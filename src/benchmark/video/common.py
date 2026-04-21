"""
Modulo condiviso per il benchmark energetico video.

Ogni script per codec (run_x265.py, run_x264.py, ...) importa da qui:
  - Le costanti (path FFmpeg, sequenze UVG, cartelle output)
  - Le funzioni di misura energetica (RAPL + Zeus, idle cal, phase measurement)
  - Le utilità di I/O (YUV loading, CSV append, bitrate computation)

Protocollo metodologico:
  1. Per ogni (seq, profilo, crf):
     a. Carica YUV raw da ~/tesi/datasets/uvg (bytes in RAM)
     b. Warmup (1 encode+decode su fetta breve)
     c. Idle cal #1 (3s CPU+GPU)
     d. Fase ENCODE: encoder_fn() → ~/tesi/bitstreams_video/{id}.bin
        → misura energia netta CPU/GPU, scrive riga CSV
     e. Idle cal #2 (3s) — ricalibrazione perché GPU si è raffreddata
     f. Fase DECODE: decoder_fn() → /dev/shm/decoded/{id}.yuv
        → misura energia netta, scrive riga CSV
        → os.remove del decoded YUV (NON misurato)
  2. I bitstream restano su disco persistente per gli script qualità,
     consentendo di aggiungere nuove metriche senza rifare l'encode.
     Il decoded YUV invece sta in tmpfs perché viene cancellato subito.

N=1 ripetizioni: ogni sequenza ha 600 frame (300 ShakeNDry) che fanno
averaging statistico interno; l'evento dura decine di secondi.
"""

import os
import gc
import time
import csv
from pathlib import Path

# ================== CONSTANTS ==================

FFMPEG = "/home/user/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

UVG_DIR = Path.home() / "tesi" / "datasets" / "uvg"
RESULTS_DIR = Path.home() / "tesi" / "results" / "video"
BITSTREAMS_DIR = Path.home() / "tesi" / "bitstreams_video"  # PERSISTENTE su disco
DECODED_DIR = Path("/dev/shm/decoded")  # tmpfs OK: cancellato subito

# Le 7 sequenze UVG 1080p canoniche.
# Attenzione: il file estratto si chiama "ReadySteadyGo", non "ReadySetGo"
# come appare sul sito UVG — usiamo il nome effettivo su disco.
UVG_SEQUENCES = [
    # (nome_logico,       nome_file,                                                    W,    H,    fps, n_frames)
    ("Beauty", "Beauty_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, 120, 600),
    ("Bosphorus", "Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, 120, 600),
    ("HoneyBee", "HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, 120, 600),
    ("Jockey", "Jockey_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, 120, 600),
    (
        "ReadySteadyGo",
        "ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv",
        1920,
        1080,
        120,
        600,
    ),
    ("ShakeNDry", "ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, 120, 300),
    ("YachtRide", "YachtRide_1920x1080_120fps_420_8bit_YUV.yuv", 1920, 1080, 120, 600),
]

# Punti operativi per codec classici: CRF canonici HEVC/AVC
# Scelta coerente con il Cap. 4 del tuo Cap. 4 (fixed CRF, bitrate osservato).
# SVT-AV1 usa CRF range diverso, lo definisce il suo script.
CLASSIC_CRF_POINTS = [22, 27, 32, 37]

# Profili temporali
PROFILES = ["LDP", "RA"]  # i codec neurali fanno solo LDP, skippano RA

IDLE_DURATION = 3.0  # secondi di idle sampling

# Schema CSV: una riga per (seq, profile, operating_point, phase)
CSV_FIELDS = [
    "codec",  # x265, x264, svtav1, dcvc_dc, dcvc_fm, dcvc_rt
    "seq",  # nome sequenza UVG
    "profile",  # LDP | RA
    "param",  # punto operativo: "crf=22" per classici, "q=3" per neurali
    "phase",  # encode | decode
    "n_frames",  # numero frame processati
    "time_s",  # durata fase (secondi)
    "energy_cpu_j",  # energia netta CPU (Joule)
    "energy_gpu_j",  # energia netta GPU (Joule) — 0 per classici
    "energy_total_j",  # somma
    "idle_cpu_w",  # potenza idle CPU misurata (W)
    "idle_gpu_w",  # potenza idle GPU misurata (W)
    "actual_mbps",  # bitrate effettivamente prodotto (osservato)
    "bitstream_path",  # path al bitstream in /dev/shm
]


# ================== RAPL (CPU energy) ==================


def read_rapl_uj():
    """Legge l'energia cumulativa del package 0 in microjoule."""
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read().strip())


def read_rapl_max():
    """Valore massimo del contatore RAPL prima del wrap-around."""
    with open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj") as f:
        return int(f.read().strip())


RAPL_MAX = read_rapl_max()


# ================== Zeus (GPU energy) ==================

# Import lazy per non forzare il caricamento se uno script non lo usa
_gpu = None


def get_gpu():
    global _gpu
    if _gpu is None:
        from zeus.device.gpu import get_gpus

        _gpu = get_gpus().gpus[0]
    return _gpu


# ================== IDLE CALIBRATION ==================


def measure_idle(duration=IDLE_DURATION, include_gpu=True):
    """
    Misura la potenza idle (CPU e opzionalmente GPU) per `duration` secondi.
    Va chiamata DOPO il warmup, con il sistema in temperatura.

    Returns: (cpu_watts, gpu_watts)
    """
    # Piccolo sleep per stabilizzare prima di iniziare la misura
    time.sleep(0.5)

    # Inizializzazioni: servono a garantire che le variabili siano sempre
    # definite anche quando include_gpu è False
    gpu = get_gpu() if include_gpu else None
    g0 = gpu.getTotalEnergyConsumption() if gpu is not None else 0
    g1 = g0

    r0 = read_rapl_uj()
    t0 = time.perf_counter()

    time.sleep(duration)

    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    if gpu is not None:
        g1 = gpu.getTotalEnergyConsumption()

    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX  # wrap-around del contatore RAPL

    cpu_w = (dr / 1e6) / dt
    gpu_w = ((g1 - g0) / 1000) / dt if include_gpu else 0.0
    return cpu_w, gpu_w


# ================== PHASE MEASUREMENT ==================


def measure_phase(run_fn, cpu_idle_w, gpu_idle_w, is_neural=False):
    """
    Esegue run_fn() una volta, misurando energia netta CPU+GPU.

    Per codec classici (is_neural=False), il contributo GPU viene considerato
    nullo (i processi ffmpeg non toccano la GPU NVIDIA).

    L'energia netta è depurata dall'assorbimento idle:
        E_net = E_gross - P_idle * dt

    Returns: dict con cpu_net_j, gpu_net_j, total_net_j, time_s
    """
    gc.disable()

    r0 = read_rapl_uj()
    gpu = get_gpu()
    g0 = gpu.getTotalEnergyConsumption()
    t0 = time.perf_counter()

    run_fn()

    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = gpu.getTotalEnergyConsumption()

    gc.enable()

    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX

    cpu_gross = dr / 1e6
    gpu_gross = (g1 - g0) / 1000

    cpu_net = max(cpu_gross - cpu_idle_w * dt, 0.0)
    gpu_net = max(gpu_gross - gpu_idle_w * dt, 0.0) if is_neural else 0.0
    total_net = cpu_net + gpu_net

    return {
        "cpu_net_j": cpu_net,
        "gpu_net_j": gpu_net,
        "total_net_j": total_net,
        "time_s": dt,
    }


# ================== YUV I/O ==================


def yuv_frame_size_bytes(width, height):
    """Dimensione di un frame YUV 4:2:0 in byte (luma + 2× chroma a 1/4)."""
    return width * height * 3 // 2


def load_yuv_raw(path, width, height, n_frames):
    """
    Carica un file YUV raw (headerless, 4:2:0 8-bit) in memoria come bytes.
    Ritorna il buffer completo. Per N frame la dimensione è N * W * H * 1.5.
    """
    expected_size = yuv_frame_size_bytes(width, height) * n_frames
    with open(path, "rb") as f:
        data = f.read(expected_size)
    if len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes from {path}, got {len(data)}")
    return data


# ================== BITRATE COMPUTATION ==================


def compute_actual_mbps(bitstream_path, n_frames, fps):
    """
    Calcola il bitrate effettivo prodotto dall'encoder, in Mbps.
        duration_s = n_frames / fps
        bitrate_bps = size_bytes * 8 / duration_s
    """
    size_bytes = os.path.getsize(bitstream_path)
    duration_s = n_frames / fps
    return (size_bytes * 8) / duration_s / 1e6


# ================== CSV ==================


def init_csv(csv_path):
    """Crea il CSV con l'header se non esiste. Idempotente."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_row(csv_path, row):
    """Appende una riga al CSV. Assume che l'header esista già."""
    # Normalizza: tutti i campi devono esistere nel dict (anche se None)
    full_row = {field: row.get(field, "") for field in CSV_FIELDS}
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writerow(full_row)


# ================== SETUP DIRECTORIES ==================


def setup_dirs():
    """Crea le cartelle necessarie all'avvio di uno script."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BITSTREAMS_DIR.mkdir(parents=True, exist_ok=True)
    DECODED_DIR.mkdir(parents=True, exist_ok=True)


# ================== UTILITY: bitstream path ==================


def bitstream_path(codec, seq, profile, op_id, ext):
    """
    Path canonico per un bitstream prodotto. Usato sia dallo script
    energetico (scrive) che dallo script qualità (legge).

    op_id è una stringa che identifica il punto operativo, es: "crf22" per
    classici, "q3" per neurali. Non contiene caratteri speciali.

    Esempio: /dev/shm/bitstreams/x265_Beauty_LDP_crf22.265
    """
    fname = f"{codec}_{seq}_{profile}_{op_id}.{ext}"
    return BITSTREAMS_DIR / fname


def decoded_path(codec, seq, profile, op_id):
    """Path temporaneo per il YUV decodificato (cancellato dopo misura)."""
    fname = f"{codec}_{seq}_{profile}_{op_id}.yuv"
    return DECODED_DIR / fname
