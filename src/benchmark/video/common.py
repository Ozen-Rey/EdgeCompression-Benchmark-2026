"""
Modulo condiviso per il benchmark energetico video. v3 — RAPL polling fix.

CAMBIAMENTI v3:
  - measure_phase ora usa un thread di polling RAPL (intervallo 10s)
  - Cattura tutti i wraparound del counter uint32 (RAPL_MAX ≈ 4.3 kJ)
  - Funziona per encode/decode di durata arbitraria

NOTA: l'overhead del polling è trascurabile (~1 ms ogni 10s = <0.01% time).
"""

import os
import gc
import time
import csv
import threading
from pathlib import Path

# ================== CONSTANTS ==================

FFMPEG = "/home/user/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"

UVG_DIR = Path.home() / "tesi" / "datasets" / "uvg"
RESULTS_DIR = Path.home() / "tesi" / "results" / "video"
BITSTREAMS_DIR = Path.home() / "tesi" / "bitstreams_video"
DECODED_DIR = Path("/dev/shm/decoded")

UVG_SEQUENCES = [
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

CLASSIC_CRF_POINTS = [22, 27, 32, 37]
PROFILES = ["LDP", "RA"]
IDLE_DURATION = 3.0

# Polling RAPL: ogni POLL_INTERVAL secondi durante measure_phase
# 9800X3D peak ~150W → 4295J/150W = 28.6s prima del wraparound
# POLL_INTERVAL=10s è ben oltre il margine di sicurezza
POLL_INTERVAL = 10.0

CSV_FIELDS = [
    "codec",
    "seq",
    "profile",
    "preset",
    "param",
    "phase",
    "n_frames",
    "time_s",
    "energy_cpu_j",
    "energy_gpu_j",
    "energy_total_j",
    "idle_cpu_w",
    "idle_gpu_w",
    "actual_mbps",
    "bitstream_path",
]


# ================== RAPL ==================


def read_rapl_uj():
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read().strip())


def read_rapl_max():
    with open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj") as f:
        return int(f.read().strip())


RAPL_MAX = read_rapl_max()


# ================== Zeus ==================

_gpu = None


def get_gpu():
    global _gpu
    if _gpu is None:
        from zeus.device.gpu import get_gpus

        _gpu = get_gpus().gpus[0]
    return _gpu


# ================== IDLE CALIBRATION ==================


def measure_idle(duration=IDLE_DURATION, include_gpu=True):
    time.sleep(0.5)

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
        dr += RAPL_MAX  # 1 wraparound max in 3s

    cpu_w = (dr / 1e6) / dt
    gpu_w = ((g1 - g0) / 1000) / dt if include_gpu else 0.0
    return cpu_w, gpu_w


# ================== RAPL POLLING (FIX WRAPAROUND) ==================


class RaplPoller:
    """
    Thread di polling RAPL che accumula energia totale gestendo wraparound.

    Uso:
      poller = RaplPoller()
      poller.start()
      ... do work ...
      total_uj = poller.stop()  # microJoule cumulativi corretti
    """

    def __init__(self, poll_interval=POLL_INTERVAL):
        self.poll_interval = poll_interval
        self.cumulative_uj = 0
        self.last_reading = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def _poll_loop(self):
        while self.running:
            time.sleep(self.poll_interval)
            if not self.running:
                break
            self._update()

    def _update(self):
        """Legge RAPL e accumula delta gestendo wraparound."""
        curr = read_rapl_uj()
        with self.lock:
            if self.last_reading is not None:
                delta = curr - self.last_reading
                if delta < 0:
                    delta += RAPL_MAX
                self.cumulative_uj += delta
            self.last_reading = curr

    def start(self):
        with self.lock:
            self.last_reading = read_rapl_uj()
            self.cumulative_uj = 0
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Ferma il polling, fa una lettura finale, ritorna cumulative_uj."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=self.poll_interval + 1.0)
        # Final read per catturare l'ultimo intervallo
        self._update()
        with self.lock:
            return self.cumulative_uj


# ================== PHASE MEASUREMENT (v3 con polling) ==================


def measure_phase(run_fn, cpu_idle_w, gpu_idle_w, is_neural=False):
    gc.disable()

    # GPU non ha problema wraparound (Zeus accumula internamente)
    gpu = get_gpu()
    g0 = gpu.getTotalEnergyConsumption()

    # CPU usa polling per gestire wraparound multipli
    poller = RaplPoller()
    poller.start()
    t0 = time.perf_counter()

    run_fn()

    t1 = time.perf_counter()
    cpu_gross_uj = poller.stop()
    g1 = gpu.getTotalEnergyConsumption()

    gc.enable()

    dt = t1 - t0
    cpu_gross = cpu_gross_uj / 1e6
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
    return width * height * 3 // 2


def load_yuv_raw(path, width, height, n_frames):
    expected_size = yuv_frame_size_bytes(width, height) * n_frames
    with open(path, "rb") as f:
        data = f.read(expected_size)
    if len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes from {path}, got {len(data)}")
    return data


# ================== BITRATE ==================


def compute_actual_mbps(bitstream_path, n_frames, fps):
    size_bytes = os.path.getsize(bitstream_path)
    duration_s = n_frames / fps
    return (size_bytes * 8) / duration_s / 1e6


# ================== CSV ==================


def init_csv(csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_row(csv_path, row):
    full_row = {field: row.get(field, "") for field in CSV_FIELDS}
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writerow(full_row)


# ================== SETUP ==================


def setup_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BITSTREAMS_DIR.mkdir(parents=True, exist_ok=True)
    DECODED_DIR.mkdir(parents=True, exist_ok=True)


# ================== PATHS ==================


def bitstream_path(codec, seq, profile, op_id, ext):
    fname = f"{codec}_{seq}_{profile}_{op_id}.{ext}"
    return BITSTREAMS_DIR / fname


def decoded_path(codec, seq, profile, op_id):
    fname = f"{codec}_{seq}_{profile}_{op_id}.yuv"
    return DECODED_DIR / fname
