"""
Benchmark audio completo: Opus vs EnCodec vs DAC vs SNAC.

Metriche per-file: PESQ, STOI, SI-SDR, mel distance, RTF
Energia: RAPL (CPU multi-zone) per Opus, Zeus (GPU) per codec neurali

ViSQOL e FAD vengono calcolati in script separati (richiedono protobuf diverso).
I file ricostruiti vengono salvati per l'analisi successiva con ViSQOL.

Output: CSV + JSON + tabella riepilogativa + WAV ricostruiti + Hardware Log
"""

import torch
import warnings
import time
import sys
import os
import csv
import json
import random
import glob
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/tesi"))


# ================== RIPRODUCIBILITÀ ==================
def set_seed(seed: int = 42):
    """Fissa il seed per tutti i generatori pseudo-casuali."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ================== CONFIG ==================
DATASETS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50": os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb": os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
}
MAX_DURATION = 10.0
TARGET_SR = 24000
PESQ_SR = 16000

OUTPUT_CSV = os.path.expanduser("~/tesi/results/audio/audio_benchmark_full.csv")
OUTPUT_JSON = os.path.expanduser("~/tesi/results/audio/audio_benchmark_full.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Limite di file per dataset per garantire tempi di run accettabili.
# 50 file * 3 dataset = 150 file totali per punto operativo.
MAX_PER_DS = 50


# ================== RAPL (MULTI-ZONE) ==================
def get_rapl_zones() -> list:
    """Trova le root zones di RAPL escludendo le sub-zones (core/uncore/dram) per evitare doppi conteggi."""
    zones = []
    for path in glob.glob("/sys/class/powercap/intel-rapl:*"):
        if path.count(":") == 1:  # Trova solo intel-rapl:0, intel-rapl:1, ecc.
            zones.append(path)
    return zones


RAPL_ZONES = get_rapl_zones()


def read_cpu_uj() -> dict:
    """Legge l'energia corrente (in microJoule) per tutte le zone fisiche della CPU."""
    energies = {}
    for zone in RAPL_ZONES:
        try:
            with open(f"{zone}/energy_uj", "r") as f:
                energies[zone] = int(f.read().strip())
        except Exception:
            pass
    return energies


def calc_energy_j(before: dict, after: dict) -> float:
    """Calcola i Joule consumati sommando i delta di tutte le zone e gestendo il wrap-around hardware."""
    total_j = 0.0
    for zone in before:
        if zone in after:
            b = before[zone]
            a = after[zone]
            if a >= b:
                total_j += (a - b) / 1e6
            else:
                try:
                    with open(f"{zone}/max_energy_range_uj", "r") as f:
                        max_val = int(f.read().strip())
                    total_j += ((max_val - b) + a) / 1e6
                except Exception:
                    total_j += (a - b) / 1e6  # Fallback di emergenza
    return total_j


# ================== ZEUS (GPU) ==================
from zeus.device.gpu import get_gpus

gpu_obj = get_gpus().gpus[0] if device.type == "cuda" else None


# ================== METRICHE ==================
from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi


def compute_si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB. Immuno al volume."""
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    # Rimuovi la media (DC offset)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    ref_energy = np.sum(ref**2)
    if ref_energy < 1e-10:
        return -100.0  # Silenzio totale

    # Proiezione ortogonale (Fattore di scala alpha)
    dot_product = np.sum(ref * est)
    alpha = dot_product / ref_energy

    target_scaled = alpha * ref
    noise = est - target_scaled

    target_energy = np.sum(target_scaled**2)
    noise_energy = np.sum(noise**2)

    if noise_energy < 1e-10:
        return 100.0  # Ricostruzione perfetta

    return float(10.0 * np.log10(target_energy / noise_energy))


def compute_metrics(
    orig_24k: np.ndarray, rec_24k: np.ndarray, duration_s: float, process_time_s: float
) -> dict:
    """Calcola PESQ, STOI, SI-SDR, mel distance, e RTF."""
    m: dict = {}

    min_len = min(len(orig_24k), len(rec_24k))
    orig_24k = orig_24k[:min_len]
    rec_24k = rec_24k[:min_len]

    orig_16k = librosa.resample(orig_24k, orig_sr=24000, target_sr=16000)
    rec_16k = librosa.resample(rec_24k, orig_sr=24000, target_sr=16000)

    try:
        m["pesq"] = compute_pesq(16000, orig_16k, rec_16k, "wb")
    except Exception:
        m["pesq"] = None

    try:
        m["stoi"] = compute_stoi(orig_16k, rec_16k, 16000, extended=False)
    except Exception:
        m["stoi"] = None

    try:
        m["sdr"] = compute_si_sdr(orig_24k, rec_24k)
    except Exception:
        m["sdr"] = None

    try:
        S_orig = librosa.feature.melspectrogram(y=orig_24k, sr=24000, n_mels=80)
        S_rec = librosa.feature.melspectrogram(y=rec_24k, sr=24000, n_mels=80)
        log_orig = np.log(np.maximum(S_orig, 1e-10))
        log_rec = np.log(np.maximum(S_rec, 1e-10))
        min_frames = min(log_orig.shape[1], log_rec.shape[1])
        m["mel_dist"] = float(
            np.mean(np.abs(log_orig[:, :min_frames] - log_rec[:, :min_frames]))
        )
    except Exception:
        m["mel_dist"] = None

    m["rtf"] = process_time_s / duration_s if duration_s > 0 else None

    return m


# ================== CODEC: OPUS (CPU) ==================
def compress_opus(audio_24k: np.ndarray, bitrate_kbps: int) -> tuple:
    import subprocess

    duration_s = len(audio_24k) / 24000
    tmp_in = "/dev/shm/opus_bench_in.wav"
    tmp_opus = "/dev/shm/opus_bench.opus"
    tmp_out = "/dev/shm/opus_bench_out.wav"
    sf.write(tmp_in, audio_24k, 24000)

    before = read_cpu_uj()
    t0 = time.perf_counter()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            tmp_in,
            "-c:a",
            "libopus",
            "-b:a",
            f"{bitrate_kbps}k",
            "-ar",
            "24000",
            tmp_opus,
        ],
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_opus, "-ar", "24000", tmp_out],
        capture_output=True,
    )

    t1 = time.perf_counter()
    after = read_cpu_uj()

    rec, _ = librosa.load(tmp_out, sr=24000, mono=True)
    file_bytes = os.path.getsize(tmp_opus) if os.path.exists(tmp_opus) else 0
    actual_kbps = (file_bytes * 8) / duration_s / 1000 if duration_s > 0 else 0

    for f in [tmp_in, tmp_opus, tmp_out]:
        if os.path.exists(f):
            os.remove(f)

    return rec, {
        "time_s": t1 - t0,
        "cpu_energy_j": calc_energy_j(before, after),
        "gpu_energy_j": None,
        "file_bytes": file_bytes,
        "actual_kbps": actual_kbps,
        "energy_source": "RAPL",
        "duration_s": duration_s,
    }


# ================== CODEC: ENCODEC (GPU) ==================
_encodec_model = None


def get_encodec():
    global _encodec_model
    if _encodec_model is None:
        from encodec.model import EncodecModel

        _encodec_model = EncodecModel.encodec_model_24khz().to(device).eval()
    return _encodec_model


def compress_encodec(audio_24k: np.ndarray, bandwidth: float) -> tuple:
    model = get_encodec()
    model.set_target_bandwidth(bandwidth)
    duration_s = len(audio_24k) / 24000
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).unsqueeze(0).to(device)

    torch.cuda.synchronize()
    before = gpu_obj.getTotalEnergyConsumption() if gpu_obj else 0.0
    t0 = time.perf_counter()

    with torch.no_grad():
        frames = model.encode(x)
        decoded = model.decode(frames)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    after = gpu_obj.getTotalEnergyConsumption() if gpu_obj else 0.0

    rec = decoded.squeeze().cpu().numpy()
    total_tokens = sum(f[0].numel() for f in frames)
    file_bytes = total_tokens * 10 // 8

    return rec, {
        "time_s": t1 - t0,
        "cpu_energy_j": None,
        "gpu_energy_j": (after - before) / 1000 if gpu_obj else None,
        "file_bytes": file_bytes,
        "actual_kbps": bandwidth,
        "energy_source": "Zeus",
        "duration_s": duration_s,
    }


# ================== CODEC: DAC (GPU) ==================
_dac_model = None


def get_dac():
    global _dac_model
    if _dac_model is None:
        import dac

        model_path = dac.utils.download(model_type="24khz")
        _dac_model = dac.DAC.load(str(model_path)).to(device).eval()
    return _dac_model


def compress_dac(audio_24k: np.ndarray) -> tuple:
    model = get_dac()
    duration_s = len(audio_24k) / 24000
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).unsqueeze(0).to(device)
    original_length = x.shape[-1]

    torch.cuda.synchronize()
    before = gpu_obj.getTotalEnergyConsumption() if gpu_obj else 0.0
    t0 = time.perf_counter()

    with torch.no_grad():
        z, codes, latents, _, _ = model.encode(x)
        audio_hat = model.decode(z)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    after = gpu_obj.getTotalEnergyConsumption() if gpu_obj else 0.0

    rec = audio_hat.squeeze().cpu().numpy()[:original_length]
    total_tokens = codes.numel()
    file_bytes = total_tokens * 10 // 8

    return rec, {
        "time_s": t1 - t0,
        "cpu_energy_j": None,
        "gpu_energy_j": (after - before) / 1000 if gpu_obj else None,
        "file_bytes": file_bytes,
        "actual_kbps": 8.0,
        "energy_source": "Zeus",
        "duration_s": duration_s,
    }


# ================== CODEC: SNAC (GPU, SOTA 2024) ==================
_snac_model = None


def get_snac():
    global _snac_model
    if _snac_model is None:
        from snac import SNAC

        _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    return _snac_model


def compress_snac(audio_24k: np.ndarray) -> tuple:
    model = get_snac()
    duration_s = len(audio_24k) / 24000
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).unsqueeze(0).to(device)

    torch.cuda.synchronize()
    before = gpu_obj.getTotalEnergyConsumption() if gpu_obj else 0.0
    t0 = time.perf_counter()

    with torch.no_grad():
        codes = model.encode(x)
        audio_hat = model.decode(codes)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    after = gpu_obj.getTotalEnergyConsumption() if gpu_obj else 0.0

    rec = audio_hat.squeeze().cpu().numpy()

    # NOTA TESI: Bitrate calcolato come "Bitrate di Payload Teorico".
    # Non essendoci un container (muxer) ufficiale, misuriamo l'entropia del latente.
    total_tokens = sum(c.numel() for c in codes)
    file_bytes = total_tokens * 12 // 8  # codebook 4096 = 12 bit
    actual_kbps = (file_bytes * 8) / duration_s / 1000 if duration_s > 0 else 0

    return rec, {
        "time_s": t1 - t0,
        "cpu_energy_j": None,
        "gpu_energy_j": (after - before) / 1000 if gpu_obj else None,
        "file_bytes": file_bytes,
        "actual_kbps": actual_kbps,
        "energy_source": "Zeus",
        "duration_s": duration_s,
    }


# ================== MAIN ==================
def main():
    print("=" * 70)
    print("HARDWARE TELEMETRY LOG")
    print("=" * 70)
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA:    {torch.version.cuda}")
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    else:
        print("GPU:     CPU Only (Neural Codecs might be extremely slow)")
    print(f"RAPL:    {len(RAPL_ZONES)} socket(s) detected")
    print("=" * 70 + "\n")

    if not RAPL_ZONES:
        print(
            "[!] Attenzione: Nessuna zona RAPL trovata. Sei su root? Modulo intel_rapl caricato?"
        )
        return

    reconstructions_dir = os.path.expanduser("~/tesi/results/audio/reconstructions")
    os.makedirs(reconstructions_dir, exist_ok=True)

    all_files = []
    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"[!] Dataset {ds_name} non trovato: {ds_path}")
            continue
        files = sorted(
            list(Path(ds_path).rglob("*.wav")) + list(Path(ds_path).rglob("*.flac"))
        )
        all_files.extend([(f, ds_name) for f in files])
        print(f"  {ds_name}: {len(files)} file trovati")

    sampled = []
    for ds_name in DATASETS:
        ds_files = [(f, d) for f, d in all_files if d == ds_name]
        # Ripristiniamo il seed localmente per sicurezza sul sampling
        np.random.seed(42)
        if len(ds_files) > MAX_PER_DS:
            indices = np.random.choice(len(ds_files), MAX_PER_DS, replace=False)
            ds_files = [ds_files[i] for i in sorted(indices)]
        sampled.extend(ds_files)
    all_files = sampled
    print(f"\nTotale Campionati per Benchmark: {len(all_files)} file\n")

    if not all_files:
        return

    CODECS = [
        ("Opus", "12", lambda a: compress_opus(a, 12)),
        ("Opus", "24", lambda a: compress_opus(a, 24)),
        ("Opus", "48", lambda a: compress_opus(a, 48)),
        ("EnCodec", "1.5", lambda a: compress_encodec(a, 1.5)),
        ("EnCodec", "3.0", lambda a: compress_encodec(a, 3.0)),
        ("EnCodec", "6.0", lambda a: compress_encodec(a, 6.0)),
        ("DAC", "8.0", lambda a: compress_dac(a)),
        ("SNAC", "0.8", lambda a: compress_snac(a)),
    ]

    ALL_COLS = [
        "file",
        "dataset",
        "codec",
        "param",
        "actual_kbps",
        "pesq",
        "stoi",
        "sdr",
        "mel_dist",
        "rtf",
        "time_s",
        "cpu_energy_j",
        "gpu_energy_j",
        "energy_source",
        "file_bytes",
        "duration_s",
    ]
    all_results = []

    print("Warmup codec neurali...")
    dummy = np.random.randn(24000).astype(np.float32) * 0.1
    for codec_name, param, fn in CODECS:
        if codec_name != "Opus":
            try:
                fn(dummy)
            except Exception as e:
                print(f"  [!] Warmup fallito per {codec_name} {param}: {e}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup completato.\n")

    print("=" * 70)
    print("BENCHMARK AUDIO (PESQ, STOI, SI-SDR, Mel Dist, RTF, Energia)")
    print("=" * 70)

    for file_idx, (file_path, ds_name) in enumerate(all_files):
        try:
            audio, sr = librosa.load(
                str(file_path), sr=TARGET_SR, mono=True, duration=MAX_DURATION
            )
        except Exception:
            continue

        if len(audio) < TARGET_SR:
            continue

        duration_s = len(audio) / TARGET_SR

        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            print(
                f"[{file_idx+1}/{len(all_files)}] Analisi in corso: {file_path.name} ({ds_name}, {duration_s:.1f}s)"
            )

        for codec_name, param, compress_fn in CODECS:
            row: dict = {
                "file": file_path.name,
                "dataset": ds_name,
                "codec": codec_name,
                "param": param,
            }
            try:
                rec, sys_m = compress_fn(audio)
                qual_m = compute_metrics(
                    audio, rec, sys_m["duration_s"], sys_m["time_s"]
                )
                row.update(sys_m)
                row.update(qual_m)

                # Salva il file per il successivo test di Google ViSQOL
                out_name = f"{file_path.stem}_{codec_name}_{param}.wav"
                out_path = os.path.join(reconstructions_dir, out_name)
                sf.write(out_path, rec, TARGET_SR)

            except Exception as e:
                for col in ALL_COLS:
                    row.setdefault(col, None)
                if (file_idx + 1) % 10 == 0:
                    print(f"    [!] Errore {codec_name} {param}: {str(e)[:60]}")

            all_results.append(row)

    # === SALVATAGGIO DATI ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nSalvataggio RAW Data completato: {OUTPUT_CSV}")

    # Aggregazione Statistica JSON
    summary: dict = {}
    groups = set((r["codec"], r["param"]) for r in all_results)
    for codec, param in sorted(groups):
        subset = [
            r
            for r in all_results
            if r["codec"] == codec and r["param"] == param and r.get("pesq") is not None
        ]
        if not subset:
            continue
        key = f"{codec}_{param}"
        summary[key] = {"codec": codec, "param": param, "n_files": len(subset)}
        for col in [
            "pesq",
            "stoi",
            "sdr",
            "mel_dist",
            "rtf",
            "time_s",
            "cpu_energy_j",
            "gpu_energy_j",
            "actual_kbps",
        ]:
            vals = [r[col] for r in subset if r.get(col) is not None]
            summary[key][col] = round(float(np.mean(vals)), 4) if vals else None

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # === TABELLA FINALE ===
    print("\n" + "=" * 110)
    print("TABELLA RIEPILOGATIVA GLOBALE (Medie)")
    print("=" * 110)
    header = (
        f"{'Codec':<18} {'kbps':>6} {'PESQ':>6} {'STOI':>6} {'SI-SDR':>7} "
        f"{'MelD':>6} {'RTF':>7} {'E_cpu':>7} {'E_gpu':>7} {'Fonte':>6}"
    )
    print(header)
    print("-" * len(header))
    for key in sorted(summary.keys()):
        s = summary[key]

        def fmt(v, f=".2f"):
            return f"{v:{f}}" if v is not None else "N/A"

        print(
            f"{key:<18} {fmt(s.get('actual_kbps'),'.1f'):>6} "
            f"{fmt(s.get('pesq'),'.2f'):>6} {fmt(s.get('stoi'),'.3f'):>6} "
            f"{fmt(s.get('sdr'),'.1f'):>7} {fmt(s.get('mel_dist'),'.2f'):>6} "
            f"{fmt(s.get('rtf'),'.4f'):>7} "
            f"{fmt(s.get('cpu_energy_j'),'.2f'):>7} {fmt(s.get('gpu_energy_j'),'.2f'):>7} "
            f"{'RAPL' if s.get('cpu_energy_j') is not None else 'Zeus':>6}"
        )

    print("\n" + "=" * 110)
    print("DETTAGLIO RIPARTITO PER DATASET")
    print("=" * 110)
    for ds in DATASETS:
        print(f"\n--- {ds.upper()} ---")
        print(
            f"  {'Codec':<18} {'PESQ':>6} {'STOI':>6} {'SI-SDR':>7} {'MelD':>6} {'RTF':>7} n"
        )
        for key in sorted(summary.keys()):
            codec, param = summary[key]["codec"], summary[key]["param"]
            subset = [
                r
                for r in all_results
                if r["codec"] == codec
                and r["param"] == param
                and r["dataset"] == ds
                and r.get("pesq") is not None
            ]
            if not subset:
                continue
            pesq_avg = np.mean([r["pesq"] for r in subset])
            stoi_avg = np.mean([r["stoi"] for r in subset if r.get("stoi") is not None])
            sdr_avg = np.mean([r["sdr"] for r in subset if r.get("sdr") is not None])
            mel_avg = np.mean(
                [r["mel_dist"] for r in subset if r.get("mel_dist") is not None]
            )
            rtf_avg = np.mean([r["rtf"] for r in subset if r.get("rtf") is not None])
            print(
                f"  {key:<18} {pesq_avg:>6.2f} {stoi_avg:>6.3f} {sdr_avg:>7.1f} "
                f"{mel_avg:>6.2f} {rtf_avg:>7.4f} {len(subset):>3}"
            )


if __name__ == "__main__":
    main()
