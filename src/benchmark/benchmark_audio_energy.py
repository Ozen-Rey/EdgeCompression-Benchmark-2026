"""
Benchmark energetico RIGOROSO audio.
Stessa metodologia delle immagini:
  1. Idle measurement (RAPL+Zeus) prima di ogni codec/param
  2. Misura encode+decode per ogni file
  3. Energia netta = misurata - idle × tempo

Codec CPU-only (Opus): energia = RAPL netta
Codec neurali (EnCodec, DAC, SNAC, WavTokenizer): energia = RAPL netta + Zeus netta
"""

import sys
import os
import time
import csv
import warnings
import subprocess
import glob
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
DATASETS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50": os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb": os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
}
MAX_PER_DS = 50
MAX_DURATION = 10.0
TARGET_SR = 24000
IDLE_DURATION = 2.0

OUTPUT_CSV = os.path.expanduser(
    "~/tesi/results/audio/full_pipeline_energy_benchmark.csv"
)
DEVICE = "cuda"
device = torch.device(DEVICE)


# ================== RAPL ==================
def read_rapl_uj():
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read().strip())


RAPL_MAX = int(open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj").read())

# ================== Zeus ==================
from zeus.device.gpu import get_gpus

gpu = get_gpus().gpus[0]

FIELDNAMES = [
    "file",
    "dataset",
    "codec",
    "param",
    "duration_s",
    "time_total_ms",
    "energy_cpu_net_j",
    "energy_gpu_net_j",
    "energy_total_net_j",
    "idle_cpu_w",
    "idle_gpu_w",
    "is_neural",
]


# ================== IDLE + MEASUREMENT ==================
def measure_idle():
    torch.cuda.synchronize()
    time.sleep(0.5)
    r0 = read_rapl_uj()
    g0 = gpu.getTotalEnergyConsumption()
    t0 = time.perf_counter()
    time.sleep(IDLE_DURATION)
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = gpu.getTotalEnergyConsumption()
    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX
    return (dr / 1e6) / dt, ((g1 - g0) / 1000) / dt


def measure_start():
    torch.cuda.synchronize()
    return read_rapl_uj(), gpu.getTotalEnergyConsumption(), time.perf_counter()


def measure_end(r0, g0, t0, cpu_idle, gpu_idle, is_neural):
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = gpu.getTotalEnergyConsumption()
    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX
    cpu_gross = dr / 1e6
    gpu_gross = (g1 - g0) / 1000
    cpu_net = max(cpu_gross - cpu_idle * dt, 0)
    gpu_net = max(gpu_gross - gpu_idle * dt, 0) if is_neural else 0.0
    return {
        "time_s": dt,
        "cpu_net": cpu_net,
        "gpu_net": gpu_net,
        "total_net": cpu_net + gpu_net,
    }


# ================== LOAD AUDIO FILES ==================
def load_audio_files():
    files = []
    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"  [!] Dataset non trovato: {ds_path}")
            continue
        ds_files = []
        for ext in ["*.flac", "*.wav"]:
            ds_files.extend(glob.glob(os.path.join(ds_path, "**", ext), recursive=True))
        ds_files = sorted(ds_files)[:MAX_PER_DS]
        for f in ds_files:
            files.append({"path": f, "dataset": ds_name, "name": Path(f).name})
    return files


def load_and_resample(path):
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True, duration=MAX_DURATION)
    return audio


# ================== OPUS (CPU) ==================
def encode_decode_opus(audio_24k, bitrate_kbps):
    tmp_in = "/dev/shm/opus_in.wav"
    tmp_opus = "/dev/shm/opus_out.opus"
    tmp_out = "/dev/shm/opus_dec.wav"

    sf.write(tmp_in, audio_24k, 24000)
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
            "-ac",
            "1",
            tmp_opus,
        ],
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_opus, "-ar", "24000", "-ac", "1", tmp_out],
        capture_output=True,
    )

    rec, _ = librosa.load(tmp_out, sr=24000, mono=True)
    for f in [tmp_in, tmp_opus, tmp_out]:
        if os.path.exists(f):
            os.remove(f)
    return rec


# ================== ENCODEC ==================
def encode_decode_encodec(model, audio_24k):
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        frames = model.encode(x)
        decoded = model.decode(frames)
    return decoded.squeeze().cpu().numpy()


# ================== DAC ==================
def encode_decode_dac(model, audio_24k):
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        z, codes, latents, _, _ = model.encode(x)
        decoded = model.decode(z)
    return decoded.squeeze().cpu().numpy()


# ================== SNAC ==================
def encode_decode_snac(model, audio_24k):
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = model.encode(x)
        decoded = model.decode(codes)
    return decoded.squeeze().cpu().numpy()


# ================== WAVTOKENIZER ==================
def encode_decode_wavtokenizer(model, audio_24k):
    x = torch.from_numpy(audio_24k).float().unsqueeze(0).to(device)
    bandwidth_id = torch.tensor([0]).to(device)
    with torch.no_grad():
        features, discrete_code = model.encode_infer(x, bandwidth_id=bandwidth_id)
        decoded = model.decode(features, bandwidth_id=bandwidth_id)
    return decoded.squeeze().cpu().numpy()


# ====================================================================
# BENCHMARK GENERICO
# ====================================================================
def benchmark_codec(
    codec_name, param, encode_decode_fn, audio_files, is_neural, warmup_fn=None
):
    # Warmup
    if warmup_fn:
        for af in audio_files[:3]:
            audio = load_and_resample(af["path"])
            try:
                warmup_fn(audio)
            except Exception:
                pass
        torch.cuda.synchronize()

    cpu_idle, gpu_idle = measure_idle()
    print(f"  {codec_name} {param} (idle: CPU={cpu_idle:.1f}W GPU={gpu_idle:.1f}W)")

    results = []
    for af in audio_files:
        audio = load_and_resample(af["path"])
        duration_s = len(audio) / TARGET_SR

        r0, g0, t0 = measure_start()
        try:
            rec = encode_decode_fn(audio)
        except Exception as e:
            print(f"    [!] {af['name']}: {e}")
            continue
        e = measure_end(r0, g0, t0, cpu_idle, gpu_idle, is_neural)

        results.append(
            {
                "file": af["name"],
                "dataset": af["dataset"],
                "codec": codec_name,
                "param": param,
                "duration_s": round(duration_s, 2),
                "time_total_ms": round(e["time_s"] * 1000, 1),
                "energy_cpu_net_j": round(e["cpu_net"], 4),
                "energy_gpu_net_j": round(e["gpu_net"], 4),
                "energy_total_net_j": round(e["total_net"], 4),
                "idle_cpu_w": round(cpu_idle, 1),
                "idle_gpu_w": round(gpu_idle, 1),
                "is_neural": is_neural,
            }
        )

    if results:
        total_dur = sum(r["duration_s"] for r in results)
        total_e = sum(r["energy_total_net_j"] for r in results)
        j_per_s = total_e / total_dur if total_dur > 0 else 0
        print(
            f"    {len(results)} files, {total_dur:.0f}s audio, "
            f"E_tot={total_e:.2f}J, J/s={j_per_s:.4f}"
        )

    return results


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 70)
    print("BENCHMARK ENERGETICO AUDIO RIGOROSO")
    print("RAPL+Zeus simultanei, idle sottratto per ogni codec/param")
    print("=" * 70)

    audio_files = load_audio_files()
    print(f"\nFile audio: {len(audio_files)}")
    for ds in DATASETS:
        n = len([f for f in audio_files if f["dataset"] == ds])
        print(f"  {ds}: {n}")

    if not audio_files:
        print("[!] Nessun file audio trovato!")
        return

    all_results = []

    # === OPUS (CPU) ===
    print("\n=== Opus ===")
    for kbps in [12, 24, 48]:
        r = benchmark_codec(
            "Opus",
            str(kbps),
            lambda a, k=kbps: encode_decode_opus(a, k),
            audio_files,
            is_neural=False,
        )
        all_results.extend(r)

    # === EnCodec (GPU) ===
    print("\n=== EnCodec ===")
    from encodec.model import EncodecModel

    for bw in [1.5, 3.0, 6.0]:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bw)
        model = model.to(device).eval()
        r = benchmark_codec(
            "EnCodec",
            str(bw),
            lambda a, m=model: encode_decode_encodec(m, a),
            audio_files,
            is_neural=True,
            warmup_fn=lambda a, m=model: encode_decode_encodec(m, a),
        )
        all_results.extend(r)
        del model
        torch.cuda.empty_cache()

    # === DAC (GPU) ===
    print("\n=== DAC ===")
    import dac

    dac_path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(dac_path).to(device).eval()  # type: ignore
    r = benchmark_codec(
        "DAC",
        "8.0",
        lambda a: encode_decode_dac(model, a),
        audio_files,
        is_neural=True,
        warmup_fn=lambda a: encode_decode_dac(model, a),
    )
    all_results.extend(r)
    del model
    torch.cuda.empty_cache()

    # === SNAC (GPU) ===
    print("\n=== SNAC ===")
    from snac import SNAC

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    r = benchmark_codec(
        "SNAC",
        "0.8",
        lambda a: encode_decode_snac(model, a),
        audio_files,
        is_neural=True,
        warmup_fn=lambda a: encode_decode_snac(model, a),
    )
    all_results.extend(r)
    del model
    torch.cuda.empty_cache()

    # === WavTokenizer (GPU) ===
    print("\n=== WavTokenizer ===")
    try:
        sys.path.insert(0, "/tmp/WavTokenizer")
        from decoder.pretrained import WavTokenizer as WT  # type: ignore

        config_path = "/home/user/.cache/huggingface/hub/models--novateur--WavTokenizer-medium-music-audio-75token/snapshots/1bcbe86db88fc1f1f86c7ea192e791b3ede17da2/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        ckpt_path = "/home/user/.cache/huggingface/hub/models--novateur--WavTokenizer-medium-music-audio-75token/snapshots/1bcbe86db88fc1f1f86c7ea192e791b3ede17da2/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"

        model = WT.from_pretrained0802(config_path, ckpt_path).to(device).eval()
        r = benchmark_codec(
            "WavTokenizer",
            "0.9",
            lambda a: encode_decode_wavtokenizer(model, a),
            audio_files,
            is_neural=True,
            warmup_fn=lambda a: encode_decode_wavtokenizer(model, a),
        )
        all_results.extend(r)
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [!] WavTokenizer: {e}")

    # === SALVA ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nCSV: {OUTPUT_CSV}")

    # === TABELLA ===
    print(f"\n{'=' * 80}")
    print("RIEPILOGO ENERGIA AUDIO (idle sottratto)")
    print(f"{'=' * 80}")
    print(
        f"{'Codec':<20} {'CPU_net':>8} {'GPU_net':>8} {'TOT_net':>8} {'J/s':>8} {'files':>6}"
    )
    print("-" * 60)

    groups = set((r["codec"], r["param"]) for r in all_results)
    for codec, param in sorted(groups):
        sub = [r for r in all_results if r["codec"] == codec and r["param"] == param]
        total_dur = sum(r["duration_s"] for r in sub)
        cpu = sum(r["energy_cpu_net_j"] for r in sub)
        gpu_e = sum(r["energy_gpu_net_j"] for r in sub)
        tot = sum(r["energy_total_net_j"] for r in sub)
        j_per_s = tot / total_dur if total_dur > 0 else 0
        print(
            f"{codec + ' ' + param:<20} {cpu:>8.2f} {gpu_e:>8.2f} {tot:>8.2f} {j_per_s:>8.4f} {len(sub):>6}"
        )


if __name__ == "__main__":
    main()
