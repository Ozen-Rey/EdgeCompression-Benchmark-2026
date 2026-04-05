"""
WavTokenizer: metriche quality + energia batch.
Usa le ricostruzioni già salvate in reconstructions/.
Richiede: protobuf >= 6.0 (ViSQOL NON funziona qui)
"""

import sys
import os
import time
import csv
import glob
import numpy as np
import librosa
import torch
from pathlib import Path
from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi
from zeus.device.gpu import get_gpus

sys.path.insert(0, "/tmp/WavTokenizer")

# ================== CONFIG ==================
REC_DIR = os.path.expanduser("~/tesi/results/audio/reconstructions")
DATASETS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50": os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb": os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
}
OUTPUT_CSV = os.path.expanduser("~/tesi/results/audio/wavtokenizer_metrics.csv")
CODEC_TAG = "WavTokenizer_0.9"
TARGET_SR = 24000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = get_gpus().gpus[0] if device.type == "cuda" else None


# ================== TROVA ORIGINALE ==================
def find_original(stem: str) -> tuple:
    for ds_name, ds_path in DATASETS.items():
        for ext in ["wav", "flac"]:
            candidate = os.path.join(ds_path, f"{stem}.{ext}")
            if os.path.exists(candidate):
                return candidate, ds_name
            matches = glob.glob(
                os.path.join(ds_path, "**", f"{stem}.{ext}"), recursive=True
            )
            if matches:
                return matches[0], ds_name
    return None, None


# ================== METRICHE ==================
def compute_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    noise = est - ref
    ref_pow = np.sum(ref**2)
    noise_pow = np.sum(noise**2)
    if noise_pow < 1e-10:
        return 100.0
    if ref_pow < 1e-10:
        return -100.0
    return float(10.0 * np.log10(ref_pow / noise_pow))


def compute_mel_distance(ref: np.ndarray, deg: np.ndarray, sr: int = 24000) -> float:
    S_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=80)
    S_deg = librosa.feature.melspectrogram(y=deg, sr=sr, n_mels=80)
    log_ref = np.log(np.maximum(S_ref, 1e-10))
    log_deg = np.log(np.maximum(S_deg, 1e-10))
    min_frames = min(log_ref.shape[1], log_deg.shape[1])
    return float(np.mean(np.abs(log_ref[:, :min_frames] - log_deg[:, :min_frames])))


# ================== FASE 1: METRICHE PER-FILE ==================
def run_metrics():
    print("=" * 60)
    print("FASE 1: Metriche quality WavTokenizer (dalle ricostruzioni)")
    print("=" * 60)

    rec_files = sorted(glob.glob(os.path.join(REC_DIR, f"*_{CODEC_TAG}.wav")))
    print(f"{len(rec_files)} ricostruzioni trovate\n")

    if not rec_files:
        print("[!] Nessuna ricostruzione. Esegui prima benchmark_wavtokenizer.py")
        return []

    results = []
    for i, rec_path in enumerate(rec_files):
        stem = Path(rec_path).stem.rsplit(f"_{CODEC_TAG}", 1)[0]
        orig_path, ds_name = find_original(stem)
        if orig_path is None:
            continue

        ref, _ = librosa.load(orig_path, sr=TARGET_SR, mono=True)
        deg, _ = librosa.load(rec_path, sr=TARGET_SR, mono=True)
        ml = min(len(ref), len(deg))
        ref, deg = ref[:ml], deg[:ml]

        # PESQ e STOI a 16kHz
        ref16 = librosa.resample(ref, orig_sr=24000, target_sr=16000)
        deg16 = librosa.resample(deg, orig_sr=24000, target_sr=16000)

        try:
            p = compute_pesq(16000, ref16, deg16, "wb")
        except Exception:
            p = None
        try:
            s = compute_stoi(ref16, deg16, 16000, extended=False)
        except Exception:
            s = None

        sdr = compute_sdr(ref, deg)
        mel_d = compute_mel_distance(ref, deg)
        duration = len(ref) / TARGET_SR

        results.append(
            {
                "file": stem,
                "dataset": ds_name,
                "codec": "WavTokenizer",
                "param": "0.9",
                "pesq": round(p, 4) if p else None,
                "stoi": round(s, 4) if s else None,
                "sdr": round(sdr, 4),
                "mel_dist": round(mel_d, 4),
                "duration_s": round(duration, 2),
            }
        )

        if (i + 1) % 50 == 0:
            pesq_vals = [r["pesq"] for r in results if r["pesq"]]
            stoi_vals = [r["stoi"] for r in results if r["stoi"]]
            print(
                f"  [{i+1}/{len(rec_files)}] "
                f"PESQ={np.mean(pesq_vals):.2f} "
                f"STOI={np.mean(stoi_vals):.3f}"
            )

    # Stampa risultati
    pesq_all = [r["pesq"] for r in results if r["pesq"]]
    stoi_all = [r["stoi"] for r in results if r["stoi"]]
    sdr_all = [r["sdr"] for r in results]
    mel_all = [r["mel_dist"] for r in results]

    print(f"\nWavTokenizer 0.9 kbps (n={len(results)}):")
    print(
        f"  PESQ: {np.mean(pesq_all):.2f}  "
        f"STOI: {np.mean(stoi_all):.3f}  "
        f"SDR: {np.mean(sdr_all):.1f}  "
        f"MelD: {np.mean(mel_all):.2f}"
    )

    for ds in DATASETS:
        sub = [r for r in results if r["dataset"] == ds]
        if sub:
            pp = [r["pesq"] for r in sub if r["pesq"]]
            ss = [r["stoi"] for r in sub if r["stoi"]]
            sd = [r["sdr"] for r in sub]
            mm = [r["mel_dist"] for r in sub]
            print(
                f"  {ds}: PESQ={np.mean(pp):.2f} STOI={np.mean(ss):.3f} "
                f"SDR={np.mean(sd):.1f} MelD={np.mean(mm):.2f} (n={len(sub)})"
            )

    # Salva CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = [
        "file",
        "dataset",
        "codec",
        "param",
        "pesq",
        "stoi",
        "sdr",
        "mel_dist",
        "duration_s",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nCSV: {OUTPUT_CSV}")

    return results


# ================== FASE 2: ENERGIA BATCH ==================
def run_energy():
    print(f"\n{'=' * 60}")
    print("FASE 2: Energia batch WavTokenizer")
    print("=" * 60)

    from decoder.pretrained import WavTokenizer  # type: ignore
    from huggingface_hub import hf_hub_download

    repo = "novateur/WavTokenizer-medium-music-audio-75token"
    config_path = hf_hub_download(
        repo_id=repo,
        filename="wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    )
    model_path = hf_hub_download(
        repo_id=repo,
        filename="wavtokenizer_medium_music_audio_320_24k_v2.ckpt",
    )
    model = WavTokenizer.from_pretrained0802(config_path, model_path).cuda().eval()
    print(
        f"Modello caricato: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params"
    )

    # Carica 50 file in RAM
    audios = []
    total_dur = 0
    files = sorted(list(Path(DATASETS["librispeech"]).rglob("*.flac")))[:50]
    for f in files:
        a, _ = librosa.load(str(f), sr=TARGET_SR, mono=True, duration=10.0)
        if len(a) >= TARGET_SR:
            audios.append(a)
            total_dur += len(a) / TARGET_SR
    print(f"{len(audios)} file, {total_dur:.1f}s totali")

    # Warmup
    dummy = torch.randn(1, TARGET_SR).cuda()
    with torch.no_grad():
        bid = torch.tensor([0]).cuda()
        ff, cc = model.encode_infer(dummy, bandwidth_id=bid)
        model.decode(ff, bandwidth_id=bid)
    torch.cuda.synchronize()

    # Batch energy
    before = gpu.getTotalEnergyConsumption()  # type: ignore
    t0 = time.perf_counter()
    for a in audios:
        wav = torch.from_numpy(a).float().unsqueeze(0).cuda()
        with torch.no_grad():
            bid = torch.tensor([0]).cuda()
            features, codes = model.encode_infer(wav, bandwidth_id=bid)
            _ = model.decode(features, bandwidth_id=bid)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    after = gpu.getTotalEnergyConsumption()  # type: ignore

    wt_j = (after - before) / 1000
    wt_s = t1 - t0
    print(
        f"\nWavTokenizer batch: {wt_j:.2f} J in {wt_s:.1f}s for {total_dur:.0f}s audio"
    )
    print(f"  -> {wt_j/total_dur:.4f} J/s audio, RTF={wt_s/total_dur:.4f}")

    print(f"\n--- Confronto energia (J/s audio) ---")
    print(f"  {'Codec':<22} {'J/s':>6} {'RTF':>8}")
    print(f"  {'-' * 40}")
    print(f"  {'SNAC 0.8':<22} {'0.32':>6} {'0.0010':>8}")
    print(f"  {'Opus 24':<22} {'0.35':>6} {'0.0076':>8}")
    print(f"  {'WavTokenizer 0.9':<22} {wt_j/total_dur:>6.2f} {wt_s/total_dur:>8.4f}")
    print(f"  {'EnCodec 3.0':<22} {'0.49':>6} {'0.0023':>8}")
    print(f"  {'DAC 8.0':<22} {'1.40':>6} {'0.0034':>8}")


# ================== MAIN ==================
if __name__ == "__main__":
    results = run_metrics()
    run_energy()
