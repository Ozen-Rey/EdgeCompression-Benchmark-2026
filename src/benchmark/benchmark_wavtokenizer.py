"""
Benchmark WavTokenizer (ICLR 2025, music-audio SOTA).
Fase 1: encode/decode + ViSQOL (richiede protobuf 3.20.3)
Fase 2: PESQ/STOI/SDR/mel + energia (dopo upgrade protobuf)

Salva ricostruzioni per FAD successivo.
"""

import sys
import os
import time
import csv
import glob
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, "/tmp/WavTokenizer")

# ================== CONFIG ==================
DATASETS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50": os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb": os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
}
REC_DIR = os.path.expanduser("~/tesi/results/audio/reconstructions")
VISQOL_CSV = os.path.expanduser("~/tesi/results/audio/visqol_benchmark.csv")
MAX_PER_DS = 50
TARGET_SR = 24000
VISQOL_SR = 16000
CODEC_TAG = "WavTokenizer_0.9"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== CARICA WAVTOKENIZER ==================
def load_wavtokenizer():
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
    model = WavTokenizer.from_pretrained0802(config_path, model_path)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"WavTokenizer music-audio: {n_params:.1f}M params")
    return model


# ================== INIT VISQOL ==================
def init_visqol():
    import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2, similarity_result_pb2  # noqa: F401

    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = VISQOL_SR  # type: ignore
    config.options.use_speech_scoring = True  # type: ignore
    api = visqol_lib_py.VisqolApi()
    api.Create(config)  # type: ignore
    print("ViSQOL inizializzato (speech mode, 16kHz)")
    return api


# ================== RACCOGLI FILE ==================
def collect_files():
    all_files = []
    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"[!] {ds_name} non trovato")
            continue
        files = sorted(
            list(Path(ds_path).rglob("*.wav")) + list(Path(ds_path).rglob("*.flac"))
        )
        all_files.extend([(f, ds_name) for f in files])
        print(f"  {ds_name}: {len(files)} file")

    # Campiona
    sampled = []
    for ds_name in DATASETS:
        ds_files = [(f, d) for f, d in all_files if d == ds_name]
        np.random.seed(42)
        if len(ds_files) > MAX_PER_DS:
            indices = np.random.choice(len(ds_files), MAX_PER_DS, replace=False)
            ds_files = [ds_files[i] for i in sorted(indices)]
        sampled.extend(ds_files)

    print(f"Campionati: {len(sampled)} file\n")
    return sampled


# ================== MAIN ==================
def main():
    print("=" * 60)
    print("BENCHMARK WavTokenizer (ICLR 2025) + ViSQOL")
    print("=" * 60)

    model = load_wavtokenizer()
    visqol_api = init_visqol()
    files = collect_files()
    os.makedirs(REC_DIR, exist_ok=True)

    # Warmup
    print("Warmup...")
    dummy = torch.randn(1, TARGET_SR).to(device)
    with torch.no_grad():
        bid = torch.tensor([0]).to(device)
        f, c = model.encode_infer(dummy, bandwidth_id=bid)
        model.decode(f, bandwidth_id=bid)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    visqol_results = []
    print(f"\n--- {CODEC_TAG} ---")

    for i, (file_path, ds_name) in enumerate(files):
        try:
            audio, sr = librosa.load(
                str(file_path), sr=TARGET_SR, mono=True, duration=10.0
            )
            if len(audio) < TARGET_SR:
                continue

            wav = torch.from_numpy(audio).float().unsqueeze(0).to(device)

            with torch.no_grad():
                bid = torch.tensor([0]).to(device)
                features, codes = model.encode_infer(wav, bandwidth_id=bid)
                audio_out = model.decode(features, bandwidth_id=bid)

            rec = audio_out.squeeze().cpu().numpy()

            # Salva ricostruzione
            rec_path = os.path.join(REC_DIR, f"{file_path.stem}_{CODEC_TAG}.wav")
            min_len = min(len(audio), len(rec))
            sf.write(rec_path, rec[:min_len], TARGET_SR)

            # ViSQOL
            ref_16k = librosa.resample(
                audio[:min_len], orig_sr=24000, target_sr=16000
            ).astype(np.float64)
            deg_16k = librosa.resample(
                rec[:min_len], orig_sr=24000, target_sr=16000
            ).astype(np.float64)

            result = visqol_api.Measure(ref_16k, deg_16k)
            visqol_results.append(
                {
                    "file": file_path.stem,
                    "dataset": ds_name,
                    "codec": "WavTokenizer",
                    "param": "0.9",
                    "visqol": round(result.moslqo, 4),
                }
            )

        except Exception as e:
            if (i + 1) % 25 == 0:
                print(f"  [!] {file_path.name}: {str(e)[:60]}")

        if (i + 1) % 25 == 0:
            scores = [r["visqol"] for r in visqol_results]
            avg = np.mean(scores) if scores else 0
            print(f"  [{i+1}/{len(files)}] ViSQOL: {avg:.3f} (n={len(scores)})")

    # === RISULTATI ===
    scores = [r["visqol"] for r in visqol_results]
    print(f"\n{'=' * 60}")
    print(f"WavTokenizer ViSQOL globale: {np.mean(scores):.3f} (n={len(scores)})")
    print(f"{'=' * 60}")

    for ds in DATASETS:
        ds_scores = [r["visqol"] for r in visqol_results if r["dataset"] == ds]
        if ds_scores:
            print(f"  {ds}: {np.mean(ds_scores):.3f} (n={len(ds_scores)})")

    # Confronto con altri codec
    print(f"\n--- Confronto ViSQOL ---")
    print(f"  SNAC 0.8 kbps:         2.992")
    print(f"  EnCodec 1.5 kbps:      3.180")
    print(f"  WavTokenizer 0.9 kbps: {np.mean(scores):.3f}")
    print(f"  DAC 8.0 kbps:          4.850")
    print(f"  Opus 24 kbps:          4.715")

    # Append CSV
    with open(VISQOL_CSV, "a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "dataset", "codec", "param", "visqol"]
        )
        for r in visqol_results:
            w.writerow(r)
    print(f"\nAppeso a {VISQOL_CSV}")


if __name__ == "__main__":
    main()
