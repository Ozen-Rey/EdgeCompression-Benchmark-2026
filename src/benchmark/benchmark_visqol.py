"""
Benchmark ViSQOL: calcola ViSQOL MOS-LQO sui file ricostruiti.
Requisiti: protobuf==3.20.3
Dopo: pip install "protobuf>=6.0"
"""

import os
import glob
import csv
import numpy as np
import librosa
from pathlib import Path

import visqol_lib_py
from visqol.pb2 import visqol_config_pb2, similarity_result_pb2

RECONSTRUCTIONS_DIR = os.path.expanduser("~/tesi/results/audio/reconstructions")
DATASETS_DIRS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50": os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb": os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
}
OUTPUT_CSV = os.path.expanduser("~/tesi/results/audio/visqol_benchmark.csv")
VISQOL_SR = 16000

CODEC_PARAMS = [
    ("Opus", "12"),
    ("Opus", "24"),
    ("Opus", "48"),
    ("EnCodec", "1.5"),
    ("EnCodec", "3.0"),
    ("EnCodec", "6.0"),
    ("DAC", "8.0"),
    ("SNAC", "0.8"),
]


def init_visqol():
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = VISQOL_SR  # type: ignore
    config.options.use_speech_scoring = True  # type: ignore
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api


def measure_visqol(api, ref_path: str, deg_path: str) -> float:
    ref, _ = librosa.load(ref_path, sr=VISQOL_SR, mono=True)
    deg, _ = librosa.load(deg_path, sr=VISQOL_SR, mono=True)
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len].astype(np.float64)
    deg = deg[:min_len].astype(np.float64)
    result = api.Measure(ref, deg)
    return result.moslqo


def find_original(filename: str) -> tuple:
    for ds_name, ds_path in DATASETS_DIRS.items():
        for ext in ["wav", "flac"]:
            candidate = os.path.join(ds_path, f"{filename}.{ext}")
            if os.path.exists(candidate):
                return candidate, ds_name
            matches = glob.glob(
                os.path.join(ds_path, "**", f"{filename}.{ext}"), recursive=True
            )
            if matches:
                return matches[0], ds_name
    return None, None


def main():
    print("=" * 60)
    print("BENCHMARK ViSQOL (MOS-LQO)")
    print("=" * 60)

    api = init_visqol()
    print("ViSQOL inizializzato (speech mode, 16kHz)\n")

    all_results = []
    fieldnames = ["file", "dataset", "codec", "param", "visqol"]

    for codec, param in CODEC_PARAMS:
        tag = f"{codec}_{param}"
        rec_files = sorted(glob.glob(os.path.join(RECONSTRUCTIONS_DIR, f"*_{tag}.wav")))

        if not rec_files:
            print(f"[!] Nessun file per {tag}")
            continue

        print(f"--- {tag}: {len(rec_files)} file ---")
        scores = []

        for i, rec_path in enumerate(rec_files):
            rec_name = Path(rec_path).stem
            orig_name = rec_name.rsplit(f"_{tag}", 1)[0]
            orig_path, dataset = find_original(orig_name)

            if orig_path is None:
                continue

            try:
                score = measure_visqol(api, orig_path, rec_path)
                scores.append(score)
                all_results.append(
                    {
                        "file": orig_name,
                        "dataset": dataset,
                        "codec": codec,
                        "param": param,
                        "visqol": round(score, 4),
                    }
                )
            except Exception as e:
                pass

            if (i + 1) % 25 == 0:
                avg = np.mean(scores) if scores else 0
                print(f"  [{i+1}/{len(rec_files)}] media: {avg:.3f} (n={len(scores)})")

        if scores:
            print(f"  MEDIA {tag}: {np.mean(scores):.3f} (n={len(scores)})")
        print()

    # Salva CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"CSV: {OUTPUT_CSV}")

    # Tabella
    print("\n" + "=" * 60)
    print("RISULTATI ViSQOL (MOS-LQO, scala 1-5)")
    print("=" * 60)
    print(f"{'Codec':<18} {'ViSQOL':>8} {'n':>5}")
    print("-" * 35)
    for codec, param in CODEC_PARAMS:
        tag = f"{codec}_{param}"
        subset = [
            r["visqol"]
            for r in all_results
            if r["codec"] == codec and r["param"] == param
        ]
        if subset:
            print(f"{tag:<18} {np.mean(subset):>8.3f} {len(subset):>5}")

    # Per dataset
    print("\n" + "=" * 60)
    print("DETTAGLIO PER DATASET")
    print("=" * 60)
    for ds in DATASETS_DIRS:
        print(f"\n--- {ds} ---")
        for codec, param in CODEC_PARAMS:
            tag = f"{codec}_{param}"
            subset = [
                r["visqol"]
                for r in all_results
                if r["codec"] == codec and r["param"] == param and r["dataset"] == ds
            ]
            if subset:
                print(f"  {tag:<18} {np.mean(subset):.3f} (n={len(subset)})")


if __name__ == "__main__":
    main()
