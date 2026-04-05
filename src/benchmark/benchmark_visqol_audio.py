"""
ViSQOL AUDIO mode (48kHz) su tutti i codec.
Speech mode già completata in visqol_benchmark.csv.
"""

import os
import glob
import csv
import numpy as np
import librosa
from pathlib import Path

import visqol_lib_py
from visqol.pb2 import visqol_config_pb2, similarity_result_pb2  # noqa: F401

RECONSTRUCTIONS_DIR = os.path.expanduser("~/tesi/results/audio/reconstructions")
DATASETS_DIRS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50": os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb": os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
}
OUTPUT_CSV = os.path.expanduser("~/tesi/results/audio/visqol_audio_mode_benchmark.csv")

CODEC_PARAMS = [
    ("Opus", "12"),
    ("Opus", "24"),
    ("Opus", "48"),
    ("EnCodec", "1.5"),
    ("EnCodec", "3.0"),
    ("EnCodec", "6.0"),
    ("DAC", "8.0"),
    ("SNAC", "0.8"),
    ("WavTokenizer", "0.9"),
]


def init_visqol_audio():
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 48000  # type: ignore
    config.options.use_speech_scoring = False  # type: ignore
    api = visqol_lib_py.VisqolApi()
    api.Create(config)  # type: ignore
    return api


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
    print("ViSQOL AUDIO MODE (48kHz) - musica/suoni")
    print("=" * 60)

    api = init_visqol_audio()
    print("ViSQOL audio mode inizializzato (48kHz)\n")

    all_results = []
    fieldnames = ["file", "dataset", "codec", "param", "visqol_audio"]

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
                ref, _ = librosa.load(orig_path, sr=48000, mono=True)
                deg, _ = librosa.load(rec_path, sr=48000, mono=True)
                min_len = min(len(ref), len(deg))
                result = api.Measure(
                    ref[:min_len].astype(np.float64),
                    deg[:min_len].astype(np.float64),
                )
                scores.append(result.moslqo)
                all_results.append(
                    {
                        "file": orig_name,
                        "dataset": dataset,
                        "codec": codec,
                        "param": param,
                        "visqol_audio": round(result.moslqo, 4),
                    }
                )
            except Exception:
                pass

            if (i + 1) % 50 == 0:
                avg = np.mean(scores) if scores else 0
                print(f"  [{i+1}/{len(rec_files)}] media: {avg:.3f} (n={len(scores)})")

        if scores:
            print(f"  MEDIA {tag}: {np.mean(scores):.3f} (n={len(scores)})")
        print()

    # Salva CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"CSV: {OUTPUT_CSV}")

    # Tabella
    print(f"\n{'=' * 60}")
    print("ViSQOL AUDIO MODE (scala 1-5, più alto = meglio)")
    print(f"{'=' * 60}")
    print(
        f"{'Codec':<22} {'Globale':>8} {'LibriS':>8} {'ESC50':>8} {'MUSDB':>8} {'n':>4}"
    )
    print("-" * 60)

    for codec, param in CODEC_PARAMS:
        subset = [r for r in all_results if r["codec"] == codec and r["param"] == param]
        scores_all = [r["visqol_audio"] for r in subset]
        if not scores_all:
            continue
        glob_avg = f"{np.mean(scores_all):.3f}"
        ds_avgs = {}
        for ds in ["librispeech", "esc50", "musdb"]:
            ds_s = [r["visqol_audio"] for r in subset if r["dataset"] == ds]
            ds_avgs[ds] = f"{np.mean(ds_s):.3f}" if ds_s else "N/A"
        tag = f"{codec}_{param}"
        print(
            f"{tag:<22} {glob_avg:>8} {ds_avgs['librispeech']:>8} {ds_avgs['esc50']:>8} {ds_avgs['musdb']:>8} {len(scores_all):>4}"
        )


if __name__ == "__main__":
    main()
