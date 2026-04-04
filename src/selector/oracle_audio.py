import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.benchmark.benchmark_audio import compress_opus, compress_encodec

# ================== CONFIG ==================
CSV_OUTPUT_PATH = os.path.expanduser("~/tesi/results/audio/oracle_audio_telemetry.csv")
SAVE_INTERVAL = 10  # checkpoint ogni 10 file audio

DATASETS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50":       os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb":       os.path.expanduser("~/tesi/datasets_audio/musdb_sample"),
}

# Parametri codec — 3 punti per famiglia (basso, medio, alto)
OPUS_KBPS    = [12, 24, 48]
ENCODEC_BW   = [1.5, 3.0, 6.0]

FIELDNAMES = [
    "file", "dataset", "codec", "param",
    "kbps", "pesq", "mel_dist", "audio_dist", "enc_ms", "status"
]

# ================== RESUME ==================
def load_done():
    """Carica tuple (file, codec, param) già processate per il resume."""
    done = set()
    if os.path.exists(CSV_OUTPUT_PATH):
        with open(CSV_OUTPUT_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "OK":
                    done.add((row["file"], row["codec"], float(row["param"])))
    return done

# ================== PROFILER ==================
def run_audio_telemetry():
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    done = load_done()
    print(f"Resume: {len(done)} misurazioni audio già completate.")

    all_files = []
    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} non trovato in {dataset_path}, skip.")
            continue
        
        # Cerca file .wav o .flac ricorsivamente (LibriSpeech usa .flac)
        files = list(Path(dataset_path).rglob("*.wav")) + list(Path(dataset_path).rglob("*.flac"))
        all_files.extend([(f, dataset_name) for f in files])
        print(f"  {dataset_name}: {len(files)} file audio")

    print(f"\nTotale: {len(all_files)} file da profilare\n")

    write_header = not os.path.exists(CSV_OUTPUT_PATH)
    csvfile = open(CSV_OUTPUT_PATH, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total_written = 0

    for idx, (file_path, dataset_name) in enumerate(all_files):
        file_name = file_path.name
        print(f"[{idx + 1}/{len(all_files)}] {file_name}")

        # ── OPUS (Classico su CPU) ────────────────────────────────
        for kbps in OPUS_KBPS:
            if (file_name, "Opus", float(kbps)) in done:
                continue
            row = {"file": file_name, "dataset": dataset_name, "codec": "Opus", "param": kbps, "status": "OK"}
            try:
                m = compress_opus(str(file_path), kbps)
                row.update(m)
            except Exception as e:
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] Opus {kbps}k: {e}")
            writer.writerow(row)
            total_written += 1

        # ── ENCODEC (Neurale su GPU) ──────────────────────────────
        for bw in ENCODEC_BW:
            if (file_name, "EnCodec", float(bw)) in done:
                continue
            row = {"file": file_name, "dataset": dataset_name, "codec": "EnCodec", "param": bw, "status": "OK"}
            try:
                m = compress_encodec(str(file_path), bw)
                row.update(m)
            except Exception as e:
                import torch
                torch.cuda.empty_cache()
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] EnCodec {bw}k: {e}")
            writer.writerow(row)
            total_written += 1

        if (idx + 1) % SAVE_INTERVAL == 0:
            csvfile.flush()

    csvfile.close()
    print(f"\nProfilazione Audio completata! File: {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    run_audio_telemetry()