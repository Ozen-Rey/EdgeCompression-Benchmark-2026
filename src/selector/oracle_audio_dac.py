import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.benchmark.benchmark_audio import compress_dac

# ================== CONFIG ==================
CSV_OUTPUT_PATH = os.path.expanduser("~/tesi/results/audio/oracle_audio_telemetry.csv")
SAVE_INTERVAL = 5

DATASETS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50":       os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb":       os.path.expanduser("~/tesi/datasets_audio/musdb_sample"),
}

# DAC 24kHz viaggia nativamente a 8.0 kbps
DAC_BW = [8.0]

FIELDNAMES = [
    "file", "dataset", "codec", "param",
    "kbps", "pesq", "mel_dist", "audio_dist", "enc_ms", "status"
]

def load_done():
    done = set()
    if os.path.exists(CSV_OUTPUT_PATH):
        with open(CSV_OUTPUT_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "OK" and row["codec"] == "DAC":
                    done.add((row["file"], row["codec"], float(row["param"])))
    return done

def run_dac_telemetry():
    done = load_done()
    print(f"Resume: {len(done)} file DAC già completati.")

    all_files = []
    for dataset_name, dataset_path in DATASETS.items():
        files = list(Path(dataset_path).rglob("*.wav")) + list(Path(dataset_path).rglob("*.flac"))
        all_files.extend([(f, dataset_name) for f in files])

    print(f"\nInizio append DAC su {len(all_files)} file...\n")

    csvfile = open(CSV_OUTPUT_PATH, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)

    for idx, (file_path, dataset_name) in enumerate(all_files):
        file_name = file_path.name
        
        for bw in DAC_BW:
            if (file_name, "DAC", float(bw)) in done:
                continue
                
            print(f"[{idx + 1}/{len(all_files)}] Eseguo DAC su {file_name}")
            row = {"file": file_name, "dataset": dataset_name, "codec": "DAC", "param": bw, "status": "OK"}
            
            try:
                m = compress_dac(str(file_path), bw)
                row.update(m)
            except Exception as e:
                import torch
                torch.cuda.empty_cache()
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] Errore DAC: {e}")
                
            writer.writerow(row)

        if (idx + 1) % SAVE_INTERVAL == 0:
            csvfile.flush()

    csvfile.close()
    print("\nTelemetria DAC completata e integrata nel CSV!")

if __name__ == "__main__":
    run_dac_telemetry()