import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.benchmark.benchmark_images import compress_jxl

# ================== CONFIG ==================
CSV_OUTPUT_PATH = os.path.expanduser("~/tesi/results/oracle/oracle_raw_telemetry.csv")
SAVE_INTERVAL = 5

DATASETS = {
    "kodak":       os.path.expanduser("~/tesi/datasets/kodak"),
    "clic_train":  os.path.expanduser("~/tesi/datasets/clic2020/train"),
    "div2k":       os.path.expanduser("~/tesi/datasets/div2k/DIV2K_valid_HR"),
    "tecnick":     os.path.expanduser("~/tesi/datasets/tecnick_flat"),
    "coco_sample": os.path.expanduser("~/tesi/datasets/coco_sample"),
}

# JXL distances (0 = lossless, numeri più alti = più compressione/minore qualità)
# Usiamo 4 livelli per coprire lo spettro R-D in modo equo rispetto a JPEG/HEVC
JXL_DISTANCES = [1.0, 3.0, 7.0, 12.0]

FIELDNAMES = [
    "image", "dataset", "codec", "param",
    "bpp", "psnr", "lpips", "enc_ms", "status"
]

def load_done():
    done = set()
    if os.path.exists(CSV_OUTPUT_PATH):
        with open(CSV_OUTPUT_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "OK" and row["codec"] == "JXL":
                    done.add((row["image"], row["codec"], float(row["param"])))
    return done

def run_jxl_telemetry():
    done = load_done()
    print(f"Resume: {len(done)} file JXL già completati.")

    all_images = []
    for dataset_name, dataset_path in DATASETS.items():
        for ext in ["*.png", "*.jpg"]:
            files = list(Path(dataset_path).rglob(ext))
            all_images.extend([(f, dataset_name) for f in files])

    print(f"\nInizio append JXL su {len(all_images)} immagini...\n")

    csvfile = open(CSV_OUTPUT_PATH, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)

    for idx, (img_path, dataset_name) in enumerate(all_images):
        # Usa il percorso assoluto in modo che combaci con il CSV esistente
        img_name = str(img_path)
        
        for dist in JXL_DISTANCES:
            if (img_name, "JXL", float(dist)) in done:
                continue
                
            print(f"[{idx + 1}/{len(all_images)}] Eseguo JXL (d={dist}) su {img_name}")
            row = {"image": img_name, "dataset": dataset_name, "codec": "JXL", "param": dist, "status": "OK"}
            
            try:
                m = compress_jxl(str(img_path), distance=dist)
                row.update(m)
            except Exception as e:
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] Errore JXL: {e}")
                
            writer.writerow(row)

        if (idx + 1) % SAVE_INTERVAL == 0:
            csvfile.flush()

    csvfile.close()
    print("\nTelemetria JXL completata e integrata nel CSV!")

if __name__ == "__main__":
    run_jxl_telemetry()