"""
Oracolo v2 - Multi-dataset (Kodak + CLIC train + COCO)
Versione anti-OOM + LPIPS migliore per codec + resume
"""
import csv
import os
import sys
import torch
from pathlib import Path
import time

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.benchmark.benchmark_images import compress_jpeg, compress_hevc
from src.benchmark.benchmark_neural_images import compress_neural

# ================== CONFIG EDGE ==================
MAX_ENC_MS = 150.0
MAX_BPP = 0.5
RESULTS_CSV = os.path.expanduser("~/tesi/results/oracle/oracle_v2_multi.csv")

FIELDNAMES = [
    "image", "dataset", "winner_codec", "winner_param",
    "winner_bpp", "winner_psnr", "winner_lpips", "winner_enc_ms",
    "jpeg_lpips", "hevc_lpips", "balle_lpips", "cheng_lpips",
    "note"
]

def load_done() -> set:
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["image"])
    return done


def select_winner(candidates: list) -> tuple[dict | None, str]:
    admissible = [c for c in candidates if c["enc_ms"] <= MAX_ENC_MS and c["bpp"] <= MAX_BPP]
    if admissible:
        return min(admissible, key=lambda x: x["lpips"]), ""
    if candidates:
        return min(candidates, key=lambda x: x["bpp"]), "FALLBACK_MIN_BPP"
    return None, "NO_CANDIDATES"


def get_best_lpips(candidates: list, codec_name: str) -> float:
    codec_cands = [c for c in candidates if c["codec"] == codec_name]
    if not codec_cands:
        return -1.0
    return min(codec_cands, key=lambda x: x["lpips"])["lpips"]


def run_oracle():
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    done = load_done()
    print(f"Resume: {len(done)} immagini già processate.")

    device = torch.device("cuda")
    from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor

    write_header = not os.path.exists(RESULTS_CSV)
    csvfile = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    # Dataset — solo Kodak e CLIC
    all_datasets = {
    "kodak": os.path.expanduser("~/tesi/datasets/kodak"),
    "clic_train": os.path.expanduser("~/tesi/datasets/clic2020/train"),
    }

    all_images = []
    for dataset_name, dataset_path in all_datasets.items():
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} non trovato, skip.")
            continue
        imgs = sorted(Path(dataset_path).glob("*.png")) + \
               sorted(Path(dataset_path).glob("*.jpg"))
        all_images.extend([(img, dataset_name) for img in imgs])
        print(f"=== {dataset_name}: {len(imgs)} immagini ===")

    print(f"\nTotale: {len(all_images)} immagini da processare")
    # =============================================================================

    for img_path, dataset_name in all_images:
        img_name = str(img_path)
        if img_name in done:
            continue

        try:
            candidates = []

            # JPEG full
            for q in [30, 60, 90]:
                start = time.time()
                m = compress_jpeg(img_path, q)
                enc_ms = (time.time() - start) * 1000
                candidates.append({
                    "codec": "JPEG", "param": q,
                    "bpp": m["bpp"], "psnr": m["psnr"], "lpips": m["lpips"], "enc_ms": enc_ms
                })

            # HEVC full
            for crf in [45, 35, 25]:
                start = time.time()
                m = compress_hevc(img_path, crf)
                enc_ms = (time.time() - start) * 1000
                candidates.append({
                    "codec": "HEVC", "param": crf,
                    "bpp": m["bpp"], "psnr": m["psnr"], "lpips": m["lpips"], "enc_ms": enc_ms
                })

            # === NEURAL LAZY LOADING (anti-OOM) ===
            # Balle2018 1-8
            for q in [2, 4, 6]:
                model = bmshj2018_hyperprior(quality=q, pretrained=True).to(device).eval()
                m = compress_neural(model, img_path)
                candidates.append({
                    "codec": "Balle2018", "param": q,
                    "bpp": m["bpp"], "psnr": m["psnr"], "lpips": m["lpips"], "enc_ms": m["enc_ms"]
                })
                del model
                torch.cuda.empty_cache()

            # Cheng2020 1-6
            for q in [2, 4, 6]:
                model = cheng2020_anchor(quality=q, pretrained=True).to(device).eval()
                m = compress_neural(model, img_path)
                candidates.append({
                    "codec": "Cheng2020", "param": q,
                    "bpp": m["bpp"], "psnr": m["psnr"], "lpips": m["lpips"], "enc_ms": m["enc_ms"]
                })
                del model
                torch.cuda.empty_cache()

            # === SELEZIONE + LPIPS MIGLIORE PER OGNI FAMIGLIA ===
            winner, note = select_winner(candidates)

            row = {
                "image": img_name,
                "dataset": dataset_name,
                "winner_codec": winner["codec"],
                "winner_param": winner["param"],
                "winner_bpp": round(winner["bpp"], 4),
                "winner_psnr": round(winner["psnr"], 4),
                "winner_lpips": round(winner["lpips"], 4),
                "winner_enc_ms": round(winner["enc_ms"], 1),
                "jpeg_lpips": round(get_best_lpips(candidates, "JPEG"), 4),
                "hevc_lpips": round(get_best_lpips(candidates, "HEVC"), 4),
                "balle_lpips": round(get_best_lpips(candidates, "Balle2018"), 4),
                "cheng_lpips": round(get_best_lpips(candidates, "Cheng2020"), 4),
                "note": note
            }

            writer.writerow(row)
            csvfile.flush()
            done.add(img_name)

            print(f" {img_path.name} → {winner['codec']} q={winner['param']} "
                  f"(lpips={winner['lpips']:.4f} {note})")

        except Exception as e:
            torch.cuda.empty_cache()
            print(f" ❌ ERROR {img_path.name}: {e}")
            error_row = {
                "image": img_name, "dataset": dataset_name,
                "winner_codec": "ERROR", "winner_param": "",
                "winner_bpp": -1, "winner_psnr": -1, "winner_lpips": -1, "winner_enc_ms": -1,
                "jpeg_lpips": -1, "hevc_lpips": -1, "balle_lpips": -1, "cheng_lpips": -1,
                "note": str(e)[:200]
            }
            writer.writerow(error_row)
            csvfile.flush()
            done.add(img_name)

    csvfile.close()
    print(f"\n🎯 Oracolo v2 completato! File: {RESULTS_CSV}")


if __name__ == "__main__":
    run_oracle()