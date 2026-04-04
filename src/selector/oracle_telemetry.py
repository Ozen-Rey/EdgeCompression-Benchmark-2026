"""
Hardware Telemetry Profiler
Formato Long (Tidy Data): una riga per ogni coppia immagine+codec.
Nessun vincitore pre-calcolato — la policy si applica a posteriori.
Supporta resume automatico e gestione errori isolata per codec.
"""

import csv
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.benchmark.benchmark_images import compress_jpeg, compress_hevc
from src.benchmark.benchmark_neural_images import compress_neural
import torch
import numpy as np
import pandas as pd

# ================== CONFIG ==================
CSV_OUTPUT_PATH = os.path.expanduser("~/tesi/results/oracle/oracle_raw_telemetry.csv")
SAVE_INTERVAL = 10  # checkpoint ogni 10 immagini

DATASETS = {
    "kodak": os.path.expanduser("~/tesi/datasets/kodak"),
    "clic_train": os.path.expanduser("~/tesi/datasets/clic2020/train"),
    "div2k": os.path.expanduser("~/tesi/datasets/div2k/DIV2K_valid_HR"),
    "tecnick": os.path.expanduser("~/tesi/datasets/tecnick_flat"),
    "coco_sample": os.path.expanduser("~/tesi/datasets/coco_sample"),
}

# Parametri codec — 3 punti per famiglia (basso, medio, alto)
JPEG_QUALITIES = [30, 60, 90]
HEVC_CRFS = [45, 35, 25]
BALLE_QUALITIES = [2, 4, 6]
CHENG_QUALITIES = [2, 4, 6]

FIELDNAMES = [
    "image",
    "dataset",
    "codec",
    "param",
    "bpp",
    "psnr",
    "lpips",
    "enc_ms",
    "status",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== RESUME ==================


def load_done() -> set[tuple[str, str, int]]:
    """Carica coppie (image, codec, param) già processate per il resume."""
    done: set[tuple[str, str, int]] = set()
    if os.path.exists(CSV_OUTPUT_PATH):
        with open(CSV_OUTPUT_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "OK":
                    done.add((row["image"], row["codec"], int(row["param"])))
    return done


# ================== PROFILER ==================


def run_telemetry_profiler() -> None:
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

    done = load_done()
    print(f"Resume: {len(done)} misurazioni già completate.")

    # Carica modelli neurali una volta sola
    print("Caricamento modelli neurali...")
    from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor

    balle_models: dict[int, torch.nn.Module] = {}
    for q in BALLE_QUALITIES:
        balle_models[q] = (
            bmshj2018_hyperprior(quality=q, pretrained=True).to(device).eval()
        )

    cheng_models: dict[int, torch.nn.Module] = {}
    for q in CHENG_QUALITIES:
        cheng_models[q] = cheng2020_anchor(quality=q, pretrained=True).to(device).eval()

    print("Modelli caricati.\n")

    # Raccogli tutte le immagini
    all_images: list[tuple[Path, str]] = []
    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} non trovato, skip.")
            continue
        imgs = sorted(Path(dataset_path).glob("*.png")) + sorted(
            Path(dataset_path).glob("*.jpg")
        )
        all_images.extend([(img, dataset_name) for img in imgs])
        print(f"  {dataset_name}: {len(imgs)} immagini")

    print(f"\nTotale: {len(all_images)} immagini\n")

    # Apri CSV in append
    write_header = not os.path.exists(CSV_OUTPUT_PATH)
    csvfile = open(CSV_OUTPUT_PATH, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total_written = 0

    for img_idx, (img_path, dataset_name) in enumerate(all_images):
        img_name = str(img_path)
        print(f"[{img_idx + 1}/{len(all_images)}] {img_path.name}")

        # ── JPEG ────────────────────────────────────────────────
        for q in JPEG_QUALITIES:
            if (img_name, "JPEG", q) in done:
                continue
            row: dict = {
                "image": img_name,
                "dataset": dataset_name,
                "codec": "JPEG",
                "param": q,
                "bpp": None,
                "psnr": None,
                "lpips": None,
                "enc_ms": None,
                "status": "OK",
            }
            try:
                start = time.perf_counter()
                m = compress_jpeg(img_path, q)
                enc_ms = (time.perf_counter() - start) * 1000
                row.update(
                    {
                        "bpp": round(m["bpp"], 5),
                        "psnr": round(m["psnr"], 4),
                        "lpips": round(m["lpips"], 5),
                        "enc_ms": round(enc_ms, 2),
                    }
                )
            except Exception as e:
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] JPEG q={q}: {e}")
            writer.writerow(row)
            total_written += 1

        # ── HEVC ────────────────────────────────────────────────
        for crf in HEVC_CRFS:
            if (img_name, "HEVC", crf) in done:
                continue
            row = {
                "image": img_name,
                "dataset": dataset_name,
                "codec": "HEVC",
                "param": crf,
                "bpp": None,
                "psnr": None,
                "lpips": None,
                "enc_ms": None,
                "status": "OK",
            }
            try:
                start = time.perf_counter()
                m = compress_hevc(img_path, crf)
                enc_ms = (time.perf_counter() - start) * 1000
                row.update(
                    {
                        "bpp": round(m["bpp"], 5),
                        "psnr": round(m["psnr"], 4),
                        "lpips": round(m["lpips"], 5),
                        "enc_ms": round(enc_ms, 2),
                    }
                )
            except Exception as e:
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] HEVC crf={crf}: {e}")
            writer.writerow(row)
            total_written += 1

        # ── Ballé2018 ───────────────────────────────────────────
        for q in BALLE_QUALITIES:
            if (img_name, "Balle2018", q) in done:
                continue
            row = {
                "image": img_name,
                "dataset": dataset_name,
                "codec": "Balle2018",
                "param": q,
                "bpp": None,
                "psnr": None,
                "lpips": None,
                "enc_ms": None,
                "status": "OK",
            }
            try:
                m = compress_neural(balle_models[q], img_path)
                row.update(
                    {
                        "bpp": round(m["bpp"], 5),
                        "psnr": round(m["psnr"], 4),
                        "lpips": round(m["lpips"], 5),
                        "enc_ms": round(m["enc_ms"], 2),
                    }
                )
            except Exception as e:
                torch.cuda.empty_cache()
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] Balle2018 q={q}: {e}")
            writer.writerow(row)
            total_written += 1

        # ── Cheng2020 ───────────────────────────────────────────
        for q in CHENG_QUALITIES:
            if (img_name, "Cheng2020", q) in done:
                continue
            row = {
                "image": img_name,
                "dataset": dataset_name,
                "codec": "Cheng2020",
                "param": q,
                "bpp": None,
                "psnr": None,
                "lpips": None,
                "enc_ms": None,
                "status": "OK",
            }
            try:
                m = compress_neural(cheng_models[q], img_path)
                row.update(
                    {
                        "bpp": round(m["bpp"], 5),
                        "psnr": round(m["psnr"], 4),
                        "lpips": round(m["lpips"], 5),
                        "enc_ms": round(m["enc_ms"], 2),
                    }
                )
            except Exception as e:
                torch.cuda.empty_cache()
                row["status"] = f"ERROR: {str(e)[:100]}"
                print(f"  [!] Cheng2020 q={q}: {e}")
            writer.writerow(row)
            total_written += 1

        # Checkpoint periodico
        if (img_idx + 1) % SAVE_INTERVAL == 0:
            csvfile.flush()
            print(f"  -> Checkpoint: {total_written} righe scritte")

    csvfile.flush()
    csvfile.close()
    print(f"\nProfilazione completata! File: {CSV_OUTPUT_PATH}")
    print(f"Righe totali scritte: {total_written}")


# ================== POLICY (applicata a posteriori) ==================

# ================== POLICY (applicata a posteriori) ==================


def apply_policy(
    csv_path: str,
    max_enc_ms: float = 150.0,
    max_bpp: float = 0.5,
    mode: str = "performance",  # "performance" o "eco"
    lpips_threshold: float = 0.15,  # usato solo in modalità eco
) -> "pd.DataFrame":
    """
    Applica una policy decisionale sul CSV raw.
    Non rilancia nessun benchmark — è solo un filtro sul foglio Excel.
    """

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()

    print(f"\nPolicy: mode={mode}, max_enc_ms={max_enc_ms}, max_bpp={max_bpp}")
    # 1. Filtro candidati fisicamente ammissibili
    if mode == "eco":
        admissible = df[df["bpp"] <= max_bpp]
    else:
        admissible = df[(df["enc_ms"] <= max_enc_ms) & (df["bpp"] <= max_bpp)]

    if admissible.empty:
        print("Nessun candidato rispetta i vincoli fisici.")
        return pd.DataFrame()

    # 2. Selezione del vincitore per ogni immagine in base al QoS (mode)
    if mode == "performance":
        # Cerca la qualità assoluta
        idx = admissible.groupby("image")["lpips"].idxmin()
        winners = admissible.loc[idx]

    elif mode == "eco":

        def eco_selector(group):
            decent_classical = group[
                (group["lpips"] <= lpips_threshold)
                & (group["codec"].isin(["JPEG", "HEVC", "JXL"]))
            ]
            if not decent_classical.empty:
                return decent_classical.loc[decent_classical["lpips"].idxmin()]
            decent_neural = group[group["lpips"] <= lpips_threshold]
            if not decent_neural.empty:
                return decent_neural.loc[decent_neural["lpips"].idxmin()]
            return group.loc[group["lpips"].idxmin()]

        winners = admissible.groupby("image", group_keys=False).apply(eco_selector)

    else:
        raise ValueError("Modalità non supportata. Usa 'performance' o 'eco'.")

    winners = (
        winners.reset_index(drop=True)
        if "image" in winners.columns
        else winners.reset_index()
    )
    print(f"\nDistribuzione vincitori (Profilo {mode.upper()}):")
    print(winners["codec"].value_counts())
    print(f"\nTotale immagini elaborate: {len(winners)}")

    # Salva il dataset bilanciato per XGBoost
    out_file = str(csv_path).replace(".csv", f"_{mode}_winners.csv")
    winners.to_csv(out_file, index=False)
    print(f"Dataset dei vincitori salvato in: {out_file}")

    return pd.DataFrame(winners)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="store_true", help="Lancia il profiler hardware su GPU"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["performance", "eco"],
        help="Applica policy sul CSV esistente",
    )
    parser.add_argument("--max-enc-ms", type=float, default=150.0)
    parser.add_argument("--max-bpp", type=float, default=0.5)
    parser.add_argument("--lpips-thresh", type=float, default=0.15)
    args = parser.parse_args()

    if args.profile:
        run_telemetry_profiler()
    elif args.policy:
        apply_policy(
            CSV_OUTPUT_PATH,
            args.max_enc_ms,
            args.max_bpp,
            mode=args.policy,
            lpips_threshold=args.lpips_thresh,
        )
    else:
        # Default: lancia il profiler
        run_telemetry_profiler()
