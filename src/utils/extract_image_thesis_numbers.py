from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

IN = ROOT / "results" / "images" / "image_4dataset_RDE_paper_ready.csv"
OUT = ROOT / "results" / "images" / "image_thesis_numbers.json"

df = pd.read_csv(IN)

# Normalizzazione minima
df["codec"] = df["codec"].astype(str)
df["param"] = df["param"].astype(str)
df["eval_dataset"] = df["eval_dataset"].astype(str)

quality_cols = [
    "bpp",
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "dists",
    "fsim",
    "gmsd",
    "vif",
    "haarpsi",
    "dss",
    "mdsi",
    "ssimulacra2",
    "energy_per_image_j",
    "energy_cpu_per_image_j",
    "energy_gpu_per_image_j",
]

missing = [c for c in quality_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Media per punto operativo su tutti i dataset e tutte le immagini
op = (
    df.groupby(["codec", "param"], as_index=False)[quality_cols]
    .mean(numeric_only=True)
)

# Media per codec
codec_summary = (
    df.groupby(["codec"], as_index=False)[quality_cols]
    .mean(numeric_only=True)
)

# Media per dataset/codec
dataset_codec_summary = (
    df.groupby(["eval_dataset", "codec"], as_index=False)[quality_cols]
    .mean(numeric_only=True)
)

def f(x, nd=4):
    if pd.isna(x):
        return None
    return round(float(x), nd)

def best_by_metric(metric, higher=True):
    s = op.copy()
    s = s.dropna(subset=[metric])
    if len(s) == 0:
        return None
    idx = s[metric].idxmax() if higher else s[metric].idxmin()
    r = s.loc[idx]
    return {
        "codec": str(r["codec"]),
        "param": str(r["param"]),
        metric: f(r[metric], 4),
        "bpp": f(r["bpp"], 4),
        "psnr": f(r["psnr"], 4),
        "energy_per_image_j": f(r["energy_per_image_j"], 4),
    }

numbers = {}

numbers["dataset"] = {
    "n_rows": int(len(df)),
    "n_datasets": int(df["eval_dataset"].nunique()),
    "datasets": sorted(df["eval_dataset"].unique().tolist()),
    "n_images_total": int(df[["eval_dataset", "image"]].drop_duplicates().shape[0])
    if "image" in df.columns else None,
    "n_codecs": int(df["codec"].nunique()),
    "codecs": sorted(df["codec"].unique().tolist()),
}

numbers["ranges"] = {
    "bpp_min": f(df["bpp"].min(), 5),
    "bpp_max": f(df["bpp"].max(), 5),
    "psnr_min": f(df["psnr"].min(), 4),
    "psnr_max": f(df["psnr"].max(), 4),
    "ssimulacra2_min": f(df["ssimulacra2"].min(), 4),
    "ssimulacra2_max": f(df["ssimulacra2"].max(), 4),
    "energy_per_image_j_min": f(op["energy_per_image_j"].min(), 6),
    "energy_per_image_j_max": f(op["energy_per_image_j"].max(), 6),
    "energy_dynamic_range_ratio": f(
        op["energy_per_image_j"].max() / op["energy_per_image_j"].min(), 2
    ),
}

numbers["best_operating_points"] = {
    "best_psnr": best_by_metric("psnr", higher=True),
    "best_ssimulacra2": best_by_metric("ssimulacra2", higher=True),
    "lowest_energy": best_by_metric("energy_per_image_j", higher=False),
    "lowest_bpp": best_by_metric("bpp", higher=False),
}

numbers["codec_summary"] = {}
for _, r in codec_summary.iterrows():
    c = str(r["codec"])
    numbers["codec_summary"][c] = {
        "bpp_mean": f(r["bpp"], 5),
        "psnr_mean": f(r["psnr"], 4),
        "ssimulacra2_mean": f(r["ssimulacra2"], 4),
        "energy_per_image_j_mean": f(r["energy_per_image_j"], 6),
        "energy_cpu_per_image_j_mean": f(r["energy_cpu_per_image_j"], 6),
        "energy_gpu_per_image_j_mean": f(r["energy_gpu_per_image_j"], 6),
    }

numbers["operating_points"] = {}
for _, r in op.iterrows():
    key = f"{r['codec']} | {r['param']}"
    numbers["operating_points"][key] = {
        "bpp": f(r["bpp"], 5),
        "psnr": f(r["psnr"], 4),
        "ssimulacra2": f(r["ssimulacra2"], 4),
        "lpips": f(r["lpips"], 5),
        "dists": f(r["dists"], 5),
        "energy_per_image_j": f(r["energy_per_image_j"], 6),
        "energy_cpu_per_image_j": f(r["energy_cpu_per_image_j"], 6),
        "energy_gpu_per_image_j": f(r["energy_gpu_per_image_j"], 6),
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")

print("[OK] Saved:", OUT)
print(json.dumps(numbers["dataset"], indent=2, ensure_ascii=False))
print()
print(json.dumps(numbers["ranges"], indent=2, ensure_ascii=False))
print()
print(json.dumps(numbers["best_operating_points"], indent=2, ensure_ascii=False))