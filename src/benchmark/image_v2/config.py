import os
from pathlib import Path

TESI_ROOT = Path(os.path.expanduser("~/tesi"))
DATASETS_ROOT = TESI_ROOT / "datasets"
RESULTS_ROOT = TESI_ROOT / "results" / "images"

DATASETS = {
    "kodak": DATASETS_ROOT / "kodak",
    "tecnick": DATASETS_ROOT / "tecnick_flat",
    "clic2020": DATASETS_ROOT / "clic2020" / "train",
    "div2k_valid": DATASETS_ROOT / "div2k" / "DIV2K_valid_HR",
    "coco_sample": DATASETS_ROOT / "coco_sample",
}

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

METRIC_COLS = [
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
]

SYSTEM_COLS = [
    "dataset",
    "codec",
    "param",
    "image",
    "width",
    "height",
    "pixels",
    "bpp",
    "pipeline",
    "time_ms",
    "energy_total_j",
    "energy_net_j",
    "params_M",
]

ALL_COLS = SYSTEM_COLS + METRIC_COLS
