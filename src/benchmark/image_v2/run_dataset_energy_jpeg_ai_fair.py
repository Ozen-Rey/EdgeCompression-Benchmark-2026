"""Fair JPEG AI energy benchmark.

This benchmark runs JPEG AI inside one persistent Python process, using
RecoEncoderProcess and RecoDecoderProcess directly.

Compared with the external CLI wrapper, this avoids:
    - conda run per image
    - Python startup per image
    - model loading per image

Quality/rate are imported from:
    results/images/<dataset>/<dataset>_12metric_paper_ready.csv

Output:
    results/images/<dataset>/<dataset>_jpeg_ai_energy_fair.csv
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATASETS = {
    "kodak": Path.home() / "tesi/datasets/kodak",
    "tecnick": Path.home() / "tesi/datasets/tecnick_flat",
    "div2k_valid": Path.home() / "tesi/datasets/div2k/DIV2K_valid_HR",
    "clic2020": Path.home() / "tesi/datasets/clic2020/train",
}

FIELDNAMES = [
    "eval_dataset",
    "codec",
    "param",
    "pipeline",
    "n_images",
    "n_repeats",
    "avg_bpp",
    "avg_psnr",
    "rate_quality_source",
    "total_time_s",
    "energy_cpu_total_j",
    "energy_gpu_total_j",
    "energy_total_j",
    "energy_cpu_per_image_j",
    "energy_gpu_per_image_j",
    "energy_per_image_j",
    "idle_cpu_w",
    "idle_gpu_w",
    "params_M",
    "is_neural",
    "execution_mode",
    "jpeg_ai_root",
    "conda_env",
]

IDLE_DURATION = 3.0

RAPL_ENERGY_PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
RAPL_MAX_PATH = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")


def rapl_available() -> bool:
    return RAPL_ENERGY_PATH.exists() and RAPL_MAX_PATH.exists()


def read_rapl_uj() -> int:
    if not rapl_available():
        return 0
    return int(RAPL_ENERGY_PATH.read_text().strip())


def read_rapl_max() -> int:
    if not rapl_available():
        return 0
    return int(RAPL_MAX_PATH.read_text().strip())


RAPL_MAX = read_rapl_max()


class GpuMeter:
    def __init__(self) -> None:
        self.mode = "none"
        self.handle = None

        if not torch.cuda.is_available():
            return

        try:
            from zeus.device.gpu import get_gpus

            gpus = get_gpus().gpus
            if len(gpus) > 0:
                self.mode = "zeus"
                self.handle = gpus[0]
                return
        except Exception:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            self.mode = "pynvml"
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self.mode = "none"
            self.handle = None

    def read_mj(self) -> float:
        if self.mode == "zeus" and self.handle is not None:
            zeus_handle = cast(Any, self.handle)
            return float(zeus_handle.getTotalEnergyConsumption())

        if self.mode == "pynvml" and self.handle is not None:
            try:
                return float(self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
            except Exception:
                return 0.0

        return 0.0


GPU = GpuMeter()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_images(dataset_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() in exts])


def paper_ready_csv(dataset_name: str) -> Path:
    return (
        Path.home()
        / "tesi"
        / "results"
        / "images"
        / dataset_name
        / f"{dataset_name}_12metric_paper_ready.csv"
    )


def load_rate_quality_lookup(dataset_name: str) -> dict[tuple[str, str], tuple[float, float]]:
    p = paper_ready_csv(dataset_name)
    if not p.exists():
        print(f"[WARN] paper_ready non trovato: {p}")
        return {}

    df = pd.read_csv(p)
    summary = (
        df.groupby(["codec", "param"])[["bpp", "psnr"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    out: dict[tuple[str, str], tuple[float, float]] = {}
    for _, r in summary.iterrows():
        out[(str(r["codec"]), str(r["param"]))] = (float(r["bpp"]), float(r["psnr"]))
    return out


def get_rate_quality(
    lookup: dict[tuple[str, str], tuple[float, float]],
    param: str,
) -> tuple[float, float, str]:
    key = ("JPEG_AI", param)
    if key not in lookup:
        print(f"[WARN] rate/quality non trovati per JPEG_AI {param}")
        return float("nan"), float("nan"), "missing"

    bpp, psnr = lookup[key]
    return float(bpp), float(psnr), "paper_ready"


def measure_idle() -> tuple[float, float]:
    sync_cuda()
    time.sleep(0.5)

    r0 = read_rapl_uj()
    g0 = GPU.read_mj()
    t0 = time.perf_counter()

    time.sleep(IDLE_DURATION)

    sync_cuda()
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = GPU.read_mj()

    dt = max(t1 - t0, 1e-9)

    dr = r1 - r0
    if rapl_available() and RAPL_MAX > 0 and dr < 0:
        dr += RAPL_MAX

    cpu_w = (dr / 1e6) / dt if rapl_available() else 0.0
    gpu_w = ((g1 - g0) / 1000.0) / dt if GPU.mode != "none" else 0.0

    return float(cpu_w), float(gpu_w)


def measure_batch_energy(
    run_fn: Callable[[], None],
    n_repeats: int,
    cpu_idle_w: float,
    gpu_idle_w: float,
) -> dict[str, float]:
    gc.disable()
    sync_cuda()

    r0 = read_rapl_uj()
    g0 = GPU.read_mj()
    t0 = time.perf_counter()

    for _ in range(n_repeats):
        run_fn()

    sync_cuda()
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = GPU.read_mj()

    gc.enable()

    dt = max(t1 - t0, 1e-9)

    dr = r1 - r0
    if rapl_available() and RAPL_MAX > 0 and dr < 0:
        dr += RAPL_MAX

    cpu_gross = dr / 1e6 if rapl_available() else 0.0
    gpu_gross = (g1 - g0) / 1000.0 if GPU.mode != "none" else 0.0

    cpu_net = max(cpu_gross - cpu_idle_w * dt, 0.0)
    gpu_net = max(gpu_gross - gpu_idle_w * dt, 0.0)

    return {
        "cpu_net": float(cpu_net),
        "gpu_net": float(gpu_net),
        "total_net": float(cpu_net + gpu_net),
        "time_s": float(dt),
    }


def target_to_arg(v: float) -> str:
    return str(int(round(v * 100)))


def target_to_param(v: float) -> str:
    # Coerente con paper_ready:
    # 0.50 -> target_bpp=0.5
    # 1.00 -> target_bpp=1.0
    return f"target_bpp={float(v)}"


def make_row(
    dataset_name: str,
    param: str,
    n_images: int,
    n_repeats: int,
    avg_bpp: float,
    avg_psnr: float,
    rq_source: str,
    energy: dict[str, float],
    idle_cpu_w: float,
    idle_gpu_w: float,
    jpeg_ai_root: Path,
    conda_env: str,
) -> dict[str, Any]:
    n_total = max(n_images * n_repeats, 1)

    cpu_total = float(energy["cpu_net"])
    gpu_total = float(energy["gpu_net"])
    total = float(energy["total_net"])
    total_time = float(energy["time_s"])

    return {
        "eval_dataset": dataset_name,
        "codec": "JPEG_AI",
        "param": param,
        "pipeline": "fair_inprocess_end_to_end_energy",
        "n_images": n_images,
        "n_repeats": n_repeats,
        "avg_bpp": round(avg_bpp, 9) if math.isfinite(avg_bpp) else float("nan"),
        "avg_psnr": round(avg_psnr, 9) if math.isfinite(avg_psnr) else float("nan"),
        "rate_quality_source": rq_source,
        "total_time_s": round(total_time, 9),
        "energy_cpu_total_j": round(cpu_total, 9),
        "energy_gpu_total_j": round(gpu_total, 9),
        "energy_total_j": round(total, 9),
        "energy_cpu_per_image_j": round(cpu_total / n_total, 12),
        "energy_gpu_per_image_j": round(gpu_total / n_total, 12),
        "energy_per_image_j": round(total / n_total, 12),
        "idle_cpu_w": round(idle_cpu_w, 9),
        "idle_gpu_w": round(idle_gpu_w, 9),
        "params_M": 0.0,
        "is_neural": True,
        "execution_mode": "persistent_python_process_reference_software",
        "jpeg_ai_root": str(jpeg_ai_root),
        "conda_env": conda_env,
    }


def write_rows_replace_existing(out_csv: Path, rows: list[dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)
    for col in FIELDNAMES:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[FIELDNAMES]

    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        for col in FIELDNAMES:
            if col not in old_df.columns:
                old_df[col] = None
        old_df = old_df[FIELDNAMES]

        keys = set(
            zip(
                new_df["eval_dataset"].astype(str),
                new_df["codec"].astype(str),
                new_df["param"].astype(str),
                new_df["pipeline"].astype(str),
            )
        )

        keep = [
            (str(r.eval_dataset), str(r.codec), str(r.param), str(r.pipeline)) not in keys
            for r in old_df.itertuples(index=False)
        ]
        old_df = old_df.loc[keep].copy()
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Salvato: {out_csv}")


def run_pair_silent(enc_proc: Any, dec_proc: Any, enc_args: list[str], dec_args: list[str]) -> None:
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            enc_proc.process(enc_args)
            dec_proc.process(dec_args)


def benchmark_target(
    dataset_name: str,
    input_paths: list[Path],
    rq_lookup: dict[tuple[str, str], tuple[float, float]],
    target_bpp: float,
    extra_encoder_args: list[str],
    jpeg_ai_root: Path,
    conda_env: str,
    n_repeats: int,
    tmp: Path,
) -> dict[str, Any]:
    encoder_mod = importlib.import_module("src.reco.coders.encoder")
    decoder_mod = importlib.import_module("src.reco.coders.decoder")

    RecoEncoderProcess = cast(Any, encoder_mod).RecoEncoderProcess
    RecoDecoderProcess = cast(Any, decoder_mod).RecoDecoderProcess

    param = target_to_param(target_bpp)
    print(f"\nJPEG_AI {param}")

    enc_proc = RecoEncoderProcess(None)
    dec_proc = RecoDecoderProcess(None)

    target_arg = target_to_arg(target_bpp)

    # Warmup: loads models and runs one encode/decode outside measurement.
    warm_bs = tmp / f"warm_{target_arg}.jai"
    warm_rec = tmp / f"warm_{target_arg}.png"

    warm_enc_args = [
        str(input_paths[0]),
        str(warm_bs),
        "--set_target_bpp",
        target_arg,
        *extra_encoder_args,
    ]
    warm_dec_args = [str(warm_bs), str(warm_rec)]

    print("warmup/load models")
    run_pair_silent(enc_proc, dec_proc, warm_enc_args, warm_dec_args)
    sync_cuda()

    cpu_idle, gpu_idle = measure_idle()

    def run_all() -> None:
        for i, inp in enumerate(input_paths):
            bitstream = tmp / f"energy_{i:04d}_{target_arg}.jai"
            rec = tmp / f"energy_{i:04d}_{target_arg}.png"

            enc_args = [
                str(inp),
                str(bitstream),
                "--set_target_bpp",
                target_arg,
                *extra_encoder_args,
            ]
            dec_args = [str(bitstream), str(rec)]

            run_pair_silent(enc_proc, dec_proc, enc_args, dec_args)

    energy = measure_batch_energy(
        run_all,
        n_repeats=n_repeats,
        cpu_idle_w=cpu_idle,
        gpu_idle_w=gpu_idle,
    )

    avg_bpp, avg_psnr, rq_source = get_rate_quality(rq_lookup, param)

    row = make_row(
        dataset_name=dataset_name,
        param=param,
        n_images=len(input_paths),
        n_repeats=n_repeats,
        avg_bpp=avg_bpp,
        avg_psnr=avg_psnr,
        rq_source=rq_source,
        energy=energy,
        idle_cpu_w=cpu_idle,
        idle_gpu_w=gpu_idle,
        jpeg_ai_root=jpeg_ai_root,
        conda_env=conda_env,
    )

    print(
        f"[OK] JPEG_AI {param} | "
        f"bpp={row['avg_bpp']} PSNR={row['avg_psnr']} "
        f"E/img={row['energy_per_image_j']} J"
    )

    return row


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--jpeg_ai_root",
        type=str,
        default=str(Path.home() / "tesi/external_codecs/jpeg-ai-reference-software"),
    )
    parser.add_argument("--conda_env", type=str, default="jpeg_ai_vm")

    parser.add_argument(
        "--target_bpps",
        nargs="+",
        type=float,
        default=[0.15, 0.25, 0.35, 0.50, 0.75, 1.00],
    )
    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument(
        "--extra_encoder_args",
        nargs=argparse.REMAINDER,
        default=["--cfg", "cfg/tools_on.json", "cfg/profiles/base.json"],
    )

    parser.add_argument("--out_csv", type=str, default=None)

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = DATASETS[dataset_name]
    jpeg_ai_root = Path(args.jpeg_ai_root).expanduser().resolve()

    if not jpeg_ai_root.exists():
        raise FileNotFoundError(f"jpeg_ai_root non trovato: {jpeg_ai_root}")

    if args.out_csv is not None:
        out_csv = Path(args.out_csv).expanduser().resolve()
    else:
        out_csv = (
            Path.home()
            / "tesi"
            / "results"
            / "images"
            / dataset_name
            / f"{dataset_name}_jpeg_ai_energy_fair.csv"
        )

    if args.overwrite and out_csv.exists():
        out_csv.unlink()

    # Make JPEG AI package importable as "src".
    os.chdir(jpeg_ai_root)
    sys.path.insert(0, str(jpeg_ai_root))

    print("=" * 80)
    print("JPEG AI ENERGY FAIR")
    print(f"dataset: {dataset_name}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"jpeg_ai_root: {jpeg_ai_root}")
    print(f"conda_env label: {args.conda_env}")
    print(f"target_bpps: {args.target_bpps}")
    print(f"repeats: {args.repeats}")
    print(f"GPU meter: {GPU.mode}")
    print(f"out_csv: {out_csv}")
    print("=" * 80)

    if not rapl_available():
        print("[WARN] RAPL non disponibile. CPU energy sarà 0.")

    rq_lookup = load_rate_quality_lookup(dataset_name)

    image_paths = get_images(dataset_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    if len(image_paths) == 0:
        raise RuntimeError(f"Nessuna immagine trovata in {dataset_dir}")

    tmp = Path("/dev/shm") / f"jpeg_ai_fair_{dataset_name}_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    try:
        input_paths: list[Path] = []

        print("\nPreparing PNG inputs in /dev/shm")
        for i, img_path in enumerate(tqdm(image_paths, desc="prep")):
            img = Image.open(img_path).convert("RGB")
            inp = tmp / f"input_{i:04d}.png"
            img.save(inp)
            input_paths.append(inp)

        for target_bpp in args.target_bpps:
            rows.append(
                benchmark_target(
                    dataset_name=dataset_name,
                    input_paths=input_paths,
                    rq_lookup=rq_lookup,
                    target_bpp=target_bpp,
                    extra_encoder_args=args.extra_encoder_args,
                    jpeg_ai_root=jpeg_ai_root,
                    conda_env=args.conda_env,
                    n_repeats=args.repeats,
                    tmp=tmp,
                )
            )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    write_rows_replace_existing(out_csv, rows)

    print("\n" + "=" * 80)
    print("[DONE] JPEG AI fair energy completato")
    print(f"CSV: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
