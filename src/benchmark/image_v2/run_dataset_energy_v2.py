"""Energy V2 image benchmark.

Scopo:
    Misurare energia e tempo dei codec immagine su dataset selezionabile.

Metodologia:
    - Questo script misura SOLO energia/tempo.
    - La qualità/rate ufficiale (avg_bpp, avg_psnr) viene importata dal CSV:
          results/images/<dataset>/<dataset>_12metric_paper_ready.csv
    - In questo modo si evita che il benchmark energetico ricalcoli qualità con
      path numerici leggermente diversi.

Codec supportati:
    - JPEG      -> end_to_end_energy
    - JXL       -> end_to_end_energy
    - HEVC      -> end_to_end_energy
    - Ballé     -> actual_bitstream_energy
    - Cheng     -> actual_bitstream_energy
    - ELIC      -> actual_bitstream_energy
    - TCM       -> actual_bitstream_energy
    - DCAE      -> actual_bitstream_energy

JPEG AI:
    - NON è incluso qui.
    - Va misurato con benchmark separato, perché richiede ambiente separato
      e il wrapper CLI/conda-run introduce overhead non confrontabile.

Output:
    ~/tesi/results/images/<dataset>/<dataset>_energy_v2.csv

Note:
    - CPU energy: Intel RAPL.
    - GPU energy: Zeus/NVML.
    - Per codec CPU-only, GPU net viene posta a 0.
    - cuDNN viene disabilitato globalmente per coerenza con
      run_dataset_neural_actual.py e per evitare inconsistenze numeriche
      nei codec neurali custom.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .config import DATASETS
from .io_utils import get_images
from .codecs_neural_actual import (
    load_balle,
    load_cheng,
    load_elic,
    load_tcm,
    load_dcae,
)
from typing import Any, Callable, cast

warnings.filterwarnings("ignore")

# Critical for numerical consistency with actual-bitstream benchmark.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# =============================================================================
# Global config
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)
transform = transforms.ToTensor()

IDLE_DURATION = 3.0

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
]


# =============================================================================
# RAPL
# =============================================================================

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


# =============================================================================
# Zeus GPU
# =============================================================================


def get_gpu_handle():
    if not torch.cuda.is_available():
        return None

    try:
        from zeus.device.gpu import get_gpus

        gpus = get_gpus().gpus
        if len(gpus) == 0:
            return None
        return gpus[0]
    except Exception as e:
        print(f"[WARN] Zeus GPU non disponibile: {e}")
        return None


GPU_HANDLE = get_gpu_handle()


def read_gpu_energy_mj() -> float:
    if GPU_HANDLE is None:
        return 0.0

    try:
        return float(GPU_HANDLE.getTotalEnergyConsumption())
    except Exception:
        return 0.0


# =============================================================================
# Utility
# =============================================================================


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_bytes(obj: Any) -> int:
    if isinstance(obj, bytes):
        return len(obj)
    if isinstance(obj, list):
        return sum(count_bytes(x) for x in obj)
    if isinstance(obj, tuple):
        return sum(count_bytes(x) for x in obj)
    return 0


def count_params_m(net: torch.nn.Module | None) -> float | None:
    if net is None:
        return None
    return float(sum(p.numel() for p in net.parameters()) / 1e6)


def preload_dataset(
    dataset_dir: Path, limit: int | None = None
) -> list[dict[str, Any]]:
    paths = get_images(dataset_dir)

    if limit is not None:
        paths = paths[:limit]

    data: list[dict[str, Any]] = []

    for img_path in tqdm(paths, desc="preload"):
        pil = Image.open(img_path).convert("RGB")
        rgb = np.array(pil)
        tensor = transform(pil).unsqueeze(0).to(device)

        data.append(
            {
                "path": img_path,
                "name": img_path.name,
                "rgb": rgb,
                "tensor": tensor,
                "h": int(rgb.shape[0]),
                "w": int(rgb.shape[1]),
            }
        )

    return data


# =============================================================================
# Paper-ready rate/quality source
# =============================================================================


def paper_ready_csv(dataset_name: str) -> Path:
    return (
        Path.home()
        / "tesi"
        / "results"
        / "images"
        / dataset_name
        / f"{dataset_name}_12metric_paper_ready.csv"
    )


def load_rate_quality_lookup(
    dataset_name: str,
) -> dict[tuple[str, str], tuple[float, float]]:
    p = paper_ready_csv(dataset_name)

    if not p.exists():
        print(f"[WARN] paper_ready non trovato: {p}")
        print("[WARN] avg_bpp/avg_psnr saranno NaN.")
        return {}

    df = pd.read_csv(p)

    required = {"codec", "param", "bpp", "psnr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"paper_ready CSV manca colonne {missing}: {p}")

    summary = (
        df.groupby(["codec", "param"])[["bpp", "psnr"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    lookup: dict[tuple[str, str], tuple[float, float]] = {}

    for _, r in summary.iterrows():
        lookup[(str(r["codec"]), str(r["param"]))] = (
            float(r["bpp"]),
            float(r["psnr"]),
        )

    return lookup


def get_rate_quality(
    lookup: dict[tuple[str, str], tuple[float, float]],
    codec: str,
    param: str,
) -> tuple[float, float, str]:
    key = (codec, param)

    if key not in lookup:
        print(f"[WARN] rate/quality non trovati in paper_ready per {codec} {param}")
        return float("nan"), float("nan"), "missing"

    bpp, psnr = lookup[key]
    return float(bpp), float(psnr), "paper_ready"


# =============================================================================
# CSV writing
# =============================================================================


def make_row(
    dataset_name: str,
    codec: str,
    param: str,
    pipeline: str,
    n_images: int,
    n_repeats: int,
    avg_bpp: float,
    avg_psnr: float,
    rate_quality_source: str,
    energy: dict[str, float],
    idle_cpu_w: float,
    idle_gpu_w: float,
    params_m: float | None,
    is_neural: bool,
) -> dict[str, Any]:
    n_total = max(n_images * n_repeats, 1)

    cpu_total = float(energy["cpu_net"])
    gpu_total = float(energy["gpu_net"])
    total = float(energy["total_net"])
    total_time = float(energy["time_s"])

    return {
        "eval_dataset": dataset_name,
        "codec": codec,
        "param": param,
        "pipeline": pipeline,
        "n_images": int(n_images),
        "n_repeats": int(n_repeats),
        "avg_bpp": (
            round(float(avg_bpp), 9) if math.isfinite(float(avg_bpp)) else float("nan")
        ),
        "avg_psnr": (
            round(float(avg_psnr), 9)
            if math.isfinite(float(avg_psnr))
            else float("nan")
        ),
        "rate_quality_source": rate_quality_source,
        "total_time_s": round(total_time, 9),
        "energy_cpu_total_j": round(cpu_total, 9),
        "energy_gpu_total_j": round(gpu_total, 9),
        "energy_total_j": round(total, 9),
        "energy_cpu_per_image_j": round(cpu_total / n_total, 12),
        "energy_gpu_per_image_j": round(gpu_total / n_total, 12),
        "energy_per_image_j": round(total / n_total, 12),
        "idle_cpu_w": round(float(idle_cpu_w), 9),
        "idle_gpu_w": round(float(idle_gpu_w), 9),
        "params_M": None if params_m is None else round(float(params_m), 9),
        "is_neural": bool(is_neural),
    }


def write_rows_replace_existing(out_csv: Path, new_rows: list[dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if len(new_rows) == 0:
        print("[WARN] Nessuna riga da scrivere.")
        return

    new_df = pd.DataFrame(new_rows)

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

        keep_mask = [
            (str(r.eval_dataset), str(r.codec), str(r.param), str(r.pipeline))
            not in keys
            for r in old_df.itertuples(index=False)
        ]

        old_df = old_df.loc[keep_mask].copy()
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Salvato: {out_csv}")


# =============================================================================
# Idle + batch energy
# =============================================================================


def measure_idle() -> tuple[float, float]:
    sync_cuda()
    time.sleep(0.5)

    r0 = read_rapl_uj()
    g0 = read_gpu_energy_mj()
    t0 = time.perf_counter()

    time.sleep(IDLE_DURATION)

    sync_cuda()
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = read_gpu_energy_mj()

    dt = max(t1 - t0, 1e-9)

    dr = r1 - r0
    if rapl_available() and RAPL_MAX > 0 and dr < 0:
        dr += RAPL_MAX

    cpu_w = (dr / 1e6) / dt if rapl_available() else 0.0
    gpu_w = ((g1 - g0) / 1000.0) / dt if GPU_HANDLE is not None else 0.0

    return float(cpu_w), float(gpu_w)


def measure_batch_energy(
    run_fn: Callable[[], None],
    n_repeats: int,
    cpu_idle_w: float,
    gpu_idle_w: float,
    is_neural: bool,
) -> dict[str, float]:
    gc.disable()
    sync_cuda()

    r0 = read_rapl_uj()
    g0 = read_gpu_energy_mj()
    t0 = time.perf_counter()

    for _ in range(n_repeats):
        run_fn()

    sync_cuda()
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = read_gpu_energy_mj()

    gc.enable()

    dt = max(t1 - t0, 1e-9)

    dr = r1 - r0
    if rapl_available() and RAPL_MAX > 0 and dr < 0:
        dr += RAPL_MAX

    cpu_gross = dr / 1e6 if rapl_available() else 0.0
    gpu_gross = (g1 - g0) / 1000.0 if GPU_HANDLE is not None else 0.0

    cpu_net = max(cpu_gross - cpu_idle_w * dt, 0.0)
    gpu_net = max(gpu_gross - gpu_idle_w * dt, 0.0) if is_neural else 0.0

    return {
        "cpu_net": float(cpu_net),
        "gpu_net": float(gpu_net),
        "total_net": float(cpu_net + gpu_net),
        "time_s": float(dt),
        "cpu_gross": float(cpu_gross),
        "gpu_gross": float(gpu_gross),
    }


# =============================================================================
# JPEG
# =============================================================================


def benchmark_jpeg(
    dataset_name: str,
    images: list[dict[str, Any]],
    rq_lookup: dict[tuple[str, str], tuple[float, float]],
    qualities=(10, 30, 60, 85),
    n_repeats: int = 20,
) -> list[dict[str, Any]]:
    import imagecodecs

    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("JPEG | end_to_end_energy")
    print("=" * 80)

    for q in qualities:
        param = f"q={q}"
        print(f"\nJPEG {param}")

        # Warmup.
        for d in images[: min(3, len(images))]:
            imagecodecs.jpeg_decode(imagecodecs.jpeg_encode(d["rgb"], level=q))

        cpu_idle, gpu_idle = measure_idle()

        def run_all() -> None:
            for d in images:
                imagecodecs.jpeg_decode(imagecodecs.jpeg_encode(d["rgb"], level=q))

        energy = measure_batch_energy(
            run_all,
            n_repeats=n_repeats,
            cpu_idle_w=cpu_idle,
            gpu_idle_w=gpu_idle,
            is_neural=False,
        )

        avg_bpp, avg_psnr, rq_source = get_rate_quality(rq_lookup, "JPEG", param)

        row = make_row(
            dataset_name=dataset_name,
            codec="JPEG",
            param=param,
            pipeline="end_to_end_energy",
            n_images=len(images),
            n_repeats=n_repeats,
            avg_bpp=avg_bpp,
            avg_psnr=avg_psnr,
            rate_quality_source=rq_source,
            energy=energy,
            idle_cpu_w=cpu_idle,
            idle_gpu_w=gpu_idle,
            params_m=0.0,
            is_neural=False,
        )

        rows.append(row)

        print(
            f"[OK] JPEG {param} | "
            f"bpp={row['avg_bpp']} PSNR={row['avg_psnr']} "
            f"E/img={row['energy_per_image_j']} J"
        )

    return rows


# =============================================================================
# JPEG XL
# =============================================================================


def benchmark_jxl(
    dataset_name: str,
    images: list[dict[str, Any]],
    rq_lookup: dict[tuple[str, str], tuple[float, float]],
    distances=(1.0, 3.0, 7.0, 12.0),
    n_repeats: int = 10,
) -> list[dict[str, Any]]:
    import imagecodecs

    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("JXL | end_to_end_energy")
    print("=" * 80)

    for dist in distances:
        param = f"d={dist}"
        print(f"\nJXL {param}")

        # Warmup.
        for d in images[: min(3, len(images))]:
            imagecodecs.jpegxl_decode(
                imagecodecs.jpegxl_encode(d["rgb"], distance=dist, effort=5)
            )

        cpu_idle, gpu_idle = measure_idle()

        def run_all() -> None:
            for d in images:
                imagecodecs.jpegxl_decode(
                    imagecodecs.jpegxl_encode(d["rgb"], distance=dist, effort=5)
                )

        energy = measure_batch_energy(
            run_all,
            n_repeats=n_repeats,
            cpu_idle_w=cpu_idle,
            gpu_idle_w=gpu_idle,
            is_neural=False,
        )

        avg_bpp, avg_psnr, rq_source = get_rate_quality(rq_lookup, "JXL", param)

        row = make_row(
            dataset_name=dataset_name,
            codec="JXL",
            param=param,
            pipeline="end_to_end_energy",
            n_images=len(images),
            n_repeats=n_repeats,
            avg_bpp=avg_bpp,
            avg_psnr=avg_psnr,
            rate_quality_source=rq_source,
            energy=energy,
            idle_cpu_w=cpu_idle,
            idle_gpu_w=gpu_idle,
            params_m=0.0,
            is_neural=False,
        )

        rows.append(row)

        print(
            f"[OK] JXL {param} | "
            f"bpp={row['avg_bpp']} PSNR={row['avg_psnr']} "
            f"E/img={row['energy_per_image_j']} J"
        )

    return rows


# =============================================================================
# HEVC intra
# =============================================================================


def benchmark_hevc(
    dataset_name: str,
    images: list[dict[str, Any]],
    rq_lookup: dict[tuple[str, str], tuple[float, float]],
    crfs=(15, 25, 35, 45),
    n_repeats: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("HEVC | end_to_end_energy")
    print("=" * 80)

    tmp = Path("/dev/shm") / f"hevc_energy_v2_{dataset_name}_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        yuv_paths: list[Path] = []

        print("\nHEVC preprocessing RGB -> YUV444")
        for i, d in enumerate(tqdm(images, desc="hevc-prep")):
            yuv_path = tmp / f"in_{i}.yuv"

            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{d['w']}x{d['h']}",
                    "-i",
                    "pipe:0",
                    "-pix_fmt",
                    "yuv444p",
                    str(yuv_path),
                ],
                input=d["rgb"].tobytes(),
                capture_output=True,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"HEVC preprocessing failed on {d['name']}:\n"
                    f"{proc.stderr.decode(errors='ignore')[-4000:]}"
                )

            yuv_paths.append(yuv_path)

        for crf in crfs:
            param = f"crf={crf}"
            print(f"\nHEVC {param}")

            # Warmup.
            for i, d in enumerate(images[: min(2, len(images))]):
                bitstream = tmp / f"warm_{i}_crf{crf}.265"

                enc = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "yuv444p",
                        "-s",
                        f"{d['w']}x{d['h']}",
                        "-i",
                        str(yuv_paths[i]),
                        "-c:v",
                        "libx265",
                        "-preset",
                        "medium",
                        "-x265-params",
                        f"crf={crf}:keyint=1",
                        "-pix_fmt",
                        "yuv444p",
                        str(bitstream),
                    ],
                    capture_output=True,
                )

                if enc.returncode != 0:
                    raise RuntimeError(
                        f"HEVC warmup encode failed on {d['name']} {param}:\n"
                        f"{enc.stderr.decode(errors='ignore')[-4000:]}"
                    )

                dec = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(bitstream),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "pipe:1",
                    ],
                    capture_output=True,
                )

                if dec.returncode != 0:
                    raise RuntimeError(
                        f"HEVC warmup decode failed on {d['name']} {param}:\n"
                        f"{dec.stderr.decode(errors='ignore')[-4000:]}"
                    )

            cpu_idle, gpu_idle = measure_idle()

            def run_all() -> None:
                for i, d in enumerate(images):
                    bitstream = tmp / f"energy_{i}_crf{crf}.265"

                    enc = subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "yuv444p",
                            "-s",
                            f"{d['w']}x{d['h']}",
                            "-i",
                            str(yuv_paths[i]),
                            "-c:v",
                            "libx265",
                            "-preset",
                            "medium",
                            "-x265-params",
                            f"crf={crf}:keyint=1",
                            "-pix_fmt",
                            "yuv444p",
                            str(bitstream),
                        ],
                        capture_output=True,
                    )

                    if enc.returncode != 0:
                        raise RuntimeError(
                            f"HEVC encode failed on {d['name']} {param}:\n"
                            f"{enc.stderr.decode(errors='ignore')[-4000:]}"
                        )

                    dec = subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(bitstream),
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "pipe:1",
                        ],
                        capture_output=True,
                    )

                    if dec.returncode != 0:
                        raise RuntimeError(
                            f"HEVC decode failed on {d['name']} {param}:\n"
                            f"{dec.stderr.decode(errors='ignore')[-4000:]}"
                        )

            energy = measure_batch_energy(
                run_all,
                n_repeats=n_repeats,
                cpu_idle_w=cpu_idle,
                gpu_idle_w=gpu_idle,
                is_neural=False,
            )

            avg_bpp, avg_psnr, rq_source = get_rate_quality(rq_lookup, "HEVC", param)

            row = make_row(
                dataset_name=dataset_name,
                codec="HEVC",
                param=param,
                pipeline="end_to_end_energy",
                n_images=len(images),
                n_repeats=n_repeats,
                avg_bpp=avg_bpp,
                avg_psnr=avg_psnr,
                rate_quality_source=rq_source,
                energy=energy,
                idle_cpu_w=cpu_idle,
                idle_gpu_w=gpu_idle,
                params_m=0.0,
                is_neural=False,
            )

            rows.append(row)

            print(
                f"[OK] HEVC {param} | "
                f"bpp={row['avg_bpp']} PSNR={row['avg_psnr']} "
                f"E/img={row['energy_per_image_j']} J"
            )

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return rows


# =============================================================================
# Neural actual bitstream
# =============================================================================


def pad_tensor(x: torch.Tensor, multiple: int, mode: str):
    h, w = x.shape[2:]

    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple

    if mode == "center":
        pl = (new_w - w) // 2
        pr = new_w - w - pl
        pt = (new_h - h) // 2
        pb = new_h - h - pt
        padding = (pl, pr, pt, pb)
        return F.pad(x, padding), padding

    pad_h = new_h - h
    pad_w = new_w - w
    padding = (0, pad_w, 0, pad_h)
    return F.pad(x, padding), padding


def benchmark_neural_actual_energy(
    dataset_name: str,
    images: list[dict[str, Any]],
    rq_lookup: dict[tuple[str, str], tuple[float, float]],
    codec_name: str,
    param: str,
    loader_fn: Callable[[], tuple[Any, float | None]],
    pad_multiple: int,
    pad_mode: str,
    n_repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"{codec_name} {param} | actual_bitstream_energy")
    print("=" * 80)

    # Load outside try/finally so Pylance knows net exists inside the block.
    net_raw, _params_m_from_loader = loader_fn()
    net: Any = cast(Any, net_raw.to(device).eval())

    if hasattr(net, "update"):
        net.update()

    params_m = count_params_m(cast(torch.nn.Module, net_raw))

    try:
        padded: list[dict[str, Any]] = []

        for d in images:
            x_pad, padding = pad_tensor(d["tensor"], pad_multiple, pad_mode)
            padded.append(
                {
                    "x_pad": x_pad,
                    "padding": padding,
                    "h": d["h"],
                    "w": d["w"],
                }
            )

        # Sanity pass: compress/decompress all images once.
        # This is not used for quality. It only verifies that the energy path runs.
        print("sanity compress/decompress")
        with torch.no_grad():
            for p in tqdm(padded, desc="sanity"):
                enc = net.compress(p["x_pad"])
                net.decompress(enc["strings"], enc["shape"])

        sync_cuda()

        # Warmup.
        with torch.no_grad():
            for _ in range(2):
                for p in padded[: min(3, len(padded))]:
                    enc = net.compress(p["x_pad"])
                    net.decompress(enc["strings"], enc["shape"])

        sync_cuda()

        cpu_idle, gpu_idle = measure_idle()

        def run_all() -> None:
            with torch.no_grad():
                for p in padded:
                    enc = net.compress(p["x_pad"])
                    net.decompress(enc["strings"], enc["shape"])

        energy = measure_batch_energy(
            run_all,
            n_repeats=n_repeats,
            cpu_idle_w=cpu_idle,
            gpu_idle_w=gpu_idle,
            is_neural=True,
        )

        avg_bpp, avg_psnr, rq_source = get_rate_quality(rq_lookup, codec_name, param)

        row = make_row(
            dataset_name=dataset_name,
            codec=codec_name,
            param=param,
            pipeline="actual_bitstream_energy",
            n_images=len(images),
            n_repeats=n_repeats,
            avg_bpp=avg_bpp,
            avg_psnr=avg_psnr,
            rate_quality_source=rq_source,
            energy=energy,
            idle_cpu_w=cpu_idle,
            idle_gpu_w=gpu_idle,
            params_m=params_m,
            is_neural=True,
        )

        rows.append(row)

        print(
            f"[OK] {codec_name} {param} | "
            f"bpp={row['avg_bpp']} PSNR={row['avg_psnr']} "
            f"E/img={row['energy_per_image_j']} J"
        )

    finally:
        del net
        del net_raw
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))

    parser.add_argument(
        "--codecs",
        nargs="+",
        default=[
            "jpeg",
            "jxl",
            "hevc",
            "balle",
            "cheng",
            "elic",
            "tcm",
            "dcae",
        ],
        choices=[
            "jpeg",
            "jxl",
            "hevc",
            "balle",
            "cheng",
            "elic",
            "tcm",
            "dcae",
        ],
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Override output CSV path.",
    )

    parser.add_argument("--jpeg_repeats", type=int, default=20)
    parser.add_argument("--jxl_repeats", type=int, default=10)
    parser.add_argument("--hevc_repeats", type=int, default=3)

    parser.add_argument("--balle_repeats", type=int, default=5)
    parser.add_argument("--cheng_repeats", type=int, default=1)
    parser.add_argument("--elic_repeats", type=int, default=3)
    parser.add_argument("--tcm_repeats", type=int, default=3)
    parser.add_argument("--dcae_repeats", type=int, default=3)

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = DATASETS[dataset_name]

    if args.out_csv is not None:
        out_csv = Path(args.out_csv).expanduser().resolve()
    else:
        out_csv = (
            Path.home()
            / "tesi"
            / "results"
            / "images"
            / dataset_name
            / f"{dataset_name}_energy_v2.csv"
        )

    if args.overwrite and out_csv.exists():
        out_csv.unlink()

    print("=" * 80)
    print("IMAGE ENERGY V2")
    print(f"dataset: {dataset_name}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"device: {DEVICE}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"codecs: {args.codecs}")
    print(f"out_csv: {out_csv}")
    print("=" * 80)

    if not rapl_available():
        print("[WARN] RAPL non disponibile. CPU energy sarà 0.")

    if GPU_HANDLE is None:
        print("[WARN] Zeus GPU non disponibile. GPU energy sarà 0.")

    rq_lookup = load_rate_quality_lookup(dataset_name)

    images = preload_dataset(dataset_dir, limit=args.limit)

    if len(images) == 0:
        raise RuntimeError(f"Nessuna immagine trovata in {dataset_dir}")

    print(f"\nImages loaded: {len(images)}")

    all_rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Classical codecs
    # -------------------------------------------------------------------------

    if "jpeg" in args.codecs:
        all_rows.extend(
            benchmark_jpeg(
                dataset_name=dataset_name,
                images=images,
                rq_lookup=rq_lookup,
                n_repeats=args.jpeg_repeats,
            )
        )

    if "jxl" in args.codecs:
        all_rows.extend(
            benchmark_jxl(
                dataset_name=dataset_name,
                images=images,
                rq_lookup=rq_lookup,
                n_repeats=args.jxl_repeats,
            )
        )

    if "hevc" in args.codecs:
        all_rows.extend(
            benchmark_hevc(
                dataset_name=dataset_name,
                images=images,
                rq_lookup=rq_lookup,
                n_repeats=args.hevc_repeats,
            )
        )

    # -------------------------------------------------------------------------
    # Neural actual bitstream codecs
    # -------------------------------------------------------------------------

    if "balle" in args.codecs:
        for q in [1, 3, 5, 7]:
            all_rows.extend(
                benchmark_neural_actual_energy(
                    dataset_name=dataset_name,
                    images=images,
                    rq_lookup=rq_lookup,
                    codec_name="Ballé",
                    param=f"q={q}",
                    loader_fn=lambda q=q: load_balle(q),
                    pad_multiple=64,
                    pad_mode="right",
                    n_repeats=args.balle_repeats,
                )
            )

    if "cheng" in args.codecs:
        for q in [1, 3, 5]:
            all_rows.extend(
                benchmark_neural_actual_energy(
                    dataset_name=dataset_name,
                    images=images,
                    rq_lookup=rq_lookup,
                    codec_name="Cheng",
                    param=f"q={q}",
                    loader_fn=lambda q=q: load_cheng(q),
                    pad_multiple=64,
                    pad_mode="right",
                    n_repeats=args.cheng_repeats,
                )
            )

    if "elic" in args.codecs:
        for lam in ["0.008", "0.032", "0.150", "0.450"]:
            all_rows.extend(
                benchmark_neural_actual_energy(
                    dataset_name=dataset_name,
                    images=images,
                    rq_lookup=rq_lookup,
                    codec_name="ELIC",
                    param=f"lam={lam}",
                    loader_fn=lambda lam=lam: load_elic(lam),
                    pad_multiple=64,
                    pad_mode="right",
                    n_repeats=args.elic_repeats,
                )
            )

    if "tcm" in args.codecs:
        for lam in ["0.0035", "0.013"]:
            all_rows.extend(
                benchmark_neural_actual_energy(
                    dataset_name=dataset_name,
                    images=images,
                    rq_lookup=rq_lookup,
                    codec_name="TCM",
                    param=f"lam={lam}",
                    loader_fn=lambda lam=lam: load_tcm(lam),
                    pad_multiple=128,
                    pad_mode="right",
                    n_repeats=args.tcm_repeats,
                )
            )

    if "dcae" in args.codecs:
        for lam in ["0.0035", "0.013"]:
            all_rows.extend(
                benchmark_neural_actual_energy(
                    dataset_name=dataset_name,
                    images=images,
                    rq_lookup=rq_lookup,
                    codec_name="DCAE",
                    param=f"lam={lam}",
                    loader_fn=lambda lam=lam: load_dcae(lam),
                    pad_multiple=128,
                    pad_mode="center",
                    n_repeats=args.dcae_repeats,
                )
            )

    write_rows_replace_existing(out_csv, all_rows)

    print("\n" + "=" * 80)
    print("[DONE] Energy V2 completato")
    print(f"CSV: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
