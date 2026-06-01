"""Kodak image mini-benchmark validation for local R-D-E sanity checks.

This script is intentionally smaller than the thesis image benchmark. It runs a
few operating points on Kodak, writes a router-ready CSV schema, records local
provenance, and optionally replays the R-D-E router.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCRIPT_VERSION = "0.46.5"
DATASET_NAME = "kodak"
DEFAULT_CODECS = "jpeg,jxl,hevc,dcae"
DEFAULT_IMAGE_GLOB = "*.png"

# Quality metric selection. PSNR needs only numpy and is the low-friction
# default for the self-contained benchmark setup; SSIMULACRA2 needs the optional
# ssimulacra2 package. Each metric maps to the matching built-in router
# domain spec so the router can be replayed on the produced CSV.
DEFAULT_QUALITY_METRIC = "psnr"
QUALITY_METRIC_COLUMNS = {"ssimulacra2": "ssimulacra2", "psnr": "psnr"}
DOMAIN_SPEC_FOR_METRIC = {
    "ssimulacra2": "image_ssimulacra2",
    "psnr": "image_psnr",
}
# The default router quality thresholds are SSIMULACRA2-scaled (50/80/90), which
# are infeasible for PSNR (dB). Point the PSNR router replay at PSNR-appropriate
# image thresholds so the on-the-spot replay yields feasible decisions.
QUALITY_THRESHOLDS_FOR_METRIC = {
    "ssimulacra2": None,
    "psnr": "configs/quality_thresholds_psnr.json",
}

# Energy proxy. On hosts without RAPL/NVML telemetry there is no measured
# energy, so router-valid rows cannot be produced. When --energy-proxy time is
# set we fill energy_per_image_j from a time-proportional proxy using a fixed
# reference power, and label every such row energy_provenance=time_proxy_non_measured.
# This is explicitly NOT a measurement; it only lets the router be exercised
# end-to-end where no telemetry exists. It is off by default so real
# measurement runs are never silently polluted.
TIME_PROXY_REFERENCE_POWER_W = 15.0
ENERGY_PROVENANCE_TIME_PROXY = "time_proxy_non_measured"

JPEG_QUALITIES = [30, 50, 70, 85, 95]
JXL_DISTANCES = [1, 2, 4, 6, 8]
HEVC_CRFS = [18, 23, 28, 33, 38]
DCAE_LAMBDAS = ["0.0035", "0.013"]

BOUNDARIES = [
    "This is a local mini-benchmark validation, not the main thesis benchmark.",
    "It does not claim hardware-invariant energy measurements.",
    "It does not redistribute Kodak images, checkpoints, or codec binaries.",
    "It is intended to check whether the R-D-E measurement pipeline can be reapplied on a controlled small image set.",
    "Generated outputs are local artifacts and should not be committed.",
]

CSV_COLUMNS = [
    "dataset",
    "image_id",
    "image_path",
    "width",
    "height",
    "codec",
    "config",
    "rate_bpp",
    "quality_metric",
    "ssimulacra2",
    "psnr",
    "energy_j_per_image",
    "time_ms",
    "compressed_size_bytes",
    "status",
    "error",
    "bpp",
    "energy_per_image_j",
    "pixels",
    "encode_time_ms",
    "decode_time_ms",
    "energy_cpu_j",
    "energy_gpu_j",
    "energy_provenance",
]

ROUTER_PROFILES = [
    "balanced",
    "energy-limited",
    "bandwidth-limited",
    "quality-first",
]


@dataclass(frozen=True)
class OperatingPoint:
    codec_key: str
    codec_label: str
    config: str
    value: Any


@dataclass
class Measurement:
    width: int
    height: int
    bpp: float
    ssimulacra2: Optional[float]
    compressed_size_bytes: int
    time_ms: float
    psnr: Optional[float] = None
    encode_time_ms: Optional[float] = None
    decode_time_ms: Optional[float] = None


@dataclass
class EnergyResult:
    energy_j: Optional[float]
    cpu_j: Optional[float]
    gpu_j: Optional[float]
    backend: str
    method: str
    warnings: list[str]
    scope: str
    measured: bool


class NoopEnergyMeter:
    name = "none"

    def start(self) -> dict[str, Any]:
        return {"t0": time.perf_counter()}

    def stop(self, state: dict[str, Any]) -> EnergyResult:
        return EnergyResult(
            energy_j=None,
            cpu_j=None,
            gpu_j=None,
            backend="none",
            method="unavailable",
            warnings=["energy_backend_none_rde_incomplete"],
            scope="none",
            measured=False,
        )


def parse_codecs(raw: str) -> list[str]:
    aliases = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "jpegxl": "jxl",
        "jxl": "jxl",
        "hevc": "hevc",
        "x265": "hevc",
        "dcae": "dcae",
    }
    codecs: list[str] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token not in aliases:
            raise ValueError(f"Unknown codec {item!r}; expected jpeg,jxl,hevc,dcae")
        normalized = aliases[token]
        if normalized not in codecs:
            codecs.append(normalized)
    return codecs


def operating_points_for_codecs(codecs: Iterable[str]) -> list[OperatingPoint]:
    points: list[OperatingPoint] = []
    selected = set(codecs)
    if "jpeg" in selected:
        points.extend(
            OperatingPoint("jpeg", "JPEG", f"q={q}", q) for q in JPEG_QUALITIES
        )
    if "jxl" in selected:
        points.extend(
            OperatingPoint("jxl", "JXL", f"distance={d}", d) for d in JXL_DISTANCES
        )
    if "hevc" in selected:
        points.extend(
            OperatingPoint("hevc", "HEVC", f"crf={crf}", crf) for crf in HEVC_CRFS
        )
    if "dcae" in selected:
        points.extend(
            OperatingPoint("dcae", "DCAE", f"lam={lam}", lam) for lam in DCAE_LAMBDAS
        )
    return points


def parse_image_globs(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or [DEFAULT_IMAGE_GLOB]


def find_images(kodak_dir: Path, patterns: Iterable[str], max_images: Optional[int]) -> list[Path]:
    seen: set[Path] = set()
    images: list[Path] = []
    for pattern in patterns:
        for path in sorted(kodak_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                images.append(path)
                seen.add(path)
    if max_images is not None:
        images = images[:max_images]
    return images


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def read_completed_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        keys = set()
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                keys.add(
                    (
                        row.get("dataset", ""),
                        row.get("image_id", ""),
                        row.get("codec", ""),
                        row.get("config", ""),
                    )
                )
        return keys


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def compute_psnr(orig_np: Any, rec_np: Any, warnings: list[str]) -> Optional[float]:
    """Compute RGB PSNR (dB) from two uint8 arrays using numpy only."""
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy is a benchmark dependency
        warning = f"psnr_unavailable:{exc.__class__.__name__}"
        if warning not in warnings:
            warnings.append(warning)
        return None

    try:
        orig = np.asarray(orig_np)
        rec = np.asarray(rec_np)
        if orig.shape != rec.shape:
            warnings.append(f"psnr_shape_mismatch:{orig.shape}!={rec.shape}")
            return None
        diff = orig.astype(np.float64) - rec.astype(np.float64)
        mse = float(np.mean(diff * diff))
        if mse <= 0.0:
            return 100.0
        return float(10.0 * np.log10((255.0 ** 2) / mse))
    except Exception as exc:
        warnings.append(f"psnr_failed:{exc.__class__.__name__}:{exc}")
        return None


def compute_ssimulacra2(orig_path: Path, rec_np: Any, warnings: list[str]) -> Optional[float]:
    try:
        import ssimulacra2 as ssimulacra2_mod  # type: ignore
        from PIL import Image
    except Exception as exc:
        warning = f"ssimulacra2_unavailable:{exc.__class__.__name__}"
        if warning not in warnings:
            warnings.append(warning)
        return None

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        Image.fromarray(rec_np).save(tmp_path)
        return float(ssimulacra2_mod.compute_ssimulacra2(str(orig_path), tmp_path))
    except Exception as exc:
        warnings.append(f"ssimulacra2_failed:{exc.__class__.__name__}:{exc}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def encode_decode_jpeg(image_path: Path, quality: int, warnings: list[str]) -> Measurement:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        img_np = np.array(image.convert("RGB"))
    height, width = img_np.shape[:2]

    try:
        import imagecodecs  # type: ignore

        t0 = time.perf_counter()
        encoded = imagecodecs.jpeg_encode(img_np, level=quality)
        t1 = time.perf_counter()
        rec_np = imagecodecs.jpeg_decode(encoded)
        t2 = time.perf_counter()
        backend = "imagecodecs"
    except Exception as exc:
        warnings.append(f"jpeg_imagecodecs_fallback_pillow:{exc.__class__.__name__}")
        buf = io.BytesIO()
        image = Image.fromarray(img_np)
        t0 = time.perf_counter()
        image.save(buf, format="JPEG", quality=quality)
        t1 = time.perf_counter()
        buf.seek(0)
        rec_np = np.array(Image.open(buf).convert("RGB"))
        t2 = time.perf_counter()
        encoded = buf.getvalue()
        backend = "pillow"

    if backend == "pillow":
        warnings.append("jpeg_backend_pillow_local_fallback")

    size = len(encoded)
    ssim2 = compute_ssimulacra2(image_path, rec_np, warnings)
    psnr = compute_psnr(img_np, rec_np, warnings)
    return Measurement(
        width=width,
        height=height,
        bpp=size * 8 / (width * height),
        ssimulacra2=ssim2,
        psnr=psnr,
        compressed_size_bytes=size,
        time_ms=(t2 - t0) * 1000,
        encode_time_ms=(t1 - t0) * 1000,
        decode_time_ms=(t2 - t1) * 1000,
    )


def encode_decode_jxl(image_path: Path, distance: int, warnings: list[str]) -> Measurement:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image:
        img_np = np.array(image.convert("RGB"))
    height, width = img_np.shape[:2]

    try:
        import imagecodecs  # type: ignore

        t0 = time.perf_counter()
        encoded = imagecodecs.jpegxl_encode(img_np, distance=distance, effort=5)
        t1 = time.perf_counter()
        rec_np = imagecodecs.jpegxl_decode(encoded)
        t2 = time.perf_counter()
        size = len(encoded)
    except Exception:
        cjxl = shutil.which("cjxl")
        djxl = shutil.which("djxl")
        if not cjxl or not djxl:
            raise RuntimeError("JPEG XL unavailable: imagecodecs jpegxl and cjxl/djxl are missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "input.png"
            bitstream = tmp / "output.jxl"
            rec = tmp / "recon.png"
            Image.fromarray(img_np).save(src)
            t0 = time.perf_counter()
            enc = subprocess.run(
                [cjxl, str(src), str(bitstream), "-d", str(distance), "-e", "5"],
                capture_output=True,
                text=True,
            )
            t1 = time.perf_counter()
            if enc.returncode != 0:
                raise RuntimeError(f"cjxl failed: {enc.stderr[-1000:]}")
            dec = subprocess.run(
                [djxl, str(bitstream), str(rec)],
                capture_output=True,
                text=True,
            )
            t2 = time.perf_counter()
            if dec.returncode != 0:
                raise RuntimeError(f"djxl failed: {dec.stderr[-1000:]}")
            rec_np = np.array(Image.open(rec).convert("RGB"))
            size = bitstream.stat().st_size
            warnings.append("jxl_backend_cjxl_djxl_local_fallback")

    ssim2 = compute_ssimulacra2(image_path, rec_np, warnings)
    psnr = compute_psnr(img_np, rec_np, warnings)
    return Measurement(
        width=width,
        height=height,
        bpp=size * 8 / (width * height),
        ssimulacra2=ssim2,
        psnr=psnr,
        compressed_size_bytes=size,
        time_ms=(t2 - t0) * 1000,
        encode_time_ms=(t1 - t0) * 1000,
        decode_time_ms=(t2 - t1) * 1000,
    )


def encode_decode_hevc(image_path: Path, crf: int, warnings: list[str]) -> Measurement:
    import numpy as np
    from PIL import Image

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("HEVC unavailable: ffmpeg missing")

    with Image.open(image_path) as image:
        img_np = np.array(image.convert("RGB"))
    height, width = img_np.shape[:2]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_in = tmp / "input.rgb"
        bitstream = tmp / "output.265"
        raw_in.write_bytes(img_np.tobytes())

        enc_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            str(raw_in),
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-x265-params",
            f"crf={crf}:keyint=1:log-level=error",
            "-pix_fmt",
            "yuv444p",
            str(bitstream),
        ]
        dec_cmd = [
            ffmpeg,
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
        ]

        t0 = time.perf_counter()
        enc = subprocess.run(enc_cmd, capture_output=True, text=True)
        t1 = time.perf_counter()
        if enc.returncode != 0:
            raise RuntimeError(f"ffmpeg libx265 encode failed: {enc.stderr[-1000:]}")
        dec = subprocess.run(dec_cmd, capture_output=True)
        t2 = time.perf_counter()
        if dec.returncode != 0:
            stderr = dec.stderr.decode(errors="ignore") if isinstance(dec.stderr, bytes) else str(dec.stderr)
            raise RuntimeError(f"ffmpeg HEVC decode failed: {stderr[-1000:]}")
        rec_np = np.frombuffer(dec.stdout, dtype=np.uint8).reshape(height, width, 3)
        size = bitstream.stat().st_size

    ssim2 = compute_ssimulacra2(image_path, rec_np, warnings)
    psnr = compute_psnr(img_np, rec_np, warnings)
    return Measurement(
        width=width,
        height=height,
        bpp=size * 8 / (width * height),
        ssimulacra2=ssim2,
        psnr=psnr,
        compressed_size_bytes=size,
        time_ms=(t2 - t0) * 1000,
        encode_time_ms=(t1 - t0) * 1000,
        decode_time_ms=(t2 - t1) * 1000,
    )


def resolve_device(requested: str, warnings: list[str]) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        warnings.append(f"torch_unavailable_for_cuda_probe:{exc.__class__.__name__}")

    if requested == "cuda" and not cuda_available:
        warnings.append("CUDA requested for DCAE but torch.cuda.is_available() is false")
        return "cuda"
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


def encode_decode_dcae(
    image_path: Path,
    lam: str,
    requested_device: str,
    warnings: list[str],
    cache: dict[str, Any],
) -> Measurement:
    try:
        import torch  # type: ignore
        from src.benchmark.image_v2 import codecs_neural_actual as neural
    except Exception as exc:
        raise RuntimeError(f"DCAE unavailable: torch/image_v2 import failed: {exc}") from exc

    resolved = resolve_device(requested_device, warnings)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DCAE unavailable: device=cuda requested but CUDA is not available")

    neural.device = torch.device(resolved)
    neural.DEVICE = resolved

    if lam not in cache:
        try:
            cache[lam] = neural.load_dcae(lam)
        except Exception as exc:
            raise RuntimeError(f"DCAE unavailable for lam={lam}: {exc}") from exc

    net, params_m = cache[lam]
    row = neural.benchmark_neural_actual_one(
        net=net,
        dataset_name=DATASET_NAME,
        codec_name="DCAE",
        param=f"lam={lam}",
        img_path=image_path,
        params_m=params_m,
        disable_cudnn=True,
    )
    width = int(row["width"])
    height = int(row["height"])
    bpp = float(row["bpp"])
    return Measurement(
        width=width,
        height=height,
        bpp=bpp,
        ssimulacra2=safe_float(row.get("ssimulacra2")),
        psnr=safe_float(row.get("psnr")),
        compressed_size_bytes=max(int(round(bpp * width * height / 8)), 0),
        time_ms=float(row["time_ms"]),
    )


def build_energy_meter(kind: str, report_warnings: list[str]) -> Any:
    if kind == "none":
        report_warnings.append("energy_backend=none; R-D-E output has incomplete energy measurements")
        return NoopEnergyMeter()
    try:
        from src.utils.energy_backends import (
            CompositeEnergyMeter,
            LinuxRaplBackend,
            NoEnergyBackend,
            NvidiaNvmlEnergyBackend,
            NvidiaNvmlPowerSamplerBackend,
        )
    except Exception as exc:
        report_warnings.append(f"energy_backends_unavailable:{exc.__class__.__name__}:{exc}")
        return NoopEnergyMeter()

    if kind == "auto":
        return CompositeEnergyMeter()
    if kind == "both":
        return CompositeEnergyMeter()
    if kind == "cpu":
        return CompositeEnergyMeter(cpu_backend=LinuxRaplBackend(), gpu_backend=NoEnergyBackend())
    if kind == "gpu":
        gpu = NvidiaNvmlEnergyBackend()
        if not gpu.available():
            gpu = NvidiaNvmlPowerSamplerBackend()
        return CompositeEnergyMeter(cpu_backend=NoEnergyBackend(), gpu_backend=gpu)
    raise ValueError(f"Unknown energy backend {kind}")


def stop_energy(meter: Any, state: Any) -> EnergyResult:
    reading = meter.stop(state)
    if isinstance(reading, EnergyResult):
        return reading
    return EnergyResult(
        energy_j=safe_float(getattr(reading, "total_j", None)),
        cpu_j=safe_float(getattr(reading, "cpu_j", None)),
        gpu_j=safe_float(getattr(reading, "gpu_j", None)),
        backend=str(getattr(reading, "energy_backend", "unknown")),
        method=str(getattr(reading, "energy_method", "unknown")),
        warnings=list(getattr(reading, "warnings", [])),
        scope=str(getattr(reading, "energy_scope", "unknown")),
        measured=bool(getattr(reading, "energy_is_measured", False)),
    )


def run_with_energy(
    operation: Callable[[], Measurement],
    meter: Any,
    warmup: int,
    repeats: int,
) -> tuple[Measurement, EnergyResult]:
    for _ in range(max(warmup, 0)):
        operation()

    measurements: list[Measurement] = []
    energy_results: list[EnergyResult] = []
    for _ in range(max(repeats, 1)):
        state = meter.start()
        try:
            measurement = operation()
        finally:
            energy = stop_energy(meter, state)
        measurements.append(measurement)
        energy_results.append(energy)

    last = measurements[-1]
    last.time_ms = sum(m.time_ms for m in measurements) / len(measurements)
    numeric_energy = [e.energy_j for e in energy_results if e.energy_j is not None]
    cpu = [e.cpu_j for e in energy_results if e.cpu_j is not None]
    gpu = [e.gpu_j for e in energy_results if e.gpu_j is not None]
    merged_warnings: list[str] = []
    for energy in energy_results:
        merged_warnings.extend(energy.warnings)
    energy = energy_results[-1]
    energy.energy_j = sum(numeric_energy) / len(numeric_energy) if numeric_energy else None
    energy.cpu_j = sum(cpu) / len(cpu) if cpu else None
    energy.gpu_j = sum(gpu) / len(gpu) if gpu else None
    energy.warnings = list(dict.fromkeys(merged_warnings))
    return last, energy


def resolve_energy_provenance(
    measurement: Measurement,
    energy: EnergyResult,
    proxy_mode: str,
) -> str:
    """Decide the energy provenance for a row, applying a time proxy if asked.

    Mutates ``energy`` in place when a proxy is applied so make_row writes the
    proxied value. Returns the provenance label for the CSV/report.
    """
    if energy.energy_j is not None:
        return (energy.backend or "measured") if energy.measured else (energy.backend or "reported_unverified")
    if proxy_mode == "time" and measurement.time_ms is not None:
        energy.energy_j = max(float(measurement.time_ms), 0.0) / 1000.0 * TIME_PROXY_REFERENCE_POWER_W
        energy.cpu_j = None
        energy.gpu_j = None
        return ENERGY_PROVENANCE_TIME_PROXY
    return "none"


def make_row(
    image_path: Path,
    op: OperatingPoint,
    measurement: Optional[Measurement],
    energy: Optional[EnergyResult],
    status: str,
    error: str = "",
    quality_metric: str = DEFAULT_QUALITY_METRIC,
    energy_provenance: str = "",
) -> dict[str, Any]:
    if measurement is None:
        try:
            width, height = image_size(image_path)
        except Exception:
            width, height = 0, 0
        measurement = Measurement(
            width=width,
            height=height,
            bpp=0.0,
            ssimulacra2=None,
            compressed_size_bytes=0,
            time_ms=0.0,
        )
    energy_j = energy.energy_j if energy else None
    return {
        "dataset": DATASET_NAME,
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "width": measurement.width,
        "height": measurement.height,
        "codec": op.codec_label,
        "config": op.config,
        "rate_bpp": measurement.bpp if status == "ok" else "",
        "quality_metric": quality_metric,
        "ssimulacra2": measurement.ssimulacra2 if status == "ok" and measurement.ssimulacra2 is not None else "",
        "psnr": measurement.psnr if status == "ok" and measurement.psnr is not None else "",
        "energy_j_per_image": energy_j if status == "ok" and energy_j is not None else "",
        "time_ms": measurement.time_ms if status == "ok" else "",
        "compressed_size_bytes": measurement.compressed_size_bytes if status == "ok" else "",
        "status": status,
        "error": error,
        "bpp": measurement.bpp if status == "ok" else "",
        "energy_per_image_j": energy_j if status == "ok" and energy_j is not None else "",
        "pixels": measurement.width * measurement.height,
        "encode_time_ms": measurement.encode_time_ms if measurement.encode_time_ms is not None and status == "ok" else "",
        "decode_time_ms": measurement.decode_time_ms if measurement.decode_time_ms is not None and status == "ok" else "",
        "energy_cpu_j": energy.cpu_j if energy and energy.cpu_j is not None and status == "ok" else "",
        "energy_gpu_j": energy.gpu_j if energy and energy.gpu_j is not None and status == "ok" else "",
        "energy_provenance": energy_provenance if status == "ok" else "",
    }


def operation_for_point(
    op: OperatingPoint,
    image_path: Path,
    device: str,
    warnings: list[str],
    dcae_cache: dict[str, Any],
) -> Callable[[], Measurement]:
    if op.codec_key == "jpeg":
        return lambda: encode_decode_jpeg(image_path, int(op.value), warnings)
    if op.codec_key == "jxl":
        return lambda: encode_decode_jxl(image_path, int(op.value), warnings)
    if op.codec_key == "hevc":
        return lambda: encode_decode_hevc(image_path, int(op.value), warnings)
    if op.codec_key == "dcae":
        return lambda: encode_decode_dcae(image_path, str(op.value), device, warnings, dcae_cache)
    raise ValueError(f"Unsupported operating point {op}")


def collect_nvidia_smi_snapshot() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "warning": "nvidia-smi missing"}
    try:
        completed = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"available": False, "warning": f"nvidia-smi failed:{exc}"}


def collect_cuda_availability() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        return {
            "torch_available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        return {
            "torch_available": False,
            "cuda_available": False,
            "warning": f"torch import failed:{exc.__class__.__name__}:{exc}",
        }


def collect_platform_fingerprint() -> dict[str, Any]:
    try:
        from src.router.adaptation.system_probe import probe_system

        return probe_system()
    except Exception as exc:
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": sys.version,
            },
            "warning": f"system_probe_failed:{exc.__class__.__name__}:{exc}",
        }


def codec_availability_summary(codecs: Iterable[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for codec in codecs:
        if codec == "jpeg":
            summary[codec] = {"available": True, "backend": "imagecodecs_or_pillow"}
        elif codec == "jxl":
            imagecodecs_ok = False
            try:
                import imagecodecs  # type: ignore

                imagecodecs_ok = hasattr(imagecodecs, "jpegxl_encode")
            except Exception:
                imagecodecs_ok = False
            summary[codec] = {
                "available": imagecodecs_ok or (shutil.which("cjxl") and shutil.which("djxl")),
                "backend": "imagecodecs_jpegxl_or_cjxl_djxl",
                "cjxl": shutil.which("cjxl"),
                "djxl": shutil.which("djxl"),
            }
        elif codec == "hevc":
            summary[codec] = {
                "available": bool(shutil.which("ffmpeg")),
                "backend": "ffmpeg_libx265",
                "ffmpeg": shutil.which("ffmpeg"),
            }
        elif codec == "dcae":
            roots = [
                Path("~/tesi/external_codecs/DCAE").expanduser(),
                Path("/tmp/DCAE"),
            ]
            root = next((p for p in roots if p.exists()), None)
            summary[codec] = {
                "available": root is not None,
                "backend": "src.benchmark.image_v2.codecs_neural_actual.load_dcae",
                "root": str(root) if root else None,
                "note": "checkpoint availability is verified when the first DCAE operating point loads",
            }
    return summary


def quality_column_for_row(row: dict[str, str]) -> str:
    metric = str(row.get("quality_metric", "") or "").strip().lower()
    return QUALITY_METRIC_COLUMNS.get(metric, "ssimulacra2")


def csv_valid_for_router(row: dict[str, str]) -> bool:
    if row.get("status") != "ok":
        return False
    quality_column = quality_column_for_row(row)
    for column in ("bpp", quality_column, "energy_per_image_j", "time_ms"):
        if str(row.get(column, "")).strip() == "":
            return False
    return True


def count_router_valid_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if csv_valid_for_router(row))


def write_router_input_subset(csv_path: Path, subset_path: Path) -> int:
    count = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as src, subset_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            if csv_valid_for_router(row):
                writer.writerow(row)
                count += 1
    return count


def run_router_profiles(
    csv_path: Path,
    out_dir: Path,
    report: dict[str, Any],
    domain_spec: str = DOMAIN_SPEC_FOR_METRIC[DEFAULT_QUALITY_METRIC],
    quality_thresholds_file: Optional[str] = None,
) -> None:
    subset_path = out_dir / "_router_valid_rows.csv"
    valid = write_router_input_subset(csv_path, subset_path)
    thresholds_path = (
        quality_thresholds_file
        if quality_thresholds_file and Path(quality_thresholds_file).exists()
        else None
    )
    report["router_replay"] = {
        "enabled": True,
        "valid_rows": valid,
        "domain_spec": domain_spec,
        "quality_thresholds_file": thresholds_path,
        "profiles": {},
    }
    if valid == 0:
        report["router_replay"]["warning"] = "no rows with numeric bpp/quality/energy/time; router skipped"
        return

    for profile in ROUTER_PROFILES:
        safe = profile.replace("-", "_")
        report_path = out_dir / f"router_{safe}_report.json"
        summary_path = out_dir / f"router_{safe}_summary.csv"
        cmd = [
            sys.executable,
            "-m",
            "src.router.rde_router",
            "--csv",
            str(subset_path),
            "--domain-spec",
            domain_spec,
            "--profile",
            profile,
            "--out",
            str(report_path),
            "--summary-out",
            str(summary_path),
        ]
        if thresholds_path:
            cmd.extend(["--quality-thresholds-file", thresholds_path])
        completed = subprocess.run(cmd, capture_output=True, text=True)
        report["router_replay"]["profiles"][profile] = {
            "returncode": completed.returncode,
            "report": str(report_path),
            "summary": str(summary_path),
            "stderr_tail": completed.stderr[-2000:],
        }


def load_rows_for_plots(csv_path: Path, quality_column: str = "psnr") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            parsed = dict(row)
            for column in ("bpp", quality_column, "energy_per_image_j", "time_ms"):
                parsed[column] = safe_float(row.get(column))
            # Generic "quality" key so the plotting code is metric-agnostic.
            parsed["quality"] = parsed.get(quality_column)
            if parsed["bpp"] is None or parsed["quality"] is None:
                continue
            rows.append(parsed)
    return rows


def scatter_plot(rows: list[dict[str, Any]], x: str, y: str, xlabel: str, ylabel: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    codecs = sorted({str(row["codec"]) for row in rows})
    for codec in codecs:
        xs = [row[x] for row in rows if row["codec"] == codec and row.get(x) is not None and row.get(y) is not None]
        ys = [row[y] for row in rows if row["codec"] == codec and row.get(x) is not None and row.get(y) is not None]
        if xs and ys:
            ax.scatter(xs, ys, label=codec, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if codecs:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def boxplot_energy(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    codecs = sorted({str(row["codec"]) for row in rows})
    data = [
        [row["energy_per_image_j"] for row in rows if row["codec"] == codec and row.get("energy_per_image_j") is not None]
        for codec in codecs
    ]
    data = [values for values in data if values]
    labels = [codec for codec in codecs if any(row["codec"] == codec and row.get("energy_per_image_j") is not None for row in rows)]
    fig, ax = plt.subplots(figsize=(7, 5))
    if data:
        ax.boxplot(data, labels=labels)
    ax.set_xlabel("Codec")
    ax.set_ylabel("Energy (J/image)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_router_profile_selection(out_dir: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts: dict[str, dict[str, int]] = {}
    for profile in ROUTER_PROFILES:
        summary = out_dir / f"router_{profile.replace('-', '_')}_summary.csv"
        if not summary.exists():
            continue
        with summary.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        profile_counts: dict[str, int] = {}
        for row in rows:
            codec = row.get("selected_codec") or row.get("codec") or ""
            if codec:
                profile_counts[codec] = profile_counts.get(codec, 0) + 1
        counts[profile] = profile_counts

    if not counts:
        return

    codecs = sorted({codec for profile_counts in counts.values() for codec in profile_counts})
    x_positions = list(range(len(counts)))
    bottom = [0] * len(counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    for codec in codecs:
        values = [counts[profile].get(codec, 0) for profile in counts]
        ax.bar(x_positions, values, bottom=bottom, label=codec)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(counts.keys()), rotation=20, ha="right")
    ax.set_ylabel("Selected rows")
    ax.set_xlabel("Router profile")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_plots(
    csv_path: Path,
    out_dir: Path,
    include_router: bool,
    quality_metric: str = DEFAULT_QUALITY_METRIC,
) -> list[str]:
    quality_column = QUALITY_METRIC_COLUMNS.get(quality_metric, "ssimulacra2")
    quality_label = quality_metric.upper()
    rows = load_rows_for_plots(csv_path, quality_column)
    plot_dir = out_dir / "plots"
    paths = [
        ("rate_quality.png", "bpp", "quality", "Rate (bpp)", quality_label),
        ("energy_quality.png", "energy_per_image_j", "quality", "Energy (J/image)", quality_label),
        ("rate_energy.png", "bpp", "energy_per_image_j", "Rate (bpp)", "Energy (J/image)"),
        ("time_quality.png", "time_ms", "quality", "Time (ms/image)", quality_label),
    ]
    generated: list[str] = []
    if rows:
        for filename, x, y, xlabel, ylabel in paths:
            if any(row.get(x) is not None and row.get(y) is not None for row in rows):
                path = plot_dir / filename
                scatter_plot(rows, x, y, xlabel, ylabel, path)
                generated.append(str(path))
        if any(row.get("energy_per_image_j") is not None for row in rows):
            path = plot_dir / "codec_energy_boxplot.png"
            boxplot_energy(rows, path)
            generated.append(str(path))
    if include_router:
        path = plot_dir / "router_profile_selection.png"
        plot_router_profile_selection(out_dir, path)
        if path.exists():
            generated.append(str(path))
    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local Kodak image mini-benchmark validation for R-D-E sanity checks."
    )
    parser.add_argument("--kodak-dir", required=True, type=Path, help="Directory containing the 24 Kodak images.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Local output directory.")
    parser.add_argument("--codecs", default=DEFAULT_CODECS, help="Comma-separated codec list. Default: jpeg,jxl,hevc,dcae.")
    parser.add_argument(
        "--quality-metric",
        choices=sorted(QUALITY_METRIC_COLUMNS),
        default=DEFAULT_QUALITY_METRIC,
        help=(
            "Quality metric used for router-valid rows and replay. "
            "'psnr' needs only numpy; 'ssimulacra2' needs the ssimulacra2 package. "
            f"Default: {DEFAULT_QUALITY_METRIC}."
        ),
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional image limit for pilot/debug runs.")
    parser.add_argument("--image-glob", default=DEFAULT_IMAGE_GLOB, help="Comma-separated glob patterns. Default: *.png.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="DCAE device.")
    parser.add_argument("--energy-backend", choices=["auto", "cpu", "gpu", "both", "none"], default="auto")
    parser.add_argument(
        "--energy-proxy",
        choices=["off", "time"],
        default="off",
        help=(
            "Fallback when energy is not measured. 'time' fills missing energy "
            "with a labeled time-proportional proxy (energy_provenance="
            "time_proxy_non_measured) so the router can be replayed without "
            "telemetry. Off by default to avoid polluting real measurements."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1, help="Measured repeats per image/config.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repeats before measurement.")
    parser.add_argument("--randomize-order", action="store_true", help="Randomize codec/config/image order.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without encoding.")
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate diagnostic PNG plots.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if a codec/config fails.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already completed in the output CSV.")
    parser.add_argument("--run-router", action="store_true", help="Replay the router on valid measured rows.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        codecs = parse_codecs(args.codecs)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = args.out_dir.expanduser().resolve()
    csv_path = out_dir / "kodak_image_rde_mini_router_ready.csv"
    report_path = out_dir / "kodak_image_rde_mini_report.json"
    report_warnings: list[str] = []

    images = find_images(
        args.kodak_dir.expanduser().resolve(),
        parse_image_globs(args.image_glob),
        args.max_images,
    )
    points = operating_points_for_codecs(codecs)
    tasks = [(image, op) for image in images for op in points]
    if args.randomize_order:
        rng = random.Random(args.seed)
        rng.shuffle(tasks)

    availability = codec_availability_summary(codecs)
    nvidia_smi = collect_nvidia_smi_snapshot()
    if not nvidia_smi.get("available"):
        report_warnings.append(str(nvidia_smi.get("warning", "nvidia-smi unavailable")))
    cuda = collect_cuda_availability()
    if args.device == "cuda" and not cuda.get("cuda_available"):
        report_warnings.append("CUDA missing but DCAE was requested with device=cuda")

    try:
        from src.router.version import ROUTER_VERSION
    except Exception:
        ROUTER_VERSION = "unknown"

    report: dict[str, Any] = {
        "timestamp": now_utc_iso(),
        "script_version": SCRIPT_VERSION,
        "repo_version": SCRIPT_VERSION,
        "router_version": ROUTER_VERSION,
        "command_line": " ".join(sys.argv if argv is None else [sys.argv[0], *argv]),
        "dataset_root": str(args.kodak_dir),
        "number_of_images_found": len(images),
        "number_of_images_processed": 0,
        "codec_list": codecs,
        "operating_points": {
            "jpeg": [f"q={q}" for q in JPEG_QUALITIES],
            "jxl": [f"distance={d}" for d in JXL_DISTANCES],
            "hevc": [f"crf={crf}" for crf in HEVC_CRFS],
            "dcae": [f"lam={lam}" for lam in DCAE_LAMBDAS],
        },
        "skipped_codecs": {},
        "failures": [],
        "warnings": report_warnings,
        "energy_backend": {
            "requested": args.energy_backend,
            "used": None,
            "method": None,
            "scope": None,
            "per_image_measurement": True,
            "normalization": "per image/config measurement; repeats are averaged when repeats > 1",
            "proxy_mode": args.energy_proxy,
            "proxy_reference_power_w": TIME_PROXY_REFERENCE_POWER_W if args.energy_proxy == "time" else None,
            "proxy_rows": 0,
            "proxy_note": (
                "time-proportional proxy fills missing energy and is labeled "
                "energy_provenance=time_proxy_non_measured; it is not a measurement"
            ),
        },
        "metric_backend": {
            "quality_metric": args.quality_metric,
            "router_domain_spec": DOMAIN_SPEC_FOR_METRIC[args.quality_metric],
            "metric_zoo": False,
        },
        "device_info": {"requested": args.device, "cuda": cuda},
        "platform_fingerprint": collect_platform_fingerprint(),
        "nvidia_smi_snapshot": nvidia_smi,
        "cuda_availability": cuda,
        "codec_availability": availability,
        "boundaries": BOUNDARIES,
        "reuse_note": {
            "reused_modules": [
                "src.utils.energy_backends",
                "src.router.adaptation.system_probe",
                "src.router.rde_router",
                "src.benchmark.image_v2.codecs_neural_actual for DCAE when available",
            ],
            "new_parts": [
                "Kodak mini-benchmark CLI orchestration",
                "router-ready CSV/report/diagnostic plot generation",
                "small operating-point selection for JPEG/JXL/HEVC/DCAE",
            ],
            "local_fallbacks": [
                "Pillow JPEG fallback when imagecodecs JPEG is unavailable",
                "cjxl/djxl JPEG XL fallback when imagecodecs JPEG XL is unavailable",
                "energy_backend=none when hardware counters are unavailable or explicitly disabled",
            ],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Dry run: {len(images)} images, {len(points)} operating points, {len(tasks)} tasks.")
        for image, op in tasks[:50]:
            print(f"{image.name}: {op.codec_label} {op.config}")
        if len(tasks) > 50:
            print(f"... {len(tasks) - 50} more tasks")
        report["dry_run"] = True
        report["planned_tasks"] = len(tasks)
        write_json(report_path, report)
        return 0

    completed = read_completed_keys(csv_path) if args.resume else set()
    meter = build_energy_meter(args.energy_backend, report_warnings)
    dcae_cache: dict[str, Any] = {}
    strict_failed = False
    processed_images: set[str] = set()
    proxy_rows = 0

    for image_path, op in tasks:
        key = (DATASET_NAME, image_path.stem, op.codec_label, op.config)
        if key in completed:
            continue
        warnings_for_row: list[str] = []
        try:
            operation = operation_for_point(op, image_path, args.device, warnings_for_row, dcae_cache)
            measurement, energy = run_with_energy(operation, meter, args.warmup, args.repeats)
            quality_value = getattr(measurement, args.quality_metric)
            if quality_value is None:
                raise RuntimeError(
                    f"{args.quality_metric.upper()} unavailable or failed; quality row not router-valid"
                )
            energy_provenance = resolve_energy_provenance(measurement, energy, args.energy_proxy)
            if energy_provenance == ENERGY_PROVENANCE_TIME_PROXY:
                proxy_rows += 1
            row = make_row(
                image_path, op, measurement, energy, "ok",
                quality_metric=args.quality_metric,
                energy_provenance=energy_provenance,
            )
            if energy.energy_j is None:
                report_warnings.append("energy measurement unavailable; R-D-E rows are incomplete for router replay")
            report["energy_backend"]["used"] = energy.backend
            report["energy_backend"]["method"] = energy.method
            report["energy_backend"]["scope"] = energy.scope
            report_warnings.extend(warnings_for_row)
            report_warnings.extend(energy.warnings)
            processed_images.add(image_path.name)
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            if op.codec_key in {"jxl", "hevc", "dcae"} and op.codec_key not in report["skipped_codecs"]:
                report["skipped_codecs"][op.codec_key] = {
                    "status": "unavailable_or_failed",
                    "first_error": error,
                }
            failure = {
                "image": str(image_path),
                "codec": op.codec_label,
                "config": op.config,
                "error": error,
            }
            report["failures"].append(failure)
            row = make_row(image_path, op, None, None, "failed", error, quality_metric=args.quality_metric)
            strict_failed = True
            print(f"[WARN] {op.codec_label} {op.config} failed on {image_path.name}: {error}", file=sys.stderr)
            if os.environ.get("KODAK_MINI_DEBUG_TRACEBACK"):
                traceback.print_exc()
        append_csv_row(csv_path, row)

    report["energy_backend"]["proxy_rows"] = proxy_rows
    if proxy_rows > 0:
        report_warnings.append(
            f"energy_provenance=time_proxy_non_measured applied to {proxy_rows} rows; "
            "these energy values are a time-scaled proxy, not a measurement"
        )
    report["number_of_images_processed"] = len(processed_images)
    report["warnings"] = sorted(set(report_warnings))
    report["router_valid_rows"] = count_router_valid_rows(csv_path)

    if args.run_router:
        run_router_profiles(
            csv_path,
            out_dir,
            report,
            domain_spec=DOMAIN_SPEC_FOR_METRIC[args.quality_metric],
            quality_thresholds_file=QUALITY_THRESHOLDS_FOR_METRIC.get(args.quality_metric),
        )
    else:
        report["router_replay"] = {"enabled": False}

    if not args.skip_plots:
        try:
            report["plots"] = generate_plots(
                csv_path,
                out_dir,
                include_router=args.run_router,
                quality_metric=args.quality_metric,
            )
        except Exception as exc:
            report["plots"] = []
            report["warnings"].append(f"plot_generation_failed:{exc.__class__.__name__}:{exc}")
    else:
        report["plots"] = []

    write_json(report_path, report)
    if strict_failed and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
