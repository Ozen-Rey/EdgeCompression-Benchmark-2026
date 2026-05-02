import os
import time
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
import imagecodecs

from .metrics import compute_all_metrics_np


def row_base(dataset_name, codec, param, img_path, img_np, bpp, time_ms, pipeline):
    h, w = img_np.shape[:2]
    return {
        "dataset": dataset_name,
        "codec": codec,
        "param": param,
        "image": Path(img_path).name,
        "width": w,
        "height": h,
        "pixels": h * w,
        "bpp": bpp,
        "pipeline": pipeline,
        "time_ms": time_ms,
        "energy_total_j": None,
        "energy_net_j": None,
        "params_M": 0,
    }


def compress_jpeg_metrics(dataset_name, img_path, quality):
    img_np = np.array(Image.open(img_path).convert("RGB"))
    h, w = img_np.shape[:2]

    t0 = time.perf_counter()
    jpg_bytes = imagecodecs.jpeg_encode(img_np, level=quality)
    rec_np = imagecodecs.jpeg_decode(jpg_bytes)
    t1 = time.perf_counter()

    bpp = len(jpg_bytes) * 8 / (h * w)

    row = row_base(
        dataset_name=dataset_name,
        codec="JPEG",
        param=f"q={quality}",
        img_path=img_path,
        img_np=img_np,
        bpp=bpp,
        time_ms=(t1 - t0) * 1000,
        pipeline="end_to_end",
    )

    row.update(compute_all_metrics_np(img_np, rec_np, img_path))
    return row


def compress_jxl_metrics(dataset_name, img_path, distance):
    img_np = np.array(Image.open(img_path).convert("RGB"))
    h, w = img_np.shape[:2]

    t0 = time.perf_counter()
    jxl_bytes = imagecodecs.jpegxl_encode(img_np, distance=distance, effort=5)
    rec_np = imagecodecs.jpegxl_decode(jxl_bytes)
    t1 = time.perf_counter()

    bpp = len(jxl_bytes) * 8 / (h * w)

    row = row_base(
        dataset_name=dataset_name,
        codec="JXL",
        param=f"d={distance}",
        img_path=img_path,
        img_np=img_np,
        bpp=bpp,
        time_ms=(t1 - t0) * 1000,
        pipeline="end_to_end",
    )

    row.update(compute_all_metrics_np(img_np, rec_np, img_path))
    return row


def compress_hevc_metrics(dataset_name, img_path, crf):
    img_np = np.array(Image.open(img_path).convert("RGB"))
    h, w = img_np.shape[:2]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        raw_in = tmpdir / "input.rgb"
        bitstream = tmpdir / "output.265"

        raw_in.write_bytes(img_np.tobytes())

        enc_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
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
            "ffmpeg",
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
        subprocess.run(enc_cmd, capture_output=True, check=True)
        dec = subprocess.run(dec_cmd, capture_output=True, check=True)
        t1 = time.perf_counter()

        rec_np = np.frombuffer(dec.stdout, dtype=np.uint8).reshape(h, w, 3)
        bpp = os.path.getsize(bitstream) * 8 / (h * w)

    row = row_base(
        dataset_name=dataset_name,
        codec="HEVC",
        param=f"crf={crf}",
        img_path=img_path,
        img_np=img_np,
        bpp=bpp,
        time_ms=(t1 - t0) * 1000,
        pipeline="end_to_end",
    )

    row.update(compute_all_metrics_np(img_np, rec_np, img_path))
    return row
