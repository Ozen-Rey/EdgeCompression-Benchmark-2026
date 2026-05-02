import os
import time
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image

from .metrics import compute_all_metrics_np


def compress_jpeg_ai_metrics(
    dataset_name,
    img_path,
    target_bpp,
    jpeg_ai_root,
    conda_env=None,
    encoder_module="src.reco.coders.encoder",
    decoder_module="src.reco.coders.decoder",
    extra_encoder_args=None,
):
    if extra_encoder_args is None:
        extra_encoder_args = []

    img = Image.open(img_path).convert("RGB")
    orig_np = np.array(img)
    h, w = orig_np.shape[:2]

    target_bppm100 = int(round(target_bpp * 100))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        input_png = tmpdir / "input.png"
        bitstream = tmpdir / "output.jai"
        rec_png = tmpdir / "rec.png"

        img.save(input_png)

        if conda_env is None:
            py_prefix = ["python"]
        else:
            py_prefix = ["conda", "run", "-n", conda_env, "python"]

        enc_cmd = (
            py_prefix
            + [
                "-m",
                encoder_module,
                str(input_png),
                str(bitstream),
                "--set_target_bpp",
                str(target_bppm100),
            ]
            + extra_encoder_args
        )

        dec_cmd = py_prefix + [
            "-m",
            decoder_module,
            str(bitstream),
            str(rec_png),
        ]

        t0 = time.perf_counter()

        enc = subprocess.run(
            enc_cmd,
            cwd=os.path.expanduser(str(jpeg_ai_root)),
            capture_output=True,
            text=True,
        )

        if enc.returncode != 0:
            raise RuntimeError(
                "JPEG AI encoder failed\n"
                f"image={img_path}\n"
                f"target_bpp={target_bpp}\n"
                f"cmd={' '.join(enc_cmd)}\n\n"
                f"STDOUT:\n{enc.stdout}\n\n"
                f"STDERR:\n{enc.stderr}"
            )

        dec = subprocess.run(
            dec_cmd,
            cwd=os.path.expanduser(str(jpeg_ai_root)),
            capture_output=True,
            text=True,
        )

        if dec.returncode != 0:
            raise RuntimeError(
                "JPEG AI decoder failed\n"
                f"image={img_path}\n"
                f"target_bpp={target_bpp}\n"
                f"cmd={' '.join(dec_cmd)}\n\n"
                f"STDOUT:\n{dec.stdout}\n\n"
                f"STDERR:\n{dec.stderr}"
            )

        t1 = time.perf_counter()

        if not rec_png.exists():
            raise RuntimeError(
                f"JPEG AI decoder did not create reconstructed image: {rec_png}"
            )

        if not bitstream.exists():
            raise RuntimeError(f"JPEG AI encoder did not create bitstream: {bitstream}")

        rec_np = np.array(Image.open(rec_png).convert("RGB"))
        bpp = os.path.getsize(bitstream) * 8 / (h * w)

    row = {
        "dataset": dataset_name,
        "codec": "JPEG_AI",
        "param": f"target_bpp={target_bpp}",
        "image": Path(img_path).name,
        "width": w,
        "height": h,
        "pixels": h * w,
        "bpp": bpp,
        "pipeline": "end_to_end",
        "time_ms": (t1 - t0) * 1000,
        "energy_total_j": None,
        "energy_net_j": None,
        "params_M": None,
    }

    row.update(compute_all_metrics_np(orig_np, rec_np, img_path))
    return row
