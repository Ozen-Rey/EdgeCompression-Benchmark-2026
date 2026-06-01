import os
import io
import sys
import time
import subprocess
import tempfile
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pytorch_msssim import ms_ssim
import lpips
import imagecodecs

loss_fn_lpips = lpips.LPIPS(net="alex").cuda()

sys.path.insert(0, os.path.expanduser("~/tesi"))
from src.utils.energy_monitor import EnergyMonitor

KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
RESULTS_DIR = os.path.expanduser("~/tesi/results/images")


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_msssim(img1_np, img2_np):
    t1 = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ms_ssim(t1, t2, data_range=1.0).item()


def compute_lpips(img1_np, img2_np):
    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t * 2.0 - 1.0  # LPIPS vuole [-1, 1]
        return t.cuda()

    with torch.no_grad():
        d = loss_fn_lpips(to_tensor(img1_np), to_tensor(img2_np))
    return d.item()


def measure_baseline(seconds=3):
    with EnergyMonitor() as em:
        time.sleep(seconds)
    return em.joules() / seconds


def compress_jpeg(img_path, quality):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    total_pixels = w * h
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    compressed_size_bits = buf.tell() * 8
    bpp = compressed_size_bits / total_pixels
    buf.seek(0)
    img_rec = Image.open(buf).convert("RGB")
    orig_np = np.array(img)
    rec_np = np.array(img_rec)
    return {
        "bpp": bpp,
        "psnr": psnr(orig_np, rec_np),
        "ms_ssim": compute_msssim(orig_np, rec_np),
        "lpips": compute_lpips(orig_np, rec_np),
    }


def compress_hevc(img_path, crf):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    total_pixels = w * h
    with tempfile.TemporaryDirectory() as tmpdir:
        input_png = os.path.join(tmpdir, "input.png")
        output_hevc = os.path.join(tmpdir, "output.hevc")
        output_png = os.path.join(tmpdir, "output.png")
        img.save(input_png)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_png,
                "-c:v",
                "libx265",
                "-crf",
                str(crf),
                "-preset",
                "medium",
                "-x265-params",
                "log-level=error",
                output_hevc,
            ],
            capture_output=True,
            check=True,
        )
        compressed_size_bits = os.path.getsize(output_hevc) * 8
        bpp = compressed_size_bits / total_pixels
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_hevc, output_png],
            capture_output=True,
            check=True,
        )
        img_rec = Image.open(output_png).convert("RGB")
        orig_np = np.array(img)
        rec_np = np.array(img_rec)
        return {
            "bpp": bpp,
            "psnr": psnr(orig_np, rec_np),
            "ms_ssim": compute_msssim(orig_np, rec_np),
            "lpips": compute_lpips(orig_np, rec_np),
        }


def run_benchmark(codec, param_name, param_values, compress_fn):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    images = sorted(Path(KODAK_DIR).glob("*.png"))
    rows = []

    for val in param_values:
        start_time = time.time()
        batch_results = [compress_fn(img_path, val) for img_path in images]
        elapsed = time.time() - start_time
        fps = len(images) / elapsed
        ms_per_image = elapsed / len(images) * 1000

        for img_path, metrics in zip(images, batch_results):
            rows.append(
                {
                    "image": img_path.name,
                    "codec": codec,
                    param_name: val,
                    **metrics,
                    "fps": round(fps, 3),
                    "ms_per_image": round(ms_per_image, 2),
                }
            )

        print(
            f"{codec} {param_name}={val} | "
            f"bpp={np.mean([r['bpp'] for r in batch_results]):.3f} | "
            f"psnr={np.mean([r['psnr'] for r in batch_results]):.2f} dB | "
            f"fps={fps:.1f} | ms/img={ms_per_image:.1f}"
        )

    df = pd.DataFrame(rows)
    rd_curve = df.groupby(param_name)[
        ["bpp", "psnr", "ms_ssim", "lpips", "fps", "ms_per_image"]
    ].mean()
    print(f"\nCurva R-D {codec} su Kodak:")
    print(rd_curve.to_string())

    out_path = os.path.join(RESULTS_DIR, f"{codec.lower()}_kodak.csv")
    df.to_csv(out_path, index=False)
    return df

def compress_jxl(img_path, distance=1.0):
    """
    Codec Classico SOTA: JPEG XL (JXL)
    distance: 0.0 = lossless, 1.0 = visually lossless (default), fino a 15+ (molto lossy)
    """
    img = np.array(Image.open(img_path).convert('RGB'))
    
    # effort=5 è il bilanciamento standard di JXL per velocità/compressione
    start = time.perf_counter()
    jxl_bytes = imagecodecs.jpegxl_encode(img, distance=distance, effort=5)
    enc_ms = (time.perf_counter() - start) * 1000
    
    img_rec = imagecodecs.jpegxl_decode(jxl_bytes)
    
    # JXL restituisce l'immagine in RGB, ce l'abbiamo già pronta come array
    bpp = (len(jxl_bytes) * 8) / (img.shape[0] * img.shape[1])
    
    psnr_val = psnr(img, img_rec)
    lpips_val = compute_lpips(img, img_rec)
    
    return {
        "bpp": round(bpp, 4),
        "psnr": round(psnr_val, 2),
        "lpips": round(lpips_val, 4),
        "enc_ms": round(enc_ms, 2)
    }


if __name__ == "__main__":
    print("=== JPEG ===")
    run_benchmark(
        "JPEG", "quality", [10, 20, 30, 40, 50, 60, 70, 80, 90, 95], compress_jpeg
    )

    print("\n=== H.265/HEVC ===")
    run_benchmark("HEVC", "crf", [51, 45, 40, 35, 30, 25, 20, 15], compress_hevc)


