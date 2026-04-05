"""
Benchmark 12 metriche per i codec MANCANTI: JPEG, HEVC, Ballé, Cheng.
Risultato: CSV compatibile con jxl_vs_elic_benchmark.csv e sota_12metric_benchmark.csv.
Poi si unisce tutto per grafici completi.

JXL + ELIC: già in jxl_vs_elic_benchmark.csv
TCM + DCAE: già in sota_12metric_benchmark.csv
"""

import os
import sys
import math
import time
import csv
import json
import subprocess
import warnings
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
OUTPUT_CSV = os.path.expanduser("~/tesi/results/images/missing_12metric_benchmark.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)
transform = transforms.ToTensor()

# ================== METRICHE ==================
from zeus.device.gpu import get_gpus
import lpips as lpips_lib
import piq
from pytorch_msssim import ms_ssim as compute_ms_ssim

gpu = get_gpus().gpus[0] if device.type == "cuda" else None
loss_fn_lpips = lpips_lib.LPIPS(net="alex").to(device)

try:
    import ssimulacra2 as ssimulacra2_mod

    HAS_SSIMULACRA2 = True
except ImportError:
    ssimulacra2_mod = None
    HAS_SSIMULACRA2 = False

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
    "codec",
    "param",
    "image",
    "bpp",
    "pipeline",
    "time_ms",
    "energy_total_j",
    "energy_net_j",
    "params_M",
]
ALL_COLS = SYSTEM_COLS + METRIC_COLS


def compute_all_metrics(x, x_hat, orig_path):
    m = {}
    m["psnr"] = piq.psnr(x, x_hat, data_range=1.0).item()
    m["ssim"] = piq.ssim(x, x_hat, data_range=1.0).item()  # type: ignore
    m["ms_ssim"] = compute_ms_ssim(x, x_hat, data_range=1.0).item()
    with torch.no_grad():
        m["lpips"] = loss_fn_lpips(x * 2 - 1, x_hat * 2 - 1).item()
    m["dists"] = piq.DISTS()(x, x_hat).item()
    m["fsim"] = piq.fsim(x, x_hat, data_range=1.0).item()
    m["gmsd"] = piq.gmsd(x, x_hat, data_range=1.0).item()
    m["vif"] = piq.vif_p(x, x_hat, data_range=1.0).item()
    m["haarpsi"] = piq.haarpsi(x, x_hat, data_range=1.0).item()
    m["dss"] = piq.dss(x, x_hat, data_range=1.0).item()
    m["mdsi"] = piq.mdsi(x, x_hat, data_range=1.0).item()

    if HAS_SSIMULACRA2 and ssimulacra2_mod is not None:
        try:
            rec_np = (
                (x_hat[0].detach().cpu().permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            tmp_path = "/tmp/ssimulacra2_temp.png"
            Image.fromarray(rec_np).save(tmp_path)
            m["ssimulacra2"] = ssimulacra2_mod.compute_ssimulacra2(
                str(orig_path), tmp_path
            )
        except Exception:
            m["ssimulacra2"] = None
    else:
        m["ssimulacra2"] = None
    return m


# ====================================================================
# JPEG
# ====================================================================
def benchmark_jpeg(img_paths, qualities=[10, 30, 60, 85]):
    import imagecodecs

    print("\n--- JPEG ---")
    results = []

    for q in qualities:
        print(f"  JPEG q={q}:")
        for i, img_path in enumerate(img_paths):
            img = np.array(Image.open(img_path).convert("RGB"))
            h, w = img.shape[:2]

            t0 = time.perf_counter()
            jpg_bytes = imagecodecs.jpeg_encode(img, level=q)
            img_rec = imagecodecs.jpeg_decode(jpg_bytes)
            t1 = time.perf_counter()

            bpp = len(jpg_bytes) * 8 / (h * w)
            x = (
                torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                / 255.0
            )
            x_hat = (
                torch.from_numpy(img_rec)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
                / 255.0
            )

            metrics = compute_all_metrics(x, x_hat, str(img_path))
            metrics.update(
                {
                    "codec": "JPEG",
                    "param": f"q={q}",
                    "image": Path(img_path).name,
                    "bpp": bpp,
                    "pipeline": "end_to_end",
                    "time_ms": (t1 - t0) * 1000,
                    "energy_total_j": None,
                    "energy_net_j": None,
                    "params_M": 0,
                }
            )
            results.append(metrics)

            if (i + 1) % 12 == 0:
                print(
                    f"    [{i+1}/{len(img_paths)}] bpp={bpp:.3f} PSNR={metrics['psnr']:.2f}"
                )

        sub = [r for r in results if r["param"] == f"q={q}"]
        print(
            f"    MEDIA: bpp={np.mean([r['bpp'] for r in sub]):.3f} PSNR={np.mean([r['psnr'] for r in sub]):.2f} SSIM2={np.mean([r['ssimulacra2'] for r in sub if r['ssimulacra2'] is not None]):.1f}"
        )

    return results


# ====================================================================
# HEVC Intra
# ====================================================================
def benchmark_hevc(img_paths, crfs=[15, 25, 35, 45]):
    print("\n--- HEVC Intra ---")
    results = []

    for crf in crfs:
        print(f"  HEVC crf={crf}:")
        for i, img_path in enumerate(img_paths):
            img_pil = Image.open(img_path).convert("RGB")
            img_rgb = np.array(img_pil)
            h, w = img_rgb.shape[:2]

            # Encode via ffmpeg
            tmp_yuv = "/dev/shm/hevc_input.yuv"
            tmp_hevc = "/dev/shm/hevc_output.265"
            tmp_rec_yuv = "/dev/shm/hevc_rec.yuv"

            # RGB -> YUV444
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-i",
                    "pipe:0",
                    "-pix_fmt",
                    "yuv444p",
                    tmp_yuv,
                ],
                input=img_rgb.tobytes(),
                capture_output=True,
            )

            t0 = time.perf_counter()
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv444p",
                    "-s",
                    f"{w}x{h}",
                    "-i",
                    tmp_yuv,
                    "-c:v",
                    "libx265",
                    "-preset",
                    "medium",
                    "-x265-params",
                    f"crf={crf}:keyint=1",
                    "-pix_fmt",
                    "yuv444p",
                    tmp_hevc,
                ],
                capture_output=True,
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_hevc,
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                capture_output=True,
            )

            # Decode
            dec_result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_hevc,
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                capture_output=True,
            )
            t1 = time.perf_counter()

            if dec_result.returncode != 0 or len(dec_result.stdout) < h * w * 3:
                continue

            img_rec = np.frombuffer(dec_result.stdout, dtype=np.uint8).reshape(h, w, 3)
            file_bytes = os.path.getsize(tmp_hevc)
            bpp = file_bytes * 8 / (h * w)

            x = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
                / 255.0
            )
            x_hat = (
                torch.from_numpy(img_rec)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
                / 255.0
            )

            metrics = compute_all_metrics(x, x_hat, str(img_path))
            metrics.update(
                {
                    "codec": "HEVC",
                    "param": f"crf={crf}",
                    "image": Path(img_path).name,
                    "bpp": bpp,
                    "pipeline": "end_to_end",
                    "time_ms": (t1 - t0) * 1000,
                    "energy_total_j": None,
                    "energy_net_j": None,
                    "params_M": 0,
                }
            )
            results.append(metrics)

            if (i + 1) % 12 == 0:
                print(
                    f"    [{i+1}/{len(img_paths)}] bpp={bpp:.3f} PSNR={metrics['psnr']:.2f}"
                )

            # Cleanup
            for f in [tmp_yuv, tmp_hevc, tmp_rec_yuv]:
                if os.path.exists(f):
                    os.remove(f)

        sub = [r for r in results if r["param"] == f"crf={crf}"]
        if sub:
            print(
                f"    MEDIA: bpp={np.mean([r['bpp'] for r in sub]):.3f} PSNR={np.mean([r['psnr'] for r in sub]):.2f}"
            )

    return results


# ====================================================================
# Ballé (CompressAI)
# ====================================================================
def benchmark_balle(img_paths, qualities=[1, 3, 5, 7]):
    from compressai.zoo import bmshj2018_hyperprior

    print("\n--- Ballé 2018 Hyperprior ---")
    results = []
    IDLE_W = 59.8

    for q in qualities:
        print(f"  Ballé q={q}:")
        net = (
            bmshj2018_hyperprior(quality=q, metric="mse", pretrained=True)
            .to(device)
            .eval()
        )
        net.update()
        pm = sum(p.numel() for p in net.parameters()) / 1e6

        # Warmup
        with torch.no_grad():
            net.forward(torch.randn(1, 3, 256, 256).to(device))
        torch.cuda.synchronize()

        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            h, w = x.shape[2:]
            pad_h = (64 - h % 64) % 64
            pad_w = (64 - w % 64) % 64
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))

            torch.cuda.synchronize()
            before = gpu.getTotalEnergyConsumption() if gpu else 0
            t0 = time.perf_counter()
            with torch.no_grad():
                out = net.forward(x_pad)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            after = gpu.getTotalEnergyConsumption() if gpu else 0

            x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
            n_px = h * w
            bpp = sum(
                torch.log(l).sum() / (-math.log(2) * n_px)
                for l in out["likelihoods"].values()
            ).item()  # type: ignore
            elapsed = t1 - t0
            e_total = (after - before) / 1000
            e_net = e_total - IDLE_W * elapsed

            metrics = compute_all_metrics(x, x_hat, str(img_path))
            metrics.update(
                {
                    "codec": "Ballé",
                    "param": f"q={q}",
                    "image": Path(img_path).name,
                    "bpp": bpp,
                    "pipeline": "forward_pass_only",
                    "time_ms": elapsed * 1000,
                    "energy_total_j": round(e_total, 4),
                    "energy_net_j": round(e_net, 4),
                    "params_M": round(pm, 1),
                }
            )
            results.append(metrics)

            if (i + 1) % 12 == 0:
                print(
                    f"    [{i+1}/{len(img_paths)}] bpp={bpp:.3f} PSNR={metrics['psnr']:.2f}"
                )

        sub = [r for r in results if r["param"] == f"q={q}"]
        print(
            f"    MEDIA: bpp={np.mean([r['bpp'] for r in sub]):.3f} PSNR={np.mean([r['psnr'] for r in sub]):.2f} SSIM2={np.mean([r['ssimulacra2'] for r in sub if r['ssimulacra2'] is not None]):.1f}"
        )

        del net
        torch.cuda.empty_cache()

    return results


# ====================================================================
# Cheng 2020 (CompressAI)
# ====================================================================
def benchmark_cheng(img_paths, qualities=[1, 3, 5]):
    from compressai.zoo import cheng2020_attn

    print("\n--- Cheng 2020 Attention ---")
    results = []
    IDLE_W = 59.8

    for q in qualities:
        print(f"  Cheng q={q}:")
        net = cheng2020_attn(quality=q, metric="mse", pretrained=True).to(device).eval()
        net.update()
        pm = sum(p.numel() for p in net.parameters()) / 1e6

        # Warmup
        with torch.no_grad():
            net.forward(torch.randn(1, 3, 256, 256).to(device))
        torch.cuda.synchronize()

        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            h, w = x.shape[2:]
            pad_h = (64 - h % 64) % 64
            pad_w = (64 - w % 64) % 64
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))

            torch.cuda.synchronize()
            before = gpu.getTotalEnergyConsumption() if gpu else 0
            t0 = time.perf_counter()
            with torch.no_grad():
                out = net.forward(x_pad)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            after = gpu.getTotalEnergyConsumption() if gpu else 0

            x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
            n_px = h * w
            bpp = sum(
                torch.log(l).sum() / (-math.log(2) * n_px)
                for l in out["likelihoods"].values()
            ).item()  # type: ignore
            elapsed = t1 - t0
            e_total = (after - before) / 1000
            e_net = e_total - IDLE_W * elapsed

            metrics = compute_all_metrics(x, x_hat, str(img_path))
            metrics.update(
                {
                    "codec": "Cheng",
                    "param": f"q={q}",
                    "image": Path(img_path).name,
                    "bpp": bpp,
                    "pipeline": "forward_pass_only",
                    "time_ms": elapsed * 1000,
                    "energy_total_j": round(e_total, 4),
                    "energy_net_j": round(e_net, 4),
                    "params_M": round(pm, 1),
                }
            )
            results.append(metrics)

            if (i + 1) % 12 == 0:
                print(
                    f"    [{i+1}/{len(img_paths)}] bpp={bpp:.3f} PSNR={metrics['psnr']:.2f}"
                )

        sub = [r for r in results if r["param"] == f"q={q}"]
        print(
            f"    MEDIA: bpp={np.mean([r['bpp'] for r in sub]):.3f} PSNR={np.mean([r['psnr'] for r in sub]):.2f} SSIM2={np.mean([r['ssimulacra2'] for r in sub if r['ssimulacra2'] is not None]):.1f}"
        )

        del net
        torch.cuda.empty_cache()

    return results


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 70)
    print("BENCHMARK 12 METRICHE: CODEC MANCANTI (JPEG, HEVC, Ballé, Cheng)")
    print("=" * 70)

    imgs = sorted(Path(KODAK_DIR).glob("*.png"))
    print(f"Kodak: {len(imgs)} immagini")
    print(f"SSIMULACRA2: {'OK' if HAS_SSIMULACRA2 else 'non disponibile'}")

    all_results = []

    # Classici
    all_results.extend(benchmark_jpeg(imgs))
    all_results.extend(benchmark_hevc(imgs))

    # Neurali (CompressAI)
    all_results.extend(benchmark_balle(imgs))
    all_results.extend(benchmark_cheng(imgs))

    # Salva CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nCSV: {OUTPUT_CSV}")

    # Tabella riepilogativa
    print(f"\n{'=' * 90}")
    print("RIEPILOGO (medie Kodak)")
    print(f"{'=' * 90}")
    print(
        f"{'Codec':<18} {'bpp':>6} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'SSIM2':>7} {'ms':>6}"
    )
    print("-" * 60)

    groups = set((r["codec"], r["param"]) for r in all_results)
    for codec, param in sorted(groups):
        sub = [r for r in all_results if r["codec"] == codec and r["param"] == param]
        bpp = np.mean([r["bpp"] for r in sub])
        psnr = np.mean([r["psnr"] for r in sub])
        ssim = np.mean([r["ssim"] for r in sub])
        lpips_v = np.mean([r["lpips"] for r in sub])
        ssim2_vals = [r["ssimulacra2"] for r in sub if r["ssimulacra2"] is not None]
        ssim2 = np.mean(ssim2_vals) if ssim2_vals else None
        t_vals = [r["time_ms"] for r in sub if r["time_ms"] is not None]
        t = np.mean(t_vals) if t_vals else None

        s2_str = f"{ssim2:.1f}" if ssim2 is not None else "N/A"
        t_str = f"{t:.0f}" if t is not None else "N/A"
        print(
            f"{codec+' '+param:<18} {bpp:>6.3f} {psnr:>6.2f} {ssim:>6.4f} {lpips_v:>6.4f} {s2_str:>7} {t_str:>6}"
        )


if __name__ == "__main__":
    main()
