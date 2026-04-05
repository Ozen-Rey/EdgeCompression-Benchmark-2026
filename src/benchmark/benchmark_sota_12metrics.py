"""
Benchmark percettivo completo TCM (CVPR 2023) e DCAE (CVPR 2025)
su Kodak (24 immagini) con le stesse 12 metriche del benchmark JXL vs ELIC.

12 metriche: PSNR, SSIM, MS-SSIM, LPIPS, DISTS, SSIMULACRA2,
             FSIM, GMSD, VIF, HaarPSI, DSS, MDSI

Requisiti:
  - /tmp/LIC_TCM clonato con checkpoints
  - /tmp/DCAE clonato con checkpoints
  - piq, lpips, pytorch_msssim, ssimulacra2, zeus-ml
"""

import sys
import os
import math
import time
import csv
import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
OUTPUT_CSV = os.path.expanduser("~/tesi/results/images/sota_12metric_benchmark.csv")
OUTPUT_JSON = os.path.expanduser("~/tesi/results/images/sota_12metric_benchmark.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDLE_W = 59.8  # GPU idle power

TCM_CHECKPOINTS = {
    "0.013": ("/tmp/LIC_TCM/checkpoints/tcm_mse_0013.pth.tar", 64),
    "0.0035": ("/tmp/LIC_TCM/checkpoints/tcm_mse_0035.pth.tar", 64),
}

DCAE_CHECKPOINTS = {
    "0.013": ("/tmp/DCAE/checkpoints/dcae_mse_0013.pth.tar", 128),
    "0.0035": ("/tmp/DCAE/checkpoints/dcae_mse_0035.pth.tar", 128),
}

device = torch.device(DEVICE)
transform = transforms.ToTensor()

# ================== METRICHE SETUP ==================
print("=" * 70)
print("BENCHMARK 12 METRICHE: TCM + DCAE su Kodak")
print("=" * 70)

from zeus.device.gpu import get_gpus
import lpips as lpips_lib
import piq
from pytorch_msssim import ms_ssim as compute_ms_ssim

gpu = get_gpus().gpus[0] if device.type == "cuda" else None
loss_fn_lpips = lpips_lib.LPIPS(net="alex").to(device)

# SSIMULACRA2
try:
    import ssimulacra2 as ssimulacra2_mod

    HAS_SSIMULACRA2 = True
    print("SSIMULACRA2: OK")
except ImportError:
    ssimulacra2_mod = None
    HAS_SSIMULACRA2 = False
    print("SSIMULACRA2: non disponibile")

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
    """x, x_hat: [1,3,H,W] su device, range [0,1]. 12 metriche."""
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

    if HAS_SSIMULACRA2:
        try:
            rec_np = (
                (x_hat[0].detach().cpu().permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            tmp_path = "/tmp/ssimulacra2_temp_rec.png"
            Image.fromarray(rec_np).save(tmp_path)
            m["ssimulacra2"] = ssimulacra2_mod.compute_ssimulacra2(  # type: ignore
                str(orig_path), tmp_path
            )
        except Exception as e:
            m["ssimulacra2"] = None
    else:
        m["ssimulacra2"] = None

    return m


def benchmark_codec(net, codec_name, lam, img_paths, pad_multiple, params_m):
    """Benchmark forward pass con 12 metriche su tutto Kodak."""
    results = []

    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        h, w = x.shape[2:]
        pad_h = (pad_multiple - h % pad_multiple) % pad_multiple
        pad_w = (pad_multiple - w % pad_multiple) % pad_multiple
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))

        # Timing + energy
        torch.cuda.synchronize()
        before = gpu.getTotalEnergyConsumption() if gpu else 0
        t0 = time.perf_counter()

        with torch.no_grad():
            out = net.forward(x_pad)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        after = gpu.getTotalEnergyConsumption() if gpu else 0

        x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
        num_pixels = h * w
        bpp = sum(
            torch.log(l).sum() / (-math.log(2) * num_pixels)
            for l in out["likelihoods"].values()
        ).item()  # type: ignore

        elapsed = t1 - t0
        e_total = (after - before) / 1000
        e_net = e_total - IDLE_W * elapsed

        # 12 metriche
        metrics = compute_all_metrics(x, x_hat, str(img_path))
        metrics.update(
            {
                "codec": codec_name,
                "param": f"lam={lam}",
                "image": Path(img_path).name,
                "bpp": bpp,
                "pipeline": "forward_pass_only",
                "time_ms": elapsed * 1000,
                "energy_total_j": round(e_total, 4),
                "energy_net_j": round(e_net, 4),
                "params_M": params_m,
            }
        )
        results.append(metrics)

        if (i + 1) % 8 == 0:
            print(
                f"    [{i+1}/{len(img_paths)}] bpp={bpp:.3f} PSNR={metrics['psnr']:.2f} "
                f"LPIPS={metrics['lpips']:.4f} SSIM2={metrics.get('ssimulacra2', 'N/A')}"
            )

    return results


def print_summary(results, codec, param):
    subset = [r for r in results if r["codec"] == codec and r["param"] == param]
    if not subset:
        return

    print(f"\n  MEDIA {codec} {param}:")
    bpp = np.mean([r["bpp"] for r in subset])
    print(f"    bpp={bpp:.3f}")

    for col in METRIC_COLS:
        vals = [r[col] for r in subset if r.get(col) is not None]
        if vals:
            print(f"    {col}: {np.mean(vals):.4f}")


def main():
    imgs = sorted(Path(KODAK_DIR).glob("*.png"))
    print(f"\nKodak: {len(imgs)} immagini\n")

    all_results = []

    # === TCM (CVPR 2023) ===
    print("=" * 70)
    print("TCM (CVPR 2023)")
    print("=" * 70)
    sys.path.insert(0, "/tmp/LIC_TCM")
    from models import TCM  # type: ignore

    for lam, (ckpt_path, pad) in TCM_CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [!] {ckpt_path} non trovato")
            continue

        print(f"\n--- TCM λ={lam} ---")
        net = TCM(
            config=[2, 2, 2, 2, 2, 2],
            head_dim=[8, 16, 32, 32, 16, 8],
            drop_path_rate=0.0,
            N=64,
            M=320,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ckpt["state_dict"])
        net = net.to(device).eval()
        net.update()
        pm = sum(p.numel() for p in net.parameters()) / 1e6

        # Warmup
        dummy = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            net.forward(dummy)
        torch.cuda.synchronize()

        res = benchmark_codec(net, "TCM", lam, imgs, pad, round(pm, 1))
        all_results.extend(res)
        print_summary(all_results, "TCM", f"lam={lam}")

        del net
        torch.cuda.empty_cache()

    # === DCAE (CVPR 2025) ===
    print("\n" + "=" * 70)
    print("DCAE (CVPR 2025)")
    print("=" * 70)

    # Pulisci import TCM
    if "/tmp/LIC_TCM" in sys.path:
        sys.path.remove("/tmp/LIC_TCM")
    for mod in list(sys.modules.keys()):
        if "models" in mod and "compressai" not in mod:
            del sys.modules[mod]

    sys.path.insert(0, "/tmp/DCAE")
    from models import DCAE  # type: ignore

    for lam, (ckpt_path, pad) in DCAE_CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [!] {ckpt_path} non trovato")
            continue

        print(f"\n--- DCAE λ={lam} ---")
        net = DCAE()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        dictory = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        net.load_state_dict(dictory)
        net = net.to(device).eval()
        net.update()
        pm = sum(p.numel() for p in net.parameters()) / 1e6

        # Warmup
        dummy = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            net.forward(dummy)
        torch.cuda.synchronize()

        res = benchmark_codec(net, "DCAE", lam, imgs, pad, round(pm, 1))
        all_results.extend(res)
        print_summary(all_results, "DCAE", f"lam={lam}")

        del net
        torch.cuda.empty_cache()

    # === SALVA CSV ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nCSV: {OUTPUT_CSV}")

    # === JSON MEDIE ===
    summary = {}
    groups = set((r["codec"], r["param"]) for r in all_results)
    for codec, param in sorted(groups):
        subset = [r for r in all_results if r["codec"] == codec and r["param"] == param]
        key = f"{codec}_{param}"
        summary[key] = {"n_images": len(subset)}
        for col in METRIC_COLS + ["bpp", "time_ms", "energy_net_j"]:
            vals = [r[col] for r in subset if r.get(col) is not None]
            summary[key][col] = round(float(np.mean(vals)), 6) if vals else None

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON: {OUTPUT_JSON}")

    # === TABELLA CONFRONTO ===
    print(f"\n{'=' * 100}")
    print("CONFRONTO 12 METRICHE (medie Kodak 24 img)")
    print(f"{'=' * 100}")
    print(
        f"{'Config':<22} {'bpp':>5} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'DISTS':>6} "
        f"{'SSIM2':>6} {'FSIM':>6} {'VIF':>6} {'HaarP':>6} {'GMSD':>6} {'DSS':>6} {'MDSI':>6}"
    )
    print("-" * 100)

    for key in sorted(summary.keys()):
        s = summary[key]

        def fv(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        print(
            f"{key:<22} {fv(s['bpp'],'.3f'):>5} {fv(s['psnr'],'.2f'):>6} "
            f"{fv(s['ssim']):>6} {fv(s['lpips']):>6} {fv(s['dists']):>6} "
            f"{fv(s.get('ssimulacra2'),'.1f'):>6} {fv(s['fsim']):>6} "
            f"{fv(s['vif']):>6} {fv(s['haarpsi']):>6} "
            f"{fv(s['gmsd']):>6} {fv(s['dss']):>6} {fv(s['mdsi']):>6}"
        )

    print(f"\nRiferimenti JXL vs ELIC (dalla tesi, lambda comparabili):")
    print(f"  JXL d=3.0:   bpp=0.74 PSNR=33.3 SSIMULACRA2=72.5 LPIPS=0.119")
    print(f"  ELIC lam=0.15: bpp=0.50 PSNR=34.5 SSIMULACRA2=64.0 LPIPS=0.088")


if __name__ == "__main__":
    main()
