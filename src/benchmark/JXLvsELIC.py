"""
Confronto completo JXL vs ELIC su Kodak (24 immagini).

ELIC misurato in DUE modalità:
  1. forward_pass_only: model(x) — solo rete neurale, no codifica entropica
  2. full_pipeline: model.compress() + model.decompress() — pipeline completa

12 metriche qualità + metriche sistema (tempo, energia, VRAM, bpp reale)

Fix applicati:
  - LPIPS range [-1,1] (libreria richzhang)
  - PIL per entrambi i codec (no cv2)
  - JXL energia = None (è CPU-only, Zeus misura solo GPU)
  - JXL timing encode + decode separati
  - VRAM separata per encode e decode
  - Shape overhead nei bytes (+8)
  - Warmup completo prima delle misurazioni
  - Seed per riproducibilità
"""

import torch
import random
import warnings
import time
import sys
import os
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/tesi"))
sys.path.insert(0, "/tmp/elic")

# ================== CONFIG & RIPRODUCIBILITÀ ==================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
ELIC_CHECKPOINTS = {
    "0.008": "/tmp/elic/elic_0008.pth.tar",
    "0.032": "/tmp/elic/elic_0032.pth.tar",
    "0.150": "/tmp/elic/elic_0150.pth.tar",
    "0.450": "/tmp/elic/elic_0450.pth.tar",
}
JXL_DISTANCES = [1.0, 3.0, 7.0, 12.0]
IDLE_W = 59.8

OUTPUT_CSV = os.path.expanduser("~/tesi/results/jxl_vs_elic_benchmark.csv")
OUTPUT_JSON = os.path.expanduser("~/tesi/results/jxl_vs_elic_benchmark.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

# ================== LOG SISTEMA ==================
print("=" * 80)
print("JXL vs ELIC BENCHMARK")
print("=" * 80)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")
print(f"Idle power (GPU): {IDLE_W} W")
print(f"Seed: {SEED}")
print("=" * 80)

# ================== METRICHE SETUP ==================
from zeus.device.gpu import get_gpus
import lpips as lpips_lib
import piq
from pytorch_msssim import ms_ssim as compute_ms_ssim
import ssimulacra2 as ssimulacra2_mod

gpu = get_gpus().gpus[0] if device.type == "cuda" else None
loss_fn_lpips = lpips_lib.LPIPS(net="alex").to(device)


def compute_all_metrics(x: torch.Tensor, x_hat: torch.Tensor, orig_path: str) -> dict:
    """
    x, x_hat: tensori [1,3,H,W] su device, range [0,1].
    Ritorna dict con 12 metriche.
    """
    m = {}

    # Pixel-level
    m["psnr"] = piq.psnr(x, x_hat, data_range=1.0).item()

    # Structural
    m["ssim"] = piq.ssim(x, x_hat, data_range=1.0).item()  # type: ignore
    m["ms_ssim"] = compute_ms_ssim(x, x_hat, data_range=1.0).item()

    # Perceptual — LPIPS richiede range [-1, 1]
    with torch.no_grad():
        x_lpips = x * 2.0 - 1.0
        x_hat_lpips = x_hat * 2.0 - 1.0
        m["lpips"] = loss_fn_lpips(x_lpips, x_hat_lpips).item()

    # Deep perceptual
    m["dists"] = piq.DISTS()(x, x_hat).item()

    # Gradient/Edge
    m["fsim"] = piq.fsim(x, x_hat, data_range=1.0).item()
    m["gmsd"] = piq.gmsd(x, x_hat, data_range=1.0).item()

    # Information-theoretic
    m["vif"] = piq.vif_p(x, x_hat, data_range=1.0).item()

    # Wavelet
    m["haarpsi"] = piq.haarpsi(x, x_hat, data_range=1.0).item()

    # Multi-scale
    m["dss"] = piq.dss(x, x_hat, data_range=1.0).item()
    m["mdsi"] = piq.mdsi(x, x_hat, data_range=1.0).item()

    # Google SSIMULACRA2 (CPU, vuole file paths)
    try:
        rec_np = (
            (x_hat[0].detach().cpu().permute(1, 2, 0).numpy() * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        tmp_path = "/tmp/ssimulacra2_temp_rec.png"
        Image.fromarray(rec_np).save(tmp_path)
        m["ssimulacra2"] = ssimulacra2_mod.compute_ssimulacra2(orig_path, tmp_path)
    except Exception as e:
        m["ssimulacra2"] = None
        print(f"    [!] SSIMULACRA2: {e}")

    return m


# ================== JXL (CPU-only, end-to-end) ==================
def benchmark_jxl(img_path: str, distance: float) -> dict:
    import imagecodecs

    # Carica con PIL (stessa ground truth di ELIC)
    img_pil = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img_pil)
    h, w = img_rgb.shape[:2]

    # Encode
    t0_c = time.perf_counter()
    jxl_bytes = imagecodecs.jpegxl_encode(img_rgb, distance=distance, effort=5)
    t1_c = time.perf_counter()

    # Decode
    t0_d = time.perf_counter()
    img_rec = imagecodecs.jpegxl_decode(jxl_bytes)
    t1_d = time.perf_counter()

    bpp = (len(jxl_bytes) * 8) / (h * w)

    x = (
        torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        / 255.0
    )
    x_hat = (
        torch.from_numpy(img_rec).permute(2, 0, 1).unsqueeze(0).float().to(device)
        / 255.0
    )

    metrics = compute_all_metrics(x, x_hat, str(img_path))
    metrics.update(
        {
            "codec": "JXL",
            "param": f"d={distance}",
            "image": Path(img_path).name,
            "bpp": bpp,
            "file_bytes": len(jxl_bytes),
            "time_compress_ms": (t1_c - t0_c) * 1000,
            "time_decompress_ms": (t1_d - t0_d) * 1000,
            "time_ms": (t1_c - t0_c + t1_d - t0_d) * 1000,
            # JXL è CPU-only: Zeus misura solo GPU, quindi energia = None
            "energy_total_j": None,
            "energy_net_j": None,
            "energy_compress_j": None,
            "energy_decompress_j": None,
            "vram_enc_mb": 0.0,
            "vram_dec_mb": 0.0,
            "pipeline": "end_to_end",
            "note": "CPU-only, no GPU energy",
        }
    )
    return metrics


# ================== ELIC ==================
def load_elic(ckpt_path: str):
    from Network import TestModel  # type: ignore

    state_dict = torch.load(ckpt_path, map_location=device)
    model = TestModel()
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    model.update()
    return model


def pad_image(x: torch.Tensor, multiple: int = 64):
    h, w = x.shape[2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, h, w


def benchmark_elic_forward(model, img_path: str, lam: str) -> dict:
    """Forward pass only: model(x). Nessuna codifica aritmetica."""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    x_pad, h, w = pad_image(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = gpu.getTotalEnergyConsumption() if gpu is not None else 0
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(x_pad)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    after = gpu.getTotalEnergyConsumption() if gpu is not None else 0
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    x_hat = out["x_hat"][:, :, :h, :w].clamp(0, 1)
    n_pixels = h * w
    bpp = sum(
        (-torch.log2(l).sum() / n_pixels).item() for l in out["likelihoods"].values()
    )
    e_total = (after - before) / 1000

    metrics = compute_all_metrics(x, x_hat, str(img_path))
    metrics.update(
        {
            "codec": "ELIC",
            "param": f"lam={lam}",
            "image": Path(img_path).name,
            "bpp": bpp,
            "file_bytes": int(bpp * n_pixels / 8),
            "time_ms": (t1 - t0) * 1000,
            "time_compress_ms": None,
            "time_decompress_ms": None,
            "energy_total_j": e_total,
            "energy_net_j": e_total - IDLE_W * (t1 - t0),
            "energy_compress_j": None,
            "energy_decompress_j": None,
            "vram_enc_mb": vram_mb,
            "vram_dec_mb": None,
            "pipeline": "forward_pass_only",
            "note": "No arithmetic coding, bpp estimated from likelihoods",
        }
    )
    return metrics


def count_bytes_recursive(obj):
    """Conta bytes ricorsivamente nella struttura nidificata di ELIC strings."""
    if isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, list):
        return sum(count_bytes_recursive(item) for item in obj)
    return 0


def benchmark_elic_full(model, img_path: str, lam: str) -> dict:
    """Pipeline completa: compress() + decompress(). Include codifica aritmetica."""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    x_pad, h, w = pad_image(x)
    n_pixels = h * w

    # === COMPRESS ===
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before_c = gpu.getTotalEnergyConsumption() if gpu is not None else 0
    t0_c = time.perf_counter()

    out_enc = model.compress(x_pad)

    torch.cuda.synchronize()
    t1_c = time.perf_counter()
    after_c = gpu.getTotalEnergyConsumption() if gpu is not None else 0
    vram_enc_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Bitstream reale + shape overhead
    total_bytes = count_bytes_recursive(out_enc["strings"]) + 8

    # === DECOMPRESS ===
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before_d = gpu.getTotalEnergyConsumption() if gpu is not None else 0
    t0_d = time.perf_counter()

    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    torch.cuda.synchronize()
    t1_d = time.perf_counter()
    after_d = gpu.getTotalEnergyConsumption() if gpu is not None else 0
    vram_dec_mb = torch.cuda.max_memory_allocated() / 1024**2

    x_hat = out_dec["x_hat"][:, :, :h, :w].clamp(0, 1)

    # Tempi
    t_compress = (t1_c - t0_c) * 1000
    t_decompress = (t1_d - t0_d) * 1000
    t_total = t_compress + t_decompress

    # Energia
    e_compress = (after_c - before_c) / 1000
    e_decompress = (after_d - before_d) / 1000
    e_total = e_compress + e_decompress
    e_net = e_total - IDLE_W * ((t1_c - t0_c) + (t1_d - t0_d))

    metrics = compute_all_metrics(x, x_hat, str(img_path))
    metrics.update(
        {
            "codec": "ELIC",
            "param": f"lam={lam}",
            "image": Path(img_path).name,
            "bpp": (total_bytes * 8) / n_pixels,
            "file_bytes": total_bytes,
            "time_ms": t_total,
            "time_compress_ms": t_compress,
            "time_decompress_ms": t_decompress,
            "energy_total_j": e_total,
            "energy_net_j": e_net,
            "energy_compress_j": e_compress,
            "energy_decompress_j": e_decompress,
            "vram_enc_mb": vram_enc_mb,
            "vram_dec_mb": vram_dec_mb,
            "pipeline": "full_pipeline",
            "note": "Includes arithmetic encode/decode",
        }
    )
    return metrics


# ================== MAIN ==================
METRIC_COLS = [
    "psnr",
    "ssim",
    "ms_ssim",
    "lpips",
    "dists",
    "fsim",
    "vif",
    "gmsd",
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
    "file_bytes",
    "time_ms",
    "time_compress_ms",
    "time_decompress_ms",
    "energy_total_j",
    "energy_net_j",
    "energy_compress_j",
    "energy_decompress_j",
    "vram_enc_mb",
    "vram_dec_mb",
    "pipeline",
    "note",
]
ALL_COLS = SYSTEM_COLS + METRIC_COLS


def print_avg(results, codec, param, pipeline=None):
    subset = [
        r
        for r in results
        if r["codec"] == codec
        and r["param"] == param
        and (pipeline is None or r["pipeline"] == pipeline)
        and r.get("bpp") is not None
    ]
    if not subset:
        return
    bpp = np.mean([r["bpp"] for r in subset])
    psnr = np.mean([r["psnr"] for r in subset])
    lpips_v = np.mean([r["lpips"] for r in subset])
    t_vals = [r["time_ms"] for r in subset if r["time_ms"] is not None]
    e_vals = [r["energy_net_j"] for r in subset if r["energy_net_j"] is not None]
    t_str = f"{np.mean(t_vals):.0f}ms" if t_vals else "N/A"
    e_str = f"{np.mean(e_vals):.2f}J" if e_vals else "N/A"

    extra = ""
    tc = [
        r["time_compress_ms"] for r in subset if r.get("time_compress_ms") is not None
    ]
    td = [
        r["time_decompress_ms"]
        for r in subset
        if r.get("time_decompress_ms") is not None
    ]
    if tc and td:
        extra = f" (enc={np.mean(tc):.0f}ms dec={np.mean(td):.0f}ms)"

    print(
        f"  MEDIA: bpp={bpp:.3f} PSNR={psnr:.2f} LPIPS={lpips_v:.4f} t={t_str}{extra} E_net={e_str}"
    )


def main():
    imgs = sorted(Path(KODAK_DIR).glob("*.png"))
    if not imgs:
        print("[!] Nessuna immagine trovata in", KODAK_DIR)
        return
    print(f"\nKodak: {len(imgs)} immagini\n")

    all_results = []

    # =============================================
    # JXL (end-to-end, CPU)
    # =============================================
    print("=" * 70)
    print("BENCHMARK JXL (end-to-end, CPU-only)")
    print("=" * 70)
    for dist in JXL_DISTANCES:
        print(f"\n--- JXL distance={dist} ---")
        for i, img in enumerate(imgs):
            m = benchmark_jxl(str(img), dist)
            all_results.append(m)
            if (i + 1) % 8 == 0:
                print(
                    f"  [{i+1}/{len(imgs)}] bpp={m['bpp']:.3f} PSNR={m['psnr']:.2f} LPIPS={m['lpips']:.4f}"
                )
        print_avg(all_results, "JXL", f"d={dist}")

    # =============================================
    # ELIC — FORWARD PASS ONLY
    # =============================================
    print("\n" + "=" * 70)
    print("BENCHMARK ELIC — FORWARD PASS ONLY (no arithmetic coding)")
    print("=" * 70)
    for lam, ckpt in ELIC_CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            print(f"\n--- ELIC lam={lam}: checkpoint non trovato ---")
            continue
        print(f"\n--- ELIC lam={lam} (forward only) ---")
        model = load_elic(ckpt)

        # Warmup: 2 immagini complete
        for warmup_img in imgs[:2]:
            x = transform(Image.open(warmup_img).convert("RGB")).unsqueeze(0).to(device)
            xp, _, _ = pad_image(x)
            with torch.no_grad():
                model(xp)
        torch.cuda.synchronize()

        for i, img in enumerate(imgs):
            m = benchmark_elic_forward(model, str(img), lam)
            all_results.append(m)
            if (i + 1) % 8 == 0:
                print(
                    f"  [{i+1}/{len(imgs)}] bpp={m['bpp']:.3f} PSNR={m['psnr']:.2f} LPIPS={m['lpips']:.4f}"
                )
        print_avg(all_results, "ELIC", f"lam={lam}", "forward_pass_only")

        del model
        torch.cuda.empty_cache()

    # =============================================
    # ELIC — FULL PIPELINE (compress + decompress)
    # =============================================
    print("\n" + "=" * 70)
    print("BENCHMARK ELIC — FULL PIPELINE (con codifica aritmetica)")
    print("=" * 70)
    for lam, ckpt in ELIC_CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            print(f"\n--- ELIC lam={lam}: checkpoint non trovato ---")
            continue
        print(f"\n--- ELIC lam={lam} (full pipeline) ---")
        model = load_elic(ckpt)

        # Warmup completo: compress + decompress su 2 immagini
        x = transform(Image.open(imgs[0]).convert("RGB")).unsqueeze(0).to(device)
        xp, _, _ = pad_image(x)
        try:
            for warmup_img in imgs[:2]:
                x_w = (
                    transform(Image.open(warmup_img).convert("RGB"))
                    .unsqueeze(0)
                    .to(device)
                )
                xp_w, _, _ = pad_image(x_w)
                out_w = model.compress(xp_w)
                model.decompress(out_w["strings"], out_w["shape"])
            torch.cuda.synchronize()
            print("  compress()/decompress() OK")
        except Exception as e:
            print(f"  [!] compress()/decompress() FAIL: {e}")
            print("  SKIP full pipeline per questo lambda")
            del model
            torch.cuda.empty_cache()
            continue

        for i, img in enumerate(imgs):
            m = benchmark_elic_full(model, str(img), lam)
            all_results.append(m)
            if (i + 1) % 8 == 0 and m.get("bpp") is not None:
                print(
                    f"  [{i+1}/{len(imgs)}] bpp={m['bpp']:.3f} PSNR={m['psnr']:.2f} "
                    f"t_enc={m['time_compress_ms']:.0f}ms t_dec={m['time_decompress_ms']:.0f}ms"
                )
        print_avg(all_results, "ELIC", f"lam={lam}", "full_pipeline")

        del model
        torch.cuda.empty_cache()

    # =============================================
    # SALVATAGGIO
    # =============================================
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # CSV per-image
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nCSV: {OUTPUT_CSV}")

    # JSON medie aggregate
    summary = {}
    groups = set((r["codec"], r["param"], r["pipeline"]) for r in all_results)
    for codec, param, pipeline in sorted(groups):
        subset = [
            r
            for r in all_results
            if r["codec"] == codec
            and r["param"] == param
            and r["pipeline"] == pipeline
            and r.get("bpp") is not None
        ]
        if not subset:
            continue
        key = f"{codec}_{param}_{pipeline}"
        summary[key] = {"pipeline": pipeline, "n_images": len(subset)}
        for col in METRIC_COLS + [
            "bpp",
            "file_bytes",
            "time_ms",
            "time_compress_ms",
            "time_decompress_ms",
            "energy_total_j",
            "energy_net_j",
            "energy_compress_j",
            "energy_decompress_j",
            "vram_enc_mb",
            "vram_dec_mb",
        ]:
            vals = [r[col] for r in subset if r.get(col) is not None]
            summary[key][col] = round(float(np.mean(vals)), 6) if vals else None

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON: {OUTPUT_JSON}")

    # =============================================
    # TABELLA RIEPILOGATIVA
    # =============================================
    print("\n" + "=" * 130)
    print("TABELLA RIEPILOGATIVA (medie su 24 immagini Kodak)")
    print("=" * 130)
    h = (
        f"{'Config':<42} {'bpp':>6} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} "
        f"{'DISTS':>6} {'SSIM2':>7} {'t_ms':>7} {'t_enc':>7} {'t_dec':>7} "
        f"{'E_net':>7} {'VRAM_e':>7} {'VRAM_d':>7}"
    )
    print(h)
    print("-" * len(h))
    for key in sorted(summary.keys()):
        s = summary[key]

        def fmt(v, f=".2f"):
            return f"{v:{f}}" if v is not None else "N/A"

        print(
            f"{key:<42} {fmt(s['bpp'],'.3f'):>6} {fmt(s['psnr']):>6} "
            f"{fmt(s['ssim'],'.4f'):>6} {fmt(s['lpips'],'.4f'):>6} "
            f"{fmt(s['dists'],'.4f'):>6} {fmt(s.get('ssimulacra2'),'.1f'):>7} "
            f"{fmt(s.get('time_ms'),'.0f'):>7} {fmt(s.get('time_compress_ms'),'.0f'):>7} "
            f"{fmt(s.get('time_decompress_ms'),'.0f'):>7} "
            f"{fmt(s.get('energy_net_j')):>7} "
            f"{fmt(s.get('vram_enc_mb'),'.0f'):>7} {fmt(s.get('vram_dec_mb'),'.0f'):>7}"
        )

    print("\n" + "=" * 130)
    print("LEGENDA:")
    print("  forward_pass_only: solo model(x), bpp stimato da log2(likelihoods)")
    print(
        "  full_pipeline: model.compress() + model.decompress(), bpp da bitstream reale"
    )
    print("  end_to_end: JXL encode + decode completo su CPU")
    print("  E_net: energia GPU netta (totale - idle), None per JXL (CPU-only)")
    print("  SSIM2: SSIMULACRA2 (metrica psicovisiva di Google, usata da JXL)")
    print("=" * 130)


if __name__ == "__main__":
    main()
