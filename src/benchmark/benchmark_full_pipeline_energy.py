"""
Benchmark energetico COMPLETO: tutti i codec, full pipeline.
CPU codecs (JPEG, JXL, HEVC): RAPL
GPU codecs (Ballé, Cheng, ELIC, TCM, DCAE): Zeus (compress + decompress)

Nessuna lacuna. Ogni codec misurato end-to-end.
Output: singolo CSV con bpp, PSNR, energia, tempo per ogni immagine Kodak.
"""

import sys
import os
import math
import time
import csv
import warnings
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
OUTPUT_CSV = os.path.expanduser(
    "~/tesi/results/images/full_pipeline_energy_benchmark.csv"
)
DEVICE = "cuda"
device = torch.device(DEVICE)
transform = transforms.ToTensor()

IDLE_W = 59.8  # GPU idle power watts


# RAPL
def read_rapl_uj():
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read().strip())


def read_rapl_max():
    with open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj") as f:
        return int(f.read().strip())


RAPL_MAX = read_rapl_max()

# Zeus
from zeus.device.gpu import get_gpus

gpu = get_gpus().gpus[0]

FIELDNAMES = [
    "codec",
    "param",
    "image",
    "bpp",
    "psnr",
    "time_total_ms",
    "time_compress_ms",
    "time_decompress_ms",
    "energy_j",
    "energy_source",
    "params_M",
    "pipeline",
]


def compute_psnr(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2).item()
    return -10 * math.log10(mse) if mse > 1e-10 else 100.0


def count_bytes(obj):
    if isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, list):
        return sum(count_bytes(item) for item in obj)
    return 0


# ====================================================================
# JPEG (CPU, RAPL)
# ====================================================================
def benchmark_jpeg(imgs, qualities=[10, 30, 60, 85]):
    import imagecodecs

    print("\n=== JPEG (CPU, RAPL) ===")
    results = []

    for q in qualities:
        for img_path in imgs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_rgb.shape[:2]

            # Warmup
            imagecodecs.jpeg_encode(img_rgb, level=q)

            before = read_rapl_uj()
            t0 = time.perf_counter()
            jpg = imagecodecs.jpeg_encode(img_rgb, level=q)
            rec = imagecodecs.jpeg_decode(jpg)
            t1 = time.perf_counter()
            after = read_rapl_uj()

            delta = after - before
            if delta < 0:
                delta += RAPL_MAX
            energy_j = delta / 1e6

            bpp = len(jpg) * 8 / (h * w)
            x = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
                / 255.0
            )
            x_hat = (
                torch.from_numpy(rec).permute(2, 0, 1).unsqueeze(0).float().to(device)
                / 255.0
            )
            psnr = compute_psnr(x, x_hat)

            results.append(
                {
                    "codec": "JPEG",
                    "param": f"q={q}",
                    "image": Path(img_path).name,
                    "bpp": round(bpp, 4),
                    "psnr": round(psnr, 2),
                    "time_total_ms": round((t1 - t0) * 1000, 1),
                    "time_compress_ms": None,
                    "time_decompress_ms": None,
                    "energy_j": round(energy_j, 4),
                    "energy_source": "RAPL",
                    "params_M": 0,
                    "pipeline": "end_to_end",
                }
            )

        avg_e = np.mean([r["energy_j"] for r in results if r["param"] == f"q={q}"])
        avg_bpp = np.mean([r["bpp"] for r in results if r["param"] == f"q={q}"])
        print(f"  q={q}: bpp={avg_bpp:.3f}, E={avg_e:.4f} J")

    return results


# ====================================================================
# JXL (CPU, RAPL)
# ====================================================================
def benchmark_jxl(imgs, distances=[1.0, 3.0, 7.0, 12.0]):
    import imagecodecs

    print("\n=== JXL (CPU, RAPL) ===")
    results = []

    for d in distances:
        for img_path in imgs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_rgb.shape[:2]

            imagecodecs.jpegxl_encode(img_rgb, distance=d, effort=5)

            before = read_rapl_uj()
            t0 = time.perf_counter()
            jxl = imagecodecs.jpegxl_encode(img_rgb, distance=d, effort=5)
            rec = imagecodecs.jpegxl_decode(jxl)
            t1 = time.perf_counter()
            after = read_rapl_uj()

            delta = after - before
            if delta < 0:
                delta += RAPL_MAX
            energy_j = delta / 1e6

            bpp = len(jxl) * 8 / (h * w)
            x = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
                / 255.0
            )
            x_hat = (
                torch.from_numpy(rec).permute(2, 0, 1).unsqueeze(0).float().to(device)
                / 255.0
            )
            psnr = compute_psnr(x, x_hat)

            results.append(
                {
                    "codec": "JXL",
                    "param": f"d={d}",
                    "image": Path(img_path).name,
                    "bpp": round(bpp, 4),
                    "psnr": round(psnr, 2),
                    "time_total_ms": round((t1 - t0) * 1000, 1),
                    "time_compress_ms": None,
                    "time_decompress_ms": None,
                    "energy_j": round(energy_j, 4),
                    "energy_source": "RAPL",
                    "params_M": 0,
                    "pipeline": "end_to_end",
                }
            )

        avg_e = np.mean([r["energy_j"] for r in results if r["param"] == f"d={d}"])
        avg_bpp = np.mean([r["bpp"] for r in results if r["param"] == f"d={d}"])
        print(f"  d={d}: bpp={avg_bpp:.3f}, E={avg_e:.4f} J")

    return results


# ====================================================================
# HEVC Intra (CPU, RAPL)
# ====================================================================
def benchmark_hevc(imgs, crfs=[25, 35, 45]):
    print("\n=== HEVC Intra (CPU, RAPL) ===")
    results = []

    for crf in crfs:
        for img_path in imgs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_rgb.shape[:2]

            tmp_yuv = "/dev/shm/hevc_in.yuv"
            tmp_265 = "/dev/shm/hevc_out.265"

            # Prepara YUV
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

            before = read_rapl_uj()
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
                    tmp_265,
                ],
                capture_output=True,
            )

            dec = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_265,
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                capture_output=True,
            )

            t1 = time.perf_counter()
            after = read_rapl_uj()

            delta = after - before
            if delta < 0:
                delta += RAPL_MAX
            energy_j = delta / 1e6

            if dec.returncode != 0 or len(dec.stdout) < h * w * 3:
                continue

            rec = np.frombuffer(dec.stdout, dtype=np.uint8).reshape(h, w, 3)
            file_bytes = os.path.getsize(tmp_265)
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
                torch.from_numpy(rec).permute(2, 0, 1).unsqueeze(0).float().to(device)
                / 255.0
            )
            psnr = compute_psnr(x, x_hat)

            results.append(
                {
                    "codec": "HEVC",
                    "param": f"crf={crf}",
                    "image": Path(img_path).name,
                    "bpp": round(bpp, 4),
                    "psnr": round(psnr, 2),
                    "time_total_ms": round((t1 - t0) * 1000, 1),
                    "time_compress_ms": None,
                    "time_decompress_ms": None,
                    "energy_j": round(energy_j, 4),
                    "energy_source": "RAPL",
                    "params_M": 0,
                    "pipeline": "end_to_end",
                }
            )

            for f in [tmp_yuv, tmp_265]:
                if os.path.exists(f):
                    os.remove(f)

        sub = [r for r in results if r["param"] == f"crf={crf}"]
        if sub:
            print(
                f"  crf={crf}: bpp={np.mean([r['bpp'] for r in sub]):.3f}, E={np.mean([r['energy_j'] for r in sub]):.4f} J"
            )

    return results


# ====================================================================
# GPU Neural codec — compress() + decompress() con Zeus
# ====================================================================
def benchmark_neural_full(net, codec_name, param_str, imgs, pad=64):
    """Full pipeline: compress() + decompress() con Zeus energy."""
    net.eval()
    net.update()
    pm = sum(p.numel() for p in net.parameters()) / 1e6
    results = []

    # Warmup (3 immagini, full pipeline)
    for wp in imgs[:3]:
        img = Image.open(wp).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        h, w = x.shape[2:]
        pad_h = (pad - h % pad) % pad
        pad_w = (pad - w % pad) % pad
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        with torch.no_grad():
            out_enc = net.compress(x_pad)
            net.decompress(out_enc["strings"], out_enc["shape"])
    torch.cuda.synchronize()

    for img_path in imgs:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        h, w = x.shape[2:]
        n_px = h * w
        pad_h = (pad - h % pad) % pad
        pad_w = (pad - w % pad) % pad
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))

        # === COMPRESS ===
        torch.cuda.synchronize()
        before_c = gpu.getTotalEnergyConsumption()
        t0_c = time.perf_counter()

        with torch.no_grad():
            out_enc = net.compress(x_pad)

        torch.cuda.synchronize()
        t1_c = time.perf_counter()
        after_c = gpu.getTotalEnergyConsumption()

        # === DECOMPRESS ===
        torch.cuda.synchronize()
        before_d = gpu.getTotalEnergyConsumption()
        t0_d = time.perf_counter()

        with torch.no_grad():
            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])

        torch.cuda.synchronize()
        t1_d = time.perf_counter()
        after_d = gpu.getTotalEnergyConsumption()

        x_hat = out_dec["x_hat"][:, :, :h, :w].clamp(0, 1)

        total_bytes = count_bytes(out_enc["strings"]) + 8
        bpp = total_bytes * 8 / n_px
        psnr = compute_psnr(x, x_hat)

        t_c = (t1_c - t0_c) * 1000
        t_d = (t1_d - t0_d) * 1000
        e_c = (after_c - before_c) / 1000
        e_d = (after_d - before_d) / 1000
        e_total = e_c + e_d

        results.append(
            {
                "codec": codec_name,
                "param": param_str,
                "image": Path(img_path).name,
                "bpp": round(bpp, 4),
                "psnr": round(psnr, 2),
                "time_total_ms": round(t_c + t_d, 1),
                "time_compress_ms": round(t_c, 1),
                "time_decompress_ms": round(t_d, 1),
                "energy_j": round(e_total, 4),
                "energy_source": "Zeus",
                "params_M": round(pm, 1),
                "pipeline": "full_pipeline",
            }
        )

    avg_e = np.mean([r["energy_j"] for r in results])
    avg_bpp = np.mean([r["bpp"] for r in results])
    avg_t = np.mean([r["time_total_ms"] for r in results])
    print(
        f"  {codec_name} {param_str}: bpp={avg_bpp:.3f}, E={avg_e:.2f} J, t={avg_t:.0f} ms (full pipeline)"
    )

    return results


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 70)
    print("BENCHMARK ENERGETICO COMPLETO — FULL PIPELINE, TUTTI I CODEC")
    print("=" * 70)

    imgs = sorted(Path(KODAK_DIR).glob("*.png"))
    print(f"Kodak: {len(imgs)} immagini\n")

    all_results = []

    # === CPU CODECS ===
    all_results.extend(benchmark_jpeg(imgs))
    all_results.extend(benchmark_jxl(imgs))
    all_results.extend(benchmark_hevc(imgs))

    # === Ballé (CompressAI) ===
    print("\n=== Ballé 2018 (GPU, Zeus, full pipeline) ===")
    from compressai.zoo import bmshj2018_hyperprior

    for q in [1, 3, 5, 7]:
        net = bmshj2018_hyperprior(quality=q, metric="mse", pretrained=True).to(device)
        all_results.extend(benchmark_neural_full(net, "Ballé", f"q={q}", imgs, pad=64))
        del net
        torch.cuda.empty_cache()

    # === Cheng (CompressAI) ===
    print("\n=== Cheng 2020 (GPU, Zeus, full pipeline) ===")
    from compressai.zoo import cheng2020_attn

    for q in [1, 3, 5]:
        net = cheng2020_attn(quality=q, metric="mse", pretrained=True).to(device)
        all_results.extend(benchmark_neural_full(net, "Cheng", f"q={q}", imgs, pad=64))
        del net
        torch.cuda.empty_cache()

    # === ELIC ===
    print("\n=== ELIC (GPU, Zeus, full pipeline) ===")
    sys.path.insert(0, "/tmp/elic")
    try:
        from Network import TestModel  # type: ignore

        elic_ckpts = {
            "lam=0.008": "/tmp/elic/elic_0008.pth.tar",
            "lam=0.032": "/tmp/elic/elic_0032.pth.tar",
            "lam=0.150": "/tmp/elic/elic_0150.pth.tar",
            "lam=0.450": "/tmp/elic/elic_0450.pth.tar",
        }
        for param, ckpt in elic_ckpts.items():
            if not os.path.exists(ckpt):
                print(f"  [!] ELIC {param}: checkpoint non trovato")
                continue
            sd = torch.load(ckpt, map_location=device)
            net = TestModel()
            net.load_state_dict(sd)
            net = net.to(device)
            all_results.extend(benchmark_neural_full(net, "ELIC", param, imgs, pad=64))
            del net
            torch.cuda.empty_cache()
    except ImportError:
        print("  [!] ELIC non disponibile (manca /tmp/elic)")

    # === TCM ===
    print("\n=== TCM (GPU, Zeus, full pipeline) ===")
    for mod in list(sys.modules.keys()):
        if "models" in mod and "compressai" not in mod:
            del sys.modules[mod]
    sys.path.insert(0, "/tmp/LIC_TCM")
    from models import TCM  # type: ignore

    tcm_ckpts = {
        "lam=0.013": "/tmp/LIC_TCM/checkpoints/tcm_mse_0013.pth.tar",
        "lam=0.0035": "/tmp/LIC_TCM/checkpoints/tcm_mse_0035.pth.tar",
    }
    for param, ckpt in tcm_ckpts.items():
        if not os.path.exists(ckpt):
            print(f"  [!] TCM {param}: checkpoint non trovato")
            continue
        net = TCM(
            config=[2, 2, 2, 2, 2, 2],
            head_dim=[8, 16, 32, 32, 16, 8],
            drop_path_rate=0.0,
            N=64,
            M=320,
        )
        sd = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(sd["state_dict"])
        net = net.to(device)
        all_results.extend(benchmark_neural_full(net, "TCM", param, imgs, pad=64))
        del net
        torch.cuda.empty_cache()

    # === DCAE ===
    print("\n=== DCAE (GPU, Zeus, full pipeline) ===")
    for mod in list(sys.modules.keys()):
        if "models" in mod and "compressai" not in mod:
            del sys.modules[mod]
    sys.path.insert(0, "/tmp/DCAE")
    from models import DCAE  # type: ignore

    dcae_ckpts = {
        "lam=0.013": "/tmp/DCAE/checkpoints/dcae_mse_0013.pth.tar",
        "lam=0.0035": "/tmp/DCAE/checkpoints/dcae_mse_0035.pth.tar",
    }
    for param, ckpt in dcae_ckpts.items():
        if not os.path.exists(ckpt):
            print(f"  [!] DCAE {param}: checkpoint non trovato")
            continue
        net = DCAE()
        sd_raw = torch.load(ckpt, map_location="cpu")
        dictory = {k.replace("module.", ""): v for k, v in sd_raw["state_dict"].items()}
        net.load_state_dict(dictory)
        net = net.to(device)
        all_results.extend(benchmark_neural_full(net, "DCAE", param, imgs, pad=128))
        del net
        torch.cuda.empty_cache()

    # === SALVA ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nCSV: {OUTPUT_CSV}")

    # === TABELLA ===
    print(f"\n{'=' * 75}")
    print("RIEPILOGO R-D-E (medie Kodak, FULL PIPELINE)")
    print(f"{'=' * 75}")
    print(
        f"{'Codec':<20} {'bpp':>6} {'PSNR':>6} {'E (J)':>8} {'t (ms)':>8} {'Source':>6} {'Pipeline':<15}"
    )
    print("-" * 75)

    groups = set((r["codec"], r["param"]) for r in all_results)
    for codec, param in sorted(groups):
        sub = [r for r in all_results if r["codec"] == codec and r["param"] == param]
        bpp = np.mean([r["bpp"] for r in sub])
        psnr = np.mean([r["psnr"] for r in sub])
        e = np.mean([r["energy_j"] for r in sub])
        t = np.mean([r["time_total_ms"] for r in sub])
        src = sub[0]["energy_source"]
        pipe = sub[0]["pipeline"]
        print(
            f"{codec+' '+param:<20} {bpp:>6.3f} {psnr:>6.2f} {e:>8.3f} {t:>8.1f} {src:>6} {pipe:<15}"
        )


if __name__ == "__main__":
    main()
