"""
Benchmark energetico RIGOROSO per la tesi.

Metodologia:
  1. Prima di ogni codec/param, misura idle power (CPU + GPU) per 2s
  2. Esegui il benchmark (compress + decompress su tutto Kodak)
  3. Energia netta = energia_misurata - idle_power × tempo

Codec CPU-only (JPEG, JXL, HEVC):
  → energia netta = RAPL_netta (Zeus ignorato — non usano GPU)

Codec neurali (Ballé, Cheng, ELIC, TCM, DCAE):
  → energia netta = RAPL_netta + Zeus_netta
  → cattura sia forward pass (GPU) sia codifica aritmetica (CPU)

Output: CSV con energy_cpu_net_j, energy_gpu_net_j, energy_total_net_j
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
torch.backends.cudnn.enabled = False  # Fix per DCAE

# ================== CONFIG ==================
KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
OUTPUT_CSV = os.path.expanduser(
    "~/tesi/results/images/full_pipeline_energy_benchmark.csv"
)
DEVICE = "cuda"
device = torch.device(DEVICE)
transform = transforms.ToTensor()
IDLE_DURATION = 2.0  # secondi per misura idle


# ================== RAPL ==================
def read_rapl_uj():
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read().strip())


def read_rapl_max():
    with open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj") as f:
        return int(f.read().strip())


RAPL_MAX = read_rapl_max()

# ================== Zeus ==================
from zeus.device.gpu import get_gpus

gpu = get_gpus().gpus[0]

FIELDNAMES = [
    "codec",
    "param",
    "image",
    "bpp",
    "psnr",
    "time_total_ms",
    "energy_cpu_gross_j",
    "energy_gpu_gross_j",
    "energy_cpu_net_j",
    "energy_gpu_net_j",
    "energy_total_net_j",
    "idle_cpu_w",
    "idle_gpu_w",
    "params_M",
    "pipeline",
    "is_neural",
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


# ================== IDLE MEASUREMENT ==================
def measure_idle():
    """Misura CPU e GPU idle power in Watts. Dura IDLE_DURATION secondi."""
    torch.cuda.synchronize()
    time.sleep(0.5)  # stabilizza

    r0 = read_rapl_uj()
    g0 = gpu.getTotalEnergyConsumption()
    t0 = time.perf_counter()

    time.sleep(IDLE_DURATION)

    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = gpu.getTotalEnergyConsumption()

    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX

    cpu_idle_w = (dr / 1e6) / dt
    gpu_idle_w = ((g1 - g0) / 1000) / dt

    return cpu_idle_w, gpu_idle_w


# ================== MEASUREMENT ==================
def measure_once():
    """Leggi RAPL+Zeus prima. Ritorna le letture iniziali."""
    torch.cuda.synchronize()
    return read_rapl_uj(), gpu.getTotalEnergyConsumption(), time.perf_counter()


def measure_end(r0, g0, t0, cpu_idle_w, gpu_idle_w, is_neural):
    """Leggi RAPL+Zeus dopo, calcola energia netta."""
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = gpu.getTotalEnergyConsumption()

    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX

    cpu_gross = dr / 1e6
    gpu_gross = (g1 - g0) / 1000

    cpu_net = cpu_gross - cpu_idle_w * dt
    gpu_net = gpu_gross - gpu_idle_w * dt

    # Per codec CPU-only, GPU net è rumore — ignora
    if not is_neural:
        gpu_net = 0.0

    # Clamp a 0 (il rumore può dare valori negativi per operazioni brevissime)
    cpu_net = max(cpu_net, 0.0)
    gpu_net = max(gpu_net, 0.0)
    total_net = cpu_net + gpu_net

    return {
        "time_s": dt,
        "cpu_gross": cpu_gross,
        "gpu_gross": gpu_gross,
        "cpu_net": cpu_net,
        "gpu_net": gpu_net,
        "total_net": total_net,
    }


# ====================================================================
# JPEG
# ====================================================================
def benchmark_jpeg(imgs, qualities=[10, 30, 60, 85]):
    import imagecodecs

    print("\n=== JPEG ===")
    results = []

    for q in qualities:
        cpu_idle, gpu_idle = measure_idle()
        print(f"  q={q} (idle: CPU={cpu_idle:.1f}W GPU={gpu_idle:.1f}W)")

        for img_path in imgs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_rgb.shape[:2]

            # Warmup
            imagecodecs.jpeg_decode(imagecodecs.jpeg_encode(img_rgb, level=q))

            r0, g0, t0 = measure_once()
            jpg = imagecodecs.jpeg_encode(img_rgb, level=q)
            rec = imagecodecs.jpeg_decode(jpg)
            e = measure_end(r0, g0, t0, cpu_idle, gpu_idle, is_neural=False)

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

            results.append(
                {
                    "codec": "JPEG",
                    "param": f"q={q}",
                    "image": Path(img_path).name,
                    "bpp": round(bpp, 4),
                    "psnr": round(compute_psnr(x, x_hat), 2),
                    "time_total_ms": round(e["time_s"] * 1000, 1),
                    "energy_cpu_gross_j": round(e["cpu_gross"], 4),
                    "energy_gpu_gross_j": round(e["gpu_gross"], 4),
                    "energy_cpu_net_j": round(e["cpu_net"], 4),
                    "energy_gpu_net_j": round(e["gpu_net"], 4),
                    "energy_total_net_j": round(e["total_net"], 4),
                    "idle_cpu_w": round(cpu_idle, 1),
                    "idle_gpu_w": round(gpu_idle, 1),
                    "params_M": 0,
                    "pipeline": "end_to_end",
                    "is_neural": False,
                }
            )

        sub = [r for r in results if r["param"] == f"q={q}"]
        print(
            f"    bpp={np.mean([r['bpp'] for r in sub]):.3f} net={np.mean([r['energy_total_net_j'] for r in sub]):.4f}J"
        )

    return results


# ====================================================================
# JXL
# ====================================================================
def benchmark_jxl(imgs, distances=[1.0, 3.0, 7.0, 12.0]):
    import imagecodecs

    print("\n=== JXL ===")
    results = []

    for d in distances:
        cpu_idle, gpu_idle = measure_idle()
        print(f"  d={d} (idle: CPU={cpu_idle:.1f}W GPU={gpu_idle:.1f}W)")

        for img_path in imgs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_rgb.shape[:2]

            imagecodecs.jpegxl_decode(
                imagecodecs.jpegxl_encode(img_rgb, distance=d, effort=5)
            )

            r0, g0, t0 = measure_once()
            jxl = imagecodecs.jpegxl_encode(img_rgb, distance=d, effort=5)
            rec = imagecodecs.jpegxl_decode(jxl)
            e = measure_end(r0, g0, t0, cpu_idle, gpu_idle, is_neural=False)

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

            results.append(
                {
                    "codec": "JXL",
                    "param": f"d={d}",
                    "image": Path(img_path).name,
                    "bpp": round(bpp, 4),
                    "psnr": round(compute_psnr(x, x_hat), 2),
                    "time_total_ms": round(e["time_s"] * 1000, 1),
                    "energy_cpu_gross_j": round(e["cpu_gross"], 4),
                    "energy_gpu_gross_j": round(e["gpu_gross"], 4),
                    "energy_cpu_net_j": round(e["cpu_net"], 4),
                    "energy_gpu_net_j": round(e["gpu_net"], 4),
                    "energy_total_net_j": round(e["total_net"], 4),
                    "idle_cpu_w": round(cpu_idle, 1),
                    "idle_gpu_w": round(gpu_idle, 1),
                    "params_M": 0,
                    "pipeline": "end_to_end",
                    "is_neural": False,
                }
            )

        sub = [r for r in results if r["param"] == f"d={d}"]
        print(
            f"    bpp={np.mean([r['bpp'] for r in sub]):.3f} net={np.mean([r['energy_total_net_j'] for r in sub]):.3f}J"
        )

    return results


# ====================================================================
# HEVC
# ====================================================================
def benchmark_hevc(imgs, crfs=[25, 35, 45]):
    print("\n=== HEVC Intra ===")
    results = []

    for crf in crfs:
        cpu_idle, gpu_idle = measure_idle()
        print(f"  crf={crf} (idle: CPU={cpu_idle:.1f}W GPU={gpu_idle:.1f}W)")

        for img_path in imgs:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_rgb.shape[:2]
            tmp_yuv = "/dev/shm/hevc_in.yuv"
            tmp_265 = "/dev/shm/hevc_out.265"

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

            r0, g0, t0 = measure_once()
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
            dec_result = subprocess.run(
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
            e = measure_end(r0, g0, t0, cpu_idle, gpu_idle, is_neural=False)

            if dec_result.returncode != 0 or len(dec_result.stdout) < h * w * 3:
                for f in [tmp_yuv, tmp_265]:
                    if os.path.exists(f):
                        os.remove(f)
                continue

            rec = np.frombuffer(dec_result.stdout, dtype=np.uint8).reshape(h, w, 3)
            bpp = os.path.getsize(tmp_265) * 8 / (h * w)
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

            results.append(
                {
                    "codec": "HEVC",
                    "param": f"crf={crf}",
                    "image": Path(img_path).name,
                    "bpp": round(bpp, 4),
                    "psnr": round(compute_psnr(x, x_hat), 2),
                    "time_total_ms": round(e["time_s"] * 1000, 1),
                    "energy_cpu_gross_j": round(e["cpu_gross"], 4),
                    "energy_gpu_gross_j": round(e["gpu_gross"], 4),
                    "energy_cpu_net_j": round(e["cpu_net"], 4),
                    "energy_gpu_net_j": round(e["gpu_net"], 4),
                    "energy_total_net_j": round(e["total_net"], 4),
                    "idle_cpu_w": round(cpu_idle, 1),
                    "idle_gpu_w": round(gpu_idle, 1),
                    "params_M": 0,
                    "pipeline": "end_to_end",
                    "is_neural": False,
                }
            )

            for f in [tmp_yuv, tmp_265]:
                if os.path.exists(f):
                    os.remove(f)

        sub = [r for r in results if r["param"] == f"crf={crf}"]
        if sub:
            print(
                f"    bpp={np.mean([r['bpp'] for r in sub]):.3f} net={np.mean([r['energy_total_net_j'] for r in sub]):.3f}J"
            )

    return results


# ====================================================================
# Neural codec generico
# ====================================================================
def benchmark_neural(net, codec_name, param_str, imgs, pad=64, pad_mode="right"):
    """Full pipeline compress()+decompress() con idle calibration."""
    net.eval()
    net.update()
    pm = sum(p.numel() for p in net.parameters()) / 1e6

    def do_pad(x):
        h, w = x.shape[2:]
        if pad_mode == "center":
            new_h = (h + pad - 1) // pad * pad
            new_w = (w + pad - 1) // pad * pad
            pl = (new_w - w) // 2
            pr = new_w - w - pl
            pt = (new_h - h) // 2
            pb = new_h - h - pt
            return F.pad(x, (pl, pr, pt, pb)), (pl, pr, pt, pb)
        else:
            pad_h = (pad - h % pad) % pad
            pad_w = (pad - w % pad) % pad
            return F.pad(x, (0, pad_w, 0, pad_h)), (0, pad_w, 0, pad_h)

    def do_crop(x_hat, padding, h, w):
        if pad_mode == "center":
            return F.pad(x_hat, (-padding[0], -padding[1], -padding[2], -padding[3]))
        else:
            return x_hat[:, :, :h, :w]

    # Warmup
    for wp in imgs[:3]:
        x = transform(Image.open(wp).convert("RGB")).unsqueeze(0).to(device)
        xp, padding = do_pad(x)
        with torch.no_grad():
            enc = net.compress(xp)
            net.decompress(enc["strings"], enc["shape"])
    torch.cuda.synchronize()

    # Idle measurement DOPO warmup (GPU è warm)
    cpu_idle, gpu_idle = measure_idle()
    print(f"  {codec_name} {param_str} (idle: CPU={cpu_idle:.1f}W GPU={gpu_idle:.1f}W)")

    results = []
    for img_path in imgs:
        x = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        h, w = x.shape[2:]
        xp, padding = do_pad(x)

        r0, g0, t0 = measure_once()
        with torch.no_grad():
            enc = net.compress(xp)
            dec = net.decompress(enc["strings"], enc["shape"])
        e = measure_end(r0, g0, t0, cpu_idle, gpu_idle, is_neural=True)

        x_hat = do_crop(dec["x_hat"], padding, h, w).clamp(0, 1)
        total_bytes = count_bytes(enc["strings"]) + 8
        bpp = total_bytes * 8 / (h * w)

        results.append(
            {
                "codec": codec_name,
                "param": param_str,
                "image": Path(img_path).name,
                "bpp": round(bpp, 4),
                "psnr": round(compute_psnr(x, x_hat), 2),
                "time_total_ms": round(e["time_s"] * 1000, 1),
                "energy_cpu_gross_j": round(e["cpu_gross"], 4),
                "energy_gpu_gross_j": round(e["gpu_gross"], 4),
                "energy_cpu_net_j": round(e["cpu_net"], 4),
                "energy_gpu_net_j": round(e["gpu_net"], 4),
                "energy_total_net_j": round(e["total_net"], 4),
                "idle_cpu_w": round(cpu_idle, 1),
                "idle_gpu_w": round(gpu_idle, 1),
                "params_M": round(pm, 1),
                "pipeline": "full_pipeline",
                "is_neural": True,
            }
        )

    avg_cpu = np.mean([r["energy_cpu_net_j"] for r in results])
    avg_gpu = np.mean([r["energy_gpu_net_j"] for r in results])
    avg_tot = np.mean([r["energy_total_net_j"] for r in results])
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_bpp = np.mean([r["bpp"] for r in results])
    print(
        f"    bpp={avg_bpp:.3f} PSNR={avg_psnr:.2f} CPU_net={avg_cpu:.3f}J GPU_net={avg_gpu:.3f}J TOT_net={avg_tot:.3f}J"
    )

    return results


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 75)
    print("BENCHMARK ENERGETICO RIGOROSO")
    print("RAPL+Zeus simultanei, idle sottratto per ogni codec/param")
    print("=" * 75)

    imgs = sorted(Path(KODAK_DIR).glob("*.png"))
    print(f"Kodak: {len(imgs)} immagini")
    print(f"Idle calibration: {IDLE_DURATION}s per ogni codec/param\n")

    all_results = []

    # === CPU CODECS ===
    all_results.extend(benchmark_jpeg(imgs))
    all_results.extend(benchmark_jxl(imgs))
    all_results.extend(benchmark_hevc(imgs))

    # === Ballé ===
    print("\n=== Ballé 2018 ===")
    from compressai.zoo import bmshj2018_hyperprior

    for q in [1, 3, 5, 7]:
        net = bmshj2018_hyperprior(quality=q, metric="mse", pretrained=True).to(device)
        all_results.extend(benchmark_neural(net, "Ballé", f"q={q}", imgs, pad=64))
        del net
        torch.cuda.empty_cache()

    # === Cheng ===
    print("\n=== Cheng 2020 ===")
    from compressai.zoo import cheng2020_attn

    for q in [1, 3, 5]:
        net = cheng2020_attn(quality=q, metric="mse", pretrained=True).to(device)
        all_results.extend(benchmark_neural(net, "Cheng", f"q={q}", imgs, pad=64))
        del net
        torch.cuda.empty_cache()

    # === ELIC ===
    print("\n=== ELIC ===")
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
                print(f"  [!] ELIC {param}: non trovato")
                continue
            sd = torch.load(ckpt, map_location=device)
            net = TestModel()
            net.load_state_dict(sd)
            net = net.to(device)
            all_results.extend(benchmark_neural(net, "ELIC", param, imgs, pad=64))
            del net
            torch.cuda.empty_cache()
    except ImportError:
        print("  [!] ELIC non disponibile (/tmp/elic)")

    # === TCM ===
    print("\n=== TCM ===")
    for mod in list(sys.modules.keys()):
        if "models" in mod and "compressai" not in mod:
            del sys.modules[mod]
    sys.path.insert(0, "/tmp/LIC_TCM")
    try:
        from models import TCM  # type: ignore

        tcm_ckpts = {
            "lam=0.013": "/tmp/LIC_TCM/checkpoints/tcm_mse_0013.pth.tar",
            "lam=0.0035": "/tmp/LIC_TCM/checkpoints/tcm_mse_0035.pth.tar",
        }
        for param, ckpt in tcm_ckpts.items():
            if not os.path.exists(ckpt):
                print(f"  [!] TCM {param}: non trovato")
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
            all_results.extend(benchmark_neural(net, "TCM", param, imgs, pad=64))
            del net
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [!] TCM: {e}")

    # === DCAE ===
    print("\n=== DCAE ===")
    for mod in list(sys.modules.keys()):
        if "models" in mod and "compressai" not in mod:
            del sys.modules[mod]
    sys.path.insert(0, "/tmp/DCAE")
    try:
        from models import DCAE  # type: ignore

        dcae_ckpts = {
            "lam=0.013": "/tmp/DCAE/checkpoints/dcae_mse_0013.pth.tar",
            "lam=0.0035": "/tmp/DCAE/checkpoints/dcae_mse_0035.pth.tar",
        }
        for param, ckpt in dcae_ckpts.items():
            if not os.path.exists(ckpt):
                print(f"  [!] DCAE {param}: non trovato")
                continue
            net = DCAE()
            net = net.to(device).eval()
            sd_raw = torch.load(ckpt, map_location=device)
            dictory = {
                k.replace("module.", ""): v for k, v in sd_raw["state_dict"].items()
            }
            net.load_state_dict(dictory)
            all_results.extend(
                benchmark_neural(net, "DCAE", param, imgs, pad=128, pad_mode="center")
            )
            del net
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [!] DCAE: {e}")

    # === SALVA ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nCSV: {OUTPUT_CSV}")

    # === TABELLA ===
    print(f"\n{'=' * 95}")
    print("RIEPILOGO R-D-E (medie Kodak, idle sottratto)")
    print(f"{'=' * 95}")
    print(
        f"{'Codec':<20} {'bpp':>6} {'PSNR':>6} {'CPU_net':>8} {'GPU_net':>8} {'TOT_net':>8} {'t(ms)':>7} {'idle_cpu':>8} {'idle_gpu':>8}"
    )
    print("-" * 95)

    groups = set((r["codec"], r["param"]) for r in all_results)
    for codec, param in sorted(groups):
        sub = [r for r in all_results if r["codec"] == codec and r["param"] == param]
        bpp = np.mean([r["bpp"] for r in sub])
        psnr = np.mean([r["psnr"] for r in sub])
        cpu_n = np.mean([r["energy_cpu_net_j"] for r in sub])
        gpu_n = np.mean([r["energy_gpu_net_j"] for r in sub])
        tot_n = np.mean([r["energy_total_net_j"] for r in sub])
        t = np.mean([r["time_total_ms"] for r in sub])
        ic = sub[0]["idle_cpu_w"]
        ig = sub[0]["idle_gpu_w"]
        print(
            f"{codec + ' ' + param:<20} {bpp:>6.3f} {psnr:>6.2f} {cpu_n:>8.3f} {gpu_n:>8.3f} {tot_n:>8.3f} {t:>7.0f} {ic:>7.1f}W {ig:>7.1f}W"
        )


if __name__ == "__main__":
    main()
