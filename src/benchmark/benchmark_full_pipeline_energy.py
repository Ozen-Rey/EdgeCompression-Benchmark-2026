"""
Benchmark energetico DEFINITIVO per la tesi.

Metodologia a batch per eliminare il rumore su codec veloci:
  1. Pre-carica TUTTE le immagini in RAM/VRAM (zero I/O durante la misura)
  2. Warmup (3 immagini × full pipeline)
  3. Misura idle per 3s (RAPL + Zeus)
  4. Loop stretto: tutte le immagini × N ripetizioni
  5. Misura RAPL + Zeus dopo il loop
  6. Energia netta = (gross - idle × tempo) / (n_immagini × n_ripetizioni)

N ripetizioni adattive:
  JPEG: 20 (1ms per immagine → 480ms totali → delta RAPL affidabile)
  JXL:  10 (~27ms → 6.5s totali)
  HEVC:  3 (~72ms → 5.2s totali)
  Ballé: 5 (~28ms → 3.4s totali)
  ELIC:  3 (~140ms → 10s totali)
  TCM:   3 (~108ms → 7.8s totali)
  DCAE:  3 (~109ms → 7.8s totali)
  Cheng: 1 (~3400ms → 82s totali — già abbastanza lungo)
 w
Codec CPU-only: GPU net = 0 (non usano GPU)
Codec neurali: tot = CPU net + GPU net

Qualità (bpp, PSNR) misurata in un passo separato non temporizzato.
"""

import sys
import os
import gc
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
torch.backends.cudnn.enabled = False  # Necessario per DCAE compress/decompress

# ================== CONFIG ==================
KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
OUTPUT_CSV = os.path.expanduser(
    "~/tesi/results/images/full_pipeline_energy_benchmark.csv"
)
DEVICE = "cuda"
device = torch.device(DEVICE)
transform = transforms.ToTensor()
IDLE_DURATION = 3.0  # secondi — più lungo = idle più stabile


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
    "n_images",
    "n_repeats",
    "avg_bpp",
    "avg_psnr",
    "total_time_s",
    "energy_cpu_net_j",
    "energy_gpu_net_j",
    "energy_total_net_j",
    "energy_per_image_j",
    "idle_cpu_w",
    "idle_gpu_w",
    "params_M",
    "is_neural",
]


def compute_psnr(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2).item()
    return -10 * math.log10(mse) if mse > 1e-10 else 100.0


def count_bytes(obj):
    if isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, list):
        return sum(count_bytes(i) for i in obj)
    return 0


# ================== IDLE ==================
def measure_idle():
    """Misura idle power per IDLE_DURATION secondi. Ritorna (cpu_W, gpu_W)."""
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

    return (dr / 1e6) / dt, ((g1 - g0) / 1000) / dt


# ================== BATCH ENERGY ==================
def measure_batch_energy(run_fn, n_repeats, cpu_idle_w, gpu_idle_w, is_neural):
    """
    Esegue run_fn() per n_repeats volte.
    run_fn() deve processare TUTTI gli item internamente.
    Ritorna energia netta totale (cpu, gpu, tot) e tempo.
    """
    gc.disable()
    torch.cuda.synchronize()

    r0 = read_rapl_uj()
    g0 = gpu.getTotalEnergyConsumption()
    t0 = time.perf_counter()

    for _ in range(n_repeats):
        run_fn()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    r1 = read_rapl_uj()
    g1 = gpu.getTotalEnergyConsumption()
    gc.enable()

    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX

    cpu_gross = dr / 1e6
    gpu_gross = (g1 - g0) / 1000
    cpu_net = max(cpu_gross - cpu_idle_w * dt, 0)
    gpu_net = max(gpu_gross - gpu_idle_w * dt, 0) if is_neural else 0.0
    total_net = cpu_net + gpu_net

    return {
        "cpu_net": cpu_net,
        "gpu_net": gpu_net,
        "total_net": total_net,
        "time_s": dt,
        "cpu_gross": cpu_gross,
        "gpu_gross": gpu_gross,
    }


# ================== PRE-LOAD ==================
def preload_kodak():
    """Pre-carica tutte le immagini Kodak come numpy + tensori GPU."""
    imgs = sorted(Path(KODAK_DIR).glob("*.png"))
    data = []
    for img_path in imgs:
        pil = Image.open(img_path).convert("RGB")
        rgb = np.array(pil)
        tensor = transform(pil).unsqueeze(0).to(device)
        data.append(
            {
                "path": img_path,
                "name": img_path.name,
                "rgb": rgb,
                "tensor": tensor,
                "h": rgb.shape[0],
                "w": rgb.shape[1],
            }
        )
    return data


# ================== JPEG ==================
def benchmark_jpeg(kodak, qualities=[10, 30, 60, 85], n_repeats=20):
    import imagecodecs

    print("\n=== JPEG ===")
    results = []

    for q in qualities:
        # Qualità (non temporizzata)
        bpps, psnrs = [], []
        for d in kodak:
            jpg = imagecodecs.jpeg_encode(d["rgb"], level=q)
            rec = imagecodecs.jpeg_decode(jpg)
            bpps.append(len(jpg) * 8 / (d["h"] * d["w"]))
            x_hat = (
                torch.from_numpy(rec).permute(2, 0, 1).unsqueeze(0).float().to(device)
                / 255.0
            )
            psnrs.append(compute_psnr(d["tensor"], x_hat))

        # Warmup
        for d in kodak[:3]:
            imagecodecs.jpeg_decode(imagecodecs.jpeg_encode(d["rgb"], level=q))

        # Idle
        cpu_idle, gpu_idle = measure_idle()

        # Batch energy
        def run_all():
            for d in kodak:
                rec = imagecodecs.jpeg_decode(
                    imagecodecs.jpeg_encode(d["rgb"], level=q)
                )

        e = measure_batch_energy(
            run_all, n_repeats, cpu_idle, gpu_idle, is_neural=False
        )
        n_total = len(kodak) * n_repeats
        e_per_img = e["total_net"] / n_total

        print(
            f"  q={q}: bpp={np.mean(bpps):.3f} PSNR={np.mean(psnrs):.2f} "
            f"E={e_per_img:.4f} J/img (batch {n_total} ops in {e['time_s']:.1f}s, "
            f"idle CPU={cpu_idle:.1f}W GPU={gpu_idle:.1f}W)"
        )

        results.append(
            {
                "codec": "JPEG",
                "param": f"q={q}",
                "n_images": len(kodak),
                "n_repeats": n_repeats,
                "avg_bpp": round(np.mean(bpps), 4),
                "avg_psnr": round(np.mean(psnrs), 2),
                "total_time_s": round(e["time_s"], 2),
                "energy_cpu_net_j": round(e["cpu_net"], 4),
                "energy_gpu_net_j": round(e["gpu_net"], 4),
                "energy_total_net_j": round(e["total_net"], 4),
                "energy_per_image_j": round(e_per_img, 6),
                "idle_cpu_w": round(cpu_idle, 1),
                "idle_gpu_w": round(gpu_idle, 1),
                "params_M": 0,
                "is_neural": False,
            }
        )

    return results


# ================== JXL ==================
def benchmark_jxl(kodak, distances=[1.0, 3.0, 7.0, 12.0], n_repeats=10):
    import imagecodecs

    print("\n=== JXL ===")
    results = []

    for dist in distances:
        bpps, psnrs = [], []
        for d in kodak:
            jxl = imagecodecs.jpegxl_encode(d["rgb"], distance=dist, effort=5)
            rec = imagecodecs.jpegxl_decode(jxl)
            bpps.append(len(jxl) * 8 / (d["h"] * d["w"]))
            x_hat = (
                torch.from_numpy(rec).permute(2, 0, 1).unsqueeze(0).float().to(device)
                / 255.0
            )
            psnrs.append(compute_psnr(d["tensor"], x_hat))

        for d in kodak[:3]:
            imagecodecs.jpegxl_decode(
                imagecodecs.jpegxl_encode(d["rgb"], distance=dist, effort=5)
            )

        cpu_idle, gpu_idle = measure_idle()

        def run_all():
            for d in kodak:
                imagecodecs.jpegxl_decode(
                    imagecodecs.jpegxl_encode(d["rgb"], distance=dist, effort=5)
                )

        e = measure_batch_energy(
            run_all, n_repeats, cpu_idle, gpu_idle, is_neural=False
        )
        n_total = len(kodak) * n_repeats
        e_per_img = e["total_net"] / n_total

        print(
            f"  d={dist}: bpp={np.mean(bpps):.3f} PSNR={np.mean(psnrs):.2f} "
            f"E={e_per_img:.4f} J/img ({e['time_s']:.1f}s, idle CPU={cpu_idle:.1f}W)"
        )

        results.append(
            {
                "codec": "JXL",
                "param": f"d={dist}",
                "n_images": len(kodak),
                "n_repeats": n_repeats,
                "avg_bpp": round(np.mean(bpps), 4),
                "avg_psnr": round(np.mean(psnrs), 2),
                "total_time_s": round(e["time_s"], 2),
                "energy_cpu_net_j": round(e["cpu_net"], 4),
                "energy_gpu_net_j": round(e["gpu_net"], 4),
                "energy_total_net_j": round(e["total_net"], 4),
                "energy_per_image_j": round(e_per_img, 6),
                "idle_cpu_w": round(cpu_idle, 1),
                "idle_gpu_w": round(gpu_idle, 1),
                "params_M": 0,
                "is_neural": False,
            }
        )

    return results


# ================== HEVC ==================
def benchmark_hevc(kodak, crfs=[25, 35, 45], n_repeats=3):
    print("\n=== HEVC Intra ===")
    results = []

    for crf in crfs:
        # Pre-converti tutte le immagini in YUV su /dev/shm
        yuv_paths = []
        for i, d in enumerate(kodak):
            yuv_path = f"/dev/shm/hevc_in_{i}.yuv"
            subprocess.run(
                [
                    "ffmpeg",
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
                    yuv_path,
                ],
                input=d["rgb"].tobytes(),
                capture_output=True,
            )
            yuv_paths.append(yuv_path)

        # Qualità (non temporizzata)
        bpps, psnrs = [], []
        for i, d in enumerate(kodak):
            hevc_path = f"/dev/shm/hevc_out_{i}.265"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv444p",
                    "-s",
                    f"{d['w']}x{d['h']}",
                    "-i",
                    yuv_paths[i],
                    "-c:v",
                    "libx265",
                    "-preset",
                    "medium",
                    "-x265-params",
                    f"crf={crf}:keyint=1",
                    "-pix_fmt",
                    "yuv444p",
                    hevc_path,
                ],
                capture_output=True,
            )
            dec = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    hevc_path,
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                capture_output=True,
            )
            if dec.returncode == 0 and len(dec.stdout) >= d["h"] * d["w"] * 3:
                rec = np.frombuffer(dec.stdout, dtype=np.uint8).reshape(
                    d["h"], d["w"], 3
                )
                bpps.append(os.path.getsize(hevc_path) * 8 / (d["h"] * d["w"]))
                x_hat = (
                    torch.from_numpy(rec)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                    / 255.0
                )
                psnrs.append(compute_psnr(d["tensor"], x_hat))

        cpu_idle, gpu_idle = measure_idle()

        def run_all():
            for i, d in enumerate(kodak):
                hevc_path = f"/dev/shm/hevc_out_{i}.265"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "yuv444p",
                        "-s",
                        f"{d['w']}x{d['h']}",
                        "-i",
                        yuv_paths[i],
                        "-c:v",
                        "libx265",
                        "-preset",
                        "medium",
                        "-x265-params",
                        f"crf={crf}:keyint=1",
                        "-pix_fmt",
                        "yuv444p",
                        hevc_path,
                    ],
                    capture_output=True,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        hevc_path,
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "pipe:1",
                    ],
                    capture_output=True,
                )

        e = measure_batch_energy(
            run_all, n_repeats, cpu_idle, gpu_idle, is_neural=False
        )
        n_total = len(kodak) * n_repeats
        e_per_img = e["total_net"] / n_total

        print(
            f"  crf={crf}: bpp={np.mean(bpps):.3f} PSNR={np.mean(psnrs):.2f} "
            f"E={e_per_img:.4f} J/img ({e['time_s']:.1f}s)"
        )

        results.append(
            {
                "codec": "HEVC",
                "param": f"crf={crf}",
                "n_images": len(kodak),
                "n_repeats": n_repeats,
                "avg_bpp": round(np.mean(bpps), 4),
                "avg_psnr": round(np.mean(psnrs), 2),
                "total_time_s": round(e["time_s"], 2),
                "energy_cpu_net_j": round(e["cpu_net"], 4),
                "energy_gpu_net_j": round(e["gpu_net"], 4),
                "energy_total_net_j": round(e["total_net"], 4),
                "energy_per_image_j": round(e_per_img, 6),
                "idle_cpu_w": round(cpu_idle, 1),
                "idle_gpu_w": round(gpu_idle, 1),
                "params_M": 0,
                "is_neural": False,
            }
        )

        # Cleanup
        for p in yuv_paths:
            if os.path.exists(p):
                os.remove(p)

    return results


# ================== NEURAL GENERICO ==================
def benchmark_neural(
    net, codec_name, param_str, kodak, pad=64, pad_mode="right", n_repeats=3
):
    """Batch energy measurement per codec neurali."""
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

    # Pre-pad tutti i tensori
    padded_data = []
    for d in kodak:
        xp, padding = do_pad(d["tensor"])
        padded_data.append({"xp": xp, "padding": padding, "h": d["h"], "w": d["w"]})

    # Qualità (non temporizzata)
    bpps, psnrs = [], []
    with torch.no_grad():
        for i, d in enumerate(kodak):
            pd_ = padded_data[i]
            enc = net.compress(pd_["xp"])
            dec = net.decompress(enc["strings"], enc["shape"])
            x_hat = do_crop(dec["x_hat"], pd_["padding"], pd_["h"], pd_["w"]).clamp(
                0, 1
            )
            total_bytes = count_bytes(enc["strings"]) + 8
            bpps.append(total_bytes * 8 / (pd_["h"] * pd_["w"]))
            psnrs.append(compute_psnr(d["tensor"], x_hat))
    torch.cuda.synchronize()

    # Warmup (3 immagini × 2 ripetizioni)
    with torch.no_grad():
        for _ in range(2):
            for pd_ in padded_data[:3]:
                enc = net.compress(pd_["xp"])
                net.decompress(enc["strings"], enc["shape"])
    torch.cuda.synchronize()

    # Idle (dopo warmup — GPU è calda)
    cpu_idle, gpu_idle = measure_idle()

    # Batch energy
    def run_all():
        with torch.no_grad():
            for pd_ in padded_data:
                enc = net.compress(pd_["xp"])
                net.decompress(enc["strings"], enc["shape"])

    e = measure_batch_energy(run_all, n_repeats, cpu_idle, gpu_idle, is_neural=True)
    n_total = len(kodak) * n_repeats
    e_per_img = e["total_net"] / n_total

    print(
        f"  {codec_name} {param_str}: bpp={np.mean(bpps):.3f} PSNR={np.mean(psnrs):.2f} "
        f"E={e_per_img:.3f} J/img "
        f"(CPU={e['cpu_net']/n_total:.3f} GPU={e['gpu_net']/n_total:.3f}) "
        f"({e['time_s']:.1f}s, idle CPU={cpu_idle:.1f}W GPU={gpu_idle:.1f}W)"
    )

    return [
        {
            "codec": codec_name,
            "param": param_str,
            "n_images": len(kodak),
            "n_repeats": n_repeats,
            "avg_bpp": round(np.mean(bpps), 4),
            "avg_psnr": round(np.mean(psnrs), 2),
            "total_time_s": round(e["time_s"], 2),
            "energy_cpu_net_j": round(e["cpu_net"] / n_total, 4),
            "energy_gpu_net_j": round(e["gpu_net"] / n_total, 4),
            "energy_total_net_j": round(e["total_net"] / n_total, 4),
            "energy_per_image_j": round(e_per_img, 6),
            "idle_cpu_w": round(cpu_idle, 1),
            "idle_gpu_w": round(gpu_idle, 1),
            "params_M": round(pm, 1),
            "is_neural": True,
        }
    ]


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 75)
    print("BENCHMARK ENERGETICO DEFINITIVO")
    print("Batch measurement, RAPL+Zeus simultanei, idle 3s, GC disabilitato")
    print("=" * 75)

    # Pre-load
    kodak = preload_kodak()
    print(f"Kodak: {len(kodak)} immagini pre-caricate\n")

    all_results = []

    # === CPU CODECS ===
    all_results.extend(benchmark_jpeg(kodak, n_repeats=20))
    all_results.extend(benchmark_jxl(kodak, n_repeats=10))
    all_results.extend(benchmark_hevc(kodak, n_repeats=3))

    # === Ballé ===
    print("\n=== Ballé 2018 ===")
    from compressai.zoo import bmshj2018_hyperprior

    for q in [1, 3, 5, 7]:
        net = bmshj2018_hyperprior(quality=q, metric="mse", pretrained=True).to(device)
        all_results.extend(
            benchmark_neural(net, "Ballé", f"q={q}", kodak, pad=64, n_repeats=5)
        )
        del net
        torch.cuda.empty_cache()

    # === Cheng ===
    print("\n=== Cheng 2020 ===")
    from compressai.zoo import cheng2020_attn

    for q in [1, 3, 5]:
        net = cheng2020_attn(quality=q, metric="mse", pretrained=True).to(device)
        all_results.extend(
            benchmark_neural(net, "Cheng", f"q={q}", kodak, pad=64, n_repeats=1)
        )
        del net
        torch.cuda.empty_cache()

    # === ELIC ===
    # NOTA: richiede fix import in /tmp/elic/Network.py per CompressAI >= 1.2.0:
    #   GaussianConditional → from compressai.entropy_models
    #   ste_round → quantize_ste
    print("\n=== ELIC ===")
    sys.path.insert(0, os.path.expanduser("~/tesi/external_codecs/elic"))
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
            all_results.extend(
                benchmark_neural(net, "ELIC", param, kodak, pad=64, n_repeats=3)
            )
            del net
            torch.cuda.empty_cache()
    except ImportError as e:
        print(f"  [!] ELIC: {e}")

    # === TCM ===
    print("\n=== TCM ===")
    # Pulisci cache moduli per evitare conflitti TCM↔DCAE
    for mod in list(sys.modules.keys()):
        if "models" in mod and "compressai" not in mod:
            del sys.modules[mod]
    sys.path.insert(0, os.path.expanduser("~/tesi/external_codecs/LIC_TCM"))
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
            all_results.extend(
                benchmark_neural(net, "TCM", param, kodak, pad=64, n_repeats=3)
            )
            del net
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [!] TCM: {e}")

    # === DCAE ===
    print("\n=== DCAE ===")
    for mod in list(sys.modules.keys()):
        if "models" in mod and "compressai" not in mod:
            del sys.modules[mod]
    sys.path.insert(0, os.path.expanduser("~/tesi/external_codecs/DCAE"))
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
                benchmark_neural(
                    net, "DCAE", param, kodak, pad=128, pad_mode="center", n_repeats=3
                )
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
    print(f"\n{'=' * 90}")
    print("RIEPILOGO R-D-E DEFINITIVO (batch, idle 3s, GC off)")
    print(f"{'=' * 90}")
    print(
        f"{'Codec':<20} {'bpp':>6} {'PSNR':>6} {'E/img':>8} {'CPU/img':>8} {'GPU/img':>8} {'reps':>5} {'time':>7}"
    )
    print("-" * 90)

    for r in sorted(all_results, key=lambda x: (x["codec"], x.get("avg_bpp", 0))):
        cpu_per = r["energy_cpu_net_j"]
        gpu_per = r["energy_gpu_net_j"]
        print(
            f"{r['codec']+' '+r['param']:<20} {r['avg_bpp']:>6.3f} {r['avg_psnr']:>6.2f} "
            f"{r['energy_per_image_j']:>8.4f} {cpu_per:>8.4f} {gpu_per:>8.4f} "
            f"{r['n_repeats']:>5} {r['total_time_s']:>6.1f}s"
        )


if __name__ == "__main__":
    main()
