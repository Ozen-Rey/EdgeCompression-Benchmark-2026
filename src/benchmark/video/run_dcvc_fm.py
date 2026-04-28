"""
Benchmark energetico + quality DCVC-FM.

Differenze chiave rispetto ai classici:
  - Modalità FORWARD-ONLY: usa forward_one_frame() invece di encode()+decompress().
    Motivazione: il round-trip encode/decode tramite range coder C++ produce
    inconsistenze numeriche su hardware Blackwell (sm_120) per via del fallback
    PyTorch grid_sample. Il forward del modello produce comunque l'x_hat
    ricostruito (identico al decoded che uscirebbe se il decoder funzionasse)
    e i bits sono stimati dall'entropy model (convenzione del paper DCVC-FM).
  - Single phase "coding": l'energia è la forward pass completa, non separabile
    in encode/decode come per i classici.
  - Profilo solo LDP (i neurali non supportano RA per design).
  - Quality calcolato inline durante lo stesso run (su YUV in tmpfs):
    - VMAF, PSNR, SSIM, MS-SSIM via libvmaf (stessa toolchain dei classici)
    - YUV decoded scritto in /dev/shm temporaneamente (~1.8 GB picco) e cancellato

Per ogni (seq, q_index):
  1. Idle cal #1 (CPU + GPU)
  2. Forward 600 frame: accumula bits, scrive x_hat → /dev/shm/dec.yuv
     Misura energia e tempo del forward.
  3. Idle cal #2 (post forward, GPU calda)
  4. Quality calc su YUV: VMAF/PSNR/SSIM/MS-SSIM via libvmaf
  5. Cancella YUV
  6. Append CSV energy + CSV quality

SMOKE_TEST=1 → solo Beauty + q_index=21 (1 config, ~5 min).
SMOKE_TEST=0 → 7 sequenze × 4 q_index = 28 config, ~3-4 ore.

Uso:
  conda activate dcvc-eval
  cd ~/tesi/src/benchmark/video
  # smoke test
  python run_dcvc_fm.py
  # full run
  SMOKE_TEST=0 nohup python -u run_dcvc_fm.py > dcvc_fm_full_run.log 2>&1 &
"""

import os
import re
import sys
import csv
import json
import subprocess
import time
from pathlib import Path

# Repo DCVC-FM a sys.path
DCVC_FM_REPO = (
    Path.home() / "tesi" / "external_codecs" / "dcvc" / "DCVC-family" / "DCVC-FM"
)
sys.path.insert(0, str(DCVC_FM_REPO))

import torch
import torch.nn.functional as F
import numpy as np

from common import (
    FFMPEG,
    UVG_DIR,
    RESULTS_DIR,
    BITSTREAMS_DIR,
    UVG_SEQUENCES,
    measure_idle,
    measure_phase,
    init_csv,
    append_row,
    setup_dirs,
)

# DCVC-FM internals
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.video_reader import YUVReader
from src.utils.video_writer import YUVWriter
from src.transforms.functional import ycbcr420_to_444, ycbcr444_to_420


# ================== CONFIG ==================

CODEC = "dcvc_fm"
ENERGY_CSV = RESULTS_DIR / "dcvc_fm_energy.csv"
QUALITY_CSV = RESULTS_DIR / "dcvc_fm_quality.csv"

CKPT_I = DCVC_FM_REPO / "checkpoints" / "cvpr2024_image.pth.tar"
CKPT_P = DCVC_FM_REPO / "checkpoints" / "cvpr2024_video.pth.tar"

# Operating points: rate_num=4, q_index linspace(0, qp_num-1) come test_video.py
Q_INDEXES = [0, 21, 42, 63]
RATE_GOP_SIZE = 8
RESET_INTERVAL = 64

# Model instantiation params
EC_THREAD = False
STREAM_PART_I = 1
STREAM_PART_P = 1

PROFILE = "LDP"  # neurali solo LDP

# Smoke test
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"

# YUV temporaneo per quality (tmpfs)
TMP_YUV_DIR = Path("/dev/shm/dcvc_fm_tmp")
TMP_YUV_DIR.mkdir(parents=True, exist_ok=True)


# CSV quality fields (identico ai classici)
QUALITY_FIELDS = [
    "codec",
    "seq",
    "profile",
    "param",
    "vmaf_mean",
    "psnr_y",
    "psnr_u",
    "psnr_v",
    "psnr_yuv",
    "ssim_y",
    "ms_ssim_y",
    "n_frames",
]


def init_quality_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=QUALITY_FIELDS)
            w.writeheader()


def append_quality(path, row):
    full_row = {f: row.get(f, "") for f in QUALITY_FIELDS}
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QUALITY_FIELDS)
        w.writerow(full_row)


# ================== UTILITY ==================


def np_image_to_tensor(img):
    return torch.from_numpy(img).to(dtype=torch.float32).unsqueeze(0)


def load_models():
    """Carica DCVC-FM I-model e P-model."""
    print(f"[load_models] loading DCVC-FM checkpoints...")
    t0 = time.time()

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    i_state_dict = get_state_dict(str(CKPT_I))
    i_model = DMCI(ec_thread=EC_THREAD, stream_part=STREAM_PART_I, inplace=True)
    i_model.load_state_dict(i_state_dict)
    i_model = i_model.cuda().eval()

    p_state_dict = get_state_dict(str(CKPT_P))
    p_model = DMC(ec_thread=EC_THREAD, stream_part=STREAM_PART_P, inplace=True)
    p_model.load_state_dict(p_state_dict)
    p_model = p_model.cuda().eval()

    i_model.update(force=True)
    p_model.update(force=True)

    print(
        f"[load_models] ready in {time.time()-t0:.1f}s, "
        f"VRAM={torch.cuda.memory_allocated()/1024**2:.1f} MB"
    )
    return i_model, p_model


# ================== FORWARD (encode + decode in single pass) ==================


def forward_sequence(
    i_model, p_model, seq_path, width, height, n_frames, q_index, dec_yuv_path
):
    """
    Forward del modello DCVC-FM su tutta la sequenza, single-pass.
    Per ogni frame:
      - Frame 0: i_model.forward_one_frame(x, q_index) → x_hat + bits
      - Frame i>0: p_model.forward_one_frame(x, dpb, q_index, fa_idx) → x_hat + bits + dpb aggiornato
    Scrive x_hat come YUV420 su dec_yuv_path durante il loop.
    Ritorna: total_bits stimati, n_i, n_p.
    """
    device = next(i_model.parameters()).device
    src_reader = YUVReader(str(seq_path), width, height)
    padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)
    recon_writer = YUVWriter(str(dec_yuv_path), width, height)

    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    dpb: dict = {}
    total_bits = 0.0
    n_i = 0
    n_p = 0

    with torch.no_grad():
        for frame_idx in range(n_frames):
            y, uv = src_reader.read_one_frame(dst_format="420")
            yuv = ycbcr420_to_444(y, uv)
            x = np_image_to_tensor(yuv).to(device)
            x_padded = F.pad(
                x, (padding_l, padding_r, padding_t, padding_b), mode="replicate"
            )

            if frame_idx == 0:
                # I-frame
                result = i_model.forward_one_frame(x_padded, q_index)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                x_hat_padded = result["x_hat"]
                total_bits += (
                    result["bit"].item()
                    if hasattr(result["bit"], "item")
                    else result["bit"]
                )
                n_i += 1
            else:
                # P-frame
                fa_idx = index_map[frame_idx % RATE_GOP_SIZE]
                if RESET_INTERVAL > 0 and frame_idx % RESET_INTERVAL == 1:
                    dpb["ref_feature"] = None
                    fa_idx = 3
                result = p_model.forward_one_frame(x_padded, dpb, q_index, fa_idx)
                dpb = result["dpb"]
                x_hat_padded = dpb["ref_frame"]
                total_bits += (
                    result["bit"].item()
                    if hasattr(result["bit"], "item")
                    else result["bit"]
                )
                n_p += 1

            # Crop to original resolution + scrivi YUV
            x_hat_padded = x_hat_padded.clamp_(0, 1)
            x_hat = F.pad(
                x_hat_padded, (-padding_l, -padding_r, -padding_t, -padding_b)
            )
            yuv_rec = x_hat.squeeze(0).cpu().numpy()
            y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
            recon_writer.write_one_frame(y=y_rec, uv=uv_rec, src_format="420")

    src_reader.close()
    recon_writer.close()
    torch.cuda.synchronize()

    return {"total_bits": total_bits, "n_i": n_i, "n_p": n_p}


# ================== QUALITY METRICS (libvmaf, su due YUV file) ==================


def compute_vmaf_yuv(ref_yuv, dec_yuv, width, height, fps, tag):
    log_path = Path("/dev/shm") / f"__vmaf_{tag}.json"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(dec_yuv),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(ref_yuv),
        "-lavfi",
        f"[0:v][1:v]libvmaf=log_fmt=json:log_path={log_path}:n_threads={os.cpu_count()}",
        "-f",
        "null",
        "-",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"VMAF failed (rc={result.returncode}):\n"
            f"{result.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    with open(log_path) as fh:
        data = json.load(fh)
    log_path.unlink()
    vmaf_mean = data["pooled_metrics"]["vmaf"]["mean"]
    print(f"      VMAF={vmaf_mean:.2f} ({elapsed:.0f}s)")
    return vmaf_mean


def compute_psnr_yuv(ref_yuv, dec_yuv, width, height, fps, tag):
    psnr_log = Path("/dev/shm") / f"__psnr_{tag}.log"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "info",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(dec_yuv),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(ref_yuv),
        "-lavfi",
        f"[0:v][1:v]psnr=stats_file={psnr_log}",
        "-f",
        "null",
        "-",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True)
    elapsed = time.time() - t0
    stderr = result.stderr.decode("utf-8", errors="replace")

    psnr_y = psnr_u = psnr_v = None
    m = re.search(
        r"PSNR\s+y:([\d.]+|inf)\s+u:([\d.]+|inf)\s+v:([\d.]+|inf)",
        stderr,
    )
    if m:

        def parse_p(s):
            return float("inf") if s == "inf" else float(s)

        psnr_y = parse_p(m.group(1))
        psnr_u = parse_p(m.group(2))
        psnr_v = parse_p(m.group(3))
    else:
        y_vals, u_vals, v_vals = [], [], []
        with open(psnr_log) as fh:
            for line in fh:
                my = re.search(r"psnr_y:([\d.]+|inf)", line)
                mu = re.search(r"psnr_u:([\d.]+|inf)", line)
                mv = re.search(r"psnr_v:([\d.]+|inf)", line)
                if my:
                    y_vals.append(
                        float("inf") if my.group(1) == "inf" else float(my.group(1))
                    )
                if mu:
                    u_vals.append(
                        float("inf") if mu.group(1) == "inf" else float(mu.group(1))
                    )
                if mv:
                    v_vals.append(
                        float("inf") if mv.group(1) == "inf" else float(mv.group(1))
                    )
        y_vals = [min(v, 100.0) for v in y_vals]
        u_vals = [min(v, 100.0) for v in u_vals]
        v_vals = [min(v, 100.0) for v in v_vals]
        psnr_y = sum(y_vals) / len(y_vals) if y_vals else 0.0
        psnr_u = sum(u_vals) / len(u_vals) if u_vals else 0.0
        psnr_v = sum(v_vals) / len(v_vals) if v_vals else 0.0

    psnr_log.unlink(missing_ok=True)
    psnr_yuv = (6 * psnr_y + psnr_u + psnr_v) / 8
    print(
        f"      PSNR y={psnr_y:.2f} u={psnr_u:.2f} v={psnr_v:.2f} weighted={psnr_yuv:.2f} ({elapsed:.0f}s)"
    )
    return psnr_y, psnr_u, psnr_v, psnr_yuv


def compute_ssim_yuv(ref_yuv, dec_yuv, width, height, fps, tag):
    ssim_log = Path("/dev/shm") / f"__ssim_{tag}.log"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "info",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(dec_yuv),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(ref_yuv),
        "-lavfi",
        f"[0:v][1:v]ssim=stats_file={ssim_log}",
        "-f",
        "null",
        "-",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True)
    elapsed = time.time() - t0
    stderr = result.stderr.decode("utf-8", errors="replace")
    ssim_y = float("nan")
    m = re.search(r"SSIM\s+Y:([\d.]+)", stderr)
    if m:
        ssim_y = float(m.group(1))
    else:
        vals = []
        if ssim_log.exists():
            with open(ssim_log) as fh:
                for line in fh:
                    mm = re.search(r"Y:([\d.]+)", line)
                    if mm:
                        vals.append(float(mm.group(1)))
        if vals:
            ssim_y = sum(vals) / len(vals)
    ssim_log.unlink(missing_ok=True)
    print(f"      SSIM Y={ssim_y:.4f} ({elapsed:.0f}s)")
    return ssim_y


def compute_ms_ssim_yuv(ref_yuv, dec_yuv, width, height, fps, tag):
    log_path = Path("/dev/shm") / f"__msssim_{tag}.json"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(dec_yuv),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(ref_yuv),
        "-lavfi",
        f"[0:v][1:v]libvmaf=feature=name=float_ms_ssim:"
        f"log_fmt=json:log_path={log_path}:n_threads={os.cpu_count()}",
        "-f",
        "null",
        "-",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"      MS-SSIM failed, returning NaN")
        log_path.unlink(missing_ok=True)
        return float("nan")
    try:
        with open(log_path) as fh:
            data = json.load(fh)
        log_path.unlink()
        ms_ssim = data["pooled_metrics"]["float_ms_ssim"]["mean"]
    except (KeyError, FileNotFoundError):
        ms_ssim = float("nan")
    print(f"      MS-SSIM Y={ms_ssim:.4f} ({elapsed:.0f}s)")
    return ms_ssim


# ================== MAIN ==================


def process_config(i_model, p_model, seq_entry, q_index):
    """
    Processa un singolo punto operativo (seq, q_index) LDP.
    Scrive 1 riga in energy CSV + 1 riga in quality CSV.
    """
    seq_name, fname, width, height, fps, n_frames = seq_entry
    seq_path = UVG_DIR / fname

    if not seq_path.exists():
        print(f"[SKIP] {seq_path} not found")
        return

    param = f"q={q_index}"
    tag = f"{seq_name}_{PROFILE}_q{q_index}"
    dec_yuv_path = TMP_YUV_DIR / f"{CODEC}_{tag}.yuv"

    print(f"\n[{seq_name} {PROFILE} {param}] {n_frames} frames @ {width}x{height}")

    # --- IDLE CAL #1 ---
    cpu_idle, gpu_idle = measure_idle()
    print(f"  idle: CPU={cpu_idle:.1f}W, GPU={gpu_idle:.1f}W")

    # --- FORWARD (measured: energy + time) ---
    fwd_info: dict = {}

    def do_forward():
        nonlocal fwd_info
        fwd_info = forward_sequence(
            i_model, p_model, seq_path, width, height, n_frames, q_index, dec_yuv_path
        )

    fwd_result = measure_phase(do_forward, cpu_idle, gpu_idle, is_neural=True)
    total_bits = fwd_info["total_bits"]
    actual_mbps = total_bits / (n_frames / fps) / 1e6
    print(
        f"  CODING: {fwd_result['time_s']:.2f}s, "
        f"CPU={fwd_result['cpu_net_j']:.1f}J GPU={fwd_result['gpu_net_j']:.1f}J, "
        f"{actual_mbps:.3f} Mbps (estimated), I={fwd_info['n_i']} P={fwd_info['n_p']}"
    )

    append_row(
        ENERGY_CSV,
        {
            "codec": CODEC,
            "seq": seq_name,
            "profile": PROFILE,
            "param": param,
            "phase": "coding",
            "n_frames": n_frames,
            "time_s": round(fwd_result["time_s"], 3),
            "energy_cpu_j": round(fwd_result["cpu_net_j"], 3),
            "energy_gpu_j": round(fwd_result["gpu_net_j"], 3),
            "energy_total_j": round(fwd_result["total_net_j"], 3),
            "idle_cpu_w": round(cpu_idle, 2),
            "idle_gpu_w": round(gpu_idle, 2),
            "actual_mbps": round(actual_mbps, 3),
        },
    )

    # --- QUALITY (NOT measured: post-forward) ---
    print(f"  quality calc on YUV ({dec_yuv_path.stat().st_size/1024**2:.0f} MB)")
    try:
        vmaf_mean = compute_vmaf_yuv(seq_path, dec_yuv_path, width, height, fps, tag)
        psnr_y, psnr_u, psnr_v, psnr_yuv = compute_psnr_yuv(
            seq_path, dec_yuv_path, width, height, fps, tag
        )
        ssim_y = compute_ssim_yuv(seq_path, dec_yuv_path, width, height, fps, tag)
        ms_ssim_y = compute_ms_ssim_yuv(seq_path, dec_yuv_path, width, height, fps, tag)

        append_quality(
            QUALITY_CSV,
            {
                "codec": CODEC,
                "seq": seq_name,
                "profile": PROFILE,
                "param": param,
                "vmaf_mean": round(vmaf_mean, 3),
                "psnr_y": round(psnr_y, 3),
                "psnr_u": round(psnr_u, 3),
                "psnr_v": round(psnr_v, 3),
                "psnr_yuv": round(psnr_yuv, 3),
                "ssim_y": round(ssim_y, 5),
                "ms_ssim_y": round(ms_ssim_y, 5),
                "n_frames": n_frames,
            },
        )
    except Exception as e:
        print(f"    [ERROR quality] {e}")
    finally:
        # Cleanup YUV temporaneo
        if dec_yuv_path.exists():
            dec_yuv_path.unlink()


def main():
    setup_dirs()
    init_csv(ENERGY_CSV)
    init_quality_csv(QUALITY_CSV)

    print("=" * 70)
    print(f"DCVC-FM ENERGY + QUALITY BENCHMARK (forward-only mode)")
    print(f"Energy CSV:   {ENERGY_CSV}")
    print(f"Quality CSV:  {QUALITY_CSV}")
    print(f"Q-indexes:    {Q_INDEXES}")
    print(f"SMOKE_TEST:   {SMOKE_TEST}")
    print("=" * 70)

    if not CKPT_I.exists() or not CKPT_P.exists():
        print(f"[FATAL] Checkpoint missing")
        sys.exit(1)

    # Load models ONCE
    i_model, p_model = load_models()

    # Select config matrix
    if SMOKE_TEST:
        seqs_to_run = [s for s in UVG_SEQUENCES if s[0] == "Beauty"]
        q_indexes_to_run = [21]
    else:
        seqs_to_run = UVG_SEQUENCES
        q_indexes_to_run = Q_INDEXES

    n_configs = len(seqs_to_run) * len(q_indexes_to_run)
    t_run_start = time.time()
    done = 0

    for seq_entry in seqs_to_run:
        for q_index in q_indexes_to_run:
            done += 1
            print(f"\n========= [{done}/{n_configs}] =========")
            process_config(i_model, p_model, seq_entry, q_index)

    total_elapsed = time.time() - t_run_start
    print(
        f"\n\nDONE. {done}/{n_configs} configs in {total_elapsed:.1f}s "
        f"({total_elapsed/60:.1f} min)"
    )
    print(f"Energy CSV:  {ENERGY_CSV}")
    print(f"Quality CSV: {QUALITY_CSV}")


if __name__ == "__main__":
    main()
