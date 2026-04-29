"""
Benchmark energetico + quality DCVC-RT (CVPR 2025).

DCVC-RT è il modello più recente, pensato per real-time. Senza i kernel CUDA
custom (che non compilano per sm_120 Blackwell), gira in fallback PyTorch puro.
Stesso bug numerico del round-trip encode/decode di FM/DC → forward-only mode.

Differenze chiave rispetto a FM e DC:

  * I-frame (DMCI image_model RT):
      result = i_model.compress(x, qp) → {x_hat, bit_stream}
      Stessa convenzione di DC.

  * P-frame (DMC video_model RT):
      result = p_model.compress(x, qp) → {bit_stream} SOLO. NIENTE x_hat.
      RT non calcola x_hat in compress (ottimizzazione lazy: lo calcolerebbe
      solo se serve come riferimento).
      → Lo ricostruiamo MANUALMENTE accedendo a:
          feature = p_model.dpb[0].feature   (salvato durante compress)
          q_recon = p_model.q_recon[qp:qp+1, :, :, :]
          x_hat = p_model.recon_generation_net(feature, q_recon).clamp_(0, 1)

  * DPB internamente gestito dal modello (non più dict gestito noi):
      p_model.clear_dpb()           per reset all'inizio di nuova sequenza
      p_model.add_ref_frame(None, x_hat)  per aggiungere I-frame
      compress() aggiorna dpb internamente

  * Two entropy coders: per UVG 1080p è obbligatorio
      i_model.set_use_two_entropy_coders(True)
      p_model.set_use_two_entropy_coders(True)

  * QP shifting: il qp del P-frame viene shiftato in base a fa_idx tramite
      curr_qp = p_model.shift_qp(qp, fa_idx)

  * use_ada_i (feature adaptor reset):
      ogni `reset_interval` frame, chiamata a `prepare_feature_adaptor_i(last_qp)`
      che resetta la feature memory.

  * Q-indexes [0, 21, 42, 63] (come FM, qp_num=64).
"""

import os
import re
import sys
import csv
import json
import subprocess
import time
from pathlib import Path

# Repo DCVC-RT (è la root del repo dcvc)
DCVC_RT_REPO = Path.home() / "tesi" / "external_codecs" / "dcvc"
sys.path.insert(0, str(DCVC_RT_REPO))

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

# DCVC-RT internals
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import get_state_dict
from src.models.common_model import (
    CompressionModel,
)  # noqa: F401  (import esplicito per accedere a get_padding_size statico)
from src.utils.video_reader import YUV420Reader
from src.utils.video_writer import YUV420Writer
from src.utils.transforms import yuv_444_to_420, ycbcr420_to_444_np

# get_padding_size è metodo statico di CompressionModel in DCVC-RT
get_padding_size = CompressionModel.get_padding_size


# ================== CONFIG ==================

CODEC = "dcvc_rt"
ENERGY_CSV = RESULTS_DIR / "dcvc_rt_energy.csv"
QUALITY_CSV = RESULTS_DIR / "dcvc_rt_quality.csv"

CKPT_I = DCVC_RT_REPO / "checkpoints" / "cvpr2025_image.pth.tar"
CKPT_P = DCVC_RT_REPO / "checkpoints" / "cvpr2025_video.pth.tar"

# Operating points: rate_num=4, q_index linspace(0, 63) come FM
Q_INDEXES = [0, 21, 42, 63]
RATE_GOP_SIZE = 8
RESET_INTERVAL = 64
INDEX_MAP = [0, 1, 0, 2, 0, 2, 0, 2]

# Two entropy coders: True per content > 720p (UVG 1080p)
USE_TWO_ENTROPY_CODERS = True

PROFILE = "LDP"

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"

TMP_YUV_DIR = Path("/dev/shm/dcvc_rt_tmp")
TMP_YUV_DIR.mkdir(parents=True, exist_ok=True)


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
    """Carica DCVC-RT DMCI (I) + DMC (P)."""
    print(f"[load_models] loading DCVC-RT checkpoints...")
    t0 = time.time()

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    i_state_dict = get_state_dict(str(CKPT_I))
    i_model = DMCI()
    i_model.load_state_dict(i_state_dict)
    i_model = i_model.cuda().eval()

    p_state_dict = get_state_dict(str(CKPT_P))
    p_model = DMC()
    p_model.load_state_dict(p_state_dict)
    p_model = p_model.cuda().eval()

    # IMPORTANTE: update() inizializza l'entropy_coder. set_use_two_entropy_coders
    # va chiamato DOPO update, altrimenti entropy_coder è ancora None.
    # NOTA RT: signature è update(force_zero_thres=None), non force=True come FM/DC
    i_model.update()
    p_model.update()

    # Set two entropy coders (per UVG 1080p) DOPO update
    i_model.set_use_two_entropy_coders(USE_TWO_ENTROPY_CODERS)
    p_model.set_use_two_entropy_coders(USE_TWO_ENTROPY_CODERS)

    print(
        f"[load_models] ready in {time.time()-t0:.1f}s, "
        f"VRAM={torch.cuda.memory_allocated()/1024**2:.1f} MB"
    )
    return i_model, p_model


# ================== FORWARD ==================


def get_p_x_hat(p_model, qp):
    """
    Recupera x_hat del P-frame appena codificato. RT non lo restituisce in
    compress, va calcolato manualmente da feature in dpb + recon_generation_net.
    """
    feature = p_model.dpb[0].feature
    q_recon = p_model.q_recon[qp : qp + 1, :, :, :]
    with torch.no_grad():
        x_hat = p_model.recon_generation_net(feature, q_recon).clamp_(0, 1)
    return x_hat


def forward_sequence(
    i_model, p_model, seq_path, width, height, n_frames, q_index, dec_yuv_path
):
    """
    Forward DCVC-RT su tutta la sequenza. Single-pass.
    """
    device = next(i_model.parameters()).device
    src_reader = YUV420Reader(str(seq_path), width, height)
    # NOTA: get_padding_size in DCVC-RT ritorna (padding_right, padding_bottom).
    # Microsoft chiama con p=16 (vedi test_video.py linea 146 di RT).
    padding_r, padding_b = get_padding_size(height, width, 16)
    padding_l = 0
    padding_t = 0
    recon_writer = YUV420Writer(str(dec_yuv_path), width, height)

    # Reset modello
    p_model.clear_dpb()

    total_bits = 0.0
    n_i = 0
    n_p = 0
    last_qp = q_index  # tracking per prepare_feature_adaptor_i

    with torch.no_grad():
        for frame_idx in range(n_frames):
            # Lettura frame YUV420 - YUV420Reader di RT ritorna UINT8.
            # Va convertito a float [0, 1] prima di ycbcr420_to_444_np.
            y_np_u8, uv_np_u8 = src_reader.read_one_frame()
            y_np = y_np_u8.astype(np.float32) / 255.0
            uv_np = uv_np_u8.astype(np.float32) / 255.0
            yuv = ycbcr420_to_444_np(y_np, uv_np)
            x = np_image_to_tensor(yuv).to(device)
            x_padded = F.pad(
                x, (padding_l, padding_r, padding_t, padding_b), mode="replicate"
            )

            if frame_idx == 0:
                # I-frame: i_model.compress(x, qp) → {x_hat, bit_stream}
                result = i_model.compress(x_padded, q_index)
                x_hat_padded = result["x_hat"]
                bit_stream = result["bit_stream"]
                total_bits += len(bit_stream) * 8
                # Aggiungi I-frame al DPB del p_model come reference
                p_model.clear_dpb()
                p_model.add_ref_frame(None, x_hat_padded)
                last_qp = q_index
                n_i += 1
            else:
                # Determina fa_idx e shifted qp
                fa_idx = INDEX_MAP[frame_idx % RATE_GOP_SIZE]
                if RESET_INTERVAL > 0 and frame_idx % RESET_INTERVAL == 1:
                    p_model.prepare_feature_adaptor_i(last_qp)
                curr_qp = p_model.shift_qp(q_index, fa_idx)
                # P-frame: compress() → {bit_stream}, dpb aggiornato internamente
                result = p_model.compress(x_padded, curr_qp)
                bit_stream = result["bit_stream"]
                total_bits += len(bit_stream) * 8
                # Recupera x_hat manualmente
                x_hat_padded = get_p_x_hat(p_model, curr_qp)
                last_qp = curr_qp
                n_p += 1

            # Crop padding (rimane tensor 4D [B, C, H, W])
            x_hat_padded = x_hat_padded.clamp_(0, 1)
            x_hat = F.pad(
                x_hat_padded, (-padding_l, -padding_r, -padding_t, -padding_b)
            )
            # yuv_444_to_420 di RT opera su tensor 4D, ritorna (y, uv) tensori
            y_t, uv_t = yuv_444_to_420(x_hat)
            # Convert to numpy uint8 per YUV420Writer (che vuole uint8)
            y_np = y_t.squeeze(0).cpu().numpy()  # 1, H, W
            uv_np = uv_t.squeeze(0).cpu().numpy()  # 2, H/2, W/2
            y_rec_u8 = np.clip(y_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
            uv_rec_u8 = np.clip(uv_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
            recon_writer.write_one_frame(y=y_rec_u8, uv=uv_rec_u8)

    src_reader.close()
    recon_writer.close()
    torch.cuda.synchronize()

    return {"total_bits": total_bits, "n_i": n_i, "n_p": n_p}


# ================== METRICS (libvmaf) ==================


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
    m = re.search(r"PSNR\s+y:([\d.]+|inf)\s+u:([\d.]+|inf)\s+v:([\d.]+|inf)", stderr)
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
    seq_name, fname, width, height, fps, n_frames = seq_entry
    seq_path = UVG_DIR / fname

    if not seq_path.exists():
        print(f"[SKIP] {seq_path} not found")
        return

    param = f"q={q_index}"
    tag = f"{seq_name}_{PROFILE}_q{q_index}"
    dec_yuv_path = TMP_YUV_DIR / f"{CODEC}_{tag}.yuv"

    print(f"\n[{seq_name} {PROFILE} {param}] {n_frames} frames @ {width}x{height}")

    cpu_idle, gpu_idle = measure_idle()
    print(f"  idle: CPU={cpu_idle:.1f}W, GPU={gpu_idle:.1f}W")

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
        if dec_yuv_path.exists():
            dec_yuv_path.unlink()


def main():
    setup_dirs()
    init_csv(ENERGY_CSV)
    init_quality_csv(QUALITY_CSV)

    print("=" * 70)
    print(f"DCVC-RT ENERGY + QUALITY BENCHMARK (forward-only mode)")
    print(f"Energy CSV:   {ENERGY_CSV}")
    print(f"Quality CSV:  {QUALITY_CSV}")
    print(f"Q-indexes:    {Q_INDEXES}")
    print(f"USE_TWO_EC:   {USE_TWO_ENTROPY_CODERS}")
    print(f"SMOKE_TEST:   {SMOKE_TEST}")
    print("=" * 70)

    if not CKPT_I.exists() or not CKPT_P.exists():
        print(f"[FATAL] Checkpoint missing: I={CKPT_I.exists()} P={CKPT_P.exists()}")
        sys.exit(1)

    i_model, p_model = load_models()

    if SMOKE_TEST:
        seqs_to_run = [s for s in UVG_SEQUENCES if s[0] == "Beauty"]
        q_indexes_to_run = [42]
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
