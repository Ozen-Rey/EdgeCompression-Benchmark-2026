"""
Calcolo qualità per i bitstream x265 prodotti da run_x265.py.

Legge i bitstream da /dev/shm/bitstreams/x265_*_crf*.265, e per ciascuno calcola:
  - VMAF           (Video Multi-method Assessment Fusion, Netflix)
  - PSNR-Y, PSNR-U, PSNR-V, PSNR-YUV (PSNR pesato 6:1:1)
  - SSIM Y         (SSIM sul canale luma)

Output:
  ~/tesi/results/video/x265_quality.csv

IMPORTANTE: questo script NON misura energia. Gira su CPU, può occupare parecchio.
Si può usare il PC nel frattempo.

Uso:
  conda activate tesi-video
  cd ~/tesi/src/benchmark/video
  nohup python -u quality_x265.py > x265_quality_run.log 2>&1 &
  echo "PID: $!"
"""

import os
import sys
import csv
import json
import subprocess
import time
import re
from pathlib import Path

from common import (
    FFMPEG,
    UVG_DIR,
    RESULTS_DIR,
    BITSTREAMS_DIR,
    UVG_SEQUENCES,
    CLASSIC_CRF_POINTS,
    PROFILES,
    setup_dirs,
    bitstream_path,
)

CODEC = "x265"
QUALITY_CSV = RESULTS_DIR / "x265_quality.csv"

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


# ================== VMAF ==================


def compute_vmaf(ref_yuv, dist_bitstream, width, height, fps):
    """
    Calcola VMAF tra sequenza originale (ref_yuv) e bitstream codificato.
    Ritorna il VMAF medio.
    """
    log_path = Path("/dev/shm") / f"__vmaf_{dist_bitstream.stem}.json"

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        # Input 0: distorted (bitstream HEVC)
        "-i",
        str(dist_bitstream),
        # Input 1: reference (YUV raw)
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


# ================== PSNR + SSIM ==================


def compute_psnr(ref_yuv, dist_bitstream, width, height, fps):
    """
    Calcola PSNR Y, U, V e PSNR-YUV pesato 6:1:1 (ITU-T standard 4:2:0).
    """
    psnr_log = Path("/dev/shm") / f"__psnr_{dist_bitstream.stem}.log"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(dist_bitstream),
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

    # Parsing dallo stderr (ffmpeg stampa summary)
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
        # Fallback: media per-frame dal log
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
        # Cap inf a 100 per la media (significa MSE=0 → identico → irrealistico per codec lossy)
        y_vals = [min(v, 100.0) for v in y_vals]
        u_vals = [min(v, 100.0) for v in u_vals]
        v_vals = [min(v, 100.0) for v in v_vals]
        psnr_y = sum(y_vals) / len(y_vals) if y_vals else 0.0
        psnr_u = sum(u_vals) / len(u_vals) if u_vals else 0.0
        psnr_v = sum(v_vals) / len(v_vals) if v_vals else 0.0

    psnr_log.unlink(missing_ok=True)

    psnr_yuv = (6 * psnr_y + psnr_u + psnr_v) / 8
    print(
        f"      PSNR y={psnr_y:.2f} u={psnr_u:.2f} v={psnr_v:.2f} "
        f"weighted={psnr_yuv:.2f} ({elapsed:.0f}s)"
    )
    return psnr_y, psnr_u, psnr_v, psnr_yuv


def compute_ssim(ref_yuv, dist_bitstream, width, height, fps):
    """SSIM Y. Restituisce la media (float 0..1)."""
    ssim_log = Path("/dev/shm") / f"__ssim_{dist_bitstream.stem}.log"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(dist_bitstream),
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


def compute_ms_ssim(ref_yuv, dist_bitstream, width, height, fps):
    """
    Multi-Scale SSIM sul canale Y. Richiede ffmpeg compilato con libvmaf
    che include ms_ssim; in alternativa usa un filtro esplicito.

    Strategia: usiamo il filtro VMAF con feature ms_ssim (più robusto
    del filtro separato ms_ssim che non è sempre compilato).
    """
    log_path = Path("/dev/shm") / f"__msssim_{dist_bitstream.stem}.json"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(dist_bitstream),
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


# ================== MAIN LOOP ==================


def main():
    print("=" * 70)
    print(f"x265 QUALITY COMPUTATION")
    print(f"Operating points: CRF ∈ {CLASSIC_CRF_POINTS}")
    print(f"Profiles: {PROFILES}")
    print("=" * 70)
    print(f"Output CSV: {QUALITY_CSV}")
    print(f"Reading bitstreams from: {BITSTREAMS_DIR}")

    setup_dirs()
    init_quality_csv(QUALITY_CSV)

    t_start = time.time()
    n_done = 0
    n_total = len(UVG_SEQUENCES) * len(PROFILES) * len(CLASSIC_CRF_POINTS)

    for seq_name, fname, W, H, fps, n_frames in UVG_SEQUENCES:
        ref_yuv = UVG_DIR / fname
        print(f"\n[{seq_name}]  (reference: {ref_yuv.name})")

        for profile in PROFILES:
            for crf in CLASSIC_CRF_POINTS:
                op_id = f"crf{crf}"
                bs_path = bitstream_path(CODEC, seq_name, profile, op_id, "265")
                if not bs_path.exists():
                    print(f"  [SKIP] {profile} {op_id}: bitstream missing at {bs_path}")
                    continue

                print(f"  {profile} {op_id} → {bs_path.name}")
                try:
                    vmaf_mean = compute_vmaf(ref_yuv, bs_path, W, H, fps)
                    psnr_y, psnr_u, psnr_v, psnr_yuv = compute_psnr(
                        ref_yuv, bs_path, W, H, fps
                    )
                    ssim_y = compute_ssim(ref_yuv, bs_path, W, H, fps)
                    ms_ssim_y = compute_ms_ssim(ref_yuv, bs_path, W, H, fps)
                    append_quality(
                        QUALITY_CSV,
                        {
                            "codec": CODEC,
                            "seq": seq_name,
                            "profile": profile,
                            "param": f"crf={crf}",
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
                    n_done += 1
                except Exception as e:
                    print(f"    [ERROR] {e}")

    elapsed = time.time() - t_start
    print(
        f"\nDONE. Computed {n_done}/{n_total} configurations in "
        f"{elapsed:.0f}s ({elapsed/60:.1f} min)"
    )
    print(f"CSV: {QUALITY_CSV}")


if __name__ == "__main__":
    main()
