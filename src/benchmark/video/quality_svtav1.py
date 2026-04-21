"""
Calcolo qualità per i bitstream SVT-AV1.

Gemello di quality_x264.py / quality_x265.py.

Output:
  ~/tesi/results/video/svtav1_quality.csv
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
    PROFILES,
    setup_dirs,
    bitstream_path,
)

# Import dei CRF points dallo script energetico per coerenza
from run_svtav1 import SVTAV1_CRF_POINTS

CODEC = "svtav1"
QUALITY_CSV = RESULTS_DIR / "svtav1_quality.csv"

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


def compute_vmaf(ref_yuv, dist_bitstream, width, height, fps):
    log_path = Path("/dev/shm") / f"__vmaf_{dist_bitstream.stem}.json"
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
            f"VMAF failed: {result.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    with open(log_path) as fh:
        data = json.load(fh)
    log_path.unlink()
    vmaf_mean = data["pooled_metrics"]["vmaf"]["mean"]
    print(f"      VMAF={vmaf_mean:.2f} ({elapsed:.0f}s)")
    return vmaf_mean


def compute_psnr(ref_yuv, dist_bitstream, width, height, fps):
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

    psnr_y = psnr_u = psnr_v = 0.0
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
        if psnr_log.exists():
            with open(psnr_log) as fh:
                for line in fh:
                    my = re.search(r"psnr_y:([\d.]+|inf)", line)
                    mu = re.search(r"psnr_u:([\d.]+|inf)", line)
                    mv = re.search(r"psnr_v:([\d.]+|inf)", line)
                    if my:
                        y_vals.append(
                            100.0 if my.group(1) == "inf" else float(my.group(1))
                        )
                    if mu:
                        u_vals.append(
                            100.0 if mu.group(1) == "inf" else float(mu.group(1))
                        )
                    if mv:
                        v_vals.append(
                            100.0 if mv.group(1) == "inf" else float(mv.group(1))
                        )
        if y_vals:
            psnr_y = sum(y_vals) / len(y_vals)
        if u_vals:
            psnr_u = sum(u_vals) / len(u_vals)
        if v_vals:
            psnr_v = sum(v_vals) / len(v_vals)

    psnr_log.unlink(missing_ok=True)
    psnr_yuv = (6 * psnr_y + psnr_u + psnr_v) / 8
    print(
        f"      PSNR y={psnr_y:.2f} u={psnr_u:.2f} v={psnr_v:.2f} "
        f"weighted={psnr_yuv:.2f} ({elapsed:.0f}s)"
    )
    return psnr_y, psnr_u, psnr_v, psnr_yuv


def compute_ssim(ref_yuv, dist_bitstream, width, height, fps):
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
    """Multi-Scale SSIM tramite libvmaf feature."""
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


def main():
    print("=" * 70)
    print(f"SVT-AV1 QUALITY COMPUTATION")
    print(f"Operating points: CRF ∈ {SVTAV1_CRF_POINTS}")
    print(f"Profiles: {PROFILES}")
    print("=" * 70)
    print(f"Output CSV: {QUALITY_CSV}")

    setup_dirs()
    init_quality_csv(QUALITY_CSV)

    t_start = time.time()
    n_done = 0
    n_total = len(UVG_SEQUENCES) * len(PROFILES) * len(SVTAV1_CRF_POINTS)

    for seq_name, fname, W, H, fps, n_frames in UVG_SEQUENCES:
        ref_yuv = UVG_DIR / fname
        print(f"\n[{seq_name}]")
        for profile in PROFILES:
            for crf in SVTAV1_CRF_POINTS:
                op_id = f"crf{crf}"
                bs_path = bitstream_path(CODEC, seq_name, profile, op_id, "ivf")
                if not bs_path.exists():
                    print(f"  [SKIP] {profile} {op_id}: bitstream missing")
                    continue
                print(f"  {profile} {op_id}")
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
    print(f"\nDONE. {n_done}/{n_total} in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"CSV: {QUALITY_CSV}")


if __name__ == "__main__":
    main()
