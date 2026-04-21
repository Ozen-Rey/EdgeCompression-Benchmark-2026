"""
Benchmark energetico VVC (H.266) tramite libvvenc.

Gemello di run_x265.py / run_svtav1.py per il quarto codec classico.

Note tecniche libvvenc:
  - Encoder: libvvenc (H.266/VVC standard, Fraunhofer HHI)
  - Preset {faster, fast, medium, slow, slower}. Usiamo "faster" per limiti pratici:
    VVC è uno standard più complesso, anche "medium" è 5-10× più lento di x265.
  - CRF range simile a AV1 (0-63), scala diversa da HEVC.
  - Parametri custom passati via -vvenc-params.

Profili LDP/RA:
  - LDP: -g n_frames + intra-period coincidente, no GOP gerarchico
  - RA:  GOP=64, struttura B (default vvenc è già RA)

CRF points scelti per coprire range simile a HEVC {22,27,32,37}:
  CRF 27 ≈ alta qualità
  CRF 33 ≈ media-alta
  CRF 39 ≈ media-bassa
  CRF 45 ≈ bassa

Output:
  ~/tesi/results/video/vvenc_energy.csv
  /dev/shm/bitstreams/vvenc_*.266

Uso:
  nohup python -u run_vvenc.py > vvenc_full_run.log 2>&1 &
"""

import os
import sys
import subprocess
import time
from pathlib import Path

from common import (
    FFMPEG,
    UVG_DIR,
    RESULTS_DIR,
    BITSTREAMS_DIR,
    DECODED_DIR,
    UVG_SEQUENCES,
    PROFILES,
    yuv_frame_size_bytes,
    load_yuv_raw,
    compute_actual_mbps,
    measure_idle,
    measure_phase,
    init_csv,
    append_row,
    setup_dirs,
    bitstream_path,
    decoded_path,
)

CODEC = "vvenc"
CSV_PATH = RESULTS_DIR / "vvenc_energy.csv"
PRESET = "medium"  # faster, fast, medium, slow, slower. Scelgo "medium" per coerenza
# con x264/x265/SVT-AV1 (preset canonico di ciascun encoder).

# CRF range AV1-like (0-63), calibrato per bitrate analoghi a HEVC {22,27,32,37}
VVENC_CRF_POINTS = [27, 33, 39, 45]


# ================== libvvenc COMMAND BUILDERS ==================


def build_vvenc_args(profile, n_frames, crf):
    """
    libvvenc in ffmpeg accetta: -preset, -qp/-crf, -g (gop length).
    Per LDP vs RA usiamo -g e -bf (numero B-frame).

    NOTE: libvvenc usa GOP gerarchico per default (RA-like).
    Per emulare LDP forziamo -bf 0 (no B) e GOP molto lungo.
    """
    common_args = [
        "-preset",
        PRESET,
        "-qp",
        str(crf),  # libvvenc usa qp parameter
    ]
    if profile == "LDP":
        return common_args + [
            "-g",
            str(n_frames),  # un solo I-frame
            "-bf",
            "0",  # no B-frame
        ]
    elif profile == "RA":
        return common_args + [
            "-g",
            "64",  # GOP 64 (HEVC CTC standard)
            "-bf",
            "7",  # B-pyramid
        ]
    else:
        raise ValueError(f"Unknown profile: {profile}")


def encode_vvenc_cmd(width, height, fps, n_frames, profile, crf, out_path):
    enc_args = build_vvenc_args(profile, n_frames, crf)
    return [
        FFMPEG,
        "-y",
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
        "pipe:0",
        "-c:v",
        "libvvenc",
        *enc_args,
        "-frames:v",
        str(n_frames),
        "-f",
        "vvc",
        str(out_path),
    ]


def decode_vvenc_cmd(bitstream_in, yuv_out, width, height):
    return [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(bitstream_in),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        str(yuv_out),
    ]


# ================== RUNNERS ==================


def run_encode(yuv_bytes, width, height, fps, n_frames, profile, crf, out_path):
    cmd = encode_vvenc_cmd(width, height, fps, n_frames, profile, crf, out_path)
    result = subprocess.run(cmd, input=yuv_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"VVC encode failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')[:800]}"
        )


def run_decode(bitstream_in, yuv_out, width, height):
    cmd = decode_vvenc_cmd(bitstream_in, yuv_out, width, height)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"VVC decode failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')[:500]}"
        )


# ================== MEASUREMENT ==================


def measure_config(yuv_bytes, seq_name, width, height, fps, n_frames, profile, crf):
    op_id = f"crf{crf}"
    bs_path = bitstream_path(CODEC, seq_name, profile, op_id, "266")
    dec_path = decoded_path(CODEC, seq_name, profile, op_id)

    # Warmup
    warmup_bytes = yuv_bytes[: yuv_frame_size_bytes(width, height) * 30]
    warmup_out = BITSTREAMS_DIR / "__warmup.266"
    try:
        run_encode(warmup_bytes, width, height, fps, 30, profile, crf, warmup_out)
    except Exception as e:
        print(f"    [warmup warning] {e}")
    finally:
        if warmup_out.exists():
            warmup_out.unlink()

    # Idle cal #1
    cpu_idle_1, gpu_idle_1 = measure_idle()

    # Encode
    def do_encode():
        run_encode(yuv_bytes, width, height, fps, n_frames, profile, crf, bs_path)

    enc_result = measure_phase(do_encode, cpu_idle_1, gpu_idle_1, is_neural=False)
    actual_mbps = compute_actual_mbps(bs_path, n_frames, fps)

    append_row(
        CSV_PATH,
        {
            "codec": CODEC,
            "seq": seq_name,
            "profile": profile,
            "param": f"crf={crf}",
            "phase": "encode",
            "n_frames": n_frames,
            "time_s": round(enc_result["time_s"], 3),
            "energy_cpu_j": round(enc_result["cpu_net_j"], 3),
            "energy_gpu_j": round(enc_result["gpu_net_j"], 3),
            "energy_total_j": round(enc_result["total_net_j"], 3),
            "idle_cpu_w": round(cpu_idle_1, 2),
            "idle_gpu_w": round(gpu_idle_1, 2),
            "actual_mbps": round(actual_mbps, 3),
            "bitstream_path": str(bs_path),
        },
    )
    print(
        f"    ENCODE: crf={crf} actual={actual_mbps:.2f}Mbps "
        f"time={enc_result['time_s']:.1f}s E_cpu={enc_result['cpu_net_j']:.1f}J"
    )

    # Idle cal #2
    cpu_idle_2, gpu_idle_2 = measure_idle()

    # Decode
    def do_decode():
        run_decode(bs_path, dec_path, width, height)

    dec_result = measure_phase(do_decode, cpu_idle_2, gpu_idle_2, is_neural=False)

    append_row(
        CSV_PATH,
        {
            "codec": CODEC,
            "seq": seq_name,
            "profile": profile,
            "param": f"crf={crf}",
            "phase": "decode",
            "n_frames": n_frames,
            "time_s": round(dec_result["time_s"], 3),
            "energy_cpu_j": round(dec_result["cpu_net_j"], 3),
            "energy_gpu_j": round(dec_result["gpu_net_j"], 3),
            "energy_total_j": round(dec_result["total_net_j"], 3),
            "idle_cpu_w": round(cpu_idle_2, 2),
            "idle_gpu_w": round(gpu_idle_2, 2),
            "actual_mbps": round(actual_mbps, 3),
            "bitstream_path": str(bs_path),
        },
    )
    print(
        f"    DECODE: time={dec_result['time_s']:.3f}s "
        f"E_cpu={dec_result['cpu_net_j']:.2f}J"
    )

    if dec_path.exists():
        dec_path.unlink()


def run_all():
    print(f"\n[Measurement] starting. CSV: {CSV_PATH}")
    t_start = time.time()

    for seq_name, fname, W, H, fps, n_frames in UVG_SEQUENCES:
        print(f"\n[{seq_name}] loading YUV ({n_frames} frames, {W}x{H})...")
        yuv_path = UVG_DIR / fname
        yuv_bytes = load_yuv_raw(yuv_path, W, H, n_frames)

        for profile in PROFILES:
            print(f"  profile={profile}")
            for crf in VVENC_CRF_POINTS:
                try:
                    measure_config(
                        yuv_bytes, seq_name, W, H, fps, n_frames, profile, crf
                    )
                except Exception as e:
                    print(f"    [ERROR] {profile} crf={crf}: {e}")

        del yuv_bytes

    elapsed = time.time() - t_start
    print(f"\n[Measurement] Complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")


def main():
    print("=" * 70)
    print(f"VVC (libvvenc) VIDEO ENERGY BENCHMARK — preset={PRESET}")
    print(f"Operating points: CRF/QP ∈ {VVENC_CRF_POINTS}")
    print(f"Profiles: {PROFILES}")
    print("=" * 70)
    print(f"Output CSV: {CSV_PATH}")

    setup_dirs()
    init_csv(CSV_PATH)

    try:
        from common import read_rapl_uj

        _ = read_rapl_uj()
    except PermissionError:
        print("\n[FATAL] RAPL non leggibile. Esegui:")
        print("  sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj")
        sys.exit(1)

    run_all()

    print(f"\nDONE. Results: {CSV_PATH}")
    print(f"Next: python -u quality_vvenc.py > vvenc_quality_run.log 2>&1 &")


if __name__ == "__main__":
    main()
