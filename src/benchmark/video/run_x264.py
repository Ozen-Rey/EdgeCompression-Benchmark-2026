"""
Benchmark energetico x264 (H.264/AVC).

Gemello di run_x265.py. Stesso approccio (CRF fissi, bitrate osservato).

Note tecniche x264:
  - Parametri passati via -x264-params (sintassi "key=val:key2=val2")
  - "no-B-frame" si ottiene con bframes=0
  - Preset "medium" coerente con x265

Output:
  ~/tesi/results/video/x264_energy.csv
  /dev/shm/bitstreams/x264_*.264

Uso:
  conda activate tesi-video
  cd ~/tesi/src/benchmark/video
  nohup python -u run_x264.py > x264_full_run.log 2>&1 &
  echo "PID: $!"
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
    CLASSIC_CRF_POINTS,
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

CODEC = "x264"
CSV_PATH = RESULTS_DIR / "x264_energy.csv"
PRESET = "medium"


# ================== x264 COMMAND BUILDERS ==================


def build_x264_params(profile, n_frames):
    """
    Costruisce la stringa -x264-params coerente col profilo.
    """
    if profile == "LDP":
        params = [
            f"keyint={n_frames}",
            "min-keyint=1",
            "bframes=0",
            "ref=1",
        ]
    elif profile == "RA":
        params = [
            "keyint=64",
            "min-keyint=64",
            "bframes=7",
            "b-pyramid=normal",
        ]
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return ":".join(params)


def encode_x264_cmd(width, height, fps, n_frames, profile, crf, out_path):
    x264_params = build_x264_params(profile, n_frames)
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
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        str(crf),
        "-x264-params",
        x264_params,
        "-frames:v",
        str(n_frames),
        "-f",
        "h264",
        str(out_path),
    ]


def decode_x264_cmd(bitstream_in, yuv_out, width, height):
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
    cmd = encode_x264_cmd(width, height, fps, n_frames, profile, crf, out_path)
    result = subprocess.run(cmd, input=yuv_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"x264 encode failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')[:500]}"
        )


def run_decode(bitstream_in, yuv_out, width, height):
    cmd = decode_x264_cmd(bitstream_in, yuv_out, width, height)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"x264 decode failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')[:500]}"
        )


# ================== MEASUREMENT ==================


def measure_config(yuv_bytes, seq_name, width, height, fps, n_frames, profile, crf):
    op_id = f"crf{crf}"
    bs_path = bitstream_path(CODEC, seq_name, profile, op_id, "264")
    dec_path = decoded_path(CODEC, seq_name, profile, op_id)

    # Warmup
    warmup_bytes = yuv_bytes[: yuv_frame_size_bytes(width, height) * 30]
    warmup_out = BITSTREAMS_DIR / "__warmup.264"
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
            for crf in CLASSIC_CRF_POINTS:
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
    print(f"x264 VIDEO ENERGY BENCHMARK — preset={PRESET}")
    print(f"Operating points: CRF ∈ {CLASSIC_CRF_POINTS}")
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
    print(f"Next: python -u quality_x264.py > x264_quality_run.log 2>&1 &")


if __name__ == "__main__":
    main()
