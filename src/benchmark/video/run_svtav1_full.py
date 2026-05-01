"""
Benchmark energetico SVT-AV1 — VERSIONE PRESET MULTIPLI.

Modifiche rispetto alla v1:
  - PRESETS è ora una LISTA di preset SVT-AV1 da testare
  - op_id include il preset → bitstream filename: svtav1_<seq>_<profile>_p<N>_crf<CRF>.ivf
  - CSV ora include colonna `preset` (post-migration)
  - I run su preset già eseguiti vengono saltati grazie a check_existing pattern

NOTA: il preset originale (p5) è GIÀ stato eseguito ed è nel CSV.
Questa versione esegue SOLO i nuovi preset (p2 = slow/quality, p10 = fast).
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

CODEC = "svtav1"
CSV_PATH = RESULTS_DIR / "svtav1_energy.csv"

# Preset NUOVI da testare (p5 già eseguito in run originale, NON ripetiamo)
PRESETS = ["2", "10"]
SVTAV1_CRF_POINTS = [27, 35, 43, 51]


def build_svtav1_params(profile, n_frames):
    if profile == "LDP":
        params = [
            "pred-struct=1",
            f"keyint={n_frames}",
            "irefresh-type=2",
        ]
    elif profile == "RA":
        params = [
            "pred-struct=2",
            "keyint=64",
            "irefresh-type=2",
        ]
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return ":".join(params)


def encode_svtav1_cmd(width, height, fps, n_frames, profile, preset, crf, out_path):
    svtav1_params = build_svtav1_params(profile, n_frames)
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
        "libsvtav1",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-b:v",
        "0",
        "-svtav1-params",
        svtav1_params,
        "-frames:v",
        str(n_frames),
        "-f",
        "ivf",
        str(out_path),
    ]


def decode_svtav1_cmd(bitstream_in, yuv_out, width, height):
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


def run_encode(yuv_bytes, width, height, fps, n_frames, profile, preset, crf, out_path):
    cmd = encode_svtav1_cmd(
        width, height, fps, n_frames, profile, preset, crf, out_path
    )
    result = subprocess.run(cmd, input=yuv_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"SVT-AV1 encode failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')[:800]}"
        )


def run_decode(bitstream_in, yuv_out, width, height):
    cmd = decode_svtav1_cmd(bitstream_in, yuv_out, width, height)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"SVT-AV1 decode failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr.decode('utf-8', errors='replace')[:500]}"
        )


def measure_config(
    yuv_bytes, seq_name, width, height, fps, n_frames, profile, preset, crf
):
    op_id = f"p{preset}_crf{crf}"
    bs_path = bitstream_path(CODEC, seq_name, profile, op_id, "ivf")
    dec_path = decoded_path(CODEC, seq_name, profile, op_id)

    if bs_path.exists() and bs_path.stat().st_size > 1000:
        print(f"      [SKIP] {bs_path.name} già esiste")
        return

    warmup_bytes = yuv_bytes[: yuv_frame_size_bytes(width, height) * 30]
    warmup_out = BITSTREAMS_DIR / "__warmup.ivf"
    try:
        run_encode(
            warmup_bytes, width, height, fps, 30, profile, preset, crf, warmup_out
        )
    except Exception as e:
        print(f"      [warmup warning] {e}")
    finally:
        if warmup_out.exists():
            warmup_out.unlink()

    cpu_idle_1, gpu_idle_1 = measure_idle()

    def do_encode():
        run_encode(
            yuv_bytes, width, height, fps, n_frames, profile, preset, crf, bs_path
        )

    enc_result = measure_phase(do_encode, cpu_idle_1, gpu_idle_1, is_neural=False)
    actual_mbps = compute_actual_mbps(bs_path, n_frames, fps)

    append_row(
        CSV_PATH,
        {
            "codec": CODEC,
            "seq": seq_name,
            "profile": profile,
            "preset": f"p{preset}",
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
        f"      ENCODE: p{preset} crf={crf} actual={actual_mbps:.2f}Mbps "
        f"time={enc_result['time_s']:.1f}s E_cpu={enc_result['cpu_net_j']:.1f}J"
    )

    cpu_idle_2, gpu_idle_2 = measure_idle()

    def do_decode():
        run_decode(bs_path, dec_path, width, height)

    dec_result = measure_phase(do_decode, cpu_idle_2, gpu_idle_2, is_neural=False)

    append_row(
        CSV_PATH,
        {
            "codec": CODEC,
            "seq": seq_name,
            "profile": profile,
            "preset": f"p{preset}",
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
        f"      DECODE: time={dec_result['time_s']:.3f}s "
        f"E_cpu={dec_result['cpu_net_j']:.2f}J"
    )

    if dec_path.exists():
        dec_path.unlink()


def run_all():
    print(f"\n[Measurement] starting. CSV: {CSV_PATH}")
    t_start = time.time()

    n_total = len(UVG_SEQUENCES) * len(PROFILES) * len(PRESETS) * len(SVTAV1_CRF_POINTS)
    n_done = 0

    for seq_name, fname, W, H, fps, n_frames in UVG_SEQUENCES:
        print(f"\n[{seq_name}] loading YUV ({n_frames} frames, {W}x{H})...")
        yuv_path = UVG_DIR / fname
        yuv_bytes = load_yuv_raw(yuv_path, W, H, n_frames)

        for profile in PROFILES:
            print(f"  profile={profile}")
            for preset in PRESETS:
                print(f"    preset=p{preset}")
                for crf in SVTAV1_CRF_POINTS:
                    n_done += 1
                    print(f"    [{n_done}/{n_total}]")
                    try:
                        measure_config(
                            yuv_bytes,
                            seq_name,
                            W,
                            H,
                            fps,
                            n_frames,
                            profile,
                            preset,
                            crf,
                        )
                    except Exception as e:
                        print(f"      [ERROR] {profile} p{preset} crf={crf}: {e}")

        del yuv_bytes

    elapsed = time.time() - t_start
    print(f"\n[Measurement] Complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")


def main():
    print("=" * 70)
    print(f"SVT-AV1 VIDEO ENERGY BENCHMARK — EXTRA PRESETS")
    print(f"Presets to run: {[f'p{p}' for p in PRESETS]}")
    print(f"  (p5 already done in original run)")
    print(f"Operating points: CRF ∈ {SVTAV1_CRF_POINTS}")
    print(f"Profiles: {PROFILES}")
    print("=" * 70)
    print(f"Output CSV: {CSV_PATH} (append mode)")

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
    print(f"Next: python -u quality_svtav1.py > svtav1_quality_extra.log 2>&1 &")


if __name__ == "__main__":
    main()
