"""
Benchmark energetico VVC (libvvenc) — VERSIONE PRESET MULTIPLI.

Modifiche rispetto alla v1:
  - PRESETS lista di preset libvvenc (escluso "medium" già fatto)
  - op_id include il preset → bitstream filename: vvenc_<seq>_<profile>_<preset>_crf<CRF>.266
  - CSV include colonna `preset` (post-migration)
  - Skip se bitstream esiste

NOTA: il preset originale (medium) è GIÀ stato eseguito ed è nel CSV.
Questa versione esegue SOLO i nuovi preset (faster, slow).

ATTENZIONE: VVC è il codec più lento in assoluto.
  - faster: ~30-50s per encode 600 frame
  - slow:  ~10-15 min per encode 600 frame (significativamente lento!)
Tempo totale stimato per i 2 nuovi preset: 8-12 ore.
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

# Preset NUOVI da testare (medium già eseguito in run originale, NON ripetiamo)
# faster=fast, fast, medium (già), slow, slower
# slow è già lentissimo (~10-15 min/encode), slower sarebbe impraticabile
PRESETS = ["faster", "slow"]

# CRF range vvenc-specific (NON usa CLASSIC_CRF_POINTS)
VVENC_CRF_POINTS = [27, 33, 39, 45]


def build_vvenc_args(profile, n_frames, preset, crf):
    common_args = [
        "-preset",
        preset,
        "-qp",
        str(crf),
    ]
    if profile == "LDP":
        return common_args + [
            "-g",
            str(n_frames),
            "-bf",
            "0",
        ]
    elif profile == "RA":
        return common_args + [
            "-g",
            "64",
            "-bf",
            "7",
        ]
    else:
        raise ValueError(f"Unknown profile: {profile}")


def encode_vvenc_cmd(width, height, fps, n_frames, profile, preset, crf, out_path):
    enc_args = build_vvenc_args(profile, n_frames, preset, crf)
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


def run_encode(yuv_bytes, width, height, fps, n_frames, profile, preset, crf, out_path):
    cmd = encode_vvenc_cmd(width, height, fps, n_frames, profile, preset, crf, out_path)
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


def measure_config(
    yuv_bytes, seq_name, width, height, fps, n_frames, profile, preset, crf
):
    op_id = f"{preset}_crf{crf}"
    bs_path = bitstream_path(CODEC, seq_name, profile, op_id, "266")
    dec_path = decoded_path(CODEC, seq_name, profile, op_id)

    if bs_path.exists() and bs_path.stat().st_size > 1000:
        print(f"      [SKIP] {bs_path.name} già esiste")
        return

    warmup_bytes = yuv_bytes[: yuv_frame_size_bytes(width, height) * 30]
    warmup_out = BITSTREAMS_DIR / "__warmup.266"
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
            "preset": preset,
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
        f"      ENCODE: {preset} crf={crf} actual={actual_mbps:.2f}Mbps "
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
            "preset": preset,
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

    n_total = len(UVG_SEQUENCES) * len(PROFILES) * len(PRESETS) * len(VVENC_CRF_POINTS)
    n_done = 0

    for seq_name, fname, W, H, fps, n_frames in UVG_SEQUENCES:
        print(f"\n[{seq_name}] loading YUV ({n_frames} frames, {W}x{H})...")
        yuv_path = UVG_DIR / fname
        yuv_bytes = load_yuv_raw(yuv_path, W, H, n_frames)

        for profile in PROFILES:
            print(f"  profile={profile}")
            for preset in PRESETS:
                print(f"    preset={preset}")
                for crf in VVENC_CRF_POINTS:
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
                        print(f"      [ERROR] {profile} {preset} crf={crf}: {e}")

        del yuv_bytes

    elapsed = time.time() - t_start
    print(f"\n[Measurement] Complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")


def main():
    print("=" * 70)
    print(f"VVC (libvvenc) VIDEO ENERGY BENCHMARK — EXTRA PRESETS")
    print(f"Presets to run: {PRESETS}")
    print(f"  (medium already done in original run)")
    print(f"Operating points: CRF/QP ∈ {VVENC_CRF_POINTS}")
    print(f"Profiles: {PROFILES}")
    print(f"WARNING: 'slow' preset is VERY slow (~10-15min/encode).")
    print(f"Estimated total time: 8-12 hours.")
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
    print(f"Next: python -u quality_vvenc.py > vvenc_quality_extra.log 2>&1 &")


if __name__ == "__main__":
    main()
