"""
Script di aggregazione master CSV per il benchmark video.

Cross-platform (Windows + Linux). Auto-detecta la cartella results/video/
salendo dalla posizione dello script.

Uso:
  python aggregate_video_master.py
  # → results/video/master.csv

  # Oppure path custom:
  python aggregate_video_master.py --results-dir C:\path\to\results\video
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict


# === Definizione codec ===

CLASSIC_CODECS = ["x264", "x265", "svtav1", "vvenc"]
NEURAL_FORWARD_ONLY = ["dcvc_dc", "dcvc_fm", "dcvc_rt"]
NEURAL_ROUND_TRIP = ["dcvc_rt_cuda"]
ALL_NEURAL = NEURAL_FORWARD_ONLY + NEURAL_ROUND_TRIP

FAMILY = {
    "x264": "AVC",
    "x265": "HEVC",
    "svtav1": "AV1",
    "vvenc": "VVC",
    "dcvc_dc": "DCVC-DC",
    "dcvc_fm": "DCVC-FM",
    "dcvc_rt": "DCVC-RT (forward)",
    "dcvc_rt_cuda": "DCVC-RT (CUDA)",
}

SEQ_FRAMES = {
    "Beauty": 600, "Bosphorus": 600, "HoneyBee": 600, "Jockey": 600,
    "ReadySteadyGo": 600, "ShakeNDry": 300, "YachtRide": 600,
}


# === Schema master ===

MASTER_FIELDS = [
    "codec", "family", "seq", "profile", "preset", "param",
    "n_frames",
    "is_neural", "is_round_trip",
    "actual_mbps",
    "vmaf_mean", "psnr_y", "psnr_u", "psnr_v", "psnr_yuv",
    "ssim_y", "ms_ssim_y",
    "time_encode_s", "energy_encode_cpu_j", "energy_encode_gpu_j",
    "energy_encode_total_j",
    "idle_cpu_w_encode", "idle_gpu_w_encode",
    "time_decode_s", "energy_decode_cpu_j", "energy_decode_gpu_j",
    "energy_decode_total_j",
    "idle_cpu_w_decode", "idle_gpu_w_decode",
    "energy_total_cpu_j", "energy_total_gpu_j",
    "energy_total_j", "energy_total_kj",
    "energy_per_frame_j",
    "bitstream_path",
]


# === Auto-detect path ===

def find_results_dir():
    """
    Cerca la cartella results/video/ partendo dalla posizione dello script
    e risalendo fino a 5 livelli. Compatibile con qualsiasi layout.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "results" / "video",
        script_dir.parent / "results" / "video",
        script_dir.parent.parent / "results" / "video",
        script_dir.parent.parent.parent / "results" / "video",
        Path.home() / "tesi" / "results" / "video",  # fallback Linux original
    ]
    for c in candidates:
        if c.exists() and (c / "x264_energy.csv").exists():
            return c
    return None


# === Helpers ===

def load_csv(path):
    if not path.exists():
        print(f"  [WARN] missing: {path.name}")
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def safe_float(s, default=None):
    if s is None or s == "":
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def make_key(row):
    preset = row.get("preset", "") or ""
    return (row["codec"], row["seq"], row["profile"], preset, row["param"])


def aggregate_codec(codec, results_dir):
    is_neural = codec in ALL_NEURAL
    is_round_trip = codec in CLASSIC_CODECS or codec in NEURAL_ROUND_TRIP

    energy_csv = results_dir / f"{codec}_energy.csv"
    quality_csv = results_dir / f"{codec}_quality.csv"

    energy_rows = load_csv(energy_csv)
    quality_rows = load_csv(quality_csv)

    if not energy_rows:
        print(f"  [SKIP] {codec}: no energy data")
        return {}

    energy_by_key = defaultdict(dict)
    for row in energy_rows:
        key = make_key(row)
        phase = row["phase"]
        energy_by_key[key][phase] = row

    quality_by_key = {}
    for row in quality_rows:
        key = make_key(row)
        quality_by_key[key] = row

    master = {}
    for key, phases in energy_by_key.items():
        codec_, seq, profile, preset, param = key

        if "encode" in phases:
            enc = phases["encode"]
        elif "coding" in phases:
            enc = phases["coding"]
        else:
            print(f"  [WARN] {key}: no encode/coding phase")
            continue

        dec = phases.get("decode", None)
        q = quality_by_key.get(key, {})

        n_frames = int(safe_float(enc.get("n_frames"), 0))
        e_enc_cpu = safe_float(enc.get("energy_cpu_j"), 0.0) or 0.0
        e_enc_gpu = safe_float(enc.get("energy_gpu_j"), 0.0) or 0.0
        e_enc_tot = e_enc_cpu + e_enc_gpu

        if dec is not None:
            e_dec_cpu = safe_float(dec.get("energy_cpu_j"), 0.0) or 0.0
            e_dec_gpu = safe_float(dec.get("energy_gpu_j"), 0.0) or 0.0
            e_dec_tot = e_dec_cpu + e_dec_gpu
            time_dec = safe_float(dec.get("time_s"), None)
            idle_cpu_dec = safe_float(dec.get("idle_cpu_w"), None)
            idle_gpu_dec = safe_float(dec.get("idle_gpu_w"), None)
        else:
            e_dec_cpu = e_dec_gpu = e_dec_tot = None
            time_dec = idle_cpu_dec = idle_gpu_dec = None

        e_tot_cpu = e_enc_cpu + (e_dec_cpu or 0.0)
        e_tot_gpu = e_enc_gpu + (e_dec_gpu or 0.0)
        e_tot = e_tot_cpu + e_tot_gpu
        e_tot_kj = e_tot / 1000.0
        e_per_frame = e_tot / n_frames if n_frames > 0 else None

        master_row = {
            "codec": codec_,
            "family": FAMILY.get(codec_, codec_),
            "seq": seq,
            "profile": profile,
            "preset": preset,
            "param": param,
            "n_frames": n_frames,
            "is_neural": int(is_neural),
            "is_round_trip": int(is_round_trip),
            "actual_mbps": safe_float(enc.get("actual_mbps"), None),
            "vmaf_mean": safe_float(q.get("vmaf_mean"), None),
            "psnr_y": safe_float(q.get("psnr_y"), None),
            "psnr_u": safe_float(q.get("psnr_u"), None),
            "psnr_v": safe_float(q.get("psnr_v"), None),
            "psnr_yuv": safe_float(q.get("psnr_yuv"), None),
            "ssim_y": safe_float(q.get("ssim_y"), None),
            "ms_ssim_y": safe_float(q.get("ms_ssim_y"), None),
            "time_encode_s": safe_float(enc.get("time_s"), None),
            "energy_encode_cpu_j": e_enc_cpu,
            "energy_encode_gpu_j": e_enc_gpu,
            "energy_encode_total_j": e_enc_tot,
            "idle_cpu_w_encode": safe_float(enc.get("idle_cpu_w"), None),
            "idle_gpu_w_encode": safe_float(enc.get("idle_gpu_w"), None),
            "time_decode_s": time_dec,
            "energy_decode_cpu_j": e_dec_cpu,
            "energy_decode_gpu_j": e_dec_gpu,
            "energy_decode_total_j": e_dec_tot,
            "idle_cpu_w_decode": idle_cpu_dec,
            "idle_gpu_w_decode": idle_gpu_dec,
            "energy_total_cpu_j": e_tot_cpu,
            "energy_total_gpu_j": e_tot_gpu,
            "energy_total_j": e_tot,
            "energy_total_kj": e_tot_kj,
            "energy_per_frame_j": e_per_frame,
            "bitstream_path": enc.get("bitstream_path", ""),
        }
        master[key] = master_row

    return master


def write_master(all_rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            clean_row = {}
            for k, v in row.items():
                if v is None:
                    clean_row[k] = ""
                elif isinstance(v, float):
                    if k == "energy_total_kj":
                        clean_row[k] = round(v, 4)
                    elif k.startswith("energy") or k.startswith("idle"):
                        clean_row[k] = round(v, 3)
                    elif k.startswith("time"):
                        clean_row[k] = round(v, 3)
                    elif k in ("vmaf_mean", "psnr_y", "psnr_u", "psnr_v", "psnr_yuv"):
                        clean_row[k] = round(v, 3)
                    elif k in ("ssim_y", "ms_ssim_y"):
                        clean_row[k] = round(v, 5)
                    elif k == "actual_mbps":
                        clean_row[k] = round(v, 3)
                    elif k == "energy_per_frame_j":
                        clean_row[k] = round(v, 4)
                    else:
                        clean_row[k] = v
                else:
                    clean_row[k] = v
            writer.writerow(clean_row)


def print_summary(all_rows):
    by_codec = defaultdict(int)
    by_family = defaultdict(int)
    by_profile = defaultdict(int)
    has_quality = 0
    has_decode = 0

    for row in all_rows:
        by_codec[row["codec"]] += 1
        by_family[row["family"]] += 1
        by_profile[row["profile"]] += 1
        if row["vmaf_mean"] is not None:
            has_quality += 1
        if row["time_decode_s"] is not None:
            has_decode += 1

    print("\n" + "=" * 60)
    print("MASTER CSV STATISTICS")
    print("=" * 60)
    print(f"Total rows: {len(all_rows)}")
    print("\nBy codec:")
    for c, n in sorted(by_codec.items()):
        print(f"  {c:20s} {n:4d}")
    print("\nBy family:")
    for f, n in sorted(by_family.items()):
        print(f"  {f:25s} {n:4d}")
    print("\nBy profile:")
    for p, n in sorted(by_profile.items()):
        print(f"  {p:5s} {n:4d}")
    total = max(len(all_rows), 1)
    print("\nCoverage:")
    print(f"  Rows with quality:  {has_quality}/{total} ({100*has_quality/total:.1f}%)")
    print(f"  Rows with decode:   {has_decode}/{total} ({100*has_decode/total:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Aggrega CSV video in master.csv")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path alla cartella results/video (auto-detect se omesso)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path file master.csv (default: <results-dir>/master.csv)")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = find_results_dir()
        if results_dir is None:
            print("[ERROR] Cartella results/video/ non trovata in posizioni standard.")
            print("Usa --results-dir per specificare il path manualmente.")
            print("Esempio:")
            print("  python aggregate_video_master.py --results-dir C:\\path\\to\\results\\video")
            sys.exit(1)

    output_path = Path(args.output) if args.output else results_dir / "master.csv"

    print("=" * 60)
    print("VIDEO BENCHMARK MASTER CSV BUILDER")
    print("=" * 60)
    print(f"Reading from:  {results_dir}")
    print(f"Writing to:    {output_path}")
    print()

    if not results_dir.exists():
        print(f"[ERROR] Cartella non esistente: {results_dir}")
        sys.exit(1)

    all_rows = []

    print("[1/2] Processing classic codecs...")
    for codec in CLASSIC_CODECS:
        print(f"  -> {codec}")
        master = aggregate_codec(codec, results_dir)
        all_rows.extend(master.values())
        print(f"     {len(master)} rows aggregated")

    print("\n[2/2] Processing neural codecs...")
    for codec in ALL_NEURAL:
        print(f"  -> {codec}")
        master = aggregate_codec(codec, results_dir)
        all_rows.extend(master.values())
        print(f"     {len(master)} rows aggregated")

    all_rows.sort(key=lambda r: (
        r["codec"], r["seq"], r["profile"], r["preset"], r["param"]
    ))

    write_master(all_rows, output_path)
    print_summary(all_rows)

    print(f"DONE. Master CSV: {output_path}")
    print(f"Lines (incl header): {len(all_rows) + 1}")


if __name__ == "__main__":
    main()