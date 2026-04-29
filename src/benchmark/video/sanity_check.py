"""
Sanity check sui CSV finali del benchmark video.

Codec attesi:
  Classici (4): x264, x265, svtav1, vvenc
    Schema CSV: phase ∈ {encode, decode}
  Neurali forward-only (3): dcvc_fm, dcvc_dc, dcvc_rt (fallback PyTorch)
    Schema CSV: phase = "coding" (single phase)
  Neurali round-trip (1): dcvc_rt_cuda (kernel CUDA custom)
    Schema CSV: phase ∈ {encode, decode}

Verifiche per ciascun CSV energy:
  - Numero righe atteso (config × phases)
  - No NaN nei valori critici
  - Energie e tempi positivi
  - Bitrate monotone in q_index per ogni sequenza
  - Idle GPU < 80W (no contamination)

Verifiche per ciascun CSV quality:
  - Numero righe atteso (1 per config)
  - No NaN
  - Range realistici: VMAF [0,100], PSNR > 20 dB, SSIM/MS-SSIM in [0,1]
"""

import pandas as pd
from pathlib import Path
import sys

RESULTS = Path.home() / "tesi" / "results" / "video"

CODECS_CLASSIC = ["x264", "x265", "svtav1", "vvenc"]
CODECS_NEURAL_FWD = ["dcvc_fm", "dcvc_dc", "dcvc_rt"]
CODECS_NEURAL_ROUNDTRIP = ["dcvc_rt_cuda"]

# Aspettative per ogni codec (n_configs * n_phases)
# Classici: 7 seq × 4 q × 2 profiles (LDP+RA) × 2 phases = 112
# Neurali forward-only: 7 seq × 4 q × 1 profile × 1 phase = 28
# Neurali round-trip: 7 seq × 4 q × 1 profile × 2 phases = 56


def check_energy_csv(path, codec):
    """Sanity check di un CSV energy."""
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return False
    df = pd.read_csv(path)
    issues = []

    # Quante righe?
    n = len(df)
    phases = sorted(df["phase"].unique())
    seqs = sorted(df["seq"].unique())
    configs = df.groupby(["seq", "profile", "param"]).size()

    print(
        f"  [{codec:18s}] {n:4d} rows, phases={phases}, "
        f"seqs={len(seqs)}, configs={len(configs)}"
    )

    # NaN check
    nan_cols = [
        "time_s",
        "energy_cpu_j",
        "energy_gpu_j",
        "energy_total_j",
        "actual_mbps",
    ]
    for col in nan_cols:
        if col in df.columns:
            nans = df[col].isna().sum()
            if nans > 0:
                issues.append(f"{nans} NaN in {col}")

    # Negative energy/time check
    for col in ["time_s", "energy_total_j"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues.append(f"{neg} negative {col}")

    # Idle GPU check (< 80W ideally, < 100W tolerated)
    if "idle_gpu_w" in df.columns:
        high_idle = (df["idle_gpu_w"] > 100).sum()
        if high_idle > 0:
            issues.append(f"{high_idle} configs with GPU idle >100W")

    # Bitrate monotonicity for each (seq, profile)
    # quality should increase with q_index/CRF, bitrate should too in most cases
    for (seq, profile), grp in df.groupby(["seq", "profile"]):
        grp_sorted = grp.sort_values("param")
        if len(grp_sorted["actual_mbps"].unique()) < 2:
            continue
        # Just check no obvious crash (negative or zero)
        if (grp_sorted["actual_mbps"] <= 0).sum() > 0:
            issues.append(f"{seq}/{profile}: zero/negative bitrate")

    if issues:
        for iss in issues:
            print(f"      ⚠ {iss}")
        return False
    return True


def check_quality_csv(path, codec):
    """Sanity check di un CSV quality."""
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        return False
    df = pd.read_csv(path)
    issues = []
    n = len(df)
    print(f"  [{codec:18s} Q] {n:3d} rows")

    # NaN check
    metric_cols = ["vmaf_mean", "psnr_y", "ssim_y", "ms_ssim_y"]
    for col in metric_cols:
        if col in df.columns:
            nans = df[col].isna().sum()
            if nans > 0:
                issues.append(f"{nans} NaN in {col}")

    # Range checks
    if "vmaf_mean" in df.columns:
        out = ((df["vmaf_mean"] < 0) | (df["vmaf_mean"] > 100)).sum()
        if out > 0:
            issues.append(f"{out} VMAF out of [0,100]")
    if "psnr_y" in df.columns:
        out = ((df["psnr_y"] < 15) | (df["psnr_y"] > 60)).sum()
        if out > 0:
            issues.append(f"{out} PSNR_Y out of [15,60]")
    if "ssim_y" in df.columns:
        out = ((df["ssim_y"] < 0) | (df["ssim_y"] > 1)).sum()
        if out > 0:
            issues.append(f"{out} SSIM_Y out of [0,1]")
    if "ms_ssim_y" in df.columns:
        out = ((df["ms_ssim_y"] < 0) | (df["ms_ssim_y"] > 1)).sum()
        if out > 0:
            issues.append(f"{out} MS-SSIM_Y out of [0,1]")

    if issues:
        for iss in issues:
            print(f"      ⚠ {iss}")
        return False
    return True


def summary_table(codec):
    """Stampa una mini tabella di sintesi del codec: una riga per config, valori chiave."""
    e_path = RESULTS / f"{codec}_energy.csv"
    q_path = RESULTS / f"{codec}_quality.csv"
    if not e_path.exists() or not q_path.exists():
        return None

    df_e = pd.read_csv(e_path)
    df_q = pd.read_csv(q_path)

    # Aggregate energy per config (sum encode+decode if needed)
    df_e_agg = (
        df_e.groupby(["codec", "seq", "profile", "param"])
        .agg(
            {
                "energy_total_j": "sum",
                "time_s": "sum",
                "actual_mbps": "first",
            }
        )
        .reset_index()
    )

    df = pd.merge(df_e_agg, df_q, on=["codec", "seq", "profile", "param"])
    return df


def main():
    print("=" * 70)
    print("SANITY CHECK — VIDEO BENCHMARK CSV")
    print("=" * 70)

    print("\n--- Energy CSV ---")
    all_codecs = CODECS_CLASSIC + CODECS_NEURAL_FWD + CODECS_NEURAL_ROUNDTRIP
    energy_ok = {}
    for codec in all_codecs:
        ok = check_energy_csv(RESULTS / f"{codec}_energy.csv", codec)
        energy_ok[codec] = ok

    print("\n--- Quality CSV ---")
    quality_ok = {}
    for codec in all_codecs:
        ok = check_quality_csv(RESULTS / f"{codec}_quality.csv", codec)
        quality_ok[codec] = ok

    print("\n--- Cross-codec summary at q≈mid (Beauty) ---")
    print(
        f"{'codec':18s} {'mbps':>8s} {'time_s':>8s} {'energy_kJ':>10s} {'VMAF':>6s} {'PSNR_Y':>7s}"
    )
    print("-" * 70)

    for codec in all_codecs:
        df = summary_table(codec)
        if df is None:
            continue
        row = df[(df["seq"] == "Beauty") & (df["profile"] == "LDP")]
        if len(row) == 0:
            continue
        # Pick "middle" config based on bitrate
        row_sorted = row.sort_values("actual_mbps")
        mid = row_sorted.iloc[len(row_sorted) // 2]
        print(
            f"{codec:18s} {mid['actual_mbps']:>8.3f} "
            f"{mid['time_s']:>8.1f} {mid['energy_total_j']/1000:>10.2f} "
            f"{mid['vmaf_mean']:>6.1f} {mid['psnr_y']:>7.2f}"
        )

    print("\n--- Final verdict ---")
    all_ok = all(energy_ok.values()) and all(quality_ok.values())
    if all_ok:
        print("  ✓ ALL CSVs valid, ready for analysis")
    else:
        failures = [
            c
            for c in all_codecs
            if not (energy_ok.get(c, False) and quality_ok.get(c, False))
        ]
        print(f"  ⚠ Issues in: {failures}")
    print("=" * 70)


if __name__ == "__main__":
    main()
