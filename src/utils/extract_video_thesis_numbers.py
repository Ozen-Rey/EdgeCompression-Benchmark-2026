from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

REF_CSV = ROOT / "plots" / "video" / "reference_preset" / "video_reference_summary.csv"
SENS_CSV = ROOT / "plots" / "video" / "preset_sensitivity" / "classical_preset_summary.csv"
OUT = ROOT / "results" / "video" / "video_thesis_numbers.json"

ref = pd.read_csv(REF_CSV)
sens = pd.read_csv(SENS_CSV)

def row(df, codec_label):
    r = df[df["codec_label"] == codec_label]
    if len(r) != 1:
        raise ValueError(f"Expected one row for {codec_label}, got {len(r)}")
    return r.iloc[0]

def f(x, nd=3):
    if pd.isna(x):
        return None
    return round(float(x), nd)

numbers = {}

# ---------------------------------------------------------------------
# Reference-preset summary
# ---------------------------------------------------------------------

numbers["reference_preset"] = {}

for _, r in ref.iterrows():
    c = str(r["codec_label"])
    numbers["reference_preset"][c] = {
        "bitrate_mbps_mean": f(r["bitrate_mbps_mean"], 3),
        "bitrate_mbps_min": f(r["bitrate_mbps_min"], 3),
        "bitrate_mbps_max": f(r["bitrate_mbps_max"], 3),
        "psnr_y_mean_db": f(r["psnr_y_mean"], 3),
        "vmaf_mean": f(r["vmaf_mean"], 3),
        "energy_kj_mean": f(r["energy_kj_mean"], 4),
        "energy_kj_min": f(r["energy_kj_min"], 4),
        "energy_kj_max": f(r["energy_kj_max"], 4),
        "encode_energy_j_mean": f(r["encode_energy_j_mean"], 3),
        "decode_energy_j_mean": f(r["decode_energy_j_mean"], 3),
    }

# ---------------------------------------------------------------------
# Important comparisons
# ---------------------------------------------------------------------

svt = row(ref, "SVT-AV1 p5")
vvenc = row(ref, "VVenC medium")
rt = row(ref, "DCVC-RT")
rt_cuda = row(ref, "DCVC-RT-CUDA")
dc = row(ref, "DCVC-DC")
fm = row(ref, "DCVC-FM")
x264 = row(ref, "x264 medium")
x265 = row(ref, "x265 medium")

numbers["comparisons"] = {
    "dcvc_rt_cuda_energy_vs_dcvc_rt_ratio": f(
        rt_cuda["energy_kj_mean"] / rt["energy_kj_mean"], 4
    ),
    "dcvc_rt_cuda_energy_reduction_vs_dcvc_rt_percent": f(
        (1.0 - rt_cuda["energy_kj_mean"] / rt["energy_kj_mean"]) * 100.0, 2
    ),
    "dcvc_rt_cuda_vmaf_delta_vs_dcvc_rt": f(
        rt_cuda["vmaf_mean"] - rt["vmaf_mean"], 3
    ),
    "dcvc_rt_cuda_psnr_delta_vs_dcvc_rt_db": f(
        rt_cuda["psnr_y_mean"] - rt["psnr_y_mean"], 3
    ),
    "vvenc_energy_vs_svtav1_ratio": f(
        vvenc["energy_kj_mean"] / svt["energy_kj_mean"], 3
    ),
    "dcvc_dc_energy_vs_svtav1_ratio": f(
        dc["energy_kj_mean"] / svt["energy_kj_mean"], 3
    ),
    "dcvc_fm_energy_vs_svtav1_ratio": f(
        fm["energy_kj_mean"] / svt["energy_kj_mean"], 3
    ),
    "dcvc_rt_cuda_energy_vs_svtav1_ratio": f(
        rt_cuda["energy_kj_mean"] / svt["energy_kj_mean"], 3
    ),
    "svtav1_vmaf_advantage_vs_x265": f(
        svt["vmaf_mean"] - x265["vmaf_mean"], 3
    ),
    "vvenc_bitrate_vs_svtav1_ratio": f(
        vvenc["bitrate_mbps_mean"] / svt["bitrate_mbps_mean"], 3
    ),
    "vvenc_bitrate_reduction_vs_svtav1_percent": f(
        (1.0 - vvenc["bitrate_mbps_mean"] / svt["bitrate_mbps_mean"]) * 100.0, 2
    ),
}

# ---------------------------------------------------------------------
# Preset sensitivity
# ---------------------------------------------------------------------

numbers["preset_sensitivity"] = {}

for codec in sorted(sens["codec"].unique()):
    sub = sens[sens["codec"] == codec].copy()
    ref_row = sub[sub["is_reference_preset"] == True]

    if len(ref_row) != 1:
        raise ValueError(f"Missing reference row for {codec}")

    ref_row = ref_row.iloc[0]
    ref_e = float(ref_row["energy_kj_mean"])
    ref_vmaf = float(ref_row["vmaf_mean"])

    numbers["preset_sensitivity"][codec] = {}

    for _, r in sub.iterrows():
        preset = str(r["preset"])
        numbers["preset_sensitivity"][codec][preset] = {
            "is_reference": bool(r["is_reference_preset"]),
            "energy_kj_mean": f(r["energy_kj_mean"], 4),
            "vmaf_mean": f(r["vmaf_mean"], 3),
            "energy_ratio_vs_reference": f(r["energy_kj_mean"] / ref_e, 3),
            "vmaf_delta_vs_reference": f(r["vmaf_mean"] - ref_vmaf, 3),
        }

# ---------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------

numbers["ranges"] = {
    "reference_energy_kj_min": f(ref["energy_kj_mean"].min(), 4),
    "reference_energy_kj_max": f(ref["energy_kj_mean"].max(), 4),
    "reference_energy_dynamic_range_ratio": f(
        ref["energy_kj_mean"].max() / ref["energy_kj_mean"].min(), 2
    ),
    "reference_vmaf_min": f(ref["vmaf_mean"].min(), 3),
    "reference_vmaf_max": f(ref["vmaf_mean"].max(), 3),
    "reference_psnr_y_min_db": f(ref["psnr_y_mean"].min(), 3),
    "reference_psnr_y_max_db": f(ref["psnr_y_mean"].max(), 3),
    "reference_bitrate_mbps_min": f(ref["bitrate_mbps_mean"].min(), 3),
    "reference_bitrate_mbps_max": f(ref["bitrate_mbps_mean"].max(), 3),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

print("[OK] Saved:", OUT)
print()
print(json.dumps(numbers, indent=2))