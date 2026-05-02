from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "results" / "video" / "master.csv"
OUT = ROOT / "results" / "video" / "video_LDP_reference_preset_paper_ready.csv"

REFERENCE_PRESETS = {
    "x264": "medium",
    "x265": "medium",
    "vvenc": "medium",
    "svtav1": "p5",
    "dcvc_dc": None,
    "dcvc_fm": None,
    "dcvc_rt": None,
    "dcvc_rt_cuda": None,
}

LABELS = {
    "x264": "x264 medium",
    "x265": "x265 medium",
    "vvenc": "VVenC medium",
    "svtav1": "SVT-AV1 p5",
    "dcvc_dc": "DCVC-DC",
    "dcvc_fm": "DCVC-FM",
    "dcvc_rt": "DCVC-RT",
    "dcvc_rt_cuda": "DCVC-RT-CUDA",
}

df = pd.read_csv(IN)

df = df[df["profile"] == "LDP"].copy()

rows = []

for codec, preset in REFERENCE_PRESETS.items():
    sub = df[df["codec"] == codec].copy()

    if preset is not None:
        if "preset" not in sub.columns:
            raise RuntimeError(f"Missing preset column for {codec}")
        sub = sub[sub["preset"].astype(str) == preset].copy()

    if len(sub) == 0:
        print(f"[WARN] No rows for {codec} preset={preset}")
        continue

    sub["codec_label"] = LABELS[codec]
    sub["reference_preset"] = "none" if preset is None else preset
    rows.append(sub)

out = pd.concat(rows, ignore_index=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("[OK] Saved:", OUT)
print("Rows:", len(out))
print()
print(out.groupby(["codec", "codec_label", "reference_preset"]).size())
print()
print("Sequences:", sorted(out["seq"].unique()))
print("Profiles:", sorted(out["profile"].unique()))