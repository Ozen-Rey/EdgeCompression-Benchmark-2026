from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "results" / "video" / "master.csv"
OUT = ROOT / "plots" / "video" / "preset_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

CLASSICAL_CODECS = ["x264", "x265", "svtav1", "vvenc"]

CODEC_LABELS = {
    "x264": "x264",
    "x265": "x265",
    "svtav1": "SVT-AV1",
    "vvenc": "VVenC",
}

PRESET_ORDER = {
    "x264": ["faster", "medium", "slow"],
    "x265": ["faster", "medium", "slow"],
    "vvenc": ["faster", "medium", "slow"],
    "svtav1": ["p10", "p5", "p2"],
}

REFERENCE_PRESET = {
    "x264": "medium",
    "x265": "medium",
    "vvenc": "medium",
    "svtav1": "p5",
}

PRESET_MARKERS = {
    "faster": "o",
    "medium": "s",
    "slow": "^",
    "p10": "o",
    "p5": "s",
    "p2": "^",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def parse_param_value(param: str) -> float:
    """
    Extracts numeric value from strings such as:
        crf=22
        qp=32
        p=5
    Used only for stable sorting of operating points.
    """
    m = re.search(r"[-+]?\d*\.?\d+", str(param))
    if m is None:
        return float("inf")
    return float(m.group(0))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN)

    df = df[
        (df["profile"] == "LDP")
        & (df["codec"].isin(CLASSICAL_CODECS))
    ].copy()

    numeric_cols = [
        "actual_mbps",
        "vmaf_mean",
        "psnr_y",
        "energy_total_j",
        "energy_total_kj",
        "energy_encode_total_j",
        "energy_decode_total_j",
        "energy_per_frame_j",
        "time_encode_s",
        "time_decode_s",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["codec_label"] = df["codec"].map(CODEC_LABELS)
    df["is_reference_preset"] = df.apply(
        lambda r: str(r["preset"]) == REFERENCE_PRESET.get(str(r["codec"]), ""),
        axis=1,
    )

    return df


def average_operating_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average over UVG sequences for each codec/preset/operating point.
    """

    return (
        df.groupby(["codec", "codec_label", "preset", "param"], as_index=False)
        .agg(
            actual_mbps=("actual_mbps", "mean"),
            psnr_y=("psnr_y", "mean"),
            vmaf_mean=("vmaf_mean", "mean"),
            energy_total_kj=("energy_total_kj", "mean"),
            energy_total_j=("energy_total_j", "mean"),
            energy_encode_total_j=("energy_encode_total_j", "mean"),
            energy_decode_total_j=("energy_decode_total_j", "mean"),
            energy_per_frame_j=("energy_per_frame_j", "mean"),
            time_encode_s=("time_encode_s", "mean"),
            time_decode_s=("time_decode_s", "mean"),
        )
    )


def make_summary(df_avg: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_avg.groupby(["codec", "codec_label", "preset"], as_index=False)
        .agg(
            bitrate_mbps_mean=("actual_mbps", "mean"),
            bitrate_mbps_min=("actual_mbps", "min"),
            bitrate_mbps_max=("actual_mbps", "max"),
            psnr_y_mean=("psnr_y", "mean"),
            vmaf_mean=("vmaf_mean", "mean"),
            energy_kj_mean=("energy_total_kj", "mean"),
            energy_kj_min=("energy_total_kj", "min"),
            energy_kj_max=("energy_total_kj", "max"),
            encode_energy_j_mean=("energy_encode_total_j", "mean"),
            decode_energy_j_mean=("energy_decode_total_j", "mean"),
            encode_time_s_mean=("time_encode_s", "mean"),
            decode_time_s_mean=("time_decode_s", "mean"),
        )
    )

    rows = []

    for codec in CLASSICAL_CODECS:
        sub = summary[summary["codec"] == codec].copy()
        order = {p: i for i, p in enumerate(PRESET_ORDER[codec])}
        sub["preset_order"] = sub["preset"].map(order)
        sub["is_reference_preset"] = sub["preset"].eq(REFERENCE_PRESET[codec])
        rows.append(sub.sort_values("preset_order"))

    out = pd.concat(rows, ignore_index=True)
    out = out.drop(columns=["preset_order"])

    out.to_csv(OUT / "classical_preset_summary.csv", index=False)
    return out


def plot_energy_spread(summary: pd.DataFrame) -> None:
    """
    Grouped bar chart:
    mean operational energy by codec and preset.
    """

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    x = np.arange(len(CLASSICAL_CODECS))
    width = 0.22

    # Three slots per codec.
    for slot in range(3):
        values = []
        labels = []

        for codec in CLASSICAL_CODECS:
            presets = PRESET_ORDER[codec]
            preset = presets[slot]

            row = summary[
                (summary["codec"] == codec)
                & (summary["preset"] == preset)
            ]

            if len(row) == 0:
                values.append(np.nan)
            else:
                values.append(float(row["energy_kj_mean"].iloc[0]))

            labels.append(preset)

        offset = (slot - 1) * width

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label="slot " + str(slot),
        )

        for bar, codec in zip(bars, CLASSICAL_CODECS):
            preset = PRESET_ORDER[codec][slot]

            if preset == REFERENCE_PRESET[codec]:
                bar.set_hatch("//")
                bar.set_linewidth(1.2)
                bar.set_edgecolor("black")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CODEC_LABELS[c] for c in CLASSICAL_CODECS])
    ax.set_ylabel("Mean operational energy [kJ/sequence]")
    ax.set_title("Classical preset sensitivity: operational energy")

    # Custom legend because slots mean different names for SVT-AV1 vs others.
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="C0", label="faster / p10"),
        Patch(facecolor="C1", hatch="//", edgecolor="black", label="medium / p5 (reference)"),
        Patch(facecolor="C2", label="slow / p2"),
    ]

    ax.legend(handles=handles, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / "classical_preset_energy_spread.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rd_spread(df_avg: pd.DataFrame) -> None:
    """
    Four-panel R-D sensitivity plot.
    Each panel is one classical codec; lines are presets.
    """

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    axes = axes.flatten()

    for ax, codec in zip(axes, CLASSICAL_CODECS):
        codec_df = df_avg[df_avg["codec"] == codec].copy()

        for preset in PRESET_ORDER[codec]:
            g = codec_df[codec_df["preset"] == preset].copy()

            if len(g) == 0:
                continue

            g["param_sort"] = g["param"].map(parse_param_value)
            g = g.sort_values("actual_mbps")

            label = preset
            if preset == REFERENCE_PRESET[codec]:
                label = f"{preset} (ref)"

            ax.plot(
                g["actual_mbps"],
                g["vmaf_mean"],
                marker=PRESET_MARKERS.get(preset, "o"),
                linewidth=1.8,
                markersize=5,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_title(CODEC_LABELS[codec])
        ax.set_xlabel("Bitrate [Mbps]")
        ax.set_ylabel("VMAF")
        ax.legend(frameon=True)

    fig.suptitle("Classical preset sensitivity: VMAF--bitrate behavior", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "classical_preset_rd_spread_vmaf.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_vmaf_energy_tradeoff(summary: pd.DataFrame) -> None:
    """
    Preset-level mean VMAF vs mean energy.
    Highlights the reference preset.
    """

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for codec in CLASSICAL_CODECS:
        g = summary[summary["codec"] == codec].copy()
        order = {p: i for i, p in enumerate(PRESET_ORDER[codec])}
        g["preset_order"] = g["preset"].map(order)
        g = g.sort_values("preset_order")

        ax.plot(
            g["energy_kj_mean"],
            g["vmaf_mean"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=CODEC_LABELS[codec],
        )

        for _, r in g.iterrows():
            preset = str(r["preset"])
            text = preset

            if preset == REFERENCE_PRESET[codec]:
                text = preset + " ref"

            ax.annotate(
                text,
                (r["energy_kj_mean"], r["vmaf_mean"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Mean operational energy [kJ/sequence]")
    ax.set_ylabel("Mean VMAF")
    ax.set_title("Classical preset sensitivity: quality--energy trade-off")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(
        OUT / "classical_preset_vmaf_energy_tradeoff.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_text_report(summary: pd.DataFrame) -> None:
    lines = []
    lines.append("Classical preset sensitivity report")
    lines.append("=" * 70)
    lines.append("Reference presets:")
    lines.append("  x264   -> medium")
    lines.append("  x265   -> medium")
    lines.append("  SVT-AV1 -> p5")
    lines.append("  VVenC  -> medium")
    lines.append("")

    for codec in CLASSICAL_CODECS:
        g = summary[summary["codec"] == codec].copy()
        ref = g[g["preset"] == REFERENCE_PRESET[codec]]

        if len(ref) == 0:
            continue

        ref_e = float(ref["energy_kj_mean"].iloc[0])
        ref_vmaf = float(ref["vmaf_mean"].iloc[0])

        lines.append(f"{CODEC_LABELS[codec]}")
        lines.append("-" * 70)

        for _, r in g.iterrows():
            preset = str(r["preset"])
            e = float(r["energy_kj_mean"])
            v = float(r["vmaf_mean"])

            e_ratio = e / ref_e if ref_e > 0 else np.nan
            dvmaf = v - ref_vmaf

            tag = " [reference]" if preset == REFERENCE_PRESET[codec] else ""

            lines.append(
                f"  {preset:<8}{tag:<13} "
                f"E={e:8.4f} kJ/seq  "
                f"E/ref={e_ratio:7.3f}  "
                f"VMAF={v:8.3f}  "
                f"ΔVMAF={dvmaf:+7.3f}"
            )

        lines.append("")

    (OUT / "classical_preset_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    setup_style()

    df = load_data()
    df_avg = average_operating_points(df)
    summary = make_summary(df_avg)

    plot_energy_spread(summary)
    plot_rd_spread(df_avg)
    plot_vmaf_energy_tradeoff(summary)
    make_text_report(summary)

    print("=" * 70)
    print("[OK] Video preset sensitivity generated")
    print("Input:", IN)
    print("Output:", OUT)
    print()

    for p in sorted(OUT.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()