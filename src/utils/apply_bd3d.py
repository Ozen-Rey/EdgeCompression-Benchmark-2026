"""
Applica le 5 metriche BD-3D a tutti i CSV del benchmark video espanso (16 varianti).
Genera heatmap 16x16 paper-grade per ciascuna metrica.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path.home() / "tesi" / "src" / "utils"))
from bd3d_metric import compute_bd_matrix

RESULTS = Path.home() / "tesi" / "results" / "video"
PLOTS = Path.home() / "tesi" / "plots" / "video" / "bd_metrics"
PLOTS.mkdir(parents=True, exist_ok=True)

# File originali da leggere
BASE_CODECS = [
    "x264",
    "x265",
    "svtav1",
    "vvenc",
    "dcvc_dc",
    "dcvc_fm",
    "dcvc_rt",
    "dcvc_rt_cuda",
]

# Nuova lista espansa per l'asse delle heatmap
ALL_VARIANTS = [
    "x264-faster",
    "x264-medium",
    "x264-slow",
    "x265-faster",
    "x265-medium",
    "x265-slow",
    "svtav1-p10",
    "svtav1-p5",
    "svtav1-p2",
    "vvenc-faster",
    "vvenc-medium",
    "vvenc-slow",
    "dcvc_dc",
    "dcvc_fm",
    "dcvc_rt",
    "dcvc_rt_cuda",
]

LABELS = {
    "x264-faster": "x264 F",
    "x264-medium": "x264 M",
    "x264-slow": "x264 S",
    "x265-faster": "x265 F",
    "x265-medium": "x265 M",
    "x265-slow": "x265 S",
    "svtav1-p10": "SVT p10",
    "svtav1-p5": "SVT p5",
    "svtav1-p2": "SVT p2",
    "vvenc-faster": "VVC F",
    "vvenc-medium": "VVC M",
    "vvenc-slow": "VVC S",
    "dcvc_dc": "DCVC-DC",
    "dcvc_fm": "DCVC-FM",
    "dcvc_rt": "DCVC-RT",
    "dcvc_rt_cuda": "DCVC-RT C",
}

DISTORTION_METRIC = "vmaf_mean"
ND_COLOR = "#e0e0e0"


def load_combined():
    energy_dfs, quality_dfs = [], []
    for codec in BASE_CODECS:
        e = RESULTS / f"{codec}_energy.csv"
        q = RESULTS / f"{codec}_quality.csv"
        if e.exists() and q.exists():
            energy_dfs.append(pd.read_csv(e))
            quality_dfs.append(pd.read_csv(q))

    pd.concat(energy_dfs).to_csv("/tmp/all_e.csv", index=False)
    pd.concat(quality_dfs).to_csv("/tmp/all_q.csv", index=False)
    return "/tmp/all_e.csv", "/tmp/all_q.csv"


def aggregate_matrix(bd_full):
    agg_funcs = {
        col: "mean"
        for col in ["bd_rate", "bd_rate_e", "bd_volume", "pareto_hv", "epsilon"]
    }
    summary = bd_full.groupby(["codec_a", "codec_b"]).agg(agg_funcs).reset_index()
    for metric in ["bd_rate", "bd_rate_e", "bd_volume", "pareto_hv", "epsilon"]:
        valid_col = f"{metric}_valid"
        if valid_col in bd_full.columns:
            n_valid = (
                bd_full.groupby(["codec_a", "codec_b"])[valid_col].sum().reset_index()
            )
            n_valid.columns = ["codec_a", "codec_b", f"{metric}_n_valid"]
            summary = pd.merge(summary, n_valid, on=["codec_a", "codec_b"])
    return summary


def make_matrix(summary, metric_col, variants=ALL_VARIANTS):
    n = len(variants)
    mat = np.full((n, n), np.nan)
    for i, ca in enumerate(variants):
        for j, cb in enumerate(variants):
            if ca == cb:
                continue
            row = summary[(summary["codec_a"] == ca) & (summary["codec_b"] == cb)]
            if len(row) > 0:
                mat[i, j] = row[metric_col].values[0]
    return mat


def plot_heatmap(
    mat, variants, title, value_label, save_path, fmt="{:+.1f}", cmap="RdYlGn_r"
):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(ND_COLOR)

    abs_max = np.nanmax(np.abs(mat))
    if abs_max == 0 or np.isnan(abs_max):
        abs_max = 1.0
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    mat_masked = np.ma.masked_invalid(mat)
    im = ax.imshow(mat_masked, cmap=cmap, norm=norm, aspect="auto")

    n = len(variants)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(
        [str(LABELS.get(c, str(c))) for c in variants],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticklabels([str(LABELS.get(c, str(c))) for c in variants], fontsize=9)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            elif np.isnan(mat[i, j]):
                pass
            else:
                v = mat[i, j]
                color = "white" if abs(v) > 0.5 * abs_max else "black"
                ax.text(
                    j,
                    i,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                    fontweight="bold",
                )

    ax.set_xlabel("Target Codec (B)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Baseline Codec (A)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_label, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.close()


def plot_panel(summary, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    metrics = [
        ("bd_rate", "BD-rate (1D)", "BD-rate (%)", "{:+.0f}", "RdYlGn_r"),
        ("bd_rate_e", "BD-rate-E (2D)", "BD-rate-E (%)", "{:+.0f}", "RdYlGn_r"),
        ("bd_volume", "BD-volume (3D)", "BD-volume (%)", "{:+.0f}", "RdYlGn_r"),
        ("pareto_hv", "Pareto-HV", "Δ HV", "{:+.2f}", "RdYlGn_r"),
        ("epsilon", "ε-indicator", "ε", "{:+.2f}", "RdYlGn_r"),
    ]

    for idx, (col, title, vlab, fmt, cmap) in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor(ND_COLOR)
        mat = make_matrix(summary, col)
        amax = np.nanmax(np.abs(mat))
        norm = TwoSlopeNorm(
            vmin=-amax if amax > 0 else -1, vcenter=0, vmax=amax if amax > 0 else 1
        )
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect="auto")

        n = len(ALL_VARIANTS)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(
            [LABELS.get(c, c) for c in ALL_VARIANTS],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticklabels([LABELS.get(c, c) for c in ALL_VARIANTS], fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(mat[i, j]):
                    v = mat[i, j]
                    ax.text(
                        j,
                        i,
                        fmt.format(v),
                        ha="center",
                        va="center",
                        color="white" if abs(v) > 0.5 * amax else "black",
                        fontsize=6,
                    )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(vlab, fontsize=9)

    ax = axes[5]
    ax.set_facecolor("white")
    n = len(ALL_VARIANTS)
    coverage = np.zeros((n, n), dtype=int)
    for col, _, _, _, _ in metrics:
        if f"{col}_n_valid" in summary.columns:
            mat_n = make_matrix(summary, f"{col}_n_valid")
            coverage += np.where(np.isnan(mat_n), 0, mat_n).astype(int)

    im = ax.imshow(coverage, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(
        [LABELS.get(c, c) for c in ALL_VARIANTS], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticklabels([LABELS.get(c, c) for c in ALL_VARIANTS], fontsize=8)
    ax.set_title("Coverage", fontsize=12, fontweight="bold")

    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{coverage[i,j]}",
                    ha="center",
                    va="center",
                    color="white" if coverage[i, j] > coverage.max() * 0.5 else "black",
                    fontsize=6,
                )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", format="pdf")
    plt.close()


def main():
    print("=" * 70)
    print(f"BD-3D MATRIX ANALYSIS (16x16) — distortion={DISTORTION_METRIC}")
    print("=" * 70)

    tmp_e, tmp_q = load_combined()  # type: ignore
    print(
        "Computing pairwise BD matrix for 16 variants (this may take up to 10 mins)..."
    )
    bd_full = compute_bd_matrix(tmp_e, tmp_q, distortion_metric=DISTORTION_METRIC)

    bd_full.to_csv(PLOTS / "bd_matrix_full.csv", index=False)
    summary = aggregate_matrix(bd_full)
    summary.to_csv(PLOTS / "bd_matrix_summary.csv", index=False)

    print("\nGenerating 16x16 paper-grade heatmaps...")
    plot_heatmap(
        make_matrix(summary, "bd_rate"),
        ALL_VARIANTS,
        "BD-rate (1D)",
        "BD-rate (%)",
        PLOTS / "heatmap_bd_rate.pdf",
    )
    plot_heatmap(
        make_matrix(summary, "bd_rate_e"),
        ALL_VARIANTS,
        "BD-rate-E (2D)",
        "BD-rate-E (%)",
        PLOTS / "heatmap_bd_rate_e.pdf",
    )
    plot_heatmap(
        make_matrix(summary, "bd_volume"),
        ALL_VARIANTS,
        "BD-volume (3D)",
        "BD-volume (%)",
        PLOTS / "heatmap_bd_volume.pdf",
    )
    plot_heatmap(
        make_matrix(summary, "pareto_hv"),
        ALL_VARIANTS,
        "Pareto-HV",
        "Δ HV",
        PLOTS / "heatmap_pareto_hv.pdf",
        fmt="{:+.2f}",
    )
    plot_heatmap(
        make_matrix(summary, "epsilon"),
        ALL_VARIANTS,
        "ε-indicator",
        "ε",
        PLOTS / "heatmap_epsilon.pdf",
        fmt="{:+.2f}",
    )

    plot_panel(summary, PLOTS / "heatmap_panel.pdf")
    print(f"\nAll plots saved to: {PLOTS}\nDONE.")


if __name__ == "__main__":
    main()
