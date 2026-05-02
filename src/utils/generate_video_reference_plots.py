from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "results" / "video" / "video_LDP_reference_preset_paper_ready.csv"
OUT = ROOT / "plots" / "video" / "reference_preset"
OUT.mkdir(parents=True, exist_ok=True)

CODEC_ORDER = [
    "x264 medium",
    "x265 medium",
    "SVT-AV1 p5",
    "VVenC medium",
    "DCVC-DC",
    "DCVC-FM",
    "DCVC-RT",
    "DCVC-RT-CUDA",
]

MARKERS = {
    "x264 medium": "o",
    "x265 medium": "s",
    "SVT-AV1 p5": "^",
    "VVenC medium": "D",
    "DCVC-DC": "P",
    "DCVC-FM": "X",
    "DCVC-RT": "v",
    "DCVC-RT-CUDA": "*",
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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN)

    # Main paper protocol: LDP reference-preset configuration.
    df = df[df["profile"] == "LDP"].copy()

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

    return df


def average_rd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average over the 7 UVG sequences for each codec and operating point.
    """

    return (
        df.groupby(["codec_label", "param"], as_index=False)
        .agg(
            actual_mbps=("actual_mbps", "mean"),
            psnr_y=("psnr_y", "mean"),
            vmaf_mean=("vmaf_mean", "mean"),
            energy_total_kj=("energy_total_kj", "mean"),
            energy_total_j=("energy_total_j", "mean"),
            energy_encode_total_j=("energy_encode_total_j", "mean"),
            energy_decode_total_j=("energy_decode_total_j", "mean"),
            energy_per_frame_j=("energy_per_frame_j", "mean"),
        )
        .sort_values(["codec_label", "actual_mbps"])
    )


def plot_rd(
    df_avg: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for codec in CODEC_ORDER:
        g = df_avg[df_avg["codec_label"] == codec].sort_values("actual_mbps")

        if len(g) == 0:
            continue

        ax.plot(
            g["actual_mbps"],
            g[y_col],
            marker=MARKERS.get(codec, "o"),
            linewidth=1.8,
            markersize=5,
            label=codec,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate [Mbps]")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)


def plot_energy_bar(df_avg: pd.DataFrame) -> None:
    summary = (
        df_avg.groupby("codec_label", as_index=False)
        .agg(energy_total_kj=("energy_total_kj", "mean"))
    )

    order = {codec: idx for idx, codec in enumerate(CODEC_ORDER)}
    summary["order"] = summary["codec_label"].map(order)
    summary = summary.sort_values("order")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    ax.bar(summary["codec_label"], summary["energy_total_kj"])
    ax.set_yscale("log")
    ax.set_ylabel("Mean operational energy [kJ/sequence]")
    ax.set_title("Operational energy, LDP reference presets")
    ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(OUT / "operational_energy_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rde_bubble(df_avg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    energy = df_avg["energy_total_kj"].to_numpy()
    e_min = np.nanmin(energy)
    e_max = np.nanmax(energy)

    sizes = 40 + 260 * (energy - e_min) / max(e_max - e_min, 1e-9)

    for codec in CODEC_ORDER:
        g = df_avg[df_avg["codec_label"] == codec].sort_values("actual_mbps")

        if len(g) == 0:
            continue

        idx = g.index
        ax.scatter(
            g["actual_mbps"],
            g["vmaf_mean"],
            s=sizes[df_avg.index.get_indexer(idx)],
            marker=MARKERS.get(codec, "o"),
            alpha=0.75,
            label=codec,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate [Mbps]")
    ax.set_ylabel("VMAF")
    ax.set_title("Rate--distortion--energy operating space")
    ax.legend(ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / "rde_bubble_vmaf_energy.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_energy_components_by_protocol(df_avg: pd.DataFrame) -> None:
    """
    Stacked energy components.

    Classical codecs and DCVC-RT-CUDA have encode/decode measurements.
    DCVC-DC, DCVC-FM, and non-CUDA DCVC-RT are forward/inference protocol
    measurements, so decode is unavailable and shown as zero only for plotting.
    """

    summary = (
        df_avg.groupby("codec_label", as_index=False)
        .agg(
            encode_j=("energy_encode_total_j", "mean"),
            decode_j=("energy_decode_total_j", "mean"),
        )
    )

    # Plotting-only handling: unavailable decode components are not zero
    # conceptually, but are displayed as absent components.
    summary["encode_j"] = summary["encode_j"].fillna(0.0)
    summary["decode_j"] = summary["decode_j"].fillna(0.0)

    order = {codec: idx for idx, codec in enumerate(CODEC_ORDER)}
    summary["order"] = summary["codec_label"].map(order)
    summary = summary.sort_values("order")

    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    encode_kj = summary["encode_j"] / 1000.0
    decode_kj = summary["decode_j"] / 1000.0

    ax.bar(x, encode_kj, label="Encode / forward")
    ax.bar(x, decode_kj, bottom=encode_kj, label="Decode")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["codec_label"], rotation=35, ha="right")
    ax.set_ylabel("Mean operational energy [kJ/sequence]")
    ax.set_title("Energy components by measurement protocol")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / "energy_components_by_protocol.pdf", bbox_inches="tight")
    plt.close(fig)


def bd_rate_curve(
    baseline: pd.DataFrame,
    target: pd.DataFrame,
    q_col: str = "vmaf_mean",
) -> float:
    """
    BD-rate of target relative to baseline.

    Negative means target needs less bitrate than baseline at equal quality.
    """

    a = baseline[["actual_mbps", q_col]].dropna().sort_values(q_col)
    b = target[["actual_mbps", q_col]].dropna().sort_values(q_col)

    if len(a) < 4 or len(b) < 4:
        return float("nan")

    qa = a[q_col].to_numpy()
    qb = b[q_col].to_numpy()
    la = np.log10(a["actual_mbps"].to_numpy())
    lb = np.log10(b["actual_mbps"].to_numpy())

    qa, idx_a = np.unique(qa, return_index=True)
    qb, idx_b = np.unique(qb, return_index=True)
    la = la[idx_a]
    lb = lb[idx_b]

    if len(qa) < 4 or len(qb) < 4:
        return float("nan")

    q_min = max(np.min(qa), np.min(qb))
    q_max = min(np.max(qa), np.max(qb))

    if q_max <= q_min:
        return float("nan")

    overlap = q_max - q_min
    min_range = min(np.max(qa) - np.min(qa), np.max(qb) - np.min(qb))

    if min_range <= 0 or overlap < 0.20 * min_range:
        return float("nan")

    fa = interp1d(qa, la, kind="cubic", fill_value="extrapolate")
    fb = interp1d(qb, lb, kind="cubic", fill_value="extrapolate")

    q_grid = np.linspace(q_min, q_max, 200)
    delta_log_rate = np.mean(fb(q_grid) - fa(q_grid))

    return (10.0**delta_log_rate - 1.0) * 100.0


def normalized_points(
    g: pd.DataFrame,
    bounds: dict[str, np.ndarray],
    q_col: str = "vmaf_mean",
) -> np.ndarray:
    """
    Lower-is-better normalized representation:
        x1 = log bitrate
        x2 = negative quality
        x3 = log energy
    """

    r = np.log10(g["actual_mbps"].to_numpy())
    q = -g[q_col].to_numpy()
    e = np.log10(g["energy_total_kj"].to_numpy())

    pts = np.column_stack([r, q, e])

    return (pts - bounds["min"]) / bounds["range"]


def additive_epsilon_indicator(a: np.ndarray, b: np.ndarray) -> float:
    """
    Additive epsilon indicator I_eps(A, B).

    Minimum additive epsilon such that A weakly dominates B.
    Lower is better.
    """

    eps = -np.inf

    for pb in b:
        best_for_b = np.inf

        for pa in a:
            best_for_b = min(best_for_b, np.max(pa - pb))

        eps = max(eps, best_for_b)

    return float(eps)


def plot_heatmap(
    mat: np.ndarray,
    title: str,
    cbar_label: str,
    out_name: str,
    fmt: str,
) -> None:
    labels = CODEC_ORDER

    fig, ax = plt.subplots(figsize=(8.2, 7.2))

    is_bd = "BD-rate" in title
    is_epsilon = "ε-indicator" in title or "epsilon" in title.lower()

    if is_epsilon:
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)

        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax == vmin:
            vmax = vmin + 1.0

        im = ax.imshow(
            np.ma.masked_invalid(mat),
            vmin=vmin,
            vmax=vmax,
            cmap="YlOrRd",
        )

        text_threshold = vmin + 0.60 * (vmax - vmin)

    else:
        if is_bd:
            # Keep real values in the cells, but clip the color scale so
            # extreme BD-rate values do not destroy readability.
            max_abs = 100.0
        else:
            max_abs = np.nanmax(np.abs(mat))
            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1.0

        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            norm=norm,
            cmap="RdYlGn_r",
        )

        text_threshold = 0.5 * max_abs

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Target codec")
    ax.set_ylabel("Baseline codec")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j or not np.isfinite(mat[i, j]):
                continue

            v = mat[i, j]

            if is_epsilon:
                color = "white" if v > text_threshold else "black"
            else:
                color = "white" if abs(v) > text_threshold else "black"

            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)


def make_heatmaps(df_avg: pd.DataFrame) -> None:
    labels = CODEC_ORDER
    n = len(labels)

    # -------------------------------------------------------------------------
    # BD-rate heatmap on VMAF.
    # -------------------------------------------------------------------------

    bd = np.full((n, n), np.nan)

    for i, baseline in enumerate(labels):
        g_baseline = df_avg[df_avg["codec_label"] == baseline]

        for j, target in enumerate(labels):
            if baseline == target:
                continue

            g_target = df_avg[df_avg["codec_label"] == target]
            bd[i, j] = bd_rate_curve(g_baseline, g_target, q_col="vmaf_mean")

    plot_heatmap(
        bd,
        "BD-rate heatmap, VMAF, LDP reference presets",
        "BD-rate of target vs baseline [%]",
        "heatmap_bd_rate_vmaf.pdf",
        "{:+.0f}",
    )

    # -------------------------------------------------------------------------
    # Additive epsilon indicator on normalized R-D-E points.
    # -------------------------------------------------------------------------

    all_log_r = np.log10(df_avg["actual_mbps"].to_numpy())
    all_neg_q = -df_avg["vmaf_mean"].to_numpy()
    all_log_e = np.log10(df_avg["energy_total_kj"].to_numpy())

    all_pts = np.column_stack([all_log_r, all_neg_q, all_log_e])

    mn = np.nanmin(all_pts, axis=0)
    mx = np.nanmax(all_pts, axis=0)
    rg = np.where(mx - mn == 0, 1.0, mx - mn)

    bounds = {"min": mn, "range": rg}

    eps = np.full((n, n), np.nan)

    for i, baseline in enumerate(labels):
        g_baseline = df_avg[df_avg["codec_label"] == baseline]
        pts_baseline = normalized_points(g_baseline, bounds)

        for j, target in enumerate(labels):
            if baseline == target:
                continue

            g_target = df_avg[df_avg["codec_label"] == target]
            pts_target = normalized_points(g_target, bounds)

            # Target relative to baseline.
            eps[i, j] = additive_epsilon_indicator(pts_target, pts_baseline)

    plot_heatmap(
        eps,
        "Additive ε-indicator heatmap, R-D-E, LDP reference presets",
        "ε(target, baseline), normalized",
        "heatmap_epsilon_rde.pdf",
        "{:+.2f}",
    )


def make_summary_csv(df_avg: pd.DataFrame) -> None:
    summary = (
        df_avg.groupby("codec_label", as_index=False)
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
        )
    )

    order = {codec: idx for idx, codec in enumerate(CODEC_ORDER)}
    summary["order"] = summary["codec_label"].map(order)
    summary = summary.sort_values("order").drop(columns=["order"])

    summary.to_csv(OUT / "video_reference_summary.csv", index=False)


def main() -> None:
    setup_style()

    df = load_data()
    df_avg = average_rd(df)

    make_summary_csv(df_avg)

    plot_rd(
        df_avg,
        y_col="psnr_y",
        y_label="PSNR-Y [dB]",
        title="Rate--distortion, PSNR-Y, LDP reference presets",
        out_name="rd_psnr_y.pdf",
    )

    plot_rd(
        df_avg,
        y_col="vmaf_mean",
        y_label="VMAF",
        title="Rate--distortion, VMAF, LDP reference presets",
        out_name="rd_vmaf.pdf",
    )

    plot_energy_bar(df_avg)
    plot_rde_bubble(df_avg)
    plot_energy_components_by_protocol(df_avg)
    make_heatmaps(df_avg)

    print("=" * 70)
    print("[OK] Video reference plots generated")
    print("Input:", IN)
    print("Output:", OUT)
    print()

    for p in sorted(OUT.glob("*.pdf")):
        print(" -", p.name)

    print(" - video_reference_summary.csv")


if __name__ == "__main__":
    main()