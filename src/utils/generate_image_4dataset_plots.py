from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

IN_CSV = ROOT / "results" / "images" / "image_4dataset_RDE_paper_ready.csv"
OUT_DIR = ROOT / "plots" / "images" / "four_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


CODEC_ORDER = [
    "JPEG",
    "JXL",
    "HEVC",
    "JPEG_AI",
    "Ballé",
    "Cheng",
    "ELIC",
    "TCM",
    "DCAE",
]

MARKERS = {
    "JPEG": "o",
    "JXL": "s",
    "HEVC": "^",
    "JPEG_AI": "D",
    "Ballé": "P",
    "Cheng": "X",
    "ELIC": "v",
    "TCM": "*",
    "DCAE": "h",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)

    required = {
        "eval_dataset",
        "codec",
        "param",
        "image",
        "bpp",
        "psnr",
        "ssimulacra2",
        "energy_per_image_j",
        "energy_cpu_per_image_j",
        "energy_gpu_per_image_j",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["codec"] = df["codec"].astype(str)
    df["param"] = df["param"].astype(str)
    df["eval_dataset"] = df["eval_dataset"].astype(str)

    return df


def operating_points(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bpp",
        "psnr",
        "ssimulacra2",
        "energy_per_image_j",
        "energy_cpu_per_image_j",
        "energy_gpu_per_image_j",
    ]

    op = (
        df.groupby(["codec", "param"], as_index=False)[cols]
        .mean(numeric_only=True)
        .copy()
    )

    op["codec_order"] = op["codec"].map(
        {codec: i for i, codec in enumerate(CODEC_ORDER)}
    ).fillna(999)

    return op.sort_values(["codec_order", "bpp"])


def savefig(name: str):
    out = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_rd(df_op: pd.DataFrame, metric: str, ylabel: str, title: str, out_name: str):
    fig, ax = plt.subplots(figsize=(10.8, 6.6))

    for codec in CODEC_ORDER:
        sub = df_op[df_op["codec"] == codec].sort_values("bpp")
        if sub.empty:
            continue

        ax.plot(
            sub["bpp"],
            sub[metric],
            marker=MARKERS.get(codec, "o"),
            linewidth=2.2,
            markersize=7,
            label=codec,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate [bpp]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=3, frameon=True)
    savefig(out_name)


def plot_energy_bar(df_op: pd.DataFrame):
    summary = (
        df_op.groupby("codec", as_index=False)
        .agg(
            energy_mean=("energy_per_image_j", "mean"),
            energy_min=("energy_per_image_j", "min"),
            energy_max=("energy_per_image_j", "max"),
        )
        .copy()
    )

    summary["codec_order"] = summary["codec"].map(
        {codec: i for i, codec in enumerate(CODEC_ORDER)}
    ).fillna(999)
    summary = summary.sort_values("codec_order")

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    x = np.arange(len(summary))

    ax.bar(x, summary["energy_mean"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["codec"], rotation=35, ha="right")
    ax.set_ylabel("Mean operational energy [J/image]")
    ax.set_title("Operational energy, 4-dataset image benchmark")

    savefig("image_operational_energy_4dataset.pdf")


def plot_energy_components(df_op: pd.DataFrame):
    summary = (
        df_op.groupby("codec", as_index=False)
        .agg(
            cpu=("energy_cpu_per_image_j", "mean"),
            gpu=("energy_gpu_per_image_j", "mean"),
            total=("energy_per_image_j", "mean"),
        )
        .copy()
    )

    summary["codec_order"] = summary["codec"].map(
        {codec: i for i, codec in enumerate(CODEC_ORDER)}
    ).fillna(999)
    summary = summary.sort_values("codec_order")

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    x = np.arange(len(summary))
    width = 0.4

    # Grouped (side-by-side) bars instead of stacked: stacking is invalid
    # under a logarithmic y-axis because log(cpu+gpu) != log(cpu)+log(gpu),
    # so the visual height of the stacked bar would not represent the sum.
    # CPU-only codecs (gpu == 0) are drawn at a small visible floor to avoid
    # the log(0) gap that would otherwise hide the zero contribution.
    gpu_plot = summary["gpu"].clip(lower=1e-3)
    ax.bar(x - width / 2, summary["cpu"], width=width, label="CPU")
    ax.bar(x + width / 2, gpu_plot, width=width, label="GPU")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["codec"], rotation=35, ha="right")
    ax.set_ylabel("Mean operational energy [J/image]")
    ax.set_title("Energy components, 4-dataset image benchmark")
    ax.legend(frameon=True)

    savefig("image_energy_components_4dataset.pdf")


_BUBBLE_SIZE_MIN = 30.0
_BUBBLE_SIZE_MAX = 1600.0


def _bubble_size_log(energy, e_min, e_max):
    """Map energy → marker area (points^2) on a log scale.

    The wider [30, 1600] range (compared to the previous [70, 620]) gives
    the eye a perceptible √(1600/30) ≈ 7× radius dynamic, which is the
    largest the page width can host while keeping the smallest bubbles
    legible. Lookups outside [e_min, e_max] are clipped.
    """
    energy = np.asarray(energy, dtype=float)
    denom = np.log10(e_max / e_min) if e_max > e_min else 1.0
    frac = np.log10(np.clip(energy, e_min, e_max) / e_min) / denom
    return _BUBBLE_SIZE_MIN + (_BUBBLE_SIZE_MAX - _BUBBLE_SIZE_MIN) * frac


def _add_bubble_size_legend(ax, e_min, e_max, unit: str, decades: int = 4):
    """Add a secondary legend with reference bubble sizes.

    Reads the same log mapping used by the scatter so the reader can
    associate a marker area with an absolute energy value.
    """
    # Pick reference values spanning the observed range with round powers
    # of 10 when possible.
    exponents = np.linspace(np.log10(e_min), np.log10(e_max), decades)
    refs = [10 ** round(e) for e in exponents]
    refs = sorted({float(f"{r:.2g}") for r in refs})

    handles = []
    labels = []
    for ref in refs:
        size = float(_bubble_size_log(np.array([ref]), e_min, e_max)[0])
        handle = ax.scatter([], [], s=size, color="gray", alpha=0.55,
                            edgecolors="black", linewidths=0.5)
        handles.append(handle)
        if ref >= 1:
            labels.append(f"{ref:g} {unit}")
        else:
            labels.append(f"{ref:.2g} {unit}")
    ax.add_artist(
        ax.legend(
            handles, labels,
            loc="lower right", title=f"Energy [{unit}]",
            frameon=True, fontsize=8, title_fontsize=8, labelspacing=1.2,
        )
    )


def plot_rde_bubble(df_op: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10.8, 6.8))

    energies = df_op["energy_per_image_j"].to_numpy(dtype=float)
    e_min = float(np.nanmin(energies))
    e_max = float(np.nanmax(energies))

    df_op = df_op.copy()
    df_op["bubble_size"] = _bubble_size_log(
        df_op["energy_per_image_j"].to_numpy(dtype=float), e_min, e_max
    )

    for codec in CODEC_ORDER:
        sub = df_op[df_op["codec"] == codec].sort_values("bpp")
        if sub.empty:
            continue

        ax.scatter(
            sub["bpp"],
            sub["ssimulacra2"],
            s=sub["bubble_size"],
            marker=MARKERS.get(codec, "o"),
            alpha=0.70,
            edgecolors="black",
            linewidths=0.4,
            label=codec,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate [bpp]")
    ax.set_ylabel("SSIMULACRA 2")
    ax.set_title("Rate--distortion--energy operating space, images")

    # Primary legend (codec). Stored explicitly so the bubble-size legend
    # below does not overwrite it.
    codec_legend = ax.legend(ncol=3, frameon=True, loc="upper left")
    ax.add_artist(codec_legend)

    # Secondary legend: bubble-size reference.
    _add_bubble_size_legend(ax, e_min, e_max, unit="J/img")

    savefig("image_rde_bubble_ssimulacra2_energy_4dataset.pdf")


def plot_dataset_energy(df: pd.DataFrame):
    dataset_summary = (
        df.groupby(["eval_dataset", "codec"], as_index=False)
        .agg(energy=("energy_per_image_j", "mean"))
        .copy()
    )

    pivot = dataset_summary.pivot(
        index="codec", columns="eval_dataset", values="energy"
    )

    pivot = pivot.reindex([c for c in CODEC_ORDER if c in pivot.index])

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    x = np.arange(len(pivot.index))
    width = 0.18

    datasets = list(pivot.columns)
    offsets = (np.arange(len(datasets)) - (len(datasets) - 1) / 2) * width

    for off, ds in zip(offsets, datasets):
        ax.bar(x + off, pivot[ds], width=width, label=ds)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("Mean operational energy [J/image]")
    ax.set_title("Energy sensitivity across image datasets")
    ax.legend(frameon=True)

    savefig("image_energy_by_dataset_4dataset.pdf")


def write_summary(df_op: pd.DataFrame):
    summary = (
        df_op.groupby("codec", as_index=False)
        .agg(
            bpp_mean=("bpp", "mean"),
            bpp_min=("bpp", "min"),
            bpp_max=("bpp", "max"),
            psnr_mean=("psnr", "mean"),
            ssimulacra2_mean=("ssimulacra2", "mean"),
            energy_j_mean=("energy_per_image_j", "mean"),
            energy_j_min=("energy_per_image_j", "min"),
            energy_j_max=("energy_per_image_j", "max"),
            cpu_j_mean=("energy_cpu_per_image_j", "mean"),
            gpu_j_mean=("energy_gpu_per_image_j", "mean"),
        )
        .copy()
    )

    summary["codec_order"] = summary["codec"].map(
        {codec: i for i, codec in enumerate(CODEC_ORDER)}
    ).fillna(999)
    summary = summary.sort_values("codec_order").drop(columns=["codec_order"])

    out = OUT_DIR / "image_4dataset_summary.csv"
    summary.to_csv(out, index=False)
    print(f"[OK] {out}")
    print()
    print(summary.to_string(index=False))


def main():
    df = load_data()
    df_op = operating_points(df)

    plot_rd(
        df_op,
        metric="psnr",
        ylabel="PSNR [dB]",
        title="Rate--distortion, PSNR, 4-dataset image benchmark",
        out_name="image_rd_psnr_4dataset.pdf",
    )

    plot_rd(
        df_op,
        metric="ssimulacra2",
        ylabel="SSIMULACRA 2",
        title="Rate--distortion, SSIMULACRA 2, 4-dataset image benchmark",
        out_name="image_rd_ssimulacra2_4dataset.pdf",
    )

    plot_energy_bar(df_op)
    plot_energy_components(df_op)
    plot_rde_bubble(df_op)
    plot_dataset_energy(df)
    write_summary(df_op)

    print("\n======================================================================")
    print("[DONE] Image 4-dataset plots generated")
    print(f"Input:  {IN_CSV}")
    print(f"Output: {OUT_DIR}")
    print("======================================================================")


if __name__ == "__main__":
    main()