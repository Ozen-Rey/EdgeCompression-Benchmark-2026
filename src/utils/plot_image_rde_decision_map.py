#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a decision map for image-domain R-D-E routing.

The script reads the image benchmark CSV and builds a heatmap over the
R-D-E weight simplex. For each pair (w_E, w_D), with w_R = 1 - w_E - w_D,
the script computes the R-D-E oracle selection for every image and reports the
dominant codec selected across the dataset.

Required CSV columns:
    dataset
    image
    codec
    param
    bpp
    ssimulacra2
    energy_per_image_j

Example usage from the project root:

python .\\src\\utils\\plot_image_rde_decision_map.py `
  --input .\\results\\images\\image_4dataset_RDE_paper_ready.csv `
  --output-dir .\\results\\images\\rde_routing `
  --quality-threshold 50 `
  --step 0.05 `
  --dpi 300
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an R-D-E routing decision map for the image benchmark."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="rde_heatmap_output",
        help="Directory where PNG, PDF and CSV outputs will be saved."
    )

    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=50.0,
        help="Minimum admissible SSIMULACRA2 value."
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Grid step for w_E and w_D."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI."
    )

    parser.add_argument(
        "--title",
        action="store_true",
        help="If set, add a short title to the plot."
    )

    parser.add_argument(
        "--annotate-cells",
        action="store_true",
        help="If set, write codec abbreviations inside feasible cells."
    )

    return parser.parse_args()


def normalize_codec_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize codec names to avoid duplicate labels in the legend."""
    df = df.copy()

    df["codec"] = df["codec"].replace(
        {
            "JPEG_AI": "JPEG AI",
            "JPEG-AI": "JPEG AI",
            "JPEGAI": "JPEG AI",
            "JPEG_AI ": "JPEG AI",
            "JPEG AI ": "JPEG AI",
        }
    )

    return df


def load_and_prepare_data(csv_path: str, quality_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_codec_names(df)

    required_cols = [
        "dataset",
        "image",
        "codec",
        "param",
        "bpp",
        "ssimulacra2",
        "energy_per_image_j",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Remove exact repeated measurements at image/configuration level.
    dedup_cols = [
        "dataset",
        "codec",
        "param",
        "image",
        "bpp",
        "ssimulacra2",
        "energy_per_image_j",
    ]

    df = df.drop_duplicates(subset=dedup_cols).copy()
    df = df[required_cols].dropna().copy()

    # Hard admissibility constraint.
    df = df[df["ssimulacra2"] >= quality_threshold].copy()

    if df.empty:
        raise ValueError(
            f"No admissible points remain after applying "
            f"SSIMULACRA2 >= {quality_threshold}."
        )

    df["sample_id"] = df["dataset"].astype(str) + "::" + df["image"].astype(str)

    return df


def normalize_rde(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize R, E and D.

    Rate and energy are normalized in log scale because their values may span
    several orders of magnitude. Distortion is obtained from SSIMULACRA2, where
    higher quality means lower distortion.
    """
    df = df.copy()

    log_r = np.log(df["bpp"].clip(lower=1e-12))
    log_e = np.log(df["energy_per_image_j"].clip(lower=1e-12))
    q = df["ssimulacra2"]

    r_min, r_max = log_r.min(), log_r.max()
    e_min, e_max = log_e.min(), log_e.max()
    q_low, q_high = q.min(), q.max()

    if r_max > r_min:
        df["R_hat"] = (log_r - r_min) / (r_max - r_min)
    else:
        df["R_hat"] = 0.0

    if e_max > e_min:
        df["E_hat"] = (log_e - e_min) / (e_max - e_min)
    else:
        df["E_hat"] = 0.0

    if q_high > q_low:
        df["D_hat"] = 1.0 - (q - q_low) / (q_high - q_low)
    else:
        df["D_hat"] = 0.0

    return df


def codec_metadata():
    """
    Fixed order, abbreviations and colors for reproducible plotting.
    Add new codecs here if needed.
    """
    codec_order = [
        "JPEG",
        "JXL",
        "HEVC",
        "JPEG AI",
        "Ballé",
        "Cheng",
        "ELIC",
        "TCM",
        "DCAE",
    ]

    codec_abbr = {
        "JPEG": "JPEG",
        "JXL": "JXL",
        "HEVC": "HEVC",
        "JPEG AI": "JAI",
        "Ballé": "BAL",
        "Cheng": "CHG",
        "ELIC": "ELIC",
        "TCM": "TCM",
        "DCAE": "DCAE",
    }

    # Deterministic categorical colors.
    codec_colors = {
        "JPEG": "#4C72B0",
        "JXL": "#2B908F",
        "HEVC": "#F1DD1F",
        "JPEG AI": "#59A14F",
        "Ballé": "#F28E2B",
        "Cheng": "#E15759",
        "ELIC": "#76B7B2",
        "TCM": "#B07AA1",
        "DCAE": "#440154",
    }

    return codec_order, codec_abbr, codec_colors


def build_decision_grid(df: pd.DataFrame, step: float):
    wE_values = np.round(np.arange(0.0, 1.0 + step, step), 10)
    wD_values = np.round(np.arange(0.0, 1.0 + step, step), 10)

    codec_order, codec_abbr, codec_colors = codec_metadata()

    # Include unexpected codec names robustly.
    observed_codecs = sorted(df["codec"].unique())
    for codec in observed_codecs:
        if codec not in codec_order:
            codec_order.append(codec)
            codec_abbr[codec] = codec
            codec_colors[codec] = "#8C564B"

    codec_to_code = {codec: idx + 1 for idx, codec in enumerate(codec_order)}
    code_to_codec = {idx + 1: codec for idx, codec in enumerate(codec_order)}

    grid = np.full((len(wD_values), len(wE_values)), np.nan)
    share_grid = np.full((len(wD_values), len(wE_values)), np.nan)

    rows = []

    for iy, wD in enumerate(wD_values):
        for ix, wE in enumerate(wE_values):
            if wE + wD > 1.0 + 1e-12:
                continue

            wR = 1.0 - wE - wD

            tmp = df.copy()
            tmp["J_RDE"] = (
                wE * tmp["E_hat"]
                + wR * tmp["R_hat"]
                + wD * tmp["D_hat"]
            )

            # Oracle-selected point per image.
            idx = tmp.groupby("sample_id")["J_RDE"].idxmin()
            chosen = tmp.loc[idx, ["sample_id", "codec", "param", "J_RDE"]].copy()

            counts = chosen["codec"].value_counts()
            dominant_codec = counts.idxmax()
            dominant_share = counts.max() / counts.sum()

            grid[iy, ix] = codec_to_code[dominant_codec]
            share_grid[iy, ix] = dominant_share

            rows.append(
                {
                    "w_E": float(wE),
                    "w_R": float(wR),
                    "w_D": float(wD),
                    "dominant_codec": dominant_codec,
                    "dominant_codec_abbr": codec_abbr[dominant_codec],
                    "dominant_share": float(dominant_share),
                }
            )

    decision_df = pd.DataFrame(rows)

    return (
        wE_values,
        wD_values,
        grid,
        share_grid,
        decision_df,
        codec_order,
        codec_to_code,
        code_to_codec,
        codec_abbr,
        codec_colors,
    )


def plot_decision_map(
    wE_values,
    wD_values,
    grid,
    quality_threshold,
    codec_order,
    code_to_codec,
    codec_abbr,
    codec_colors,
    output_png,
    output_pdf,
    dpi,
    step,
    show_title=False,
    annotate_cells=False,
):
    max_code = int(np.nanmax(grid))

    ordered_codecs_present = []
    for code in range(1, max_code + 1):
        codec = code_to_codec.get(code)
        if codec is not None and np.any(grid == code):
            ordered_codecs_present.append(codec)

    colors = [codec_colors[codec] for codec in codec_order[:max_code]]
    cmap = ListedColormap(colors)

    bounds = np.arange(0.5, max_code + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    masked = np.ma.masked_invalid(grid)

    ax.imshow(
        masked,
        origin="lower",
        extent=[
            wE_values.min() - step / 2,
            wE_values.max() + step / 2,
            wD_values.min() - step / 2,
            wD_values.max() + step / 2,
        ],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # Feasible simplex boundary: w_E >= 0, w_D >= 0, w_E + w_D <= 1.
    ax.plot([0.0, 1.0], [1.0, 0.0], linewidth=1.2, color="black")
    ax.plot([0.0, 1.0], [0.0, 0.0], linewidth=1.2, color="black")
    ax.plot([0.0, 0.0], [0.0, 1.0], linewidth=1.2, color="black")

    if annotate_cells:
        for iy, wD in enumerate(wD_values):
            for ix, wE in enumerate(wE_values):
                if np.isnan(grid[iy, ix]):
                    continue
                code = int(grid[iy, ix])
                codec = code_to_codec[code]
                label = codec_abbr.get(codec, codec)
                ax.text(
                    wE,
                    wD,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="black",
                )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel(r"$w_E$")
    ax.set_ylabel(r"$w_D$")

    if show_title:
        ax.set_title(
            "Image R-D-E routing decision map\n"
            + f"SSIMULACRA2 $\\geq$ {quality_threshold:.0f}, "
            + r"$w_R = 1 - w_E - w_D$"
        )

    # Legend only for codecs that appear in the map.
    legend_handles = []
    for codec in ordered_codecs_present:
        legend_handles.append(
            Patch(
                facecolor=codec_colors[codec],
                edgecolor="black",
                label=f"{codec_abbr.get(codec, codec)} = {codec}",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        title="Codec",
        title_fontsize=9,
    )

    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(args.input, args.quality_threshold)
    df = normalize_rde(df)

    (
        wE_values,
        wD_values,
        grid,
        share_grid,
        decision_df,
        codec_order,
        codec_to_code,
        code_to_codec,
        codec_abbr,
        codec_colors,
    ) = build_decision_grid(df, args.step)

    threshold_tag = str(int(args.quality_threshold))
    step_tag = str(args.step).replace(".", "p")

    output_png = output_dir / f"image_rde_decision_map_ss{threshold_tag}_step{step_tag}.png"
    output_pdf = output_dir / f"image_rde_decision_map_ss{threshold_tag}_step{step_tag}.pdf"
    output_csv = output_dir / f"image_rde_decision_map_ss{threshold_tag}_step{step_tag}.csv"

    plot_decision_map(
        wE_values=wE_values,
        wD_values=wD_values,
        grid=grid,
        quality_threshold=args.quality_threshold,
        codec_order=codec_order,
        code_to_codec=code_to_codec,
        codec_abbr=codec_abbr,
        codec_colors=codec_colors,
        output_png=output_png,
        output_pdf=output_pdf,
        dpi=args.dpi,
        step=args.step,
        show_title=args.title,
        annotate_cells=args.annotate_cells,
    )

    decision_df.to_csv(output_csv, index=False)

    print(f"Saved PNG figure to: {output_png}")
    print(f"Saved PDF figure to: {output_pdf}")
    print(f"Saved decision grid CSV to: {output_csv}")


if __name__ == "__main__":
    main()