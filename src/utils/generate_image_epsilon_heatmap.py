from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "results" / "images" / "image_4dataset_RDE_paper_ready.csv"
OUT_DIR = ROOT / "plots" / "images" / "four_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)

    numeric_cols = [
        "bpp",
        "psnr",
        "ssimulacra2",
        "energy_per_image_j",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # tieni solo righe valide per la heatmap R-D-E
    df = df.dropna(subset=["codec", "param", "bpp", "ssimulacra2", "energy_per_image_j"]).copy()

    # uniforma eventuale label
    if "codec_label" not in df.columns:
        df["codec_label"] = df["codec"]

    return df


def aggregate_points(df: pd.DataFrame) -> pd.DataFrame:
    # media sui 4 dataset per ogni punto operativo del codec
    df_avg = (
        df.groupby(["codec_label", "param"], as_index=False)
        .agg(
            bpp=("bpp", "mean"),
            psnr=("psnr", "mean"),
            ssimulacra2=("ssimulacra2", "mean"),
            energy_per_image_j=("energy_per_image_j", "mean"),
        )
        .sort_values(["codec_label", "bpp"])
    )

    order_map = {codec: i for i, codec in enumerate(CODEC_ORDER)}
    df_avg["order"] = df_avg["codec_label"].map(order_map)
    df_avg = df_avg.sort_values(["order", "bpp"]).drop(columns="order")

    return df_avg


def normalized_points(g: pd.DataFrame, bounds: dict[str, np.ndarray]) -> np.ndarray:
    """
    Lower-is-better representation:
        x1 = log10(bpp)
        x2 = -SSIMULACRA2
        x3 = log10(energy_per_image_j)

    Poi min-max normalization.
    """
    r = np.log10(g["bpp"].to_numpy())
    q = -g["ssimulacra2"].to_numpy()
    e = np.log10(g["energy_per_image_j"].to_numpy())

    pts = np.column_stack([r, q, e])
    return (pts - bounds["min"]) / bounds["range"]


def additive_epsilon_indicator(a: np.ndarray, b: np.ndarray) -> float:
    """
    Additive epsilon indicator I_eps(A, B).

    Restituisce il minimo epsilon additivo tale che A weakly dominates B.
    Lower is better.
    """
    vals = []
    for b_i in b:
        inner = []
        for a_i in a:
            inner.append(np.max(a_i - b_i))
        vals.append(np.min(inner))
    return float(np.max(vals))


def plot_heatmap(mat: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.2))

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)

    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Codec target")
    ax.set_ylabel("Codec baseline")
    ax.set_title("Additive ε-indicator heatmap, image R-D-E (4 datasets)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ε(target, baseline), normalized")

    text_threshold = vmin + 0.60 * (vmax - vmin)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            if np.isnan(v):
                text = "—"
                color = "black"
            else:
                text = f"{v:+.2f}"
                color = "white" if v > text_threshold else "black"

            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "image_heatmap_epsilon_rde_ssimulacra2_4dataset.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def save_summary(mat: np.ndarray, labels: list[str]) -> None:
    df = pd.DataFrame(mat, index=labels, columns=labels)
    df.to_csv(OUT_DIR / "image_heatmap_epsilon_rde_ssimulacra2_4dataset.csv", index=True)


def main() -> None:
    df = load_data()
    df_avg = aggregate_points(df)

    labels = [c for c in CODEC_ORDER if c in set(df_avg["codec_label"])]

    all_log_r = np.log10(df_avg["bpp"].to_numpy())
    all_neg_q = -df_avg["ssimulacra2"].to_numpy()
    all_log_e = np.log10(df_avg["energy_per_image_j"].to_numpy())

    all_pts = np.column_stack([all_log_r, all_neg_q, all_log_e])

    bounds = {
        "min": np.nanmin(all_pts, axis=0),
        "range": np.maximum(np.nanmax(all_pts, axis=0) - np.nanmin(all_pts, axis=0), 1e-12),
    }

    eps = np.full((len(labels), len(labels)), np.nan)

    for i, baseline in enumerate(labels):
        g_baseline = df_avg[df_avg["codec_label"] == baseline]
        pts_baseline = normalized_points(g_baseline, bounds)

        for j, target in enumerate(labels):
            if baseline == target:
                continue

            g_target = df_avg[df_avg["codec_label"] == target]
            pts_target = normalized_points(g_target, bounds)

            eps[i, j] = additive_epsilon_indicator(pts_target, pts_baseline)

    plot_heatmap(eps, labels)
    save_summary(eps, labels)

    print("[OK]", OUT_DIR / "image_heatmap_epsilon_rde_ssimulacra2_4dataset.pdf")
    print("[OK]", OUT_DIR / "image_heatmap_epsilon_rde_ssimulacra2_4dataset.csv")


if __name__ == "__main__":
    main()