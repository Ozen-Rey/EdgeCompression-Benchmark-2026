"""
Genera tutti i plot R-D-E del capitolo video.

Output (in ./plots/):
  1. rd_curves_<seq>.pdf  — Rate vs Distortion (PSNR/VMAF) per ogni sequenza
  2. re_curves_<seq>.pdf  — Rate vs Energy per ogni sequenza
  3. de_curves_<seq>.pdf  — Distortion vs Energy per ogni sequenza
  4. rde_3d_<seq>.pdf     — Scatter 3D R-D-E con Pareto fronts evidenziati
  5. rd_summary.pdf       — Curve aggregate (media su tutte le sequenze)
  6. re_summary.pdf       — Curve energy aggregate
  7. cross_codec_table.csv — Tabella riassuntiva per la tesi

Stile coerente: ogni codec ha un colore fisso, marker per paradigma (classic vs neural).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registra proiezione)

# ================== CONFIG ==================

RESULTS = Path.home() / "tesi" / "results" / "video"
PLOTS = Path.home() / "tesi" / "plots" / "video"
PLOTS.mkdir(parents=True, exist_ok=True)

CODECS_CLASSIC = ["x264", "x265", "svtav1", "vvenc"]
CODECS_NEURAL = ["dcvc_dc", "dcvc_fm", "dcvc_rt", "dcvc_rt_cuda"]
ALL_CODECS = CODECS_CLASSIC + CODECS_NEURAL

# Stile: colore + marker per ogni codec
STYLE = {
    "x264": {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "x264"},
    "x265": {"color": "#2ca02c", "marker": "o", "ls": "-", "label": "x265 (HEVC)"},
    "svtav1": {"color": "#ff7f0e", "marker": "o", "ls": "-", "label": "SVT-AV1"},
    "vvenc": {"color": "#d62728", "marker": "o", "ls": "-", "label": "vvenc (VVC)"},
    "dcvc_dc": {"color": "#9467bd", "marker": "s", "ls": "--", "label": "DCVC-DC"},
    "dcvc_fm": {"color": "#8c564b", "marker": "s", "ls": "--", "label": "DCVC-FM"},
    "dcvc_rt": {
        "color": "#e377c2",
        "marker": "^",
        "ls": ":",
        "label": "DCVC-RT (fallback)",
    },
    "dcvc_rt_cuda": {
        "color": "#17becf",
        "marker": "^",
        "ls": "-",
        "label": "DCVC-RT (CUDA)",
    },
}

# Profilo da plottare
PROFILE = "LDP"


# ================== LOAD DATA ==================


def load_all_data():
    """Carica i CSV di tutti i codec, aggrega encode+decode per i classici/RT-cuda."""
    energy_dfs = []
    quality_dfs = []
    for codec in ALL_CODECS:
        e_path = RESULTS / f"{codec}_energy.csv"
        q_path = RESULTS / f"{codec}_quality.csv"
        if e_path.exists():
            df_e = pd.read_csv(e_path)
            energy_dfs.append(df_e)
        if q_path.exists():
            df_q = pd.read_csv(q_path)
            quality_dfs.append(df_q)

    df_e = pd.concat(energy_dfs, ignore_index=True)
    df_q = pd.concat(quality_dfs, ignore_index=True)

    # Aggregate energy: per i classici e RT-cuda è encode+decode; per neurali fwd è una riga
    df_e_agg = (
        df_e.groupby(["codec", "seq", "profile", "param"])
        .agg(
            {
                "energy_total_j": "sum",
                "energy_cpu_j": "sum",
                "energy_gpu_j": "sum",
                "time_s": "sum",
                "actual_mbps": "first",
                "n_frames": "first",
            }
        )
        .reset_index()
    )

    df = pd.merge(df_e_agg, df_q, on=["codec", "seq", "profile", "param"])
    df = df[df["profile"] == PROFILE]
    return df


# ================== PLOT FUNCTIONS ==================


def plot_rd_curve(df, seq, distortion="vmaf_mean", save_path=None):
    """Rate-Distortion curve per una sequenza."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    df_seq = df[df["seq"] == seq].sort_values("actual_mbps")
    for codec in ALL_CODECS:
        df_c = df_seq[df_seq["codec"] == codec].sort_values("actual_mbps")
        if len(df_c) < 2:
            continue
        st = STYLE[codec]
        ax.plot(
            df_c["actual_mbps"],
            df_c[distortion],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            label=st["label"],
            markersize=8,
            linewidth=1.8,
        )

    dist_name = {
        "vmaf_mean": "VMAF",
        "psnr_y": "PSNR-Y (dB)",
        "psnr_yuv": "PSNR-YUV (dB)",
    }.get(distortion, distortion)
    ax.set_xlabel("Bitrate (Mbps)")
    ax.set_ylabel(dist_name)
    ax.set_title(f"Rate-Distortion — {seq}")
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_re_curve(df, seq, save_path=None):
    """Rate-Energy curve per una sequenza."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    df_seq = df[df["seq"] == seq].sort_values("actual_mbps")
    for codec in ALL_CODECS:
        df_c = df_seq[df_seq["codec"] == codec].sort_values("actual_mbps")
        if len(df_c) < 2:
            continue
        st = STYLE[codec]
        ax.plot(
            df_c["actual_mbps"],
            df_c["energy_total_j"] / 1000,  # in kJ
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            label=st["label"],
            markersize=8,
            linewidth=1.8,
        )

    ax.set_xlabel("Bitrate (Mbps)")
    ax.set_ylabel("Energy (kJ)")
    ax.set_title(f"Rate-Energy — {seq}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_de_curve(df, seq, distortion="vmaf_mean", save_path=None):
    """Distortion-Energy curve per una sequenza."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    df_seq = df[df["seq"] == seq].sort_values(distortion)
    for codec in ALL_CODECS:
        df_c = df_seq[df_seq["codec"] == codec].sort_values(distortion)
        if len(df_c) < 2:
            continue
        st = STYLE[codec]
        ax.plot(
            df_c[distortion],
            df_c["energy_total_j"] / 1000,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            label=st["label"],
            markersize=8,
            linewidth=1.8,
        )

    dist_name = {"vmaf_mean": "VMAF", "psnr_y": "PSNR-Y (dB)"}.get(
        distortion, distortion
    )
    ax.set_xlabel(dist_name)
    ax.set_ylabel("Energy (kJ)")
    ax.set_title(f"Quality-Energy — {seq}")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def is_pareto_3d(rates, dists, energies):
    """Calcola maschera Pareto front in 3D: (rate↓, dist↑, energy↓)."""
    pts = np.stack([rates, -dists, energies], axis=1)  # nego dist per "minimizzare"
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                is_pareto[i] = False
                break
    return is_pareto


def plot_rde_3d(df, seq, distortion="vmaf_mean", save_path=None):
    """Scatter 3D R-D-E con Pareto fronts evidenziati per ogni codec."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    df_seq = df[df["seq"] == seq]

    # Scatter di tutti i punti, e per ogni codec evidenziamo il fronte di Pareto
    for codec in ALL_CODECS:
        df_c = df_seq[df_seq["codec"] == codec]
        if len(df_c) == 0:
            continue
        st = STYLE[codec]
        rates = df_c["actual_mbps"].values
        dists = df_c[distortion].values
        energies = df_c["energy_total_j"].values / 1000  # kJ

        # Tutti i punti in trasparenza
        ax.scatter(
            rates,
            dists,
            energies,
            c=st["color"],
            marker=st["marker"],
            s=60,
            alpha=0.4,
            label=st["label"],
        )

        # Connetti in ordine di rate (linea trend)
        order = np.argsort(rates)
        ax.plot(
            rates[order],
            dists[order],
            energies[order],
            color=st["color"],
            linestyle=st["ls"],
            linewidth=1.5,
            alpha=0.6,
        )

    ax.set_xlabel("Bitrate (Mbps)")
    ax.set_ylabel("VMAF" if distortion == "vmaf_mean" else "PSNR-Y")
    ax.set_zlabel("Energy (kJ)")
    ax.set_title(f"R-D-E 3D — {seq}")
    # Note: zscale log non funziona se ci sono valori molto piccoli;
    # usiamo lineare ma con i classici a basso valore comprimi visivamente.
    # Se vuoi log: ax.set_zscale('log') + assicurarsi tutti i valori > 0.
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_aggregate_rd(df, distortion="vmaf_mean", save_path=None):
    """Curve aggregate: media su tutte le sequenze per ogni codec."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for codec in ALL_CODECS:
        df_c = df[df["codec"] == codec]
        if len(df_c) == 0:
            continue
        # Per ogni param (q_index/CRF), media su sequenze
        agg = (
            df_c.groupby("param")
            .agg(
                {
                    "actual_mbps": "mean",
                    distortion: "mean",
                    "energy_total_j": "mean",
                }
            )
            .reset_index()
            .sort_values("actual_mbps")
        )
        st = STYLE[codec]
        ax.plot(
            agg["actual_mbps"],
            agg[distortion],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            label=st["label"],
            markersize=10,
            linewidth=2,
        )

    dist_name = {"vmaf_mean": "VMAF", "psnr_y": "PSNR-Y (dB)"}.get(
        distortion, distortion
    )
    ax.set_xlabel("Bitrate (Mbps)")
    ax.set_ylabel(dist_name)
    ax.set_title(f"Aggregate Rate-Distortion (mean across UVG sequences)")
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_aggregate_re(df, save_path=None):
    """Curve aggregate energy."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for codec in ALL_CODECS:
        df_c = df[df["codec"] == codec]
        if len(df_c) == 0:
            continue
        agg = (
            df_c.groupby("param")
            .agg(
                {
                    "actual_mbps": "mean",
                    "energy_total_j": "mean",
                }
            )
            .reset_index()
            .sort_values("actual_mbps")
        )
        st = STYLE[codec]
        ax.plot(
            agg["actual_mbps"],
            agg["energy_total_j"] / 1000,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["ls"],
            label=st["label"],
            markersize=10,
            linewidth=2,
        )

    ax.set_xlabel("Bitrate (Mbps)")
    ax.set_ylabel("Energy (kJ)")
    ax.set_title("Aggregate Rate-Energy (mean across UVG sequences)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def export_summary_table(df, out_path):
    """Tabella di riepilogo per la tesi: per codec, valori a bitrate medio."""
    rows = []
    for codec in ALL_CODECS:
        df_c = df[df["codec"] == codec]
        if len(df_c) == 0:
            continue
        # Pick mid q_index per ogni sequenza, poi media
        for seq in df_c["seq"].unique():
            df_cs = df_c[df_c["seq"] == seq].sort_values("actual_mbps")
            if len(df_cs) == 0:
                continue
            mid = df_cs.iloc[len(df_cs) // 2]
            rows.append(
                {
                    "codec": codec,
                    "seq": seq,
                    "bitrate_mbps": mid["actual_mbps"],
                    "vmaf": mid["vmaf_mean"],
                    "psnr_y": mid["psnr_y"],
                    "time_s": mid["time_s"],
                    "energy_kJ": mid["energy_total_j"] / 1000,
                }
            )

    summary = pd.DataFrame(rows)

    # Pivot: codec × seq con vari valori
    summary.to_csv(out_path, index=False)

    # Summary per codec (media su seq)
    print("\n=== Summary per codec (mean across sequences, mid bitrate) ===")
    by_codec = (
        summary.groupby("codec")
        .agg(
            {
                "bitrate_mbps": "mean",
                "vmaf": "mean",
                "psnr_y": "mean",
                "time_s": "mean",
                "energy_kJ": "mean",
            }
        )
        .round(3)
    )
    print(by_codec.to_string())


# ================== MAIN ==================


def main():
    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} (codec, seq, q) entries")

    sequences = sorted(df["seq"].unique())
    print(f"Sequences: {sequences}")

    print("\nGenerating per-sequence plots...")
    for seq in sequences:
        plot_rd_curve(df, seq, "vmaf_mean", PLOTS / f"rd_vmaf_{seq}.pdf")
        plot_rd_curve(df, seq, "psnr_y", PLOTS / f"rd_psnr_{seq}.pdf")
        plot_re_curve(df, seq, PLOTS / f"re_{seq}.pdf")
        plot_de_curve(df, seq, "vmaf_mean", PLOTS / f"de_vmaf_{seq}.pdf")
        plot_rde_3d(df, seq, "vmaf_mean", PLOTS / f"rde_3d_{seq}.pdf")
        print(f"  ✓ {seq}")

    print("\nGenerating aggregate plots...")
    plot_aggregate_rd(df, "vmaf_mean", PLOTS / "rd_aggregate_vmaf.pdf")
    plot_aggregate_rd(df, "psnr_y", PLOTS / "rd_aggregate_psnr.pdf")
    plot_aggregate_re(df, PLOTS / "re_aggregate.pdf")
    print("  ✓ aggregate")

    print("\nExporting summary table...")
    export_summary_table(df, PLOTS / "cross_codec_summary.csv")
    print(f"  ✓ {PLOTS / 'cross_codec_summary.csv'}")

    print(f"\nAll plots saved to {PLOTS}")


if __name__ == "__main__":
    main()
