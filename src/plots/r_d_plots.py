"""
Grafici R-D-E per la tesi — TUTTE e tre le dimensioni.
8 codec immagini (full pipeline) + 5 codec audio.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

OUT = os.path.expanduser("~/tesi/figures")
os.makedirs(OUT, exist_ok=True)

C = {
    "JPEG": "#7f7f7f",
    "JXL": "#2ca02c",
    "HEVC": "#17becf",
    "Ballé": "#ff7f0e",
    "Cheng": "#d62728",
    "ELIC": "#1f77b4",
    "TCM": "#9467bd",
    "DCAE": "#8c564b",
    "Opus": "#2ca02c",
    "EnCodec": "#1f77b4",
    "DAC": "#d62728",
    "SNAC": "#ff7f0e",
    "WavTokenizer": "#9467bd",
}
M = {
    "JPEG": "x",
    "JXL": "s",
    "HEVC": "p",
    "Ballé": "^",
    "Cheng": "v",
    "ELIC": "o",
    "TCM": "D",
    "DCAE": "P",
    "Opus": "s",
    "EnCodec": "o",
    "DAC": "D",
    "SNAC": "^",
    "WavTokenizer": "P",
}


# ====================================================================
# DATI
# ====================================================================
def load_rde():
    """Tutti i codec immagini con R-D-E full pipeline."""
    df = pd.read_csv(
        os.path.expanduser("~/tesi/results/images/full_pipeline_energy_benchmark.csv")
    )

    # Aggiungi ELIC dal benchmark precedente
    try:
        elic_df = pd.read_csv(
            os.path.expanduser("~/tesi/results/jxl_vs_elic_benchmark.csv")
        )
        elic_full = elic_df[
            (elic_df["codec"] == "ELIC") & (elic_df["pipeline"] == "full_pipeline")
        ].copy()
        elic_rows = []
        for _, row in elic_full.iterrows():
            elic_rows.append(
                {
                    "codec": "ELIC",
                    "param": row["param"],
                    "image": row["image"],
                    "bpp": row["bpp"],
                    "psnr": row["psnr"],
                    "time_total_ms": row.get("time_ms", None),
                    "energy_j": row.get("energy_net_j", None),
                    "energy_source": "Zeus",
                    "params_M": 36.9,
                    "pipeline": "full_pipeline",
                }
            )
        if elic_rows:
            df = pd.concat([df, pd.DataFrame(elic_rows)], ignore_index=True)
    except Exception as e:
        print(f"  [!] ELIC non caricato: {e}")

    return df.dropna(subset=["bpp", "psnr", "energy_j"])


def load_quality():
    """12 metriche (per SSIMULACRA2 plot)."""
    dfs = []
    for path in [
        "~/tesi/results/jxl_vs_elic_benchmark.csv",
        "~/tesi/results/images/sota_12metric_benchmark.csv",
        "~/tesi/results/images/missing_12metric_benchmark.csv",
    ]:
        try:
            df = pd.read_csv(os.path.expanduser(path))
            if "pipeline" in df.columns:
                df = df[
                    (df["codec"] != "ELIC") | (df["pipeline"] == "forward_pass_only")
                ]
            dfs.append(df)
        except Exception:
            pass
    cols = ["codec", "param", "image", "bpp", "psnr", "ssimulacra2", "lpips", "ssim"]
    return pd.concat(
        [d[[c for c in cols if c in d.columns]] for d in dfs], ignore_index=True
    )


def load_audio():
    d = {}
    d["full"] = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/audio_benchmark_full.csv")
    )
    d["wt"] = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/wavtokenizer_metrics.csv")
    )
    d["vsp"] = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/visqol_benchmark.csv")
    )
    d["vau"] = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/visqol_audio_mode_benchmark.csv")
    )
    d["fad"] = pd.read_csv(os.path.expanduser("~/tesi/results/audio/fad_benchmark.csv"))
    return d


# ====================================================================
# HELPER
# ====================================================================
def avg(df, codec):
    s = (
        df[df["codec"] == codec]
        .groupby("param")
        .agg(bpp=("bpp", "mean"), psnr=("psnr", "mean"), energy=("energy_j", "mean"))
        .reset_index()
        .sort_values("bpp")
    )
    return s


CODEC_ORDER_IMG = ["JPEG", "JXL", "HEVC", "Ballé", "Cheng", "ELIC", "TCM", "DCAE"]


# ====================================================================
# FIG 1: BUBBLE R-D-E (bpp vs PSNR, bolla = energia)
# ====================================================================
def fig1(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_e = df.groupby(["codec", "param"])["energy_j"].mean()
    e_min = max(all_e.min(), 0.01)

    for codec in CODEC_ORDER_IMG:
        s = avg(df, codec)
        if len(s) == 0:
            continue
        ax.plot(s["bpp"], s["psnr"], color=C[codec], alpha=0.4, linewidth=1, zorder=3)
        sizes = np.sqrt(np.maximum(s["energy"].values, e_min) / e_min) * 30
        ax.scatter(
            s["bpp"],
            s["psnr"],
            s=sizes,
            c=C[codec],
            marker=M[codec],
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
            alpha=0.85,
            label=codec,
        )

    ax.set_xlabel("Rate — Bitrate (bpp)")
    ax.set_ylabel("Distortion — PSNR (dB) ↑")
    ax.set_title(
        "Framework R-D-E: Rate vs Distortion vs Energy\n(dimensione bolla ∝ √energia, full pipeline)"
    )
    ax.set_xlim(0, 2.0)
    ax.set_ylim(25, 45)

    handles = [
        Line2D(
            [0], [0], color=C[c], marker=M[c], linestyle="none", markersize=8, label=c
        )
        for c in CODEC_ORDER_IMG
        if c in df["codec"].unique()
    ]
    leg1 = ax.legend(handles=handles, loc="lower right", title="Codec", ncol=2)
    ax.add_artist(leg1)

    for e_ref in [0.05, 1.0, 15, 400]:
        ax.scatter(
            [],
            [],
            s=np.sqrt(e_ref / e_min) * 30,
            c="gray",
            alpha=0.4,
            edgecolors="black",
            linewidth=0.5,
            label=f"{e_ref:.0f} J" if e_ref >= 1 else f"{e_ref:.2f} J",
        )
    ax.legend(loc="upper right", title="Energia (J)", labelspacing=1.2, framealpha=0.9)

    fig.savefig(os.path.join(OUT, "fig1_rde_bubble.pdf"))
    plt.close()
    print("  fig1_rde_bubble.pdf")


# ====================================================================
# FIG 2: R-D SSIMULACRA2 (8 codec)
# ====================================================================
def fig2(dfq):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for codec in CODEC_ORDER_IMG:
        s = dfq[dfq["codec"] == codec].dropna(subset=["ssimulacra2"])
        if len(s) == 0:
            continue
        a = (
            s.groupby("param")
            .agg(bpp=("bpp", "mean"), s2=("ssimulacra2", "mean"))
            .reset_index()
            .sort_values("bpp")
        )
        lw = 2.8 if codec == "JXL" else 1.8
        ax.plot(
            a["bpp"],
            a["s2"],
            color=C[codec],
            marker=M.get(codec, "o"),
            label=codec,
            zorder=5,
            linewidth=lw,
        )

    ax.set_xlabel("Rate (bpp)")
    ax.set_ylabel("SSIMULACRA2 (↑ meglio)")
    ax.set_title("Qualità psicovisiva vs Bitrate — JXL domina anche vs SOTA 2025")
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlim(0, 2.0)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    fig.savefig(os.path.join(OUT, "fig2_rd_ssimulacra2.pdf"))
    plt.close()
    print("  fig2_rd_ssimulacra2.pdf")


# ====================================================================
# FIG 3: PSNR vs Energia (efficienza qualità/costo)
# ====================================================================
def fig3(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for codec in CODEC_ORDER_IMG:
        s = avg(df, codec)
        if len(s) == 0:
            continue
        ax.plot(
            s["energy"],
            s["psnr"],
            color=C[codec],
            marker=M[codec],
            label=codec,
            zorder=5,
            linewidth=1.8,
        )

    ax.set_xlabel("Energy — Energia full pipeline (J) →")
    ax.set_ylabel("Distortion — PSNR (dB) ↑")
    ax.set_title("Trade-off Qualità vs Energia (full pipeline)")
    ax.set_xscale("log")
    ax.legend(loc="lower right", ncol=2)

    fig.savefig(os.path.join(OUT, "fig3_psnr_vs_energy.pdf"))
    plt.close()
    print("  fig3_psnr_vs_energy.pdf")


# ====================================================================
# FIG 4: Energy bar chart a ~34 dB (confronto fair)
# ====================================================================
def fig4(df):
    # Punti a ~34 dB PSNR per confronto fair
    targets = [
        ("JPEG", "q=85", 36.46, 0.07),
        ("JXL", "d=1.0", 37.89, 1.37),
        ("HEVC", "crf=25", 38.63, 5.65),
        ("Ballé", "q=5", 37.32, 1.32),
        ("Cheng", "q=5", 34.95, 404.0),
        ("TCM", "lam=0.013", 34.14, 14.41),
        ("DCAE", "lam=0.013", 34.52, 29.04),
    ]

    # Aggiungi ELIC se disponibile
    elic_sub = df[(df["codec"] == "ELIC") & (df["param"] == "lam=0.150")]
    if len(elic_sub) > 0:
        targets.append(
            ("ELIC", "lam=0.150", elic_sub["psnr"].mean(), elic_sub["energy_j"].mean())
        )

    targets.sort(key=lambda x: x[3])

    labels = [f"{t[0]}\n{t[1]}\n({t[2]:.0f} dB)" for t in targets]
    energies = [t[3] for t in targets]
    colors = [C[t[0]] for t in targets]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(targets)),
        energies,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energia full pipeline (J)")
    ax.set_title(
        "Costo energetico per codec (Kodak, full pipeline)\nCheng: codifica aritmetica = 99.6% del costo"
    )
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)

    for i, v in enumerate(energies):
        label = f"{v:.2f}" if v < 1 else f"{v:.0f}"
        ax.text(i, v * 1.3, f"{label} J", ha="center", fontsize=8, fontweight="bold")

    fig.savefig(os.path.join(OUT, "fig4_energy_bar.pdf"))
    plt.close()
    print("  fig4_energy_bar.pdf")


# ====================================================================
# FIG 5: Bottleneck (forward vs full)
# ====================================================================
def fig5():
    codecs = [
        "Ballé\n(5.1M)",
        "ELIC\n(36.9M)",
        "TCM\n(45.2M)",
        "DCAE\n(119.4M)",
        "Cheng\n(26.6M)",
    ]
    fwd = [3, 15, 39, 91, 15]
    full = [27, 134, 100, 127, 3336]
    arith = [f - fw for f, fw in zip(full, fwd)]

    x = np.arange(len(codecs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - w / 2,
        fwd,
        w,
        label="Forward pass",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(x + w / 2, fwd, w, color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(
        x + w / 2,
        arith,
        w,
        bottom=fwd,
        label="Codifica aritmetica",
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    for i in range(len(codecs)):
        ax.text(
            x[i] + w / 2,
            full[i] * 1.08,
            f"{full[i]} ms",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
        pct = arith[i] / full[i] * 100
        if arith[i] > 20:
            ax.text(
                x[i] + w / 2,
                fwd[i] + arith[i] / 2,
                f"{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(codecs, fontsize=9)
    ax.set_ylabel("Tempo (ms)")
    ax.set_title("Forward pass vs Codifica aritmetica")
    ax.set_yscale("log")
    ax.set_ylim(1, 8000)
    ax.legend(loc="upper left")

    fig.savefig(os.path.join(OUT, "fig5_bottleneck.pdf"))
    plt.close()
    print("  fig5_bottleneck.pdf")


# ====================================================================
# FIG 6: Audio — ViSQOL dual mode
# ====================================================================
def fig6(aud):
    kmap = {
        "12": 12,
        "24": 24,
        "48": 48,
        "1.5": 1.5,
        "3.0": 3.0,
        "6.0": 6.0,
        "8.0": 8.0,
        "0.8": 0.8,
        "0.9": 0.9,
    }

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for codec in ["Opus", "EnCodec", "DAC", "SNAC", "WavTokenizer"]:
        for ax, dkey, vcol in [(a1, "vsp", "visqol"), (a2, "vau", "visqol_audio")]:
            s = aud[dkey][aud[dkey]["codec"] == codec].copy()
            s["kbps"] = s["param"].astype(str).map(kmap)
            s = s.dropna(subset=["kbps"])
            a = s.groupby("kbps")[vcol].mean().reset_index().sort_values("kbps")
            if len(a) > 0:
                ax.plot(
                    a["kbps"], a[vcol], color=C[codec], marker=M[codec], label=codec
                )

    for ax, title in [(a1, "ViSQOL Speech (16 kHz)"), (a2, "ViSQOL Audio (48 kHz)")]:
        ax.set_xlabel("Bitrate (kbps)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 60)
        ax.set_ylim(2.5, 5.0)

    a1.set_ylabel("ViSQOL MOS-LQO (↑)")
    fig.suptitle("Qualità percettiva audio vs bitrate", fontsize=13)
    plt.subplots_adjust(top=0.85, wspace=0.08)

    fig.savefig(os.path.join(OUT, "fig6_audio_visqol.pdf"))
    plt.close()
    print("  fig6_audio_visqol.pdf")


# ====================================================================
# FIG 7: Audio ViSQOL per dataset
# ====================================================================
def fig7(aud):
    va = aud["vau"]
    codec_order = [
        ("SNAC", "0.8"),
        ("WavTokenizer", "0.9"),
        ("EnCodec", "1.5"),
        ("EnCodec", "6.0"),
        ("DAC", "8.0"),
        ("Opus", "12"),
        ("Opus", "48"),
    ]
    datasets = ["librispeech", "esc50", "musdb"]
    ds_labels = ["LibriSpeech", "ESC-50", "MUSDB18"]
    ds_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(codec_order))
    w = 0.22

    for i, (ds, dl, dc) in enumerate(zip(datasets, ds_labels, ds_colors)):
        vals = []
        for codec, param in codec_order:
            s = va[
                (va["codec"] == codec)
                & (va["param"].astype(str) == param)
                & (va["dataset"] == ds)
            ]
            vals.append(s["visqol_audio"].mean() if len(s) > 0 else 0)
        ax.bar(
            x + i * w,
            vals,
            w,
            label=dl,
            color=dc,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.85,
        )

    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{c}\n{p}k" for c, p in codec_order], fontsize=8)
    ax.set_ylabel("ViSQOL Audio (48 kHz, ↑)")
    ax.set_title("Dipendenza dal contenuto: ViSQOL per dataset")
    ax.legend()
    ax.set_ylim(1.5, 5.0)

    fig.savefig(os.path.join(OUT, "fig7_visqol_per_dataset.pdf"))
    plt.close()
    print("  fig7_visqol_per_dataset.pdf")


# ====================================================================
# FIG 8: Audio energy
# ====================================================================
def fig8():
    codecs = [
        "SNAC\n0.8 kbps",
        "Opus\n24 kbps",
        "EnCodec\n3.0 kbps",
        "WavTok\n0.9 kbps",
        "DAC\n8.0 kbps",
    ]
    j = [0.32, 0.35, 0.49, 0.51, 1.40]
    cc = [C["SNAC"], C["Opus"], C["EnCodec"], C["WavTokenizer"], C["DAC"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        range(len(codecs)), j, color=cc, edgecolor="black", linewidth=0.5, alpha=0.85
    )
    for i, v in enumerate(j):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(codecs)))
    ax.set_xticklabels(codecs, fontsize=9)
    ax.set_ylabel("Energia (J / s audio)")
    ax.set_title("Consumo energetico audio — SNAC (neurale) < Opus (classico)")
    ax.set_ylim(0, 1.7)
    ax.axhline(y=0.35, color="gray", linestyle="--", alpha=0.5)

    fig.savefig(os.path.join(OUT, "fig8_energy_audio.pdf"))
    plt.close()
    print("  fig8_energy_audio.pdf")


# ====================================================================
# FIG 9: Cross-domain asymmetry
# ====================================================================
def fig9():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Energy gap
    a1.bar(
        ["Immagini\n(Cheng/JPEG)"],
        [3700],
        color="#d62728",
        edgecolor="black",
        alpha=0.85,
    )
    a1.bar(["Audio\n(DAC/SNAC)"], [4.4], color="#2ca02c", edgecolor="black", alpha=0.85)
    a1.set_ylabel("Divario energetico massimo (×)")
    a1.set_title("Gap energetico: classico ↔ neurale")
    a1.set_yscale("log")
    a1.set_ylim(1, 10000)
    a1.text(
        0, 5000, "3700×", ha="center", fontsize=14, fontweight="bold", color="#d62728"
    )
    a1.text(1, 8, "4.4×", ha="center", fontsize=14, fontweight="bold", color="#2ca02c")

    # Perceptual winner
    x = np.arange(2)
    w = 0.3
    a2.bar(
        x - w / 2,
        [72.5, 3.57],
        w,
        label="Classico (JXL/Opus)",
        color="#2ca02c",
        edgecolor="black",
    )
    a2.bar(
        x + w / 2,
        [56.1, 3.67],
        w,
        label="Neurale SOTA (DCAE/DAC)",
        color="#1f77b4",
        edgecolor="black",
    )
    a2.set_xticks(x)
    a2.set_xticklabels(["Immagini\n(SSIMULACRA2)", "Audio\n(ViSQOL audio)"])
    a2.set_ylabel("Score percettivo (↑)")
    a2.set_title("Vincitore percettivo per dominio")
    a2.legend()

    fig.suptitle("Asimmetria cross-dominio", fontsize=14)
    plt.subplots_adjust(top=0.85, wspace=0.3)

    fig.savefig(os.path.join(OUT, "fig9_cross_domain.pdf"))
    plt.close()
    print("  fig9_cross_domain.pdf")


# ====================================================================
# FIG 10: R-D PSNR (8 codec, per confronto classico)
# ====================================================================
def fig10(dfq):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for codec in CODEC_ORDER_IMG:
        s = dfq[dfq["codec"] == codec].dropna(subset=["psnr"])
        if len(s) == 0:
            continue
        a = (
            s.groupby("param")
            .agg(bpp=("bpp", "mean"), psnr=("psnr", "mean"))
            .reset_index()
            .sort_values("bpp")
        )
        ax.plot(
            a["bpp"],
            a["psnr"],
            color=C[codec],
            marker=M.get(codec, "o"),
            label=codec,
            zorder=5,
        )

    ax.set_xlabel("Rate (bpp)")
    ax.set_ylabel("PSNR (dB) ↑")
    ax.set_title("Rate-Distortion su Kodak (24 immagini)")
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(25, 45)

    fig.savefig(os.path.join(OUT, "fig10_rd_psnr.pdf"))
    plt.close()
    print("  fig10_rd_psnr.pdf")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("GENERAZIONE GRAFICI R-D-E TESI")
    print("=" * 60)

    print("\nCaricamento dati...")
    df_rde = load_rde()
    print(f"  R-D-E: {len(df_rde)} righe, codec: {sorted(df_rde['codec'].unique())}")
    dfq = load_quality()
    print(f"  Quality: {len(dfq)} righe, codec: {sorted(dfq['codec'].unique())}")
    aud = load_audio()
    print(f"  Audio: visqol_sp={len(aud['vsp'])}, visqol_au={len(aud['vau'])}")

    print("\nGenerazione grafici:")
    fig1(df_rde)  # Bubble R-D-E
    fig2(dfq)  # SSIMULACRA2 vs bpp
    fig3(df_rde)  # PSNR vs Energy
    fig4(df_rde)  # Energy bar
    fig5()  # Bottleneck
    fig6(aud)  # Audio ViSQOL dual
    fig7(aud)  # Audio ViSQOL per dataset
    fig8()  # Audio energy
    fig9()  # Cross-domain
    fig10(dfq)  # R-D PSNR classico

    n = len([f for f in os.listdir(OUT) if f.endswith(".pdf")])
    print(f"\nTutti i grafici in: {OUT}/")
    print(f"Totale: {n} PDF")


if __name__ == "__main__":
    main()
