"""
Grafici R-D-E definitivi per la tesi.
Adattato per la metodologia BATCH: gestisce correttamente le medie
e i tempi totali scalati per numero di immagini e ripetizioni.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
CODEC_ORDER = ["JPEG", "JXL", "HEVC", "Ballé", "Cheng", "ELIC", "TCM", "DCAE"]


# ====================================================================
# CARICAMENTO DATI
# ====================================================================
def load_quality():
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
    cols = ["codec", "param", "image", "bpp", "psnr", "ssimulacra2", "lpips"]
    merged = [d[[c for c in cols if c in d.columns]] for d in dfs]
    return pd.concat(merged, ignore_index=True)


def load_energy():
    return pd.read_csv(
        os.path.expanduser("~/tesi/results/images/full_pipeline_energy_benchmark.csv")
    )


def load_rde(dfq, dfe):
    """
    FIX POTENTE 1:
    dfq è per-immagine, dfe è per-batch.
    Dobbiamo fare la media di dfq PRIMA di unire, e non usare la colonna "image".
    """
    # 1. Media delle metriche di qualità
    q_avg = (
        dfq.groupby(["codec", "param"])
        .agg({"bpp": "mean", "psnr": "mean", "ssimulacra2": "mean"})
        .reset_index()
    )

    # 2. Estrai l'energia per immagine dal file batch
    e = dfe[["codec", "param", "energy_per_image_j"]].copy()
    e = e.rename(columns={"energy_per_image_j": "energy"})

    # 3. Unione pulita
    merged = q_avg.merge(e, on=["codec", "param"], how="inner")
    return merged.dropna(subset=["bpp", "psnr", "energy"])


def load_audio():
    d = {}
    d["vsp"] = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/visqol_benchmark.csv")
    )
    d["vau"] = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/visqol_audio_mode_benchmark.csv")
    )
    # FIX: Punta al nuovo file batch
    d["energy"] = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/audio_energy_rigorous_batch.csv")
    )
    return d


# ====================================================================
# HELPERS
# ====================================================================
def avg_rde(df_rde, codec):
    # df_rde è già una media, basta filtrare
    return df_rde[df_rde["codec"] == codec].sort_values("bpp")


def avg_metric(dfq, codec, metric):
    s = dfq[dfq["codec"] == codec].dropna(subset=[metric])
    if len(s) == 0:
        return pd.DataFrame()
    return (
        s.groupby("param")
        .agg(bpp=("bpp", "mean"), val=(metric, "mean"))
        .reset_index()
        .sort_values("bpp")
    )


# ====================================================================
# FIG 1: Bubble R-D-E
# ====================================================================
def fig1(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    for codec in CODEC_ORDER:
        s = avg_rde(df, codec)
        if len(s) == 0:
            continue
        ax.plot(s["bpp"], s["psnr"], color=C[codec], alpha=0.4, linewidth=1, zorder=3)
        sizes = np.sqrt(np.maximum(s["energy"].values, 0.01) / 0.01) * 25
        ax.scatter(
            s["bpp"],
            s["psnr"],
            s=sizes,
            c=C[codec],
            marker=M[codec],
            edgecolors="black" if M[codec] != "x" else C[codec],
            linewidth=0.5,
            zorder=5,
            alpha=0.85,
            label=codec,
        )

    ax.set_xlabel("Rate — Bitrate (bpp)")
    ax.set_ylabel("Distortion — PSNR (dB) ↑")
    ax.set_title(
        "Framework R-D-E: Rate vs Distortion vs Energy\n(dimensione bolla ∝ √energia netta, full pipeline)"
    )
    ax.set_xlim(0, 2.0)
    ax.set_ylim(25, 45)

    handles = [
        Line2D(
            [0], [0], color=C[c], marker=M[c], linestyle="none", markersize=8, label=c
        )
        for c in CODEC_ORDER
        if c in df["codec"].unique()
    ]
    leg1 = ax.legend(handles=handles, loc="lower right", title="Codec", ncol=2)
    ax.add_artist(leg1)

    for e_ref in [0.04, 1, 10, 150]:
        ax.scatter(
            [],
            [],
            s=np.sqrt(e_ref / 0.01) * 25,
            c="gray",
            alpha=0.4,
            edgecolors="black",
            linewidth=0.5,
            label=f"{e_ref:.2f} J" if e_ref < 1 else f"{e_ref:.0f} J",
        )
    ax.legend(
        loc="upper right", title="Energia netta (J)", labelspacing=1.2, framealpha=0.9
    )

    fig.savefig(os.path.join(OUT, "fig1_rde_bubble.pdf"))
    plt.close()
    print("  fig1_rde_bubble.pdf")


# ====================================================================
# FIG 2: SSIMULACRA2 vs bpp
# ====================================================================
def fig2(dfq):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for codec in CODEC_ORDER:
        s = avg_metric(dfq, codec, "ssimulacra2")
        if len(s) == 0:
            continue
        lw = 2.8 if codec == "JXL" else 1.8
        ax.plot(
            s["bpp"],
            s["val"],
            color=C[codec],
            marker=M.get(codec, "o"),
            label=codec,
            zorder=5,
            linewidth=lw,
        )

    ax.set_xlabel("Rate (bpp)")
    ax.set_ylabel("SSIMULACRA2 (↑ meglio)")
    ax.set_title("Qualità psicovisiva (Butteraugli/SSIMULACRA2) vs Bitrate")
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlim(0, 2.0)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    fig.savefig(os.path.join(OUT, "fig2_rd_ssimulacra2.pdf"))
    plt.close()
    print("  fig2_rd_ssimulacra2.pdf")


# ====================================================================
# FIG 3: PSNR vs Energia
# ====================================================================
def fig3(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for codec in CODEC_ORDER:
        s = avg_rde(df, codec)
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

    ax.set_xlabel("Energia netta per immagine (J) →")
    ax.set_ylabel("PSNR (dB) ↑")
    ax.set_title("Trade-off Qualità vs Energia (full pipeline, idle sottratto)")
    ax.set_xscale("log")
    ax.legend(loc="lower right", ncol=2)

    fig.savefig(os.path.join(OUT, "fig3_psnr_vs_energy.pdf"))
    plt.close()
    print("  fig3_psnr_vs_energy.pdf")


# ====================================================================
# FIG 4: Energy bar chart immagini
# ====================================================================
def fig4(df):
    targets_def = [
        ("JPEG", "q=85"),
        ("JXL", "d=1.0"),
        ("HEVC", "crf=25"),
        ("Ballé", "q=5"),
        ("Cheng", "q=5"),
        ("ELIC", "lam=0.150"),
        ("TCM", "lam=0.013"),
        ("DCAE", "lam=0.013"),
    ]
    targets = []
    for codec, param in targets_def:
        s = df[(df["codec"] == codec) & (df["param"] == param)]
        if len(s) > 0:
            targets.append((codec, param, s["psnr"].values[0], s["energy"].values[0]))
    targets.sort(key=lambda x: x[3])

    labels = [f"{t[0]}\n{t[1]}\n({t[2]:.0f} dB)" for t in targets]
    energies = [t[3] for t in targets]
    colors = [C[t[0]] for t in targets]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(targets)),
        energies,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energia netta per Immagine (J)")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)
    ax.set_title("Costo energetico per codec immagini (full pipeline, idle sottratto)")

    for i, v in enumerate(energies):
        label = f"{v:.2f}" if v < 1 else f"{v:.0f}"
        ax.text(i, v * 1.4, f"{label} J", ha="center", fontsize=8, fontweight="bold")

    fig.savefig(os.path.join(OUT, "fig4_energy_bar.pdf"))
    plt.close()
    print("  fig4_energy_bar.pdf")


# ====================================================================
# FIG 5: Bottleneck
# ====================================================================
def fig5(dfe):
    codecs_info = [
        ("Ballé", "q=5", 3),
        ("ELIC", "lam=0.150", 15),
        ("TCM", "lam=0.013", 39),
        ("DCAE", "lam=0.013", 91),
        ("Cheng", "q=5", 15),
    ]
    codecs_data = []
    for codec, param, fwd_ms in codecs_info:
        s = dfe[(dfe["codec"] == codec) & (dfe["param"] == param)]
        if len(s) > 0:
            # FIX POTENTE 2: Converti il tempo batch in ms per singola immagine
            n_img = s["n_images"].values[0]
            n_rep = s["n_repeats"].values[0]
            t_tot_s = s["total_time_s"].values[0]
            time_per_img_ms = (t_tot_s * 1000) / (n_img * n_rep)
            codecs_data.append((codec, fwd_ms, time_per_img_ms))

    labels = [c[0] for c in codecs_data]
    fwd = [c[1] for c in codecs_data]
    full = [c[2] for c in codecs_data]
    arith = [f - fw for f, fw in zip(full, fwd)]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - w / 2,
        fwd,
        w,
        label="Forward pass (GPU)",
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
        label="Codifica aritmetica (CPU)",
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    for i in range(len(labels)):
        ax.text(
            x[i] + w / 2,
            full[i] * 1.08,
            f"{full[i]:.0f} ms",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
        if arith[i] > 5:
            pct = arith[i] / full[i] * 100
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
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tempo (ms per immagine)")
    ax.set_title("Forward pass vs Codifica aritmetica")
    ax.set_yscale("log")
    ax.set_ylim(1, 8000)
    ax.legend(loc="upper left")

    fig.savefig(os.path.join(OUT, "fig5_bottleneck.pdf"))
    plt.close()
    print("  fig5_bottleneck.pdf")


# ====================================================================
# FIG 6: Audio ViSQOL dual mode
# ====================================================================
def fig6(aud):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for codec in ["Opus", "EnCodec", "DAC", "SNAC", "WavTokenizer"]:
        for ax, dkey, vcol in [(a1, "vsp", "visqol"), (a2, "vau", "visqol_audio")]:
            s = aud[dkey][aud[dkey]["codec"] == codec].copy()
            s["kbps"] = pd.to_numeric(s["param"], errors="coerce")
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
        ("SNAC", 0.8),
        ("WavTokenizer", 0.9),
        ("EnCodec", 1.5),
        ("EnCodec", 6.0),
        ("DAC", 8.0),
        ("Opus", 12),
        ("Opus", 48),
    ]
    datasets = ["librispeech", "esc50", "musdb"]
    ds_labels = ["LibriSpeech", "ESC-50", "MUSDB18"]
    ds_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(codec_order))
    w = 0.22

    for i, (ds, dl, dc) in enumerate(zip(datasets, ds_labels, ds_colors)):
        vals = []
        for codec, param_f in codec_order:
            s = va[(va["codec"] == codec) & (va["dataset"] == ds)].copy()
            s["kbps"] = pd.to_numeric(s["param"], errors="coerce")
            s = s[np.isclose(s["kbps"], param_f, rtol=0.05)]
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
# FIG 8: Audio energy (dal CSV rigoroso batch)
# ====================================================================
def fig8(aud):
    ae = aud["energy"]
    codec_order = [
        ("Opus", 24),
        ("Opus", 12),
        ("Opus", 48),
        ("SNAC", 0.8),
        ("WavTokenizer", 0.9),
        ("EnCodec", 3.0),
        ("EnCodec", 1.5),
        ("DAC", 8.0),
    ]

    labels, j_per_s, colors = [], [], []
    for codec, param in codec_order:
        # FIX POTENTE 3: Converte tutto in float sicuro per matchare "12" con "12.0"
        s = ae[
            (ae["codec"] == codec)
            & (pd.to_numeric(ae["param"], errors="coerce") == float(param))
        ]
        if len(s) == 0:
            continue

        # FIX POTENTE 4: Il nuovo CSV ha già j_per_s
        val = s["j_per_s"].values[0]
        labels.append(f"{codec}\n{param} kbps")
        j_per_s.append(val)
        colors.append(C[codec])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(
        range(len(labels)),
        j_per_s,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    for i, v in enumerate(j_per_s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energia netta (J / s audio)")
    ax.set_title("Consumo energetico audio (idle sottratto, RAPL+Zeus batch)")
    ax.set_ylim(0, 1.4)

    fig.savefig(os.path.join(OUT, "fig8_energy_audio.pdf"))
    plt.close()
    print("  fig8_energy_audio.pdf")


# ====================================================================
# FIG 9: Cross-domain
# ====================================================================
def fig9():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    a1.bar(
        ["Immagini\n(Cheng/JPEG)"],
        [3980],
        color="#d62728",
        edgecolor="black",
        alpha=0.85,
    )

    # Aggiornato con il nuovo gap misurato (14.5x)
    a1.bar(
        ["Audio\n(DAC/Opus)"], [14.5], color="#2ca02c", edgecolor="black", alpha=0.85
    )

    a1.set_ylabel("Divario energetico massimo (×)")
    a1.set_title("Gap energetico: classico ↔ neurale")
    a1.set_yscale("log")
    a1.set_ylim(1, 10000)
    a1.text(
        0, 5500, "3980×", ha="center", fontsize=14, fontweight="bold", color="#d62728"
    )
    a1.text(
        1, 20, "14.5×", ha="center", fontsize=14, fontweight="bold", color="#2ca02c"
    )

    classical_pct = [72.5 / 72.5 * 100, 3.57 / 3.67 * 100]
    neural_pct = [56.1 / 72.5 * 100, 3.67 / 3.67 * 100]
    x = np.arange(2)
    w = 0.3
    a2.bar(
        x - w / 2,
        classical_pct,
        w,
        label="Classico (JXL / Opus 48)",
        color="#2ca02c",
        edgecolor="black",
    )
    a2.bar(
        x + w / 2,
        neural_pct,
        w,
        label="Neurale SOTA (DCAE / DAC)",
        color="#1f77b4",
        edgecolor="black",
    )

    for xi, cv, nv, cr, nr in [
        (0, classical_pct[0], neural_pct[0], "72.5", "56.1"),
        (1, classical_pct[1], neural_pct[1], "3.57", "3.67"),
    ]:
        a2.text(
            xi - w / 2,
            cv + 2,
            cr,
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#2ca02c",
        )
        a2.text(
            xi + w / 2,
            nv + 2,
            nr,
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1f77b4",
        )

    a2.set_xticks(x)
    a2.set_xticklabels(["Immagini\n(SSIMULACRA2)", "Audio\n(ViSQOL audio)"])
    a2.set_ylabel("% del migliore per dominio")
    a2.set_ylim(0, 115)
    a2.set_title("Vincitore percettivo per dominio")
    a2.legend()
    a2.annotate(
        "Classico vince",
        xy=(0, 108),
        fontsize=9,
        ha="center",
        color="#2ca02c",
        fontweight="bold",
    )
    a2.annotate(
        "Neurale vince",
        xy=(1, 108),
        fontsize=9,
        ha="center",
        color="#1f77b4",
        fontweight="bold",
    )

    fig.suptitle("Asimmetria cross-dominio", fontsize=14)
    plt.subplots_adjust(top=0.85, wspace=0.3)

    fig.savefig(os.path.join(OUT, "fig9_cross_domain.pdf"))
    plt.close()
    print("  fig9_cross_domain.pdf")


# ====================================================================
# FIG 10: R-D PSNR classico
# ====================================================================
def fig10(dfq):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for codec in CODEC_ORDER:
        s = avg_metric(dfq, codec, "psnr")
        if len(s) == 0:
            continue
        ax.plot(
            s["bpp"],
            s["val"],
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
    print("GENERAZIONE GRAFICI R-D-E TESI (definitivi)")
    print("=" * 60)

    print("\nCaricamento dati...")
    dfq = load_quality()
    dfe = load_energy()
    df_rde = load_rde(dfq, dfe)
    aud = load_audio()

    print(f"  Qualità img: {len(dfq)} righe, codec: {sorted(dfq['codec'].unique())}")
    print(f"  Energia img: {len(dfe)} righe, codec: {sorted(dfe['codec'].unique())}")
    print(f"  R-D-E merge: {len(df_rde)} righe aggregate")
    print(f"  Audio energy: {len(aud['energy'])} righe")

    print("\nGenerazione grafici:")
    fig1(df_rde)
    fig2(dfq)
    fig3(df_rde)
    fig4(df_rde)
    fig5(dfe)
    fig6(aud)
    fig7(aud)
    fig8(aud)
    fig9()
    fig10(dfq)

    n = len([f for f in os.listdir(OUT) if f.endswith(".pdf")])
    print(f"\nTutti i grafici in: {OUT}/")
    print(f"Totale: {n} PDF")


if __name__ == "__main__":
    main()
