"""
Grafici publication-quality per la tesi.
Genera PDF per inclusione diretta in LaTeX.

Cap. 6: Immagini (JXL, ELIC, Ballé, Cheng, TCM, DCAE)
Cap. 7: Audio (Opus, EnCodec, DAC, SNAC, WavTokenizer)
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ================== STYLE ==================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

OUT_DIR = os.path.expanduser("~/tesi/figures")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "JXL": "#2ca02c",
    "ELIC": "#1f77b4",
    "Ballé": "#ff7f0e",
    "Cheng": "#d62728",
    "TCM": "#9467bd",
    "DCAE": "#8c564b",
    "Opus": "#2ca02c",
    "EnCodec": "#1f77b4",
    "DAC": "#d62728",
    "SNAC": "#ff7f0e",
    "WavTokenizer": "#9467bd",
}

MARKERS = {
    "JXL": "s",
    "ELIC": "o",
    "Ballé": "^",
    "Cheng": "v",
    "TCM": "D",
    "DCAE": "P",
    "Opus": "s",
    "EnCodec": "o",
    "DAC": "D",
    "SNAC": "^",
    "WavTokenizer": "P",
}


# ====================================================================
# CARICAMENTO DATI
# ====================================================================
def load_data():
    data = {}

    # JXL vs ELIC (12 metriche, per-image)
    jxl_elic = pd.read_csv(
        os.path.expanduser("~/tesi/results/jxl_vs_elic_benchmark.csv")
    )
    data["jxl_elic"] = jxl_elic

    # SOTA 12 metriche (TCM + DCAE, per-image)
    sota_12 = pd.read_csv(
        os.path.expanduser("~/tesi/results/images/sota_12metric_benchmark.csv")
    )
    data["sota_12"] = sota_12

    # CPU energy (JPEG, JXL, HEVC)
    cpu_energy = pd.read_csv(
        os.path.expanduser("~/tesi/results/cpu_energy_rapl_benchmark.csv")
    )
    data["cpu_energy"] = cpu_energy

    # Audio full (PESQ, STOI, SDR, mel, RTF)
    audio_full = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/audio_benchmark_full.csv")
    )
    data["audio_full"] = audio_full

    # WavTokenizer metrics
    wt_metrics = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/wavtokenizer_metrics.csv")
    )
    data["wt_metrics"] = wt_metrics

    # ViSQOL speech
    visqol_sp = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/visqol_benchmark.csv")
    )
    data["visqol_speech"] = visqol_sp

    # ViSQOL audio
    visqol_au = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/visqol_audio_mode_benchmark.csv")
    )
    data["visqol_audio"] = visqol_au

    # FAD
    fad = pd.read_csv(os.path.expanduser("~/tesi/results/audio/fad_benchmark.csv"))
    data["fad"] = fad

    # Energy batch
    energy_batch = pd.read_csv(
        os.path.expanduser("~/tesi/results/audio/energy_batch_benchmark.csv")
    )
    data["energy_batch"] = energy_batch

    return data


# ====================================================================
# FIG 1: R-D Curve (bpp vs PSNR) — tutti i codec immagini
# ====================================================================
def fig1_rd_curve(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # JXL
    jxl = data["jxl_elic"][data["jxl_elic"]["codec"] == "JXL"]
    jxl_avg = jxl.groupby("param")[["bpp", "psnr"]].mean().sort_values("bpp")
    ax.plot(
        jxl_avg["bpp"],
        jxl_avg["psnr"],
        color=COLORS["JXL"],
        marker=MARKERS["JXL"],
        label="JXL (2021)",
        zorder=5,
    )

    # ELIC forward
    elic_fwd = data["jxl_elic"][
        (data["jxl_elic"]["codec"] == "ELIC")
        & (data["jxl_elic"]["pipeline"] == "forward_pass_only")
    ]
    elic_avg = elic_fwd.groupby("param")[["bpp", "psnr"]].mean().sort_values("bpp")
    ax.plot(
        elic_avg["bpp"],
        elic_avg["psnr"],
        color=COLORS["ELIC"],
        marker=MARKERS["ELIC"],
        label="ELIC (2022)",
        zorder=5,
    )

    # TCM
    tcm = data["sota_12"][data["sota_12"]["codec"] == "TCM"]
    tcm_avg = tcm.groupby("param")[["bpp", "psnr"]].mean().sort_values("bpp")
    ax.plot(
        tcm_avg["bpp"],
        tcm_avg["psnr"],
        color=COLORS["TCM"],
        marker=MARKERS["TCM"],
        label="TCM (CVPR 2023)",
        zorder=5,
    )

    # DCAE
    dcae = data["sota_12"][data["sota_12"]["codec"] == "DCAE"]
    dcae_avg = dcae.groupby("param")[["bpp", "psnr"]].mean().sort_values("bpp")
    ax.plot(
        dcae_avg["bpp"],
        dcae_avg["psnr"],
        color=COLORS["DCAE"],
        marker=MARKERS["DCAE"],
        label="DCAE (CVPR 2025)",
        zorder=5,
    )

    ax.set_xlabel("Bitrate (bpp)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate-Distortion: Kodak (24 immagini)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.6)

    path = os.path.join(OUT_DIR, "fig1_rd_curve_images.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 2: bpp vs SSIMULACRA2 — il grafico killer
# ====================================================================
def fig2_ssimulacra2(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # JXL
    jxl = data["jxl_elic"][data["jxl_elic"]["codec"] == "JXL"]
    jxl_avg = jxl.groupby("param")[["bpp", "ssimulacra2"]].mean().sort_values("bpp")
    ax.plot(
        jxl_avg["bpp"],
        jxl_avg["ssimulacra2"],
        color=COLORS["JXL"],
        marker=MARKERS["JXL"],
        label="JXL (2021)",
        zorder=5,
        linewidth=2.5,
    )

    # ELIC forward
    elic = data["jxl_elic"][
        (data["jxl_elic"]["codec"] == "ELIC")
        & (data["jxl_elic"]["pipeline"] == "forward_pass_only")
    ]
    elic_avg = elic.groupby("param")[["bpp", "ssimulacra2"]].mean().sort_values("bpp")
    ax.plot(
        elic_avg["bpp"],
        elic_avg["ssimulacra2"],
        color=COLORS["ELIC"],
        marker=MARKERS["ELIC"],
        label="ELIC (2022)",
        zorder=5,
    )

    # TCM
    tcm = data["sota_12"][data["sota_12"]["codec"] == "TCM"]
    tcm_avg = tcm.groupby("param")[["bpp", "ssimulacra2"]].mean().sort_values("bpp")
    ax.plot(
        tcm_avg["bpp"],
        tcm_avg["ssimulacra2"],
        color=COLORS["TCM"],
        marker=MARKERS["TCM"],
        label="TCM (CVPR 2023)",
        zorder=5,
    )

    # DCAE
    dcae = data["sota_12"][data["sota_12"]["codec"] == "DCAE"]
    dcae_avg = dcae.groupby("param")[["bpp", "ssimulacra2"]].mean().sort_values("bpp")
    ax.plot(
        dcae_avg["bpp"],
        dcae_avg["ssimulacra2"],
        color=COLORS["DCAE"],
        marker=MARKERS["DCAE"],
        label="DCAE (CVPR 2025)",
        zorder=5,
    )

    ax.set_xlabel("Bitrate (bpp)")
    ax.set_ylabel("SSIMULACRA2 (↑)")
    ax.set_title("Qualità psicovisiva: JXL domina anche contro SOTA 2025")
    ax.legend(loc="lower right")

    path = os.path.join(OUT_DIR, "fig2_ssimulacra2_vs_bpp.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 3: Radar 12 metriche a ~34 dB (JXL d=3, ELIC lam=0.15, TCM, DCAE)
# ====================================================================
def fig3_radar(data):
    # Metriche da confrontare (normalizzate: più alto = meglio)
    metrics = [
        "psnr",
        "ssim",
        "ms_ssim",
        "fsim",
        "vif",
        "haarpsi",
        "dss",
        "ssimulacra2",
    ]
    # Per LPIPS, DISTS, GMSD: invertiamo (più basso = meglio)
    inv_metrics = ["lpips", "dists", "gmsd"]

    # Raccogli medie
    codecs_data = {}

    # JXL d=3.0
    jxl = data["jxl_elic"][
        (data["jxl_elic"]["codec"] == "JXL") & (data["jxl_elic"]["param"] == "d=3.0")
    ]
    codecs_data["JXL d=3.0"] = jxl[metrics + inv_metrics].mean()

    # ELIC lam=0.150
    elic = data["jxl_elic"][
        (data["jxl_elic"]["codec"] == "ELIC")
        & (data["jxl_elic"]["param"] == "lam=0.150")
        & (data["jxl_elic"]["pipeline"] == "forward_pass_only")
    ]
    codecs_data["ELIC λ=0.15"] = elic[metrics + inv_metrics].mean()

    # TCM lam=0.013
    tcm = data["sota_12"][
        (data["sota_12"]["codec"] == "TCM") & (data["sota_12"]["param"] == "lam=0.013")
    ]
    codecs_data["TCM λ=0.013"] = tcm[metrics + inv_metrics].mean()

    # DCAE lam=0.013
    dcae = data["sota_12"][
        (data["sota_12"]["codec"] == "DCAE") & (data["sota_12"]["param"] == "lam=0.013")
    ]
    codecs_data["DCAE λ=0.013"] = dcae[metrics + inv_metrics].mean()

    # Normalizza tutto in [0,1] per il radar
    all_labels = metrics + inv_metrics
    mins = {m: min(cd[m] for cd in codecs_data.values()) for m in all_labels}
    maxs = {m: max(cd[m] for cd in codecs_data.values()) for m in all_labels}

    def normalize(val, m):
        rng = maxs[m] - mins[m]
        if rng < 1e-10:
            return 0.5
        norm = (val - mins[m]) / rng
        # Inverti per metriche dove basso è meglio
        if m in inv_metrics:
            norm = 1 - norm
        return norm

    # Plot
    labels = [
        "PSNR",
        "SSIM",
        "MS-SSIM",
        "FSIM",
        "VIF",
        "HaarPSI",
        "DSS",
        "SSIM2",
        "1-LPIPS",
        "1-DISTS",
        "1-GMSD",
    ]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))

    colors_radar = ["#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    for idx, (name, vals) in enumerate(codecs_data.items()):
        values = [normalize(vals[m], m) for m in all_labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.08, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Confronto 11 metriche percettive\n(normalizzate, ↑ = meglio)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = os.path.join(OUT_DIR, "fig3_radar_12metrics.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 4: Energy bar chart — immagini (CPU RAPL + GPU Zeus)
# ====================================================================
def fig4_energy_images(data):
    # Dati manuali aggregati (dalla tesi)
    codecs = [
        "JPEG\nq=60",
        "JXL\nd=3.0",
        "HEVC\ncrf=35",
        "Ballé\nλ=0.15",
        "ELIC\nλ=0.15",
        "TCM\nλ=0.013",
        "DCAE\nλ=0.013",
        "Cheng\nλ=0.15",
    ]
    energy_j = [0.05, 1.11, 5.72, 1.67, 5.56, 12.88, 19.02, 185.0]
    colors_bar = [
        "#2ca02c",
        "#2ca02c",
        "#2ca02c",
        "#1f77b4",
        "#1f77b4",
        "#9467bd",
        "#8c564b",
        "#d62728",
    ]
    hatches = ["//", "//", "//", "", "", "", "", ""]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        range(len(codecs)),
        energy_j,
        color=colors_bar,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(range(len(codecs)))
    ax.set_xticklabels(codecs, fontsize=9)
    ax.set_ylabel("Energia per immagine (J)")
    ax.set_title("Consumo energetico per codec (Kodak 768×512)")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 500)

    # Annotazioni
    for i, v in enumerate(energy_j):
        ax.text(
            i,
            v * 1.15,
            f"{v:.1f}" if v >= 1 else f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Legenda
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#2ca02c", hatch="//", edgecolor="black", label="Classici (CPU)"
        ),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Neurali (GPU)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    path = os.path.join(OUT_DIR, "fig4_energy_images.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 5: Arithmetic coding bottleneck
# ====================================================================
def fig5_bottleneck(data):
    codecs = ["Ballé\n(5.1M)", "ELIC\n(36.9M)", "Cheng\n(26.6M)"]
    fwd_ms = [3, 15, 15]
    full_ms = [56, 134, 3536]
    arith_ms = [53, 119, 3521]

    x = np.arange(len(codecs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar(
        x - w / 2,
        fwd_ms,
        w,
        label="Forward pass",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x + w / 2,
        arith_ms,
        w,
        bottom=np.array(fwd_ms),
        label="Codifica aritmetica",
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    b3 = ax.bar(
        x + w / 2,
        fwd_ms,
        w,
        label="Forward (in pipeline)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )

    # Annotazioni totali
    for i in range(len(codecs)):
        ax.text(
            x[i] + w / 2,
            full_ms[i] + 50,
            f"{full_ms[i]} ms",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
        pct = arith_ms[i] / full_ms[i] * 100
        ax.text(
            x[i] + w / 2,
            fwd_ms[i] + arith_ms[i] / 2,
            f"{pct:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(codecs)
    ax.set_ylabel("Tempo (ms)")
    ax.set_title("Collo di bottiglia: codifica aritmetica")
    ax.set_yscale("log")
    ax.set_ylim(1, 8000)
    ax.legend(loc="upper left")

    path = os.path.join(OUT_DIR, "fig5_bottleneck.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 6: Audio — kbps vs ViSQOL (speech + audio mode)
# ====================================================================
def fig6_audio_visqol(data):
    # Aggrega per codec
    vspeech = (
        data["visqol_speech"].groupby(["codec", "param"])["visqol"].mean().reset_index()
    )
    vaudio = (
        data["visqol_audio"]
        .groupby(["codec", "param"])["visqol_audio"]
        .mean()
        .reset_index()
    )

    kbps_map = {
        "12": 12,
        "24": 24,
        "48": 48,
        "1.5": 1.5,
        "3.0": 3.0,
        "6.0": 6.0,
        "8.0": 8.0,
        "0.8": 0.8,
        "0.9": 0.9,
        12: 12,
        24: 24,
        48: 48,
        1.5: 1.5,
        3.0: 3.0,
        6.0: 6.0,
        8.0: 8.0,
        0.8: 0.8,
        0.9: 0.9,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for codec_name in ["Opus", "EnCodec", "DAC", "SNAC", "WavTokenizer"]:
        # Speech
        sub = vspeech[vspeech["codec"] == codec_name].copy()
        sub["kbps"] = sub["param"].astype(str).map(kbps_map)
        sub = sub.dropna(subset=["kbps"]).sort_values("kbps")
        if len(sub) > 0:
            ax1.plot(
                sub["kbps"],
                sub["visqol"],
                color=COLORS.get(codec_name, "gray"),
                marker=MARKERS.get(codec_name, "o"),
                label=codec_name,
            )

        # Audio
        sub2 = vaudio[vaudio["codec"] == codec_name].copy()
        sub2["kbps"] = sub2["param"].astype(str).map(kbps_map)
        sub2 = sub2.dropna(subset=["kbps"]).sort_values("kbps")
        if len(sub2) > 0:
            ax2.plot(
                sub2["kbps"],
                sub2["visqol_audio"],
                color=COLORS.get(codec_name, "gray"),
                marker=MARKERS.get(codec_name, "o"),
                label=codec_name,
            )

    ax1.set_xlabel("Bitrate (kbps)")
    ax1.set_ylabel("ViSQOL MOS-LQO (↑)")
    ax1.set_title("ViSQOL Speech Mode (16 kHz)")
    ax1.legend(fontsize=8)
    try:
        ax1.set_xscale("log")
    except ValueError:
        pass

    ax2.set_xlabel("Bitrate (kbps)")
    ax2.set_title("ViSQOL Audio Mode (48 kHz)")
    ax2.legend(fontsize=8)
    try:
        ax2.set_xscale("log")
    except ValueError:
        pass

    fig.suptitle("Qualità percettiva audio: bitrate vs ViSQOL", fontsize=13)
    plt.subplots_adjust(top=0.88, wspace=0.15)

    path = os.path.join(OUT_DIR, "fig6_audio_visqol.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 7: Audio — ViSQOL per dataset (grouped bar)
# ====================================================================
def fig7_visqol_per_dataset(data):
    vaudio = data["visqol_audio"]
    vaudio["tag"] = vaudio["codec"] + " " + vaudio["param"].astype(str)

    codecs_order = [
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

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(codecs_order))
    width = 0.25

    for i, (ds, ds_label) in enumerate(zip(datasets, ds_labels)):
        vals = []
        for codec, param in codecs_order:
            sub = vaudio[
                (vaudio["codec"] == codec)
                & (vaudio["param"].astype(str) == str(param))
                & (vaudio["dataset"] == ds)
            ]
            vals.append(sub["visqol_audio"].mean() if len(sub) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=ds_label, alpha=0.85)

    labels = [f"{c}\n{p}kbps" for c, p in codecs_order]
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ViSQOL Audio Mode (↑)")
    ax.set_title("Dipendenza dal contenuto: ViSQOL per dataset")
    ax.legend()

    path = os.path.join(OUT_DIR, "fig7_visqol_per_dataset.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 8: Audio energy bar chart
# ====================================================================
def fig8_energy_audio(data):
    codecs = [
        "SNAC\n0.8 kbps",
        "Opus\n24 kbps",
        "EnCodec\n3.0 kbps",
        "WavTok\n0.9 kbps",
        "DAC\n8.0 kbps",
    ]
    j_per_s = [0.32, 0.35, 0.49, 0.51, 1.40]
    colors_bar = [
        COLORS["SNAC"],
        COLORS["Opus"],
        COLORS["EnCodec"],
        COLORS["WavTokenizer"],
        COLORS["DAC"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        range(len(codecs)),
        j_per_s,
        color=colors_bar,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    for i, v in enumerate(j_per_s):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(codecs)))
    ax.set_xticklabels(codecs, fontsize=9)
    ax.set_ylabel("Energia (J/s di audio)")
    ax.set_title(
        "Consumo energetico audio — SNAC neurale più economico di Opus classico"
    )
    ax.set_ylim(0, 1.7)

    # Linea Opus come riferimento
    ax.axhline(y=0.35, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(4.4, 0.37, "Opus", fontsize=8, color="gray")

    path = os.path.join(OUT_DIR, "fig8_energy_audio.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 9: Cross-domain asymmetry
# ====================================================================
def fig9_cross_domain(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Energy gap
    domains = ["Immagini", "Audio"]
    gaps = [3700, 4.4]
    colors_d = ["#d62728", "#2ca02c"]
    ax1.bar(domains, gaps, color=colors_d, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax1.set_ylabel("Divario energetico max (×)")
    ax1.set_title("Divario energetico: classico vs neurale")
    ax1.set_yscale("log")
    ax1.set_ylim(1, 10000)
    for i, v in enumerate(gaps):
        ax1.text(
            i,
            v * 1.3,
            f"{v:.0f}×" if v > 10 else f"{v:.1f}×",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # Right: Perceptual winner
    categories = ["Immagini\n(SSIMULACRA2)", "Audio\n(ViSQOL)"]
    classical = [72.5, 3.57]  # JXL ssim2, Opus 48 visqol_au
    neural = [56.1, 3.67]  # DCAE ssim2, DAC visqol_au
    x = np.arange(len(categories))
    w = 0.3
    ax2.bar(
        x - w / 2,
        classical,
        w,
        label="Classico (JXL / Opus)",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.bar(
        x + w / 2,
        neural,
        w,
        label="Neurale SOTA (DCAE / DAC)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Score percettivo (↑)")
    ax2.set_title("Vincitore percettivo per dominio")
    ax2.legend()

    fig.suptitle("Asimmetria cross-dominio", fontsize=14)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "fig9_cross_domain.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# FIG 10: Audio PESQ vs kbps
# ====================================================================
def fig10_audio_pesq(data):
    af = data["audio_full"]
    wt = data["wt_metrics"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for codec_name in ["Opus", "EnCodec", "DAC", "SNAC"]:
        sub = (
            af[af["codec"] == codec_name]
            .groupby("param")
            .agg(pesq=("pesq", "mean"), kbps=("actual_kbps", "mean"))
            .reset_index()
            .sort_values("kbps")
        )
        if len(sub) > 0:
            ax.plot(
                sub["kbps"],
                sub["pesq"],
                color=COLORS[codec_name],
                marker=MARKERS[codec_name],
                label=codec_name,
            )

    # WavTokenizer
    wt_pesq = wt["pesq"].dropna().mean()
    ax.plot(
        0.9,
        wt_pesq,
        color=COLORS["WavTokenizer"],
        marker=MARKERS["WavTokenizer"],
        markersize=10,
        label="WavTokenizer",
        zorder=5,
    )

    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("PESQ (↑)")
    ax.set_title("PESQ vs Bitrate")
    ax.legend(fontsize=8)
    try:
        ax.set_xscale("log")
    except ValueError:
        pass

    path = os.path.join(OUT_DIR, "fig10_audio_pesq.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  {path}")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("Caricamento dati...")
    data = load_data()
    print("Dati caricati.\n")

    print("Generazione grafici:")
    fig1_rd_curve(data)
    fig2_ssimulacra2(data)
    fig3_radar(data)
    fig4_energy_images(data)
    fig5_bottleneck(data)
    fig6_audio_visqol(data)
    fig7_visqol_per_dataset(data)
    fig8_energy_audio(data)
    fig9_cross_domain(data)
    fig10_audio_pesq(data)

    print(f"\nTutti i grafici salvati in: {OUT_DIR}/")
    print(f"Totale: {len(os.listdir(OUT_DIR))} file PDF")


if __name__ == "__main__":
    main()
