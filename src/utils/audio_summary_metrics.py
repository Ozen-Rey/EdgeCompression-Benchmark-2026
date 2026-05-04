from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "results" / "audio"

FULL_METRICS = AUDIO_DIR / "audio_benchmark_full.csv"
WAVTOKENIZER_METRICS = AUDIO_DIR / "wavtokenizer_metrics.csv"
VISQOL_SPEECH = AUDIO_DIR / "visqol_benchmark.csv"
VISQOL_AUDIO = AUDIO_DIR / "visqol_audio_mode_benchmark.csv"
FAD = AUDIO_DIR / "fad_benchmark.csv"
ENERGY = AUDIO_DIR / "audio_energy_rigorous_batch.csv"

OUT_CSV = AUDIO_DIR / "audio_summary_full.csv"
OUT_TEX = AUDIO_DIR / "audio_summary_full.tex"


def load_aux_metrics() -> pd.DataFrame:
    full = pd.read_csv(FULL_METRICS)

    aux = (
        full.groupby(["codec", "param"], as_index=False)
        .agg(
            actual_kbps=("actual_kbps", "mean"),
            pesq=("pesq", "mean"),
            stoi=("stoi", "mean"),
            sdr=("sdr", "mean"),
            mel_dist=("mel_dist", "mean"),
            rtf=("rtf", "mean"),
        )
    )

    if WAVTOKENIZER_METRICS.exists():
        wav = pd.read_csv(WAVTOKENIZER_METRICS)
        wav_aux = (
            wav.groupby(["codec", "param"], as_index=False)
            .agg(
                pesq=("pesq", "mean"),
                stoi=("stoi", "mean"),
                sdr=("sdr", "mean"),
                mel_dist=("mel_dist", "mean"),
            )
        )
        wav_aux["actual_kbps"] = wav_aux["param"]
        wav_aux["rtf"] = pd.NA
        aux = pd.concat([aux, wav_aux], ignore_index=True)

    return aux


def load_main_metrics() -> pd.DataFrame:
    visqol_speech = (
        pd.read_csv(VISQOL_SPEECH)
        .groupby(["codec", "param"], as_index=False)
        .agg(visqol_speech=("visqol", "mean"))
    )

    visqol_audio = (
        pd.read_csv(VISQOL_AUDIO)
        .groupby(["codec", "param"], as_index=False)
        .agg(visqol_audio=("visqol_audio", "mean"))
    )

    fad = pd.read_csv(FAD).rename(columns={"fad_vggish": "fad"})
    energy = pd.read_csv(ENERGY)

    return visqol_speech, visqol_audio, fad, energy


def make_summary() -> pd.DataFrame:
    aux = load_aux_metrics()
    visqol_speech, visqol_audio, fad, energy = load_main_metrics()

    df = aux.merge(visqol_speech, on=["codec", "param"], how="outer")
    df = df.merge(visqol_audio, on=["codec", "param"], how="outer")
    df = df.merge(fad, on=["codec", "param"], how="outer")
    df = df.merge(energy, on=["codec", "param"], how="outer")

    order = [
        ("SNAC", 0.8),
        ("WavTokenizer", 0.9),
        ("EnCodec", 1.5),
        ("EnCodec", 3.0),
        ("EnCodec", 6.0),
        ("DAC", 8.0),
        ("Opus", 12.0),
        ("Opus", 24.0),
        ("Opus", 48.0),
    ]

    order_map = {item: i for i, item in enumerate(order)}
    df["order"] = df.apply(
        lambda r: order_map.get((r["codec"], float(r["param"])), 999),
        axis=1,
    )
    df = df.sort_values("order").drop(columns=["order"])

    return df


def write_latex_table(df: pd.DataFrame) -> None:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Metriche audio ausiliarie aggregate sui campioni valutati. "
        r"Per PESQ, STOI e SDR valori maggiori indicano qualità superiore; "
        r"per Mel distance valori minori indicano maggiore vicinanza allo spettrogramma originale.}"
    )
    lines.append(r"    \label{tab:audio_aux_metrics}")
    lines.append(r"    \small")
    lines.append(r"    \begin{tabular}{lrrrrr}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textit{Codec} & kbps & PESQ $\uparrow$ & STOI $\uparrow$ & SDR $\uparrow$ & Mel dist. $\downarrow$ \\"
    )
    lines.append(r"        \midrule")

    for _, r in df.iterrows():
        codec = str(r["codec"])
        kbps = float(r["param"])
        pesq = float(r["pesq"])
        stoi = float(r["stoi"])
        sdr = float(r["sdr"])
        mel = float(r["mel_dist"])

        lines.append(
            f"        {codec:<13} & {kbps:.1f} & {pesq:.3f} & {stoi:.3f} & {sdr:.2f} & {mel:.3f} \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = make_summary()

    cols = [
        "codec",
        "param",
        "actual_kbps",
        "visqol_speech",
        "visqol_audio",
        "fad",
        "j_per_s",
        "pesq",
        "stoi",
        "sdr",
        "mel_dist",
        "rtf",
    ]

    available_cols = [c for c in cols if c in df.columns]
    df[available_cols].to_csv(OUT_CSV, index=False)
    write_latex_table(df)

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TEX}")


if __name__ == "__main__":
    main()