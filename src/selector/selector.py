"""
Selettore adattivo del metodo di compressione.
Deriva le regole decisionali empiricamente dai dati del benchmark.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.expanduser("~/tesi/results")


# ── Strutture dati ───────────────────────────────────────────────


@dataclass
class UserConstraints:
    """Vincoli specificati dall'utente."""

    max_bpp: Optional[float] = None  # bitrate massimo
    min_psnr: Optional[float] = None  # qualità minima (matematica)
    max_lpips: Optional[float] = None  # qualità minima (percettiva)
    max_enc_ms: Optional[float] = None  # latenza massima di codifica
    domain: str = "image"  # "image", "audio", "video"


@dataclass
class SelectionResult:
    """Risultato della selezione."""

    codec: str
    param_name: str
    param_value: float
    predicted_bpp: float
    predicted_psnr: float
    predicted_lpips: float
    predicted_enc_ms: float
    reason: str


# ── Helper ───────────────────────────────────────────────────────


from typing import Any, Optional


def _row_to_result(best: Any, reason: str) -> SelectionResult:
    """Converte una riga pandas in SelectionResult con cast espliciti."""
    return SelectionResult(
        codec=str(best["codec"]),
        param_name=str(best["param_name"]),
        param_value=float(best["param_value"]),
        predicted_bpp=float(best["bpp"]),
        predicted_psnr=float(best["psnr"]),
        predicted_lpips=float(best.get("lpips", float("nan"))),
        predicted_enc_ms=float(best["enc_ms"]),
        reason=reason,
    )


# ── Caricamento dati benchmark ───────────────────────────────────


def load_image_data() -> pd.DataFrame:
    """Carica e unifica tutti i dati benchmark immagini."""
    dfs = []

    # JPEG
    df = pd.read_csv(f"{RESULTS_DIR}/images/jpeg_kodak.csv")
    df = (
        df.groupby("quality")[
            ["bpp", "psnr", "ms_ssim", "lpips", "fps", "ms_per_image"]
        ]
        .mean()
        .reset_index()
    )
    df["codec"] = "JPEG"
    df["param_name"] = "quality"
    df.rename(
        columns={"quality": "param_value", "ms_per_image": "enc_ms"}, inplace=True
    )
    dfs.append(df)

    # HEVC
    df = pd.read_csv(f"{RESULTS_DIR}/images/hevc_kodak.csv")
    df = (
        df.groupby("crf")[["bpp", "psnr", "ms_ssim", "lpips", "fps", "ms_per_image"]]
        .mean()
        .reset_index()
    )
    df["codec"] = "HEVC"
    df["param_name"] = "crf"
    df.rename(columns={"crf": "param_value", "ms_per_image": "enc_ms"}, inplace=True)
    dfs.append(df)

    # Ballé2018
    df = pd.read_csv(f"{RESULTS_DIR}/images/balle2018_kodak.csv")
    df = (
        df.groupby("quality")[["bpp", "psnr", "ms_ssim", "lpips", "enc_ms", "dec_ms"]]
        .mean()
        .reset_index()
    )
    df["codec"] = "Balle2018"
    df["param_name"] = "quality"
    df.rename(columns={"quality": "param_value"}, inplace=True)
    dfs.append(df)

    # Cheng2020
    df = pd.read_csv(f"{RESULTS_DIR}/images/cheng2020_kodak.csv")
    df = (
        df.groupby("quality")[["bpp", "psnr", "ms_ssim", "lpips", "enc_ms", "dec_ms"]]
        .mean()
        .reset_index()
    )
    df["codec"] = "Cheng2020"
    df["param_name"] = "quality"
    df.rename(columns={"quality": "param_value"}, inplace=True)
    dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["domain"] = "image"
    return df_all


def load_audio_data() -> pd.DataFrame:
    """Carica e unifica tutti i dati benchmark audio."""
    df = pd.read_csv(f"{RESULTS_DIR}/audio/audio_benchmark.csv")

    result = (
        df.groupby(["codec", "target_kbps"])[
            ["bitrate_kbps", "snr_db", "pesq", "stoi", "si_sdr", "mel_dist", "enc_ms"]
        ]
        .mean()
        .reset_index()
    )

    result.rename(
        columns={
            "target_kbps": "param_value",
            "bitrate_kbps": "bpp",
            "pesq": "psnr",  # PESQ come metrica qualità principale
            "mel_dist": "lpips",  # mel_dist come proxy percettivo (più basso = meglio)
        },
        inplace=True,
    )

    result["param_name"] = "bitrate_kbps"
    result["domain"] = "audio"
    return result


def load_video_data() -> pd.DataFrame:
    """Carica e unifica tutti i dati benchmark video."""
    df = pd.read_csv(f"{RESULTS_DIR}/video/video_benchmark.csv")

    result = (
        df.groupby(["codec", "crf"])[
            ["bpp", "psnr", "ms_ssim", "lpips", "fps_enc", "enc_time_s"]
        ]
        .mean()
        .reset_index()
    )

    result["enc_ms"] = result["enc_time_s"] * 1000 / 120  # ms per frame
    result.rename(columns={"crf": "param_value", "fps_enc": "fps"}, inplace=True)
    result["param_name"] = "crf"
    result["codec"] = result["codec"].str.upper()
    result["domain"] = "video"
    return result


# ── Frontiera di Pareto ──────────────────────────────────────────


def compute_pareto_frontier(
    df: pd.DataFrame, max_enc_ms: Optional[float] = None
) -> pd.DataFrame:
    candidates = df.copy()
    if max_enc_ms is not None:
        candidates = candidates[candidates["enc_ms"] <= max_enc_ms]
    if candidates.empty:
        candidates = df.copy()

    pareto_mask = []

    # usa lpips se disponibile, altrimenti solo 3 dimensioni
    has_lpips = "lpips" in candidates.columns and candidates["lpips"].notna().all()

    if has_lpips:
        records = candidates[["bpp", "psnr", "lpips", "enc_ms"]].values
    else:
        records = candidates[["bpp", "psnr", "enc_ms"]].values

    for i, point in enumerate(records):
        dominated = False
        for j, other in enumerate(records):
            if i == j:
                continue
            if has_lpips:
                # domina se migliore o uguale su tutte e 4 le dimensioni
                # bpp↓  psnr↑  lpips↓  enc_ms↓
                if (
                    other[0] <= point[0]  # bpp più basso
                    and other[1] >= point[1]  # psnr più alto
                    and other[2] <= point[2]  # lpips più basso
                    and other[3] <= point[3]  # enc_ms più basso
                    and (
                        other[0] < point[0]
                        or other[1] > point[1]
                        or other[2] < point[2]
                        or other[3] < point[3]
                    )
                ):
                    dominated = True
                    break
            else:
                if (
                    other[0] <= point[0]
                    and other[1] >= point[1]
                    and other[2] <= point[2]
                    and (
                        other[0] < point[0]
                        or other[1] > point[1]
                        or other[2] < point[2]
                    )
                ):
                    dominated = True
                    break
        pareto_mask.append(not dominated)

    return candidates[pareto_mask].copy()


# ── Selettore principale ─────────────────────────────────────────


class AdaptiveSelector:
    """
    Selettore adattivo del metodo di compressione.
    Deriva le regole dai dati empirici del benchmark.
    """

    def __init__(self) -> None:
        self.image_data = load_image_data()
        self.audio_data = load_audio_data()
        self.video_data = load_video_data()

    def select(self, constraints: UserConstraints) -> SelectionResult:
        """Seleziona il metodo ottimo dato i vincoli utente."""
        if constraints.domain == "image":
            return self._select_image(constraints)
        elif constraints.domain == "audio":
            return self._select_audio(constraints)
        elif constraints.domain == "video":
            return self._select_video(constraints)
        else:
            raise ValueError(f"Dominio non supportato: {constraints.domain}")

    def _filter_by_constraints(
        self, df: pd.DataFrame, constraints: UserConstraints
    ) -> pd.DataFrame:
        """Filtra i punti che rispettano tutti i vincoli."""
        filtered = df.copy()
        if constraints.max_bpp is not None:
            filtered = filtered[filtered["bpp"] <= constraints.max_bpp]
        if constraints.min_psnr is not None:
            filtered = filtered[filtered["psnr"] >= constraints.min_psnr]
        if constraints.max_lpips is not None:
            filtered = filtered[filtered["lpips"] <= constraints.max_lpips]
        if constraints.max_enc_ms is not None:
            filtered = filtered[filtered["enc_ms"] <= constraints.max_enc_ms]
        return filtered

    def _select_image(self, constraints: UserConstraints) -> SelectionResult:
        df = self.image_data.copy()
        filtered = self._filter_by_constraints(df, constraints)

        if filtered.empty:
            filtered = df.copy()
            reason = (
                "Nessun codec rispetta tutti i vincoli. "
                "Selezionato il migliore disponibile."
            )
        else:
            reason = "Selezionato il punto ottimo sulla frontiera di Pareto."

        if constraints.max_lpips is not None or constraints.min_psnr is not None:
            best = filtered.loc[filtered["lpips"].idxmin()]
        else:
            best = filtered.loc[filtered["psnr"].idxmax()]

        return _row_to_result(best, reason)

    def _select_audio(self, constraints: UserConstraints) -> SelectionResult:
        df = self.audio_data.copy()

        # per l'audio filtra su param_value (target kbps) non su bpp effettivo
        filtered = df.copy()
        if constraints.max_bpp is not None:
            filtered = filtered[filtered["param_value"] <= constraints.max_bpp]
        if constraints.min_psnr is not None:
            filtered = filtered[filtered["psnr"] >= constraints.min_psnr]
        if constraints.max_enc_ms is not None:
            filtered = filtered[filtered["enc_ms"] <= constraints.max_enc_ms]

        if filtered.empty:
            filtered = df.copy()
            reason = (
                "Nessun codec rispetta tutti i vincoli. "
                "Selezionato il migliore disponibile."
            )
        else:
            reason = "Selezionato il punto ottimo."

        best = filtered.loc[filtered["psnr"].idxmax()]
        return _row_to_result(best, reason)

    def _select_video(self, constraints: UserConstraints) -> SelectionResult:
        df = self.video_data.copy()
        filtered = self._filter_by_constraints(df, constraints)

        if filtered.empty:
            filtered = df.copy()
            reason = (
                "Nessun codec rispetta tutti i vincoli. "
                "Selezionato il migliore disponibile."
            )
        else:
            reason = "Selezionato il punto ottimo."

        best = filtered.loc[filtered["psnr"].idxmax()]
        return _row_to_result(best, reason)

    def get_pareto_frontier(
        self, domain: str, max_enc_ms: Optional[float] = None
    ) -> pd.DataFrame:
        """Restituisce la frontiera di Pareto per il dominio specificato."""
        if domain == "image":
            return compute_pareto_frontier(self.image_data, max_enc_ms)
        elif domain == "audio":
            return compute_pareto_frontier(self.audio_data, max_enc_ms)
        elif domain == "video":
            return compute_pareto_frontier(self.video_data, max_enc_ms)
        else:
            raise ValueError(f"Dominio non supportato: {domain}")

    def explain(self, constraints: UserConstraints) -> str:
        """Spiega la scelta del selettore con il trade-off."""
        result = self.select(constraints)
        lines = [
            f"Codec selezionato: {result.codec}",
            f"Parametro: {result.param_name} = {result.param_value}",
            f"Bitrate previsto: {result.predicted_bpp:.3f}",
            f"PSNR previsto: {result.predicted_psnr:.2f} dB",
            f"LPIPS previsto: {result.predicted_lpips:.4f}",
            f"Latenza codifica: {result.predicted_enc_ms:.1f} ms",
            f"Motivazione: {result.reason}",
        ]
        return "\n".join(lines)


# ── Test rapido ──────────────────────────────────────────────────

if __name__ == "__main__":
    selector = AdaptiveSelector()

    print("=" * 60)
    print("TEST SELETTORE ADATTIVO")
    print("=" * 60)

    # Scenario 1: immagine, basso bitrate, qualità percettiva
    print("\nScenario 1: immagine, max 0.3 bpp, qualità percettiva")
    result = selector.select(
        UserConstraints(domain="image", max_bpp=0.3, max_lpips=0.3)
    )
    print(f"  → {result.codec} (q={result.param_value})")
    print(
        f"     bpp={result.predicted_bpp:.3f} | "
        f"psnr={result.predicted_psnr:.2f}dB | "
        f"lpips={result.predicted_lpips:.4f} | "
        f"enc={result.predicted_enc_ms:.1f}ms"
    )
    print(f"     {result.reason}")

    # Scenario 2: immagine, vincolo latenza stretto
    print("\nScenario 2: immagine, max 30ms codifica, max 1 bpp")
    result = selector.select(
        UserConstraints(domain="image", max_bpp=1.0, max_enc_ms=30.0)
    )
    print(f"  → {result.codec} (q={result.param_value})")
    print(
        f"     bpp={result.predicted_bpp:.3f} | "
        f"psnr={result.predicted_psnr:.2f}dB | "
        f"enc={result.predicted_enc_ms:.1f}ms"
    )
    print(f"     {result.reason}")

    # Scenario 3: audio, basso bitrate
    print("\nScenario 3: audio, max 12 kbps")
    result = selector.select(UserConstraints(domain="audio", max_bpp=12.0))
    print(f"  → {result.codec} ({result.param_value}kbps)")
    print(
        f"     PESQ={result.predicted_psnr:.2f} | "
        f"mel_dist={result.predicted_lpips:.3f} | "
        f"enc={result.predicted_enc_ms:.1f}ms"
    )
    print(f"     {result.reason}")

    # Scenario 4: video, real-time
    print("\nScenario 4: video, real-time (min 24 fps codifica)")
    df_video = selector.video_data
    rt = df_video[df_video["fps"] >= 24].copy()
    if not rt.empty:
        best = rt.loc[rt["psnr"].idxmax()]
        print(f"  → {best['codec']} crf={best['param_value']}")
        print(
            f"     bpp={best['bpp']:.4f} | psnr={best['psnr']:.2f}dB | "
            f"fps={best['fps']:.1f}"
        )

    # Scenario 5: immagine, qualità alta senza vincolo latenza
    print("\nScenario 5: immagine, min 35 dB PSNR")
    result = selector.select(UserConstraints(domain="image", min_psnr=35.0))
    print(f"  → {result.codec} (q={result.param_value})")
    print(
        f"     bpp={result.predicted_bpp:.3f} | "
        f"psnr={result.predicted_psnr:.2f}dB | "
        f"lpips={result.predicted_lpips:.4f} | "
        f"enc={result.predicted_enc_ms:.1f}ms"
    )
    print(f"     {result.reason}")

    # Frontiere di Pareto immagini
    print("\nFrontiera di Pareto — Immagini (latenza ≤ 100ms, uso pratico):")
    pareto_fast = selector.get_pareto_frontier("image", max_enc_ms=100.0)
    print(
        pareto_fast[
            ["codec", "param_value", "bpp", "psnr", "lpips", "enc_ms"]
        ].to_string(index=False)
    )

    print("\nFrontiera di Pareto — Immagini (alta qualità, nessun vincolo latenza):")
    pareto_all = selector.get_pareto_frontier("image")
    print(
        pareto_all[
            ["codec", "param_value", "bpp", "psnr", "lpips", "enc_ms"]
        ].to_string(index=False)
    )

    print("\nFrontiera di Pareto — Video:")
    pareto_video = selector.get_pareto_frontier("video")
    print(
        pareto_video[
            ["codec", "param_value", "bpp", "psnr", "lpips", "fps", "enc_ms"]
        ].to_string(index=False)
    )
