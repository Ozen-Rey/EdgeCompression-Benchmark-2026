import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _log10_values(values: Iterable[float]) -> List[float]:
    out = []
    for v in values:
        if v is not None and v > 0:
            out.append(math.log10(v))
    return out


def _linear_values(values: Iterable[float]) -> List[float]:
    out = []
    for v in values:
        if v is not None:
            out.append(float(v))
    return out


def _min_max(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


def build_normalization_profile(
    points,
    domain: str,
    rate_transform: str = "log10",
    energy_transform: str = "log10",
    quality_transform: str = "linear",
    mode: str = "global",
    source: Optional[str] = None,
    comparability: Optional[str] = None,
    warning: Optional[str] = None,
    build_scope: str = "all",
) -> Dict[str, Any]:
    rates = [_safe_float(getattr(p, "rate", None)) for p in points]
    energies = [_safe_float(getattr(p, "energy", None)) for p in points]
    qualities = [_safe_float(getattr(p, "quality", None)) for p in points]

    if rate_transform == "log10":
        rate_values = _log10_values(rates)
    else:
        rate_values = _linear_values(rates)

    if energy_transform == "log10":
        energy_values = _log10_values(energies)
    else:
        energy_values = _linear_values(energies)

    quality_values = _linear_values(qualities)

    return {
        "version": "0.4",
        "domain": domain,
        "mode": mode,
        "source": source,
        "comparability": comparability or (
            "global" if mode == "global" else "local_only"
        ),
        "warning": warning,
        "build_scope": build_scope,
        "description": "Precomputed normalization profile for R-D-E router.",
        "transforms": {
            "rate": rate_transform,
            "energy": energy_transform,
            "quality": quality_transform,
            "distortion": "quality_to_distortion",
        },
        "scales": {
            "rate": _min_max(rate_values),
            "energy": _min_max(energy_values),
            "quality": _min_max(quality_values),
        },
        "num_points": len(points),
    }


def save_normalization_profile(profile: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def load_normalization_profile(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Normalization profile non trovato: {p}")

    with p.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    _validate_normalization_profile(profile)
    return profile


def _validate_normalization_profile(profile: Dict[str, Any]) -> None:
    scales = profile.get("scales", {})

    for key in ["rate", "energy", "quality"]:
        if key not in scales:
            raise ValueError(f"Normalization profile non valido: scala mancante '{key}'")

        mn = scales[key].get("min")
        mx = scales[key].get("max")

        if mn is None or mx is None:
            raise ValueError(f"Normalization profile non valido: min/max mancanti per '{key}'")

        if float(mx) < float(mn):
            raise ValueError(f"Normalization profile non valido: max < min per '{key}'")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_from_minmax(value: float, mn: float, mx: float) -> float:
    if mx <= mn:
        return 0.0
    return _clamp01((value - mn) / (mx - mn))


def normalize_with_profile(point, profile: Dict[str, Any]) -> Dict[str, float]:
    transforms = profile.get("transforms", {})
    scales = profile.get("scales", {})

    rate = _safe_float(getattr(point, "rate", None))
    energy = _safe_float(getattr(point, "energy", None))
    quality = _safe_float(getattr(point, "quality", None))

    if rate is None or energy is None or quality is None:
        raise ValueError("Punto R-D-E incompleto: rate/energy/quality mancanti.")

    if transforms.get("rate") == "log10":
        if rate <= 0:
            norm_rate = 1.0
        else:
            norm_rate = _norm_from_minmax(
                math.log10(rate),
                float(scales["rate"]["min"]),
                float(scales["rate"]["max"]),
            )
    else:
        norm_rate = _norm_from_minmax(
            rate,
            float(scales["rate"]["min"]),
            float(scales["rate"]["max"]),
        )

    if transforms.get("energy") == "log10":
        if energy <= 0:
            norm_energy = 1.0
        else:
            norm_energy = _norm_from_minmax(
                math.log10(energy),
                float(scales["energy"]["min"]),
                float(scales["energy"]["max"]),
            )
    else:
        norm_energy = _norm_from_minmax(
            energy,
            float(scales["energy"]["min"]),
            float(scales["energy"]["max"]),
        )

    q_min = float(scales["quality"]["min"])
    q_max = float(scales["quality"]["max"])

    # Distorsione normalizzata: qualità alta -> distorsione bassa.
    norm_quality = _norm_from_minmax(quality, q_min, q_max)
    norm_distortion = _clamp01(1.0 - norm_quality)

    return {
        "norm_rate": norm_rate,
        "norm_energy": norm_energy,
        "norm_distortion": norm_distortion,
    }
