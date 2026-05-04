import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .normalization_profile import normalize_with_profile
except ImportError:
    from normalization_profile import normalize_with_profile


@dataclass
class RDEPoint:
    codec: str
    config: str
    rate: float
    quality: float
    energy: float
    raw: Dict[str, Any]
    time_ms: Optional[float] = None


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _parse_float(value: Any) -> float:
    if value is None:
        raise ValueError("Valore numerico mancante")

    text = str(value).strip()

    if "," in text and "." not in text:
        text = text.replace(",", ".")

    text = text.replace("−", "-")

    match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.IGNORECASE)

    if not match:
        raise ValueError(f"Impossibile convertire in float: {value}")

    return float(match.group(0))


def _parse_time_ms(value: Any, column_name: str) -> Optional[float]:
    if value in (None, ""):
        return None

    t = _parse_float(value)
    norm = _normalize_name(column_name)

    if "ms" in norm:
        return t

    if norm.endswith("s") or "seconds" in norm or "totaltimes" in norm:
        return t * 1000.0

    return t


def _find_column(
    headers: Iterable[str],
    aliases: list[str],
    override: Optional[str] = None,
) -> Optional[str]:
    headers = list(headers)

    if override:
        override_norm = _normalize_name(override)
        for h in headers:
            if _normalize_name(h) == override_norm:
                return h

    alias_norms = {_normalize_name(a) for a in aliases}

    for h in headers:
        if _normalize_name(h) in alias_norms:
            return h

    return None


def _get_value_by_alias(row: Dict[str, Any], aliases: list[str]) -> Optional[str]:
    alias_norms = {_normalize_name(a) for a in aliases}

    for key, value in row.items():
        if _normalize_name(key) in alias_norms and value not in (None, ""):
            return str(value)

    return None


def _build_config(row: Dict[str, Any], config_col: Optional[str]) -> str:
    if config_col and row.get(config_col):
        return str(row[config_col]).strip()

    parts = []

    for aliases, name in [
        (["q", "quality_level", "qualitylevel"], "q"),
        (["crf"], "crf"),
        (["lambda", "lam"], "lambda"),
        (["d", "distance"], "d"),
        (["qp"], "qp"),
        (["preset"], "preset"),
        (["bitrate", "kbps"], "bitrate"),
        (["param"], "param"),
    ]:
        value = _get_value_by_alias(row, aliases)
        if value:
            parts.append(f"{name}={value}")

    if parts:
        return ", ".join(parts)

    return "unknown"


def _quantile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("Quantile richiesto su lista vuota.")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    q = max(0.0, min(1.0, q))
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0

    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def load_rde_points(
    csv_path: str | Path,
    codec_col: Optional[str] = None,
    config_col: Optional[str] = None,
    rate_col: Optional[str] = None,
    quality_col: Optional[str] = None,
    energy_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> List[RDEPoint]:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        codec_key = _find_column(
            headers,
            ["codec", "model", "method", "name", "encoder"],
            codec_col,
        )

        config_key = _find_column(
            headers,
            [
                "config",
                "configuration",
                "setting",
                "op_point",
                "operating_point",
                "point",
                "param",
            ],
            config_col,
        )

        rate_key = _find_column(
            headers,
            [
                "bpp",
                "rate",
                "bitrate",
                "bitrate_mbps",
                "kbps",
                "mean_bpp",
                "bpp_mean",
            ],
            rate_col,
        )

        quality_key = _find_column(
            headers,
            [
                "ssimulacra2",
                "ssimulacra_2",
                "ssimulacra 2",
                "quality",
                "psnr",
                "vmaf",
                "visqol",
                "mos",
            ],
            quality_col,
        )

        energy_key = _find_column(
            headers,
            [
                "energy",
                "energy_j",
                "joules",
                "j_img",
                "j/img",
                "j_per_img",
                "energy_j_per_image",
                "energy_per_image_j",
                "j_per_image",
                "j_s",
                "j/s",
                "kj_seq",
                "kj/seq",
                "energy_kj_per_sequence",
            ],
            energy_col,
        )

        time_key = _find_column(
            headers,
            [
                "time_ms",
                "pipeline_time_ms",
                "total_time_ms",
                "time_compress_ms",
                "time_decompress_ms",
                "total_time_s",
                "time_s",
                "time",
            ],
            time_col,
        )

        missing = []
        if codec_key is None:
            missing.append("codec")
        if rate_key is None:
            missing.append("rate")
        if quality_key is None:
            missing.append("quality")
        if energy_key is None:
            missing.append("energy")

        if missing:
            raise ValueError(
                "Colonne mancanti: "
                + ", ".join(missing)
                + "\nColonne trovate nel CSV: "
                + ", ".join(headers)
                + "\nUsa --codec-col, --rate-col, --quality-col, --energy-col per specificarle manualmente."
            )

        points: List[RDEPoint] = []

        for row in reader:
            try:
                codec = str(row[codec_key]).strip()
                config = _build_config(row, config_key)
                rate = _parse_float(row[rate_key])
                quality = _parse_float(row[quality_key])
                energy = _parse_float(row[energy_key])

                if not codec:
                    continue

                time_ms = None
                if time_key is not None:
                    time_ms = _parse_time_ms(row.get(time_key), time_key)

                raw = dict(row)
                raw.update(
                    {
                        "quality_mean": quality,
                        "quality_min": quality,
                        "quality_p10": quality,
                        "quality_p25": quality,
                        "quality_std": 0.0,
                        "time_mean_ms": time_ms,
                        "time_p90_ms": time_ms,
                        "time_max_ms": time_ms,
                    }
                )

                points.append(
                    RDEPoint(
                        codec=codec,
                        config=config,
                        rate=rate,
                        quality=quality,
                        energy=energy,
                        raw=raw,
                        time_ms=time_ms,
                    )
                )

            except Exception:
                continue

    if not points:
        raise ValueError("Nessun punto R-D-E valido trovato nel CSV.")

    return points


def aggregate_points_by_config(points: List[RDEPoint]) -> List[RDEPoint]:
    grouped: Dict[tuple[str, str], list[RDEPoint]] = {}

    for p in points:
        key = (p.codec, p.config)
        grouped.setdefault(key, []).append(p)

    aggregated: List[RDEPoint] = []

    for (codec, config), group in grouped.items():
        rates = [p.rate for p in group]
        qualities = [p.quality for p in group]
        energies = [p.energy for p in group]
        times = [p.time_ms for p in group if p.time_ms is not None]

        mean_rate = _mean(rates)
        mean_quality = _mean(qualities)
        mean_energy = _mean(energies)
        mean_time_ms = _mean(times) if times else None

        raw = {
            "aggregation": "mean_by_codec_config",
            "n_samples": len(group),
            "quality_mean": mean_quality,
            "quality_min": min(qualities),
            "quality_p10": _quantile(qualities, 0.10),
            "quality_p25": _quantile(qualities, 0.25),
            "quality_max": max(qualities),
            "quality_std": _std(qualities),
            "time_mean_ms": mean_time_ms,
            "time_min_ms": min(times) if times else None,
            "time_p90_ms": _quantile(times, 0.90) if times else None,
            "time_max_ms": max(times) if times else None,
        }

        aggregated.append(
            RDEPoint(
                codec=codec,
                config=config,
                rate=mean_rate,
                quality=mean_quality,
                energy=mean_energy,
                raw=raw,
                time_ms=mean_time_ms,
            )
        )

    return aggregated


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _make_minmax_transform(
    values: list[float],
    use_log: bool = False,
    high_is_good: bool = False,
):
    if not values:
        raise ValueError("Impossibile normalizzare: lista valori vuota.")

    if use_log:
        transformed = [math.log10(max(v, 1e-12)) for v in values]
    else:
        transformed = values[:]

    v_min = min(transformed)
    v_max = max(transformed)

    def transform(value: float) -> float:
        if use_log:
            value_t = math.log10(max(value, 1e-12))
        else:
            value_t = value

        if abs(v_max - v_min) < 1e-12:
            return 1.0 if high_is_good else 0.0

        return _clamp01((value_t - v_min) / (v_max - v_min))

    return transform


def _normalization_metadata(reference_points: List[RDEPoint]) -> Dict[str, Any]:
    return {
        "num_reference_points": len(reference_points),
        "rate_scale": "log10",
        "energy_scale": "log10",
        "quality_scale": "linear",
        "rate_min": min(p.rate for p in reference_points),
        "rate_max": max(p.rate for p in reference_points),
        "energy_min": min(p.energy for p in reference_points),
        "energy_max": max(p.energy for p in reference_points),
        "quality_min": min(p.quality for p in reference_points),
        "quality_max": max(p.quality for p in reference_points),
    }


def _get_quality_stat(point: RDEPoint, stat: str) -> float:
    stat = stat.lower().strip()

    if stat == "mean":
        return float(point.raw.get("quality_mean", point.quality))
    if stat == "min":
        return float(point.raw.get("quality_min", point.quality))
    if stat == "p10":
        return float(point.raw.get("quality_p10", point.quality))
    if stat == "p25":
        return float(point.raw.get("quality_p25", point.quality))

    raise ValueError(f"Statistica qualità non supportata: {stat}")


def _passes_base_constraints(
    point: RDEPoint,
    max_rate: Optional[float],
    max_energy: Optional[float],
    max_time_ms: Optional[float],
) -> bool:
    if max_rate is not None and point.rate > max_rate:
        return False

    if max_energy is not None and point.energy > max_energy:
        return False

    if max_time_ms is not None:
        time_ms = point.time_ms
        if time_ms is None:
            return False
        if time_ms > max_time_ms:
            return False

    return True


def select_best_rde(
    points: List[RDEPoint],
    weights: Dict[str, float],
    min_quality: Optional[float] = None,
    max_rate: Optional[float] = None,
    max_energy: Optional[float] = None,
    max_time_ms: Optional[float] = None,
    quality_constraint_stat: str = "mean",
    near_quality_floor: Optional[float] = None,
    allow_degraded_fallback: bool = False,
    top_k: int = 5,
    normalization_points: Optional[List[RDEPoint]] = None,
    normalization_profile=None,
) -> Dict[str, Any]:
    safe_pool: List[RDEPoint] = []
    near_pool: List[RDEPoint] = []

    excluded_by_base_constraints = 0
    excluded_by_quality_guard = 0

    for p in points:
        if not _passes_base_constraints(p, max_rate, max_energy, max_time_ms):
            excluded_by_base_constraints += 1
            continue

        q_guard = _get_quality_stat(p, quality_constraint_stat)

        if min_quality is None or q_guard >= min_quality:
            safe_pool.append(p)
            continue

        excluded_by_quality_guard += 1

        if (
            allow_degraded_fallback
            and near_quality_floor is not None
            and q_guard >= near_quality_floor
        ):
            near_pool.append(p)

    decision_mode = "safe"

    if safe_pool:
        active_pool = safe_pool
    elif allow_degraded_fallback and near_pool:
        active_pool = near_pool
        decision_mode = "degraded_fallback"
    else:
        raise ValueError(
            "Nessuna configurazione soddisfa il vincolo di usabilità. "
            f"quality_constraint_stat={quality_constraint_stat}, "
            f"quality_floor={min_quality}, "
            f"near_quality_floor={near_quality_floor}, "
            f"safe_pool={len(safe_pool)}, near_pool={len(near_pool)}."
        )

    reference_points = normalization_points if normalization_points is not None else points

    if normalization_profile is None:
        if not reference_points:
            raise ValueError("Pool di riferimento per la normalizzazione vuoto.")

        rate_transform = _make_minmax_transform(
            [p.rate for p in reference_points],
            use_log=True,
            high_is_good=False,
        )

        energy_transform = _make_minmax_transform(
            [p.energy for p in reference_points],
            use_log=True,
            high_is_good=False,
        )

        quality_transform = _make_minmax_transform(
            [p.quality for p in reference_points],
            use_log=False,
            high_is_good=True,
        )

        normalization_reference = _normalization_metadata(reference_points)
    else:
        normalization_reference = {
            "source": "precomputed_profile",
            "profile_version": normalization_profile.get("version"),
            "profile_domain": normalization_profile.get("domain"),
            "num_reference_points": normalization_profile.get("num_points"),
            "transforms": normalization_profile.get("transforms", {}),
            "scales": normalization_profile.get("scales", {}),
        }

    scored = []

    for p in active_pool:
        if normalization_profile is not None:
            norm = normalize_with_profile(p, normalization_profile)
            r_n = norm["norm_rate"]
            d_n = norm["norm_distortion"]
            e_n = norm["norm_energy"]
        else:
            r_n = rate_transform(p.rate)
            e_n = energy_transform(p.energy)
            q_n = quality_transform(p.quality)
            d_n = 1.0 - q_n

        cost = (
            weights["w_R"] * r_n
            + weights["w_E"] * e_n
            + weights["w_D"] * d_n
        )

        q_guard = _get_quality_stat(p, quality_constraint_stat)

        scored.append(
            {
                "codec": p.codec,
                "config": p.config,
                "rate": p.rate,
                "quality": p.quality,
                "energy": p.energy,
                "time_ms": p.time_ms,
                "quality_constraint_stat": quality_constraint_stat,
                "quality_constraint_value": q_guard,
                "quality_stats": {
                    "mean": float(p.raw.get("quality_mean", p.quality)),
                    "min": float(p.raw.get("quality_min", p.quality)),
                    "p10": float(p.raw.get("quality_p10", p.quality)),
                    "p25": float(p.raw.get("quality_p25", p.quality)),
                    "std": float(p.raw.get("quality_std", 0.0)),
                },
                "normalized": {
                    "rate": r_n,
                    "distortion": d_n,
                    "energy": e_n,
                },
                "cost": cost,
                "decision_mode": decision_mode,
                "raw": p.raw,
            }
        )

    scored.sort(key=lambda x: x["cost"])

    return {
        "selected": scored[0],
        "top_k": scored[:top_k],
        "num_points_total": len(points),
        "num_points_admissible": len(active_pool),
        "num_points_safe": len(safe_pool),
        "num_points_near": len(near_pool),
        "excluded_by_base_constraints": excluded_by_base_constraints,
        "excluded_by_quality_guard": excluded_by_quality_guard,
        "decision_mode": decision_mode,
        "quality_guard": {
            "hard_constraint": True,
            "stat": quality_constraint_stat,
            "floor": min_quality,
            "near_floor": near_quality_floor,
            "allow_degraded_fallback": allow_degraded_fallback,
            "max_time_ms": max_time_ms,
        },
        "normalization_reference": normalization_reference,
    }
