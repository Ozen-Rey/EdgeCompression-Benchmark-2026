"""Neural-inclusive R-D-E oracle audit (offline / paper-facing).

The earlier content-aware analysis (router_content_aware,
policy_comparison, content_predictor_interpretability) restricted the
candidate pool to the deployable classical triple JPEG / JXL / HEVC.
That subset is a *classic-only ablation*; it does not answer the
central R-D-E question — "when, and under which operational profile,
do neural codecs become oracle-optimal once they are included in the
candidate pool?".

This module is read-only against an existing aggregate R-D-E CSV
(e.g. ``image_4dataset_energy_v2_with_jpeg_ai.csv``) that carries one
row per ``(group, codec, param)`` triple with rate, quality and
energy measurements. It produces per-pool / per-profile / per-floor
oracles, a classic-vs-full pool comparison, and per-group target
labels that downstream work can use to train a neural-inclusive
predictor. The router runtime, the ranking score, the operational
report schema, the CLI flags of ``rde_router`` and the benchmark
data itself are unchanged.

Pools:

- ``classic_pool`` = {JPEG, JXL, HEVC}
- ``neural_pool`` = {JPEG_AI, Ballé, Cheng, ELIC, TCM, DCAE}
- ``full_pool`` = classic_pool ∪ neural_pool

Normalization is computed **once on the full pool** before any
admissibility filtering, so the J_RDE scores of classical and neural
candidates are directly comparable. Re-normalizing per pool would
break that comparability; the audit explicitly records the chosen
scope as ``full_pool_global``.

Profiles follow the official router profiles in
``src/router/core/profiles.py`` (``balanced``, ``energy-limited``,
``bandwidth-limited``, ``quality-first``). The exact ``(w_R, w_E,
w_D)`` weights used are written into the report's provenance block so
the analysis is self-contained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from src.router.core.profiles import PROFILES as ROUTER_PROFILES


__all__ = [
    "load_pool_rows",
    "classify_codec_family",
    "normalize_full_pool",
    "select_oracle_for_group",
    "evaluate_pool",
    "build_pool_summary",
    "compute_pool_comparison",
    "build_per_group_labels",
    "build_interpretation",
    "main",
]


# ---------------------------------------------------------------------------
# Family classification
# ---------------------------------------------------------------------------


_FAMILY_NEURAL = {
    "jpegai",
    "balle",
    "balle2018",
    "cheng",
    "cheng2020",
    "elic",
    "tcm",
    "dcae",
}

_FAMILY_CLASSICAL = {
    "jpeg",
    "jxl",
    "jpegxl",
    "hevc",
    "h265",
    "vvc",
    "vvenc",
}


def _normalize_codec_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    # Strip accents on Ballé and similar (common in benchmark CSV variants
    # written with Windows-1252 or UTF-8 indistinctly).
    text = text.replace("é", "e").replace("è", "e").replace("ë", "e")
    text = "".join(ch for ch in text if ch.isalnum())
    return text


def classify_codec_family(codec: Any) -> str:
    """Classify a codec name into ``classical`` / ``neural`` / ``unknown``."""
    key = _normalize_codec_name(codec)
    if key in _FAMILY_NEURAL:
        return "neural"
    if key in _FAMILY_CLASSICAL:
        return "classical"
    return "unknown"


_CLASSIC_POOL_LABELS = {"JPEG", "JXL", "HEVC"}
_NEURAL_POOL_LABELS = {
    "JPEG_AI",
    "JPEGAI",
    "Ballé",
    "Balle",
    "Cheng",
    "ELIC",
    "TCM",
    "DCAE",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _safe_log10(value: float) -> float:
    return math.log10(max(value, 1e-12))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _minmax(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp01((value - lo) / (hi - lo))


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    q = max(0.0, min(1.0, q))
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Optional[float], digits: int = 5) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_GLOBAL_GROUP_KEY = "__global__"


def _load_energy_lookup(
    csv_path: str,
    *,
    codec_col: str,
    config_col: str,
    energy_col: str,
) -> Dict[Tuple[str, str], float]:
    """Build ``(codec, config) -> mean(energy)`` from an aggregate side file.

    The side file is allowed to carry one row per ``(codec, config,
    eval_dataset)`` triple, in which case energy values are averaged
    across the rows that share the same ``(codec, config)`` key.
    """
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            codec = str(raw.get(codec_col, "")).strip()
            config = str(raw.get(config_col, "")).strip()
            energy = _to_float(raw.get(energy_col))
            if not codec or not config or energy is None:
                continue
            key = (codec, config)
            sums[key] += energy
            counts[key] += 1

    if not sums:
        raise ValueError(
            f"No usable (codec, config, {energy_col}) tuples in {csv_path}"
        )

    return {key: sums[key] / counts[key] for key in sums}


def load_pool_rows(
    csv_path: str,
    *,
    codec_col: str,
    config_col: str,
    rate_col: str,
    quality_col: str,
    energy_col: str,
    image_id_col: Optional[str],
    dataset_col: Optional[str] = None,
    energy_csv_path: Optional[str] = None,
    energy_csv_codec_col: Optional[str] = None,
    energy_csv_config_col: Optional[str] = None,
    energy_csv_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load aggregate R-D-E rows from ``csv_path``.

    Each row must carry the codec name, the configuration string, the
    rate and the quality. The energy term is sourced from one of two
    places:

    1. **Single-file mode**: when ``energy_csv_path`` is ``None``, the
       energy is read from the ``energy_col`` column of ``csv_path``
       itself.
    2. **Two-file mode**: when ``energy_csv_path`` is provided, the
       energy is read from that side file and joined on
       ``(codec, config)``. Rows in the side file that share the same
       ``(codec, config)`` (e.g. one per ``eval_dataset``) are
       averaged before the join. This is the right mode when the main
       metrics file carries quality but not energy.

    ``image_id_col`` defines the per-group oracle unit. When it is
    ``None`` or absent from the CSV, every row is assigned to the
    single synthetic group ``__global__``; the oracle is then computed
    once per ``(pool, profile, quality_floor)`` over the entire pool.
    """
    energy_lookup: Optional[Dict[Tuple[str, str], float]] = None
    if energy_csv_path is not None:
        energy_lookup = _load_energy_lookup(
            energy_csv_path,
            codec_col=energy_csv_codec_col or codec_col,
            config_col=energy_csv_config_col or config_col,
            energy_col=energy_csv_col or energy_col,
        )

    id_columns: List[str] = []
    if image_id_col is not None and str(image_id_col).strip() != "":
        id_columns = [
            token.strip()
            for token in str(image_id_col).split(",")
            if token.strip()
        ]

    rows: List[Dict[str, Any]] = []

    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        usable_id_columns = [c for c in id_columns if c in fieldnames]

        for raw in reader:
            codec = str(raw.get(codec_col, "")).strip()
            config = str(raw.get(config_col, "")).strip()

            if not codec or not config:
                continue

            if usable_id_columns:
                parts = [
                    str(raw.get(col, "")).strip()
                    for col in usable_id_columns
                ]
                if any(p == "" for p in parts):
                    continue
                image_id = "::".join(parts)
            else:
                image_id = _GLOBAL_GROUP_KEY

            rate = _to_float(raw.get(rate_col))
            quality = _to_float(raw.get(quality_col))

            if energy_lookup is not None:
                energy = energy_lookup.get((codec, config))
            else:
                energy = _to_float(raw.get(energy_col))

            if rate is None or quality is None or energy is None:
                continue

            dataset = (
                str(raw.get(dataset_col, image_id)).strip()
                if dataset_col and dataset_col in fieldnames
                else image_id
            )

            rows.append(
                {
                    "image_id": image_id,
                    "dataset": dataset,
                    "codec": codec,
                    "config": config,
                    "codec_family": classify_codec_family(codec),
                    "rate": rate,
                    "quality": quality,
                    "energy": energy,
                    "is_neural_csv": str(raw.get("is_neural", "")).strip().lower()
                    in {"1", "true", "yes"},
                    "raw": raw,
                }
            )

    if not rows:
        raise ValueError(f"No usable R-D-E rows loaded from {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Normalization on the full pool
# ---------------------------------------------------------------------------


def normalize_full_pool(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute log-min-max normalization on the full pool, in-place.

    Adds ``norm_rate``, ``norm_energy``, ``norm_distortion`` to every
    row. The scaling is global across the full pool, so J_RDE values
    derived from these fields remain comparable when downstream code
    filters by pool or by quality floor.
    """
    if not rows:
        raise ValueError("normalize_full_pool requires at least one row.")

    log_rates = [_safe_log10(r["rate"]) for r in rows]
    log_energies = [_safe_log10(r["energy"]) for r in rows]
    qualities = [r["quality"] for r in rows]

    rate_lo, rate_hi = min(log_rates), max(log_rates)
    energy_lo, energy_hi = min(log_energies), max(log_energies)
    q_lo, q_hi = min(qualities), max(qualities)

    for r in rows:
        r_n = _minmax(_safe_log10(r["rate"]), rate_lo, rate_hi)
        e_n = _minmax(_safe_log10(r["energy"]), energy_lo, energy_hi)
        q_n = _minmax(r["quality"], q_lo, q_hi)
        r["norm_rate"] = r_n
        r["norm_energy"] = e_n
        r["norm_distortion"] = 1.0 - q_n

    return {
        "rate_log10_min": rate_lo,
        "rate_log10_max": rate_hi,
        "energy_log10_min": energy_lo,
        "energy_log10_max": energy_hi,
        "quality_min": q_lo,
        "quality_max": q_hi,
    }


def _j_rde(row: Dict[str, Any], weights: Dict[str, float]) -> float:
    return (
        weights["w_R"] * row["norm_rate"]
        + weights["w_E"] * row["norm_energy"]
        + weights["w_D"] * row["norm_distortion"]
    )


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


_DEFAULT_PROFILES = [
    "balanced",
    "energy-limited",
    "bandwidth-limited",
    "quality-first",
]


def _resolve_profile_weights(name: str) -> Dict[str, float]:
    key = name.strip().lower().replace("_", "-")
    if key not in ROUTER_PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Known profiles: "
            f"{', '.join(sorted(ROUTER_PROFILES))}"
        )
    profile = ROUTER_PROFILES[key]
    total = profile.w_e + profile.w_r + profile.w_d
    if total <= 0:
        raise ValueError(f"Profile '{name}' has non-positive weight sum.")
    return {
        "w_R": profile.w_r / total,
        "w_E": profile.w_e / total,
        "w_D": profile.w_d / total,
    }


# ---------------------------------------------------------------------------
# Pool filtering
# ---------------------------------------------------------------------------


def _filter_pool(
    rows: List[Dict[str, Any]],
    *,
    pool: str,
) -> List[Dict[str, Any]]:
    if pool == "full_pool":
        return list(rows)
    if pool == "classic_pool":
        return [r for r in rows if r["codec_family"] == "classical"]
    if pool == "neural_pool":
        return [r for r in rows if r["codec_family"] == "neural"]
    raise ValueError(f"Unknown pool '{pool}'.")


# ---------------------------------------------------------------------------
# Oracle selection per group
# ---------------------------------------------------------------------------


def select_oracle_for_group(
    candidates: List[Dict[str, Any]],
    *,
    weights: Dict[str, float],
    quality_floor: Optional[float],
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    feasible = [
        r for r in candidates
        if quality_floor is None or r["quality"] >= quality_floor
    ]
    if not feasible:
        return None

    best: Optional[Dict[str, Any]] = None
    best_cost = math.inf
    for row in feasible:
        cost = _j_rde(row, weights)
        if cost < best_cost:
            best_cost = cost
            best = row

    if best is None:
        return None

    return {
        "image_id": best["image_id"],
        "dataset": best["dataset"],
        "selected_codec": best["codec"],
        "selected_config": best["config"],
        "selected_family": best["codec_family"],
        "rate": best["rate"],
        "quality": best["quality"],
        "energy": best["energy"],
        "J_RDE": best_cost,
        "num_candidates": len(candidates),
        "num_safe_candidates": len(feasible),
    }


# ---------------------------------------------------------------------------
# Per-pool oracle evaluation
# ---------------------------------------------------------------------------


def evaluate_pool(
    rows: List[Dict[str, Any]],
    *,
    pool: str,
    profile_name: str,
    weights: Dict[str, float],
    quality_floor: Optional[float],
) -> List[Dict[str, Any]]:
    pool_rows = _filter_pool(rows, pool=pool)
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool_rows:
        by_group[r["image_id"]].append(r)

    out: List[Dict[str, Any]] = []
    for image_id in sorted(by_group):
        selection = select_oracle_for_group(
            by_group[image_id],
            weights=weights,
            quality_floor=quality_floor,
        )
        if selection is None:
            out.append(
                {
                    "image_id": image_id,
                    "dataset": by_group[image_id][0]["dataset"],
                    "pool": pool,
                    "profile": profile_name,
                    "quality_floor": quality_floor,
                    "selected_codec": None,
                    "selected_config": None,
                    "selected_family": None,
                    "rate": None,
                    "quality": None,
                    "energy": None,
                    "J_RDE": None,
                    "num_candidates": len(by_group[image_id]),
                    "num_safe_candidates": 0,
                    "has_safe_selection": False,
                }
            )
            continue

        selection["pool"] = pool
        selection["profile"] = profile_name
        selection["quality_floor"] = quality_floor
        selection["has_safe_selection"] = True
        out.append(selection)

    return out


# ---------------------------------------------------------------------------
# Pool summary statistics
# ---------------------------------------------------------------------------


def _aggregate_optional(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    real = [v for v in values if v is not None]
    if not real:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": mean(real),
        "median": median(real),
        "p90": _quantile(real, 0.90),
        "max": max(real),
    }


def build_pool_summary(
    selections: List[Dict[str, Any]],
    *,
    pool: str,
    profile_name: str,
    quality_floor: Optional[float],
) -> Dict[str, Any]:
    num_images = len(selections)
    safe = [s for s in selections if s["has_safe_selection"]]
    num_safe = len(safe)

    codec_counter = Counter(
        f"{s['selected_codec']}|{s['selected_config']}"
        for s in safe
    )
    family_counter = Counter(s["selected_family"] for s in safe)

    neural_count = family_counter.get("neural", 0)
    neural_selection_rate = (
        neural_count / num_safe if num_safe > 0 else None
    )

    rates = _aggregate_optional([s["rate"] for s in safe])
    qualities = _aggregate_optional([s["quality"] for s in safe])
    energies = _aggregate_optional([s["energy"] for s in safe])
    j_rde = _aggregate_optional([s["J_RDE"] for s in safe])

    return {
        "pool": pool,
        "profile": profile_name,
        "quality_floor": quality_floor,
        "num_images": num_images,
        "num_images_with_safe_candidate": num_safe,
        "oracle_codec_counts": dict(codec_counter),
        "oracle_family_counts": dict(family_counter),
        "neural_selection_count": neural_count,
        "neural_selection_rate": neural_selection_rate,
        "mean_rate": rates["mean"],
        "mean_quality": qualities["mean"],
        "mean_energy": energies["mean"],
        "mean_J_RDE": j_rde["mean"],
        "median_J_RDE": j_rde["median"],
        "p90_J_RDE": j_rde["p90"],
        "max_J_RDE": j_rde["max"],
    }


# ---------------------------------------------------------------------------
# Pool comparison: classic vs full
# ---------------------------------------------------------------------------


def compute_pool_comparison(
    classic_selections: List[Dict[str, Any]],
    full_selections: List[Dict[str, Any]],
    *,
    profile_name: str,
    quality_floor: Optional[float],
) -> Dict[str, Any]:
    classic_by_image = {s["image_id"]: s for s in classic_selections}
    full_by_image = {s["image_id"]: s for s in full_selections}

    paired_image_ids = sorted(
        set(classic_by_image) & set(full_by_image)
    )

    regrets: List[float] = []
    rate_gains_when_neural: List[float] = []
    energy_penalties_when_neural: List[float] = []
    quality_deltas_when_neural: List[float] = []

    neural_count = 0
    classic_only_full_count = 0

    for image_id in paired_image_ids:
        c = classic_by_image[image_id]
        f = full_by_image[image_id]

        if not (c["has_safe_selection"] and f["has_safe_selection"]):
            continue

        regret = c["J_RDE"] - f["J_RDE"]
        regrets.append(regret)

        if f["selected_family"] == "neural":
            neural_count += 1
            rate_gains_when_neural.append(c["rate"] - f["rate"])
            energy_penalties_when_neural.append(f["energy"] - c["energy"])
            quality_deltas_when_neural.append(f["quality"] - c["quality"])
        else:
            classic_only_full_count += 1

    regret_stats = _aggregate_optional(regrets)
    mean_classic_j = (
        mean(c["J_RDE"] for c in classic_selections if c["has_safe_selection"])
        if any(c["has_safe_selection"] for c in classic_selections)
        else None
    )
    mean_full_j = (
        mean(f["J_RDE"] for f in full_selections if f["has_safe_selection"])
        if any(f["has_safe_selection"] for f in full_selections)
        else None
    )

    relative_improvement = (
        (mean_classic_j - mean_full_j) / mean_classic_j
        if mean_classic_j is not None
        and mean_full_j is not None
        and mean_classic_j > 0
        else None
    )

    neural_selection_rate_in_full = (
        neural_count / len(paired_image_ids) if paired_image_ids else None
    )

    return {
        "profile": profile_name,
        "quality_floor": quality_floor,
        "num_paired_images": len(paired_image_ids),
        "mean_classic_J_RDE": mean_classic_j,
        "mean_full_J_RDE": mean_full_j,
        "mean_regret_classic_vs_full": regret_stats["mean"],
        "median_regret_classic_vs_full": regret_stats["median"],
        "p90_regret_classic_vs_full": regret_stats["p90"],
        "max_regret_classic_vs_full": regret_stats["max"],
        "relative_mean_improvement_full_vs_classic": relative_improvement,
        "neural_selection_count_in_full": neural_count,
        "classic_only_selection_count_in_full": classic_only_full_count,
        "neural_selection_rate_in_full": neural_selection_rate_in_full,
        "mean_rate_gain_when_neural_selected": (
            mean(rate_gains_when_neural) if rate_gains_when_neural else None
        ),
        "mean_energy_penalty_when_neural_selected": (
            mean(energy_penalties_when_neural)
            if energy_penalties_when_neural
            else None
        ),
        "mean_quality_delta_when_neural_selected": (
            mean(quality_deltas_when_neural)
            if quality_deltas_when_neural
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-group labels for downstream classifier training
# ---------------------------------------------------------------------------


def build_per_group_labels(
    *,
    classic_selections: List[Dict[str, Any]],
    full_selections: List[Dict[str, Any]],
    profile_name: str,
    quality_floor: Optional[float],
) -> List[Dict[str, Any]]:
    classic_by_image = {s["image_id"]: s for s in classic_selections}
    full_by_image = {s["image_id"]: s for s in full_selections}

    image_ids = sorted(set(classic_by_image) | set(full_by_image))
    out: List[Dict[str, Any]] = []

    for image_id in image_ids:
        c = classic_by_image.get(image_id)
        f = full_by_image.get(image_id)

        c_label = (
            f"{c['selected_codec']}|{c['selected_config']}"
            if c and c["has_safe_selection"]
            else None
        )
        f_label = (
            f"{f['selected_codec']}|{f['selected_config']}"
            if f and f["has_safe_selection"]
            else None
        )
        f_family = (
            f["selected_family"] if f and f["has_safe_selection"] else None
        )

        regret = None
        if (
            c is not None
            and f is not None
            and c["has_safe_selection"]
            and f["has_safe_selection"]
        ):
            regret = c["J_RDE"] - f["J_RDE"]

        out.append(
            {
                "image_id": image_id,
                "dataset": (c or f or {}).get("dataset"),
                "profile": profile_name,
                "quality_floor": quality_floor,
                "classic_pool_oracle_label": c_label,
                "full_pool_oracle_label": f_label,
                "full_pool_oracle_family": f_family,
                "regret_classic_vs_full": regret,
            }
        )

    return out


# ---------------------------------------------------------------------------
# Auto-generated interpretation
# ---------------------------------------------------------------------------


_CLOSE_REGRET_THRESHOLD = 0.005


def _index_summaries(
    summaries: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, Optional[float]], Dict[str, Any]]:
    return {
        (s["pool"], s["profile"], s["quality_floor"]): s
        for s in summaries
    }


def _index_comparisons(
    comparisons: List[Dict[str, Any]],
) -> Dict[Tuple[str, Optional[float]], Dict[str, Any]]:
    return {
        (c["profile"], c["quality_floor"]): c
        for c in comparisons
    }


def build_interpretation(
    *,
    summaries: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
    profiles: List[str],
    quality_floors: List[Optional[float]],
) -> List[str]:
    notes: List[str] = []

    summary_index = _index_summaries(summaries)
    comparison_index = _index_comparisons(comparisons)

    full_summaries = [
        s for s in summaries if s["pool"] == "full_pool"
    ]
    any_neural = any(
        (s.get("neural_selection_count") or 0) > 0 for s in full_summaries
    )

    if any_neural:
        regimes_with_neural = sum(
            1
            for s in full_summaries
            if (s.get("neural_selection_count") or 0) > 0
        )
        notes.append(
            f"Neural codecs are selected as oracle under "
            f"{regimes_with_neural} of the evaluated full-pool "
            f"(profile, quality_floor) combinations within this benchmark; "
            "this does not imply universal dominance of neural over "
            "classical codecs."
        )
    else:
        notes.append(
            "No neural codec is selected as oracle under any of the "
            "evaluated full-pool (profile, quality_floor) combinations "
            "within this benchmark; this does not imply that neural "
            "codecs cannot win under other profiles or constraints."
        )

    # Profile concentration of neural selections.
    profile_neural_rate: Dict[str, List[float]] = defaultdict(list)
    for s in full_summaries:
        rate = s.get("neural_selection_rate")
        if rate is not None:
            profile_neural_rate[s["profile"]].append(rate)

    if profile_neural_rate:
        averaged = {
            profile: mean(rates)
            for profile, rates in profile_neural_rate.items()
        }
        sorted_profiles = sorted(
            averaged.items(), key=lambda item: item[1], reverse=True
        )
        top_profile, top_rate = sorted_profiles[0]
        if top_rate > 0:
            notes.append(
                f"The '{top_profile}' profile shows the highest average "
                f"neural selection rate "
                f"({top_rate:.2%}) within this benchmark, which is "
                "consistent with neural codecs being preferred when their "
                "associated weighting emphasises the dimension on which "
                "they hold an advantage."
            )

        if "energy-limited" in averaged:
            energy_rate = averaged["energy-limited"]
            other_rates = [
                rate
                for profile, rate in averaged.items()
                if profile != "energy-limited"
            ]
            if other_rates and energy_rate < min(other_rates):
                notes.append(
                    "Under the 'energy-limited' profile the neural "
                    "selection rate is the lowest among the evaluated "
                    "profiles, which suggests that the bitrate advantage "
                    "of neural codecs does not compensate their energy "
                    "cost when energy receives a high weight, within "
                    "this benchmark."
                )

    # Classic-only proximity to full.
    classic_only_close: List[Tuple[str, Optional[float], float]] = []
    classic_only_far: List[Tuple[str, Optional[float], float]] = []
    for cmp in comparisons:
        mean_regret = cmp.get("mean_regret_classic_vs_full")
        if mean_regret is None:
            continue
        if mean_regret <= _CLOSE_REGRET_THRESHOLD:
            classic_only_close.append(
                (cmp["profile"], cmp["quality_floor"], mean_regret)
            )
        else:
            classic_only_far.append(
                (cmp["profile"], cmp["quality_floor"], mean_regret)
            )

    if classic_only_close and not classic_only_far:
        notes.append(
            "The classic-only pool is close to the full-pool oracle "
            "under every evaluated (profile, quality_floor); the "
            "neural codecs do not contribute a meaningful regret "
            "reduction within this benchmark."
        )
    elif classic_only_far:
        classic_only_far.sort(key=lambda item: item[2], reverse=True)
        worst_profile, worst_floor, worst_regret = classic_only_far[0]
        notes.append(
            f"The classic-only pool incurs the largest regret relative "
            f"to the full pool under profile='{worst_profile}', "
            f"quality_floor={worst_floor}: mean regret = "
            f"{worst_regret:.5f}, which suggests that in that regime "
            "the neural codecs occupy a region of the R-D-E space not "
            "reachable by the classical pool."
        )

    # Rate-vs-energy trade-off when neural wins.
    rate_gains = [
        cmp.get("mean_rate_gain_when_neural_selected")
        for cmp in comparisons
        if cmp.get("mean_rate_gain_when_neural_selected") is not None
    ]
    energy_penalties = [
        cmp.get("mean_energy_penalty_when_neural_selected")
        for cmp in comparisons
        if cmp.get("mean_energy_penalty_when_neural_selected") is not None
    ]

    if rate_gains and energy_penalties:
        mean_rate_gain = mean(rate_gains)
        mean_energy_penalty = mean(energy_penalties)
        if mean_rate_gain > 0 and mean_energy_penalty > 0:
            notes.append(
                "When a neural codec is selected as oracle, the "
                "associated full-pool selection reduces rate "
                f"(mean delta_bpp = {mean_rate_gain:.4f} vs classical "
                "selection) at the cost of higher energy "
                f"(mean delta_energy_j = {mean_energy_penalty:.4f}); "
                "this is consistent with neural codecs trading "
                "bandwidth for computational cost in this benchmark."
            )
        elif mean_rate_gain > 0 and mean_energy_penalty <= 0:
            notes.append(
                "When a neural codec is selected as oracle, the "
                "associated full-pool selection reduces rate "
                f"(mean delta_bpp = {mean_rate_gain:.4f}) without an "
                "average energy penalty within this benchmark."
            )

    notes.append(
        "These observations describe patterns within the current "
        "image R-D-E benchmark and do not establish universal "
        "generalization to arbitrary natural-image distributions or "
        "hardware platforms."
    )

    return notes


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_summary_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        rows.append(
            {
                "pool": s["pool"],
                "profile": s["profile"],
                "quality_floor": s["quality_floor"],
                "num_images": s["num_images"],
                "num_images_with_safe_candidate":
                    s["num_images_with_safe_candidate"],
                "neural_selection_count": s["neural_selection_count"],
                "neural_selection_rate":
                    _fmt(s.get("neural_selection_rate")),
                "mean_rate": _fmt(s.get("mean_rate")),
                "mean_quality": _fmt(s.get("mean_quality")),
                "mean_energy": _fmt(s.get("mean_energy")),
                "mean_J_RDE": _fmt(s.get("mean_J_RDE")),
                "median_J_RDE": _fmt(s.get("median_J_RDE")),
                "p90_J_RDE": _fmt(s.get("p90_J_RDE")),
                "max_J_RDE": _fmt(s.get("max_J_RDE")),
                "oracle_codec_counts":
                    "; ".join(
                        f"{k}={v}"
                        for k, v in sorted(
                            s["oracle_codec_counts"].items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    ),
                "oracle_family_counts":
                    "; ".join(
                        f"{k}={v}"
                        for k, v in sorted(
                            s["oracle_family_counts"].items()
                        )
                    ),
            }
        )
    _write_csv(path, rows)


def _write_by_image_csv(
    path: Path,
    selections_by_pool: Dict[str, List[Dict[str, Any]]],
) -> None:
    rows: List[Dict[str, Any]] = []
    for pool, selections in selections_by_pool.items():
        for sel in selections:
            rows.append(
                {
                    "image_id": sel["image_id"],
                    "dataset": sel.get("dataset"),
                    "pool": pool,
                    "profile": sel.get("profile"),
                    "quality_floor": sel.get("quality_floor"),
                    "has_safe_selection": sel["has_safe_selection"],
                    "selected_codec": sel.get("selected_codec"),
                    "selected_config": sel.get("selected_config"),
                    "selected_family": sel.get("selected_family"),
                    "rate": _fmt(sel.get("rate")),
                    "quality": _fmt(sel.get("quality")),
                    "energy": _fmt(sel.get("energy")),
                    "J_RDE": _fmt(sel.get("J_RDE")),
                    "num_candidates": sel.get("num_candidates"),
                    "num_safe_candidates": sel.get("num_safe_candidates"),
                }
            )
    _write_csv(path, rows)


def _write_pool_comparison_csv(
    path: Path,
    comparisons: List[Dict[str, Any]],
) -> None:
    rows = [
        {
            "profile": c["profile"],
            "quality_floor": c["quality_floor"],
            "num_paired_images": c["num_paired_images"],
            "mean_classic_J_RDE": _fmt(c.get("mean_classic_J_RDE")),
            "mean_full_J_RDE": _fmt(c.get("mean_full_J_RDE")),
            "mean_regret_classic_vs_full": _fmt(
                c.get("mean_regret_classic_vs_full")
            ),
            "median_regret_classic_vs_full": _fmt(
                c.get("median_regret_classic_vs_full")
            ),
            "p90_regret_classic_vs_full": _fmt(
                c.get("p90_regret_classic_vs_full")
            ),
            "max_regret_classic_vs_full": _fmt(
                c.get("max_regret_classic_vs_full")
            ),
            "relative_mean_improvement_full_vs_classic": _fmt(
                c.get("relative_mean_improvement_full_vs_classic")
            ),
            "neural_selection_count_in_full":
                c.get("neural_selection_count_in_full"),
            "classic_only_selection_count_in_full":
                c.get("classic_only_selection_count_in_full"),
            "neural_selection_rate_in_full": _fmt(
                c.get("neural_selection_rate_in_full")
            ),
            "mean_rate_gain_when_neural_selected": _fmt(
                c.get("mean_rate_gain_when_neural_selected")
            ),
            "mean_energy_penalty_when_neural_selected": _fmt(
                c.get("mean_energy_penalty_when_neural_selected")
            ),
            "mean_quality_delta_when_neural_selected": _fmt(
                c.get("mean_quality_delta_when_neural_selected")
            ),
        }
        for c in comparisons
    ]
    _write_csv(path, rows)


def _write_per_image_labels_csv(
    path: Path,
    labels: List[Dict[str, Any]],
) -> None:
    rows = [
        {
            "image_id": lab["image_id"],
            "dataset": lab.get("dataset"),
            "profile": lab["profile"],
            "quality_floor": lab["quality_floor"],
            "classic_pool_oracle_label": lab.get("classic_pool_oracle_label"),
            "full_pool_oracle_label": lab.get("full_pool_oracle_label"),
            "full_pool_oracle_family": lab.get("full_pool_oracle_family"),
            "regret_classic_vs_full": _fmt(lab.get("regret_classic_vs_full")),
        }
        for lab in labels
    ]
    _write_csv(path, rows)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.router.analysis.neural_inclusive_oracle",
        description=(
            "Offline neural-inclusive R-D-E oracle audit: per-pool / "
            "per-profile / per-floor oracles, classic-vs-full pool "
            "comparison, and per-group target labels for downstream "
            "neural-inclusive predictor training. Read-only against "
            "an existing aggregate R-D-E CSV; does not change the "
            "router runtime."
        ),
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--out-dir",
        default="results/routing_context/neural_inclusive_oracle",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-summary-csv", default=None)

    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="param")
    parser.add_argument("--rate-col", default="bpp")
    parser.add_argument("--quality-col", default="ssimulacra2")
    parser.add_argument("--energy-col", default="energy_per_image_j")
    parser.add_argument(
        "--image-id-col",
        default="image_id",
        help=(
            "Column that defines the per-group oracle unit. When the "
            "column is absent from the CSV, every row is assigned to "
            "the single synthetic group '__global__'."
        ),
    )
    parser.add_argument("--dataset-col", default=None)

    parser.add_argument(
        "--energy-csv",
        default=None,
        help=(
            "Optional side CSV that provides the energy column. Joined "
            "on (codec, config); rows sharing the same (codec, config) "
            "in the side file are averaged. Use this when the main "
            "metrics file carries quality but not energy."
        ),
    )
    parser.add_argument(
        "--energy-csv-codec-col",
        default=None,
        help="Codec column in --energy-csv; defaults to --codec-col.",
    )
    parser.add_argument(
        "--energy-csv-config-col",
        default=None,
        help="Config column in --energy-csv; defaults to --config-col.",
    )
    parser.add_argument(
        "--energy-csv-col",
        default=None,
        help="Energy column in --energy-csv; defaults to --energy-col.",
    )

    parser.add_argument(
        "--profiles",
        default=",".join(_DEFAULT_PROFILES),
        help=(
            "Comma-separated profile names. Resolved against the "
            "official router profiles in src/router/core/profiles.py."
        ),
    )
    parser.add_argument(
        "--quality-floors",
        default="60,70,80",
        help=(
            "Comma-separated quality floors interpreted in the units "
            "of the --quality-col column."
        ),
    )

    parser.add_argument("--codec-family-map", default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_floor_list(value: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for token in _parse_csv_list(value):
        if token.lower() in {"none", "null", ""}:
            out.append(None)
            continue
        out.append(float(token))
    if not out:
        raise ValueError("--quality-floors must contain at least one entry.")
    return out


def _load_optional_family_map(path: Optional[str]) -> None:
    if path is None:
        return
    extras = _read_codec_family_overrides(path)
    for codec_name, family in extras.items():
        key = _normalize_codec_name(codec_name)
        if family == "classical":
            _FAMILY_CLASSICAL.add(key)
        elif family == "neural":
            _FAMILY_NEURAL.add(key)
        else:
            raise ValueError(
                f"Codec family override for '{codec_name}' must be "
                "'classical' or 'neural'."
            )


def _read_codec_family_overrides(path: str) -> Dict[str, str]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["codec"]: row["family"] for row in reader}


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    _load_optional_family_map(args.codec_family_map)

    rows = load_pool_rows(
        args.csv,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        image_id_col=args.image_id_col,
        dataset_col=args.dataset_col,
        energy_csv_path=args.energy_csv,
        energy_csv_codec_col=args.energy_csv_codec_col,
        energy_csv_config_col=args.energy_csv_config_col,
        energy_csv_col=args.energy_csv_col,
    )

    normalization_stats = normalize_full_pool(rows)

    profile_names = _parse_csv_list(args.profiles)
    quality_floors = _parse_floor_list(args.quality_floors)

    profile_weights: Dict[str, Dict[str, float]] = {
        name: _resolve_profile_weights(name) for name in profile_names
    }

    summaries: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    selections_by_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    labels: List[Dict[str, Any]] = []

    for profile_name in profile_names:
        weights = profile_weights[profile_name]
        for floor in quality_floors:
            classic_selections = evaluate_pool(
                rows,
                pool="classic_pool",
                profile_name=profile_name,
                weights=weights,
                quality_floor=floor,
            )
            full_selections = evaluate_pool(
                rows,
                pool="full_pool",
                profile_name=profile_name,
                weights=weights,
                quality_floor=floor,
            )

            for pool_name, sels in [
                ("classic_pool", classic_selections),
                ("full_pool", full_selections),
            ]:
                selections_by_pool[pool_name].extend(sels)
                summaries.append(
                    build_pool_summary(
                        sels,
                        pool=pool_name,
                        profile_name=profile_name,
                        quality_floor=floor,
                    )
                )

            comparisons.append(
                compute_pool_comparison(
                    classic_selections,
                    full_selections,
                    profile_name=profile_name,
                    quality_floor=floor,
                )
            )
            labels.extend(
                build_per_group_labels(
                    classic_selections=classic_selections,
                    full_selections=full_selections,
                    profile_name=profile_name,
                    quality_floor=floor,
                )
            )

    interpretation = build_interpretation(
        summaries=summaries,
        comparisons=comparisons,
        profiles=profile_names,
        quality_floors=quality_floors,
    )

    codec_inventory = Counter(
        f"{r['codec']}|{r['codec_family']}" for r in rows
    )

    report = {
        "generated_at_unix": int(time.time()),
        "router_target": "neural_inclusive_oracle",
        "inputs": {
            "csv": args.csv,
            "codec_col": args.codec_col,
            "config_col": args.config_col,
            "rate_col": args.rate_col,
            "quality_col": args.quality_col,
            "energy_col": args.energy_col,
            "image_id_col": args.image_id_col,
            "dataset_col": args.dataset_col,
            "codec_family_map": args.codec_family_map,
        },
        "num_rows_loaded": len(rows),
        "codec_inventory": dict(codec_inventory),
        "normalization": {
            "scope": "full_pool_global",
            **normalization_stats,
        },
        "profiles": profile_names,
        "profile_weights": profile_weights,
        "quality_floors": quality_floors,
        "pool_summaries": summaries,
        "pool_comparisons": comparisons,
        "per_group_labels_count": len(labels),
        "interpretation": interpretation,
        "provenance": {
            "normalization_scope": "full_pool_global",
            "uses_router_profiles": True,
            "profile_source": "src.router.core.profiles.PROFILES",
        },
    }
    report["_selections_by_pool"] = dict(selections_by_pool)
    report["_per_group_labels"] = labels
    return report


def _write_outputs(
    report: Dict[str, Any],
    *,
    out_dir: Path,
    out_json: Optional[Path],
    out_summary_csv: Optional[Path],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (
        out_summary_csv
        if out_summary_csv is not None
        else (out_dir / "neural_inclusive_oracle_summary.csv")
    )
    _write_summary_csv(summary_csv, report["pool_summaries"])

    by_image_csv = out_dir / "neural_inclusive_oracle_by_image.csv"
    _write_by_image_csv(by_image_csv, report["_selections_by_pool"])

    pool_comparison_csv = out_dir / "neural_inclusive_pool_comparison.csv"
    _write_pool_comparison_csv(pool_comparison_csv, report["pool_comparisons"])

    labels_csv = out_dir / "neural_inclusive_per_group_labels.csv"
    _write_per_image_labels_csv(labels_csv, report["_per_group_labels"])

    json_path = (
        out_json
        if out_json is not None
        else (out_dir / "neural_inclusive_oracle_report.json")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip internal underscore-prefixed keys from the JSON payload.
    public_report = {
        k: v for k, v in report.items() if not k.startswith("_")
    }
    json_path.write_text(
        json.dumps(public_report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    return {
        "summary_csv": str(summary_csv),
        "by_image_csv": str(by_image_csv),
        "pool_comparison_csv": str(pool_comparison_csv),
        "labels_csv": str(labels_csv),
        "json": str(json_path),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    report = _run(args)
    out_dir = Path(args.out_dir)
    out_json = Path(args.out_json) if args.out_json else None
    out_summary_csv = (
        Path(args.out_summary_csv) if args.out_summary_csv else None
    )
    written = _write_outputs(
        report,
        out_dir=out_dir,
        out_json=out_json,
        out_summary_csv=out_summary_csv,
    )

    print("\n=== R-D-E Neural-Inclusive Oracle Audit ===")
    print(f"Rows loaded:               {report['num_rows_loaded']}")
    print(f"Profiles:                  {', '.join(report['profiles'])}")
    print(f"Quality floors:            {report['quality_floors']}")
    print(f"Codec inventory:           {report['codec_inventory']}")
    print(f"Normalization scope:       {report['normalization']['scope']}")
    print(f"Summary CSV:               {written['summary_csv']}")
    print(f"Pool-comparison CSV:       {written['pool_comparison_csv']}")
    print(f"Per-group labels CSV:      {written['labels_csv']}")
    print(f"JSON report:               {written['json']}")
    for note in report["interpretation"]:
        print(f"- {note}")


if __name__ == "__main__":
    main()
