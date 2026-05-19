"""Operational-regime simulation for predictive R-D-E routing.

This module is an offline, paper/demo-facing audit. It simulates how a
predictive router behaves under different operational regimes without
changing the runtime router, the J_RDE formula, report schemas, CLI flags,
raw benchmark data, or execution backends.

The central no-leakage rule is the same as in the neural-inclusive
predictive-router audit: predictive policies may use training folds and test
image metadata, but they do not inspect the test image's measured R-D-E
candidates while choosing. Test-image R-D-E rows are used only afterwards to
realise the selected pair and compute energy, rate, quality and regret.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.router.analysis.neural_inclusive_oracle import (
    classify_codec_family,
    load_pool_rows,
    normalize_full_pool,
)
from src.router.analysis.neural_inclusive_predictive_router import (
    _derive_metadata_from_rde_rows,
    _fit_numeric_stats,
    _build_metadata_columns,
    _knn_predict_label,
    _label,
    _split_label,
)
from src.router.core.profiles import PROFILES as ROUTER_PROFILES


__all__ = [
    "REGIME_ORDER",
    "POLICIES",
    "build_regime_definitions",
    "evaluate_operational_regimes",
    "build_interpretation",
    "main",
]


Pair = Tuple[str, str]

REGIME_ORDER = [
    "normal",
    "bandwidth_limited",
    "energy_saving",
    "battery_pressure",
    "thermal_pressure",
    "no_cuda",
    "low_memory_or_vram_pressure",
]

POLICIES = [
    "robust_global_full_pool_baseline",
    "metadata_only_full_pool",
    "system_only_full_pool",
    "metadata_plus_system_full_pool",
    "metadata_only_classic_pool",
    "full_pool_oracle",
]

RATE_PRESSURE_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


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


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _relative_reduction(
    policy_value: Optional[float],
    baseline_value: Optional[float],
) -> Optional[float]:
    ratio = _safe_div(policy_value, baseline_value)
    if ratio is None:
        return None
    return 1.0 - ratio


def _fmt(value: Optional[float], digits: int = 8) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalised_profile_weights(name: str) -> Dict[str, float]:
    key = name.strip().lower().replace("_", "-")
    if key not in ROUTER_PROFILES:
        raise ValueError(f"Unknown router profile: {name}")
    profile = ROUTER_PROFILES[key]
    total = profile.w_r + profile.w_e + profile.w_d
    if total <= 0:
        raise ValueError(f"Profile {name!r} has non-positive weights.")
    return {
        "w_R": profile.w_r / total,
        "w_E": profile.w_e / total,
        "w_D": profile.w_d / total,
    }


def _j_rde(row: Dict[str, Any], weights: Dict[str, float]) -> float:
    return (
        weights["w_R"] * row["norm_rate"]
        + weights["w_E"] * row["norm_energy"]
        + weights["w_D"] * row["norm_distortion"]
    )


def _operational_cost(row: Dict[str, Any], regime: Dict[str, Any]) -> float:
    cost = _j_rde(row, regime["weights"])
    if row.get("codec_family") == "neural":
        cost += float(regime.get("neural_penalty") or 0.0)
    return cost


def _pair(row: Dict[str, Any]) -> Pair:
    return str(row["codec"]), str(row["config"])


def _pair_family(pair: Optional[Pair]) -> Optional[str]:
    if pair is None:
        return None
    return classify_codec_family(pair[0])


def _quality_floor_key(value: Optional[float]) -> str:
    return "none" if value is None else _fmt(value)


def _parse_csv_list(value: str) -> List[str]:
    return [token.strip().lower() for token in value.split(",") if token.strip()]


def _parse_quality_floors(value: str) -> List[Optional[float]]:
    floors: List[Optional[float]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "null"}:
            floors.append(None)
        else:
            floors.append(float(token))
    return floors


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = weights["w_R"] + weights["w_E"] + weights["w_D"]
    if total <= 0:
        raise ValueError("Regime weights must have positive sum.")
    return {
        "w_R": weights["w_R"] / total,
        "w_E": weights["w_E"] / total,
        "w_D": weights["w_D"] / total,
    }


def build_regime_definitions() -> Dict[str, Dict[str, Any]]:
    """Return explicit offline regime definitions.

    ``simulation_only`` is true for regimes that are not a one-to-one runtime
    policy. They are transparent stress scenarios over the measured R-D-E CSV.
    """
    normal = _normalised_profile_weights("balanced")
    bandwidth = _normalised_profile_weights("bandwidth-limited")
    energy = _normalised_profile_weights("energy-limited")

    regimes = {
        "normal": {
            "weights": normal,
            "router_profile_source": "balanced",
            "neural_allowed": True,
            "effective_exclude_neural": False,
            "neural_penalty": 0.0,
            "system_penalty": 0.0,
            "max_energy": None,
            "max_time": None,
            "simulation_only": False,
            "description": "Balanced R-D-E regime with no additional system penalty.",
        },
        "bandwidth_limited": {
            "weights": bandwidth,
            "router_profile_source": "bandwidth-limited",
            "neural_allowed": True,
            "effective_exclude_neural": False,
            "neural_penalty": 0.0,
            "system_penalty": 0.0,
            "max_energy": None,
            "max_time": None,
            "simulation_only": False,
            "description": "Rate-constrained regime using the official bandwidth-limited profile.",
        },
        "energy_saving": {
            "weights": energy,
            "router_profile_source": "energy-limited",
            "neural_allowed": True,
            "effective_exclude_neural": False,
            "neural_penalty": 0.0,
            "system_penalty": 0.0,
            "max_energy": None,
            "max_time": None,
            "simulation_only": False,
            "description": "Energy-oriented regime using the official energy-limited profile.",
        },
        "battery_pressure": {
            "weights": _normalise_weights({"w_R": 0.20, "w_E": 0.70, "w_D": 0.10}),
            "router_profile_source": "simulation",
            "neural_allowed": True,
            "effective_exclude_neural": False,
            "neural_penalty": 0.15,
            "system_penalty": 0.15,
            "max_energy": None,
            "max_time": None,
            "simulation_only": True,
            "description": "Offline battery-pressure proxy: high energy weight plus neural-family penalty.",
        },
        "thermal_pressure": {
            "weights": _normalise_weights({"w_R": 0.25, "w_E": 0.55, "w_D": 0.20}),
            "router_profile_source": "simulation",
            "neural_allowed": True,
            "effective_exclude_neural": False,
            "neural_penalty": 0.20,
            "system_penalty": 0.20,
            "max_energy": "norm_energy<=0.85",
            "max_time": None,
            "max_norm_energy": 0.85,
            "simulation_only": True,
            "description": "Offline thermal-pressure proxy: penalises neural candidates and filters very high normalised energy.",
        },
        "no_cuda": {
            "weights": normal,
            "router_profile_source": "simulation",
            "neural_allowed": False,
            "effective_exclude_neural": True,
            "neural_penalty": 0.0,
            "system_penalty": 0.0,
            "max_energy": None,
            "max_time": None,
            "simulation_only": True,
            "description": "Capability-aware no-CUDA proxy: neural-family candidates are excluded.",
        },
        "low_memory_or_vram_pressure": {
            "weights": _normalise_weights({"w_R": 0.25, "w_E": 0.45, "w_D": 0.30}),
            "router_profile_source": "simulation",
            "neural_allowed": True,
            "effective_exclude_neural": False,
            "neural_penalty": 0.18,
            "system_penalty": 0.18,
            "max_energy": None,
            "max_time": None,
            "simulation_only": True,
            "proxy_note": "memory/vram regime is proxy-based and uses neural-family penalty/exclusion.",
            "description": "Offline memory/VRAM-pressure proxy using neural-family penalty because no RAM/VRAM metadata is available.",
        },
    }
    return regimes


# ---------------------------------------------------------------------------
# Pool and feasibility
# ---------------------------------------------------------------------------


def _pool_matches(row: Dict[str, Any], pool: str) -> bool:
    family = row.get("codec_family")
    if pool == "full_pool":
        return family in {"classical", "neural"}
    if pool == "classic_pool":
        return family == "classical"
    if pool == "neural_pool":
        return family == "neural"
    raise ValueError(f"Unknown pool: {pool}")


def _row_allowed_by_regime(row: Dict[str, Any], regime: Dict[str, Any]) -> bool:
    if not regime.get("neural_allowed", True) and row.get("codec_family") == "neural":
        return False
    max_norm_energy = regime.get("max_norm_energy")
    if max_norm_energy is not None and row.get("norm_energy", 0.0) > float(max_norm_energy):
        return False
    return True


def _rows_for_selection(
    rows: Iterable[Dict[str, Any]],
    *,
    pool: str,
    regime: Dict[str, Any],
    quality_floor: Optional[float],
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if not _pool_matches(row, pool):
            continue
        if not _row_allowed_by_regime(row, regime):
            continue
        if quality_floor is not None and row["quality"] < quality_floor:
            continue
        out.append(row)
    return out


def _pair_allowed_by_regime(
    pair: Optional[Pair],
    *,
    regime: Dict[str, Any],
    training_rows: List[Dict[str, Any]],
) -> bool:
    if pair is None:
        return False
    family = classify_codec_family(pair[0])
    if not regime.get("neural_allowed", True) and family == "neural":
        return False
    max_norm_energy = regime.get("max_norm_energy")
    if max_norm_energy is None:
        return True
    candidates = [
        r for r in training_rows
        if r["codec"] == pair[0] and r["config"] == pair[1]
    ]
    if not candidates:
        return True
    return mean(float(r["norm_energy"]) for r in candidates) <= float(max_norm_energy)


def _candidate_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    return {
        (str(r["image_id"]), str(r["codec"]), str(r["config"])): r
        for r in rows
    }


def _loio_training_rows(rows: List[Dict[str, Any]], image_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["image_id"] != image_id]


def _lodo_training_rows(rows: List[Dict[str, Any]], dataset: str) -> List[Dict[str, Any]]:
    return [r for r in rows if str(r.get("dataset")) != str(dataset)]


# ---------------------------------------------------------------------------
# Oracles and policies
# ---------------------------------------------------------------------------


def _oracle_for_image(
    rows: List[Dict[str, Any]],
    *,
    image_id: str,
    pool: str,
    regime: Dict[str, Any],
    quality_floor: Optional[float],
) -> Optional[Dict[str, Any]]:
    candidates = _rows_for_selection(
        [r for r in rows if r["image_id"] == image_id],
        pool=pool,
        regime=regime,
        quality_floor=quality_floor,
    )
    if not candidates:
        return None
    best = min(candidates, key=lambda r: (_operational_cost(r, regime), r["codec"], r["config"]))
    return {
        "image_id": image_id,
        "dataset": best.get("dataset"),
        "codec": best["codec"],
        "config": best["config"],
        "family": best["codec_family"],
        "rate": best["rate"],
        "quality": best["quality"],
        "energy": best["energy"],
        "cost": _operational_cost(best, regime),
        "pair": _pair(best),
    }


def _oracle_labels(
    rows: List[Dict[str, Any]],
    *,
    pool: str,
    regime: Dict[str, Any],
    quality_floor: Optional[float],
) -> Dict[str, Optional[Dict[str, Any]]]:
    labels: Dict[str, Optional[Dict[str, Any]]] = {}
    for image_id in sorted({r["image_id"] for r in rows}):
        labels[image_id] = _oracle_for_image(
            rows,
            image_id=image_id,
            pool=pool,
            regime=regime,
            quality_floor=quality_floor,
        )
    return labels


def _policy_robust_global(
    *,
    training_rows: List[Dict[str, Any]],
    pool: str,
    regime: Dict[str, Any],
    quality_floor: Optional[float],
) -> Optional[Pair]:
    usable = _rows_for_selection(
        training_rows,
        pool=pool,
        regime=regime,
        quality_floor=quality_floor,
    )
    if not usable:
        return None
    training_images = {r["image_id"] for r in training_rows}
    grouped: Dict[Pair, List[Dict[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[_pair(row)].append(row)

    best_pair: Optional[Pair] = None
    best_score: Optional[Tuple[int, float, str, str]] = None
    for pair, candidates in grouped.items():
        covered = len({r["image_id"] for r in candidates})
        mean_cost = mean(_operational_cost(r, regime) for r in candidates)
        full_coverage_bonus = 1 if covered == len(training_images) else 0
        score = (full_coverage_bonus, covered, -mean_cost, pair[0] + pair[1])
        if best_score is None or score > best_score:
            best_score = score
            best_pair = pair
    return best_pair


def _policy_knn_metadata(
    *,
    training_rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    pool: str,
    regime: Dict[str, Any],
    quality_floor: Optional[float],
    test_meta: Dict[str, Any],
    k: int,
) -> Tuple[Optional[Pair], Optional[float], Optional[Pair]]:
    labels = _oracle_labels(
        training_rows,
        pool=pool,
        regime=regime,
        quality_floor=quality_floor,
    )
    training_metas: List[Dict[str, Any]] = []
    training_labels: List[str] = []
    for image_id, oracle in labels.items():
        if oracle is None:
            continue
        meta = metadata_by_image.get(image_id)
        if meta is None:
            continue
        training_metas.append(meta)
        training_labels.append(_label(oracle["pair"]))

    fallback = _policy_robust_global(
        training_rows=training_rows,
        pool=pool,
        regime=regime,
        quality_floor=quality_floor,
    )
    if not training_metas:
        return None, None, fallback

    _columns, levels = _build_metadata_columns(training_metas + [test_meta])
    numeric_stats = _fit_numeric_stats(training_metas)
    label, confidence = _knn_predict_label(
        test_meta=test_meta,
        training_metas=training_metas,
        training_labels=training_labels,
        columns=_columns,
        levels=levels,
        numeric_stats=numeric_stats,
        k=k,
    )
    return _split_label(label), confidence, fallback


def _policy_context(
    *,
    policy: str,
    regime: Dict[str, Any],
    normal_regime: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], bool]:
    """Return ``(pool, selection_regime, enforce_active_system)``."""
    if policy == "robust_global_full_pool_baseline":
        return "full_pool", normal_regime, True
    if policy == "metadata_only_full_pool":
        return "full_pool", normal_regime, True
    if policy == "system_only_full_pool":
        return "full_pool", regime, False
    if policy == "metadata_plus_system_full_pool":
        return "full_pool", regime, False
    if policy == "metadata_only_classic_pool":
        return "classic_pool", regime, False
    if policy == "full_pool_oracle":
        return "full_pool", regime, False
    raise ValueError(f"Unknown policy: {policy}")


def _select_pair(
    *,
    policy: str,
    training_rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    image_id: str,
    pool: str,
    selection_regime: Dict[str, Any],
    active_regime: Dict[str, Any],
    normal_regime: Dict[str, Any],
    quality_floor: Optional[float],
    k: int,
    oracle: Optional[Dict[str, Any]],
    enforce_active_system: bool,
) -> Tuple[Optional[Pair], Optional[float], bool, str]:
    if policy == "full_pool_oracle":
        return (oracle["pair"] if oracle else None), 1.0 if oracle else None, False, "oracle_reference_no_fallback"

    if policy in {"robust_global_full_pool_baseline", "system_only_full_pool"}:
        pair = _policy_robust_global(
            training_rows=training_rows,
            pool=pool,
            regime=selection_regime,
            quality_floor=quality_floor,
        )
        fallback = _policy_robust_global(
            training_rows=training_rows,
            pool="full_pool",
            regime=active_regime,
            quality_floor=quality_floor,
        )
        selected, used_fallback, reason = _apply_system_fallback(
            pair=pair,
            fallback=fallback,
            training_rows=training_rows,
            active_regime=active_regime,
            enforce_active_system=enforce_active_system,
            default_reason="no_training_baseline",
        )
        return selected, None, used_fallback, reason

    if policy in {
        "metadata_only_full_pool",
        "metadata_plus_system_full_pool",
        "metadata_only_classic_pool",
    }:
        test_meta = metadata_by_image.get(image_id)
        if test_meta is None:
            fallback = _policy_robust_global(
                training_rows=training_rows,
                pool=pool,
                regime=selection_regime,
                quality_floor=quality_floor,
            )
            return fallback, None, True, "no_metadata_for_test_image"
        pair, confidence, fallback = _policy_knn_metadata(
            training_rows=training_rows,
            metadata_by_image=metadata_by_image,
            pool=pool,
            regime=selection_regime,
            quality_floor=quality_floor,
            test_meta=test_meta,
            k=k,
        )
        if pair is None:
            return fallback, confidence, True, "knn_no_training_labels"
        active_fallback = fallback
        if enforce_active_system:
            active_fallback = _policy_robust_global(
                training_rows=training_rows,
                pool="full_pool",
                regime=active_regime,
                quality_floor=quality_floor,
            )
        selected, used, reason = _apply_system_fallback(
            pair=pair,
            fallback=active_fallback,
            training_rows=training_rows,
            active_regime=active_regime,
            enforce_active_system=enforce_active_system,
            default_reason="",
        )
        return selected, confidence, used, reason

    raise ValueError(f"Unknown policy: {policy}")


def _apply_system_fallback(
    *,
    pair: Optional[Pair],
    fallback: Optional[Pair],
    training_rows: List[Dict[str, Any]],
    active_regime: Dict[str, Any],
    enforce_active_system: bool,
    default_reason: str,
) -> Tuple[Optional[Pair], bool, str]:
    if pair is None:
        return fallback, True, default_reason
    if enforce_active_system and not _pair_allowed_by_regime(
        pair,
        regime=active_regime,
        training_rows=training_rows,
    ):
        return fallback, True, "system_context_rejected_predicted_pair"
    return pair, False, default_reason


def _realise_decision(
    *,
    pair: Optional[Pair],
    image_id: str,
    dataset: str,
    protocol: str,
    regime_name: str,
    policy: str,
    quality_floor: Optional[float],
    confidence: Optional[float],
    fallback_used: bool,
    fallback_reason: str,
    candidate_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    active_regime: Dict[str, Any],
    oracle: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    base = {
        "image_id": image_id,
        "dataset": dataset,
        "protocol": protocol,
        "regime": regime_name,
        "policy": policy,
        "quality_floor": quality_floor,
        "selected_codec": None,
        "selected_config": None,
        "selected_family": None,
        "oracle_codec": oracle["codec"] if oracle else None,
        "oracle_config": oracle["config"] if oracle else None,
        "oracle_family": oracle["family"] if oracle else None,
        "oracle_status": "feasible" if oracle else "infeasible",
        "selected_cost": None,
        "oracle_cost": oracle["cost"] if oracle else None,
        "regret": None,
        "excluded_from_regret": oracle is None,
        "selected_energy": None,
        "selected_rate": None,
        "selected_quality": None,
        "oracle_quality": oracle["quality"] if oracle else None,
        "selected_quality_margin": None,
        "oracle_quality_margin": (
            oracle["quality"] - quality_floor
            if oracle is not None and quality_floor is not None
            else None
        ),
        "baseline_quality_margin": None,
        "realized_quality_violation": False,
        "quality_violation": False,
        "exact_match": None,
        "family_match": None,
        "confidence": confidence,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "provenance": "no_predicted_pair_available",
    }
    if pair is None:
        return base

    row = candidate_index.get((image_id, pair[0], pair[1]))
    family = classify_codec_family(pair[0])
    base.update(
        {
            "selected_codec": pair[0],
            "selected_config": pair[1],
            "selected_family": family,
        }
    )
    if row is None:
        base["provenance"] = "predicted_pair_not_present_on_test_image"
        return base

    selected_raw_objective = _operational_cost(row, active_regime)
    quality_violation = (
        quality_floor is not None and row["quality"] < quality_floor
    )
    selected_cost = selected_raw_objective
    objective_consistency_note = ""
    # Regret is defined against the oracle over the feasible candidate set.
    # A predictive no-leakage choice may realise below the target quality
    # floor; in that case its raw objective can be numerically lower than the
    # feasible oracle, but it is not an admissible improvement. Keep the raw
    # objective for objective-gain diagnostics and clamp the comparison
    # objective to the oracle lower bound for regret accounting.
    if oracle is not None and selected_cost < oracle["cost"] - 1e-9:
        selected_cost = oracle["cost"]
        objective_consistency_note = "selected_raw_objective_below_feasible_oracle_clamped_for_regret"
    regret = None
    if oracle is not None:
        regret = selected_cost - oracle["cost"]
    base.update(
        {
            "selected_cost": selected_cost,
            "selected_raw_objective": selected_raw_objective,
            "objective_consistency_note": objective_consistency_note,
            "regret": regret,
            "selected_energy": row["energy"],
            "selected_rate": row["rate"],
            "selected_quality": row["quality"],
            "selected_quality_margin": (
                row["quality"] - quality_floor
                if quality_floor is not None
                else None
            ),
            "oracle_quality": oracle["quality"] if oracle else None,
            "oracle_quality_margin": (
                oracle["quality"] - quality_floor
                if oracle is not None and quality_floor is not None
                else None
            ),
            "realized_quality_violation": quality_violation,
            "quality_violation": quality_violation,
            "exact_match": (
                oracle is not None
                and pair[0] == oracle["codec"]
                and pair[1] == oracle["config"]
            ),
            "family_match": oracle is not None and family == oracle["family"],
            "provenance": "predicted_pair_realised_on_test_image",
        }
    )
    return base


# ---------------------------------------------------------------------------
# Summaries and plot-data tables
# ---------------------------------------------------------------------------


def _summarise_decisions(
    decisions: List[Dict[str, Any]],
    *,
    regime: str,
    policy: str,
    protocol: str,
    quality_floor: Optional[float],
) -> Dict[str, Any]:
    n = len(decisions)
    regret_eligible = [d for d in decisions if not d.get("excluded_from_regret")]
    regrets = [float(d["regret"]) for d in regret_eligible if d.get("regret") is not None]
    objectives = [
        float(d["selected_raw_objective"])
        for d in decisions
        if d.get("selected_raw_objective") is not None
    ]
    energies = [float(d["selected_energy"]) for d in decisions if d.get("selected_energy") is not None]
    rates = [float(d["selected_rate"]) for d in decisions if d.get("selected_rate") is not None]
    qualities = [float(d["selected_quality"]) for d in decisions if d.get("selected_quality") is not None]
    selected_families = [d.get("selected_family") for d in decisions if d.get("selected_family")]
    oracle_families = [d.get("oracle_family") for d in decisions if d.get("oracle_family")]
    neural_pred = sum(1 for f in selected_families if f == "neural")
    neural_oracle = sum(1 for f in oracle_families if f == "neural")
    tp = sum(
        1 for d in decisions
        if d.get("selected_family") == "neural" and d.get("oracle_family") == "neural"
    )
    fp = sum(
        1 for d in decisions
        if d.get("selected_family") == "neural" and d.get("oracle_family") == "classical"
    )
    fn = sum(
        1 for d in decisions
        if d.get("selected_family") == "classical" and d.get("oracle_family") == "neural"
    )
    return {
        "regime": regime,
        "policy": policy,
        "protocol": protocol,
        "quality_floor": quality_floor,
        "num_images": n,
        "num_regret_eligible": len(regret_eligible),
        "infeasible_rate": (
            sum(1 for d in decisions if d.get("oracle_status") == "infeasible") / n
            if n else None
        ),
        "mean_energy": mean(energies) if energies else None,
        "energy_saving_vs_global_baseline": None,
        "mean_rate": mean(rates) if rates else None,
        "rate_reduction_vs_global_baseline": None,
        "mean_quality": mean(qualities) if qualities else None,
        "quality_violation_rate": (
            sum(1 for d in decisions if d.get("quality_violation") is True) / n
            if n else None
        ),
        "mean_selected_objective": mean(objectives) if objectives else None,
        "objective_gain_vs_global_baseline": None,
        "mean_regret": mean(regrets) if regrets else None,
        "median_regret": median(regrets) if regrets else None,
        "p90_regret": _quantile(regrets, 0.90) if regrets else None,
        "max_regret": max(regrets) if regrets else None,
        "regret_reduction_vs_global_baseline": None,
        "neural_selection_rate": neural_pred / n if n else None,
        "oracle_neural_rate": neural_oracle / n if n else None,
        "neural_family_precision": tp / (tp + fp) if (tp + fp) else None,
        "neural_family_recall": tp / (tp + fn) if (tp + fn) else None,
        "exact_match_rate": (
            sum(1 for d in decisions if d.get("exact_match") is True) / n
            if n else None
        ),
        "family_match_rate": (
            sum(1 for d in decisions if d.get("family_match") is True) / n
            if n else None
        ),
        "fallback_rate": (
            sum(1 for d in decisions if d.get("fallback_used") is True) / n
            if n else None
        ),
    }


def _attach_baseline_reductions(summaries: List[Dict[str, Any]]) -> None:
    baseline_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in summaries:
        if row["policy"] == "robust_global_full_pool_baseline":
            baseline_by_key[
                (row["regime"], row["protocol"], _quality_floor_key(row["quality_floor"]))
            ] = row

    for row in summaries:
        baseline = baseline_by_key.get(
            (row["regime"], row["protocol"], _quality_floor_key(row["quality_floor"]))
        )
        if not baseline:
            continue
        row["energy_saving_vs_global_baseline"] = _relative_reduction(
            row.get("mean_energy"), baseline.get("mean_energy")
        )
        row["rate_reduction_vs_global_baseline"] = _relative_reduction(
            row.get("mean_rate"), baseline.get("mean_rate")
        )
        row["regret_reduction_vs_global_baseline"] = _relative_reduction(
            row.get("mean_regret"), baseline.get("mean_regret")
        )
        row["objective_gain_vs_global_baseline"] = _relative_reduction(
            row.get("mean_selected_objective"),
            baseline.get("mean_selected_objective"),
        )


def _plot_data_rows(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        row = {
            "regime": s["regime"],
            "policy": s["policy"],
            "protocol": s["protocol"],
            "quality_floor": s["quality_floor"],
            "mean_energy": s["mean_energy"],
            "mean_rate": s["mean_rate"],
            "mean_quality": s["mean_quality"],
            "infeasible_rate": s["infeasible_rate"],
            "mean_regret": s["mean_regret"],
            "energy_saving_vs_global_baseline": s["energy_saving_vs_global_baseline"],
            "rate_reduction_vs_global_baseline": s["rate_reduction_vs_global_baseline"],
            "regret_reduction_vs_global_baseline": s["regret_reduction_vs_global_baseline"],
            "quality_violation_rate": s["quality_violation_rate"],
            "mean_selected_objective": s["mean_selected_objective"],
            "objective_gain_vs_global_baseline": s["objective_gain_vs_global_baseline"],
            "neural_selection_rate": s["neural_selection_rate"],
            "oracle_neural_rate": s["oracle_neural_rate"],
            "neural_family_precision": s["neural_family_precision"],
            "neural_family_recall": s["neural_family_recall"],
            "exact_match_rate": s["exact_match_rate"],
            "family_match_rate": s["family_match_rate"],
            "x_energy_saving": s["energy_saving_vs_global_baseline"],
            "y_regret_reduction": s["regret_reduction_vs_global_baseline"],
        }
        rows.append(row)
    return rows


def _winner_distribution(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for d in decisions:
        key = (d["regime"], d["policy"], d["protocol"], _quality_floor_key(d["quality_floor"]))
        selected = (
            d.get("selected_family") or "none",
            d.get("selected_codec") or "none",
            d.get("selected_config") or "none",
        )
        grouped[key][selected] += 1
        totals[key] += 1
    rows: List[Dict[str, Any]] = []
    for key, counter in sorted(grouped.items()):
        total = totals[key]
        for selected, count in sorted(counter.items()):
            rows.append(
                {
                    "regime": key[0],
                    "policy": key[1],
                    "protocol": key[2],
                    "quality_floor": key[3],
                    "selected_family": selected[0],
                    "selected_codec": selected[1],
                    "selected_config": selected[2],
                    "count": count,
                    "selection_rate": count / total if total else None,
                }
            )
    return rows


def _oracle_vs_prediction(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for d in decisions:
        grouped[
            (d["regime"], d["policy"], d["protocol"], _quality_floor_key(d["quality_floor"]))
        ].append(d)
    rows: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        families = sorted(
            {
                "classical",
                "neural",
                *(str(d.get("oracle_family")) for d in items if d.get("oracle_family")),
                *(str(d.get("selected_family")) for d in items if d.get("selected_family")),
            }
        )
        total = len(items)
        for family in families:
            oracle_count = sum(1 for d in items if d.get("oracle_family") == family)
            predicted_count = sum(1 for d in items if d.get("selected_family") == family)
            oracle_rate = oracle_count / total if total else None
            predicted_rate = predicted_count / total if total else None
            rows.append(
                {
                    "regime": key[0],
                    "policy": key[1],
                    "protocol": key[2],
                    "quality_floor": key[3],
                    "family": family,
                    "oracle_count": oracle_count,
                    "oracle_rate": oracle_rate,
                    "predicted_count": predicted_count,
                    "predicted_rate": predicted_rate,
                    "prediction_gap": (
                        predicted_rate - oracle_rate
                        if predicted_rate is not None and oracle_rate is not None
                        else None
                    ),
                }
            )
    return rows


def _family_confusion(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for d in decisions:
        key = (d["regime"], d["policy"], d["protocol"], _quality_floor_key(d["quality_floor"]))
        oracle_family = d.get("oracle_family") or "none"
        predicted_family = d.get("selected_family") or "none"
        grouped[key][(oracle_family, predicted_family)] += 1
        totals[key] += 1
    rows: List[Dict[str, Any]] = []
    for key, counter in sorted(grouped.items()):
        total = totals[key]
        for (oracle_family, predicted_family), count in sorted(counter.items()):
            rows.append(
                {
                    "regime": key[0],
                    "policy": key[1],
                    "protocol": key[2],
                    "quality_floor": key[3],
                    "oracle_family": oracle_family,
                    "predicted_family": predicted_family,
                    "count": count,
                    "rate": count / total if total else None,
                }
            )
    return rows


def _attach_baseline_quality_margins(decisions: List[Dict[str, Any]]) -> None:
    baseline_by_key = {
        (
            d["image_id"],
            d["regime"],
            d["protocol"],
            _quality_floor_key(d["quality_floor"]),
        ): d
        for d in decisions
        if d["policy"] == "robust_global_full_pool_baseline"
    }
    for d in decisions:
        baseline = baseline_by_key.get(
            (
                d["image_id"],
                d["regime"],
                d["protocol"],
                _quality_floor_key(d["quality_floor"]),
            )
        )
        if baseline is None:
            continue
        d["baseline_quality_margin"] = baseline.get("selected_quality_margin")


def build_oracle_quality_contract(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    oracle_rows = [d for d in decisions if d["policy"] == "full_pool_oracle"]
    violation_count = sum(
        1 for d in oracle_rows
        if d.get("oracle_status") == "feasible"
        and d.get("oracle_quality_margin") is not None
        and float(d["oracle_quality_margin"]) < -1e-9
    )
    return {
        "hard_quality_floor": True,
        "infeasible_cases_reported_separately": True,
        "oracle_never_violates_quality_floor": violation_count == 0,
        "oracle_violation_count": violation_count,
        "infeasible_case_count": sum(
            1 for d in oracle_rows if d.get("oracle_status") == "infeasible"
        ),
    }


# ---------------------------------------------------------------------------
# Neural/classical switch analysis
# ---------------------------------------------------------------------------


SWITCH_REASONS = [
    "neural_necessary_for_quality",
    "neural_rde_efficient",
    "classical_sufficient",
    "neural_too_energy_expensive",
    "no_neural_feasible",
    "no_classic_feasible",
    "no_feasible_candidate",
]


def _best_family_candidate(
    rows: List[Dict[str, Any]],
    *,
    image_id: str,
    family: str,
    regime: Dict[str, Any],
    quality_floor: Optional[float],
) -> Optional[Dict[str, Any]]:
    candidates = [
        r for r in rows
        if r["image_id"] == image_id
        and r["codec_family"] == family
        and _row_allowed_by_regime(r, regime)
        and (quality_floor is None or r["quality"] >= quality_floor)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (_operational_cost(r, regime), r["codec"], r["config"]))


def _best_family_candidate_ignoring_floor(
    rows: List[Dict[str, Any]],
    *,
    image_id: str,
    family: str,
    regime: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    candidates = [
        r for r in rows
        if r["image_id"] == image_id
        and r["codec_family"] == family
        and _row_allowed_by_regime(r, regime)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (_operational_cost(r, regime), r["codec"], r["config"]))


def _switch_reason(
    *,
    best_classic: Optional[Dict[str, Any]],
    best_neural: Optional[Dict[str, Any]],
    floorless_classic: Optional[Dict[str, Any]],
    floorless_neural: Optional[Dict[str, Any]],
) -> str:
    if best_classic is None and best_neural is None:
        if floorless_classic is not None and floorless_neural is not None:
            return "no_feasible_candidate"
        if floorless_neural is None and floorless_classic is not None:
            return "no_neural_feasible"
        if floorless_classic is None and floorless_neural is not None:
            return "no_classic_feasible"
        return "no_feasible_candidate"
    if best_classic is None:
        return "neural_necessary_for_quality"
    if best_neural is None:
        return "no_neural_feasible"

    delta_rate = best_neural["rate"] - best_classic["rate"]
    delta_energy = best_neural["energy"] - best_classic["energy"]
    delta_j = _operational_cost(best_neural, best_neural["active_regime"]) - _operational_cost(
        best_classic, best_classic["active_regime"]
    )
    if delta_j < -1e-12:
        return "neural_rde_efficient"
    if delta_rate < 0 and delta_energy > 0 and delta_j >= -1e-12:
        return "neural_too_energy_expensive"
    return "classical_sufficient"


def build_switch_analysis(
    *,
    rows: List[Dict[str, Any]],
    quality_floors: List[Optional[float]],
    protocols: List[str],
    regimes: Dict[str, Dict[str, Any]],
    rate_weight: Optional[float] = None,
    rate_regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    image_ids = sorted({r["image_id"] for r in rows})
    image_to_dataset: Dict[str, str] = {}
    for row in rows:
        image_to_dataset.setdefault(row["image_id"], str(row.get("dataset")))

    by_image: List[Dict[str, Any]] = []
    regime_items = (
        [("rate_pressure", rate_regime)]
        if rate_regime is not None
        else [(name, regimes[name]) for name in REGIME_ORDER]
    )
    for protocol in protocols:
        for regime_name, regime in regime_items:
            assert regime is not None
            for floor in quality_floors:
                for image_id in image_ids:
                    best_classic = _best_family_candidate(
                        rows,
                        image_id=image_id,
                        family="classical",
                        regime=regime,
                        quality_floor=floor,
                    )
                    best_neural = _best_family_candidate(
                        rows,
                        image_id=image_id,
                        family="neural",
                        regime=regime,
                        quality_floor=floor,
                    )
                    floorless_classic = _best_family_candidate_ignoring_floor(
                        rows,
                        image_id=image_id,
                        family="classical",
                        regime=regime,
                    )
                    floorless_neural = _best_family_candidate_ignoring_floor(
                        rows,
                        image_id=image_id,
                        family="neural",
                        regime=regime,
                    )
                    # Carry the active regime on the row references so the
                    # reason helper can compare the same objective without
                    # changing the public row schema.
                    if best_classic is not None:
                        best_classic = {**best_classic, "active_regime": regime}
                    if best_neural is not None:
                        best_neural = {**best_neural, "active_regime": regime}
                    reason = _switch_reason(
                        best_classic=best_classic,
                        best_neural=best_neural,
                        floorless_classic=floorless_classic,
                        floorless_neural=floorless_neural,
                    )
                    winner = best_neural if reason in {
                        "neural_necessary_for_quality",
                        "neural_rde_efficient",
                    } else best_classic
                    if reason == "no_classic_feasible":
                        winner = best_neural
                    if reason in {"no_neural_feasible", "no_feasible_candidate"}:
                        winner = best_classic
                    row = _switch_row(
                        image_id=image_id,
                        dataset=image_to_dataset[image_id],
                        protocol=protocol,
                        regime=regime_name,
                        quality_floor=floor,
                        rate_weight=rate_weight,
                        best_classic=best_classic,
                        best_neural=best_neural,
                        winner=winner,
                        switch_reason=reason,
                    )
                    by_image.append(row)
    return {
        "by_image": by_image,
        "summary": _switch_summary(by_image),
        "reason_by_rate_weight": _switch_reason_by_rate_weight(by_image),
        "tradeoff_scatter": _switch_tradeoff_scatter(by_image),
        "quality_floor_summary": _quality_floor_switch_summary(by_image),
    }


def _switch_row(
    *,
    image_id: str,
    dataset: str,
    protocol: str,
    regime: str,
    quality_floor: Optional[float],
    rate_weight: Optional[float],
    best_classic: Optional[Dict[str, Any]],
    best_neural: Optional[Dict[str, Any]],
    winner: Optional[Dict[str, Any]],
    switch_reason: str,
) -> Dict[str, Any]:
    classic_j = _operational_cost(best_classic, best_classic["active_regime"]) if best_classic else None
    neural_j = _operational_cost(best_neural, best_neural["active_regime"]) if best_neural else None
    classic_rate = best_classic["rate"] if best_classic else None
    neural_rate = best_neural["rate"] if best_neural else None
    classic_quality = best_classic["quality"] if best_classic else None
    neural_quality = best_neural["quality"] if best_neural else None
    classic_energy = best_classic["energy"] if best_classic else None
    neural_energy = best_neural["energy"] if best_neural else None
    return {
        "image_id": image_id,
        "dataset": dataset,
        "regime": regime,
        "protocol": protocol,
        "quality_floor": quality_floor,
        "rate_weight": rate_weight,
        "best_classic_codec": best_classic["codec"] if best_classic else None,
        "best_classic_config": best_classic["config"] if best_classic else None,
        "best_classic_rate": classic_rate,
        "best_classic_quality": classic_quality,
        "best_classic_energy": classic_energy,
        "best_classic_J": classic_j,
        "best_neural_codec": best_neural["codec"] if best_neural else None,
        "best_neural_config": best_neural["config"] if best_neural else None,
        "best_neural_rate": neural_rate,
        "best_neural_quality": neural_quality,
        "best_neural_energy": neural_energy,
        "best_neural_J": neural_j,
        "delta_rate_neural_minus_classic": (
            neural_rate - classic_rate
            if neural_rate is not None and classic_rate is not None
            else None
        ),
        "bitrate_reduction_neural_vs_classic": (
            classic_rate - neural_rate
            if neural_rate is not None and classic_rate is not None
            else None
        ),
        "delta_quality_neural_minus_classic": (
            neural_quality - classic_quality
            if neural_quality is not None and classic_quality is not None
            else None
        ),
        "delta_energy_neural_minus_classic": (
            neural_energy - classic_energy
            if neural_energy is not None and classic_energy is not None
            else None
        ),
        "delta_J_neural_minus_classic": (
            neural_j - classic_j
            if neural_j is not None and classic_j is not None
            else None
        ),
        "winner_family": winner["codec_family"] if winner else None,
        "winner_codec": winner["codec"] if winner else None,
        "switch_reason": switch_reason,
    }


def _switch_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["protocol"], row["quality_floor"], row["rate_weight"])].append(row)
    out: List[Dict[str, Any]] = []
    for (regime, protocol, floor, rate_weight), items in sorted(grouped.items(), key=lambda x: str(x[0])):
        n = len(items)
        reason_counts = Counter(item["switch_reason"] for item in items)
        neural_wins = sum(1 for item in items if item.get("winner_family") == "neural")
        classic_wins = sum(1 for item in items if item.get("winner_family") == "classical")
        neural_win_items = [i for i in items if i.get("winner_family") == "neural"]
        neural_lose_items = [i for i in items if i.get("winner_family") != "neural"]
        out.append(
            {
                "regime": regime,
                "protocol": protocol,
                "quality_floor": floor,
                "rate_weight": rate_weight,
                "num_images": n,
                "neural_win_rate": neural_wins / n if n else None,
                "classic_win_rate": classic_wins / n if n else None,
                "neural_necessary_rate": reason_counts["neural_necessary_for_quality"] / n if n else None,
                "neural_rde_efficient_rate": reason_counts["neural_rde_efficient"] / n if n else None,
                "classical_sufficient_rate": reason_counts["classical_sufficient"] / n if n else None,
                "neural_too_energy_expensive_rate": reason_counts["neural_too_energy_expensive"] / n if n else None,
                "no_neural_feasible_rate": reason_counts["no_neural_feasible"] / n if n else None,
                "no_classic_feasible_rate": reason_counts["no_classic_feasible"] / n if n else None,
                "mean_bitrate_reduction_when_neural_wins": _mean_field(neural_win_items, "bitrate_reduction_neural_vs_classic"),
                "mean_energy_penalty_when_neural_wins": _mean_field(neural_win_items, "delta_energy_neural_minus_classic"),
                "mean_quality_delta_when_neural_wins": _mean_field(neural_win_items, "delta_quality_neural_minus_classic"),
                "mean_bitrate_reduction_when_neural_loses": _mean_field(neural_lose_items, "bitrate_reduction_neural_vs_classic"),
                "mean_energy_penalty_when_neural_loses": _mean_field(neural_lose_items, "delta_energy_neural_minus_classic"),
                "mean_quality_delta_when_neural_loses": _mean_field(neural_lose_items, "delta_quality_neural_minus_classic"),
            }
        )
    return out


def _mean_field(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
    values = [float(r[field]) for r in rows if r.get(field) is not None]
    return mean(values) if values else None


def _switch_reason_by_rate_weight(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("rate_weight") is None:
            continue
        grouped[(row["protocol"], row["quality_floor"], row["rate_weight"])].append(row)
    out: List[Dict[str, Any]] = []
    for (protocol, floor, rate_weight), items in sorted(grouped.items(), key=lambda x: str(x[0])):
        n = len(items)
        counts = Counter(item["switch_reason"] for item in items)
        for reason in SWITCH_REASONS:
            out.append(
                {
                    "protocol": protocol,
                    "quality_floor": floor,
                    "rate_weight": rate_weight,
                    "switch_reason": reason,
                    "count": counts[reason],
                    "share": counts[reason] / n if n else None,
                }
            )
    return out


def _switch_tradeoff_scatter(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "image_id": row["image_id"],
            "dataset": row["dataset"],
            "regime": row["regime"],
            "protocol": row["protocol"],
            "quality_floor": row["quality_floor"],
            "rate_weight": row["rate_weight"],
            "bitrate_reduction_neural_vs_classic": row["bitrate_reduction_neural_vs_classic"],
            "delta_energy_neural_minus_classic": row["delta_energy_neural_minus_classic"],
            "delta_quality_neural_minus_classic": row["delta_quality_neural_minus_classic"],
            "winner_family": row["winner_family"],
            "switch_reason": row["switch_reason"],
        }
        for row in rows
        if row["bitrate_reduction_neural_vs_classic"] is not None
        and row["delta_energy_neural_minus_classic"] is not None
    ]


def _quality_floor_switch_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("rate_weight") is not None:
            continue
        grouped[(row["quality_floor"], row["protocol"])].append(row)
    out: List[Dict[str, Any]] = []
    for (floor, protocol), items in sorted(grouped.items(), key=lambda x: str(x[0])):
        n = len(items)
        neural_necessary = sum(1 for i in items if i["switch_reason"] == "neural_necessary_for_quality")
        neural_wins = sum(1 for i in items if i.get("winner_family") == "neural")
        out.append(
            {
                "quality_metric": "input_quality_col",
                "quality_floor": floor,
                "protocol": protocol,
                "neural_necessary_rate": neural_necessary / n if n else None,
                "neural_win_rate": neural_wins / n if n else None,
            }
        )
    return out


def _rate_pressure_switch_transition(
    *,
    switch_summary: List[Dict[str, Any]],
    rate_pressure_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    weights = sorted({float(r["rate_weight"]) for r in switch_summary if r.get("rate_weight") is not None})
    first_feasible = None
    first_neural_wins = None
    for weight in weights:
        rows_at = [r for r in switch_summary if r.get("rate_weight") == weight]
        if first_feasible is None and any((r.get("no_neural_feasible_rate") or 0.0) < 1.0 for r in rows_at):
            first_feasible = weight
        if first_neural_wins is None and any((r.get("neural_win_rate") or 0.0) > 0.0 for r in rows_at):
            first_neural_wins = weight

    predicted_rates: Dict[float, float] = defaultdict(float)
    oracle_rates: Dict[float, float] = defaultdict(float)
    for row in rate_pressure_rows:
        if row.get("policy") != "metadata_plus_system_full_pool":
            continue
        weight = float(row["rate_weight"])
        if row.get("selected_family") == "neural":
            predicted_rates[weight] = max(predicted_rates[weight], float(row.get("selection_rate") or 0.0))
        oracle_rates[weight] = max(oracle_rates[weight], float(row.get("oracle_neural_rate") or 0.0))
    first_pred = _first_weight_positive(predicted_rates)
    first_oracle = _first_weight_positive(oracle_rates)
    return {
        "first_rate_weight_neural_feasible": first_feasible,
        "first_rate_weight_neural_wins": first_neural_wins,
        "first_rate_weight_predicted_neural_wins": first_pred,
        "first_rate_weight_oracle_neural_wins": first_oracle,
        "predicted_shift_vs_oracle": _shift_relation(first_pred, first_oracle),
    }


def _first_weight_positive(values: Dict[float, float]) -> Optional[float]:
    for weight in sorted(values):
        if values[weight] > 0.0:
            return weight
    return None


def _shift_relation(predicted: Optional[float], oracle: Optional[float]) -> str:
    if predicted is None and oracle is None:
        return "no_shift_observed"
    if predicted is None:
        return "late_or_absent"
    if oracle is None:
        return "early_without_oracle_shift"
    if predicted < oracle:
        return "early"
    if predicted > oracle:
        return "late"
    return "aligned"


# ---------------------------------------------------------------------------
# Expected-quality safety gate (shadow / no target leakage)
# ---------------------------------------------------------------------------


def _expected_quality_stats(
    training_rows: List[Dict[str, Any]],
    pair: Optional[Pair],
) -> Dict[str, Optional[float]]:
    if pair is None:
        return {"expected_quality_min": None, "expected_quality_p10": None}
    values = [
        float(r["quality"])
        for r in training_rows
        if r["codec"] == pair[0] and r["config"] == pair[1]
    ]
    if not values:
        return {"expected_quality_min": None, "expected_quality_p10": None}
    return {
        "expected_quality_min": min(values),
        "expected_quality_p10": _quantile(values, 0.10),
    }


def evaluate_expected_quality_gate_shadow(
    *,
    rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    quality_floors: List[Optional[float]],
    protocols: List[str],
    regimes: Dict[str, Dict[str, Any]],
    k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    candidate_index = _candidate_index(rows)
    image_ids = sorted({r["image_id"] for r in rows})
    image_to_dataset: Dict[str, str] = {}
    for row in rows:
        image_to_dataset.setdefault(row["image_id"], str(row.get("dataset")))

    decisions: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for protocol in protocols:
        for regime_name in REGIME_ORDER:
            active_regime = regimes[regime_name]
            for floor in quality_floors:
                oracle_by_image = {
                    image_id: _oracle_for_image(
                        rows,
                        image_id=image_id,
                        pool="full_pool",
                        regime=active_regime,
                        quality_floor=floor,
                    )
                    for image_id in image_ids
                }
                for gate_policy in [
                    "policy_without_expected_quality_gate",
                    "policy_with_expected_quality_gate_shadow",
                ]:
                    policy_decisions: List[Dict[str, Any]] = []
                    for image_id in image_ids:
                        dataset = image_to_dataset[image_id]
                        training_rows = (
                            _loio_training_rows(rows, image_id)
                            if protocol == "loio"
                            else _lodo_training_rows(rows, dataset)
                        )
                        test_meta = metadata_by_image.get(image_id)
                        predicted: Optional[Pair] = None
                        confidence: Optional[float] = None
                        fallback = _policy_robust_global(
                            training_rows=training_rows,
                            pool="full_pool",
                            regime=active_regime,
                            quality_floor=floor,
                        )
                        if test_meta is not None:
                            predicted, confidence, fallback = _policy_knn_metadata(
                                training_rows=training_rows,
                                metadata_by_image=metadata_by_image,
                                pool="full_pool",
                                regime=active_regime,
                                quality_floor=floor,
                                test_meta=test_meta,
                                k=k,
                            )
                        selected = predicted or fallback
                        expected = _expected_quality_stats(training_rows, predicted)
                        fallback_used = predicted is None
                        fallback_reason = "knn_no_training_labels" if fallback_used else ""
                        if gate_policy == "policy_with_expected_quality_gate_shadow":
                            expected_min = expected.get("expected_quality_min")
                            expected_p10 = expected.get("expected_quality_p10")
                            gate_fails = (
                                floor is not None
                                and (
                                    expected_min is None
                                    or expected_min < floor
                                    or (expected_p10 is not None and expected_p10 < floor)
                                )
                            )
                            if gate_fails:
                                selected = fallback
                                fallback_used = True
                                fallback_reason = "expected_quality_gate_training_floor_failure"
                        decision = _realise_decision(
                            pair=selected,
                            image_id=image_id,
                            dataset=dataset,
                            protocol=protocol,
                            regime_name=regime_name,
                            policy=gate_policy,
                            quality_floor=floor,
                            confidence=confidence,
                            fallback_used=fallback_used,
                            fallback_reason=fallback_reason,
                            candidate_index=candidate_index,
                            active_regime=active_regime,
                            oracle=oracle_by_image[image_id],
                        )
                        decision.update(expected)
                        policy_decisions.append(decision)
                        decisions.append(decision)
                    summaries.append(
                        _summarise_decisions(
                            policy_decisions,
                            regime=regime_name,
                            policy=gate_policy,
                            protocol=protocol,
                            quality_floor=floor,
                        )
                    )
    _attach_baseline_quality_margins(decisions)
    _attach_quality_gate_comparison_metrics(summaries)
    return {"decisions": decisions, "summaries": summaries}


def _attach_quality_gate_comparison_metrics(summaries: List[Dict[str, Any]]) -> None:
    ungated_by_key = {
        (s["regime"], s["protocol"], _quality_floor_key(s["quality_floor"])): s
        for s in summaries
        if s["policy"] == "policy_without_expected_quality_gate"
    }
    for summary in summaries:
        summary["quality_violation_reduction_vs_ungated"] = None
        summary["objective_gain_vs_global_baseline"] = None
        if summary["policy"] != "policy_with_expected_quality_gate_shadow":
            continue
        ungated = ungated_by_key.get(
            (summary["regime"], summary["protocol"], _quality_floor_key(summary["quality_floor"]))
        )
        if not ungated:
            continue
        summary["quality_violation_reduction_vs_ungated"] = _relative_reduction(
            summary.get("quality_violation_rate"),
            ungated.get("quality_violation_rate"),
        )
        summary["objective_gain_vs_global_baseline"] = _relative_reduction(
            summary.get("mean_selected_objective"),
            ungated.get("mean_selected_objective"),
        )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_operational_regimes(
    *,
    rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    quality_floors: List[Optional[float]],
    protocols: List[str],
    k: int,
) -> Dict[str, Any]:
    regimes = build_regime_definitions()
    candidate_index = _candidate_index(rows)
    image_ids = sorted({r["image_id"] for r in rows})
    image_to_dataset: Dict[str, str] = {}
    for row in rows:
        image_to_dataset.setdefault(row["image_id"], str(row.get("dataset")))

    all_decisions: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    normal_regime = regimes["normal"]

    for protocol in protocols:
        if protocol not in {"loio", "lodo"}:
            raise ValueError(f"Unknown protocol: {protocol}")
        for regime_name in REGIME_ORDER:
            active_regime = regimes[regime_name]
            for floor in quality_floors:
                oracle_by_image = {
                    image_id: _oracle_for_image(
                        rows,
                        image_id=image_id,
                        pool="full_pool",
                        regime=active_regime,
                        quality_floor=floor,
                    )
                    for image_id in image_ids
                }
                for policy in POLICIES:
                    policy_decisions: List[Dict[str, Any]] = []
                    for image_id in image_ids:
                        dataset = image_to_dataset[image_id]
                        training_rows = (
                            _loio_training_rows(rows, image_id)
                            if protocol == "loio"
                            else _lodo_training_rows(rows, dataset)
                        )
                        pool, selection_regime, enforce_system = _policy_context(
                            policy=policy,
                            regime=active_regime,
                            normal_regime=normal_regime,
                        )
                        pair, confidence, fallback_used, fallback_reason = _select_pair(
                            policy=policy,
                            training_rows=training_rows,
                            metadata_by_image=metadata_by_image,
                            image_id=image_id,
                            pool=pool,
                            selection_regime=selection_regime,
                            active_regime=active_regime,
                            normal_regime=normal_regime,
                            quality_floor=floor,
                            k=k,
                            oracle=oracle_by_image.get(image_id),
                            enforce_active_system=enforce_system,
                        )
                        decision = _realise_decision(
                            pair=pair,
                            image_id=image_id,
                            dataset=dataset,
                            protocol=protocol,
                            regime_name=regime_name,
                            policy=policy,
                            quality_floor=floor,
                            confidence=confidence,
                            fallback_used=fallback_used,
                            fallback_reason=fallback_reason,
                            candidate_index=candidate_index,
                            active_regime=active_regime,
                            oracle=oracle_by_image.get(image_id),
                        )
                        policy_decisions.append(decision)
                        all_decisions.append(decision)
                    summaries.append(
                        _summarise_decisions(
                            policy_decisions,
                            regime=regime_name,
                            policy=policy,
                            protocol=protocol,
                            quality_floor=floor,
                        )
                    )

    _attach_baseline_quality_margins(all_decisions)
    _attach_baseline_reductions(summaries)
    plot_data = _plot_data_rows(summaries)
    winners = _winner_distribution(all_decisions)
    oracle_prediction = _oracle_vs_prediction(all_decisions)
    confusion = _family_confusion(all_decisions)
    return {
        "regime_definitions": regimes,
        "decisions": all_decisions,
        "summaries": summaries,
        "plot_data": plot_data,
        "winner_distribution": winners,
        "oracle_vs_prediction": oracle_prediction,
        "family_confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Rate-pressure sweep
# ---------------------------------------------------------------------------


def _rate_pressure_regime(rate_weight: float) -> Dict[str, Any]:
    remaining = 1.0 - rate_weight
    energy_weight = remaining / 2.0
    quality_weight = remaining / 2.0
    return {
        "weights": {
            "w_R": rate_weight,
            "w_E": energy_weight,
            "w_D": quality_weight,
        },
        "router_profile_source": "rate_pressure_sweep",
        "neural_allowed": True,
        "effective_exclude_neural": False,
        "neural_penalty": 0.0,
        "system_penalty": 0.0,
        "max_energy": None,
        "max_time": None,
        "simulation_only": True,
        "description": (
            "Rate-pressure sweep with w_E=(1-w_R)/2 and w_D=(1-w_R)/2."
        ),
    }


def evaluate_rate_pressure_sweep(
    *,
    rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    quality_floors: List[Optional[float]],
    protocols: List[str],
    k: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    candidate_index = _candidate_index(rows)
    image_ids = sorted({r["image_id"] for r in rows})
    image_to_dataset: Dict[str, str] = {}
    for row in rows:
        image_to_dataset.setdefault(row["image_id"], str(row.get("dataset")))

    baseline_regime = _rate_pressure_regime(RATE_PRESSURE_GRID[0])
    for protocol in protocols:
        for floor in quality_floors:
            baseline_decisions = _rate_pressure_policy_decisions(
                rows=rows,
                metadata_by_image=metadata_by_image,
                candidate_index=candidate_index,
                image_ids=image_ids,
                image_to_dataset=image_to_dataset,
                protocol=protocol,
                quality_floor=floor,
                regime=baseline_regime,
                policy="metadata_plus_system_full_pool",
                k=k,
            )
            baseline_summary = _summarise_decisions(
                baseline_decisions,
                regime="rate_pressure",
                policy="metadata_plus_system_full_pool",
                protocol=protocol,
                quality_floor=floor,
            )
            for rate_weight in RATE_PRESSURE_GRID:
                regime = _rate_pressure_regime(rate_weight)
                for policy in ["metadata_plus_system_full_pool", "full_pool_oracle"]:
                    decisions = _rate_pressure_policy_decisions(
                        rows=rows,
                        metadata_by_image=metadata_by_image,
                        candidate_index=candidate_index,
                        image_ids=image_ids,
                        image_to_dataset=image_to_dataset,
                        protocol=protocol,
                        quality_floor=floor,
                        regime=regime,
                        policy=policy,
                        k=k,
                    )
                    summary = _summarise_decisions(
                        decisions,
                        regime="rate_pressure",
                        policy=policy,
                        protocol=protocol,
                        quality_floor=floor,
                    )
                    family_counts = Counter(
                        d.get("selected_family") or "none" for d in decisions
                    )
                    for family in ("classical", "neural"):
                        family_counts.setdefault(family, 0)
                    codec_counts = Counter(
                        f"{d.get('selected_codec') or 'none'}|{d.get('selected_config') or 'none'}"
                        for d in decisions
                    )
                    total = len(decisions)
                    top_family, top_family_count = sorted(
                        family_counts.items(), key=lambda item: (-item[1], item[0])
                    )[0]
                    top_codec, top_codec_count = sorted(
                        codec_counts.items(), key=lambda item: (-item[1], item[0])
                    )[0]
                    for selected_family, count in sorted(family_counts.items()):
                        output.append(
                            {
                                "rate_weight": rate_weight,
                                "energy_weight": regime["weights"]["w_E"],
                                "quality_weight": regime["weights"]["w_D"],
                                "policy": policy,
                                "protocol": protocol,
                                "quality_floor": floor,
                                "mean_rate": summary["mean_rate"],
                                "mean_energy": summary["mean_energy"],
                                "mean_quality": summary["mean_quality"],
                                "mean_regret": summary["mean_regret"],
                                "rate_reduction_vs_baseline": _relative_reduction(
                                    summary["mean_rate"], baseline_summary["mean_rate"]
                                ),
                                "energy_delta_vs_baseline": (
                                    _safe_div(summary["mean_energy"], baseline_summary["mean_energy"]) - 1.0
                                    if _safe_div(summary["mean_energy"], baseline_summary["mean_energy"]) is not None
                                    else None
                                ),
                                "regret_reduction_vs_baseline": _relative_reduction(
                                    summary["mean_regret"], baseline_summary["mean_regret"]
                                ),
                                "neural_selection_rate": summary["neural_selection_rate"],
                                "oracle_neural_rate": summary["oracle_neural_rate"],
                                "top_selected_codec": top_codec,
                                "top_selected_family": top_family,
                                "top_selected_codec_rate": top_codec_count / total if total else None,
                                "top_selected_family_rate": top_family_count / total if total else None,
                                "selected_family": selected_family,
                                "selection_rate": count / total if total else None,
                            }
                        )
    return output


def _rate_pressure_policy_decisions(
    *,
    rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    candidate_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    image_ids: List[str],
    image_to_dataset: Dict[str, str],
    protocol: str,
    quality_floor: Optional[float],
    regime: Dict[str, Any],
    policy: str,
    k: int,
) -> List[Dict[str, Any]]:
    oracle_by_image = {
        image_id: _oracle_for_image(
            rows,
            image_id=image_id,
            pool="full_pool",
            regime=regime,
            quality_floor=quality_floor,
        )
        for image_id in image_ids
    }
    decisions: List[Dict[str, Any]] = []
    for image_id in image_ids:
        dataset = image_to_dataset[image_id]
        training_rows = (
            _loio_training_rows(rows, image_id)
            if protocol == "loio"
            else _lodo_training_rows(rows, dataset)
        )
        if policy == "full_pool_oracle":
            pair = oracle_by_image[image_id]["pair"] if oracle_by_image[image_id] else None
            confidence = 1.0 if pair else None
            fallback_used = False
            fallback_reason = "oracle_reference_no_fallback"
        else:
            pair, confidence, fallback = _policy_knn_metadata(
                training_rows=training_rows,
                metadata_by_image=metadata_by_image,
                pool="full_pool",
                regime=regime,
                quality_floor=quality_floor,
                test_meta=metadata_by_image[image_id],
                k=k,
            )
            fallback_used = pair is None
            fallback_reason = "knn_no_training_labels" if fallback_used else ""
            if pair is None:
                pair = fallback
        decisions.append(
            _realise_decision(
                pair=pair,
                image_id=image_id,
                dataset=dataset,
                protocol=protocol,
                regime_name="rate_pressure",
                policy=policy,
                quality_floor=quality_floor,
                confidence=confidence,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                candidate_index=candidate_index,
                active_regime=regime,
                oracle=oracle_by_image[image_id],
            )
        )
    return decisions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _maybe_generate_plots(
    *,
    out_dir: Path,
    plot_data: List[Dict[str, Any]],
    winner_distribution: List[Dict[str, Any]],
    oracle_vs_prediction: List[Dict[str, Any]],
    family_confusion: List[Dict[str, Any]],
    rate_pressure: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {
            "generated": False,
            "plot_paths": [],
            "skipped_reason": "matplotlib_unavailable",
        }

    plot_paths: List[str] = []

    def save(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths.append(str(path))

    try:
        _plot_scatter_energy_regret(plt, plot_data)
        save("energy_saving_vs_regret_reduction.png")

        _plot_bar_metric(plt, plot_data, "neural_selection_rate", "neural_selection_rate_by_regime.png")
        save("neural_selection_rate_by_regime.png")

        _plot_stacked_winners(plt, winner_distribution, "selected_family")
        save("winner_family_by_regime.png")

        _plot_stacked_winners(plt, winner_distribution, "selected_codec")
        save("winner_codec_by_regime.png")

        _plot_oracle_vs_predicted(plt, oracle_vs_prediction)
        save("oracle_vs_predicted_neural_rate.png")

        _plot_family_confusion(plt, family_confusion)
        save("family_confusion_heatmap.png")

        _plot_rate_pressure_family(plt, rate_pressure)
        save("rate_pressure_family_shift.png")

        _plot_rate_pressure_codec(plt, rate_pressure)
        save("rate_pressure_codec_shift.png")

        _plot_rate_energy_sweep(plt, rate_pressure)
        save("rate_reduction_vs_energy_penalty_sweep.png")

        _plot_bar_metric(plt, plot_data, "quality_violation_rate", "quality_violation_by_regime.png")
        save("quality_violation_by_regime.png")
    except Exception as exc:
        return {
            "generated": False,
            "plot_paths": plot_paths,
            "skipped_reason": f"plot_generation_failed:{type(exc).__name__}",
        }

    return {"generated": True, "plot_paths": plot_paths, "skipped_reason": None}


def _maybe_generate_switch_plots(
    *,
    out_dir: Path,
    reason_by_rate: List[Dict[str, Any]],
    tradeoff_scatter: List[Dict[str, Any]],
    quality_floor_summary: List[Dict[str, Any]],
    switch_rate_summary: List[Dict[str, Any]],
    rate_pressure: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {
            "generated": False,
            "plot_paths": [],
            "skipped_reason": "matplotlib_unavailable",
        }

    plot_paths: List[str] = []

    def save(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        plot_paths.append(str(path))

    try:
        _plot_switch_reason_by_rate(plt, reason_by_rate)
        save("switch_reason_by_rate_weight.png")
        _plot_neural_classic_tradeoff(plt, tradeoff_scatter)
        save("neural_vs_classic_tradeoff_scatter.png")
        _plot_quality_floor_neural_necessity(plt, quality_floor_summary)
        save("quality_floor_vs_neural_necessity.png")
        _plot_rate_weight_switch_boundary(plt, switch_rate_summary, rate_pressure)
        save("rate_weight_switch_boundary.png")
    except Exception as exc:
        return {
            "generated": False,
            "plot_paths": plot_paths,
            "skipped_reason": f"switch_plot_generation_failed:{type(exc).__name__}",
        }

    return {"generated": True, "plot_paths": plot_paths, "skipped_reason": None}


def _plot_switch_reason_by_rate(plt: Any, rows: List[Dict[str, Any]]) -> None:
    filtered = [r for r in rows if r["protocol"] == "loio"] or rows
    weights = sorted({float(r["rate_weight"]) for r in filtered})
    bottoms = [0.0 for _ in weights]
    plt.figure(figsize=(8, 5))
    for reason in SWITCH_REASONS:
        values = [
            sum(
                float(r.get("share") or 0.0)
                for r in filtered
                if float(r["rate_weight"]) == weight and r["switch_reason"] == reason
            )
            for weight in weights
        ]
        plt.bar(weights, values, bottom=bottoms, width=0.06, label=reason)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    plt.xlabel("Rate weight")
    plt.ylabel("Share")
    plt.legend(fontsize=6)


def _plot_neural_classic_tradeoff(plt: Any, rows: List[Dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 5))
    for row in rows:
        x = row.get("bitrate_reduction_neural_vs_classic")
        y = row.get("delta_energy_neural_minus_classic")
        if x is None or y is None:
            continue
        size = 30 + 60 * abs(float(row.get("delta_quality_neural_minus_classic") or 0.0))
        plt.scatter(float(x), float(y), s=size, alpha=0.65, label=row.get("switch_reason"))
    plt.xlabel("Bitrate reduction neural vs classic")
    plt.ylabel("Delta energy neural minus classic")


def _plot_quality_floor_neural_necessity(plt: Any, rows: List[Dict[str, Any]]) -> None:
    filtered = [r for r in rows if r["protocol"] == "loio"] or rows
    floors = sorted({float(r["quality_floor"]) for r in filtered if r.get("quality_floor") is not None})
    necessity = []
    wins = []
    for floor in floors:
        matches = [r for r in filtered if float(r["quality_floor"]) == floor]
        necessity.append(mean(float(r.get("neural_necessary_rate") or 0.0) for r in matches))
        wins.append(mean(float(r.get("neural_win_rate") or 0.0) for r in matches))
    plt.figure(figsize=(7, 5))
    plt.plot(floors, necessity, marker="o", label="neural_necessary_rate")
    plt.plot(floors, wins, marker="s", label="neural_win_rate")
    plt.xlabel("Quality floor")
    plt.ylabel("Rate")
    plt.legend()


def _plot_rate_weight_switch_boundary(
    plt: Any,
    switch_rows: List[Dict[str, Any]],
    rate_pressure: List[Dict[str, Any]],
) -> None:
    weights = sorted({float(r["rate_weight"]) for r in switch_rows if r.get("rate_weight") is not None})
    neural_win = []
    predicted = []
    oracle = []
    for weight in weights:
        sw = [r for r in switch_rows if float(r["rate_weight"]) == weight]
        neural_win.append(mean(float(r.get("neural_win_rate") or 0.0) for r in sw) if sw else 0.0)
        rp = [
            r for r in rate_pressure
            if float(r["rate_weight"]) == weight
            and r["policy"] == "metadata_plus_system_full_pool"
        ]
        predicted.append(max([float(r.get("neural_selection_rate") or 0.0) for r in rp] or [0.0]))
        oracle.append(max([float(r.get("oracle_neural_rate") or 0.0) for r in rp] or [0.0]))
    plt.figure(figsize=(8, 5))
    plt.plot(weights, neural_win, marker="o", label="switch_neural_win_rate")
    plt.plot(weights, predicted, marker="s", label="predicted_neural_rate")
    plt.plot(weights, oracle, marker="^", label="oracle_neural_rate")
    plt.xlabel("Rate weight")
    plt.ylabel("Rate")
    plt.legend()


def _main_plot_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if r["protocol"] == "loio"
        and str(r["quality_floor"]) in {"30.0", "30", "none"}
        and r["policy"] in {
            "metadata_only_full_pool",
            "system_only_full_pool",
            "metadata_plus_system_full_pool",
        }
    ]


def _plot_scatter_energy_regret(plt: Any, plot_data: List[Dict[str, Any]]) -> None:
    rows = _main_plot_rows(plot_data) or plot_data
    plt.figure(figsize=(8, 5))
    for row in rows:
        x = row.get("energy_saving_vs_global_baseline")
        y = row.get("regret_reduction_vs_global_baseline")
        if x is None or y is None:
            continue
        size = 40 + 200 * float(row.get("neural_selection_rate") or 0.0)
        plt.scatter(x, y, s=size, label=f"{row['regime']}:{row['policy']}", alpha=0.7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Energy saving vs global baseline")
    plt.ylabel("Regret reduction vs global baseline")
    plt.legend(fontsize=6, loc="best")


def _plot_bar_metric(plt: Any, plot_data: List[Dict[str, Any]], metric: str, _name: str) -> None:
    rows = _main_plot_rows(plot_data) or plot_data
    regimes = REGIME_ORDER
    policies = sorted({r["policy"] for r in rows})
    width = 0.8 / max(1, len(policies))
    plt.figure(figsize=(9, 5))
    for p_idx, policy in enumerate(policies):
        values = []
        xs = []
        for r_idx, regime in enumerate(regimes):
            matches = [r for r in rows if r["policy"] == policy and r["regime"] == regime]
            value = mean([float(m.get(metric) or 0.0) for m in matches]) if matches else 0.0
            xs.append(r_idx + p_idx * width)
            values.append(value)
        plt.bar(xs, values, width=width, label=policy)
    plt.xticks([i + width for i in range(len(regimes))], regimes, rotation=35, ha="right")
    plt.ylabel(metric)
    plt.legend(fontsize=7)


def _plot_stacked_winners(plt: Any, rows: List[Dict[str, Any]], field: str) -> None:
    filtered = [
        r for r in rows
        if r["protocol"] == "loio"
        and r["policy"] == "metadata_plus_system_full_pool"
        and str(r["quality_floor"]) in {"30.0", "30", "none"}
    ] or rows
    regimes = REGIME_ORDER
    labels = sorted({str(r[field]) for r in filtered})
    bottoms = [0.0 for _ in regimes]
    plt.figure(figsize=(9, 5))
    for label in labels:
        values = []
        for regime in regimes:
            value = sum(
                float(r.get("selection_rate") or 0.0)
                for r in filtered
                if r["regime"] == regime and str(r[field]) == label
            )
            values.append(value)
        plt.bar(regimes, values, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Selection rate")
    plt.legend(fontsize=7)


def _plot_oracle_vs_predicted(plt: Any, rows: List[Dict[str, Any]]) -> None:
    filtered = [
        r for r in rows
        if r["policy"] == "metadata_plus_system_full_pool"
        and r["protocol"] == "loio"
        and r["family"] == "neural"
        and str(r["quality_floor"]) in {"30.0", "30", "none"}
    ] or [r for r in rows if r["family"] == "neural"]
    regimes = REGIME_ORDER
    oracle = []
    pred = []
    for regime in regimes:
        matches = [r for r in filtered if r["regime"] == regime]
        oracle.append(mean([float(m.get("oracle_rate") or 0.0) for m in matches]) if matches else 0.0)
        pred.append(mean([float(m.get("predicted_rate") or 0.0) for m in matches]) if matches else 0.0)
    plt.figure(figsize=(9, 5))
    xs = list(range(len(regimes)))
    plt.bar([x - 0.2 for x in xs], oracle, width=0.4, label="oracle")
    plt.bar([x + 0.2 for x in xs], pred, width=0.4, label="predicted")
    plt.xticks(xs, regimes, rotation=35, ha="right")
    plt.ylabel("Neural rate")
    plt.legend()


def _plot_family_confusion(plt: Any, rows: List[Dict[str, Any]]) -> None:
    target = [
        r for r in rows
        if r["regime"] == "bandwidth_limited"
        and r["policy"] == "metadata_plus_system_full_pool"
        and r["protocol"] == "loio"
    ] or rows
    families = sorted(
        {
            *(r["oracle_family"] for r in target),
            *(r["predicted_family"] for r in target),
        }
    )
    matrix = []
    for oracle_family in families:
        matrix.append(
            [
                sum(
                    float(r.get("rate") or 0.0)
                    for r in target
                    if r["oracle_family"] == oracle_family and r["predicted_family"] == pred_family
                )
                for pred_family in families
            ]
        )
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(families)), families)
    plt.yticks(range(len(families)), families)
    plt.xlabel("Predicted family")
    plt.ylabel("Oracle family")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")


def _plot_rate_pressure_family(plt: Any, rows: List[Dict[str, Any]]) -> None:
    filtered = [
        r for r in rows
        if r["policy"] == "metadata_plus_system_full_pool"
        and r["protocol"] == "loio"
    ] or rows
    families = sorted({r["selected_family"] for r in filtered})
    weights = sorted({float(r["rate_weight"]) for r in filtered})
    plt.figure(figsize=(8, 5))
    for family in families:
        values = []
        for w in weights:
            matches = [r for r in filtered if float(r["rate_weight"]) == w and r["selected_family"] == family]
            values.append(mean([float(m.get("selection_rate") or 0.0) for m in matches]) if matches else 0.0)
        plt.plot(weights, values, marker="o", label=family)
    plt.xlabel("Rate weight")
    plt.ylabel("Selection share")
    plt.legend()


def _plot_rate_pressure_codec(plt: Any, rows: List[Dict[str, Any]]) -> None:
    filtered = [
        r for r in rows
        if r["policy"] == "metadata_plus_system_full_pool"
        and r["protocol"] == "loio"
    ] or rows
    weights = sorted({float(r["rate_weight"]) for r in filtered})
    codecs = sorted({r["top_selected_codec"] for r in filtered})
    plt.figure(figsize=(9, 5))
    for codec in codecs:
        values = []
        for w in weights:
            matches = [r for r in filtered if float(r["rate_weight"]) == w and r["top_selected_codec"] == codec]
            values.append(mean([float(m.get("top_selected_codec_rate") or 0.0) for m in matches]) if matches else 0.0)
        plt.plot(weights, values, marker="o", label=codec)
    plt.xlabel("Rate weight")
    plt.ylabel("Top codec selection share")
    plt.legend(fontsize=7)


def _plot_rate_energy_sweep(plt: Any, rows: List[Dict[str, Any]]) -> None:
    filtered = [
        r for r in rows
        if r["policy"] == "metadata_plus_system_full_pool"
        and r["protocol"] == "loio"
    ] or rows
    plt.figure(figsize=(7, 5))
    for row in filtered:
        x = row.get("rate_reduction_vs_baseline")
        y = row.get("energy_delta_vs_baseline")
        if x is None or y is None:
            continue
        plt.scatter(x, y, c=float(row["rate_weight"]), cmap="viridis", vmin=0.1, vmax=0.8)
    plt.xlabel("Rate reduction vs baseline")
    plt.ylabel("Energy delta vs baseline")
    plt.colorbar(label="Rate weight")


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def build_interpretation(
    *,
    summaries: List[Dict[str, Any]],
    rate_pressure: List[Dict[str, Any]],
) -> List[str]:
    notes: List[str] = []

    def main_summary(regime: str, policy: str = "metadata_plus_system_full_pool") -> Optional[Dict[str, Any]]:
        matches = [
            s for s in summaries
            if s["regime"] == regime
            and s["policy"] == policy
            and s["protocol"] == "loio"
        ]
        if not matches:
            return None
        return matches[0]

    energy = main_summary("energy_saving")
    if (
        energy
        and (energy.get("energy_saving_vs_global_baseline") or 0.0) > 0.0
        and (energy.get("quality_violation_rate") or 0.0) == 0.0
    ):
        notes.append(
            "The energy-saving regime reduces mean energy while preserving the quality floor in this benchmark."
        )

    bandwidth = main_summary("bandwidth_limited")
    normal = main_summary("normal")
    if bandwidth and normal:
        b_rate = bandwidth.get("neural_selection_rate") or 0.0
        n_rate = normal.get("neural_selection_rate") or 0.0
        if b_rate > n_rate:
            notes.append(
                "The bandwidth-limited regime activates neural selections because rate has higher weight."
            )
        if (bandwidth.get("oracle_neural_rate") or 0.0) > (normal.get("oracle_neural_rate") or 0.0):
            notes.append(
                "The oracle neural rate increases under bandwidth pressure, indicating that neural codecs occupy the rate-constrained region of the R-D-E space."
            )

    no_cuda = main_summary("no_cuda")
    if no_cuda and (no_cuda.get("neural_selection_rate") or 0.0) == 0.0:
        notes.append(
            "The no-CUDA regime suppresses neural selections through capability-aware filtering."
        )

    battery = main_summary("battery_pressure")
    if battery and normal:
        if (battery.get("neural_selection_rate") or 0.0) < (normal.get("neural_selection_rate") or 0.0):
            notes.append("Energy-saving and no-CUDA regimes suppress neural selections.")

    content_system = [
        s for s in summaries
        if s["policy"] == "metadata_plus_system_full_pool"
    ]
    content_only = [
        s for s in summaries
        if s["policy"] == "metadata_only_full_pool"
    ]
    better = False
    content_by_key = {
        (s["regime"], s["protocol"], _quality_floor_key(s["quality_floor"])): s
        for s in content_only
    }
    for row in content_system:
        other = content_by_key.get(
            (row["regime"], row["protocol"], _quality_floor_key(row["quality_floor"]))
        )
        if other and row.get("mean_regret") is not None and other.get("mean_regret") is not None:
            if row["mean_regret"] < other["mean_regret"]:
                better = True
                break
    if better:
        notes.append("Combining content and system context can lower the active R-D-E objective.")
    else:
        notes.append(
            "In this benchmark, the predictive gain is mainly content/profile-driven; system-aware context primarily modulates feasibility and energy exposure."
        )

    main_rows = [
        r for r in rate_pressure
        if r["policy"] == "metadata_plus_system_full_pool"
        and r["protocol"] == "loio"
        and r["selected_family"] == "neural"
    ]
    if main_rows:
        by_weight: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for row in main_rows:
            by_weight[float(row["rate_weight"])].append(row)
        weights = sorted(by_weight)
        first = mean(float(r.get("selection_rate") or 0.0) for r in by_weight[weights[0]])
        last = mean(float(r.get("selection_rate") or 0.0) for r in by_weight[weights[-1]])
        notes.append(
            "As rate pressure increases, the predicted neural selection rate increases from "
            f"{first:.3f} to {last:.3f}."
        )

    quality_rates = [
        s.get("quality_violation_rate")
        for s in summaries
        if s["policy"] == "metadata_plus_system_full_pool"
        and s.get("quality_violation_rate") is not None
    ]
    if quality_rates:
        notes.append(
            "Quality violation remains at "
            f"{max(float(x) for x in quality_rates):.3f}, indicating that the observed gains are not obtained by bypassing the quality floor."
        )

    gaps = [
        abs((s.get("oracle_neural_rate") or 0.0) - (s.get("neural_selection_rate") or 0.0))
        for s in summaries
        if s["policy"] == "metadata_plus_system_full_pool"
    ]
    if gaps and max(gaps) > 0.0:
        notes.append(
            "A gap between oracle neural rate and predicted neural rate indicates conservative or imperfect neural opportunity detection."
        )

    return _sanitize_interpretation(notes)


def _quality_metric_contract(quality_col: str) -> Dict[str, Any]:
    metric = str(quality_col).strip().lower()
    if metric == "ssimulacra2":
        return {
            "quality_col": quality_col,
            "role": "preferred_perceptual_image_metric",
            "interpretation": (
                "This run is aligned with the image router's perceptual quality metric."
            ),
        }
    if metric == "psnr":
        return {
            "quality_col": quality_col,
            "role": "rate_oriented_stress_test",
            "interpretation": (
                "PSNR30 is a useful rate-pressure stress test but does not guarantee perceptual quality."
            ),
        }
    return {
        "quality_col": quality_col,
        "role": "custom_quality_metric",
        "interpretation": "Interpret quality floors according to the supplied metric contract.",
    }


def _sanitize_interpretation(notes: List[str]) -> List[str]:
    banned = [
        "prov" + "es",
        "best " + "possible",
        "neural codecs are " + "better",
        "classical codecs are " + "obsolete",
        "statistically " + "significant",
    ]
    clean: List[str] = []
    for note in notes:
        lowered = note.lower()
        if any(term in lowered for term in banned):
            continue
        clean.append(note)
    return clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline operational-regime simulation for predictive R-D-E routing."
        )
    )
    parser.add_argument("--rde-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-summary-csv")
    parser.add_argument("--out-decisions-csv")
    parser.add_argument("--image-id-col", default="dataset,image")
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="param")
    parser.add_argument("--rate-col", default="bpp")
    parser.add_argument("--quality-col", default="psnr")
    parser.add_argument("--energy-col", default="energy_per_image_j")
    parser.add_argument("--quality-floors", default="30,60,70,80")
    parser.add_argument("--protocols", default="loio,lodo")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pool_rows(
        args.rde_csv,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        image_id_col=args.image_id_col,
        dataset_col=args.dataset_col,
    )
    normalization = normalize_full_pool(rows)
    metadata_by_image = _derive_metadata_from_rde_rows(rows)
    if not metadata_by_image:
        raise ValueError(
            "Metadata-only policies require width/height columns in --rde-csv."
        )

    quality_floors = _parse_quality_floors(args.quality_floors)
    protocols = _parse_csv_list(args.protocols)

    result = evaluate_operational_regimes(
        rows=rows,
        metadata_by_image=metadata_by_image,
        quality_floors=quality_floors,
        protocols=protocols,
        k=args.k,
    )
    rate_pressure = evaluate_rate_pressure_sweep(
        rows=rows,
        metadata_by_image=metadata_by_image,
        quality_floors=quality_floors,
        protocols=protocols,
        k=args.k,
    )
    quality_gate = evaluate_expected_quality_gate_shadow(
        rows=rows,
        metadata_by_image=metadata_by_image,
        quality_floors=quality_floors,
        protocols=protocols,
        regimes=result["regime_definitions"],
        k=args.k,
    )
    switch_analysis = build_switch_analysis(
        rows=rows,
        quality_floors=quality_floors,
        protocols=protocols,
        regimes=result["regime_definitions"],
    )
    switch_rate_by_image: List[Dict[str, Any]] = []
    switch_rate_summary: List[Dict[str, Any]] = []
    switch_reason_by_rate: List[Dict[str, Any]] = []
    for rate_weight in RATE_PRESSURE_GRID:
        rate_switch = build_switch_analysis(
            rows=rows,
            quality_floors=quality_floors,
            protocols=protocols,
            regimes=result["regime_definitions"],
            rate_weight=rate_weight,
            rate_regime=_rate_pressure_regime(rate_weight),
        )
        switch_rate_by_image.extend(rate_switch["by_image"])
        switch_rate_summary.extend(rate_switch["summary"])
        switch_reason_by_rate.extend(rate_switch["reason_by_rate_weight"])
    switch_transition = _rate_pressure_switch_transition(
        switch_summary=switch_rate_summary,
        rate_pressure_rows=rate_pressure,
    )

    summary_path = Path(args.out_summary_csv) if args.out_summary_csv else out_dir / "operational_regime_summary.csv"
    decisions_path = Path(args.out_decisions_csv) if args.out_decisions_csv else out_dir / "operational_regime_decisions.csv"
    plot_data_path = out_dir / "operational_regime_plot_data.csv"
    winners_path = out_dir / "operational_regime_winner_distribution.csv"
    oracle_pred_path = out_dir / "operational_regime_oracle_vs_prediction.csv"
    confusion_path = out_dir / "operational_regime_family_confusion.csv"
    sweep_path = out_dir / "operational_regime_rate_pressure_sweep.csv"
    switch_summary_path = out_dir / "operational_regime_switch_summary.csv"
    switch_by_image_path = out_dir / "operational_regime_switch_by_image.csv"
    switch_report_path = out_dir / "operational_regime_switch_report.json"
    switch_reason_path = out_dir / "switch_reason_by_rate_weight.csv"
    switch_scatter_path = out_dir / "neural_vs_classic_tradeoff_scatter.csv"
    switch_floor_path = out_dir / "quality_floor_switch_summary.csv"
    gate_summary_path = out_dir / "quality_gate_comparison_summary.csv"
    gate_decisions_path = out_dir / "quality_gate_comparison_decisions.csv"
    report_path = Path(args.out_json) if args.out_json else out_dir / "operational_regime_report.json"

    _write_csv(summary_path, result["summaries"])
    _write_csv(decisions_path, result["decisions"])
    _write_csv(plot_data_path, result["plot_data"])
    _write_csv(winners_path, result["winner_distribution"])
    _write_csv(oracle_pred_path, result["oracle_vs_prediction"])
    _write_csv(confusion_path, result["family_confusion"])
    _write_csv(sweep_path, rate_pressure)
    _write_csv(switch_summary_path, switch_analysis["summary"])
    _write_csv(switch_by_image_path, switch_analysis["by_image"])
    _write_csv(switch_reason_path, switch_reason_by_rate)
    _write_csv(switch_scatter_path, switch_analysis["tradeoff_scatter"] + _switch_tradeoff_scatter(switch_rate_by_image))
    _write_csv(switch_floor_path, switch_analysis["quality_floor_summary"])
    _write_csv(gate_summary_path, quality_gate["summaries"])
    _write_csv(gate_decisions_path, quality_gate["decisions"])
    _write_json(
        switch_report_path,
        {
            "schema_version": "operational_regime_switch_analysis_v1",
            "quality_metric": args.quality_col,
            "metric_contract": _quality_metric_contract(args.quality_col),
            "rate_pressure_transition": switch_transition,
            "switch_reasons": SWITCH_REASONS,
        },
    )

    plot_info = _maybe_generate_plots(
        out_dir=out_dir,
        plot_data=result["plot_data"],
        winner_distribution=result["winner_distribution"],
        oracle_vs_prediction=result["oracle_vs_prediction"],
        family_confusion=result["family_confusion"],
        rate_pressure=rate_pressure,
    )
    plot_info["plot_data_paths"] = [
        str(plot_data_path),
        str(winners_path),
        str(oracle_pred_path),
        str(confusion_path),
        str(sweep_path),
        str(switch_reason_path),
        str(switch_scatter_path),
        str(switch_floor_path),
        str(gate_summary_path),
        str(gate_decisions_path),
    ]
    switch_plot_info = _maybe_generate_switch_plots(
        out_dir=out_dir,
        reason_by_rate=switch_reason_by_rate,
        tradeoff_scatter=switch_analysis["tradeoff_scatter"] + _switch_tradeoff_scatter(switch_rate_by_image),
        quality_floor_summary=switch_analysis["quality_floor_summary"],
        switch_rate_summary=switch_rate_summary,
        rate_pressure=rate_pressure,
    )
    if switch_plot_info["plot_paths"]:
        plot_info["plot_paths"].extend(switch_plot_info["plot_paths"])
        plot_info["generated"] = plot_info["generated"] or switch_plot_info["generated"]

    report = {
        "schema_version": "operational_regime_simulation_v1",
        "created_at_unix": started,
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "rde_csv": args.rde_csv,
            "image_id_col": args.image_id_col,
            "dataset_col": args.dataset_col,
            "codec_col": args.codec_col,
            "config_col": args.config_col,
            "rate_col": args.rate_col,
            "quality_col": args.quality_col,
            "energy_col": args.energy_col,
            "quality_floors": quality_floors,
            "protocols": protocols,
            "k": args.k,
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
        },
        "provenance": {
            "offline_read_only": True,
            "policy_does_not_see_test_image_rde": True,
            "test_image_rde_used_only_for_realisation_and_oracle": True,
            "runtime_router_changed": False,
            "j_rde_formula_changed": False,
            "rde_router_cli_changed": False,
            "codecs_executed": False,
            "benchmark_raw_data_modified": False,
        },
        "normalization": normalization,
        "regime_definitions": result["regime_definitions"],
        "oracle_quality_contract": build_oracle_quality_contract(result["decisions"]),
        "rate_pressure_sweep": {
            "rate_weights": RATE_PRESSURE_GRID,
            "weight_rule": "w_R=grid_value; w_E=(1-w_R)/2; w_D=(1-w_R)/2",
        },
        "quality_metric_contract": _quality_metric_contract(args.quality_col),
        "switch_analysis": {
            "summary_csv": str(switch_summary_path),
            "by_image_csv": str(switch_by_image_path),
            "report_json": str(switch_report_path),
            "rate_pressure_transition": switch_transition,
        },
        "expected_quality_gate": {
            "mode": "shadow_report_only",
            "uses_target_quality_for_decision": False,
            "summary_csv": str(gate_summary_path),
            "decisions_csv": str(gate_decisions_path),
        },
        "policies": POLICIES,
        "summaries": result["summaries"],
        "interpretation": build_interpretation(
            summaries=result["summaries"],
            rate_pressure=rate_pressure,
        ),
        "plot_artifacts": plot_info,
        "outputs": {
            "summary_csv": str(summary_path),
            "decisions_csv": str(decisions_path),
            "plot_data_csv": str(plot_data_path),
            "winner_distribution_csv": str(winners_path),
            "oracle_vs_prediction_csv": str(oracle_pred_path),
            "family_confusion_csv": str(confusion_path),
            "rate_pressure_sweep_csv": str(sweep_path),
            "switch_summary_csv": str(switch_summary_path),
            "switch_by_image_csv": str(switch_by_image_path),
            "switch_report_json": str(switch_report_path),
            "switch_reason_by_rate_weight_csv": str(switch_reason_path),
            "neural_vs_classic_tradeoff_scatter_csv": str(switch_scatter_path),
            "quality_floor_switch_summary_csv": str(switch_floor_path),
            "quality_gate_comparison_summary_csv": str(gate_summary_path),
            "quality_gate_comparison_decisions_csv": str(gate_decisions_path),
            "report_json": str(report_path),
        },
    }
    _write_json(report_path, report)


if __name__ == "__main__":
    main()
