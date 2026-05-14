"""Pipeline helpers extracted from rde_router.py.

This module is the first step of breaking up the monolithic CLI entry point.
It hosts side-effect-free orchestration helpers that translate between the
parsed argparse namespace, the router report, and CSV-export rows.

Anything that performs argparse parsing, writes JSON/CSV files, prints to
stdout, executes the backend command, or writes feedback receipts still
lives in rde_router.py.
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple

from src.router.adaptation.context_policy import compute_context_policy
from src.router.core.profiles import get_profile
from src.router.core.rde_database import RDEPoint


def _normalize_weights(w_e: float, w_r: float, w_d: float) -> Dict[str, float]:
    total = w_e + w_r + w_d

    if total <= 0:
        raise ValueError("The sum of the weights must be positive.")

    return {
        "w_E": w_e / total,
        "w_R": w_r / total,
        "w_D": w_d / total,
    }


def build_weights_for_profile(
    args: argparse.Namespace,
    profile_name: str,
) -> Tuple[Dict[str, float], Optional[float], str, Optional[Dict[str, Any]]]:
    if args.auto_weights:
        policy = compute_context_policy(
            power_mode=args.power_mode,
            battery_percent=args.battery_percent,
            thermal_state=args.thermal_state,
            network_profile=args.network_profile,
            quality_target=args.quality_target,
            system_load=args.system_load,
        )

        weights = policy["weights"]

        min_quality = args.quality_floor
        if args.min_quality is not None:
            min_quality = max(args.min_quality, min_quality)

        return weights, min_quality, "context_policy", policy

    profile = get_profile(profile_name)

    w_e = args.wE if args.wE is not None else profile.w_e
    w_r = args.wR if args.wR is not None else profile.w_r
    w_d = args.wD if args.wD is not None else profile.w_d

    weights = _normalize_weights(w_e, w_r, w_d)

    min_quality = args.quality_floor
    if args.min_quality is not None:
        min_quality = max(args.min_quality, min_quality)

    return weights, min_quality, "manual_profile", None


def annotate_points_with_calibration_provenance(
    points: List[RDEPoint],
    calibration_report: Dict[str, Any],
) -> None:
    applied_by_key = {
        (str(item.get("codec")), str(item.get("config"))): item
        for item in calibration_report.get("applied", [])
        if item.get("codec") is not None and item.get("config") is not None
    }

    for point in points:
        calibration = applied_by_key.get((str(point.codec), str(point.config)))
        if calibration is None:
            continue

        raw = dict(getattr(point, "raw", {}) or {})
        for key in (
            "energy_is_measured",
            "energy_usable_for_total",
            "energy_scope",
            "energy_backend",
            "energy_method",
            "energy_quality",
            "energy_scaling_method",
            "current_method",
        ):
            if key in calibration:
                raw[key] = calibration.get(key)

        point.raw = raw


def summary_row_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    selected = report["decision"]["selected"]
    weights = report["weights"]
    constraints = report["constraints"]
    filtering = report["codec_filtering"]
    normalization = report["normalization"]
    context_policy = report["context_policy"]

    return {
        "profile": report["profile"],
        "weight_source": report["weight_source"],
        "decision_mode": report["decision"]["decision_mode"],
        "selected_codec": selected["codec"],
        "selected_config": selected["config"],
        "rate": selected["rate"],
        "quality_mean": selected["quality"],
        "quality_constraint_stat": selected["quality_constraint_stat"],
        "quality_constraint_value": selected["quality_constraint_value"],
        "quality_min": selected["quality_stats"]["min"],
        "quality_p10": selected["quality_stats"]["p10"],
        "quality_p25": selected["quality_stats"]["p25"],
        "energy": selected["energy"],
        "time_ms": selected["time_ms"],
        "J_RDE": selected["cost"],
        "num_rows_loaded": report["num_rows_loaded_before_aggregation"],
        "num_points_before_codec_filtering": filtering["num_before_codec_filtering"],
        "num_points_after_codec_filtering": filtering["num_after_codec_filtering"],
        "num_candidate_points": report["decision"]["num_points_total"],
        "num_admissible_points": report["decision"]["num_points_admissible"],
        "num_points_safe": report["decision"]["num_points_safe"],
        "num_points_near": report["decision"]["num_points_near"],
        "normalization_scope": normalization["scope"],
        "num_normalization_reference_points": normalization["num_reference_points"],
        "min_quality": constraints["min_quality"],
        "quality_floor": constraints["quality_floor"],
        "near_quality_floor": constraints["near_quality_floor"],
        "allow_degraded_fallback": constraints["allow_degraded_fallback"],
        "max_rate": constraints["max_rate"],
        "max_energy": constraints["max_energy"],
        "max_time_ms": constraints["max_time_ms"],
        "w_E": weights["w_E"],
        "w_R": weights["w_R"],
        "w_D": weights["w_D"],
        "aggregate_by_config": report["aggregate_by_config"],
        "exclude_neural": filtering["exclude_neural"],
        "context_power_mode": (
            context_policy["context"]["power_mode"] if context_policy is not None else None
        ),
        "context_battery_percent": (
            context_policy["context"]["battery_percent"] if context_policy is not None else None
        ),
        "context_network_profile": (
            context_policy["context"]["network_profile"] if context_policy is not None else None
        ),
        "context_thermal_state": (
            context_policy["context"]["thermal_state"] if context_policy is not None else None
        ),
        "context_quality_target": (
            context_policy["context"]["quality_target"] if context_policy is not None else None
        ),
    }


def topk_rows_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    weights = report["weights"]
    constraints = report["constraints"]
    filtering = report["codec_filtering"]
    normalization = report["normalization"]

    for rank, item in enumerate(report["decision"]["top_k"], start=1):
        norm = item["normalized"]

        rows.append(
            {
                "profile": report["profile"],
                "rank": rank,
                "decision_mode": item["decision_mode"],
                "codec": item["codec"],
                "config": item["config"],
                "rate": item["rate"],
                "quality_mean": item["quality"],
                "quality_constraint_stat": item["quality_constraint_stat"],
                "quality_constraint_value": item["quality_constraint_value"],
                "quality_min": item["quality_stats"]["min"],
                "quality_p10": item["quality_stats"]["p10"],
                "quality_p25": item["quality_stats"]["p25"],
                "energy": item["energy"],
                "time_ms": item["time_ms"],
                "J_RDE": item["cost"],
                "norm_rate": norm["rate"],
                "norm_distortion": norm["distortion"],
                "norm_energy": norm["energy"],
                "normalization_scope": normalization["scope"],
                "num_normalization_reference_points": normalization["num_reference_points"],
                "min_quality": constraints["min_quality"],
                "quality_floor": constraints["quality_floor"],
                "near_quality_floor": constraints["near_quality_floor"],
                "allow_degraded_fallback": constraints["allow_degraded_fallback"],
                "max_rate": constraints["max_rate"],
                "max_energy": constraints["max_energy"],
                "max_time_ms": constraints["max_time_ms"],
                "w_E": weights["w_E"],
                "w_R": weights["w_R"],
                "w_D": weights["w_D"],
                "aggregate_by_config": report["aggregate_by_config"],
                "exclude_neural": filtering["exclude_neural"],
            }
        )

    return rows
