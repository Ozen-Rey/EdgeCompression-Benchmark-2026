import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .calibration.calibration_apply import apply_local_calibration
    from .calibration.calibration_bundle import (
        validate_calibration_bundle_manifest,
        validate_calibration_bundle_validation,
    )
    from .codecs.codec_capabilities import (
        filter_points_by_capabilities,
        is_neural_codec,
        load_external_codec_registry,
    )
    from .cli import build_router_arg_parser
    from .context import RouterContext
    from .adaptation.content_policy import (
        build_content_policy_report,
        get_content_policy_preferred_candidate,
    )
    from .observability.decision_receipt import build_decision_receipt
    from .codecs.external_codec_registry import load_external_codec_points
    from .adaptation.content_classifier_model import (
        build_metadata_no_source_features,
        extract_metadata_features_from_image,
        load_content_classifier_config,
        load_training_rows_from_config,
        predict_content_classifier,
    )
    from .adaptation.context_policy import compute_context_policy
    from .execution import (
        apply_execution_result,
        build_feedback_row as _build_feedback_row,
        write_feedback_report,
    )
    from .core.normalization_profile import load_normalization_profile
    from .core.profiles import available_profiles, get_profile
    from .core.quality_thresholds import resolve_quality_floor
    from .core.rde_database import (
        RDEPoint,
        aggregate_points_by_config,
        filter_points_by_raw_column,
        load_rde_points,
        select_best_rde,
    )
    from .report import build_router_report
    from .core.router_config import expand_argv_with_config
    from .observability.run_manifest import build_run_manifest
    from .adaptation.system_features import build_system_features
    from .adaptation.system_penalty import (
        build_system_penalty_context,
        load_system_penalty_weights,
        make_system_penalty_fn,
    )
    from .adaptation.system_policy import (
        apply_system_policy_simulation,
        build_system_policy,
        parse_system_policy_simulation,
    )
    from .adaptation.system_probe import probe_system
except ImportError:
    from calibration_apply import apply_local_calibration
    from calibration_bundle import (
        validate_calibration_bundle_manifest,
        validate_calibration_bundle_validation,
    )
    from codec_capabilities import (
        filter_points_by_capabilities,
        is_neural_codec,
        load_external_codec_registry,
    )
    from cli import build_router_arg_parser
    from context import RouterContext
    from content_policy import (
        build_content_policy_report,
        get_content_policy_preferred_candidate,
    )
    from decision_receipt import build_decision_receipt
    from external_codec_registry import load_external_codec_points
    from content_classifier_model import (
        build_metadata_no_source_features,
        extract_metadata_features_from_image,
        load_content_classifier_config,
        load_training_rows_from_config,
        predict_content_classifier,
    )
    from context_policy import compute_context_policy
    from execution import (
        apply_execution_result,
        build_feedback_row as _build_feedback_row,
        write_feedback_report,
    )
    from normalization_profile import load_normalization_profile
    from profiles import available_profiles, get_profile
    from quality_thresholds import resolve_quality_floor
    from rde_database import (
        RDEPoint,
        aggregate_points_by_config,
        filter_points_by_raw_column,
        load_rde_points,
        select_best_rde,
    )
    from report import build_router_report
    from router_config import expand_argv_with_config
    from run_manifest import build_run_manifest
    from system_features import build_system_features
    from system_penalty import (
        build_system_penalty_context,
        load_system_penalty_weights,
        make_system_penalty_fn,
    )
    from system_policy import (
        apply_system_policy_simulation,
        build_system_policy,
        parse_system_policy_simulation,
    )
    from system_probe import probe_system


def _normalize_weights(w_e: float, w_r: float, w_d: float) -> Dict[str, float]:
    total = w_e + w_r + w_d

    if total <= 0:
        raise ValueError("La somma dei pesi deve essere positiva.")

    return {
        "w_E": w_e / total,
        "w_R": w_r / total,
        "w_D": w_d / total,
    }


def _normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _parse_codec_list(value: Optional[str]) -> Optional[set[str]]:
    if value is None or value.strip() == "":
        return None

    return {
        _normalize_token(item)
        for item in value.split(",")
        if item.strip()
    }


def _is_neural_codec(codec_name: str) -> bool:
    return is_neural_codec(codec_name)


def _filter_points_by_codec_availability(
    points: List[RDEPoint],
    available_codecs: Optional[set[str]],
    exclude_codecs: Optional[set[str]],
    exclude_neural: bool,
) -> tuple[List[RDEPoint], Dict[str, Any]]:
    filtered: List[RDEPoint] = []

    excluded_by_available = 0
    excluded_by_exclude_list = 0
    excluded_by_neural = 0

    for p in points:
        codec_norm = _normalize_token(p.codec)

        if available_codecs is not None and codec_norm not in available_codecs:
            excluded_by_available += 1
            continue

        if exclude_codecs is not None and codec_norm in exclude_codecs:
            excluded_by_exclude_list += 1
            continue

        if exclude_neural and _is_neural_codec(p.codec):
            excluded_by_neural += 1
            continue

        filtered.append(p)

    filter_report = {
        "available_codecs": sorted(available_codecs) if available_codecs is not None else None,
        "exclude_codecs": sorted(exclude_codecs) if exclude_codecs is not None else None,
        "exclude_neural": exclude_neural,
        "num_before_codec_filtering": len(points),
        "num_after_codec_filtering": len(filtered),
        "excluded_by_available_codecs": excluded_by_available,
        "excluded_by_exclude_codecs": excluded_by_exclude_list,
        "excluded_by_exclude_neural": excluded_by_neural,
    }

    if not filtered:
        raise ValueError(
            "Pool vuoto dopo i filtri codec. "
            "Controlla --available-codecs, --exclude-codecs o --exclude-neural."
        )

    return filtered, filter_report


def _apply_system_aware_policy(
    system_state: Dict[str, Any],
    enabled: bool,
    simulate_no_cuda: bool,
    exclude_neural_requested: bool,
    capability_aware_enabled: bool = False,
) -> tuple[bool, Dict[str, Any]]:
    cuda_available = bool(system_state.get("cuda", {}).get("available", False))

    if simulate_no_cuda:
        cuda_available = False

    effective_exclude_neural = exclude_neural_requested
    rules_applied: list[str] = []

    if exclude_neural_requested:
        rules_applied.append("manual_exclude_neural")

    if enabled:
        if not cuda_available:
            if capability_aware_enabled:
                rules_applied.append(
                    "cuda_unavailable_defer_neural_filtering_to_codec_capabilities"
                )
            else:
                effective_exclude_neural = True
                rules_applied.append("cuda_unavailable_exclude_neural_candidates")
        else:
            rules_applied.append("cuda_available_keep_neural_candidates")

    return effective_exclude_neural, {
        "enabled": enabled,
        "simulate_no_cuda": simulate_no_cuda,
        "cuda_available": cuda_available,
        "exclude_neural_requested": exclude_neural_requested,
        "effective_exclude_neural": effective_exclude_neural,
        "capability_aware_enabled": capability_aware_enabled,
        "rules_applied": rules_applied,
    }


def _build_time_guard_report(
    points,
    max_time_ms,
    strict_time: bool = False,
) -> Dict[str, Any]:
    if max_time_ms is None:
        return {
            "enabled": False,
            "max_time_ms": None,
        }

    total = len(points)

    with_time = []
    missing_time = []
    over_limit = []
    within_limit = []

    for p in points:
        time_ms = getattr(p, "time_ms", None)

        if time_ms is None:
            missing_time.append(
                {
                    "codec": p.codec,
                    "config": p.config,
                    "reason": "missing_time_ms",
                }
            )
            continue

        with_time.append(p)

        if float(time_ms) <= float(max_time_ms):
            within_limit.append(p)
        else:
            over_limit.append(
                {
                    "codec": p.codec,
                    "config": p.config,
                    "time_ms": float(time_ms),
                    "max_time_ms": float(max_time_ms),
                }
            )

    report = {
        "enabled": True,
        "max_time_ms": float(max_time_ms),
        "strict_time": strict_time,
        "num_candidate_points": total,
        "num_with_time": len(with_time),
        "num_missing_time": len(missing_time),
        "num_within_limit": len(within_limit),
        "num_over_limit": len(over_limit),
        "missing_time_preview": missing_time[:20],
        "over_limit_preview": over_limit[:20],
        "warnings": [],
    }

    if total == 0:
        report["warnings"].append("time_guard_enabled_but_candidate_pool_empty")
        return report

    if len(with_time) == 0:
        raise ValueError(
            "Time constraint requested but no time data is available. "
            "You used --max-time-ms, but no candidate point has time_ms. "
            "Provide --time-col with a valid column, remove --max-time-ms, "
            "or use a CSV containing timing data."
        )

    if strict_time and missing_time:
        raise ValueError(
            "Strict time guard requested but some candidate points have no time_ms. "
            f"Missing time for {len(missing_time)} / {total} candidate points."
        )

    if missing_time:
        report["warnings"].append(
            f"{len(missing_time)} candidate points have no time_ms and will be "
            "excluded by the time constraint."
        )

    if len(within_limit) == 0:
        report["warnings"].append(
            "No candidate point with available timing satisfies max_time_ms."
        )

    return report


def _safe_profile_filename(profile_name: str) -> str:
    return profile_name.strip().lower().replace("-", "_").replace(" ", "_")


def _build_content_classifier_router_report(args: argparse.Namespace) -> Dict[str, Any]:
    enabled = bool(getattr(args, "content_classifier", False))
    mode = str(getattr(args, "content_classifier_mode", "report-only"))

    report = {
        "enabled": enabled,
        "mode": mode,
        "applied": False,
        "config": getattr(args, "content_classifier_config", None),
        "prediction": None,
        "features": None,
        "warnings": [],
        "reasons": [],
    }

    if not enabled:
        report["reasons"].append("content_classifier_disabled")
        return report

    if mode not in {"report-only", "apply"}:
        raise ValueError("content classifier mode must be 'report-only' or 'apply'.")

    config_path = getattr(args, "content_classifier_config", None)

    if not config_path:
        report["warnings"].append("content_classifier_enabled_but_missing_config")
        report["reasons"].append("missing_classifier_config")
        return report

    config = load_content_classifier_config(config_path)

    image_path = getattr(args, "content_classifier_image", None)
    width = getattr(args, "content_classifier_width", None)
    height = getattr(args, "content_classifier_height", None)

    if image_path:
        features = extract_metadata_features_from_image(image_path)
    elif width is not None and height is not None:
        features = build_metadata_no_source_features(
            width=int(width),
            height=int(height),
        )
    else:
        report["warnings"].append(
            "content_classifier_enabled_but_no_image_or_dimensions"
        )
        report["reasons"].append("missing_classifier_features")
        return report

    training_rows = load_training_rows_from_config(config)

    classifier_report = predict_content_classifier(
        config=config,
        content_features=features,
        training_rows=training_rows,
    )

    classifier_report["mode"] = mode
    classifier_report["applied"] = False
    classifier_report["config"] = config_path

    return classifier_report


def _build_weights_for_profile(
    args: argparse.Namespace,
    profile_name: str,
) -> tuple[Dict[str, float], float | None, str, Dict[str, Any] | None]:
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


def _write_json_report(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _router_context(args: argparse.Namespace) -> Optional[RouterContext]:
    context = getattr(args, "_router_context", None)
    if isinstance(context, RouterContext):
        return context
    return None


def _context_or_args(
    args: argparse.Namespace,
    context_field: str,
    args_field: str,
    default: Any,
) -> Any:
    context = _router_context(args)
    if context is not None:
        value = getattr(context, context_field)
        if value is not None:
            return value
    return getattr(args, args_field, default)


def _annotate_points_with_calibration_provenance(
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


def _summary_row_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
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


def _write_summary_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _topk_rows_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _write_topk_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_single_decision(report: Dict[str, Any], json_path: Path) -> None:
    selected = report["decision"]["selected"]
    weights = report["weights"]
    filtering = report["codec_filtering"]
    normalization = report["normalization"]
    constraints = report["constraints"]

    print("\n=== R-D-E Router Decision ===")
    print(f"Profile: {report['profile']}")
    router_config = report.get("router_config", {})
    if router_config.get("enabled", False):
        print(f"Router config: {router_config.get('source')}")
        print(f"Experiment: {router_config.get('experiment_name')}")
    run_manifest = report.get("run_manifest", {})
    if run_manifest.get("enabled", False):
        git_info = run_manifest.get("git", {})
        print(
            "Run manifest: "
            f"git={git_info.get('commit_short')}, "
            f"dirty={git_info.get('dirty_worktree')}"
        )
    system_features = report.get("system_features", {})
    print(f"System features: {system_features.get('enabled', False)}")

    if system_features.get("enabled", False):
        overhead = system_features.get("probe_overhead", {})
        system_constraints = system_features.get("derived_constraints", {})
        classes = system_constraints.get("classes", {})

        print(
            "System feature probe: "
            f"level={system_features.get('probe_level')}, "
            f"overhead_ms={overhead.get('total_probe_ms'):.3f}"
        )

        print(
            "System constraints: "
            f"cpu={classes.get('cpu')}, "
            f"memory={classes.get('memory')}, "
            f"battery={classes.get('battery')}, "
            f"gpu={classes.get('gpu')}, "
            f"thermal={classes.get('thermal')}, "
            f"disk={classes.get('disk')}"
        )

        efficiency = report.get("system_probe_efficiency", {})
        if efficiency.get("enabled", False):
            print(
                "System probe efficiency: "
                f"{efficiency.get('classification')} "
                f"(ratio={efficiency.get('overhead_ratio'):.4f})"
            )
    system_policy = report.get("system_policy", {})
    print(f"System policy: {system_policy.get('enabled', False)}")

    if system_policy.get("enabled", False):
        print(
            "System policy mode: "
            f"{system_policy.get('mode')} "
            f"(applied={system_policy.get('applied')})"
        )

        simulation = report.get("system_policy_simulation", {})
        if simulation.get("enabled", False):
            print(
                "System policy simulation: "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in simulation.get("classes", {}).items()
                )
            )

        rules = system_policy.get("rules_applied") or []
        if rules:
            print(f"System policy rules: {', '.join(rules)}")

        suggested = system_policy.get("suggested_weights", {})
        if suggested:
            print(
                "System suggested weights: "
                f"E={suggested.get('w_E'):.3f}, "
                f"R={suggested.get('w_R'):.3f}, "
                f"D={suggested.get('w_D'):.3f}"
            )

    content_policy = report.get("content_policy", {})

    if content_policy.get("enabled", False):
        print(
            "Content policy: "
            f"{content_policy.get('mode')} "
            f"(suggestion={content_policy.get('suggestion') is not None})"
        )

        if content_policy.get("policy_value"):
            print(
                "Content source: "
                f"{content_policy.get('policy_key')}={content_policy.get('policy_value')}"
            )

        suggestion = content_policy.get("suggestion")
        if suggestion:
            print(
                "Content suggestion: "
                f"{suggestion.get('codec')} {suggestion.get('config')}"
            )

        warnings = content_policy.get("warnings") or []
        if warnings:
            print("Content policy warnings: " + "; ".join(warnings))

    content_classifier = report.get("content_classifier", {})

    if content_classifier.get("enabled", False):
        print(
            "Content classifier: "
            f"{content_classifier.get('mode')} "
            f"(prediction={content_classifier.get('prediction') is not None})"
        )

        prediction = content_classifier.get("prediction")
        if prediction:
            print(
                "Content classifier prediction: "
                f"{prediction.get('codec')} {prediction.get('config')}"
            )

        features = content_classifier.get("features") or {}
        if features:
            print(
                "Content classifier features: "
                f"resolution={features.get('resolution_class')}, "
                f"orientation={features.get('orientation_class')}, "
                f"mp={features.get('megapixels')}"
            )

        warnings = content_classifier.get("warnings") or []
        if warnings:
            print("Content classifier warnings: " + "; ".join(warnings))

    content_filter = report.get("content_filter", {})

    if content_filter.get("enabled", False):
        print(
            "Content filter: "
            f"{content_filter.get('column')}={content_filter.get('value')} "
            f"({content_filter.get('before_count')} -> {content_filter.get('after_count')})"
        )

        warnings = content_filter.get("warnings") or []
        if warnings:
            print("Content filter warnings: " + "; ".join(warnings))

    codec_registry = report.get("codec_registry", {})
    print(f"Codec registry: {codec_registry.get('enabled', False)}")
    if codec_registry.get("enabled", False):
        print(f"Codec registry file: {codec_registry.get('source')}")
        print(f"External codecs: {', '.join(codec_registry.get('codecs', []))}")
    print(f"Weight source: {report['weight_source']}")
    print(f"Decision mode: {report['decision']['decision_mode']}")
    print(f"Aggregate by config: {report['aggregate_by_config']}")
    print(f"Exclude neural: {filtering['exclude_neural']}")
    system_aware = filtering.get("system_aware", {})
    print(f"System-aware: {system_aware.get('enabled', False)}")
    if system_aware.get("enabled", False):
        print(f"CUDA available: {system_aware.get('cuda_available')}")
        print(f"Effective exclude neural: {system_aware.get('effective_exclude_neural')}")
    capability = filtering.get("capability_aware", {})
    print(f"Capability-aware: {capability.get('enabled', False)}")
    if capability.get("enabled", False):
        print(f"Strict executables: {capability.get('strict_executables')}")
        print(
            "Capability filtering: "
            f"{capability.get('num_after_capability_filtering')} / "
            f"{capability.get('num_before_capability_filtering')}"
        )
    calibration = report.get("calibration", {})
    print(f"Calibration: {calibration.get('enabled', False)}")
    if calibration.get("enabled", False):
        print(f"Calibration file: {calibration.get('source')}")
        print(f"Calibration level: {calibration.get('level')}")
        print(f"Calibration applied points: {calibration.get('num_applied')}")
    calibration_bundle = report.get("calibration_bundle", {})
    print(f"Calibration bundle: {calibration_bundle.get('enabled', False)}")
    if calibration_bundle.get("enabled", False):
        print(f"Bundle manifest: {calibration_bundle.get('manifest_path')}")
        print(f"Bundle CSV: {calibration_bundle.get('calibrated_csv_path')}")
        print(f"Bundle validated: {calibration_bundle.get('validated')}")
    calibration_bundle_validation = report.get("calibration_bundle_validation", {})
    print(
        "Calibration bundle validation: "
        f"{calibration_bundle_validation.get('enabled', False)}"
    )
    if calibration_bundle_validation.get("enabled", False):
        print(
            "Bundle validation accepted: "
            f"{calibration_bundle_validation.get('accepted')}"
        )
        print(
            "Bundle validation file: "
            f"{calibration_bundle_validation.get('validation_path')}"
        )
    normalization_profile = report.get("normalization_profile", {})
    if normalization_profile.get("enabled", False):
        print(
            "Normalization: "
            f"{normalization_profile.get('mode')} "
            f"({normalization_profile.get('source')})"
        )
    else:
        print(f"Normalization: {normalization_profile.get('mode', 'runtime')}")
    if normalization_profile.get("warning"):
        print(f"Normalization warning: {normalization_profile.get('warning')}")
    print(
        f"Quality guard: {constraints['quality_constraint_stat']} >= "
        f"{constraints['min_quality']} "
        f"(near={constraints['near_quality_floor']}, degraded={constraints['allow_degraded_fallback']})"
    )
    quality_thresholds = report.get("quality_thresholds", {})

    if quality_thresholds.get("enabled", False):
        print(
            "Quality thresholds: "
            f"domain={quality_thresholds.get('domain')}, "
            f"metric={quality_thresholds.get('quality_metric')}, "
            f"target={quality_thresholds.get('quality_target')}, "
            f"target_floor={quality_thresholds.get('target_floor')}, "
            f"user_floor={quality_thresholds.get('user_quality_floor')}, "
            f"effective_floor={quality_thresholds.get('effective_quality_floor')}"
        )

    time_guard = report.get("time_guard", {})

    if time_guard.get("enabled", False):
        print(
            "Time guard: "
            f"max_time_ms <= {time_guard.get('max_time_ms')} "
            f"(with_time={time_guard.get('num_with_time')}/"
            f"{time_guard.get('num_candidate_points')})"
        )

        for warning in time_guard.get("warnings", []):
            print(f"Time guard warning: {warning}")

    print(
        f"Weights: "
        f"E={weights['w_E']:.3f}, "
        f"R={weights['w_R']:.3f}, "
        f"D={weights['w_D']:.3f}"
    )
    print(f"Loaded rows: {report['num_rows_loaded_before_aggregation']}")
    print(f"Candidate points after codec filtering: {filtering['num_after_codec_filtering']}")
    print(f"Candidate points: {report['decision']['num_points_total']}")
    print(
        f"Safe points: {report['decision']['num_points_safe']} | "
        f"Near points: {report['decision']['num_points_near']}"
    )
    print(
        f"Admissible points: "
        f"{report['decision']['num_points_admissible']} / "
        f"{report['decision']['num_points_total']}"
    )
    print()
    print(f"Selected codec:             {selected['codec']}")
    print(f"Selected config:            {selected['config']}")
    print(f"Rate:                       {selected['rate']}")
    print(f"Quality mean:               {selected['quality']}")
    print(f"Quality guard value:        {selected['quality_constraint_value']}")
    print(f"Quality min / p10 / p25:    {selected['quality_stats']['min']} / {selected['quality_stats']['p10']} / {selected['quality_stats']['p25']}")
    print(f"Energy:                     {selected['energy']}")
    print(f"Time ms:                    {selected['time_ms']}")
    print(f"J_RDE:                      {selected['cost']:.6f}")
    system_penalty = selected.get("system_penalty", {})
    if system_penalty.get("enabled", False):
        print(
            "System penalty:            "
            f"P={system_penalty.get('penalty_norm'):.6f}, "
            f"lambda={system_penalty.get('lambda_sys')}, "
            f"weighted={system_penalty.get('weighted_penalty'):.6f}"
        )
        print(f"J_total:                   {selected.get('J_total'):.6f}")

        penalty_rules = system_penalty.get("rules_applied") or []
        if penalty_rules:
            print(f"System penalty rules:      {', '.join(penalty_rules)}")

    cost_decomp = selected.get("cost_decomposition", {})
    if cost_decomp:
        print(
            "Cost decomposition:         "
            f"R={cost_decomp.get('term_R'):.6f}, "
            f"E={cost_decomp.get('term_E'):.6f}, "
            f"D={cost_decomp.get('term_D'):.6f}"
        )

    decision_trace = report.get("decision", {}).get("decision_trace", {})
    if decision_trace.get("enabled", False):
        print(f"Selected reason:            {decision_trace.get('selected_reason')}")

    selected_calibration = report.get("selected_calibration", {})

    if selected_calibration.get("enabled", False):
        print()
        print("Selected calibration:")
        print(f"  rate:    {selected_calibration.get('rate_before')} -> {selected_calibration.get('rate_after')}")
        print(f"  energy:  {selected_calibration.get('energy_before')} -> {selected_calibration.get('energy_after')}")
        print(f"  time ms: {selected_calibration.get('time_ms_before')} -> {selected_calibration.get('time_ms_after')}")
        print(f"  method:  {selected_calibration.get('energy_scaling_method')}")
        print(f"  backend: {selected_calibration.get('energy_backend')}")
        print(f"  scope:   {selected_calibration.get('energy_scope')}")
        print(f"  usable total: {selected_calibration.get('energy_usable_for_total')}")

    execution_plan = report.get("execution_plan", {})
    if execution_plan.get("requested", False):
        print()
        print("Execution plan:")
        print(f"  backend:      {execution_plan.get('execution_backend')}")
        print(f"  can_execute:  {execution_plan.get('can_execute')}")
        print(f"  output:       {execution_plan.get('output')}")

        command = execution_plan.get("command")
        if command:
            print(f"  command:      {' '.join(command)}")

        reasons = execution_plan.get("reasons") or []
        if reasons:
            print(f"  reasons:      {', '.join(reasons)}")

        warnings = execution_plan.get("warnings") or []
        if warnings:
            print(f"  warnings:     {', '.join(warnings)}")

    print()
    print(f"Report written to: {json_path}")


def _run_profile(
    args: argparse.Namespace,
    profile_name: str,
    points: List[RDEPoint],
    normalization_points: List[RDEPoint],
    normalization_scope: str,
    csv_path: str,
    num_rows_loaded: int,
    system_state: Dict[str, Any],
    filter_report: Dict[str, Any],
) -> Dict[str, Any]:
    weights, min_quality, weight_source, context_policy = _build_weights_for_profile(
        args,
        profile_name,
    )
    router_context = _router_context(args)

    system_policy_report = build_system_policy(
        base_weights=weights,
        system_features_report=_context_or_args(
            args,
            "system_features_report",
            "_system_features_report",
            {"enabled": False},
        ),
        enabled=args.system_policy,
        mode=args.system_policy_mode,
    )

    if router_context is not None:
        router_context.system_policy_report = system_policy_report

    if system_policy_report.get("enabled", False) and system_policy_report.get("applied", False):
        weights = system_policy_report["effective_weights"]
        weight_source = f"{weight_source}+system_policy"

    content_policy_report = build_content_policy_report(
        enabled=bool(getattr(args, "content_policy", False)),
        mode=str(getattr(args, "content_policy_mode", "report-only")),
        rules_file=getattr(args, "content_policy_rules_file", None),
        policy_key=str(getattr(args, "content_policy_key", "dataset")),
        policy_value=getattr(args, "content_source", None),
        fallback="router",
    )

    if router_context is not None:
        router_context.content_policy_report = content_policy_report

    content_classifier_report = _build_content_classifier_router_report(args)
    if router_context is not None:
        router_context.content_classifier_report = content_classifier_report

    preferred_codec = None
    preferred_config = None
    preferred_reason = None
    preferred_source = None

    content_preferred = get_content_policy_preferred_candidate(content_policy_report)

    if content_preferred is not None:
        preferred_codec, preferred_config = content_preferred
        preferred_reason = "content_policy_preferred_candidate"
        preferred_source = "content_policy"

    classifier_prediction = (
        content_classifier_report.get("prediction")
        if content_classifier_report
        else None
    )

    if (
        preferred_codec is None
        and content_classifier_report.get("enabled")
        and content_classifier_report.get("mode") == "apply"
        and classifier_prediction
    ):
        preferred_codec = str(classifier_prediction.get("codec"))
        preferred_config = str(classifier_prediction.get("config"))
        preferred_reason = str(
            content_classifier_report.get(
                "selection_reason",
                "content_classifier_preferred_candidate",
            )
        )
        preferred_source = "content_classifier"

    if router_context is not None:
        router_context.preferred_candidate_source = preferred_source

    system_penalty_weights_report = load_system_penalty_weights(
        args.system_penalty_weights_file
    )

    if router_context is not None:
        router_context.system_penalty_weights_report = system_penalty_weights_report

    system_penalty_context = build_system_penalty_context(
        enabled=args.system_penalty,
        mode=args.system_penalty_mode,
        lambda_sys=args.system_penalty_lambda,
        system_features_report=_context_or_args(
            args,
            "system_features_report",
            "_system_features_report",
            {"enabled": False},
        ),
        latency_constrained=args.max_time_ms is not None,
        execution_requested=bool(args.execute),
        penalty_weights=system_penalty_weights_report["weights"],
        penalty_weights_source=system_penalty_weights_report["source"],
    )

    if router_context is not None:
        router_context.system_penalty_report = system_penalty_context

    system_penalty_fn = (
        make_system_penalty_fn(system_penalty_context)
        if system_penalty_context.get("enabled", False)
        else None
    )

    near_quality_floor = args.near_quality_floor

    if args.allow_degraded_fallback and near_quality_floor is None:
        near_quality_floor = max(0.0, min_quality - 10.0)

    time_guard_report = _build_time_guard_report(
        points=points,
        max_time_ms=args.max_time_ms,
        strict_time=args.strict_time,
    )

    if router_context is not None:
        router_context.time_guard_report = time_guard_report

    decision = select_best_rde(
        points=points,
        weights=weights,
        min_quality=min_quality,
        max_rate=args.max_rate,
        max_energy=args.max_energy,
        max_time_ms=args.max_time_ms,
        quality_constraint_stat=args.quality_constraint_stat,
        near_quality_floor=near_quality_floor,
        allow_degraded_fallback=args.allow_degraded_fallback,
        top_k=args.top_k,
        normalization_points=normalization_points,
        normalization_profile=_context_or_args(
            args,
            "normalization_profile",
            "_normalization_profile",
            None,
        ),
        system_penalty_fn=system_penalty_fn,
        system_penalty_apply=(
            system_penalty_context.get("enabled", False)
            and system_penalty_context.get("applied", False)
        ),
        preferred_codec=preferred_codec,
        preferred_config=preferred_config,
        preferred_reason=preferred_reason or "preferred_candidate",
    )

    if content_policy_report.get("enabled") and content_policy_report.get("mode") == "apply":
        selected = decision.get("selected", {})
        suggestion = content_policy_report.get("suggestion")

        if suggestion:
            suggested_codec = str(suggestion.get("codec"))
            suggested_config = str(suggestion.get("config"))

            selected_codec = str(selected.get("codec"))
            selected_config = str(selected.get("config"))

            if selected_codec == suggested_codec and selected_config == suggested_config:
                content_policy_report["applied"] = True
                content_policy_report["reasons"].append(
                    "content_policy_suggestion_selected"
                )
            else:
                content_policy_report["applied"] = False

                preferred_audit = (
                    decision.get("decision_trace", {})
                    .get("preferred_candidate")
                )

                content_policy_report["decision_audit"] = preferred_audit

                if preferred_audit and preferred_audit.get("admissible") is True:
                    content_policy_report["warnings"].append(
                        "content_policy_suggestion_not_j_total_competitive_fallback_to_router"
                    )
                    content_policy_report["reasons"].append(
                        "suggestion_admissible_but_not_competitive"
                    )
                else:
                    content_policy_report["warnings"].append(
                        "content_policy_suggestion_not_admissible_fallback_to_router"
                    )
                    content_policy_report["reasons"].append(
                        "suggestion_not_admissible"
                    )

                content_policy_report["reasons"].append(
                    "fallback_to_router_selection"
                )

    if (
        content_classifier_report.get("enabled")
        and content_classifier_report.get("mode") == "apply"
    ):
        selected = decision.get("selected", {})
        prediction = content_classifier_report.get("prediction")

        if prediction:
            predicted_codec = str(prediction.get("codec"))
            predicted_config = str(prediction.get("config"))

            selected_codec = str(selected.get("codec"))
            selected_config = str(selected.get("config"))

            if selected_codec == predicted_codec and selected_config == predicted_config:
                content_classifier_report["applied"] = True
                content_classifier_report["reasons"].append(
                    "content_classifier_prediction_selected"
                )
            else:
                content_classifier_report["applied"] = False

                preferred_audit = (
                    decision.get("decision_trace", {})
                    .get("preferred_candidate")
                )

                content_classifier_report["decision_audit"] = preferred_audit

                if preferred_audit and preferred_audit.get("admissible") is True:
                    content_classifier_report["warnings"].append(
                        "content_classifier_prediction_not_j_total_competitive_fallback_to_router"
                    )
                    content_classifier_report["reasons"].append(
                        "prediction_admissible_but_not_competitive"
                    )
                else:
                    content_classifier_report["warnings"].append(
                        "content_classifier_prediction_not_admissible_fallback_to_router"
                    )
                    content_classifier_report["reasons"].append(
                        "prediction_not_admissible"
                    )

                content_classifier_report["reasons"].append(
                    "fallback_to_router_selection"
                )

    return build_router_report(
        args=args,
        profile_name=profile_name,
        weights=weights,
        min_quality=min_quality,
        near_quality_floor=near_quality_floor,
        decision=decision,
        csv_path=csv_path,
        num_rows_loaded=num_rows_loaded,
        system_state=system_state,
        filter_report=filter_report,
        normalization_scope=normalization_scope,
        normalization_reference_count=len(normalization_points),
        weight_source=weight_source,
        context_policy=context_policy,
    )


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    original_argv = list(argv)

    argv, router_config_report = expand_argv_with_config(argv)

    expanded_argv = list(argv)

    parser = build_router_arg_parser()
    args = parser.parse_args(argv)
    router_context = RouterContext()
    args._router_context = router_context
    router_context.router_config_report = router_config_report
    if args.system_features:
        router_context.system_features_report = build_system_features(
            probe_level=args.system_probe_level,
            cache_ttl_s=args.system_feature_cache_ttl_s,
            cpu_interval_s=args.system_feature_cpu_interval_s,
        )
    else:
        router_context.system_features_report = {
            "enabled": False,
        }

    simulated_system_classes = parse_system_policy_simulation(
        args.system_policy_simulate
    )

    router_context.system_policy_simulation = {
        "enabled": bool(simulated_system_classes),
        "classes": simulated_system_classes,
    }

    if simulated_system_classes:
        router_context.system_features_report = apply_system_policy_simulation(
            system_features_report=router_context.system_features_report,
            simulated_classes=simulated_system_classes,
        )

    router_context.run_manifest = build_run_manifest(
        original_argv=original_argv,
        expanded_argv=expanded_argv,
        args=args,
        router_config_report=router_config_report,
    )

    if args.codec_registry_file:
        registry_report = load_external_codec_registry(args.codec_registry_file)
    else:
        registry_report = {
            "enabled": False,
        }

    router_context.codec_registry_report = registry_report

    if args.execute:
        args.generate_command = True

    if args.execute and args.all_profiles:
        raise ValueError("--execute è supportato solo in modalità singolo profilo, non con --all-profiles.")

    if args.execute and args.input is None:
        raise ValueError("--execute richiede --input.")

    if args.auto_weights and args.all_profiles:
        raise ValueError(
            "--auto-weights genera un singolo profilo contestuale; "
            "non usarlo insieme a --all-profiles."
        )

    if args.safe_mode:
        if args.quality_constraint_stat is None:
            args.quality_constraint_stat = "p10"
        if args.quality_floor is None:
            args.quality_floor = 60.0
        else:
            args.quality_floor = max(args.quality_floor, 60.0)
    else:
        if args.quality_constraint_stat is None:
            args.quality_constraint_stat = "mean"

    quality_threshold_report = resolve_quality_floor(
        domain=args.domain,
        quality_metric=args.quality_metric or args.quality_col,
        quality_target=args.quality_target,
        user_quality_floor=args.quality_floor,
        thresholds_file=args.quality_thresholds_file,
    )

    args.quality_floor = quality_threshold_report["effective_quality_floor"]
    router_context.quality_threshold_report = quality_threshold_report

    effective_csv_path = args.csv
    if args.calibration_bundle_validation and not args.calibration_bundle_manifest:
        raise ValueError(
            "--calibration-bundle-validation requires "
            "--calibration-bundle-manifest. No automatic bundle discovery is "
            "performed."
        )

    if args.calibration_bundle_manifest:
        if args.calibration_file:
            raise ValueError(
                "--calibration-bundle-manifest cannot be combined with "
                "--calibration-file. Use a prebuilt calibrated CSV bundle, or "
                "apply local calibration separately before routing."
            )

        calibration_bundle_report = validate_calibration_bundle_manifest(
            args.calibration_bundle_manifest
        )

        if args.calibration_bundle_validation:
            calibration_bundle_validation_report = (
                validate_calibration_bundle_validation(
                    args.calibration_bundle_validation,
                    bundle_manifest_path=args.calibration_bundle_manifest,
                )
            )
            if calibration_bundle_validation_report.get("accepted") is not True:
                reasons = calibration_bundle_validation_report.get(
                    "rejection_reasons",
                    [],
                )
                raise ValueError(
                    "Calibration bundle validation was not accepted: "
                    + ", ".join(str(reason) for reason in reasons)
                )
        else:
            calibration_bundle_validation_report = {
                "enabled": False,
            }

        effective_csv_path = calibration_bundle_report["calibrated_csv_path"]
    else:
        calibration_bundle_report = {
            "enabled": False,
        }
        calibration_bundle_validation_report = {
            "enabled": False,
        }

    router_context.calibration_bundle_report = calibration_bundle_report
    router_context.calibration_bundle_validation_report = (
        calibration_bundle_validation_report
    )

    points = load_rde_points(
        csv_path=effective_csv_path,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        time_col=args.time_col,
    )

    if args.external_codec_manifest:
        external_points, external_codecs_report = load_external_codec_points(
            args.external_codec_manifest
        )
        points.extend(external_points)
    else:
        external_codecs_report = {
            "enabled": False,
        }

    router_context.external_codecs_report = external_codecs_report

    content_filter_report = {
        "enabled": bool(getattr(args, "content_source_filter", False)),
        "applied": False,
        "column": None,
        "value": None,
        "before_count": len(points),
        "after_count": len(points),
        "warnings": [],
    }

    if getattr(args, "content_source_filter", False):
        filter_column = (
            getattr(args, "content_filter_column", None)
            or getattr(args, "content_policy_key", "dataset")
        )

        filter_value = (
            getattr(args, "content_filter_value", None)
            or getattr(args, "content_source", None)
        )

        content_filter_report["column"] = filter_column
        content_filter_report["value"] = filter_value

        if filter_value is None or str(filter_value).strip() == "":
            content_filter_report["warnings"].append(
                "content_source_filter_enabled_but_no_filter_value"
            )
        else:
            filtered_points = filter_points_by_raw_column(
                points,
                column=filter_column,
                value=filter_value,
            )

            content_filter_report["after_count"] = len(filtered_points)
            content_filter_report["applied"] = True

            if len(filtered_points) == 0:
                raise ValueError(
                    "Content/source filter removed all candidate rows: "
                f"{filter_column}={filter_value}"
                )

            points = filtered_points

    router_context.content_filter_report = content_filter_report

    num_rows_loaded = len(points)

    if args.aggregate_by_config:
        points = aggregate_points_by_config(points)

    if args.calibration_file:
        points, calibration_report = apply_local_calibration(
            points=points,
            calibration_file=args.calibration_file,
        )
        _annotate_points_with_calibration_provenance(
            points=points,
            calibration_report=calibration_report,
        )
    else:
        calibration_report = {
            "enabled": False,
        }

    router_context.calibration_report = calibration_report

    normalization_mode = args.normalization_mode

    if normalization_mode == "runtime":
        if args.normalization_file:
            raise ValueError(
                "--normalization-mode runtime non deve essere usato insieme a --normalization-file."
            )

        normalization_profile = None
        normalization_scope_label = "runtime_global_before_codec_filtering"
        normalization_report = {
            "enabled": False,
            "mode": "runtime",
            "source": None,
            "scope": normalization_scope_label,
            "comparability": "run_local",
            "warning": (
                "Normalization is computed at runtime; J_RDE values may not be "
                "comparable across runs with different candidate pools."
            ),
        }

    elif normalization_mode == "auto":
        if args.normalization_file:
            normalization_profile = load_normalization_profile(args.normalization_file)
            profile_mode = normalization_profile.get("mode", "profile")
            normalization_scope_label = f"precomputed_{profile_mode}_profile"
            normalization_report = {
                "enabled": True,
                "mode": profile_mode,
                "source": args.normalization_file,
                "scope": normalization_scope_label,
                "comparability": normalization_profile.get("comparability"),
                "warning": normalization_profile.get("warning"),
            }
        else:
            normalization_profile = None
            normalization_scope_label = "runtime_global_before_codec_filtering"
            normalization_report = {
                "enabled": False,
                "mode": "runtime",
                "source": None,
                "scope": normalization_scope_label,
                "comparability": "run_local",
                "warning": (
                    "Normalization is computed at runtime; J_RDE values may not be "
                    "comparable across runs with different candidate pools."
                ),
            }

    else:
        if not args.normalization_file:
            raise ValueError(
                f"--normalization-mode {normalization_mode} richiede --normalization-file."
            )

        normalization_profile = load_normalization_profile(args.normalization_file)
        profile_mode = normalization_profile.get("mode")

        if profile_mode and profile_mode != normalization_mode:
            raise ValueError(
                f"Normalization mode mismatch: CLI mode={normalization_mode}, "
                f"profile mode={profile_mode}."
            )

        normalization_scope_label = f"precomputed_{normalization_mode}_profile"
        normalization_report = {
            "enabled": True,
            "mode": normalization_mode,
            "source": args.normalization_file,
            "scope": normalization_scope_label,
            "comparability": normalization_profile.get("comparability"),
            "warning": normalization_profile.get("warning"),
        }

    router_context.normalization_profile = normalization_profile
    router_context.normalization_report = normalization_report

    global_normalization_points = list(points)

    system_state = probe_system()

    effective_exclude_neural, system_aware_report = _apply_system_aware_policy(
        system_state=system_state,
        enabled=args.system_aware,
        simulate_no_cuda=args.simulate_no_cuda,
        exclude_neural_requested=args.exclude_neural,
        capability_aware_enabled=args.capability_aware,
    )

    available_codecs = _parse_codec_list(args.available_codecs)
    exclude_codecs = _parse_codec_list(args.exclude_codecs)

    points, filter_report = _filter_points_by_codec_availability(
        points=points,
        available_codecs=available_codecs,
        exclude_codecs=exclude_codecs,
        exclude_neural=effective_exclude_neural,
    )

    filter_report["system_aware"] = system_aware_report

    if args.capability_aware:
        points, capability_report = filter_points_by_capabilities(
            points=points,
            system_state=system_state,
            strict_executables=args.strict_executables,
            simulate_no_cuda=args.simulate_no_cuda,
        )

        filter_report["capability_aware"] = capability_report
    else:
        filter_report["capability_aware"] = {
            "enabled": False,
        }

    if normalization_profile is not None:
        normalization_points = global_normalization_points
        normalization_scope_label = normalization_report["scope"]
    elif args.normalization_scope == "global":
        normalization_points = global_normalization_points
        normalization_scope_label = normalization_report["scope"]
    else:
        normalization_points = points
        normalization_scope_label = "filtered_after_codec_filtering"
        normalization_report["scope"] = normalization_scope_label
        router_context.normalization_report = normalization_report

    if args.all_profiles:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[Dict[str, Any]] = []
        topk_rows: List[Dict[str, Any]] = []

        print("\n=== R-D-E Router: all profiles ===")
        print(f"Loaded rows: {num_rows_loaded}")
        print(f"Candidate points after aggregation/filtering: {len(points)}")
        print(f"Normalization: {normalization_scope_label} ({len(normalization_points)} reference points)")
        print(f"Aggregate by config: {args.aggregate_by_config}")
        print(f"Available codecs: {args.available_codecs}")
        print(f"Exclude codecs: {args.exclude_codecs}")
        print(f"Exclude neural: {args.exclude_neural}")
        print(f"System-aware: {args.system_aware}")
        if args.system_aware:
            print(f"CUDA available: {filter_report['system_aware']['cuda_available']}")
            print(f"Effective exclude neural: {filter_report['system_aware']['effective_exclude_neural']}")
        print(f"Capability-aware: {args.capability_aware}")
        print(f"Strict executables: {args.strict_executables}")
        print(f"Safe mode: {args.safe_mode}")
        print(f"Quality guard: {args.quality_constraint_stat} >= {args.quality_floor}")
        print(f"Export top-k: {args.export_topk}")
        print()

        for profile_name in available_profiles():
            report = _run_profile(
                args=args,
                profile_name=profile_name,
                points=points,
                normalization_points=normalization_points,
        normalization_scope=normalization_scope_label,
                csv_path=effective_csv_path,
                num_rows_loaded=num_rows_loaded,
                system_state=system_state,
                filter_report=filter_report,
            )

            safe_name = _safe_profile_filename(profile_name)
            json_path = out_dir / f"router_decision_report_{safe_name}.json"
            report["decision_receipt"] = build_decision_receipt(report)
            _write_json_report(report, json_path)

            summary_rows.append(_summary_row_from_report(report))

            if args.export_topk:
                topk_rows.extend(_topk_rows_from_report(report))

            selected = report["decision"]["selected"]
            print(
                f"{profile_name:18s} -> "
                f"{selected['codec']} {selected['config']} "
                f"| mode={report['decision']['decision_mode']} "
                f"| R={selected['rate']:.6f}, "
                f"Qmean={selected['quality']:.2f}, "
                f"Qguard={selected['quality_constraint_value']:.2f}, "
                f"E={selected['energy']:.6f}, "
                f"J={selected['cost']:.6f}"
            )

        summary_path = (
            Path(args.summary_out)
            if args.summary_out is not None
            else out_dir / "router_summary.csv"
        )

        _write_summary_csv(summary_rows, summary_path)

        if args.export_topk:
            topk_path = out_dir / "router_topk.csv"
            _write_topk_csv(topk_rows, topk_path)

        print()
        print(f"JSON reports written to: {out_dir}")
        print(f"Summary written to:      {summary_path}")

        if args.export_topk:
            print(f"Top-k written to:        {topk_path}")

    else:
        report = _run_profile(
            args=args,
            profile_name="context-auto" if args.auto_weights else args.profile,
            points=points,
            normalization_points=normalization_points,
            normalization_scope=normalization_scope_label,
            csv_path=effective_csv_path,
            num_rows_loaded=num_rows_loaded,
            system_state=system_state,
            filter_report=filter_report,
        )

        out_path = Path(args.out)

        apply_execution_result(report, execute=args.execute)

        report["decision_receipt"] = build_decision_receipt(report)
        _write_json_report(report, out_path)

        if args.execute and args.feedback_out:
            write_feedback_report(
                report,
                feedback_out=args.feedback_out,
                report_path=out_path,
            )
            _write_json_report(report, out_path)

        if args.export_topk:
            topk_path = out_path.with_name(out_path.stem + "_topk.csv")
            _write_topk_csv(_topk_rows_from_report(report), topk_path)

        _print_single_decision(report, out_path)

        if args.execute:
            result = report["execution_result"]
            print()
            print("Execution result:")
            print(f"  executed:     {result.get('executed')}")
            print(f"  success:      {result.get('success')}")
            print(f"  returncode:   {result.get('returncode')}")

            if result.get("output"):
                print(f"  output:       {result.get('output')}")

            if result.get("reason"):
                print(f"  reason:       {result.get('reason')}")

            if result.get("stderr") and not result.get("success"):
                print("  stderr:")
                print(result.get("stderr"))

            validation = report.get("execution_validation", {})
            if validation.get("enabled", False):
                print()
                print("Execution validation:")
                print(f"  output exists: {validation.get('output_exists')}")
                print(f"  output size:   {validation.get('output_size_bytes')}")
                print(f"  nonempty:      {validation.get('output_nonempty')}")
                print(f"  ext valid:     {validation.get('extension_valid')}")
                print(f"  time ms:       {validation.get('execution_time_ms')}")

                warnings = validation.get("warnings") or []
                if warnings:
                    print(f"  warnings:      {', '.join(warnings)}")

        if args.export_topk:
            print(f"Top-k written to: {topk_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print("\n=== R-D-E Router: infeasible request ===")
        print(str(exc))
        print()
        print("No codec/configuration can satisfy the current constraints.")
        print("Try one of the following:")
        print("  - relax --max-rate")
        print("  - lower --quality-floor or --near-quality-floor")
        print("  - disable --simulate-no-cuda if CUDA codecs are actually available")
        print("  - enable more codecs in the admissible pool")
        raise SystemExit(2)
