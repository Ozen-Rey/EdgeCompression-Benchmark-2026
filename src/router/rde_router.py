import argparse
import csv
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.router.version import DOMAIN_SUPPORT, FEATURE_LEVEL, ROUTER_VERSION
    from src.utils.energy_backends import CompositeEnergyMeter
    from .calibration_apply import apply_local_calibration
    from .calibration_bundle import (
        validate_calibration_bundle_manifest,
        validate_calibration_bundle_validation,
    )
    from .codec_capabilities import (
        build_execution_plan,
        filter_points_by_capabilities,
        is_neural_codec,
        load_external_codec_registry,
    )
    from .content_policy import (
        build_content_policy_report,
        get_content_policy_preferred_candidate,
    )
    from .decision_receipt import build_decision_receipt
    from .energy_provenance_compatibility import (
        build_energy_provenance_compatibility_audit,
    )
    from .energy_provenance import build_energy_provenance_summary
    from .content_classifier_model import (
        build_metadata_no_source_features,
        extract_metadata_features_from_image,
        load_content_classifier_config,
        load_training_rows_from_config,
        predict_content_classifier,
    )
    from .context_policy import compute_context_policy
    from .execution_validation import validate_execution_output
    from .feedback_logger import append_feedback_row
    from .normalization_consistency import (
        NormalizationAuditLoadError,
        compare_normalization_audits,
        load_previous_normalization_audit,
    )
    from .normalization_profile import load_normalization_profile
    from .profiles import available_profiles, get_profile
    from .quality_thresholds import resolve_quality_floor
    from .rde_database import (
        RDEPoint,
        aggregate_points_by_config,
        filter_points_by_raw_column,
        load_rde_points,
        select_best_rde,
    )
    from .router_config import expand_argv_with_config
    from .run_manifest import build_run_manifest
    from .system_features import build_system_features, estimate_probe_efficiency
    from .system_penalty import (
        build_system_penalty_context,
        load_system_penalty_weights,
        make_system_penalty_fn,
    )
    from .system_policy import (
        apply_system_policy_simulation,
        build_system_policy,
        parse_system_policy_simulation,
    )
    from .system_probe import probe_system
except ImportError:
    from version import DOMAIN_SUPPORT, FEATURE_LEVEL, ROUTER_VERSION
    sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
    from energy_backends import CompositeEnergyMeter
    from calibration_apply import apply_local_calibration
    from calibration_bundle import (
        validate_calibration_bundle_manifest,
        validate_calibration_bundle_validation,
    )
    from codec_capabilities import (
        build_execution_plan,
        filter_points_by_capabilities,
        is_neural_codec,
        load_external_codec_registry,
    )
    from content_policy import (
        build_content_policy_report,
        get_content_policy_preferred_candidate,
    )
    from decision_receipt import build_decision_receipt
    from energy_provenance_compatibility import (
        build_energy_provenance_compatibility_audit,
    )
    from energy_provenance import build_energy_provenance_summary
    from content_classifier_model import (
        build_metadata_no_source_features,
        extract_metadata_features_from_image,
        load_content_classifier_config,
        load_training_rows_from_config,
        predict_content_classifier,
    )
    from context_policy import compute_context_policy
    from execution_validation import validate_execution_output
    from feedback_logger import append_feedback_row
    from normalization_consistency import (
        NormalizationAuditLoadError,
        compare_normalization_audits,
        load_previous_normalization_audit,
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
    from router_config import expand_argv_with_config
    from run_manifest import build_run_manifest
    from system_features import build_system_features, estimate_probe_efficiency
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


def _build_energy_provenance_report(
    calibration_report: Optional[Dict[str, Any]] = None,
    selected_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_calibration = selected_calibration or {}
    calibration_report = calibration_report or {}

    energy_backend = "benchmark_csv"
    energy_method = "benchmark_energy_from_input_csv"
    energy_is_measured = False
    energy_quality = "benchmark_derived"
    energy_scope = "none"
    energy_usable_for_total = False
    local_measurement_energy_is_measured = False
    current_method = "benchmark_energy"
    energy_mode = calibration_report.get("energy_mode", "auto")

    if selected_calibration.get("enabled", False):
        energy_backend = selected_calibration.get("energy_backend") or energy_backend
        energy_method = selected_calibration.get("energy_method") or energy_method
        energy_quality = selected_calibration.get("energy_quality") or energy_quality
        energy_scope = selected_calibration.get("energy_scope") or energy_scope
        energy_usable_for_total = bool(
            selected_calibration.get("energy_usable_for_total", False)
        )
        local_measurement_energy_is_measured = bool(
            selected_calibration.get("energy_is_measured", False)
        )

        scaling_method = selected_calibration.get("energy_scaling_method")
        if scaling_method == "local_hardware_energy_total":
            current_method = "local_hardware_energy_total"
            energy_is_measured = True
        elif scaling_method == "benchmark_only_energy_mode":
            current_method = "benchmark_only_energy_mode"
            energy_is_measured = False
        elif str(scaling_method).startswith("benchmark_energy_scaled_by_time_ratio"):
            current_method = str(scaling_method)
            energy_is_measured = False

    return {
        "local_energy_measurement": "hardware_backend_or_fallback",
        "energy_mode": energy_mode,
        "current_method": current_method,
        "energy_backend": energy_backend,
        "energy_method": energy_method,
        "energy_is_measured": energy_is_measured,
        "energy_quality": energy_quality,
        "energy_scope": energy_scope,
        "energy_usable_for_total": energy_usable_for_total,
        "local_measurement_energy_is_measured": local_measurement_energy_is_measured,
        "hardware_backends": [
            part
            for part in str(energy_backend).split(";")
            if part
            and not part.endswith("=none")
            and part not in {"benchmark_csv", "benchmark_only"}
        ],
        "fallback": "benchmark_energy_scaled_by_time_ratio_when_calibration_is_used",
        "calibration_energy_measurement": calibration_report.get(
            "energy_measurement", {}
        ),
        "warning": (
            "Energy values are hardware-measured only when energy_is_measured=true. "
            "Otherwise they are benchmark-derived or time-scaled estimates."
        ),
    }


def _build_normalization_audit(
    normalization_report: Dict[str, Any],
    normalization_profile: Optional[Dict[str, Any]],
    normalization_reference: Dict[str, Any],
    quality_metric: Optional[str],
) -> Dict[str, Any]:
    mode = normalization_report.get("mode", "runtime")
    source = normalization_report.get("source")
    computed_at_runtime = not normalization_report.get("enabled", False)

    audit: Dict[str, Any] = {
        "mode": mode,
        "scales_source": source or "computed_at_runtime",
        "computed_at_runtime": computed_at_runtime,
        "scope": normalization_report.get("scope"),
        "comparability": normalization_report.get("comparability"),
        "quality_metric": quality_metric,
        "quality_direction": "higher_is_better",
        "num_reference_points": normalization_reference.get("num_reference_points"),
    }

    if normalization_profile is not None:
        scales = normalization_profile.get("scales", {})
        transforms = normalization_profile.get("transforms", {})
        audit.update({
            "rate_scale": transforms.get("rate", "log10"),
            "rate_min": (scales.get("rate") or {}).get("min"),
            "rate_max": (scales.get("rate") or {}).get("max"),
            "energy_scale": transforms.get("energy", "log10"),
            "energy_min": (scales.get("energy") or {}).get("min"),
            "energy_max": (scales.get("energy") or {}).get("max"),
            "quality_scale": transforms.get("quality", "linear"),
            "quality_min": (scales.get("quality") or {}).get("min"),
            "quality_max": (scales.get("quality") or {}).get("max"),
        })
    else:
        audit.update({
            "rate_scale": normalization_reference.get("rate_scale", "log10"),
            "rate_min": normalization_reference.get("rate_min"),
            "rate_max": normalization_reference.get("rate_max"),
            "energy_scale": normalization_reference.get("energy_scale", "log10"),
            "energy_min": normalization_reference.get("energy_min"),
            "energy_max": normalization_reference.get("energy_max"),
            "quality_scale": normalization_reference.get("quality_scale", "linear"),
            "quality_min": normalization_reference.get("quality_min"),
            "quality_max": normalization_reference.get("quality_max"),
        })

    return audit


def _build_normalization_consistency_report(
    previous_receipt_path: Optional[str],
    current_audit: Dict[str, Any],
) -> Dict[str, Any]:
    if not previous_receipt_path:
        return {
            "enabled": False,
        }

    base: Dict[str, Any] = {
        "enabled": True,
        "previous_receipt_path": str(previous_receipt_path),
        "previous_receipt_loaded": False,
        "comparable": False,
        "warnings": [],
        "differences": compare_normalization_audits(
            current_audit,
            None,
        )["differences"],
    }

    try:
        previous_audit = load_previous_normalization_audit(previous_receipt_path)
    except NormalizationAuditLoadError as exc:
        base["warnings"] = [f"previous_receipt_load_failed: {exc}"]
        base["error"] = str(exc)
        return base

    base["previous_receipt_loaded"] = True
    comparison = compare_normalization_audits(
        current=current_audit,
        previous=previous_audit,
    )
    base.update(comparison)
    return base


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


def _make_report(
    args: argparse.Namespace,
    profile_name: str,
    weights: Dict[str, float],
    min_quality: float | None,
    near_quality_floor: float | None,
    decision: Dict[str, Any],
    csv_path: str,
    num_rows_loaded: int,
    system_state: Dict[str, Any],
    filter_report: Dict[str, Any],
    normalization_scope: str,
    normalization_reference_count: int,
    weight_source: str,
    context_policy: Dict[str, Any] | None,
) -> Dict[str, Any]:
    execution_plan = build_execution_plan(
        codec_name=decision["selected"]["codec"],
        config=decision["selected"]["config"],
        input_path=args.input,
        output_path=args.output,
        system_state=system_state,
        requested=args.generate_command or args.input is not None or args.execute,
    )

    calibration_report = getattr(args, "_calibration_report", {"enabled": False})
    selected_calibration = _find_selected_calibration(
        calibration_report=calibration_report,
        decision=decision,
    )

    system_features_report = getattr(
        args,
        "_system_features_report",
        {
            "enabled": False,
        },
    )

    if system_features_report.get("enabled", False):
        selected_time_ms = decision.get("selected", {}).get("time_ms")
        system_probe_efficiency = estimate_probe_efficiency(
            probe_overhead_ms=system_features_report
            .get("probe_overhead", {})
            .get("total_probe_ms", 0.0),
            reference_time_ms=selected_time_ms,
        )
    else:
        system_probe_efficiency = {
            "enabled": False,
            "reason": "system_features_disabled",
        }

    normalization_audit = _build_normalization_audit(
        normalization_report=getattr(
            args,
            "_normalization_report",
            {"mode": "runtime", "enabled": False},
        ),
        normalization_profile=getattr(args, "_normalization_profile", None),
        normalization_reference=decision.get("normalization_reference", {}),
        quality_metric=getattr(args, "quality_metric", None),
    )
    normalization_consistency = _build_normalization_consistency_report(
        previous_receipt_path=getattr(args, "previous_decision_receipt", None),
        current_audit=normalization_audit,
    )
    energy_provenance_summary = build_energy_provenance_summary(
        selected=decision.get("selected", {}),
        scored_candidate_pool=decision.get("scored_candidate_pool", []),
        unscored_candidate_pool=decision.get("unscored_candidate_pool", []),
    )
    energy_provenance_compatibility = (
        build_energy_provenance_compatibility_audit(
            selected=decision.get("selected", {}),
            scored_candidate_pool=decision.get("scored_candidate_pool", []),
            unscored_candidate_pool=decision.get("unscored_candidate_pool", []),
        )
    )

    return {
        "router_version": ROUTER_VERSION,
        "feature_level": dict(FEATURE_LEVEL),
        "domain_support": dict(DOMAIN_SUPPORT),
        "domain": args.domain,
        "profile": profile_name,
        "weight_source": weight_source,
        "context_policy": context_policy,
        "calibration": calibration_report,
        "calibration_bundle": getattr(
            args,
            "_calibration_bundle_report",
            {
                "enabled": False,
            },
        ),
        "calibration_bundle_validation": getattr(
            args,
            "_calibration_bundle_validation_report",
            {
                "enabled": False,
            },
        ),
        "energy_provenance": _build_energy_provenance_report(
            calibration_report=calibration_report,
            selected_calibration=selected_calibration,
        ),
        "energy_provenance_summary": energy_provenance_summary,
        "energy_provenance_compatibility": energy_provenance_compatibility,
        "codec_registry": getattr(
            args,
            "_codec_registry_report",
            {
                "enabled": False,
            },
        ),
        "router_config": getattr(
            args,
            "_router_config_report",
            {
                "enabled": False,
            },
        ),
        "run_manifest": getattr(
            args,
            "_run_manifest",
            {
                "enabled": False,
            },
        ),
        "resolved_args": {
            k: v for k, v in vars(args).items()
            if not k.startswith("_")
        },
        "selected_calibration": selected_calibration,
        "aggregate_by_config": args.aggregate_by_config,
        "num_rows_loaded_before_aggregation": num_rows_loaded,
        "codec_filtering": filter_report,
        "normalization": {
            "scope": normalization_scope,
            "num_reference_points": normalization_reference_count,
        },
        "normalization_audit": normalization_audit,
        "normalization_consistency": normalization_consistency,
        "normalization_profile": getattr(
            args,
            "_normalization_report",
            {
                "enabled": False,
                "mode": "runtime",
            },
        ),
        "quality_thresholds": getattr(
            args,
            "_quality_threshold_report",
            {
                "enabled": False,
            },
        ),
        "time_guard": getattr(
            args,
            "_time_guard_report",
            {
                "enabled": False,
                "max_time_ms": None,
            },
        ),
        "weights": weights,
        "constraints": {
            "min_quality": min_quality,
            "quality_floor": args.quality_floor,
            "quality_constraint_stat": args.quality_constraint_stat,
            "near_quality_floor": near_quality_floor,
            "allow_degraded_fallback": args.allow_degraded_fallback,
            "max_rate": args.max_rate,
            "max_energy": args.max_energy,
            "max_time_ms": args.max_time_ms,
        },
        "csv": str(Path(csv_path)),
        "decision": decision,
        "execution_plan": execution_plan,
        "system_state": system_state,
        "system_features": system_features_report,
        "system_probe_efficiency": system_probe_efficiency,
        "system_policy": getattr(
            args,
            "_system_policy_report",
            {
                "enabled": False,
            },
        ),
        "content_policy": getattr(
            args,
            "_content_policy_report",
            {
                "enabled": False,
                "mode": "report-only",
                "suggestion": None,
            },
        ),
        "content_classifier": getattr(
            args,
            "_content_classifier_report",
            {
                "enabled": False,
                "mode": "report-only",
                "prediction": None,
            },
        ),
        "content_filter": getattr(
            args,
            "_content_filter_report",
            {
                "enabled": False,
                "applied": False,
            },
        ),
        "system_policy_simulation": getattr(
            args,
            "_system_policy_simulation",
            {
                "enabled": False,
                "classes": {},
            },
        ),
        "system_penalty": getattr(
            args,
            "_system_penalty_report",
            {
                "enabled": False,
            },
        ),
        "system_penalty_weights": getattr(
            args,
            "_system_penalty_weights_report",
            {
                "source": None,
                "source_exists": False,
            },
        ),
    }


def _write_json_report(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _image_pixel_count(path: Optional[str]) -> Optional[int]:
    if not path:
        return None

    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
            return int(width * height)
    except Exception:
        return None


def _execute_plan(execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    command = execution_plan.get("command")

    if not execution_plan.get("requested", False):
        return {
            "requested": False,
            "executed": False,
            "success": False,
            "reason": "execution_not_requested",
        }

    if not execution_plan.get("can_execute", False):
        return {
            "requested": True,
            "executed": False,
            "success": False,
            "reason": "execution_plan_not_executable",
            "plan_reasons": execution_plan.get("reasons", []),
        }

    if not command:
        return {
            "requested": True,
            "executed": False,
            "success": False,
            "reason": "missing_command",
        }

    output_path = execution_plan.get("output")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    meter = CompositeEnergyMeter()

    def run_once():
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    result, energy = meter.measure_callable(run_once)

    return {
        "requested": True,
        "executed": True,
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": output_path,
        "execution_time_ms": energy.time_s * 1000.0,
        "local_cpu_energy_j": energy.cpu_j,
        "local_gpu_energy_j": energy.gpu_j,
        "local_energy_j": (
            energy.total_j if energy.energy_usable_for_total else None
        ),
        "energy_scope": energy.energy_scope,
        "energy_is_measured": energy.energy_is_measured,
        "energy_usable_for_total": energy.energy_usable_for_total,
        "energy_backend": energy.energy_backend,
        "energy_method": energy.energy_method,
        "energy_quality": energy.energy_quality,
        "energy_warnings": list(energy.warnings),
    }


def _build_feedback_row(
    *,
    report: Dict[str, Any],
    execution_result: Dict[str, Any],
    execution_validation: Dict[str, Any],
    report_path: Path,
) -> Dict[str, Any]:
    selected = report.get("decision", {}).get("selected", {})
    cost_decomp = selected.get("cost_decomposition", {}) or {}
    run_manifest = report.get("run_manifest", {}) or {}
    git_info = run_manifest.get("git", {}) or {}
    input_path = (
        report.get("execution_plan", {}).get("input")
        or report.get("resolved_args", {}).get("input")
    )

    output_bytes = execution_validation.get("output_size_bytes")
    input_pixels = _image_pixel_count(input_path)
    actual_rate = None
    if output_bytes is not None and input_pixels:
        actual_rate = (float(output_bytes) * 8.0) / float(input_pixels)

    error = execution_result.get("reason")
    if not execution_result.get("success", False):
        stderr = str(execution_result.get("stderr") or "").strip()
        if stderr:
            error = stderr

    return {
        "router_version": report.get("router_version"),
        "feature_level": report.get("feature_level"),
        "domain": report.get("domain"),
        "input_path": input_path,
        "input_id": Path(input_path).stem if input_path else None,
        "selected_codec": selected.get("codec"),
        "selected_config": selected.get("config"),
        "decision_mode": report.get("decision", {}).get("decision_mode"),
        "profile": report.get("profile"),
        "predicted_rate": selected.get("rate"),
        "predicted_quality": selected.get("quality"),
        "predicted_energy": selected.get("energy"),
        "predicted_time_ms": selected.get("time_ms"),
        "predicted_cost": selected.get("cost"),
        "term_R": cost_decomp.get("term_R"),
        "term_E": cost_decomp.get("term_E"),
        "term_D": cost_decomp.get("term_D"),
        "actual_output_bytes": output_bytes,
        "actual_rate": actual_rate,
        "actual_time_ms": execution_result.get("execution_time_ms"),
        "local_cpu_energy_j": execution_result.get("local_cpu_energy_j"),
        "local_gpu_energy_j": execution_result.get("local_gpu_energy_j"),
        "local_energy_j": execution_result.get("local_energy_j"),
        "energy_scope": execution_result.get("energy_scope"),
        "energy_is_measured": execution_result.get("energy_is_measured"),
        "energy_usable_for_total": execution_result.get("energy_usable_for_total"),
        "energy_backend": execution_result.get("energy_backend"),
        "energy_method": execution_result.get("energy_method"),
        "energy_quality": execution_result.get("energy_quality"),
        "execution_requested": execution_result.get("requested"),
        "execution_success": execution_result.get("success"),
        "output_exists": execution_validation.get("output_exists"),
        "output_nonempty": execution_validation.get("output_nonempty"),
        "error": error,
        "report_path": str(report_path),
        "git_commit_short": git_info.get("commit_short"),
    }


def _find_selected_calibration(calibration_report: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    selected = decision.get("selected", {})
    selected_codec = str(selected.get("codec"))
    selected_config = str(selected.get("config"))

    if not calibration_report.get("enabled", False):
        return {
            "enabled": False,
        }

    for item in calibration_report.get("applied", []):
        if (
            str(item.get("codec")) == selected_codec
            and str(item.get("config")) == selected_config
        ):
            out = dict(item)
            out["enabled"] = True
            return out

    return {
        "enabled": False,
        "reason": "selected_point_not_calibrated",
        "codec": selected_codec,
        "config": selected_config,
    }


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

    system_policy_report = build_system_policy(
        base_weights=weights,
        system_features_report=getattr(
            args,
            "_system_features_report",
            {
                "enabled": False,
            },
        ),
        enabled=args.system_policy,
        mode=args.system_policy_mode,
    )

    args._system_policy_report = system_policy_report

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

    args._content_policy_report = content_policy_report

    content_classifier_report = _build_content_classifier_router_report(args)
    args._content_classifier_report = content_classifier_report

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

    args._preferred_candidate_source = preferred_source

    system_penalty_weights_report = load_system_penalty_weights(
        args.system_penalty_weights_file
    )

    args._system_penalty_weights_report = system_penalty_weights_report

    system_penalty_context = build_system_penalty_context(
        enabled=args.system_penalty,
        mode=args.system_penalty_mode,
        lambda_sys=args.system_penalty_lambda,
        system_features_report=getattr(
            args,
            "_system_features_report",
            {
                "enabled": False,
            },
        ),
        latency_constrained=args.max_time_ms is not None,
        execution_requested=bool(args.execute),
        penalty_weights=system_penalty_weights_report["weights"],
        penalty_weights_source=system_penalty_weights_report["source"],
    )

    args._system_penalty_report = system_penalty_context

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

    args._time_guard_report = time_guard_report

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
        normalization_profile=getattr(args, "_normalization_profile", None),
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

    return _make_report(
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

    parser = argparse.ArgumentParser(
        description="Prototype R-D-E router for adaptive codec selection."
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Router configuration JSON file. Expanded before normal argument parsing.",
    )

    parser.add_argument("--csv", required=True, help="Path del CSV con i punti R-D-E.")

    parser.add_argument(
        "--calibration-bundle-manifest",
        default=None,
        help=(
            "Explicit calibration bundle manifest. When provided, the router "
            "validates the manifest and uses its calibrated CSV as the R-D-E input."
        ),
    )

    parser.add_argument(
        "--calibration-bundle-validation",
        default=None,
        help=(
            "Optional explicit shadow decision validation report. When provided, "
            "it must be accepted before the calibration bundle can be used."
        ),
    )

    parser.add_argument(
        "--calibration-file",
        default=None,
        help="File JSON di calibrazione locale da applicare ai punti R-D-E.",
    )

    parser.add_argument(
        "--normalization-file",
        default=None,
        help="File JSON con scale di normalizzazione precomputate.",
    )

    parser.add_argument(
        "--normalization-mode",
        default="auto",
        choices=["auto", "runtime", "global", "dataset", "local"],
        help=(
            "Politica di normalizzazione: auto, runtime, global, dataset, local. "
            "runtime usa la normalizzazione calcolata al volo; global/dataset/local "
            "richiedono --normalization-file."
        ),
    )

    parser.add_argument(
        "--previous-decision-receipt",
        default=None,
        help=(
            "Optional explicit previous decision receipt/router report used only "
            "to audit normalization comparability in the output report."
        ),
    )

    parser.add_argument(
        "--input",
        default=None,
        help="File di input da usare per generare un piano di esecuzione.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="File di output desiderato per il piano di esecuzione.",
    )

    parser.add_argument(
        "--generate-command",
        action="store_true",
        help="Genera un execution plan per il codec selezionato.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Esegue direttamente il comando generato se il piano è eseguibile.",
    )

    parser.add_argument(
        "--domain",
        default="image",
        choices=["image", "audio", "video"],
        help="Dominio multimediale.",
    )

    parser.add_argument(
        "--profile",
        default="balanced",
        choices=available_profiles(),
        help="Profilo operativo da usare se --all-profiles non è attivo.",
    )

    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Esegue il router su tutti i profili disponibili e genera un summary CSV.",
    )

    parser.add_argument(
        "--auto-weights",
        action="store_true",
        help="Calcola automaticamente i pesi R-D-E dal contesto operativo.",
    )

    parser.add_argument(
        "--power-mode",
        choices=["ac", "battery", "unknown"],
        default="ac",
        help="Modalità alimentazione usata dalla policy contestuale.",
    )

    parser.add_argument(
        "--battery-percent",
        type=float,
        default=None,
        help="Percentuale batteria usata dalla policy contestuale.",
    )

    parser.add_argument(
        "--thermal-state",
        choices=["nominal", "warm", "hot", "critical"],
        default="nominal",
        help="Stato termico usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--network-profile",
        choices=["normal", "limited", "very-limited"],
        default="normal",
        help="Profilo rete usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--quality-target",
        choices=["preview", "normal", "high", "very-high"],
        default="normal",
        help="Target qualità usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--quality-thresholds-file",
        default="configs/quality_thresholds.json",
        help="File JSON con soglie qualità domain-specific.",
    )

    parser.add_argument(
        "--codec-registry-file",
        default=None,
        help="External codec/backend registry JSON file.",
    )

    parser.add_argument(
        "--system-load",
        choices=["normal", "high", "very-high"],
        default="normal",
        help="Carico sistema usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--aggregate-by-config",
        action="store_true",
        help="Aggrega le righe per codec+config usando la media di rate, qualità ed energia.",
    )

    parser.add_argument(
        "--normalization-scope",
        choices=["global", "filtered"],
        default="global",
        help=(
            "global = normalizza sui punti prima dei filtri codec; "
            "filtered = normalizza solo sui punti rimasti dopo i filtri."
        ),
    )

    parser.add_argument(
        "--available-codecs",
        default=None,
        help="Lista separata da virgole dei codec disponibili. Esempio: JPEG,JXL,HEVC",
    )

    parser.add_argument(
        "--exclude-codecs",
        default=None,
        help="Lista separata da virgole dei codec da escludere. Esempio: DCAE,JPEG_AI",
    )

    parser.add_argument(
        "--exclude-neural",
        action="store_true",
        help="Esclude codec neurali o basati su modelli appresi.",
    )

    parser.add_argument(
        "--system-aware",
        action="store_true",
        help="Usa il profilo del sistema reale per filtrare automaticamente il pool ammissibile.",
    )

    parser.add_argument(
        "--system-features",
        action="store_true",
        help="Extract cheap system-aware features and include them in the report.",
    )

    parser.add_argument(
        "--system-probe-level",
        default="basic",
        choices=["basic", "gpu", "full"],
        help="System feature probe level: basic, gpu, or full.",
    )

    parser.add_argument(
        "--system-feature-cache-ttl-s",
        type=float,
        default=5.0,
        help="Cache TTL in seconds for system feature probes.",
    )

    parser.add_argument(
        "--system-feature-cpu-interval-s",
        type=float,
        default=0.0,
        help="psutil CPU sampling interval for system feature extraction.",
    )

    parser.add_argument(
        "--system-policy",
        action="store_true",
        help="Build a system-aware policy from extracted system features.",
    )

    parser.add_argument(
        "--system-policy-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="report-only records suggested changes; apply uses adjusted weights.",
    )

    parser.add_argument(
        "--system-policy-simulate",
        default=None,
        help=(
            "Comma-separated simulated system classes, e.g. "
            "'battery=critical,cpu=busy,memory=constrained'. "
            "Overrides measured classes for system-policy evaluation."
        ),
    )

    parser.add_argument(
        "--system-penalty",
        action="store_true",
        help="Compute a codec/backend system penalty from resource profiles.",
    )

    parser.add_argument(
        "--system-penalty-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="report-only records J_total; apply ranks by J_total and applies hard exclusions.",
    )

    parser.add_argument(
        "--system-penalty-lambda",
        type=float,
        default=0.25,
        help="Weight of the system penalty term in J_total.",
    )

    parser.add_argument(
        "--system-penalty-weights-file",
        default=None,
        help="Optional JSON file with configurable system penalty coefficients.",
    )

    parser.add_argument(
        "--content-policy",
        action="store_true",
        help="Enable source-aware content policy reporting.",
    )

    parser.add_argument(
        "--content-policy-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="Content policy mode: report-only or safe apply integration.",
    )

    parser.add_argument(
        "--content-policy-rules-file",
        default=None,
        help="CSV rules file produced by content metadata policy evaluation.",
    )

    parser.add_argument(
        "--content-policy-key",
        default="dataset",
        help="Metadata key used by the content policy rules, e.g. dataset/source.",
    )

    parser.add_argument(
        "--content-source",
        default=None,
        help="Optional homogeneous content source/dataset label, e.g. tecnick, kodak, clic2020.",
    )

    parser.add_argument(
        "--content-source-filter",
        action="store_true",
        help="Filter the benchmark candidate pool using the provided content source/context.",
    )

    parser.add_argument(
        "--content-filter-column",
        default=None,
        help="CSV/raw column used for source-conditioned filtering. Defaults to --content-policy-key.",
    )

    parser.add_argument(
        "--content-filter-value",
        default=None,
        help="Value used for source-conditioned filtering. Defaults to --content-source.",
    )

    parser.add_argument(
        "--content-classifier",
        action="store_true",
        help="Enable source-agnostic content classifier reporting.",
    )

    parser.add_argument(
        "--content-classifier-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="Content classifier mode: report-only or safe apply integration.",
    )

    parser.add_argument(
        "--content-classifier-config",
        default=None,
        help="JSON config for the source-agnostic content classifier.",
    )

    parser.add_argument(
        "--content-classifier-image",
        default=None,
        help="Optional image path used to extract source-agnostic content classifier features.",
    )

    parser.add_argument(
        "--content-classifier-width",
        type=int,
        default=None,
        help="Optional image width used when no content-classifier image path is provided.",
    )

    parser.add_argument(
        "--content-classifier-height",
        type=int,
        default=None,
        help="Optional image height used when no content-classifier image path is provided.",
    )

    parser.add_argument(
        "--capability-aware",
        action="store_true",
        help="Filtra i codec usando il registry dei requisiti hardware/software.",
    )

    parser.add_argument(
        "--strict-executables",
        action="store_true",
        help="Se attivo, esclude i codec i cui eseguibili richiesti non sono nel PATH.",
    )

    parser.add_argument(
        "--simulate-no-cuda",
        action="store_true",
        help="Debug: simula assenza di CUDA per testare il filtro system-aware.",
    )

    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Attiva guardia qualità robusta: usa p10 se non specificato e floor minimo 60.",
    )

    parser.add_argument(
        "--quality-constraint-stat",
        choices=["mean", "p25", "p10", "min"],
        default=None,
        help="Statistica usata come vincolo duro di qualità.",
    )

    parser.add_argument(
        "--quality-floor",
        type=float,
        default=None,
        help="Soglia minima assoluta di qualità accettabile.",
    )

    parser.add_argument(
        "--near-quality-floor",
        type=float,
        default=None,
        help="Soglia qualità quasi-usabile per fallback degradato.",
    )

    parser.add_argument(
        "--allow-degraded-fallback",
        action="store_true",
        help="Permette fallback degradato se nessun punto supera la soglia sicura.",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="Qualità minima ammissibile. Se assente, usa quella del profilo/policy.",
    )

    parser.add_argument("--max-rate", type=float, default=None, help="Rate massimo ammissibile.")
    parser.add_argument("--max-energy", type=float, default=None, help="Energia massima ammissibile.")
    parser.add_argument("--max-time-ms", type=float, default=None, help="Tempo massimo ammissibile in millisecondi.")
    parser.add_argument(
        "--strict-time",
        action="store_true",
        help=(
            "Se usato con --max-time-ms, richiede che tutti i punti candidati "
            "abbiano time_ms disponibile."
        ),
    )

    parser.add_argument("--wE", "--w-e", dest="wE", type=float, default=None, help="Peso energia custom.")
    parser.add_argument("--wR", "--w-r", dest="wR", type=float, default=None, help="Peso rate custom.")
    parser.add_argument("--wD", "--w-d", dest="wD", type=float, default=None, help="Peso distorsione custom.")

    parser.add_argument("--codec-col", default=None)
    parser.add_argument("--config-col", default=None)
    parser.add_argument("--rate-col", default=None)
    parser.add_argument("--quality-col", default=None)
    parser.add_argument("--quality-metric", default=None)
    parser.add_argument("--energy-col", default=None)
    parser.add_argument("--time-col", default=None)

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Numero di configurazioni migliori da salvare nel report JSON.",
    )

    parser.add_argument(
        "--export-topk",
        action="store_true",
        help="Esporta anche i top-k candidati in CSV.",
    )

    parser.add_argument(
        "--feedback-out",
        default="results/routing_context/online_feedback.csv",
        help=(
            "Append-only CSV for observed execution feedback. "
            "Written only when --execute is used."
        ),
    )

    parser.add_argument(
        "--out",
        default="results/routing/router_decision_report.json",
        help="Path del report JSON quando si usa un solo profilo.",
    )

    parser.add_argument(
        "--out-dir",
        default="results/routing",
        help="Cartella di output quando si usa --all-profiles.",
    )

    parser.add_argument(
        "--summary-out",
        default=None,
        help="Path del summary CSV. Se assente, usa results/routing/router_summary.csv.",
    )

    args = parser.parse_args(argv)
    args._router_config_report = router_config_report
    if args.system_features:
        args._system_features_report = build_system_features(
            probe_level=args.system_probe_level,
            cache_ttl_s=args.system_feature_cache_ttl_s,
            cpu_interval_s=args.system_feature_cpu_interval_s,
        )
    else:
        args._system_features_report = {
            "enabled": False,
        }

    simulated_system_classes = parse_system_policy_simulation(
        args.system_policy_simulate
    )

    args._system_policy_simulation = {
        "enabled": bool(simulated_system_classes),
        "classes": simulated_system_classes,
    }

    if simulated_system_classes:
        args._system_features_report = apply_system_policy_simulation(
            system_features_report=args._system_features_report,
            simulated_classes=simulated_system_classes,
        )

    args._run_manifest = build_run_manifest(
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

    args._codec_registry_report = registry_report

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
    args._quality_threshold_report = quality_threshold_report

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

    args._calibration_bundle_report = calibration_bundle_report
    args._calibration_bundle_validation_report = calibration_bundle_validation_report

    points = load_rde_points(
        csv_path=effective_csv_path,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        time_col=args.time_col,
    )

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

    args._content_filter_report = content_filter_report

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

    args._calibration_report = calibration_report

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

    args._normalization_profile = normalization_profile
    args._normalization_report = normalization_report

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
        args._normalization_report = normalization_report

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

        if args.execute:
            execution_result = _execute_plan(report.get("execution_plan", {}))
            report["execution_result"] = execution_result
            report["execution_validation"] = validate_execution_output(
                execution_plan=report.get("execution_plan", {}),
                execution_result=execution_result,
            )
        else:
            report["execution_result"] = {
                "requested": False,
                "executed": False,
            }
            report["execution_validation"] = {
                "enabled": False,
                "reason": "execution_not_requested",
            }

        report["decision_receipt"] = build_decision_receipt(report)
        _write_json_report(report, out_path)

        if args.execute and args.feedback_out:
            feedback_row = _build_feedback_row(
                report=report,
                execution_result=report["execution_result"],
                execution_validation=report["execution_validation"],
                report_path=out_path,
            )
            feedback_report = {
                "enabled": True,
                "path": args.feedback_out,
                "written": False,
            }

            try:
                append_feedback_row(args.feedback_out, feedback_row)
                feedback_report["written"] = True
            except Exception as exc:
                feedback_report["error"] = str(exc)

            report["online_feedback"] = feedback_report
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
