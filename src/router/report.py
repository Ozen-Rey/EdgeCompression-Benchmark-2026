from pathlib import Path
from typing import Any, Dict, Optional

try:
    from src.router.version import DOMAIN_SUPPORT, FEATURE_LEVEL, ROUTER_VERSION
    from .codecs.codec_capabilities import build_execution_plan
    from .context import RouterContext
    from .energy_provenance import build_energy_provenance_summary
    from .energy_provenance_compatibility import (
        build_energy_provenance_compatibility_audit,
    )
    from .energy_tier_policy import build_energy_tier_policy_shadow
    from .normalization_consistency import (
        NormalizationAuditLoadError,
        compare_normalization_audits,
        load_previous_normalization_audit,
    )
    from .system_features import estimate_probe_efficiency
except ImportError:  # pragma: no cover - direct script fallback
    from version import DOMAIN_SUPPORT, FEATURE_LEVEL, ROUTER_VERSION
    from codec_capabilities import build_execution_plan
    from context import RouterContext
    from energy_provenance import build_energy_provenance_summary
    from energy_provenance_compatibility import (
        build_energy_provenance_compatibility_audit,
    )
    from energy_tier_policy import build_energy_tier_policy_shadow
    from normalization_consistency import (
        NormalizationAuditLoadError,
        compare_normalization_audits,
        load_previous_normalization_audit,
    )
    from system_features import estimate_probe_efficiency


def _router_context(args: Any) -> Optional[RouterContext]:
    context = getattr(args, "_router_context", None)
    if isinstance(context, RouterContext):
        return context
    return None


def _context_or_args(
    args: Any,
    context: Optional[RouterContext],
    context_field: str,
    args_field: str,
    default: Any,
) -> Any:
    if context is not None:
        value = getattr(context, context_field)
        if value is not None:
            return value
    return getattr(args, args_field, default)


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


def _find_selected_calibration(
    calibration_report: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
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


def build_router_report(
    args: Any,
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
    router_context = _router_context(args)

    execution_plan = build_execution_plan(
        codec_name=decision["selected"]["codec"],
        config=decision["selected"]["config"],
        input_path=args.input,
        output_path=args.output,
        system_state=system_state,
        requested=args.generate_command or args.input is not None or args.execute,
    )

    calibration_report = _context_or_args(
        args,
        router_context,
        "calibration_report",
        "_calibration_report",
        {"enabled": False},
    )
    selected_calibration = _find_selected_calibration(
        calibration_report=calibration_report,
        decision=decision,
    )

    system_features_report = _context_or_args(
        args,
        router_context,
        "system_features_report",
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
        normalization_report=_context_or_args(
            args,
            router_context,
            "normalization_report",
            "_normalization_report",
            {"mode": "runtime", "enabled": False},
        ),
        normalization_profile=_context_or_args(
            args,
            router_context,
            "normalization_profile",
            "_normalization_profile",
            None,
        ),
        normalization_reference=decision.get("normalization_reference", {}),
        quality_metric=getattr(args, "quality_metric", None),
    )
    normalization_consistency = _context_or_args(
        args,
        router_context,
        "normalization_consistency_report",
        "_normalization_consistency_report",
        None,
    )
    if normalization_consistency is None:
        normalization_consistency = _build_normalization_consistency_report(
            previous_receipt_path=getattr(args, "previous_decision_receipt", None),
            current_audit=normalization_audit,
        )
    energy_provenance_report = _context_or_args(
        args,
        router_context,
        "energy_provenance_report",
        "_energy_provenance_report",
        None,
    )
    if energy_provenance_report is None:
        energy_provenance_report = _build_energy_provenance_report(
            calibration_report=calibration_report,
            selected_calibration=selected_calibration,
        )
    energy_provenance_summary = _context_or_args(
        args,
        router_context,
        "energy_provenance_summary",
        "_energy_provenance_summary",
        None,
    )
    if energy_provenance_summary is None:
        energy_provenance_summary = build_energy_provenance_summary(
            selected=decision.get("selected", {}),
            scored_candidate_pool=decision.get("scored_candidate_pool", []),
            unscored_candidate_pool=decision.get("unscored_candidate_pool", []),
        )
    energy_provenance_compatibility = _context_or_args(
        args,
        router_context,
        "energy_provenance_compatibility",
        "_energy_provenance_compatibility",
        None,
    )
    if energy_provenance_compatibility is None:
        energy_provenance_compatibility = (
            build_energy_provenance_compatibility_audit(
                selected=decision.get("selected", {}),
                scored_candidate_pool=decision.get("scored_candidate_pool", []),
                unscored_candidate_pool=decision.get("unscored_candidate_pool", []),
            )
        )
    energy_tier_policy = _context_or_args(
        args,
        router_context,
        "energy_tier_policy",
        "_energy_tier_policy",
        None,
    )
    if energy_tier_policy is None:
        energy_tier_policy = build_energy_tier_policy_shadow(
            selected=decision.get("selected", {}),
            scored_candidate_pool=decision.get("scored_candidate_pool", []),
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
        "calibration_bundle": _context_or_args(
            args,
            router_context,
            "calibration_bundle_report",
            "_calibration_bundle_report",
            {
                "enabled": False,
            },
        ),
        "calibration_bundle_validation": _context_or_args(
            args,
            router_context,
            "calibration_bundle_validation_report",
            "_calibration_bundle_validation_report",
            {
                "enabled": False,
            },
        ),
        "energy_provenance": energy_provenance_report,
        "energy_provenance_summary": energy_provenance_summary,
        "energy_provenance_compatibility": energy_provenance_compatibility,
        "energy_tier_policy": energy_tier_policy,
        "codec_registry": _context_or_args(
            args,
            router_context,
            "codec_registry_report",
            "_codec_registry_report",
            {
                "enabled": False,
            },
        ),
        "external_codecs": _context_or_args(
            args,
            router_context,
            "external_codecs_report",
            "_external_codecs_report",
            {
                "enabled": False,
            },
        ),
        "router_config": _context_or_args(
            args,
            router_context,
            "router_config_report",
            "_router_config_report",
            {
                "enabled": False,
            },
        ),
        "run_manifest": _context_or_args(
            args,
            router_context,
            "run_manifest",
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
        "normalization_profile": _context_or_args(
            args,
            router_context,
            "normalization_report",
            "_normalization_report",
            {
                "enabled": False,
                "mode": "runtime",
            },
        ),
        "quality_thresholds": _context_or_args(
            args,
            router_context,
            "quality_threshold_report",
            "_quality_threshold_report",
            {
                "enabled": False,
            },
        ),
        "time_guard": _context_or_args(
            args,
            router_context,
            "time_guard_report",
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
        "system_policy": _context_or_args(
            args,
            router_context,
            "system_policy_report",
            "_system_policy_report",
            {
                "enabled": False,
            },
        ),
        "content_policy": _context_or_args(
            args,
            router_context,
            "content_policy_report",
            "_content_policy_report",
            {
                "enabled": False,
                "mode": "report-only",
                "suggestion": None,
            },
        ),
        "content_classifier": _context_or_args(
            args,
            router_context,
            "content_classifier_report",
            "_content_classifier_report",
            {
                "enabled": False,
                "mode": "report-only",
                "prediction": None,
            },
        ),
        "content_filter": _context_or_args(
            args,
            router_context,
            "content_filter_report",
            "_content_filter_report",
            {
                "enabled": False,
                "applied": False,
            },
        ),
        "system_policy_simulation": _context_or_args(
            args,
            router_context,
            "system_policy_simulation",
            "_system_policy_simulation",
            {
                "enabled": False,
                "classes": {},
            },
        ),
        "system_penalty": _context_or_args(
            args,
            router_context,
            "system_penalty_report",
            "_system_penalty_report",
            {
                "enabled": False,
            },
        ),
        "system_penalty_weights": _context_or_args(
            args,
            router_context,
            "system_penalty_weights_report",
            "_system_penalty_weights_report",
            {
                "source": None,
                "source_exists": False,
            },
        ),
    }
