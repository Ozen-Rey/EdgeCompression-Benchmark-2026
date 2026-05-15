"""Pipeline helpers and high-level orchestration for the router.

Hosts:

- side-effect-free helpers extracted from rde_router.py
  (build_weights_for_profile, annotate_points_with_calibration_provenance,
  summary_row_from_report, topk_rows_from_report);
- run_router(...) — orchestrates a single CLI invocation from a parsed
  argparse namespace and a freshly constructed RouterContext. It owns
  everything that used to live in main() after the parser step:
  feature probes, calibration bundle handling, CSV loading + row
  diagnostics, codec filtering, normalization mode resolution, the
  per-profile loop, and the single-profile branch.

argparse construction itself still lives in rde_router.main(); the
``--help`` entry points and the ``__main__`` exception wrapper are
unchanged. _run_profile and its local helpers also stay in rde_router
for now and are imported lazily here to avoid a circular import.
"""

import argparse
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.router.adaptation.context_policy import compute_context_policy
from src.router.adaptation.system_features import build_system_features
from src.router.adaptation.system_policy import (
    apply_system_policy_simulation,
    parse_system_policy_simulation,
)
from src.router.adaptation.system_probe import probe_system
from src.router.calibration.calibration_apply import apply_local_calibration
from src.router.calibration.calibration_bundle import (
    validate_calibration_bundle_manifest,
    validate_calibration_bundle_validation,
)
from src.router.codecs.codec_capabilities import (
    filter_points_by_capabilities,
    is_neural_codec,
    load_external_codec_registry,
)
from src.router.codecs.external_codec_registry import load_external_codec_points
from src.router.context import RouterContext
from src.router.core.normalization_profile import load_normalization_profile
from src.router.core.profiles import available_profiles, get_profile
from src.router.core.quality_thresholds import resolve_quality_floor
from src.router.core.rde_database import (
    RDEPoint,
    aggregate_points_by_config,
    filter_points_by_raw_column,
    load_rde_points_with_diagnostics,
)
from src.router.execution import apply_execution_result, write_feedback_report
from src.router.observability.decision_receipt import build_decision_receipt
from src.router.observability.run_manifest import build_run_manifest
from src.router.outputs import (
    safe_profile_filename,
    write_json_report,
    write_summary_csv,
    write_topk_csv,
)
from src.router.presentation import (
    print_all_profiles_footer,
    print_all_profiles_header,
    print_all_profiles_selection,
    print_single_decision,
)


def _normalize_weights(w_e: float, w_r: float, w_d: float) -> Dict[str, float]:
    total = w_e + w_r + w_d

    if total <= 0:
        raise ValueError("The sum of the weights must be positive.")

    return {
        "w_E": w_e / total,
        "w_R": w_r / total,
        "w_D": w_d / total,
    }


def normalize_token(text: str) -> str:
    """Lowercase + strip accents/punctuation; used to canonicalize codec names."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text.lower() if ch.isalnum())


def parse_codec_list(value: Optional[str]) -> Optional[set[str]]:
    """Parse ``--available-codecs``/``--exclude-codecs`` CLI values.

    Returns ``None`` for empty/missing values, otherwise a set of
    canonical codec tokens normalized via :func:`normalize_token`.
    """
    if value is None or value.strip() == "":
        return None

    return {
        normalize_token(item)
        for item in value.split(",")
        if item.strip()
    }


def filter_points_by_codec_availability(
    points: List[RDEPoint],
    available_codecs: Optional[set[str]],
    exclude_codecs: Optional[set[str]],
    exclude_neural: bool,
) -> Tuple[List[RDEPoint], Dict[str, Any]]:
    """Apply CLI codec-availability filters and return survivors + report."""
    filtered: List[RDEPoint] = []

    excluded_by_available = 0
    excluded_by_exclude_list = 0
    excluded_by_neural = 0

    for p in points:
        codec_norm = normalize_token(p.codec)

        if available_codecs is not None and codec_norm not in available_codecs:
            excluded_by_available += 1
            continue

        if exclude_codecs is not None and codec_norm in exclude_codecs:
            excluded_by_exclude_list += 1
            continue

        if exclude_neural and is_neural_codec(p.codec):
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


def run_router(
    *,
    args: argparse.Namespace,
    router_context: RouterContext,
    original_argv: List[str],
    expanded_argv: List[str],
    router_config_report: Dict[str, Any],
) -> None:
    """Run the router pipeline end-to-end from a parsed CLI namespace.

    Imports ``_run_profile`` and ``_apply_system_aware_policy`` from
    rde_router lazily to avoid a circular module-load import. The codec
    availability/normalization helpers (:func:`parse_codec_list`,
    :func:`filter_points_by_codec_availability`) live in this module so
    they no longer participate in the lazy cycle. Everything else
    (system probes, calibration bundle handling, CSV loading,
    normalization mode resolution, per-profile loop, single-profile
    branch, execution result printing) is orchestrated here.
    """
    from src.router.rde_router import (
        _apply_system_aware_policy,
        _run_profile,
    )

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
        raise ValueError("--execute is only supported in single-profile mode, not with --all-profiles.")

    if args.execute and args.input is None:
        raise ValueError("--execute requires --input.")

    if args.auto_weights and args.all_profiles:
        raise ValueError(
            "--auto-weights produces a single contextual profile; "
            "do not combine it with --all-profiles."
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

    points, csv_row_diagnostics = load_rde_points_with_diagnostics(
        csv_path=effective_csv_path,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        time_col=args.time_col,
    )
    router_context.csv_row_diagnostics = csv_row_diagnostics

    if csv_row_diagnostics["dropped_rows"] > 0:
        reasons_summary = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(csv_row_diagnostics["reasons"].items())
        )
        print(
            f"Warning: dropped {csv_row_diagnostics['dropped_rows']} CSV row(s) "
            f"from {effective_csv_path} ({reasons_summary}). "
            "See report.csv_row_diagnostics for details."
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
        annotate_points_with_calibration_provenance(
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
                "--normalization-mode runtime must not be combined with --normalization-file."
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

    available_codecs = parse_codec_list(args.available_codecs)
    exclude_codecs = parse_codec_list(args.exclude_codecs)

    points, filter_report = filter_points_by_codec_availability(
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

        print_all_profiles_header(
            args,
            num_rows_loaded=num_rows_loaded,
            num_points_after_filter=len(points),
            normalization_scope_label=normalization_scope_label,
            num_normalization_points=len(normalization_points),
            filter_report=filter_report,
        )

        for profile_name in available_profiles():
            report = _run_profile(
                args=args,
                router_context=router_context,
                profile_name=profile_name,
                points=points,
                normalization_points=normalization_points,
                normalization_scope=normalization_scope_label,
                csv_path=effective_csv_path,
                num_rows_loaded=num_rows_loaded,
                system_state=system_state,
                filter_report=filter_report,
            )

            safe_name = safe_profile_filename(profile_name)
            json_path = out_dir / f"router_decision_report_{safe_name}.json"
            report["decision_receipt"] = build_decision_receipt(report)
            write_json_report(report, json_path)

            summary_rows.append(summary_row_from_report(report))

            if args.export_topk:
                topk_rows.extend(topk_rows_from_report(report))

            print_all_profiles_selection(profile_name, report)

        summary_path = (
            Path(args.summary_out)
            if args.summary_out is not None
            else out_dir / "router_summary.csv"
        )

        write_summary_csv(summary_rows, summary_path)

        topk_path: Optional[Path] = None
        if args.export_topk:
            topk_path = out_dir / "router_topk.csv"
            write_topk_csv(topk_rows, topk_path)

        print_all_profiles_footer(
            out_dir=out_dir,
            summary_path=summary_path,
            topk_path=topk_path,
        )

    else:
        report = _run_profile(
            args=args,
            router_context=router_context,
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
        write_json_report(report, out_path)

        if args.execute and args.feedback_out:
            write_feedback_report(
                report,
                feedback_out=args.feedback_out,
                report_path=out_path,
            )
            write_json_report(report, out_path)

        if args.export_topk:
            topk_path = out_path.with_name(out_path.stem + "_topk.csv")
            write_topk_csv(topk_rows_from_report(report), topk_path)

        print_single_decision(report, out_path)

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
