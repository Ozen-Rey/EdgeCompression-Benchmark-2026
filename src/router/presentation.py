"""Human-readable CLI presentation of router decision reports.

Extracted from rde_router.py. Functions here only consume fully built
router report dicts (and, for the --all-profiles header, the parsed
argparse namespace) and print to stdout: no decision, ranking,
scoring, normalization, I/O, or report assembly happens in this module.
"""

from pathlib import Path
from typing import Any, Dict, Optional


def print_single_decision(report: Dict[str, Any], json_path: Path) -> None:
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


def print_all_profiles_header(
    args: Any,
    *,
    num_rows_loaded: int,
    num_points_after_filter: int,
    normalization_scope_label: str,
    num_normalization_points: int,
    filter_report: Dict[str, Any],
) -> None:
    print("\n=== R-D-E Router: all profiles ===")
    print(f"Loaded rows: {num_rows_loaded}")
    print(f"Candidate points after aggregation/filtering: {num_points_after_filter}")
    print(
        f"Normalization: {normalization_scope_label} "
        f"({num_normalization_points} reference points)"
    )
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


def print_all_profiles_selection(profile_name: str, report: Dict[str, Any]) -> None:
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


def print_all_profiles_footer(
    *,
    out_dir: Path,
    summary_path: Path,
    topk_path: Optional[Path] = None,
) -> None:
    print()
    print(f"JSON reports written to: {out_dir}")
    print(f"Summary written to:      {summary_path}")
    if topk_path is not None:
        print(f"Top-k written to:        {topk_path}")
