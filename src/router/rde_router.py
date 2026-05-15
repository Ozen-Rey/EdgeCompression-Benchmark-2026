import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.router.codecs.codec_capabilities import is_neural_codec
from src.router.cli import build_router_arg_parser
from src.router.context import RouterContext
from src.router.adaptation.content_policy import (
    build_content_policy_report,
    get_content_policy_preferred_candidate,
)
from src.router.adaptation.content_classifier_model import (
    build_metadata_no_source_features,
    extract_metadata_features_from_image,
    load_content_classifier_config,
    load_training_rows_from_config,
    predict_content_classifier,
)
from src.router.core.rde_database import RDEPoint, select_best_rde
from src.router.pipeline import build_weights_for_profile, run_router
from src.router.profile_runner import apply_preferred_candidate_override
from src.router.report import build_router_report
from src.router.core.router_config import expand_argv_with_config
from src.router.adaptation.system_penalty import (
    build_system_penalty_context,
    load_system_penalty_weights,
    make_system_penalty_fn,
)
from src.router.adaptation.system_policy import build_system_policy


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


def _run_profile(
    args: argparse.Namespace,
    router_context: RouterContext,
    profile_name: str,
    points: List[RDEPoint],
    normalization_points: List[RDEPoint],
    normalization_scope: str,
    csv_path: str,
    num_rows_loaded: int,
    system_state: Dict[str, Any],
    filter_report: Dict[str, Any],
) -> Dict[str, Any]:
    weights, min_quality, weight_source, context_policy = build_weights_for_profile(
        args,
        profile_name,
    )

    system_features_report = router_context.system_features_report or {"enabled": False}

    system_policy_report = build_system_policy(
        base_weights=weights,
        system_features_report=system_features_report,
        enabled=args.system_policy,
        mode=args.system_policy_mode,
    )

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

    router_context.content_policy_report = content_policy_report

    content_classifier_report = _build_content_classifier_router_report(args)
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

    router_context.preferred_candidate_source = preferred_source

    system_penalty_weights_report = load_system_penalty_weights(
        args.system_penalty_weights_file
    )

    router_context.system_penalty_weights_report = system_penalty_weights_report

    system_penalty_context = build_system_penalty_context(
        enabled=args.system_penalty,
        mode=args.system_penalty_mode,
        lambda_sys=args.system_penalty_lambda,
        system_features_report=system_features_report,
        latency_constrained=args.max_time_ms is not None,
        execution_requested=bool(args.execute),
        penalty_weights=system_penalty_weights_report["weights"],
        penalty_weights_source=system_penalty_weights_report["source"],
    )

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
        normalization_profile=router_context.normalization_profile,
        system_penalty_fn=system_penalty_fn,
        system_penalty_apply=(
            system_penalty_context.get("enabled", False)
            and system_penalty_context.get("applied", False)
        ),
        preferred_codec=preferred_codec,
        preferred_config=preferred_config,
        preferred_reason=preferred_reason or "preferred_candidate",
    )

    apply_preferred_candidate_override(
        report=content_policy_report,
        decision=decision,
        candidate_key="suggestion",
        label_prefix="content_policy",
    )

    apply_preferred_candidate_override(
        report=content_classifier_report,
        decision=decision,
        candidate_key="prediction",
        label_prefix="content_classifier",
    )

    return build_router_report(
        args=args,
        context=router_context,
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

    run_router(
        args=args,
        router_context=router_context,
        original_argv=original_argv,
        expanded_argv=expanded_argv,
        router_config_report=router_config_report,
    )


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
