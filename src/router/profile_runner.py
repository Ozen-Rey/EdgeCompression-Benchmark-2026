"""Profile runner.

Hosts the per-profile orchestration that used to live in
``src.router.rde_router._run_profile``, along with the side-effect-free
helpers that feed it. After v0.42.34 ``pipeline.run_router`` imports
:func:`run_profile` directly from this module so the router pipeline no
longer depends on :mod:`src.router.rde_router` at all; rde_router only
keeps the CLI entrypoint and the ``__main__`` exception wrapper. The
weight-resolution helper :func:`build_weights_for_profile` also lives
here (re-exported by ``src.router.pipeline``) so the pipeline module
has a one-way dependency on this one with no lazy back-edge.

Helpers here are written so they can be exercised by unit tests without
spinning up the full router pipeline.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

from src.router.adaptation.content_classifier_model import (
    build_metadata_no_source_features,
    extract_metadata_features_from_image,
    load_content_classifier_config,
    load_training_rows_from_config,
    predict_content_classifier,
)
from src.router.adaptation.content_policy import (
    build_content_policy_report,
    get_content_policy_preferred_candidate,
)
from src.router.adaptation.context_policy import compute_context_policy
from src.router.adaptation.system_penalty import (
    build_system_penalty_context,
    load_system_penalty_weights,
    make_system_penalty_fn,
)
from src.router.adaptation.system_policy import build_system_policy
from src.router.context import RouterContext
from src.router.core.profiles import get_profile
from src.router.core.rde_database import RDEPoint, select_best_rde
from src.router.report import build_router_report


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


def apply_preferred_candidate_override(
    *,
    report: Dict[str, Any],
    decision: Dict[str, Any],
    candidate_key: str,
    label_prefix: str,
) -> None:
    """Resolve whether a preferred candidate (suggestion/prediction) was selected.

    Shared between the content-policy and content-classifier ``apply`` paths.
    Mutates ``report`` in place: sets ``applied``, appends to ``reasons``/
    ``warnings``, and records ``decision_audit`` when the router falls back
    from the preferred candidate to its own J_RDE-ranked choice. No effect
    when the report is disabled, in report-only mode, or has no candidate.
    """
    if not (report.get("enabled") and report.get("mode") == "apply"):
        return

    candidate = report.get(candidate_key)
    if not candidate:
        return

    candidate_codec = str(candidate.get("codec"))
    candidate_config = str(candidate.get("config"))

    selected = decision.get("selected", {})
    selected_codec = str(selected.get("codec"))
    selected_config = str(selected.get("config"))

    if selected_codec == candidate_codec and selected_config == candidate_config:
        report["applied"] = True
        report["reasons"].append(f"{label_prefix}_{candidate_key}_selected")
        return

    report["applied"] = False

    preferred_audit = (
        decision.get("decision_trace", {}).get("preferred_candidate")
    )
    report["decision_audit"] = preferred_audit

    if preferred_audit and preferred_audit.get("admissible") is True:
        report["warnings"].append(
            f"{label_prefix}_{candidate_key}_not_j_total_competitive_fallback_to_router"
        )
        report["reasons"].append(
            f"{candidate_key}_admissible_but_not_competitive"
        )
    else:
        report["warnings"].append(
            f"{label_prefix}_{candidate_key}_not_admissible_fallback_to_router"
        )
        report["reasons"].append(
            f"{candidate_key}_not_admissible"
        )

    report["reasons"].append("fallback_to_router_selection")


def build_time_guard_report(
    points,
    max_time_ms,
    strict_time: bool = False,
) -> Dict[str, Any]:
    """Build the ``time_guard`` section of the router report.

    Returns a disabled report when ``max_time_ms`` is None. With a real
    constraint, classifies candidates into ``within_limit``,
    ``over_limit``, and ``missing_time`` buckets and raises ``ValueError``
    when timing data is unavailable (or required by ``strict_time``).
    """
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


def build_content_classifier_router_report(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build the content-classifier section of the router report.

    Returns a disabled report when ``--content-classifier`` is off; with
    the flag enabled, validates the mode, loads the classifier config,
    extracts metadata features (from image or width/height), and returns
    the classifier prediction. Schema is stable across all paths so the
    router report can rely on a fixed set of keys.
    """
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


def run_profile(
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
    """Run the router for a single profile and return its router report.

    Resolves weights and system policy, builds the content-policy and
    content-classifier reports, picks a preferred candidate (if any),
    invokes :func:`select_best_rde` with the J_RDE objective, and
    finally assembles the structured router report via
    :func:`build_router_report`. Side-effects are limited to mutations
    on ``router_context`` and the report dicts it carries.
    """
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

    content_classifier_report = build_content_classifier_router_report(args)
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

    time_guard_report = build_time_guard_report(
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
