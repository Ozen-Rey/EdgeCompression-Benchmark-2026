"""Report-only shadow policy for energy provenance tiers."""

from __future__ import annotations

from typing import Any

try:
    from .energy_provenance import classify_energy_provenance
except ImportError:  # pragma: no cover - direct script fallback
    from energy_provenance import classify_energy_provenance


RELIABILITY_ORDER = [
    "measured_hw_total",
    "derived_time_scaled",
    "benchmark_reference",
    "measured_hw_partial",
    "unknown",
]

RELIABILITY_RANK = {
    tier: rank for rank, tier in enumerate(RELIABILITY_ORDER)
}


def build_energy_tier_policy_shadow(
    *,
    selected: dict[str, Any],
    scored_candidate_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    """Simulate a future strict-compatible tier policy without applying it."""

    current_selected = _candidate_summary(selected)
    base_report: dict[str, Any] = {
        "enabled": True,
        "mode": "report-only",
        "policy": "strict-compatible",
        "status": None,
        "would_change_decision": False,
        "current_selected": current_selected,
        "shadow_selected": current_selected,
        "rejected_by_policy": [],
        "warnings": [],
    }

    if not scored_candidate_pool:
        base_report["status"] = "no_action_empty_scored_pool"
        base_report["warnings"] = ["scored_candidate_pool_empty"]
        return base_report

    tiers = {
        classify_energy_provenance(candidate)
        for candidate in scored_candidate_pool
    }

    if len(tiers) == 1:
        base_report["status"] = "no_action_single_tier_pool"
        return base_report

    selected_tier = classify_energy_provenance(selected)
    best_available_tier = _most_reliable_tier(tiers)

    if _is_more_reliable(best_available_tier, selected_tier):
        alternatives = [
            candidate
            for candidate in scored_candidate_pool
            if classify_energy_provenance(candidate) == best_available_tier
        ]
        if alternatives:
            shadow_selected = alternatives[0]
            base_report["status"] = "shadow_selection_changed"
            base_report["would_change_decision"] = True
            base_report["shadow_selected"] = _candidate_summary(shadow_selected)
            base_report["reason"] = (
                "selected_energy_tier_less_reliable_than_available_alternative"
            )
            base_report["warnings"] = [
                "selected_energy_tier_less_reliable_than_available_alternative"
            ]
            base_report["rejected_by_policy"] = _rejected_candidates(
                scored_candidate_pool,
                compatible_tier=best_available_tier,
            )
            return base_report

        base_report["status"] = "no_action_no_compatible_alternative"
        base_report["reason"] = "no_more_reliable_compatible_candidate_found"
        base_report["warnings"] = ["no_more_reliable_compatible_candidate_found"]
        return base_report

    base_report["status"] = "no_action_selected_tier_is_most_reliable_available"
    base_report["rejected_by_policy"] = _rejected_candidates(
        scored_candidate_pool,
        compatible_tier=selected_tier,
    )
    return base_report


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "codec": candidate.get("codec"),
        "config": candidate.get("config"),
        "rank": candidate.get("rank"),
        "cost": candidate.get("cost"),
        "ranking_cost": candidate.get("ranking_cost"),
        "energy_provenance_tier": classify_energy_provenance(candidate),
    }


def _most_reliable_tier(tiers: set[str]) -> str:
    return min(
        tiers,
        key=lambda tier: RELIABILITY_RANK.get(tier, len(RELIABILITY_RANK)),
    )


def _is_more_reliable(candidate_tier: str, selected_tier: str) -> bool:
    return RELIABILITY_RANK.get(candidate_tier, len(RELIABILITY_RANK)) < (
        RELIABILITY_RANK.get(selected_tier, len(RELIABILITY_RANK))
    )


def _rejected_candidates(
    scored_candidate_pool: list[dict[str, Any]],
    *,
    compatible_tier: str,
) -> list[dict[str, Any]]:
    rejected = []
    for candidate in scored_candidate_pool:
        tier = classify_energy_provenance(candidate)
        if tier == compatible_tier:
            continue
        item = _candidate_summary(candidate)
        item["reason"] = "incompatible_energy_provenance_tier"
        item["compatible_tier"] = compatible_tier
        rejected.append(item)
    return rejected
