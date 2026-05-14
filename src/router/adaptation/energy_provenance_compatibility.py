"""Report-only compatibility audit for energy provenance tiers."""

from __future__ import annotations

from typing import Any

from src.router.adaptation.energy_provenance import (
    ENERGY_PROVENANCE_TIERS,
    classify_energy_provenance,
    summarize_energy_provenance_tiers,
)


def build_energy_provenance_compatibility_audit(
    *,
    selected: dict[str, Any],
    scored_candidate_pool: list[dict[str, Any]],
    unscored_candidate_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_tier = classify_energy_provenance(selected)
    scored_pool_tiers = summarize_energy_provenance_tiers(scored_candidate_pool)
    unscored_pool_tiers = summarize_energy_provenance_tiers(unscored_candidate_pool)
    all_candidate_tiers = summarize_energy_provenance_tiers(
        list(scored_candidate_pool) + list(unscored_candidate_pool)
    )

    nonzero_scored_tiers = [
        tier for tier in ENERGY_PROVENANCE_TIERS
        if scored_pool_tiers.get(tier, 0) > 0
    ]
    mixed_tiers = len(nonzero_scored_tiers) > 1
    warnings: list[str] = []
    severity = "ok"
    compatible = True

    if mixed_tiers:
        compatible = False
        severity = "warning"
        warnings.append("mixed_energy_provenance_tiers_in_scored_pool")

    if selected_tier == "unknown":
        compatible = False
        severity = _max_severity(severity, "warning")
        warnings.append("selected_energy_provenance_unknown")

    if selected_tier == "measured_hw_partial":
        compatible = False
        severity = "critical"
        warnings.append("selected_energy_is_partial_not_total")

    return {
        "enabled": True,
        "compatible": compatible,
        "selected_tier": selected_tier,
        "scored_pool_tiers": scored_pool_tiers,
        "unscored_pool_tiers": unscored_pool_tiers,
        "all_candidate_tiers": all_candidate_tiers,
        "mixed_tiers": mixed_tiers,
        "warnings": warnings,
        "severity": severity,
    }


def _max_severity(left: str, right: str) -> str:
    order = {
        "ok": 0,
        "warning": 1,
        "critical": 2,
    }
    return left if order[left] >= order[right] else right
