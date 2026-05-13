"""Report-only energy provenance tier classification."""

from __future__ import annotations

from typing import Any


ENERGY_PROVENANCE_TIERS = [
    "measured_hw_total",
    "measured_hw_partial",
    "derived_time_scaled",
    "benchmark_reference",
    "unknown",
]

LOCAL_ENERGY_METADATA_KEYS = {
    "energy_is_measured",
    "energy_usable_for_total",
    "energy_scope",
    "energy_backend",
    "energy_method",
    "energy_quality",
    "energy_scaling_method",
    "current_method",
}


def classify_energy_provenance(record: Any) -> str:
    """Classify energy provenance without changing router scoring."""

    data = _as_mapping(record)

    explicit_tier = data.get("energy_provenance_tier")
    if explicit_tier in ENERGY_PROVENANCE_TIERS:
        return str(explicit_tier)

    energy_is_measured = _parse_bool_or_none(data.get("energy_is_measured"))
    energy_usable_for_total = _parse_bool_or_none(
        data.get("energy_usable_for_total")
    )

    if energy_is_measured is True and energy_usable_for_total is True:
        return "measured_hw_total"

    if energy_is_measured is True and energy_usable_for_total is False:
        return "measured_hw_partial"

    method_text = " ".join(
        str(data.get(key, "") or "")
        for key in (
            "current_method",
            "energy_method",
            "energy_scaling_method",
            "calibration_method",
        )
    ).lower()

    if (
        "benchmark_energy_scaled_by_time_ratio" in method_text
        or "time_scaled" in method_text
        or "time-scaling" in method_text
    ):
        return "derived_time_scaled"

    if _has_energy_value(data) and not _has_local_energy_metadata(data):
        return "benchmark_reference"

    return "unknown"


def summarize_energy_provenance_tiers(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {tier: 0 for tier in ENERGY_PROVENANCE_TIERS}

    for record in records:
        tier = classify_energy_provenance(record)
        counts[tier] = counts.get(tier, 0) + 1

    return counts


def build_energy_provenance_summary(
    *,
    selected: dict[str, Any],
    scored_candidate_pool: list[dict[str, Any]],
    unscored_candidate_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    all_candidates = list(scored_candidate_pool) + list(unscored_candidate_pool)

    return {
        "enabled": True,
        "selected_tier": classify_energy_provenance(selected),
        "counts": summarize_energy_provenance_tiers(all_candidates),
        "scored_candidate_pool": summarize_energy_provenance_tiers(
            scored_candidate_pool
        ),
        "unscored_candidate_pool": summarize_energy_provenance_tiers(
            unscored_candidate_pool
        ),
    }


def _as_mapping(record: Any) -> dict[str, Any]:
    if record is None:
        return {}

    if isinstance(record, dict):
        data = dict(record)
        raw = data.get("raw")
        if isinstance(raw, dict):
            merged = dict(raw)
            merged.update(data)
            return merged
        return data

    raw = getattr(record, "raw", None)
    data = dict(raw) if isinstance(raw, dict) else {}

    for key in (
        "energy",
        "energy_is_measured",
        "energy_usable_for_total",
        "energy_method",
        "energy_scaling_method",
        "current_method",
    ):
        if hasattr(record, key):
            data[key] = getattr(record, key)

    return data


def _parse_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _has_energy_value(data: dict[str, Any]) -> bool:
    value = data.get("energy")
    if value in (None, ""):
        return False

    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _has_local_energy_metadata(data: dict[str, Any]) -> bool:
    for key in LOCAL_ENERGY_METADATA_KEYS:
        value = data.get(key)
        if value not in (None, ""):
            return True
    return False
