from src.router.adaptation.energy_provenance_compatibility import (
    build_energy_provenance_compatibility_audit,
)


def _candidate(codec: str, tier: str) -> dict:
    return {
        "codec": codec,
        "config": "cfg",
        "energy": 1.0,
        "energy_provenance_tier": tier,
    }


def test_all_benchmark_reference_is_compatible_ok():
    scored = [
        _candidate("JPEG", "benchmark_reference"),
        _candidate("JXL", "benchmark_reference"),
    ]

    audit = build_energy_provenance_compatibility_audit(
        selected=scored[0],
        scored_candidate_pool=scored,
        unscored_candidate_pool=[],
    )

    assert audit["enabled"] is True
    assert audit["compatible"] is True
    assert audit["mixed_tiers"] is False
    assert audit["severity"] == "ok"
    assert audit["warnings"] == []


def test_mixed_benchmark_and_time_scaled_scored_pool_warns():
    scored = [
        _candidate("JPEG", "benchmark_reference"),
        _candidate("JXL", "derived_time_scaled"),
    ]

    audit = build_energy_provenance_compatibility_audit(
        selected=scored[0],
        scored_candidate_pool=scored,
        unscored_candidate_pool=[],
    )

    assert audit["compatible"] is False
    assert audit["mixed_tiers"] is True
    assert audit["severity"] == "warning"
    assert "mixed_energy_provenance_tiers_in_scored_pool" in audit["warnings"]


def test_selected_measured_partial_is_critical():
    selected = _candidate("GPUOnly", "measured_hw_partial")

    audit = build_energy_provenance_compatibility_audit(
        selected=selected,
        scored_candidate_pool=[selected],
        unscored_candidate_pool=[],
    )

    assert audit["compatible"] is False
    assert audit["severity"] == "critical"
    assert "selected_energy_is_partial_not_total" in audit["warnings"]


def test_selected_unknown_warns():
    selected = _candidate("Unknown", "unknown")

    audit = build_energy_provenance_compatibility_audit(
        selected=selected,
        scored_candidate_pool=[selected],
        unscored_candidate_pool=[],
    )

    assert audit["compatible"] is False
    assert audit["severity"] == "warning"
    assert "selected_energy_provenance_unknown" in audit["warnings"]


def test_all_candidate_tiers_include_unscored_pool():
    scored = [_candidate("JPEG", "benchmark_reference")]
    unscored = [_candidate("GPUOnly", "measured_hw_partial")]

    audit = build_energy_provenance_compatibility_audit(
        selected=scored[0],
        scored_candidate_pool=scored,
        unscored_candidate_pool=unscored,
    )

    assert audit["scored_pool_tiers"]["benchmark_reference"] == 1
    assert audit["unscored_pool_tiers"]["measured_hw_partial"] == 1
    assert audit["all_candidate_tiers"]["benchmark_reference"] == 1
    assert audit["all_candidate_tiers"]["measured_hw_partial"] == 1
