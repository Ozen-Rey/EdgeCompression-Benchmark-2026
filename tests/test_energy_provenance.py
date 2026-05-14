from src.router.adaptation.energy_provenance import (
    build_energy_provenance_summary,
    classify_energy_provenance,
)
from src.router.core.rde_database import RDEPoint


def test_benchmark_point_without_local_energy_metadata_is_reference():
    point = RDEPoint(
        codec="JPEG",
        config="q=85",
        rate=1.0,
        quality=90.0,
        energy=0.1,
        raw={},
    )

    assert classify_energy_provenance(point) == "benchmark_reference"


def test_measured_total_energy_tier():
    record = {
        "energy": 0.1,
        "energy_is_measured": True,
        "energy_usable_for_total": True,
    }

    assert classify_energy_provenance(record) == "measured_hw_total"


def test_measured_partial_energy_tier_never_total():
    record = {
        "energy": 0.1,
        "energy_is_measured": True,
        "energy_usable_for_total": False,
        "energy_scope": "gpu",
    }

    assert classify_energy_provenance(record) == "measured_hw_partial"


def test_time_scaled_energy_tier():
    record = {
        "energy": 0.1,
        "energy_scaling_method": "benchmark_energy_scaled_by_time_ratio",
    }

    assert classify_energy_provenance(record) == "derived_time_scaled"


def test_energy_provenance_summary_counts_tiers():
    selected = {"energy": 0.1, "energy_provenance_tier": "measured_hw_total"}
    scored = [
        selected,
        {"energy": 0.2, "energy_provenance_tier": "benchmark_reference"},
    ]
    unscored = [
        {"energy": 0.3, "energy_provenance_tier": "measured_hw_partial"},
        {"energy": 0.4, "energy_provenance_tier": "derived_time_scaled"},
    ]

    summary = build_energy_provenance_summary(
        selected=selected,
        scored_candidate_pool=scored,
        unscored_candidate_pool=unscored,
    )

    assert summary["selected_tier"] == "measured_hw_total"
    assert summary["counts"]["measured_hw_total"] == 1
    assert summary["counts"]["benchmark_reference"] == 1
    assert summary["counts"]["measured_hw_partial"] == 1
    assert summary["counts"]["derived_time_scaled"] == 1
    assert summary["counts"]["unknown"] == 0
