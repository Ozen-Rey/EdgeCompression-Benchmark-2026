from src.router.adaptation.energy_tier_policy import build_energy_tier_policy_shadow


def _candidate(
    codec: str,
    tier: str,
    *,
    rank: int,
    cost: float,
) -> dict:
    return {
        "codec": codec,
        "config": "cfg",
        "rank": rank,
        "cost": cost,
        "ranking_cost": cost,
        "energy": 1.0,
        "energy_provenance_tier": tier,
    }


def test_single_tier_benchmark_pool_no_action():
    selected = _candidate("JPEG", "benchmark_reference", rank=1, cost=0.1)
    scored = [
        selected,
        _candidate("JXL", "benchmark_reference", rank=2, cost=0.2),
    ]

    report = build_energy_tier_policy_shadow(
        selected=selected,
        scored_candidate_pool=scored,
    )

    assert report["enabled"] is True
    assert report["mode"] == "report-only"
    assert report["policy"] == "strict-compatible"
    assert report["status"] == "no_action_single_tier_pool"
    assert report["would_change_decision"] is False
    assert report["shadow_selected"]["codec"] == "JPEG"
    assert report["rejected_by_policy"] == []


def test_selected_benchmark_with_all_benchmark_pool_is_unchanged():
    selected = _candidate("JPEG", "benchmark_reference", rank=1, cost=0.1)

    report = build_energy_tier_policy_shadow(
        selected=selected,
        scored_candidate_pool=[selected],
    )

    assert report["would_change_decision"] is False
    assert report["current_selected"] == report["shadow_selected"]


def test_selected_partial_changes_to_benchmark_reference_alternative():
    selected = _candidate("GPUOnly", "measured_hw_partial", rank=1, cost=0.1)
    alternative = _candidate("JPEG", "benchmark_reference", rank=2, cost=0.2)

    report = build_energy_tier_policy_shadow(
        selected=selected,
        scored_candidate_pool=[selected, alternative],
    )

    assert report["would_change_decision"] is True
    assert report["shadow_selected"]["codec"] == "JPEG"
    assert report["reason"] == (
        "selected_energy_tier_less_reliable_than_available_alternative"
    )
    assert report["rejected_by_policy"][0]["codec"] == "GPUOnly"


def test_selected_unknown_changes_to_measured_total_alternative():
    selected = _candidate("Unknown", "unknown", rank=1, cost=0.1)
    alternative = _candidate("Measured", "measured_hw_total", rank=2, cost=0.2)

    report = build_energy_tier_policy_shadow(
        selected=selected,
        scored_candidate_pool=[selected, alternative],
    )

    assert report["would_change_decision"] is True
    assert report["shadow_selected"]["codec"] == "Measured"
    assert report["shadow_selected"]["energy_provenance_tier"] == (
        "measured_hw_total"
    )


def test_selected_measured_total_not_replaced_by_benchmark_reference():
    selected = _candidate("Measured", "measured_hw_total", rank=1, cost=0.1)
    benchmark = _candidate("JPEG", "benchmark_reference", rank=2, cost=0.2)

    report = build_energy_tier_policy_shadow(
        selected=selected,
        scored_candidate_pool=[selected, benchmark],
    )

    assert report["would_change_decision"] is False
    assert report["shadow_selected"]["codec"] == "Measured"
    assert report["status"] == "no_action_selected_tier_is_most_reliable_available"


def test_report_only_policy_does_not_reorder_scored_pool():
    selected = _candidate("Unknown", "unknown", rank=1, cost=0.1)
    alternative = _candidate("Measured", "measured_hw_total", rank=2, cost=0.2)
    scored = [selected, alternative]

    build_energy_tier_policy_shadow(
        selected=selected,
        scored_candidate_pool=scored,
    )

    assert [item["codec"] for item in scored] == ["Unknown", "Measured"]
    assert [item["rank"] for item in scored] == [1, 2]
