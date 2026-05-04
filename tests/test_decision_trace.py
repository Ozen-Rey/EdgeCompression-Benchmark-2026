from src.router.rde_database import RDEPoint, select_best_rde


def test_cost_decomposition_sums_to_j_rde():
    points = [
        RDEPoint(
            codec="A",
            config="q=1",
            rate=1.0,
            quality=80.0,
            energy=10.0,
            time_ms=100.0,
            raw={},
        ),
        RDEPoint(
            codec="B",
            config="q=2",
            rate=2.0,
            quality=90.0,
            energy=20.0,
            time_ms=200.0,
            raw={},
        ),
    ]

    weights = {
        "w_R": 0.3,
        "w_E": 0.3,
        "w_D": 0.4,
    }

    decision = select_best_rde(
        points=points,
        weights=weights,
        min_quality=70.0,
        max_rate=None,
        max_energy=None,
        max_time_ms=None,
        quality_constraint_stat="mean",
        near_quality_floor=None,
        allow_degraded_fallback=False,
        top_k=2,
    )

    selected = decision["selected"]
    decomp = selected["cost_decomposition"]

    assert decomp["sum"] == selected["cost"]

    reconstructed = (
        decomp["term_R"]
        + decomp["term_E"]
        + decomp["term_D"]
    )

    assert abs(reconstructed - selected["cost"]) < 1e-12


def test_decision_trace_records_safe_pool_reason():
    points = [
        RDEPoint(
            codec="A",
            config="q=1",
            rate=1.0,
            quality=80.0,
            energy=10.0,
            time_ms=100.0,
            raw={},
        ),
        RDEPoint(
            codec="B",
            config="q=2",
            rate=2.0,
            quality=90.0,
            energy=20.0,
            time_ms=200.0,
            raw={},
        ),
    ]

    weights = {
        "w_R": 0.3,
        "w_E": 0.3,
        "w_D": 0.4,
    }

    decision = select_best_rde(
        points=points,
        weights=weights,
        min_quality=70.0,
        quality_constraint_stat="mean",
        allow_degraded_fallback=False,
        top_k=2,
    )

    trace = decision["decision_trace"]

    assert trace["enabled"] is True
    assert trace["decision_mode"] == "safe"
    assert trace["active_pool"] == "safe_pool"
    assert trace["selected_reason"] == "lowest_J_RDE_in_safe_pool"
    assert trace["ranking_key"] == "minimize_J_RDE"
