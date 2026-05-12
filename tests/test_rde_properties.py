from src.router.rde_database import RDEPoint, select_best_rde


def _three_point_decision(**kwargs):
    points = [
        RDEPoint(codec="JPEG", config="q=85", rate=0.4, quality=95.0, energy=1.0, time_ms=10.0, raw={}),
        RDEPoint(codec="JXL", config="d=1.0", rate=0.8, quality=99.0, energy=3.0, time_ms=20.0, raw={}),
        RDEPoint(codec="HEVC", config="crf=15", rate=0.2, quality=40.0, energy=0.1, time_ms=1.0, raw={}),
    ]
    return select_best_rde(
        points=points,
        weights={"w_R": 0.3, "w_E": 0.3, "w_D": 0.4},
        min_quality=50.0,
        quality_constraint_stat="mean",
        **kwargs,
    )


def test_scored_candidate_pool_contains_selected():
    decision = _three_point_decision()
    pool = decision["scored_candidate_pool"]
    selected = decision["selected"]

    keys = {(item["codec"], item["config"]) for item in pool}
    assert (selected["codec"], selected["config"]) in keys


def test_scored_candidate_pool_rank_one_matches_selected():
    decision = _three_point_decision()
    pool = decision["scored_candidate_pool"]
    selected = decision["selected"]

    rank1 = next(item for item in pool if item["rank"] == 1)
    assert rank1["codec"] == selected["codec"]
    assert rank1["config"] == selected["config"]


def test_scored_candidate_pool_cost_matches_selected_cost():
    decision = _three_point_decision()
    pool = decision["scored_candidate_pool"]
    selected = decision["selected"]

    rank1 = next(item for item in pool if item["rank"] == 1)
    assert abs(rank1["cost"] - selected["cost"]) < 1e-10


def test_scored_candidate_pool_has_cost_provenance_and_status():
    decision = _three_point_decision()
    for item in decision["scored_candidate_pool"]:
        assert item["cost_provenance"] == "router_scored"
        assert item["candidate_status"] == "router_scored"


def test_scored_candidate_pool_ranks_are_unique_and_sequential():
    decision = _three_point_decision()
    pool = decision["scored_candidate_pool"]
    ranks = sorted(item["rank"] for item in pool)
    assert ranks == list(range(1, len(pool) + 1))


def test_unscored_candidate_pool_contains_quality_guard_violations():
    decision = _three_point_decision()
    unscored = decision["unscored_candidate_pool"]

    hevc_unscored = [
        item for item in unscored
        if item["codec"] == "HEVC" and item["config"] == "crf=15"
    ]
    assert len(hevc_unscored) == 1
    assert hevc_unscored[0]["reason"] == "quality_guard_violation"
    assert hevc_unscored[0]["candidate_status"] == "infeasible_quality_guard"
    assert hevc_unscored[0]["cost_provenance"] == "unavailable_filtered"
    assert "feasible" not in hevc_unscored[0]


def test_unscored_pool_candidate_status_never_equals_router_scored():
    decision = _three_point_decision()
    for item in decision["unscored_candidate_pool"]:
        assert item["candidate_status"] != "router_scored"
        assert item["cost_provenance"] != "router_scored"


def test_unscored_candidate_pool_does_not_include_scored_candidates():
    decision = _three_point_decision()
    scored_keys = {(item["codec"], item["config"]) for item in decision["scored_candidate_pool"]}
    for item in decision["unscored_candidate_pool"]:
        assert (item["codec"], item["config"]) not in scored_keys


def test_scored_pool_supersedes_top_k_no_decision_change():
    decision = _three_point_decision(top_k=1)
    assert len(decision["top_k"]) == 1
    assert len(decision["scored_candidate_pool"]) >= 1
    assert decision["scored_candidate_pool"][0]["codec"] == decision["selected"]["codec"]


def test_energy_monotonicity_when_rate_and_quality_are_equal():
    points = [
        RDEPoint(
            codec="HighEnergy",
            config="same-rq",
            rate=1.0,
            quality=90.0,
            energy=100.0,
            time_ms=100.0,
            raw={},
        ),
        RDEPoint(
            codec="LowEnergy",
            config="same-rq",
            rate=1.0,
            quality=90.0,
            energy=1.0,
            time_ms=100.0,
            raw={},
        ),
    ]

    for energy_weight in [0.1, 0.3, 0.6, 0.9]:
        remaining = 1.0 - energy_weight

        decision = select_best_rde(
            points=points,
            weights={
                "w_E": energy_weight,
                "w_R": remaining / 2.0,
                "w_D": remaining / 2.0,
            },
            min_quality=50.0,
            quality_constraint_stat="mean",
        )

        assert decision["selected"]["codec"] == "LowEnergy"
