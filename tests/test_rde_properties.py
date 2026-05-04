from src.router.rde_database import RDEPoint, select_best_rde


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
