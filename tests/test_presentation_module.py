"""Smoke checks for the router presentation module.

Verifies that ``print_single_decision`` is importable from
``src.router.presentation`` and that it emits the two anchor lines a
caller would visually look for (selected codec and report path).
"""

from pathlib import Path

from src.router import presentation
from src.router.presentation import print_single_decision


def _minimal_report() -> dict:
    return {
        "profile": "balanced",
        "weight_source": "manual_profile",
        "aggregate_by_config": False,
        "num_rows_loaded_before_aggregation": 4,
        "weights": {"w_E": 0.3, "w_R": 0.3, "w_D": 0.4},
        "constraints": {
            "quality_constraint_stat": "mean",
            "min_quality": 50.0,
            "near_quality_floor": None,
            "allow_degraded_fallback": False,
        },
        "codec_filtering": {
            "exclude_neural": False,
            "num_after_codec_filtering": 4,
        },
        "normalization": {
            "scope": "global",
            "num_reference_points": 4,
        },
        "decision": {
            "decision_mode": "safe",
            "num_points_total": 4,
            "num_points_admissible": 4,
            "num_points_safe": 4,
            "num_points_near": 0,
            "selected": {
                "codec": "HEVC",
                "config": "crf=15",
                "rate": 0.42,
                "quality": 90.0,
                "quality_constraint_value": 90.0,
                "quality_stats": {"min": 88.0, "p10": 89.0, "p25": 89.5},
                "energy": 1.0,
                "time_ms": 10.0,
                "cost": 0.123456,
            },
        },
    }


def test_presentation_module_exports_print_single_decision():
    assert callable(getattr(presentation, "print_single_decision", None))


def test_print_single_decision_prints_selected_and_report_path(capsys, tmp_path: Path):
    json_path = tmp_path / "decision.json"
    print_single_decision(_minimal_report(), json_path)

    out = capsys.readouterr().out
    assert "=== R-D-E Router Decision ===" in out
    assert "Selected codec:             HEVC" in out
    assert "Selected config:            crf=15" in out
    assert f"Report written to: {json_path}" in out
