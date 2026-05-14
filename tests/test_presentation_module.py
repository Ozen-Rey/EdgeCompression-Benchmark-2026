"""Smoke checks for the router presentation module.

Verifies that ``print_single_decision`` is importable from
``src.router.presentation`` and that it emits the two anchor lines a
caller would visually look for (selected codec and report path). The
all-profiles helpers added in v0.42.26 are also exercised with capsys.
"""

from argparse import Namespace
from pathlib import Path

from src.router import presentation
from src.router.presentation import (
    print_all_profiles_footer,
    print_all_profiles_header,
    print_all_profiles_selection,
    print_single_decision,
)


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


def test_presentation_module_exports_all_profiles_helpers():
    for name in (
        "print_all_profiles_header",
        "print_all_profiles_selection",
        "print_all_profiles_footer",
    ):
        assert callable(getattr(presentation, name, None)), f"missing: {name}"


def test_print_single_decision_prints_selected_and_report_path(capsys, tmp_path: Path):
    json_path = tmp_path / "decision.json"
    print_single_decision(_minimal_report(), json_path)

    out = capsys.readouterr().out
    assert "=== R-D-E Router Decision ===" in out
    assert "Selected codec:             HEVC" in out
    assert "Selected config:            crf=15" in out
    assert f"Report written to: {json_path}" in out


def _all_profiles_args() -> Namespace:
    return Namespace(
        aggregate_by_config=False,
        available_codecs="JPEG,JXL,HEVC",
        exclude_codecs=None,
        exclude_neural=False,
        system_aware=False,
        capability_aware=False,
        strict_executables=False,
        safe_mode=False,
        quality_constraint_stat="mean",
        quality_floor=80.0,
        export_topk=False,
    )


def test_print_all_profiles_header_prints_anchor_lines(capsys):
    print_all_profiles_header(
        _all_profiles_args(),
        num_rows_loaded=12,
        num_points_after_filter=8,
        normalization_scope_label="runtime_global_before_codec_filtering",
        num_normalization_points=8,
        filter_report={},
    )

    out = capsys.readouterr().out
    assert "=== R-D-E Router: all profiles ===" in out
    assert "Loaded rows: 12" in out
    assert "Candidate points after aggregation/filtering: 8" in out
    assert "Normalization: runtime_global_before_codec_filtering (8 reference points)" in out
    assert "Quality guard: mean >= 80.0" in out


def test_print_all_profiles_header_includes_cuda_lines_when_system_aware(capsys):
    args = _all_profiles_args()
    args.system_aware = True
    filter_report = {
        "system_aware": {
            "cuda_available": True,
            "effective_exclude_neural": False,
        }
    }

    print_all_profiles_header(
        args,
        num_rows_loaded=1,
        num_points_after_filter=1,
        normalization_scope_label="runtime_global_before_codec_filtering",
        num_normalization_points=1,
        filter_report=filter_report,
    )

    out = capsys.readouterr().out
    assert "CUDA available: True" in out
    assert "Effective exclude neural: False" in out


def test_print_all_profiles_selection_emits_single_line_per_profile(capsys):
    report = {
        "decision": {
            "decision_mode": "safe",
            "selected": {
                "codec": "HEVC",
                "config": "crf=15",
                "rate": 0.42,
                "quality": 91.0,
                "quality_constraint_value": 90.0,
                "energy": 1.5,
                "cost": 0.123456,
            },
        },
    }

    print_all_profiles_selection("balanced", report)

    out = capsys.readouterr().out
    assert "balanced" in out
    assert "HEVC crf=15" in out
    assert "mode=safe" in out
    assert "R=0.420000" in out
    assert "Qmean=91.00" in out
    assert "Qguard=90.00" in out
    assert "E=1.500000" in out
    assert "J=0.123456" in out


def test_print_all_profiles_footer_without_topk(capsys, tmp_path: Path):
    print_all_profiles_footer(
        out_dir=tmp_path / "routing",
        summary_path=tmp_path / "routing" / "summary.csv",
    )

    out = capsys.readouterr().out
    assert "JSON reports written to:" in out
    assert "Summary written to:" in out
    assert "Top-k written to:" not in out


def test_print_all_profiles_footer_with_topk(capsys, tmp_path: Path):
    print_all_profiles_footer(
        out_dir=tmp_path / "routing",
        summary_path=tmp_path / "routing" / "summary.csv",
        topk_path=tmp_path / "routing" / "topk.csv",
    )

    out = capsys.readouterr().out
    assert "Top-k written to:" in out
