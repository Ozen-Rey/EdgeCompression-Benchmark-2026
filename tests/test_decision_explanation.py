"""Tests for ``src.router.observability.decision_explanation``.

Covers: importability, structured-build correctness on representative
router-report fixtures (minimal, system-penalty active, content-policy
accepted, content-classifier rejected, degraded-fallback, infeasible),
Markdown render invariants, and the CLI smoke (--help and writing
--out-md / --out-json to disk).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from src.router.observability import decision_explanation
from src.router.observability.decision_explanation import (
    build_decision_explanation,
    render_decision_explanation_markdown,
)
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "decision_explanation"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _minimal_report() -> Dict[str, Any]:
    return {
        "router_version": "0.43.1",
        "domain": "image",
        "profile": "balanced",
        "weight_source": "manual_profile",
        "weights": {"w_R": 0.2, "w_E": 0.2, "w_D": 0.6},
        "constraints": {
            "min_quality": 70.0,
            "quality_floor": 70.0,
            "quality_constraint_stat": "p10",
            "near_quality_floor": None,
            "allow_degraded_fallback": False,
            "max_rate": None,
            "max_energy": None,
            "max_time_ms": None,
        },
        "normalization_audit": {
            "mode": "runtime",
            "scope": "global",
            "comparability": "comparable_under_global_normalization",
            "quality_metric": "ssimulacra2",
        },
        "codec_filtering": {
            "available_codecs": "JPEG,JXL,HEVC",
            "excluded_codecs": None,
            "strict_executables": True,
            "capability_filtering": True,
        },
        "decision": {
            "selected": {
                "codec": "JXL",
                "config": "d=1.0",
                "rate": 1.372,
                "quality": 85.18,
                "energy": 2.552,
                "time_ms": 250.0,
                "cost": 0.234,
                "J_total": 0.234,
                "system_penalty": 0.0,
                "ranking_cost": 0.234,
                "decision_mode": "safe",
                "cost_decomposition": {
                    "w_R": 0.2,
                    "w_E": 0.2,
                    "w_D": 0.6,
                    "norm_rate": 0.31,
                    "norm_energy": 0.52,
                    "norm_distortion": 0.10,
                    "term_R": 0.062,
                    "term_E": 0.104,
                    "term_D": 0.06,
                    "sum": 0.234,
                },
            },
            "decision_trace": {
                "enabled": True,
                "selected_reason": "lowest_J_RDE_in_safe_pool",
                "active_pool": "safe_pool",
                "decision_mode": "safe",
                "quality_guard_applied": True,
                "quality_constraint_stat": "p10",
                "quality_floor": 70.0,
                "near_quality_floor": None,
                "allow_degraded_fallback": False,
                "normalization": "runtime_minmax",
                "ranking_key": "minimize_J_RDE",
                "system_penalty_applied": False,
                "preferred_candidate": None,
                "cost_formula": "J_RDE = w_R*R_norm + w_E*E_norm + w_D*D_norm",
            },
            "quality_guard": {
                "hard_constraint": True,
                "stat": "p10",
                "floor": 70.0,
                "near_floor": None,
                "allow_degraded_fallback": False,
                "max_time_ms": None,
            },
        },
        "content_policy": {"enabled": False, "mode": "report-only", "suggestion": None},
        "content_classifier": {
            "enabled": False,
            "mode": "report-only",
            "prediction": None,
        },
        "system_policy": {"enabled": False},
        "system_penalty": {"enabled": False},
        "calibration": {"enabled": False},
        "calibration_bundle": {"enabled": False},
        "energy_provenance": {
            "current_method": "benchmark_energy",
            "energy_is_measured": False,
            "energy_quality": "benchmark_derived",
            "energy_scope": "none",
            "energy_usable_for_total": False,
        },
        "time_guard": {"enabled": False, "max_time_ms": None},
    }


def test_module_is_importable():
    assert hasattr(decision_explanation, "build_decision_explanation")
    assert hasattr(decision_explanation, "render_decision_explanation_markdown")
    assert hasattr(decision_explanation, "main")


def test_build_minimal_report_extracts_selected_and_why():
    explanation = build_decision_explanation(_minimal_report())

    selected = explanation["selected"]
    assert selected["codec"] == "JXL"
    assert selected["config"] == "d=1.0"
    assert selected["selected_reason"] == "lowest_J_RDE_in_safe_pool"
    assert selected["active_pool"] == "safe_pool"
    assert selected["active_ranking_name"] == "J_RDE"
    assert selected["active_ranking_value"] == 0.234
    assert selected["system_penalty_applied"] is False

    why = explanation["why_selected"]
    assert len(why) >= 2
    assert any("quality guard" in line for line in why)
    assert any("safe admissible pool" in line for line in why)

    fallback = explanation["fallback_safety"]
    assert fallback["category"] == "safe_pool_selection"


def test_markdown_render_contains_required_sections():
    explanation = build_decision_explanation(_minimal_report())
    md = render_decision_explanation_markdown(explanation)

    assert md.startswith("# R-D-E Router Decision Explanation")
    for header in [
        "## Selected candidate",
        "## Why this candidate was selected",
        "## Active constraints",
        "## Predictor role",
        "## Fallback / safety",
        "## Cost decomposition",
        "## Framing notes",
    ]:
        assert header in md
    assert "JXL" in md
    assert "d=1.0" in md
    assert "lowest_J_RDE_in_safe_pool" in md


def test_content_classifier_accepted_path():
    report = _minimal_report()
    report["content_classifier"] = {
        "enabled": True,
        "mode": "apply",
        "applied": True,
        "prediction": {"codec": "JXL", "config": "d=1.0"},
        "reasons": ["content_classifier_prediction_selected"],
        "warnings": [],
    }
    report["decision"]["decision_trace"]["selected_reason"] = (
        "content_classifier_preferred_candidate"
    )
    report["decision"]["decision_trace"]["preferred_candidate"] = {
        "codec": "JXL",
        "config": "d=1.0",
        "reason": "content_classifier_preferred_candidate",
        "admissible": True,
        "selected": True,
        "competitive": True,
        "ranking_key": "J_RDE",
        "preferred_ranking_cost": 0.234,
        "best_ranking_cost": 0.234,
    }

    explanation = build_decision_explanation(report)
    fallback = explanation["fallback_safety"]

    assert fallback["category"] == "preferred_candidate_accepted"
    predictor = explanation["predictor_role"]
    assert predictor["content_classifier_enabled"] is True
    assert predictor["content_classifier"]["applied"] is True

    why = explanation["why_selected"]
    assert any("matches a preferred candidate" in line for line in why)


def test_content_policy_rejected_falls_back():
    report = _minimal_report()
    report["content_policy"] = {
        "enabled": True,
        "mode": "apply",
        "applied": False,
        "suggestion": {"codec": "HEVC", "config": "crf=15"},
        "reasons": [
            "suggestion_admissible_but_not_competitive",
            "fallback_to_router_selection",
        ],
        "warnings": [
            "content_policy_suggestion_not_j_total_competitive_fallback_to_router",
        ],
    }
    report["decision"]["decision_trace"]["preferred_candidate"] = {
        "codec": "HEVC",
        "config": "crf=15",
        "reason": "content_policy_preferred_candidate",
        "admissible": True,
        "selected": False,
        "competitive": False,
        "ranking_key": "J_RDE",
        "preferred_ranking_cost": 0.9,
        "best_ranking_cost": 0.234,
    }

    explanation = build_decision_explanation(report)
    fallback = explanation["fallback_safety"]

    assert fallback["category"] == "preferred_candidate_rejected"
    assert explanation["selected"]["codec"] == "JXL"
    assert "fallback_to_router_selection" in explanation["predictor_role"][
        "content_policy"
    ]["reasons"]
    assert any(
        "admissible but not competitive" in line
        for line in explanation["why_selected"]
    )


def test_system_penalty_changes_active_ranking_to_j_total():
    report = _minimal_report()
    report["system_penalty"] = {
        "enabled": True,
        "applied": True,
        "mode": "apply",
        "lambda_sys": 0.1,
    }
    report["decision"]["selected"]["J_total"] = 0.300
    report["decision"]["selected"]["system_penalty"] = 0.066
    report["decision"]["selected"]["ranking_cost"] = 0.300
    report["decision"]["decision_trace"]["system_penalty_applied"] = True
    report["decision"]["decision_trace"]["ranking_key"] = "minimize_J_total"
    report["decision"]["decision_trace"]["selected_reason"] = (
        "lowest_J_total_in_safe_pool"
    )

    explanation = build_decision_explanation(report)
    selected = explanation["selected"]

    assert selected["active_ranking_name"] == "J_total"
    assert selected["active_ranking_value"] == 0.300
    assert selected["system_penalty_applied"] is True

    decomposition = explanation["cost_decomposition"]
    assert decomposition["active_ranking"] == "J_total"
    assert decomposition["lambda_sys"] == 0.1
    assert decomposition["J_total"] == 0.300


def test_degraded_fallback_category():
    report = _minimal_report()
    report["decision"]["decision_trace"]["decision_mode"] = "degraded_fallback"
    report["decision"]["decision_trace"]["active_pool"] = "near_pool"
    report["decision"]["decision_trace"]["selected_reason"] = (
        "lowest_J_RDE_in_degraded_fallback_pool"
    )
    report["decision"]["selected"]["decision_mode"] = "degraded_fallback"

    explanation = build_decision_explanation(report)
    fallback = explanation["fallback_safety"]
    assert fallback["category"] == "degraded_fallback_selection"
    assert any(
        "near-floor (degraded fallback)" in line
        for line in explanation["why_selected"]
    )


def test_infeasible_report_handled_gracefully():
    report = _minimal_report()
    report["decision"] = {}

    explanation = build_decision_explanation(report)
    assert explanation["selected"] is None
    assert explanation["fallback_safety"]["category"] == "infeasible_request"
    assert explanation["cost_decomposition"] is None

    md = render_decision_explanation_markdown(explanation)
    assert "infeasible" in md.lower()


def test_missing_optional_blocks_render_as_unavailable():
    report = _minimal_report()
    # Strip the cost decomposition; ensure the section renders as unavailable
    # rather than carrying invented numbers.
    del report["decision"]["selected"]["cost_decomposition"]

    explanation = build_decision_explanation(report)
    assert explanation["cost_decomposition"] is None

    md = render_decision_explanation_markdown(explanation)
    assert "## Cost decomposition" in md
    assert "_unavailable_" in md.split("## Cost decomposition", 1)[1]


def test_build_is_deterministic_for_same_input():
    report = _minimal_report()
    a = build_decision_explanation(copy.deepcopy(report))
    b = build_decision_explanation(copy.deepcopy(report))
    assert a == b


def test_cli_help_exits_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.router.observability.decision_explanation",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "report" in result.stdout.lower()


def test_cli_writes_out_md_and_out_json(monkeypatch):
    report_path = _tmp_path("report.json")
    report_path.write_text(json.dumps(_minimal_report()), encoding="utf-8")

    out_md = _tmp_path("explanation.md")
    out_json = _tmp_path("explanation.json")

    decision_explanation.main(
        [
            "--report",
            str(report_path),
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ]
    )

    assert out_md.is_file()
    assert out_json.is_file()

    md_text = out_md.read_text(encoding="utf-8")
    assert md_text.startswith("# R-D-E Router Decision Explanation")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["selected"]["codec"] == "JXL"
    assert payload["fallback_safety"]["category"] == "safe_pool_selection"


def test_cli_prints_to_stdout_when_no_outputs_requested(capsys):
    report_path = _tmp_path("report_stdout.json")
    report_path.write_text(json.dumps(_minimal_report()), encoding="utf-8")

    decision_explanation.main(["--report", str(report_path)])
    captured = capsys.readouterr()
    assert "# R-D-E Router Decision Explanation" in captured.out
