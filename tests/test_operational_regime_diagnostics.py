"""Tests for operational-regime diagnostics."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.router.analysis import operational_regime_diagnostics as ordx


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rde_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset, image, width, height in [
        ("small", "s1", 640, 512),
        ("small", "s2", 720, 540),
        ("large", "l1", 2560, 1440),
        ("large", "l2", 3840, 2160),
    ]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "image": image,
                    "codec": "JPEG",
                    "param": "q=85",
                    "bpp": 1.2,
                    "psnr": 34.0,
                    "ssimulacra2": 82.0,
                    "energy_per_image_j": 0.05,
                    "width": width,
                    "height": height,
                },
                {
                    "dataset": dataset,
                    "image": image,
                    "codec": "JXL",
                    "param": "d=1.0",
                    "bpp": 0.8,
                    "psnr": 36.0,
                    "ssimulacra2": 86.0,
                    "energy_per_image_j": 0.08,
                    "width": width,
                    "height": height,
                },
                {
                    "dataset": dataset,
                    "image": image,
                    "codec": "Balle",
                    "param": "lambda=0.005",
                    "bpp": 0.18,
                    "psnr": 29.0 if image == "s1" else 35.0,
                    "ssimulacra2": 58.0 if image == "s1" else 81.0,
                    "energy_per_image_j": 9.0,
                    "width": width,
                    "height": height,
                },
            ]
        )
    return rows


def _diagnostic_inputs(tmp_path: Path) -> Dict[str, Path]:
    rde_csv = tmp_path / "rde.csv"
    _write_csv(rde_csv, _rde_rows())

    decisions = tmp_path / "operational_regime_decisions.csv"
    _write_csv(
        decisions,
        [
            {
                "image_id": "small::s1",
                "dataset": "small",
                "protocol": "loio",
                "regime": "bandwidth_limited",
                "policy": "robust_global_full_pool_baseline",
                "quality_floor": "30",
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "selected_family": "classical",
                "oracle_codec": "JXL",
                "oracle_config": "d=1.0",
                "oracle_family": "classical",
                "selected_cost": "0.50",
                "oracle_cost": "0.40",
                "regret": "0.10",
                "selected_quality": "34.0",
                "quality_violation": "False",
            },
            {
                "image_id": "small::s1",
                "dataset": "small",
                "protocol": "loio",
                "regime": "bandwidth_limited",
                "policy": "metadata_plus_system_full_pool",
                "quality_floor": "30",
                "selected_codec": "Balle",
                "selected_config": "lambda=0.005",
                "selected_family": "neural",
                "oracle_codec": "JXL",
                "oracle_config": "d=1.0",
                "oracle_family": "classical",
                "selected_cost": "0.30",
                "oracle_cost": "0.40",
                "regret": "-0.10",
                "selected_quality": "29.0",
                "quality_violation": "True",
            },
        ],
    )

    summary = tmp_path / "operational_regime_summary.csv"
    _write_csv(
        summary,
        [
            {
                "regime": "bandwidth_limited",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "mean_regret": "-0.10",
            }
        ],
    )

    rate_pressure = tmp_path / "operational_regime_rate_pressure_sweep.csv"
    _write_csv(
        rate_pressure,
        [
            {
                "rate_weight": "0.1",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "selected_family": "neural",
                "selection_rate": "0.0",
                "oracle_neural_rate": "0.0",
            },
            {
                "rate_weight": "0.2",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "selected_family": "neural",
                "selection_rate": "0.5",
                "oracle_neural_rate": "0.25",
            },
        ],
    )

    winners = tmp_path / "operational_regime_winner_distribution.csv"
    _write_csv(
        winners,
        [
            {
                "regime": "normal",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "selected_family": "classical",
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "count": "1",
                "selection_rate": "1.0",
            },
            {
                "regime": "normal",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "lodo",
                "quality_floor": "30",
                "selected_family": "classical",
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "count": "1",
                "selection_rate": "1.0",
            },
        ],
    )

    return {
        "rde": rde_csv,
        "decisions": decisions,
        "summary": summary,
        "rate_pressure": rate_pressure,
        "winners": winners,
        "out_dir": tmp_path / "out",
        "report": tmp_path / "out" / "operational_regime_diagnostic_report.json",
    }


def _run_diagnostics(tmp_path: Path) -> Dict[str, Path]:
    paths = _diagnostic_inputs(tmp_path)
    ordx.main(
        [
            "--rde-csv",
            str(paths["rde"]),
            "--operational-summary-csv",
            str(paths["summary"]),
            "--operational-decisions-csv",
            str(paths["decisions"]),
            "--rate-pressure-csv",
            str(paths["rate_pressure"]),
            "--winner-distribution-csv",
            str(paths["winners"]),
            "--out-dir",
            str(paths["out_dir"]),
            "--out-json",
            str(paths["report"]),
            "--quality-metrics",
            "psnr,ssimulacra2",
            "--quality-floors-psnr",
            "30",
            "--quality-floors-ssimulacra2",
            "60,70,80",
            "--k",
            "1",
        ]
    )
    return paths


def test_module_importable():
    assert hasattr(ordx, "main")
    assert hasattr(ordx, "run_diagnostics")


def test_cli_help_exit_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.router.analysis.operational_regime_diagnostics",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "--operational-decisions-csv" in result.stdout


def test_outputs_and_recommended_options_written(tmp_path):
    paths = _run_diagnostics(tmp_path)
    expected = [
        "negative_regret_rows.csv",
        "quality_violation_rows.csv",
        "metric_comparison_summary.csv",
        "objective_consistency_summary.csv",
        "rate_pressure_transition_diagnostics.csv",
        "winner_distribution_duplicate_diagnostics.csv",
    ]
    for name in expected:
        assert (paths["out_dir"] / name).exists()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["provenance"]["read_only"] is True
    assert report["provenance"]["operational_regime_simulation_behavior_changed"] is False
    assert report["recommended_fix_options"]


def test_negative_regret_and_quality_violation_rows_identified(tmp_path):
    paths = _run_diagnostics(tmp_path)
    negatives = _read_csv(paths["out_dir"] / "negative_regret_rows.csv")
    violations = _read_csv(paths["out_dir"] / "quality_violation_rows.csv")
    assert len(negatives) == 1
    assert negatives[0]["reason_hypothesis"] in {
        "quality_gate_mismatch",
        "candidate_pool_mismatch",
        "possible_policy_or_oracle_system_penalty_mismatch",
        "unknown",
    }
    assert len(violations) == 1
    assert violations[0]["violation_type"] in {
        "metric_mismatch",
        "floor_not_used_as_preselection_gate_due_to_no_leakage",
        "predictor_selected_low_quality_target",
    }


def test_objective_mismatch_classified_possible(tmp_path):
    paths = _run_diagnostics(tmp_path)
    rows = _read_csv(paths["out_dir"] / "objective_consistency_summary.csv")
    assert any("possible_" in row["finding"] for row in rows)


def test_metric_comparison_has_psnr_and_ssimulacra2(tmp_path):
    paths = _run_diagnostics(tmp_path)
    rows = _read_csv(paths["out_dir"] / "metric_comparison_summary.csv")
    metrics = {row["quality_metric"] for row in rows}
    assert {"psnr", "ssimulacra2"}.issubset(metrics)


def test_rate_pressure_finds_first_neural_weight(tmp_path):
    paths = _run_diagnostics(tmp_path)
    rows = _read_csv(paths["out_dir"] / "rate_pressure_transition_diagnostics.csv")
    target = rows[0]
    assert target["first_predicted_neural_rate_weight"] == "0.2"
    assert target["first_oracle_neural_rate_weight"] == "0.2"


def test_duplicate_diagnostic_distinguishes_real_and_apparent(tmp_path):
    paths = _run_diagnostics(tmp_path)
    rows = _read_csv(paths["out_dir"] / "winner_distribution_duplicate_diagnostics.csv")
    by_type = {row["key_type"]: row for row in rows}
    assert by_type["full_expected_key"]["duplicate_count"] == "0"
    assert int(by_type["displayed_truncated_key"]["duplicate_count"]) > 0


def test_report_wording_is_prudent(tmp_path):
    paths = _run_diagnostics(tmp_path)
    text = paths["report"].read_text(encoding="utf-8").lower()
    assert "prov" + "es" not in text
    assert "best " + "possible" not in text
    assert "neural codecs are " + "better" not in text
    assert "classical codecs are " + "obsolete" not in text
