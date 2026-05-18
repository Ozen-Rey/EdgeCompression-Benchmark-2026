"""Tests for ``src.router.analysis.policy_comparison``.

Covers: importability, deterministic bootstrap, aggregate statistics on a
mini synthetic fixture, preservation of null/unavailable provenance when
a policy decisions CSV is missing optional columns, and the CLI --help.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.router.analysis import policy_comparison
from src.router.analysis.policy_comparison import (
    bootstrap_pair_ci,
    build_baseline_result,
    build_oracle_result,
    build_policy_result,
    load_decisions_csv,
    load_oracle_by_image,
)
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "policy_comparison"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _oracle_by_image_fixture() -> List[Dict[str, Any]]:
    return [
        {
            "image_id": f"A::{i}",
            "dataset": "A",
            "image": f"img{i}",
            "regret": regret,
            "global_feasible": "True",
            "oracle_cost": 0.10,
            "global_cost": 0.10 + regret,
        }
        for i, regret in enumerate(
            [0.10, 0.05, 0.20, 0.00, 0.15, 0.08], start=1
        )
    ]


def _metadata_decisions_fixture() -> List[Dict[str, Any]]:
    """Six images aligned with the oracle fixture; policy strictly beats baseline."""
    baseline = [0.10, 0.05, 0.20, 0.00, 0.15, 0.08]
    policy = [0.02, 0.01, 0.05, 0.00, 0.03, 0.02]
    rows: List[Dict[str, Any]] = []
    for i, (b, p) in enumerate(zip(baseline, policy), start=1):
        rows.append(
            {
                "image_id": f"A::{i}",
                "regret": p,
                "global_regret": b,
                "selected_feasible": "True",
                "fallback_used": "False",
                "correct_oracle_match": "True" if p == 0.0 else "False",
            }
        )
    return rows


def test_module_is_importable():
    assert hasattr(policy_comparison, "main")
    assert hasattr(policy_comparison, "bootstrap_pair_ci")


def test_load_oracle_by_image_parses_basic_fixture():
    path = _tmp_path("load_oracle_basic.csv")
    _write_csv(path, _oracle_by_image_fixture())

    rows = load_oracle_by_image(str(path))
    assert len(rows) == 6
    assert rows[0]["regret"] == pytest.approx(0.10)
    assert rows[0]["global_feasible"] is True


def test_baseline_result_aggregates_match_hand_computed_values():
    oracle = load_oracle_by_image_from_fixture()

    result = build_baseline_result(oracle, iterations=200, seed=7)

    assert result["policy_name"] == "robust_global_baseline"
    assert result["num_images"] == 6
    # Hand-computed on [0.10, 0.05, 0.20, 0.00, 0.15, 0.08]
    assert result["mean_regret"] == pytest.approx(0.58 / 6)
    assert result["median_regret"] == pytest.approx((0.08 + 0.10) / 2)
    assert result["max_regret"] == pytest.approx(0.20)
    assert result["relative_reduction_vs_global"] == 0.0
    assert result["coverage_rate"] == pytest.approx(1.0)
    assert result["quality_violations"] == pytest.approx(0.0)
    assert result["fallback_rate"] is None
    assert result["provenance"]["fallback_rate"] == "not_applicable"
    assert result["mean_regret_ci_low"] is not None
    assert result["mean_regret_ci_high"] is not None
    assert (
        result["mean_regret_ci_low"]
        <= result["mean_regret"]
        <= result["mean_regret_ci_high"]
    )
    # Relative reduction CI is degenerate at zero for the self-reference row.
    assert result["relative_reduction_ci_low"] == 0.0
    assert result["relative_reduction_ci_high"] == 0.0


def load_oracle_by_image_from_fixture() -> List[Dict[str, Any]]:
    path = _tmp_path("oracle_by_image_for_baseline.csv")
    _write_csv(path, _oracle_by_image_fixture())
    return load_oracle_by_image(str(path))


def test_oracle_result_is_degenerate_zero_with_full_relative_reduction():
    oracle = load_oracle_by_image_from_fixture()
    result = build_oracle_result(oracle)

    assert result["policy_name"] == "per_image_oracle"
    assert result["num_images"] == 6
    assert result["mean_regret"] == 0.0
    assert result["median_regret"] == 0.0
    assert result["p90_regret"] == 0.0
    assert result["max_regret"] == 0.0
    assert result["relative_reduction_vs_global"] == 1.0
    assert result["mean_regret_ci_low"] == 0.0
    assert result["mean_regret_ci_high"] == 0.0
    assert result["relative_reduction_ci_low"] == 1.0
    assert result["relative_reduction_ci_high"] == 1.0
    assert result["quality_violations"] == 0.0
    assert result["coverage_rate"] == 1.0


def test_policy_result_paired_bootstrap_and_aggregates():
    decisions_path = _tmp_path("metadata_decisions.csv")
    _write_csv(decisions_path, _metadata_decisions_fixture())
    decisions = load_decisions_csv(str(decisions_path))

    result = build_policy_result(
        decisions,
        policy_name="source_aware_metadata_majority",
        protocol="leave-one-out",
        iterations=500,
        seed=42,
    )

    # Hand-computed on policy regrets [0.02, 0.01, 0.05, 0.00, 0.03, 0.02]
    assert result["num_images"] == 6
    assert result["mean_regret"] == pytest.approx(0.13 / 6)
    assert result["max_regret"] == pytest.approx(0.05)
    assert result["fallback_rate"] == pytest.approx(0.0)
    assert result["coverage_rate"] == pytest.approx(1.0)
    # mean baseline = 0.58/6, mean policy = 0.13/6, reduction = 1 - 13/58
    expected_rel = (0.58 - 0.13) / 0.58
    assert result["relative_reduction_vs_global"] == pytest.approx(expected_rel)
    assert (
        result["mean_regret_ci_low"]
        <= result["mean_regret"]
        <= result["mean_regret_ci_high"]
    )
    assert (
        result["relative_reduction_ci_low"]
        <= result["relative_reduction_vs_global"]
        <= result["relative_reduction_ci_high"]
    )


def test_bootstrap_is_deterministic_for_same_seed():
    policy = [0.02, 0.01, 0.05, 0.00, 0.03, 0.02]
    baseline = [0.10, 0.05, 0.20, 0.00, 0.15, 0.08]

    a = bootstrap_pair_ci(
        policy_regrets=policy,
        baseline_regrets=baseline,
        iterations=300,
        seed=2026,
    )
    b = bootstrap_pair_ci(
        policy_regrets=policy,
        baseline_regrets=baseline,
        iterations=300,
        seed=2026,
    )
    assert a == b

    c = bootstrap_pair_ci(
        policy_regrets=policy,
        baseline_regrets=baseline,
        iterations=300,
        seed=2027,
    )
    assert a != c


def test_missing_fallback_column_is_preserved_as_unavailable():
    rows = [
        {
            "image_id": f"A::{i}",
            "regret": p,
            "global_regret": b,
            "selected_feasible": "True",
            # fallback_used intentionally omitted
        }
        for i, (p, b) in enumerate(
            zip(
                [0.02, 0.01, 0.05, 0.00, 0.03, 0.02],
                [0.10, 0.05, 0.20, 0.00, 0.15, 0.08],
            ),
            start=1,
        )
    ]
    path = _tmp_path("decisions_no_fallback.csv")
    _write_csv(path, rows)

    loaded = load_decisions_csv(str(path))
    result = build_policy_result(
        loaded,
        policy_name="custom_policy",
        protocol="custom",
        iterations=50,
        seed=1,
    )

    assert result["fallback_rate"] is None
    assert result["provenance"].get("fallback_rate") == "unavailable"
    # mean_regret must still be a real number; the missing column does
    # not corrupt the regret distribution.
    assert result["mean_regret"] is not None


def test_end_to_end_emits_csv_and_json_with_expected_policies(monkeypatch):
    oracle_path = _tmp_path("oracle_by_image_e2e.csv")
    decisions_path = _tmp_path("metadata_decisions_e2e.csv")
    _write_csv(oracle_path, _oracle_by_image_fixture())
    _write_csv(decisions_path, _metadata_decisions_fixture())

    out_csv = _tmp_path("out_e2e.csv")
    out_json = _tmp_path("out_e2e.json")

    argv = [
        "--oracle-by-image",
        str(oracle_path),
        "--metadata-policy-decisions",
        str(decisions_path),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--bootstrap-iterations",
        "100",
        "--seed",
        "5",
    ]
    policy_comparison.main(argv)

    assert out_csv.is_file()
    assert out_json.is_file()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    names = [row["policy_name"] for row in rows]
    assert names == [
        "robust_global_baseline",
        "source_aware_metadata_majority",
        "per_image_oracle",
    ]

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["bootstrap_iterations"] == 100
    assert payload["seed"] == 5
    assert "inputs" in payload
    assert payload["inputs"]["oracle_by_image"] == str(oracle_path)
    assert any(
        p["policy_name"] == "robust_global_baseline" for p in payload["policies"]
    )


def test_baseline_and_oracle_only_when_no_optional_inputs():
    oracle_path = _tmp_path("oracle_by_image_only.csv")
    _write_csv(oracle_path, _oracle_by_image_fixture())

    out_csv = _tmp_path("out_baseline_only.csv")
    out_json = _tmp_path("out_baseline_only.json")

    argv = [
        "--oracle-by-image",
        str(oracle_path),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--bootstrap-iterations",
        "50",
        "--seed",
        "11",
    ]
    policy_comparison.main(argv)

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    names = [row["policy_name"] for row in rows]
    assert names == ["robust_global_baseline", "per_image_oracle"]


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "src.router.analysis.policy_comparison", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "policy" in result.stdout.lower()
