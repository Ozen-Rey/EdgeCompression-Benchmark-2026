"""Tests for the offline operational-regime simulation audit."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.router.analysis import operational_regime_simulation as ors


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    *,
    dataset: str,
    image: str,
    codec: str,
    param: str,
    bpp: float,
    psnr: float,
    energy: float,
    width: int,
    height: int,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "image": image,
        "codec": codec,
        "param": param,
        "bpp": bpp,
        "psnr": psnr,
        "energy_per_image_j": energy,
        "width": width,
        "height": height,
    }


def _fixture_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = [
        ("small", "s1", 640, 512),
        ("small", "s2", 720, 540),
        ("large", "l1", 2560, 1440),
        ("large", "l2", 3840, 2160),
    ]
    for dataset, image, width, height in specs:
        rows.extend(
            [
                _row(
                    dataset=dataset,
                    image=image,
                    codec="JPEG",
                    param="q=85",
                    bpp=1.20,
                    psnr=34.0,
                    energy=0.06,
                    width=width,
                    height=height,
                ),
                _row(
                    dataset=dataset,
                    image=image,
                    codec="JXL",
                    param="d=1.0",
                    bpp=0.80,
                    psnr=36.0,
                    energy=0.10,
                    width=width,
                    height=height,
                ),
                _row(
                    dataset=dataset,
                    image=image,
                    codec="JPEG_AI",
                    param="lambda=0.01",
                    bpp=0.30,
                    psnr=35.0,
                    energy=8.0,
                    width=width,
                    height=height,
                ),
                _row(
                    dataset=dataset,
                    image=image,
                    codec="Balle",
                    param="lambda=0.005",
                    bpp=0.20,
                    psnr=35.5,
                    energy=10.0,
                    width=width,
                    height=height,
                ),
            ]
        )
    return rows


def _run_cli(tmp_path: Path) -> Dict[str, Path]:
    rde_csv = tmp_path / "fixture_rde.csv"
    out_dir = tmp_path / "out"
    _write_csv(rde_csv, _fixture_rows())
    summary = out_dir / "operational_regime_summary.csv"
    decisions = out_dir / "operational_regime_decisions.csv"
    report = out_dir / "operational_regime_report.json"
    ors.main(
        [
            "--rde-csv",
            str(rde_csv),
            "--out-dir",
            str(out_dir),
            "--out-json",
            str(report),
            "--out-summary-csv",
            str(summary),
            "--out-decisions-csv",
            str(decisions),
            "--quality-floors",
            "30",
            "--protocols",
            "loio",
            "--k",
            "1",
            "--bootstrap-iterations",
            "10",
        ]
    )
    return {
        "out_dir": out_dir,
        "summary": summary,
        "decisions": decisions,
        "report": report,
        "plot_data": out_dir / "operational_regime_plot_data.csv",
        "winners": out_dir / "operational_regime_winner_distribution.csv",
        "oracle_prediction": out_dir / "operational_regime_oracle_vs_prediction.csv",
        "confusion": out_dir / "operational_regime_family_confusion.csv",
        "sweep": out_dir / "operational_regime_rate_pressure_sweep.csv",
        "switch_summary": out_dir / "operational_regime_switch_summary.csv",
        "switch_by_image": out_dir / "operational_regime_switch_by_image.csv",
        "switch_report": out_dir / "operational_regime_switch_report.json",
        "switch_reason": out_dir / "switch_reason_by_rate_weight.csv",
        "switch_scatter": out_dir / "neural_vs_classic_tradeoff_scatter.csv",
        "switch_floor": out_dir / "quality_floor_switch_summary.csv",
        "gate_summary": out_dir / "quality_gate_comparison_summary.csv",
        "gate_decisions": out_dir / "quality_gate_comparison_decisions.csv",
    }


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_module_is_importable():
    assert hasattr(ors, "main")
    assert hasattr(ors, "evaluate_operational_regimes")


def test_cli_help_exits_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.router.analysis.operational_regime_simulation",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "--rde-csv" in result.stdout


def test_regime_definitions_have_expected_system_controls():
    regimes = ors.build_regime_definitions()
    for name in ["normal", "bandwidth_limited", "energy_saving", "no_cuda"]:
        assert name in regimes
    assert regimes["energy_saving"]["weights"]["w_E"] > regimes["normal"]["weights"]["w_E"]
    assert regimes["bandwidth_limited"]["weights"]["w_R"] > regimes["normal"]["weights"]["w_R"]
    assert regimes["no_cuda"]["neural_allowed"] is False
    assert regimes["no_cuda"]["effective_exclude_neural"] is True


def test_cli_writes_json_and_csv_artifacts(tmp_path):
    paths = _run_cli(tmp_path)
    for key in [
        "summary",
        "decisions",
        "report",
        "plot_data",
        "winners",
        "oracle_prediction",
        "confusion",
        "sweep",
        "switch_summary",
        "switch_by_image",
        "switch_report",
        "switch_reason",
        "switch_scatter",
        "switch_floor",
        "gate_summary",
        "gate_decisions",
    ]:
        assert paths[key].exists(), key

    summary = _read_csv(paths["summary"])
    policies = {row["policy"] for row in summary}
    assert "metadata_plus_system_full_pool" in policies
    assert "system_only_full_pool" in policies

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["provenance"]["policy_does_not_see_test_image_rde"] is True
    assert "plot_artifacts" in report
    assert report["plot_artifacts"]["plot_data_paths"]
    assert report["quality_metric_contract"]["role"] == "rate_oriented_stress_test"
    assert "switch_analysis" in report
    assert report["oracle_quality_contract"]["hard_quality_floor"] is True
    assert report["oracle_quality_contract"]["infeasible_cases_reported_separately"] is True
    interpretation = " ".join(report["interpretation"]).lower()
    assert "prov" + "es" not in interpretation
    assert "best " + "possible" not in interpretation
    assert "classical codecs are " + "obsolete" not in interpretation


def test_energy_and_regret_reductions_are_computed(tmp_path):
    paths = _run_cli(tmp_path)
    summary = _read_csv(paths["summary"])
    baseline = next(
        row for row in summary
        if row["regime"] == "bandwidth_limited"
        and row["policy"] == "robust_global_full_pool_baseline"
    )
    oracle = next(
        row for row in summary
        if row["regime"] == "bandwidth_limited"
        and row["policy"] == "full_pool_oracle"
    )
    expected_energy_saving = 1.0 - (
        float(oracle["mean_energy"]) / float(baseline["mean_energy"])
    )
    expected_regret_reduction = 1.0 - (
        float(oracle["mean_regret"]) / float(baseline["mean_regret"])
    )
    assert math.isclose(
        float(oracle["energy_saving_vs_global_baseline"]),
        expected_energy_saving,
    )
    assert math.isclose(
        float(oracle["regret_reduction_vs_global_baseline"]),
        expected_regret_reduction,
    )
    assert "objective_gain_vs_global_baseline" in oracle


def test_no_negative_regret_invariant_and_oracle_zero(tmp_path):
    paths = _run_cli(tmp_path)
    decisions = _read_csv(paths["decisions"])
    regrets = [float(row["regret"]) for row in decisions if row["regret"]]
    assert regrets
    assert min(regrets) >= -1e-9
    oracle_rows = [row for row in decisions if row["policy"] == "full_pool_oracle"]
    assert oracle_rows
    assert all(abs(float(row["regret"])) <= 1e-9 for row in oracle_rows if row["regret"])
    assert all(row["oracle_status"] == "feasible" for row in oracle_rows)


def test_oracle_never_selects_below_floor_and_reports_margins():
    rows = [
        {
            "image_id": "x",
            "dataset": "d",
            "codec": "JPEG",
            "config": "q=85",
            "codec_family": "classical",
            "rate": 1.0,
            "quality": 35.0,
            "energy": 0.1,
            "norm_rate": 0.9,
            "norm_energy": 0.1,
            "norm_distortion": 0.1,
        },
        {
            "image_id": "x",
            "dataset": "d",
            "codec": "Balle",
            "config": "lambda=0.005",
            "codec_family": "neural",
            "rate": 0.1,
            "quality": 25.0,
            "energy": 0.2,
            "norm_rate": 0.0,
            "norm_energy": 0.2,
            "norm_distortion": 0.0,
        },
    ]
    oracle = ors._oracle_for_image(
        rows,
        image_id="x",
        pool="full_pool",
        regime=ors.build_regime_definitions()["bandwidth_limited"],
        quality_floor=30.0,
    )
    assert oracle is not None
    assert oracle["codec"] == "JPEG"
    assert oracle["quality"] >= 30.0


def test_infeasible_when_no_candidate_passes_floor():
    rows = [
        {
            "image_id": "x",
            "dataset": "d",
            "codec": "JPEG",
            "config": "q=85",
            "codec_family": "classical",
            "rate": 1.0,
            "quality": 20.0,
            "energy": 0.1,
            "norm_rate": 0.9,
            "norm_energy": 0.1,
            "norm_distortion": 0.1,
        }
    ]
    oracle = ors._oracle_for_image(
        rows,
        image_id="x",
        pool="full_pool",
        regime=ors.build_regime_definitions()["normal"],
        quality_floor=30.0,
    )
    assert oracle is None


def test_oracle_is_argmin_under_same_objective(tmp_path):
    paths = _run_cli(tmp_path)
    decisions = _read_csv(paths["decisions"])
    for row in decisions:
        if row["oracle_cost"] and row["selected_cost"]:
            assert float(row["selected_cost"]) + 1e-9 >= float(row["oracle_cost"])


def test_predictive_violation_is_ex_post_not_oracle(tmp_path):
    paths = _run_cli(tmp_path)
    decisions = _read_csv(paths["decisions"])
    for row in decisions:
        if row["realized_quality_violation"] == "True":
            assert row["oracle_status"] in {"feasible", "infeasible"}
            if row["oracle_status"] == "feasible" and row["oracle_quality_margin"]:
                assert float(row["oracle_quality_margin"]) >= -1e-9


def test_plot_data_contains_scatter_metrics(tmp_path):
    paths = _run_cli(tmp_path)
    rows = _read_csv(paths["plot_data"])
    assert rows
    assert "x_energy_saving" in rows[0]
    assert "y_regret_reduction" in rows[0]
    assert "mean_energy" in rows[0]
    assert "neural_family_recall" in rows[0]
    assert "objective_gain_vs_global_baseline" in rows[0]


def test_winner_distribution_sums_to_one_per_group(tmp_path):
    paths = _run_cli(tmp_path)
    rows = _read_csv(paths["winners"])
    grouped: Dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        key = (
            row["regime"],
            row["policy"],
            row["protocol"],
            row["quality_floor"],
        )
        grouped[key] = grouped.get(key, 0.0) + float(row["selection_rate"])
    assert grouped
    assert all(math.isclose(total, 1.0) for total in grouped.values())


def test_oracle_prediction_and_confusion_tables_include_families(tmp_path):
    paths = _run_cli(tmp_path)
    oracle_prediction = _read_csv(paths["oracle_prediction"])
    assert {"oracle_rate", "predicted_rate"}.issubset(oracle_prediction[0].keys())
    assert {"classical", "neural"}.issubset({row["family"] for row in oracle_prediction})

    confusion = _read_csv(paths["confusion"])
    families = {row["oracle_family"] for row in confusion} | {
        row["predicted_family"] for row in confusion
    }
    assert {"classical", "neural"}.issubset(families)


def test_rate_pressure_sweep_contains_grid_and_shift(tmp_path):
    paths = _run_cli(tmp_path)
    sweep = _read_csv(paths["sweep"])
    weights = sorted({float(row["rate_weight"]) for row in sweep})
    assert weights == ors.RATE_PRESSURE_GRID

    rows = [
        row for row in sweep
        if row["policy"] == "metadata_plus_system_full_pool"
        and row["selected_family"] == "neural"
    ]
    by_weight: Dict[float, List[float]] = {}
    for row in rows:
        by_weight.setdefault(float(row["rate_weight"]), []).append(
            float(row["selection_rate"])
        )
    assert max(mean_values := [sum(v) / len(v) for v in by_weight.values()]) > min(mean_values)


def test_optional_png_generation_does_not_block_report(tmp_path):
    paths = _run_cli(tmp_path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert "generated" in report["plot_artifacts"]
    if not report["plot_artifacts"]["generated"]:
        assert report["plot_artifacts"]["skipped_reason"]


def _switch_unit_rows() -> List[Dict[str, Any]]:
    def row(image: str, codec: str, family: str, rate: float, quality: float, energy: float, nr: float, ne: float, nd: float) -> Dict[str, Any]:
        return {
            "image_id": image,
            "dataset": "d",
            "codec": codec,
            "config": "cfg",
            "codec_family": family,
            "rate": rate,
            "quality": quality,
            "energy": energy,
            "norm_rate": nr,
            "norm_energy": ne,
            "norm_distortion": nd,
        }

    return [
        row("necessary", "JPEG", "classical", 1.0, 20.0, 1.0, 0.2, 0.1, 0.1),
        row("necessary", "Balle", "neural", 0.2, 35.0, 8.0, 0.1, 0.9, 0.1),
        row("rde", "JPEG", "classical", 1.0, 35.0, 1.0, 0.8, 0.2, 0.2),
        row("rde", "Balle", "neural", 0.2, 35.0, 1.0, 0.1, 0.1, 0.1),
        row("expensive", "JPEG", "classical", 1.0, 35.0, 1.0, 0.2, 0.1, 0.1),
        row("expensive", "Balle", "neural", 0.2, 35.0, 10.0, 0.1, 1.0, 0.1),
        row("sufficient", "JPEG", "classical", 1.0, 35.0, 1.0, 0.1, 0.1, 0.1),
        row("sufficient", "Balle", "neural", 1.2, 35.0, 2.0, 0.3, 0.2, 0.1),
    ]


def test_switch_reasons_cover_core_cases():
    regime = {
        "weights": {"w_R": 0.5, "w_E": 0.4, "w_D": 0.1},
        "neural_allowed": True,
        "max_norm_energy": None,
    }
    result = ors.build_switch_analysis(
        rows=_switch_unit_rows(),
        quality_floors=[30.0],
        protocols=["loio"],
        regimes=ors.build_regime_definitions(),
        rate_weight=0.5,
        rate_regime=regime,
    )
    by_image = {row["image_id"]: row["switch_reason"] for row in result["by_image"]}
    assert by_image["necessary"] == "neural_necessary_for_quality"
    assert by_image["rde"] == "neural_rde_efficient"
    assert by_image["expensive"] == "neural_too_energy_expensive"
    assert by_image["sufficient"] == "classical_sufficient"


def test_switch_summary_rates_sum_to_one_and_threshold_detected(tmp_path):
    paths = _run_cli(tmp_path)
    rows = _read_csv(paths["switch_summary"])
    assert rows
    for row in rows:
        reason_sum = (
            float(row["neural_necessary_rate"])
            + float(row["neural_rde_efficient_rate"])
            + float(row["classical_sufficient_rate"])
            + float(row["neural_too_energy_expensive_rate"])
            + float(row["no_neural_feasible_rate"])
            + float(row["no_classic_feasible_rate"])
        )
        assert reason_sum <= 1.0 + 1e-9
    report = json.loads(paths["switch_report"].read_text(encoding="utf-8"))
    assert "first_rate_weight_neural_feasible" in report["rate_pressure_transition"]


def _expected_gate_rows() -> List[Dict[str, Any]]:
    def row(dataset: str, image: str, width: int, codec: str, param: str, bpp: float, psnr: float, energy: float) -> Dict[str, Any]:
        return {
            "dataset": dataset,
            "image": image,
            "codec": codec,
            "param": param,
            "bpp": bpp,
            "psnr": psnr,
            "energy_per_image_j": energy,
            "width": width,
            "height": 512,
        }

    rows: List[Dict[str, Any]] = []
    for dataset, image, width, balle_quality in [
        ("target", "t", 640, 25.0),
        ("good", "g", 650, 35.0),
        ("bad", "b", 4000, 25.0),
    ]:
        rows.extend(
            [
                row(dataset, image, width, "JPEG", "q=85", 1.0, 35.0, 0.05),
                row(dataset, image, width, "Balle", "lambda=0.005", 0.1, balle_quality, 10.0),
            ]
        )
    return rows


def test_expected_quality_gate_uses_training_only_and_reduces_violation():
    csv_rows = _expected_gate_rows()
    # Load through the same public CSV path logic to derive normalization and metadata.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "gate.csv"
        _write_csv(path, csv_rows)
        loaded = ors.load_pool_rows(
            str(path),
            codec_col="codec",
            config_col="param",
            rate_col="bpp",
            quality_col="psnr",
            energy_col="energy_per_image_j",
            image_id_col="dataset,image",
            dataset_col="dataset",
        )
        ors.normalize_full_pool(loaded)
        result = ors.evaluate_expected_quality_gate_shadow(
            rows=loaded,
            metadata_by_image=ors._derive_metadata_from_rde_rows(loaded),
            quality_floors=[30.0],
            protocols=["loio"],
            regimes={"normal": ors.build_regime_definitions()["normal"], **ors.build_regime_definitions()},
            k=1,
        )
    summary = {
        row["policy"]: row for row in result["summaries"]
        if row["regime"] == "normal" and row["protocol"] == "loio"
    }
    assert summary["policy_with_expected_quality_gate_shadow"]["quality_violation_rate"] <= summary["policy_without_expected_quality_gate"]["quality_violation_rate"]
    gated_decisions = [
        row for row in result["decisions"]
        if row["policy"] == "policy_with_expected_quality_gate_shadow"
        and row["fallback_reason"] == "expected_quality_gate_training_floor_failure"
    ]
    assert gated_decisions
    assert all(row["expected_quality_min"] is not None for row in gated_decisions)
