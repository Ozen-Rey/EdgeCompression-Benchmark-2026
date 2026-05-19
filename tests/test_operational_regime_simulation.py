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


def test_plot_data_contains_scatter_metrics(tmp_path):
    paths = _run_cli(tmp_path)
    rows = _read_csv(paths["plot_data"])
    assert rows
    assert "x_energy_saving" in rows[0]
    assert "y_regret_reduction" in rows[0]
    assert "mean_energy" in rows[0]
    assert "neural_family_recall" in rows[0]


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
