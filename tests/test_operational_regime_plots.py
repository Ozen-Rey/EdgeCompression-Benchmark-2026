"""Tests for operational-regime plot rendering from CSV artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.router.analysis import operational_regime_plots as plots


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_fixture(input_dir: Path, *, tradeoff_rows: int = 5) -> None:
    _write_csv(
        input_dir / "operational_regime_plot_data.csv",
        [
            {
                "regime": "normal",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "x_energy_saving": "0.1",
                "y_regret_reduction": "0.2",
                "neural_selection_rate": "0.25",
                "quality_violation_rate": "0.0",
            }
        ],
    )
    _write_csv(
        input_dir / "operational_regime_winner_distribution.csv",
        [
            {
                "regime": "normal",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "selected_family": "classical",
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "selection_rate": "1.0",
            }
        ],
    )
    _write_csv(
        input_dir / "operational_regime_oracle_vs_prediction.csv",
        [
            {
                "regime": "normal",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "family": "neural",
                "oracle_rate": "0.5",
                "predicted_rate": "0.25",
            }
        ],
    )
    _write_csv(
        input_dir / "operational_regime_family_confusion.csv",
        [
            {
                "regime": "normal",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "oracle_family": "classical",
                "predicted_family": "classical",
                "rate": "1.0",
            }
        ],
    )
    _write_csv(
        input_dir / "operational_regime_rate_pressure_sweep.csv",
        [
            {
                "rate_weight": "0.1",
                "policy": "metadata_plus_system_full_pool",
                "protocol": "loio",
                "quality_floor": "30",
                "selected_family": "classical",
                "selection_rate": "1.0",
                "oracle_neural_rate": "0.0",
                "neural_selection_rate": "0.0",
                "top_selected_codec": "JPEG",
                "top_selected_codec_rate": "1.0",
                "rate_reduction_vs_baseline": "0.1",
                "energy_delta_vs_baseline": "0.2",
            }
        ],
    )
    _write_csv(
        input_dir / "operational_regime_switch_summary.csv",
        [
            {
                "regime": "rate_pressure",
                "protocol": "loio",
                "quality_floor": "30",
                "rate_weight": "0.1",
                "neural_win_rate": "0.0",
            }
        ],
    )
    _write_csv(
        input_dir / "switch_reason_by_rate_weight.csv",
        [
            {
                "protocol": "loio",
                "quality_floor": "30",
                "rate_weight": "0.1",
                "switch_reason": "classical_sufficient",
                "count": "1",
                "share": "1.0",
            }
        ],
    )
    _write_csv(
        input_dir / "neural_vs_classic_tradeoff_scatter.csv",
        [
            {
                "image_id": f"img{i}",
                "bitrate_reduction_neural_vs_classic": str(0.1 * i),
                "delta_energy_neural_minus_classic": str(0.2 * i),
                "delta_quality_neural_minus_classic": "0.5",
                "winner_family": "classical",
                "switch_reason": "classical_sufficient",
            }
            for i in range(tradeoff_rows)
        ],
    )
    _write_csv(
        input_dir / "quality_floor_switch_summary.csv",
        [
            {
                "quality_metric": "psnr",
                "quality_floor": "30",
                "protocol": "loio",
                "neural_necessary_rate": "0.0",
                "neural_win_rate": "0.0",
            }
        ],
    )


def _read_report(out_dir: Path) -> Dict[str, Any]:
    return json.loads((out_dir / "operational_regime_plot_report.json").read_text(encoding="utf-8"))


def _matplotlib_unavailable(report: Dict[str, Any]) -> bool:
    return bool(report["skipped_plots"]) and all(
        row.get("reason") == "matplotlib_unavailable"
        for row in report["skipped_plots"]
    )


def test_module_importable():
    assert hasattr(plots, "main")
    assert hasattr(plots, "generate_plots")


def test_cli_help_exit_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.router.analysis.operational_regime_plots",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "--plot-mode" in result.stdout


def test_basic_plotter_writes_report_and_only_basic_names(tmp_path):
    _write_fixture(tmp_path)
    out_dir = tmp_path / "plots"
    plots.generate_plots(
        input_dir=tmp_path,
        out_dir=out_dir,
        plot_mode="basic",
        max_plot_points=2000,
        seed=42,
    )
    report = _read_report(out_dir)
    assert report["plot_mode"] == "basic"
    assert set(Path(p).name for p in report["generated_plots"]).issubset(set(plots.BASIC_PLOTS))
    assert not (set(Path(p).name for p in report["generated_plots"]) & set(plots.ALL_EXTRA_PLOTS))


def test_all_plotter_includes_heavy_plot_names(tmp_path):
    _write_fixture(tmp_path)
    out_dir = tmp_path / "plots"
    plots.generate_plots(
        input_dir=tmp_path,
        out_dir=out_dir,
        plot_mode="all",
        max_plot_points=2000,
        seed=42,
    )
    report = _read_report(out_dir)
    mentioned = {
        Path(p).name for p in report["generated_plots"]
    } | {
        row["plot"] for row in report["failed_plots"]
    } | {
        row["plot"] for row in report["skipped_plots"]
    }
    assert set(plots.ALL_EXTRA_PLOTS).issubset(mentioned)


def test_plot_failure_is_reported_without_abort(tmp_path, monkeypatch):
    _write_fixture(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic plot failure")

    original = plots.PLOT_SPECS["energy_saving_vs_regret_reduction.png"]["fn"]
    monkeypatch.setitem(
        plots.PLOT_SPECS["energy_saving_vs_regret_reduction.png"],
        "fn",
        boom,
    )
    try:
        report = plots.generate_plots(
            input_dir=tmp_path,
            out_dir=tmp_path / "plots",
            plot_mode="basic",
            max_plot_points=2000,
            seed=42,
        )
    finally:
        monkeypatch.setitem(
            plots.PLOT_SPECS["energy_saving_vs_regret_reduction.png"],
            "fn",
            original,
        )
    if _matplotlib_unavailable(report):
        assert report["skipped_plots"]
        return
    assert any(row["plot"] == "energy_saving_vs_regret_reduction.png" for row in report["failed_plots"])


def test_tradeoff_scatter_sampling_is_deterministic(tmp_path):
    _write_fixture(tmp_path, tradeoff_rows=10)
    out_dir = tmp_path / "plots"
    report = plots.generate_plots(
        input_dir=tmp_path,
        out_dir=out_dir,
        plot_mode="all",
        max_plot_points=2,
        seed=7,
    )
    if _matplotlib_unavailable(report):
        assert report["skipped_plots"]
        return
    sampled = [
        row for row in report["sampled_plots"]
        if row["plot"] == "neural_vs_classic_tradeoff_scatter.png"
    ]
    assert sampled
    assert sampled[0]["plotted_rows"] == 2


def test_missing_input_csv_is_skipped_clearly(tmp_path):
    _write_fixture(tmp_path)
    (tmp_path / "operational_regime_winner_distribution.csv").unlink()
    report = plots.generate_plots(
        input_dir=tmp_path,
        out_dir=tmp_path / "plots",
        plot_mode="basic",
        max_plot_points=2000,
        seed=42,
    )
    assert any(
        row["reason"] == "missing_input_csv"
        and row["detail"] == "operational_regime_winner_distribution.csv"
        for row in report["skipped_plots"]
    )


def test_matplotlib_unavailable_is_reported(tmp_path, monkeypatch):
    _write_fixture(tmp_path)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    report = plots.generate_plots(
        input_dir=tmp_path,
        out_dir=tmp_path / "plots",
        plot_mode="basic",
        max_plot_points=2000,
        seed=42,
    )
    assert report["generated_plots"] == []
    assert all(row["reason"] == "matplotlib_unavailable" for row in report["skipped_plots"])
