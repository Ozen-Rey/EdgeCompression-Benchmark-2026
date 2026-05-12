import csv
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.feedback_analysis import analyze_feedback
from src.router.feedback_logger import FEEDBACK_FIELDS


def _tmp_dir(name: str) -> Path:
    root = Path(__file__).with_name("_tmp") / "feedback_analysis" / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_feedback(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in FEEDBACK_FIELDS
        }
    )
    fieldnames = FEEDBACK_FIELDS + extra_fields

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_feedback_analysis_empty_csv():
    root = _tmp_dir("empty")
    feedback = root / "feedback.csv"
    _write_feedback(feedback, [])

    report = analyze_feedback(
        feedback=feedback,
        out_dir=root / "analysis",
    )

    assert report["summary"]["num_rows"] == 0
    assert report["summary"]["num_executed"] == 0
    assert report["summary"]["success_rate"] == 0.0
    assert report["summary"]["rate_abs_error_mean"] is None
    assert _read_csv(root / "analysis" / "feedback_by_codec.csv") == []


def test_feedback_analysis_success_and_failure_counts():
    root = _tmp_dir("counts")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "execution_requested": "true",
                "execution_success": "true",
                "output_exists": "true",
                "output_nonempty": "true",
            },
            {
                "selected_codec": "JXL",
                "execution_requested": "1",
                "execution_success": "False",
                "output_exists": "false",
                "output_nonempty": "0",
                "error": "encoder failed",
            },
        ],
    )

    report = analyze_feedback(
        feedback=feedback,
        out_dir=root / "analysis",
    )

    assert report["summary"]["num_rows"] == 2
    assert report["summary"]["num_executed"] == 2
    assert report["summary"]["num_success"] == 1
    assert report["summary"]["num_failed"] == 1
    assert report["summary"]["success_rate"] == 0.5

    errors = _read_csv(root / "analysis" / "feedback_errors.csv")
    assert len(errors) == 1
    assert errors[0]["selected_codec"] == "JXL"
    assert errors[0]["error"] == "encoder failed"


def test_feedback_analysis_rate_and_time_errors():
    root = _tmp_dir("rate_time")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "execution_requested": "true",
                "execution_success": "true",
                "predicted_rate": "1.0",
                "actual_rate": "1.2",
                "predicted_time_ms": "100",
                "actual_time_ms": "150",
            },
            {
                "selected_codec": "JPEG",
                "execution_requested": "true",
                "execution_success": "true",
                "predicted_rate": "2.0",
                "actual_rate": "1.0",
                "predicted_time_ms": "200",
                "actual_time_ms": "100",
            },
        ],
    )

    report = analyze_feedback(
        feedback=feedback,
        out_dir=root / "analysis",
    )

    summary = report["summary"]
    assert summary["rate_abs_error_mean"] == pytest.approx(0.6)
    assert summary["rate_rel_error_mean"] == pytest.approx(0.35)
    assert summary["time_abs_error_ms_mean"] == pytest.approx(75.0)
    assert summary["time_rel_error_mean"] == pytest.approx(0.5)


def test_feedback_analysis_ignores_partial_energy():
    root = _tmp_dir("partial_energy")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JXL",
                "execution_requested": "true",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_gpu_energy_j": "3.0",
                "local_energy_j": "3.0",
                "energy_scope": "gpu",
                "energy_is_measured": "true",
                "energy_usable_for_total": "false",
            }
        ],
    )

    report = analyze_feedback(
        feedback=feedback,
        out_dir=root / "analysis",
    )

    assert report["summary"]["num_with_usable_total_energy"] == 0
    assert report["summary"]["energy_abs_error_mean"] is None
    assert report["summary"]["energy_rel_error_mean"] is None
    assert report["by_codec"][0]["num_usable_energy"] == 0
    assert report["by_codec"][0]["mean_energy_rel_error"] is None


def test_feedback_analysis_uses_only_usable_total_energy():
    root = _tmp_dir("usable_energy")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "execution_requested": "true",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_energy_j": "12.0",
                "energy_scope": "cpu",
                "energy_usable_for_total": "true",
            },
            {
                "selected_codec": "JPEG",
                "execution_requested": "true",
                "execution_success": "true",
                "predicted_energy": "100.0",
                "local_energy_j": "1.0",
                "energy_scope": "gpu",
                "energy_usable_for_total": "false",
            },
        ],
    )

    report = analyze_feedback(
        feedback=feedback,
        out_dir=root / "analysis",
    )

    assert report["summary"]["num_with_usable_total_energy"] == 1
    assert report["summary"]["energy_abs_error_mean"] == pytest.approx(2.0)
    assert report["summary"]["energy_rel_error_mean"] == pytest.approx(0.2)
    assert report["by_codec"][0]["mean_local_energy_j"] == pytest.approx(12.0)


def test_feedback_analysis_writes_expected_outputs():
    root = _tmp_dir("outputs")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "HEVC",
                "execution_requested": "true",
                "execution_success": "true",
                "predicted_rate": "2.0",
                "actual_rate": "2.5",
            }
        ],
    )

    analyze_feedback(
        feedback=feedback,
        out_dir=root / "analysis",
    )

    expected = {
        "feedback_summary.json",
        "feedback_summary.csv",
        "feedback_by_codec.csv",
        "feedback_errors.csv",
    }
    assert expected == {p.name for p in (root / "analysis").iterdir()}

    summary = json.loads(
        (root / "analysis" / "feedback_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["semantics"]["read_only"] is True
    assert summary["semantics"]["router_decision_impact"] == "none"
