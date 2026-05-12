import csv
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.feedback_calibration_proposal import (
    build_feedback_calibration_proposal,
    main,
    write_feedback_calibration_proposal,
)
from src.router.feedback_logger import FEEDBACK_FIELDS


def _tmp_dir(name: str) -> Path:
    root = Path(__file__).with_name("_tmp") / "feedback_proposal" / f"{name}_{uuid4().hex}"
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


def test_empty_feedback_generates_valid_empty_proposal():
    root = _tmp_dir("empty")
    feedback = root / "feedback.csv"
    _write_feedback(feedback, [])

    report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=3,
    )

    assert report["mode"] == "shadow_proposal_only"
    assert report["applied_by_router"] is False
    assert report["global"]["num_rows"] == 0
    assert report["global"]["num_groups"] == 0
    assert report["proposals"] == []


def test_rate_and_time_scales_are_computed():
    root = _tmp_dir("rate_time")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_rate": "1.0",
                "actual_rate": "1.2",
                "predicted_time_ms": "100",
                "actual_time_ms": "50",
            },
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_rate": "2.0",
                "actual_rate": "1.6",
                "predicted_time_ms": "100",
                "actual_time_ms": "200",
            },
        ],
    )

    report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=2,
    )

    proposal = report["proposals"][0]
    assert proposal["rate_scale"] == pytest.approx(1.0)
    assert proposal["time_scale"] == pytest.approx(1.25)
    assert proposal["rate_confidence"] == "low"
    assert proposal["time_confidence"] == "low"
    assert proposal["usable_for_rate"] is True
    assert proposal["usable_for_time"] is True


def test_energy_scale_requires_usable_total_energy():
    root = _tmp_dir("energy_total")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_energy_j": "12.0",
                "energy_scope": "cpu",
                "energy_usable_for_total": "true",
            },
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_energy_j": "1.0",
                "energy_scope": "gpu",
                "energy_usable_for_total": "false",
            },
        ],
    )

    report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=1,
    )

    proposal = report["proposals"][0]
    assert proposal["energy_scale"] == pytest.approx(1.2)
    assert proposal["energy_confidence"] == "low"
    assert proposal["usable_for_energy"] is True


def test_gpu_only_energy_is_ignored():
    root = _tmp_dir("gpu_only")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JXL",
                "selected_config": "effort=7",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_energy_j": "3.0",
                "local_gpu_energy_j": "3.0",
                "energy_scope": "gpu",
                "energy_is_measured": "true",
                "energy_usable_for_total": "false",
            }
        ],
    )

    report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=1,
    )

    proposal = report["proposals"][0]
    assert proposal["energy_scale"] is None
    assert proposal["energy_confidence"] == "insufficient"
    assert proposal["usable_for_energy"] is False


def test_min_samples_controls_confidence():
    root = _tmp_dir("min_samples")
    feedback = root / "feedback.csv"
    rows = [
        {
            "selected_codec": "HEVC",
            "selected_config": "crf=28",
            "execution_success": "true",
            "predicted_rate": "1.0",
            "actual_rate": "1.0",
        },
        {
            "selected_codec": "HEVC",
            "selected_config": "crf=28",
            "execution_success": "true",
            "predicted_rate": "2.0",
            "actual_rate": "2.0",
        },
    ]
    _write_feedback(feedback, rows)

    strict_report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=3,
    )
    loose_report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=2,
    )

    assert strict_report["proposals"][0]["rate_confidence"] == "insufficient"
    assert strict_report["proposals"][0]["usable_for_rate"] is False
    assert loose_report["proposals"][0]["rate_confidence"] == "low"
    assert loose_report["proposals"][0]["usable_for_rate"] is True


def test_out_of_bounds_scale_is_not_usable():
    root = _tmp_dir("bounds")
    feedback = root / "feedback.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=95",
                "execution_success": "true",
                "predicted_rate": "1.0",
                "actual_rate": "20.0",
            }
        ],
    )

    report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=1,
    )

    proposal = report["proposals"][0]
    assert proposal["rate_scale"] == pytest.approx(20.0)
    assert proposal["rate_confidence"] == "low"
    assert proposal["usable_for_rate"] is False
    assert proposal["warnings"] == ["rate_scale_out_of_bounds"]


def test_cli_writes_json_and_csv():
    root = _tmp_dir("cli")
    feedback = root / "feedback.csv"
    out = root / "proposal.json"
    summary_out = root / "proposal.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_rate": "1.0",
                "actual_rate": "1.1",
            }
        ],
    )

    main(
        [
            "--feedback",
            str(feedback),
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
            "--min-samples",
            "1",
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    rows = _read_csv(summary_out)

    assert report["mode"] == "shadow_proposal_only"
    assert report["applied_by_router"] is False
    assert report["proposals"][0]["codec"] == "JPEG"
    assert rows[0]["codec"] == "JPEG"
    assert rows[0]["config"] == "q=85"


def test_writer_writes_json_and_csv():
    root = _tmp_dir("writer")
    feedback = root / "feedback.csv"
    out = root / "proposal.json"
    summary_out = root / "proposal.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
            }
        ],
    )

    write_feedback_calibration_proposal(
        feedback=feedback,
        out=out,
        summary_out=summary_out,
        min_samples=1,
    )

    assert out.exists()
    assert summary_out.exists()
