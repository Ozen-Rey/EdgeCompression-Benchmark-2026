import csv
import json
import math
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.observability.feedback_logger import FEEDBACK_FIELDS
from src.router.observability.feedback_proposal_validation import (
    main,
    validate_feedback_proposals,
)


def _tmp_dir(name: str) -> Path:
    root = Path(__file__).with_name("_tmp") / "feedback_validation" / f"{name}_{uuid4().hex}"
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


def _write_proposal(path: Path, proposals: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "0.14.0",
        "mode": "shadow_proposal_only",
        "applied_by_router": False,
        "global": {
            "num_groups": len(proposals),
        },
        "proposals": proposals,
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _result(report: dict[str, object], codec: str, config: str, axis: str) -> dict[str, object]:
    for item in report["results"]:
        if (
            item["codec"] == codec
            and item["config"] == config
            and item["axis"] == axis
        ):
            return item
    raise AssertionError(f"Missing result for {codec} {config} {axis}")


def test_empty_inputs_generate_valid_empty_validation():
    root = _tmp_dir("empty")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    _write_feedback(feedback, [])
    _write_proposal(proposal, [])

    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    assert report["mode"] == "offline_validation_only"
    assert report["applied_by_router"] is False
    assert report["global"]["num_feedback_rows"] == 0
    assert report["global"]["num_proposal_groups"] == 0
    assert report["results"] == []


def test_time_scale_improves_log_error():
    root = _tmp_dir("time_improves")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_time_ms": "100",
                "actual_time_ms": "200",
            }
        ],
    )
    _write_proposal(
        proposal,
        [
            {
                "codec": "JPEG",
                "config": "q=85",
                "time_scale": 2.0,
                "usable_for_time": True,
            }
        ],
    )

    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    time = _result(report, "JPEG", "q=85", "time")
    assert time["num_eval_rows"] == 1
    assert time["mean_abs_log_error_before"] == pytest.approx(math.log(2.0))
    assert time["mean_abs_log_error_after"] == pytest.approx(0.0)
    assert time["improvement_ratio"] == pytest.approx(1.0)
    assert time["improved"] is True


def test_bad_scale_can_worsen_error():
    root = _tmp_dir("bad_scale")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=95",
                "execution_success": "true",
                "predicted_rate": "100",
                "actual_rate": "200",
            }
        ],
    )
    _write_proposal(
        proposal,
        [
            {
                "codec": "JPEG",
                "config": "q=95",
                "rate_scale": 8.0,
                "usable_for_rate": True,
            }
        ],
    )

    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    rate = _result(report, "JPEG", "q=95", "rate")
    assert rate["mean_abs_log_error_before"] == pytest.approx(math.log(2.0))
    assert rate["mean_abs_log_error_after"] == pytest.approx(abs(math.log(0.25)))
    assert rate["improvement_ratio"] < 0.0
    assert rate["improved"] is False


def test_missing_proposal_is_reported():
    root = _tmp_dir("missing")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JXL",
                "selected_config": "d=1.0",
                "execution_success": "true",
                "predicted_rate": "1.0",
                "actual_rate": "1.2",
            }
        ],
    )
    _write_proposal(proposal, [])

    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    rate = _result(report, "JXL", "d=1.0", "rate")
    assert rate["num_eval_rows"] == 1
    assert rate["usable_proposal"] is False
    assert rate["mean_abs_log_error_after"] is None
    assert rate["warnings"] == ["missing_proposal"]


def test_unusable_proposal_is_not_applied():
    root = _tmp_dir("unusable")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "HEVC",
                "selected_config": "crf=15",
                "execution_success": "true",
                "predicted_rate": "1.0",
                "actual_rate": "1.4",
            }
        ],
    )
    _write_proposal(
        proposal,
        [
            {
                "codec": "HEVC",
                "config": "crf=15",
                "rate_scale": 1.4,
                "usable_for_rate": False,
            }
        ],
    )

    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    rate = _result(report, "HEVC", "crf=15", "rate")
    assert rate["scale"] == pytest.approx(1.4)
    assert rate["usable_proposal"] is False
    assert rate["mean_abs_log_error_after"] is None
    assert rate["warnings"] == ["proposal_not_usable"]


def test_energy_validation_requires_total_usable_energy():
    root = _tmp_dir("energy_total")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_energy_j": "20.0",
                "energy_scope": "cpu",
                "energy_usable_for_total": "true",
            },
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_energy_j": "40.0",
                "energy_scope": "gpu",
                "energy_usable_for_total": "false",
            },
        ],
    )
    _write_proposal(
        proposal,
        [
            {
                "codec": "JPEG",
                "config": "q=85",
                "energy_scale": 2.0,
                "usable_for_energy": True,
            }
        ],
    )

    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    energy = _result(report, "JPEG", "q=85", "energy")
    assert energy["num_eval_rows"] == 1
    assert energy["mean_abs_log_error_before"] == pytest.approx(math.log(2.0))
    assert energy["mean_abs_log_error_after"] == pytest.approx(0.0)
    assert energy["improved"] is True


def test_gpu_only_energy_is_ignored():
    root = _tmp_dir("gpu_only")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JXL",
                "selected_config": "d=1.0",
                "execution_success": "true",
                "predicted_energy": "10.0",
                "local_energy_j": "2.0",
                "local_gpu_energy_j": "2.0",
                "energy_scope": "gpu",
                "energy_is_measured": "true",
                "energy_usable_for_total": "false",
            }
        ],
    )
    _write_proposal(
        proposal,
        [
            {
                "codec": "JXL",
                "config": "d=1.0",
                "energy_scale": 0.2,
                "usable_for_energy": True,
            }
        ],
    )

    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    energy = _result(report, "JXL", "d=1.0", "energy")
    assert energy["num_eval_rows"] == 0
    assert energy["mean_abs_log_error_before"] is None
    assert energy["mean_abs_log_error_after"] is None
    assert energy["improved"] is False


def test_cli_writes_json_and_csv():
    root = _tmp_dir("cli")
    feedback = root / "feedback.csv"
    proposal = root / "proposal.json"
    out = root / "validation.json"
    summary_out = root / "validation.csv"
    _write_feedback(
        feedback,
        [
            {
                "selected_codec": "JPEG",
                "selected_config": "q=85",
                "execution_success": "true",
                "predicted_time_ms": "100",
                "actual_time_ms": "100",
            }
        ],
    )
    _write_proposal(
        proposal,
        [
            {
                "codec": "JPEG",
                "config": "q=85",
                "time_scale": 1.0,
                "usable_for_time": True,
            }
        ],
    )

    main(
        [
            "--feedback",
            str(feedback),
            "--proposal",
            str(proposal),
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    rows = _read_csv(summary_out)

    assert report["mode"] == "offline_validation_only"
    assert report["applied_by_router"] is False
    assert out.exists()
    assert summary_out.exists()
    assert any(row["axis"] == "time" for row in rows)
