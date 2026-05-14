import csv
import json
from pathlib import Path
from uuid import uuid4

from src.router.observability.feedback_calibration_promotion import (
    build_feedback_calibration_promotion,
    main,
)
from tests.conftest import scratch_root


def _tmp_dir(name: str) -> Path:
    root = scratch_root() / "feedback_promotion" / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_proposal(path: Path, proposals: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "0.14.0",
                "mode": "shadow_proposal_only",
                "applied_by_router": False,
                "proposals": proposals,
            }
        ),
        encoding="utf-8",
    )


def _write_validation(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "0.15.0",
                "mode": "offline_validation_only",
                "applied_by_router": False,
                "results": results,
            }
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _base_proposal(**updates: object) -> dict[str, object]:
    out: dict[str, object] = {
        "codec": "JPEG",
        "config": "q=85",
        "rate_scale": 1.2,
        "usable_for_rate": True,
    }
    out.update(updates)
    return out


def _base_validation(**updates: object) -> dict[str, object]:
    out: dict[str, object] = {
        "codec": "JPEG",
        "config": "q=85",
        "axis": "rate",
        "num_eval_rows": 5,
        "scale": 1.2,
        "usable_proposal": True,
        "mean_abs_log_error_before": 0.4,
        "mean_abs_log_error_after": 0.2,
        "improvement_ratio": 0.5,
        "improved": True,
        "warnings": [],
    }
    out.update(updates)
    return out


def _run(
    root: Path,
    *,
    proposals: list[dict[str, object]],
    validations: list[dict[str, object]],
    min_samples: int = 3,
    min_improvement: float = 0.10,
    max_after_error: float = 0.25,
) -> dict[str, object]:
    proposal = root / "proposal.json"
    validation = root / "validation.json"
    _write_proposal(proposal, proposals)
    _write_validation(validation, validations)

    return build_feedback_calibration_promotion(
        proposal=proposal,
        validation=validation,
        min_samples=min_samples,
        min_improvement=min_improvement,
        max_after_error=max_after_error,
    )


def test_accepts_scale_when_validation_passes_thresholds():
    root = _tmp_dir("accept")

    report = _run(
        root,
        proposals=[_base_proposal()],
        validations=[_base_validation()],
    )

    assert report["global"]["num_accepted_scales"] == 1
    assert report["global"]["num_rejected_scales"] == 0
    accepted = report["calibration_profile"][0]
    assert accepted["codec"] == "JPEG"
    assert accepted["config"] == "q=85"
    assert accepted["axis"] == "rate"
    assert accepted["status"] == "accepted"
    assert accepted["scale"] == 1.2


def test_rejects_insufficient_samples():
    root = _tmp_dir("samples")

    report = _run(
        root,
        proposals=[_base_proposal()],
        validations=[_base_validation(num_eval_rows=2)],
    )

    rejected = report["rejected"][0]
    assert rejected["status"] == "rejected"
    assert "insufficient_samples" in rejected["reasons"]


def test_rejects_negative_or_zero_scale():
    root = _tmp_dir("scale")

    report = _run(
        root,
        proposals=[_base_proposal(rate_scale=0.0)],
        validations=[_base_validation(scale=0.0)],
    )

    rejected = report["rejected"][0]
    assert "non_positive_scale" in rejected["reasons"]


def test_rejects_unimproved_validation():
    root = _tmp_dir("unimproved")

    report = _run(
        root,
        proposals=[_base_proposal()],
        validations=[
            _base_validation(
                improved=False,
                improvement_ratio=-0.1,
                mean_abs_log_error_after=0.5,
            )
        ],
        max_after_error=1.0,
    )

    rejected = report["rejected"][0]
    assert "not_improved" in rejected["reasons"]


def test_rejects_small_improvement():
    root = _tmp_dir("small_improvement")

    report = _run(
        root,
        proposals=[_base_proposal()],
        validations=[
            _base_validation(
                improvement_ratio=0.05,
                mean_abs_log_error_after=0.2,
            )
        ],
        min_improvement=0.10,
    )

    rejected = report["rejected"][0]
    assert "improvement_below_threshold" in rejected["reasons"]


def test_rejects_large_after_error():
    root = _tmp_dir("large_after")

    report = _run(
        root,
        proposals=[_base_proposal()],
        validations=[
            _base_validation(
                mean_abs_log_error_after=0.30,
                improvement_ratio=0.30,
            )
        ],
        max_after_error=0.25,
    )

    rejected = report["rejected"][0]
    assert "after_error_above_threshold" in rejected["reasons"]


def test_rejects_unusable_proposal():
    root = _tmp_dir("unusable")

    report = _run(
        root,
        proposals=[_base_proposal(usable_for_rate=False)],
        validations=[_base_validation(usable_proposal=False)],
    )

    rejected = report["rejected"][0]
    assert "unusable_proposal" in rejected["reasons"]
    assert "proposal_not_marked_usable" in rejected["reasons"]


def test_energy_requires_total_usable_validation():
    root = _tmp_dir("energy")

    report = _run(
        root,
        proposals=[
            {
                "codec": "JXL",
                "config": "d=1.0",
                "energy_scale": 1.2,
                "usable_for_energy": True,
            }
        ],
        validations=[
            {
                "codec": "JXL",
                "config": "d=1.0",
                "axis": "energy",
                "num_eval_rows": 0,
                "scale": 1.2,
                "usable_proposal": True,
                "mean_abs_log_error_before": None,
                "mean_abs_log_error_after": 0.1,
                "improvement_ratio": 0.5,
                "improved": True,
            }
        ],
        min_samples=1,
    )

    rejected = report["rejected"][0]
    assert "energy_not_total_usable" in rejected["reasons"]


def test_missing_validation_for_proposal_is_rejected():
    root = _tmp_dir("missing_validation")

    report = _run(
        root,
        proposals=[_base_proposal(time_scale=0.8, usable_for_time=True)],
        validations=[],
    )

    rejected = report["rejected"]
    time = [item for item in rejected if item["axis"] == "time"][0]
    assert time["scale"] == 0.8
    assert "missing_validation" in time["reasons"]


def test_cli_writes_json_and_csv():
    root = _tmp_dir("cli")
    proposal = root / "proposal.json"
    validation = root / "validation.json"
    out = root / "candidate.json"
    summary_out = root / "candidate.csv"
    _write_proposal(proposal, [_base_proposal()])
    _write_validation(validation, [_base_validation()])

    main(
        [
            "--proposal",
            str(proposal),
            "--validation",
            str(validation),
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
            "--min-samples",
            "3",
            "--min-improvement",
            "0.10",
            "--max-after-error",
            "0.25",
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    rows = _read_csv(summary_out)

    assert report["mode"] == "candidate_profile_only"
    assert report["applied_by_router"] is False
    assert report["global"]["num_accepted_scales"] == 1
    assert rows[0]["status"] == "accepted"
