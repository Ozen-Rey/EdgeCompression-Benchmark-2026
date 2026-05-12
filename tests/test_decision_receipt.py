import json
from pathlib import Path
from uuid import uuid4

from src.router.calibration_bundle import sha256_file
from src.router.decision_receipt import (
    build_decision_receipt,
    sanitize_replay_argv,
)
from src.router.version import ROUTER_VERSION


def _tmp_dir(name: str) -> Path:
    root = (
        Path(__file__).with_name("_tmp")
        / "decision_receipt"
        / f"{name}_{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_sanitize_replay_argv_removes_output_and_execution_side_effects():
    argv = [
        "--csv",
        "points.csv",
        "--execute",
        "--out",
        "old_report.json",
        "--feedback-out=feedback.csv",
        "--export-topk",
        "--quality-floor",
        "90",
    ]

    sanitized = sanitize_replay_argv(argv)

    assert sanitized == [
        "--csv",
        "points.csv",
        "--quality-floor",
        "90",
    ]


def test_build_decision_receipt_records_decision_and_input_hashes():
    root = _tmp_dir("build")
    csv_path = root / "points.csv"
    csv_path.write_text(
        "codec,config,rate,quality,energy\nJPEG,q=85,1,90,0.1\n",
        encoding="utf-8",
    )

    report = {
        "router_version": ROUTER_VERSION,
        "profile": "balanced",
        "csv": str(csv_path),
        "run_manifest": {
            "argv": {
                "expanded": [
                    "--csv",
                    str(csv_path),
                    "--out",
                    str(root / "report.json"),
                ],
            },
        },
        "resolved_args": {
            "csv": str(csv_path),
            "quality_thresholds_file": None,
        },
        "decision": {
            "decision_mode": "safe",
            "selected": {
                "codec": "JPEG",
                "config": "q=85",
                "rate": 1.0,
                "quality": 90.0,
                "energy": 0.1,
                "time_ms": 5.0,
                "cost": 0.25,
                "quality_constraint_stat": "mean",
                "quality_constraint_value": 90.0,
            },
            "num_points_total": 1,
            "num_points_admissible": 1,
            "num_points_safe": 1,
            "num_points_near": 0,
        },
        "weights": {
            "w_E": 0.33,
            "w_R": 0.33,
            "w_D": 0.34,
        },
        "constraints": {
            "quality_floor": 50.0,
        },
        "normalization": {
            "scope": "runtime",
        },
        "calibration_bundle": {
            "enabled": False,
        },
        "calibration_bundle_validation": {
            "enabled": False,
        },
    }

    receipt = build_decision_receipt(report)
    artifacts = {
        item["name"]: item
        for item in receipt["input_artifacts"]
    }

    assert receipt["artifact_type"] == "router_decision_receipt"
    assert receipt["router_version"] == ROUTER_VERSION
    assert receipt["receipt_schema_version"] == "0.29.0"
    assert receipt["replay"]["argv"] == ["--csv", str(csv_path)]
    assert receipt["decision"]["selected_codec"] == "JPEG"
    assert receipt["decision"]["selected_config"] == "q=85"
    assert artifacts["effective_csv"]["sha256"] == sha256_file(csv_path)
    assert json.dumps(receipt)
