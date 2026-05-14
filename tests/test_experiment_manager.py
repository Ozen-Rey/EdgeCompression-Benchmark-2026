import json
from pathlib import Path

from src.router.experiment_manager import (
    build_experiment_command,
    load_experiment_suite,
    summarize_report,
)
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "experiment_manager"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_build_experiment_command_places_out_before_experiment_args():
    command = build_experiment_command(
        base_config="configs/router_image_v05.json",
        report_path="results/report.json",
        experiment_args=["--available-codecs", "JXL"],
    )

    assert "-m" in command
    assert "src.router.rde_router" in command
    assert "--config" in command
    assert "configs/router_image_v05.json" in command
    assert "--out" in command
    assert "results/report.json" in command

    assert command[-2] == "--available-codecs"
    assert command[-1] == "JXL"


def test_load_experiment_suite_validates_required_fields():
    suite_path = _tmp_path("suite.json")

    suite_path.write_text(
        json.dumps(
            {
                "version": "0.7",
                "base_config": "configs/router_image_v05.json",
                "experiments": [
                    {
                        "name": "jxl_only",
                        "args": ["--available-codecs", "JXL"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = load_experiment_suite(str(suite_path))

    assert suite["version"] == "0.7"
    assert suite["base_config"] == "configs/router_image_v05.json"
    assert len(suite["experiments"]) == 1


def test_summarize_report_extracts_selected_fields():
    report_path = _tmp_path("report.json")

    report = {
        "run_manifest": {
            "git": {
                "commit_short": "abc123",
                "dirty_worktree": False,
            }
        },
        "decision": {
            "decision_mode": "safe",
            "decision_trace": {
                "selected_reason": "lowest_J_RDE_in_safe_pool",
            },
            "selected": {
                "codec": "JXL",
                "config": "d=1.0",
                "rate": 1.2,
                "quality": 85.0,
                "quality_constraint_value": 80.0,
                "energy": 2.5,
                "time_ms": 100.0,
                "cost": 0.25,
                "cost_decomposition": {
                    "term_R": 0.1,
                    "term_E": 0.05,
                    "term_D": 0.1,
                },
            },
        },
        "execution_result": {
            "requested": True,
            "success": True,
        },
        "execution_validation": {
            "output_exists": True,
            "output_nonempty": True,
            "extension_valid": True,
            "output_size_bytes": 123,
        },
    }

    report_path.write_text(json.dumps(report), encoding="utf-8")

    row = summarize_report(
        experiment_name="jxl_only",
        run_result={
            "success": True,
            "returncode": 0,
            "report_path": str(report_path),
        },
    )

    assert row["experiment"] == "jxl_only"
    assert row["selected_codec"] == "JXL"
    assert row["selected_config"] == "d=1.0"
    assert row["decision_mode"] == "safe"
    assert row["selected_reason"] == "lowest_J_RDE_in_safe_pool"
    assert row["J_RDE"] == 0.25
    assert row["term_R"] == 0.1
    assert row["term_E"] == 0.05
    assert row["term_D"] == 0.1
    assert row["execution_requested"] is True
    assert row["execution_success"] is True
    assert row["output_exists"] is True
    assert row["git_commit_short"] == "abc123"


def test_summarize_report_dry_run_does_not_require_report():
    row = summarize_report(
        experiment_name="dry_run_test",
        run_result={
            "success": True,
            "returncode": 0,
            "dry_run": True,
            "report_path": "this/report/does/not/exist.json",
            "command": ["python", "-m", "src.router.rde_router"],
        },
    )

    assert row["experiment"] == "dry_run_test"
    assert row["success"] is True
    assert row["returncode"] == 0
    assert row["dry_run"] is True
    assert row["selected_codec"] is None
    assert "python -m src.router.rde_router" == row["command"]
