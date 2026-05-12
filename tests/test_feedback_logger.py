import csv
import json
from pathlib import Path
from uuid import uuid4

from src.router.feedback_logger import FEEDBACK_FIELDS, append_feedback_row
from src.router.rde_router import _build_feedback_row


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _tmp_feedback_path(name: str) -> Path:
    root = Path(__file__).with_name("_tmp") / "feedback_logger"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}_{uuid4().hex}.csv"


def test_feedback_logger_creates_file_with_header():
    path = _tmp_feedback_path("online")

    append_feedback_row(
        path,
        {
            "router_version": "0.12.0",
            "selected_codec": "JPEG",
            "execution_success": True,
        },
    )

    assert path.exists()
    rows = _read_rows(path)
    assert len(rows) == 1
    assert rows[0]["router_version"] == "0.12.0"
    assert rows[0]["selected_codec"] == "JPEG"
    assert rows[0]["execution_success"] == "True"

    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))

    assert header[: len(FEEDBACK_FIELDS)] == FEEDBACK_FIELDS


def test_feedback_logger_appends_without_overwriting():
    path = _tmp_feedback_path("append")

    append_feedback_row(path, {"selected_codec": "JPEG"})
    append_feedback_row(path, {"selected_codec": "JXL"})

    rows = _read_rows(path)
    assert [row["selected_codec"] for row in rows] == ["JPEG", "JXL"]


def test_feedback_logger_serializes_nested_values():
    path = _tmp_feedback_path("nested")

    append_feedback_row(
        path,
        {
            "feature_level": {"online_feedback": "append_only_observational"},
            "nested_extra": {"warnings": ["partial_gpu_energy"]},
        },
    )

    rows = _read_rows(path)
    assert json.loads(rows[0]["feature_level"]) == {
        "online_feedback": "append_only_observational"
    }
    assert json.loads(rows[0]["nested_extra"]) == {
        "warnings": ["partial_gpu_energy"]
    }


def test_feedback_logger_records_failed_execution():
    path = _tmp_feedback_path("failed")

    append_feedback_row(
        path,
        {
            "execution_requested": True,
            "execution_success": False,
            "error": "execution_plan_not_executable",
            "output_exists": False,
            "output_nonempty": False,
        },
    )

    rows = _read_rows(path)
    assert rows[0]["execution_requested"] == "True"
    assert rows[0]["execution_success"] == "False"
    assert rows[0]["error"] == "execution_plan_not_executable"
    assert rows[0]["output_exists"] == "False"


def test_feedback_row_keeps_gpu_only_energy_partial_not_total():
    report_path = _tmp_feedback_path("report").with_suffix(".json")
    report = {
        "router_version": "0.12.0",
        "feature_level": {"online_feedback": "append_only_observational"},
        "domain": "image",
        "profile": "balanced",
        "execution_plan": {"input": "tests/fixtures/image.png"},
        "run_manifest": {"git": {"commit_short": "abc123"}},
        "decision": {
            "decision_mode": "safe",
            "selected": {
                "codec": "JPEG",
                "config": "q=85",
                "rate": 1.0,
                "quality": 90.0,
                "energy": 2.0,
                "time_ms": 10.0,
                "cost": 0.1,
                "cost_decomposition": {
                    "term_R": 0.01,
                    "term_E": 0.02,
                    "term_D": 0.03,
                },
            },
        },
    }
    execution_result = {
        "requested": True,
        "success": True,
        "execution_time_ms": 12.0,
        "local_cpu_energy_j": None,
        "local_gpu_energy_j": 1.5,
        "local_energy_j": None,
        "energy_scope": "gpu",
        "energy_is_measured": True,
        "energy_usable_for_total": False,
        "energy_backend": "cpu=none;gpu=nvml_total_counter",
        "energy_method": "cpu=unavailable;gpu=nvml_total_energy_counter_delta",
        "energy_quality": "cpu=not_measured;gpu=hardware_counter",
    }
    execution_validation = {
        "output_size_bytes": 100,
        "output_exists": True,
        "output_nonempty": True,
    }

    row = _build_feedback_row(
        report=report,
        execution_result=execution_result,
        execution_validation=execution_validation,
        report_path=report_path,
    )

    assert row["local_gpu_energy_j"] == 1.5
    assert row["local_energy_j"] is None
    assert row["energy_scope"] == "gpu"
    assert row["energy_is_measured"] is True
    assert row["energy_usable_for_total"] is False
