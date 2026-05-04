from pathlib import Path

from src.router.execution_validation import validate_execution_output


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "execution_validation"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_execution_validation_accepts_existing_nonempty_output():
    output = _tmp_path("encoded.jxl")
    output.write_bytes(b"fake encoded data")

    execution_plan = {
        "canonical": "JXL",
        "output": str(output),
    }

    execution_result = {
        "requested": True,
        "executed": True,
        "success": True,
        "output": str(output),
        "execution_time_ms": 12.5,
    }

    validation = validate_execution_output(
        execution_plan=execution_plan,
        execution_result=execution_result,
    )

    assert validation["enabled"] is True
    assert validation["output_exists"] is True
    assert validation["output_size_bytes"] > 0
    assert validation["output_nonempty"] is True
    assert validation["extension_valid"] is True
    assert validation["execution_time_ms"] == 12.5
    assert validation["warnings"] == []


def test_execution_validation_reports_missing_output():
    output = _tmp_path("missing.jxl")
    if output.exists():
        output.unlink()

    execution_plan = {
        "canonical": "JXL",
        "output": str(output),
    }

    execution_result = {
        "requested": True,
        "executed": True,
        "success": True,
        "output": str(output),
        "execution_time_ms": 5.0,
    }

    validation = validate_execution_output(
        execution_plan=execution_plan,
        execution_result=execution_result,
    )

    assert validation["enabled"] is True
    assert validation["output_exists"] is False
    assert validation["output_nonempty"] is False
    assert "output_file_missing" in validation["warnings"]


def test_execution_validation_reports_wrong_extension():
    output = _tmp_path("encoded.mp4")
    output.write_bytes(b"fake encoded data")

    execution_plan = {
        "canonical": "JXL",
        "output": str(output),
    }

    execution_result = {
        "requested": True,
        "executed": True,
        "success": True,
        "output": str(output),
        "execution_time_ms": 7.0,
    }

    validation = validate_execution_output(
        execution_plan=execution_plan,
        execution_result=execution_result,
    )

    assert validation["output_exists"] is True
    assert validation["output_nonempty"] is True
    assert validation["extension_valid"] is False
    assert "unexpected_output_extension:.mp4" in validation["warnings"]


def test_execution_validation_disabled_when_not_requested():
    validation = validate_execution_output(
        execution_plan={},
        execution_result={
            "requested": False,
            "executed": False,
        },
    )

    assert validation["enabled"] is False
    assert validation["reason"] == "execution_not_requested"
