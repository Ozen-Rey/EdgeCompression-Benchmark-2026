from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup" / "setup_router.py"
DOCTOR = ROOT / "scripts" / "setup" / "doctor.py"


def _run(args: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def test_setup_router_help_exits_zero() -> None:
    result = _run([str(SETUP), "--help"])

    assert result.returncode == 0
    assert "router Python development environment" in result.stdout


def test_setup_router_dry_run_exits_zero() -> None:
    result = _run([str(SETUP), "--dry-run", "--venv", ".tmp_setup_test_venv"])

    assert result.returncode == 0
    assert '"dry_run": true' in result.stdout
    assert not (ROOT / ".tmp_setup_test_venv").exists()


def test_doctor_help_exits_zero() -> None:
    result = _run([str(DOCTOR), "--help"])

    assert result.returncode == 0
    assert "Read-only doctor" in result.stdout


def test_doctor_report_out_writes_valid_json(tmp_path: Path) -> None:
    report_path = tmp_path / "environment_doctor_report.json"

    result = _run([str(DOCTOR), "--report-out", str(report_path)])

    assert result.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["router"]["import_ok"] is True
    assert "optional_external_tools" in report


def test_optional_external_tools_are_non_fatal(tmp_path: Path) -> None:
    report_path = tmp_path / "doctor.json"

    result = _run([str(DOCTOR), "--report-out", str(report_path)])

    assert result.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for tool in report["optional_external_tools"].values():
        assert tool["optional_for_benchmark_or_execution"] is True
    assert not any("optional" in error.lower() for error in report["errors"])


def test_prompt_default_no_does_not_create_venv_or_install(tmp_path: Path) -> None:
    venv_path = tmp_path / "router_venv"

    result = _run([str(SETUP), "--venv", str(venv_path)], input_text="\n\n\n")

    assert result.returncode == 0
    assert not venv_path.exists()
    assert '"name": "create_venv"' in result.stdout
    assert '"skipped": true' in result.stdout


def test_setup_router_source_has_no_system_or_benchmark_installs() -> None:
    source = SETUP.read_text(encoding="utf-8")

    forbidden = [
        "apt install",
        "pacman -S",
        "winget install",
        "choco install",
        "brew install",
        "git clone",
        "gdown",
        "wget",
    ]
    for pattern in forbidden:
        assert pattern not in source
