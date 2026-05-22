from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import ModuleType
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup" / "setup_router.py"
DOCTOR = ROOT / "scripts" / "setup" / "doctor.py"


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert "bootstrap_prerequisites" in result.stdout
    assert "system_package_hints" in result.stdout
    assert "dependency_boundaries" in result.stdout
    assert "router_core_dependency_probe" in result.stdout
    assert '"module": "PIL"' in result.stdout
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
    assert report["python_dependencies"]["Pillow"]["import_name"] == "PIL"
    assert report["python_dependencies"]["Pillow"]["required"] is True
    assert report["python_dependencies"]["Pillow"]["install_scope"] == "python_package"
    assert "bootstrap_prerequisites" in report
    assert "system_package_hints" in report
    assert report["bootstrap_prerequisites"]["pip"]["managed_by_setup"] is False
    assert report["bootstrap_prerequisites"]["unzip"]["optional_for_source_archives"] is True
    assert "optional_external_tools" in report


def test_optional_external_tools_are_non_fatal(tmp_path: Path) -> None:
    report_path = tmp_path / "doctor.json"

    result = _run([str(DOCTOR), "--report-out", str(report_path)])

    assert result.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for tool in report["optional_external_tools"].values():
        assert tool["optional_for_benchmark_or_execution"] is True
    assert not any("optional" in error.lower() for error in report["errors"])


def test_system_package_hints_cover_supported_platforms() -> None:
    setup = _load_script(SETUP, "setup_router_for_hint_test")

    assert setup._system_package_hints({"id": "arch", "id_like": []}) == [
        "sudo pacman -S --needed git python python-pip unzip"
    ]
    assert setup._system_package_hints({"id": "ubuntu", "id_like": []}) == [
        "sudo apt update",
        "sudo apt install git python3 python3-pip python3-venv unzip",
    ]
    assert setup._system_package_hints({"id": "fedora", "id_like": []}) == [
        "sudo dnf install git python3 python3-pip unzip"
    ]
    assert "official installers" in setup._system_package_hints(
        {"id": "macos", "id_like": []}
    )[0]
    assert "setup.ps1" in setup._system_package_hints(
        {"id": "windows", "id_like": []}
    )[0]


def test_venv_failure_hint_is_distro_aware() -> None:
    setup = _load_script(SETUP, "setup_router_for_venv_hint_test")

    assert setup._venv_failure_hint({"id": "ubuntu", "id_like": []}) == (
        "Install python3-venv and re-run setup."
    )
    assert setup._venv_failure_hint({"id": "arch", "id_like": []}) == (
        "Verify that the python package is installed, then re-run setup."
    )
    assert "venv support" in setup._venv_failure_hint({"id": "linux", "id_like": []})


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
        "git clone",
        "gdown",
        "wget",
    ]
    for pattern in forbidden:
        assert pattern not in source
    assert "system_package_hints" in source
    assert "does not install system packages" in source
