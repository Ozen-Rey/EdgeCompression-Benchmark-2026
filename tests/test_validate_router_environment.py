from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup" / "validate_router_environment.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_router_environment_for_tests",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_exits_zero() -> None:
    result = _run(["--help"])

    assert result.returncode == 0
    assert "router environment validation report" in result.stdout


def test_skip_tests_no_doctor_generates_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "validation"

    result = _run(
        [
            "--label",
            "unit_test_env",
            "--out-dir",
            str(out_dir),
            "--skip-tests",
            "--no-doctor",
        ]
    )

    assert result.returncode == 0
    manifest_path = out_dir / "router_environment_validation_manifest.json"
    summary_path = out_dir / "VALIDATION_SUMMARY.md"
    metadata_path = out_dir / "environment_metadata.json"
    pip_freeze_path = out_dir / "pip_freeze.txt"
    full_fingerprint_path = out_dir / "platform_fingerprint_full.json"
    sanitized_fingerprint_path = out_dir / "platform_fingerprint_sanitized.json"

    assert manifest_path.exists()
    assert summary_path.exists()
    assert metadata_path.exists()
    assert pip_freeze_path.exists()
    assert full_fingerprint_path.exists()
    assert sanitized_fingerprint_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation_scope"] == "router environment validation"
    assert manifest["not_benchmark_reproduction"] is True
    assert manifest["doctor"]["skipped"] is True
    assert manifest["fixture_runs"] == {}
    assert "No benchmark was executed." in manifest["boundaries"]
    assert manifest["platform_fingerprint"]["schema"] == "rde_platform_fingerprint_v1"
    assert manifest["platform_fingerprint"]["full_path"] == "platform_fingerprint_full.json"
    assert manifest["platform_fingerprint"]["sanitized_path"] == (
        "platform_fingerprint_sanitized.json"
    )
    sanitized = json.loads(sanitized_fingerprint_path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert manifest["platform_fingerprint"]["sanitized_sha256"] == (
        hashlib.sha256(canonical).hexdigest()
    )
    assert "summary" in manifest["platform_fingerprint"]


def test_command_runner_handles_failure_without_crashing() -> None:
    module = _load_script()

    result = module.run_command(
        [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"]
    )

    assert result["ok"] is False
    assert result["returncode"] == 7
    assert "bad" in result["stdout"]


def test_output_directory_is_created(tmp_path: Path) -> None:
    out_dir = tmp_path / "new" / "nested" / "validation"

    result = _run(["--label", "mkdir_test", "--out-dir", str(out_dir), "--skip-tests", "--no-doctor"])

    assert result.returncode == 0
    assert out_dir.is_dir()


def test_refuses_results_output_directory() -> None:
    result = _run(
        [
            "--label",
            "bad_results",
            "--out-dir",
            str(ROOT / "results" / "validation_should_not_exist"),
            "--skip-tests",
            "--no-doctor",
        ]
    )

    assert result.returncode != 0
    assert not (ROOT / "results" / "validation_should_not_exist").exists()


def test_source_has_no_install_or_venv_creation_commands() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    forbidden = [
        "pip install",
        '"pip", "install"',
        "'pip', 'install'",
        '"-m", "venv"',
        "'-m', 'venv'",
        "setup_router.py --yes",
    ]
    for pattern in forbidden:
        assert pattern not in source


def test_outputs_stay_inside_requested_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "validation"
    result = _run(["--label", "containment", "--out-dir", str(out_dir), "--skip-tests", "--no-doctor"])

    assert result.returncode == 0
    assert not (ROOT / "results" / "router_environment_validation_manifest.json").exists()
    for path in out_dir.rglob("*"):
        assert out_dir in path.resolve().parents or path.resolve() == out_dir.resolve()
