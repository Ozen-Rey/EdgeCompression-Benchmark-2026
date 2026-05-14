import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.router.codecs.external_codec_probe import main, probe_external_codec_spec


def _valid_spec(**overrides):
    spec = {
        "schema_version": "0.36.0",
        "codec_id": "example_codec",
        "display_name": "Example Codec",
        "domain": "image",
        "family": "classical",
        "runtime": {
            "type": "external_command",
            "executable": sys.executable,
            "max_runtime_seconds": 30,
        },
        "version_probe": {
            "command": ["{executable}", "--version"],
        },
        "encode": {
            "command_template": [
                "{binary}",
                "--input",
                "{input}",
                "--output",
                "{output}",
                "--quality",
                "{quality}",
            ],
        },
        "decode": {
            "command_template": [
                "{binary}",
                "--decode",
                "{input}",
                "--output",
                "{output}",
            ],
        },
        "parameters": [
            {
                "name": "quality",
                "type": "integer",
                "values": [60, 75, 90],
            }
        ],
        "output": {
            "extension": ".exi",
        },
        "rate": {
            "metric": "bpp",
        },
        "quality": {
            "metric": "ssimulacra2",
            "direction": "higher_is_better",
        },
        "measurement": {
            "time": "wall_clock",
        },
        "requirements": {
            "binaries": ["example-codec"],
        },
        "security": {
            "allow_shell": False,
        },
    }
    spec.update(overrides)
    return spec


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    spec_path = tmp_path / "external_codec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


def _probe(report: dict) -> dict:
    return report["external_codec_probe"]


def test_valid_external_command_probe_fingerprints_python_executable(tmp_path: Path):
    spec_path = _write_spec(tmp_path, _valid_spec())

    report = _probe(probe_external_codec_spec(spec_path))

    assert report["valid_spec"] is True
    assert report["executable_exists"] is True
    assert report["binary_sha256"]
    assert report["version_probe_executed"] is True
    assert report["version_string"]
    assert report["version_returncode"] == 0
    assert report["available"] is True
    assert report["errors"] == []


def test_missing_executable_reports_controlled_error(tmp_path: Path):
    missing = tmp_path / "missing-codec.exe"
    spec_path = _write_spec(
        tmp_path,
        _valid_spec(runtime={
            "type": "external_command",
            "executable": str(missing),
            "max_runtime_seconds": 30,
        }),
    )

    report = _probe(probe_external_codec_spec(spec_path))

    assert report["available"] is False
    assert report["executable_exists"] is False
    assert "executable_not_found" in report["errors"]


def test_invalid_spec_does_not_run_subprocess(tmp_path: Path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid specs must not run subprocess")

    monkeypatch.setattr("src.router.codecs.external_codec_probe.subprocess.run", fail_if_called)
    spec_path = _write_spec(tmp_path, _valid_spec(codec_id="Bad Codec!"))

    report = _probe(probe_external_codec_spec(spec_path))

    assert report["valid_spec"] is False
    assert report["version_probe_executed"] is False
    assert "invalid_codec_id" in report["errors"]


def test_version_probe_timeout_reports_controlled_error(tmp_path: Path):
    spec_path = _write_spec(
        tmp_path,
        _valid_spec(version_probe={
            "command": [
                "{executable}",
                "-c",
                "import time; time.sleep(2)",
            ],
        }),
    )

    report = _probe(probe_external_codec_spec(spec_path, timeout_s=0.1))

    assert report["available"] is False
    assert report["version_probe_executed"] is True
    assert "version_probe_timeout" in report["errors"]


def test_version_probe_uses_shell_false(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="codec 1.0\n",
            stderr="",
        )

    monkeypatch.setattr("src.router.codecs.external_codec_probe.subprocess.run", fake_run)
    spec_path = _write_spec(tmp_path, _valid_spec())

    report = _probe(probe_external_codec_spec(spec_path))

    assert report["available"] is True
    assert calls
    assert calls[0]["shell"] is False


def test_python_module_spec_does_not_import_module_by_default(tmp_path: Path):
    module_name = "external_codec_probe_should_not_import_me"
    sys.modules.pop(module_name, None)
    spec_path = _write_spec(
        tmp_path,
        _valid_spec(
            codec_id="python_codec",
            runtime={
                "type": "python_module",
                "module": module_name,
                "max_runtime_seconds": 30,
            },
        ),
    )

    report = _probe(probe_external_codec_spec(spec_path))

    assert report["valid_spec"] is True
    assert report["available"] == "unknown"
    assert report["version_probe_executed"] is False
    assert module_name not in sys.modules
    assert "python_module_not_imported" in report["warnings"]


def test_report_contains_safety_flags(tmp_path: Path):
    spec_path = _write_spec(tmp_path, _valid_spec())

    report = _probe(probe_external_codec_spec(spec_path, run_version_probe=False))

    assert report["safety"] == {
        "shell_used": False,
        "encode_executed": False,
        "decode_executed": False,
        "benchmark_executed": False,
    }


def test_cli_writes_probe_report(tmp_path: Path, capsys):
    spec_path = _write_spec(tmp_path, _valid_spec())
    out_path = tmp_path / "probe_report.json"

    report = main(["--spec", str(spec_path), "--out", str(out_path)])
    captured = capsys.readouterr()

    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == report
    assert json.loads(captured.out)["external_codec_probe"]["available"] is True


def test_cli_no_version_probe_skips_execution(tmp_path: Path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("--no-version-probe must not run subprocess")

    monkeypatch.setattr("src.router.codecs.external_codec_probe.subprocess.run", fail_if_called)
    spec_path = _write_spec(tmp_path, _valid_spec())
    out_path = tmp_path / "probe_report.json"

    report = _probe(main([
        "--spec",
        str(spec_path),
        "--out",
        str(out_path),
        "--no-version-probe",
    ]))

    assert report["available"] is True
    assert report["version_probe_executed"] is False
    assert "version_probe_skipped" in report["warnings"]


def test_cli_rejects_non_positive_timeout(tmp_path: Path):
    spec_path = _write_spec(tmp_path, _valid_spec())
    out_path = tmp_path / "probe_report.json"

    with pytest.raises(ValueError, match="--timeout-s must be positive"):
        main([
            "--spec",
            str(spec_path),
            "--out",
            str(out_path),
            "--timeout-s",
            "0",
        ])
