import json
import subprocess
import sys
from pathlib import Path

from src.router.external_codec_dry_run import main, dry_run_external_codec_spec


def _fake_codec(tmp_path: Path) -> Path:
    script = tmp_path / "fake_codec.py"
    script.write_text(
        "\n".join([
            "import shutil",
            "import sys",
            "import time",
            "mode = sys.argv[1]",
            "if mode == 'copy':",
            "    shutil.copyfile(sys.argv[2], sys.argv[3])",
            "elif mode == 'empty':",
            "    open(sys.argv[3], 'wb').close()",
            "elif mode == 'missing':",
            "    pass",
            "elif mode == 'sleep':",
            "    time.sleep(5)",
            "elif mode == 'decode':",
            "    shutil.copyfile(sys.argv[2], sys.argv[3])",
            "else:",
            "    raise SystemExit(2)",
        ]),
        encoding="utf-8",
    )
    return script


def _input_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.png"
    path.write_bytes(b"fake image bytes")
    return path


def _valid_spec(tmp_path: Path, **overrides):
    fake = _fake_codec(tmp_path)
    spec = {
        "schema_version": "0.36.0",
        "codec_id": "fake_codec",
        "display_name": "Fake Codec",
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
                "{executable}",
                str(fake),
                "copy",
                "{input}",
                "{output}",
                "{quality}",
            ],
        },
        "decode": {
            "available": False,
            "command_template": [
                "{executable}",
                str(fake),
                "decode",
                "{input}",
                "{output}",
            ],
        },
        "parameters": [
            {
                "name": "quality",
                "type": "integer",
                "values": [50, 75, 90],
            }
        ],
        "output": {
            "extension": ".fake",
            "must_be_nonempty": True,
        },
        "rate": {
            "metric": "bytes",
        },
        "quality": {
            "metric": "contract_only",
            "direction": "higher_is_better",
        },
        "measurement": {
            "time": "dry_run_wall_clock",
        },
        "requirements": {
            "binaries": ["python"],
        },
        "security": {
            "allow_shell": False,
        },
    }
    spec.update(overrides)
    return spec


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _run(tmp_path: Path, spec: dict, **kwargs) -> dict:
    spec_path = _write_spec(tmp_path, spec)
    report = dry_run_external_codec_spec(
        spec_path,
        input_path=kwargs.pop("input_path", _input_file(tmp_path)),
        out_dir=kwargs.pop("out_dir", tmp_path / "out"),
        params=kwargs.pop("params", {"quality": "50"}),
        timeout_s=kwargs.pop("timeout_s", 5),
    )
    return report["external_codec_dry_run"]


def test_valid_encode_dry_run_with_fake_copy_codec(tmp_path: Path):
    report = _run(tmp_path, _valid_spec(tmp_path))

    assert report["valid_spec"] is True
    assert report["executable_checked"] is True
    assert report["encode"]["executed"] is True
    assert report["encode"]["output_exists"] is True
    assert report["encode"]["output_size_bytes"] > 0
    assert report["encode"]["output_extension_valid"] is True
    assert report["decode"]["requested"] is False
    assert report["success"] is True


def test_valid_encode_decode_dry_run(tmp_path: Path):
    spec = _valid_spec(
        tmp_path,
        decode={
            "available": True,
            "must_be_nonempty": True,
            "reconstruction_extension": ".png",
            "command_template": [
                "{executable}",
                str(tmp_path / "fake_codec.py"),
                "decode",
                "{input}",
                "{output}",
            ],
        },
    )

    report = _run(tmp_path, spec)

    assert report["encode"]["success"] is True
    assert report["decode"]["requested"] is True
    assert report["decode"]["executed"] is True
    assert report["decode"]["reconstruction_exists"] is True
    assert report["decode"]["reconstruction_size_bytes"] > 0
    assert report["decode"]["success"] is True
    assert report["success"] is True


def test_missing_output_reports_controlled_error(tmp_path: Path):
    spec = _valid_spec(tmp_path)
    spec["encode"]["command_template"][2] = "missing"

    report = _run(tmp_path, spec)

    assert report["success"] is False
    assert report["encode"]["success"] is False
    assert "encode_output_missing" in report["errors"]


def test_empty_output_with_nonempty_contract_fails(tmp_path: Path):
    spec = _valid_spec(tmp_path)
    spec["encode"]["command_template"][2] = "empty"

    report = _run(tmp_path, spec)

    assert report["success"] is False
    assert report["encode"]["output_size_bytes"] == 0
    assert "encode_output_empty" in report["errors"]


def test_encode_timeout_reports_controlled_error(tmp_path: Path):
    spec = _valid_spec(tmp_path)
    spec["encode"]["command_template"][2] = "sleep"

    report = _run(tmp_path, spec, timeout_s=0.1)

    assert report["success"] is False
    assert report["encode"]["timeout"] is True
    assert "encode_timeout" in report["errors"]


def test_subprocess_uses_shell_false(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        Path(args[0][4]).write_bytes(b"fake")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr("src.router.external_codec_dry_run.subprocess.run", fake_run)

    _run(tmp_path, _valid_spec(tmp_path))

    assert calls
    assert calls[0]["shell"] is False


def test_undeclared_parameter_is_rejected(tmp_path: Path):
    report = _run(
        tmp_path,
        _valid_spec(tmp_path),
        params={"quality": "50", "speed": "fast"},
    )

    assert report["success"] is False
    assert "undeclared_parameter:speed" in report["errors"]
    assert report["encode"]["executed"] is False


def test_unresolved_placeholder_is_rejected(tmp_path: Path):
    spec = _valid_spec(tmp_path)
    spec["encode"]["command_template"].append("{effort}")

    report = _run(tmp_path, spec)

    assert report["success"] is False
    assert "unresolved_placeholder:effort" in report["errors"]
    assert report["encode"]["executed"] is False


def test_output_path_traversal_is_rejected(tmp_path: Path):
    spec = _valid_spec(
        tmp_path,
        output={
            "extension": ".fake",
            "filename": "../escape.fake",
            "must_be_nonempty": True,
        },
    )

    report = _run(tmp_path, spec)

    assert report["success"] is False
    assert "output_filename_unsafe" in report["errors"]
    assert report["safety"]["output_confined_to_out_dir"] is False
    assert report["encode"]["executed"] is False


def test_invalid_spec_blocks_dry_run_without_subprocess(tmp_path: Path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid spec must not execute subprocess")

    monkeypatch.setattr("src.router.external_codec_dry_run.subprocess.run", fail_if_called)

    report = _run(tmp_path, _valid_spec(tmp_path, codec_id="Bad Codec!"))

    assert report["valid_spec"] is False
    assert "invalid_codec_id" in report["errors"]
    assert report["encode"]["executed"] is False


def test_decode_available_false_does_not_execute_decode(tmp_path: Path):
    report = _run(tmp_path, _valid_spec(tmp_path))

    assert report["decode"]["requested"] is False
    assert report["decode"]["executed"] is False
    assert report["success"] is True


def test_safety_flags_are_present_and_correct(tmp_path: Path):
    report = _run(tmp_path, _valid_spec(tmp_path))

    assert report["safety"] == {
        "shell_used": False,
        "benchmark_executed": False,
        "router_candidate_registered": False,
        "output_confined_to_out_dir": True,
    }


def test_cli_writes_dry_run_report(tmp_path: Path, capsys):
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_path = _input_file(tmp_path)
    out_dir = tmp_path / "out"
    report_path = tmp_path / "dry_run.json"

    report = main([
        "--spec",
        str(spec_path),
        "--input",
        str(input_path),
        "--out-dir",
        str(out_dir),
        "--param",
        "quality=50",
        "--timeout-s",
        "5",
        "--out",
        str(report_path),
    ])
    captured = capsys.readouterr()

    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(captured.out)["external_codec_dry_run"]["success"] is True


def test_dry_run_does_not_import_router_decision_module(tmp_path: Path):
    sys.modules.pop("src.router.rde_router", None)

    report = _run(tmp_path, _valid_spec(tmp_path))

    assert report["success"] is True
    assert "src.router.rde_router" not in sys.modules
