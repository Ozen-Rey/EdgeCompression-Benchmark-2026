import csv
import json
import subprocess
import sys
from pathlib import Path

from src.router.external_codec_benchmark import (
    CSV_COLUMNS,
    main,
    run_external_codec_benchmark,
)


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


def _input_file(tmp_path: Path, name: str, payload: bytes = b"fake image") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _valid_spec(tmp_path: Path, **overrides) -> dict:
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
                "values": [40, 80],
            }
        ],
        "output": {
            "extension": ".fake",
            "must_be_nonempty": True,
        },
        "rate": {
            "metric": "output_size_bytes",
        },
        "quality": {
            "metric": "contract_only",
            "direction": "higher_is_better",
        },
        "measurement": {
            "time": "benchmark_wall_clock",
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


def _rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_benchmark_runs_explicit_grid_and_writes_csv(tmp_path: Path):
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_a = _input_file(tmp_path, "a.png", b"a")
    input_b = _input_file(tmp_path, "b.png", b"bb")
    csv_path = tmp_path / "measurements.csv"
    report_path = tmp_path / "report.json"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_a, input_b],
        out_dir=tmp_path / "runs",
        report_out=report_path,
        csv_out=csv_path,
        param_sets=[{"quality": "40"}, {"quality": "80"}],
        timeout_s=5,
    )["external_codec_benchmark"]

    rows = _rows(csv_path)
    assert csv_path.exists()
    assert report_path.exists()
    assert set(rows[0]) == set(CSV_COLUMNS)
    assert report["num_runs"] == 4
    assert report["num_successful_runs"] == 4
    assert report["success"] is True
    assert len(rows) == 4
    assert {row["param_json"] for row in rows} == {
        '{"quality": "40"}',
        '{"quality": "80"}',
    }
    assert all(row["encode_success"] == "True" for row in rows)
    assert all(row["rate_metric"] == "output_size_bytes" for row in rows)
    assert all(row["energy_j"] == "" for row in rows)
    assert all(row["energy_provenance_tier"] == "unknown" for row in rows)


def test_default_parameter_grid_uses_declared_values(tmp_path: Path):
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_path = _input_file(tmp_path, "a.png")
    csv_path = tmp_path / "measurements.csv"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        csv_out=csv_path,
        timeout_s=5,
    )["external_codec_benchmark"]

    assert report["num_param_sets"] == 2
    assert [row["param_json"] for row in _rows(csv_path)] == [
        '{"quality": "40"}',
        '{"quality": "80"}',
    ]


def test_failed_run_produces_failed_csv_row(tmp_path: Path):
    spec = _valid_spec(tmp_path)
    spec["encode"]["command_template"][2] = "missing"
    spec_path = _write_spec(tmp_path, spec)
    input_path = _input_file(tmp_path, "a.png")
    csv_path = tmp_path / "measurements.csv"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        csv_out=csv_path,
        param_sets=[{"quality": "40"}],
        timeout_s=5,
    )["external_codec_benchmark"]

    rows = _rows(csv_path)
    assert report["success"] is False
    assert report["num_successful_runs"] == 0
    assert len(rows) == 1
    assert rows[0]["encode_success"] == "False"
    assert "encode_output_missing" in rows[0]["error"]


def test_decode_success_is_recorded_when_decode_available(tmp_path: Path):
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
    spec_path = _write_spec(tmp_path, spec)
    input_path = _input_file(tmp_path, "a.png")
    csv_path = tmp_path / "measurements.csv"

    run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        csv_out=csv_path,
        param_sets=[{"quality": "40"}],
        timeout_s=5,
    )

    row = _rows(csv_path)[0]
    assert row["encode_success"] == "True"
    assert row["decode_success"] == "True"
    assert row["reconstruction_path"]


def test_timeout_run_is_reported_not_dropped(tmp_path: Path):
    spec = _valid_spec(tmp_path)
    spec["encode"]["command_template"][2] = "sleep"
    spec_path = _write_spec(tmp_path, spec)
    input_path = _input_file(tmp_path, "a.png")
    csv_path = tmp_path / "measurements.csv"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        csv_out=csv_path,
        param_sets=[{"quality": "40"}],
        timeout_s=0.1,
    )["external_codec_benchmark"]

    rows = _rows(csv_path)
    assert report["success"] is False
    assert len(rows) == 1
    assert "encode_timeout" in rows[0]["error"]


def test_undeclared_parameter_grid_is_rejected_before_runs(tmp_path: Path):
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_path = _input_file(tmp_path, "a.png")
    csv_path = tmp_path / "measurements.csv"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        csv_out=csv_path,
        param_sets=[{"speed": "fast"}],
        timeout_s=5,
    )["external_codec_benchmark"]

    assert report["success"] is False
    assert "undeclared_parameter:speed" in report["errors"]
    assert _rows(csv_path) == []


def test_output_path_traversal_in_spec_produces_failed_row(tmp_path: Path):
    spec = _valid_spec(
        tmp_path,
        output={
            "extension": ".fake",
            "filename": "../escape.fake",
            "must_be_nonempty": True,
        },
    )
    spec_path = _write_spec(tmp_path, spec)
    input_path = _input_file(tmp_path, "a.png")
    csv_path = tmp_path / "measurements.csv"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        csv_out=csv_path,
        param_sets=[{"quality": "40"}],
        timeout_s=5,
    )["external_codec_benchmark"]

    rows = _rows(csv_path)
    assert report["success"] is False
    assert len(rows) == 1
    assert "output_filename_unsafe" in rows[0]["error"]
    assert rows[0]["output_path"] == ""


def test_invalid_spec_does_not_execute_and_writes_empty_csv(tmp_path: Path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid spec must not execute subprocess")

    monkeypatch.setattr("src.router.external_codec_dry_run.subprocess.run", fail_if_called)
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path, codec_id="Bad Codec!"))
    input_path = _input_file(tmp_path, "a.png")
    csv_path = tmp_path / "measurements.csv"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        csv_out=csv_path,
        param_sets=[{"quality": "40"}],
        timeout_s=5,
    )["external_codec_benchmark"]

    assert report["valid_spec"] is False
    assert "invalid_codec_id" in report["errors"]
    assert _rows(csv_path) == []


def test_benchmark_subprocess_uses_shell_false(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        Path(args[0][4]).write_bytes(b"fake")
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr("src.router.external_codec_dry_run.subprocess.run", fake_run)
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_path = _input_file(tmp_path, "a.png")

    run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        param_sets=[{"quality": "40"}],
        timeout_s=5,
    )

    assert calls
    assert calls[0]["shell"] is False


def test_outputs_are_confined_to_out_dir(tmp_path: Path):
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_path = _input_file(tmp_path, "a.png")
    out_dir = tmp_path / "runs"
    csv_path = tmp_path / "measurements.csv"

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=out_dir,
        csv_out=csv_path,
        param_sets=[{"quality": "40"}],
        timeout_s=5,
    )["external_codec_benchmark"]

    row = _rows(csv_path)[0]
    assert report["safety"]["output_confined_to_out_dir"] is True
    assert Path(row["output_path"]).resolve().is_relative_to(out_dir.resolve())


def test_cli_writes_report_and_csv(tmp_path: Path, capsys):
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_path = _input_file(tmp_path, "a.png")
    report_path = tmp_path / "report.json"
    csv_path = tmp_path / "measurements.csv"

    report = main([
        "--spec",
        str(spec_path),
        "--input",
        str(input_path),
        "--out-dir",
        str(tmp_path / "runs"),
        "--param-set",
        '{"quality":"40"}',
        "--timeout-s",
        "5",
        "--out",
        str(report_path),
        "--csv",
        str(csv_path),
    ])
    captured = capsys.readouterr()

    assert report_path.exists()
    assert csv_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(captured.out)["external_codec_benchmark"]["success"] is True


def test_benchmark_does_not_import_router_decision_module(tmp_path: Path):
    sys.modules.pop("src.router.rde_router", None)
    spec_path = _write_spec(tmp_path, _valid_spec(tmp_path))
    input_path = _input_file(tmp_path, "a.png")

    report = run_external_codec_benchmark(
        spec_path,
        input_paths=[input_path],
        out_dir=tmp_path / "runs",
        param_sets=[{"quality": "40"}],
        timeout_s=5,
    )["external_codec_benchmark"]

    assert report["success"] is True
    assert "src.router.rde_router" not in sys.modules
