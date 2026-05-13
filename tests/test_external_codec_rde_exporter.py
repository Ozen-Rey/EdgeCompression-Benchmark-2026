import csv
import json
import struct
import subprocess
from pathlib import Path

from src.router.external_codec_rde_exporter import (
    RDE_COLUMNS,
    export_external_codec_rde,
    main,
)


RAW_COLUMNS = [
    "codec_id",
    "domain",
    "input_path",
    "input_id",
    "param_set_id",
    "param_json",
    "encode_success",
    "decode_success",
    "output_path",
    "reconstruction_path",
    "output_size_bytes",
    "encode_time_ms",
    "decode_time_ms",
    "rate_metric",
    "rate_value",
    "energy_j",
    "energy_provenance_tier",
    "error",
]


def _spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.json"
    payload = {
        "schema_version": "0.36.0",
        "codec_id": "fake_codec",
        "display_name": "Fake Codec",
        "domain": "image",
        "family": "classical",
        "runtime": {
            "type": "external_command",
            "executable": "fake",
            "max_runtime_seconds": 30,
        },
        "version_probe": {
            "command": ["{executable}", "--version"],
        },
        "encode": {
            "command_template": [
                "{executable}",
                "--input",
                "{input}",
                "--output",
                "{output}",
                "--quality",
                "{quality}",
            ],
        },
        "decode": {
            "available": False,
            "command_template": ["{executable}", "--decode", "{input}", "{output}"],
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
            "metric": "ssimulacra2",
            "direction": "higher_is_better",
        },
        "measurement": {
            "time": "benchmark_wall_clock",
        },
        "requirements": {
            "binaries": ["fake"],
        },
        "security": {
            "allow_shell": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _png(path: Path, width: int, height: int) -> Path:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )
    return path


def _raw_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "raw.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for row in rows:
            payload = {
                "codec_id": "fake_codec",
                "domain": "image",
                "input_path": str(_png(tmp_path / f"{row.get('input_id', 'a')}.png", 4, 2)),
                "input_id": row.get("input_id", "a"),
                "param_set_id": row.get("param_set_id", "p000"),
                "param_json": row.get("param_json", '{"quality": "40"}'),
                "encode_success": row.get("encode_success", "True"),
                "decode_success": row.get("decode_success", ""),
                "output_path": row.get("output_path", "encoded.fake"),
                "reconstruction_path": row.get("reconstruction_path", ""),
                "output_size_bytes": row.get("output_size_bytes", "16"),
                "encode_time_ms": row.get("encode_time_ms", "2.5"),
                "decode_time_ms": row.get("decode_time_ms", ""),
                "rate_metric": row.get("rate_metric", "output_size_bytes"),
                "rate_value": row.get("rate_value", "16"),
                "energy_j": row.get("energy_j", ""),
                "energy_provenance_tier": row.get("energy_provenance_tier", "unknown"),
                "error": row.get("error", ""),
            }
            writer.writerow(payload)
    return path


def _quality_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "quality.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["input_id", "param_set_id", "ssimulacra2"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _export(tmp_path: Path, raw: Path, **kwargs):
    out = tmp_path / "external_rde.csv"
    report = tmp_path / "report.json"
    result = export_external_codec_rde(
        spec_path=_spec(tmp_path),
        raw_csv=raw,
        out=out,
        report_out=report,
        **kwargs,
    )
    return result["external_codec_rde_export"], _read(out), json.loads(report.read_text())


def test_raw_csv_with_two_successes_exports_two_rde_rows(tmp_path: Path):
    raw = _raw_csv(tmp_path, [
        {"input_id": "a", "param_set_id": "p000"},
        {"input_id": "b", "param_set_id": "p001", "param_json": '{"quality": "80"}'},
    ])

    report, rows, _ = _export(tmp_path, raw)

    assert len(rows) == 2
    assert report["raw_rows"] == 2
    assert report["successful_raw_rows"] == 2
    assert report["failed_raw_rows"] == 0
    assert report["exported_rows"] == 2
    assert set(rows[0]) == set(RDE_COLUMNS)
    assert rows[0]["codec"] == "fake_codec"
    assert rows[0]["success"] == "True"


def test_failed_raw_row_is_counted_and_not_exported(tmp_path: Path):
    raw = _raw_csv(tmp_path, [
        {"input_id": "a"},
        {
            "input_id": "b",
            "encode_success": "False",
            "error": "encode_timeout",
        },
    ])

    report, rows, _ = _export(tmp_path, raw)

    assert len(rows) == 1
    assert report["raw_rows"] == 2
    assert report["successful_raw_rows"] == 1
    assert report["failed_raw_rows"] == 1
    assert "failed_raw_rows_not_exported" in report["warnings"]


def test_image_bpp_is_calculated_from_png_dimensions(tmp_path: Path):
    raw = _raw_csv(tmp_path, [{"input_id": "a", "output_size_bytes": "16"}])

    report, rows, _ = _export(tmp_path, raw, rate_mode="image_bpp")

    assert report["rate_mode"] == "image_bpp"
    assert float(rows[0]["rate"]) == 16.0


def test_missing_quality_is_not_invented_and_router_not_ready(tmp_path: Path):
    raw = _raw_csv(tmp_path, [{"input_id": "a"}])

    report, rows, _ = _export(tmp_path, raw)

    assert rows[0]["quality"] == ""
    assert report["quality_available"] is False
    assert report["router_ready"] is False


def test_missing_energy_is_not_invented_and_tier_unknown(tmp_path: Path):
    raw = _raw_csv(tmp_path, [{"input_id": "a"}])

    report, rows, _ = _export(tmp_path, raw)

    assert rows[0]["energy"] == ""
    assert rows[0]["energy_provenance_tier"] == "unknown"
    assert report["energy_available"] is False


def test_quality_csv_is_joined_by_input_and_param_set(tmp_path: Path):
    raw = _raw_csv(tmp_path, [
        {"input_id": "a", "param_set_id": "p000", "energy_j": "0.1"},
    ])
    quality = _quality_csv(tmp_path, [
        {"input_id": "a", "param_set_id": "p000", "ssimulacra2": "91.5"},
    ])

    report, rows, _ = _export(
        tmp_path,
        raw,
        quality_csv=quality,
        quality_column="ssimulacra2",
    )

    assert rows[0]["quality"] == "91.5"
    assert report["quality_available"] is True
    assert report["energy_available"] is True
    assert report["router_ready"] is True


def test_exporter_does_not_invoke_subprocess(tmp_path: Path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("exporter must not execute subprocess")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    raw = _raw_csv(tmp_path, [{"input_id": "a"}])

    report, rows, _ = _export(tmp_path, raw)

    assert len(rows) == 1
    assert report["safety"]["codec_executed"] is False


def test_report_safety_flags_are_execution_false(tmp_path: Path):
    raw = _raw_csv(tmp_path, [{"input_id": "a"}])

    report, _, _ = _export(tmp_path, raw)

    assert report["safety"] == {
        "codec_executed": False,
        "encode_executed": False,
        "decode_executed": False,
        "router_candidate_registered": False,
    }


def test_cli_writes_csv_and_report_json(tmp_path: Path, capsys):
    raw = _raw_csv(tmp_path, [{"input_id": "a"}])
    spec = _spec(tmp_path)
    out = tmp_path / "external_rde.csv"
    report_path = tmp_path / "report.json"

    result = main([
        "--spec",
        str(spec),
        "--raw-csv",
        str(raw),
        "--out",
        str(out),
        "--report-out",
        str(report_path),
    ])
    captured = capsys.readouterr()

    assert out.exists()
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == result
    assert json.loads(captured.out)["external_codec_rde_export"]["exported_rows"] == 1


def test_output_csv_has_future_router_compatible_columns_but_can_be_not_ready(tmp_path: Path):
    raw = _raw_csv(tmp_path, [{"input_id": "a"}])

    report, rows, _ = _export(tmp_path, raw)

    assert list(rows[0].keys()) == RDE_COLUMNS
    assert report["structurally_valid_csv"] is True
    assert report["router_ready"] is False
