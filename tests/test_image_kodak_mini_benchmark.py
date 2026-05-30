import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.validation import image_kodak_mini_benchmark as bench


def test_cli_help_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validation/image_kodak_mini_benchmark.py", "--help"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--kodak-dir" in completed.stdout
    assert "--run-router" in completed.stdout


def test_dry_run_with_fake_image_does_not_execute_encoder(tmp_path: Path, monkeypatch) -> None:
    kodak_dir = tmp_path / "kodak"
    out_dir = tmp_path / "out"
    kodak_dir.mkdir()
    (kodak_dir / "kodim01.png").write_text("not a real image", encoding="utf-8")

    def fail_encoder(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("encoder should not run during dry-run")

    monkeypatch.setattr(bench, "encode_decode_jpeg", fail_encoder)

    rc = bench.main(
        [
            "--dry-run",
            "--kodak-dir",
            str(kodak_dir),
            "--out-dir",
            str(out_dir),
            "--codecs",
            "jpeg",
        ]
    )

    assert rc == 0
    report = json.loads((out_dir / "kodak_image_rde_mini_report.json").read_text())
    assert report["dry_run"] is True
    assert report["planned_tasks"] == 5


def test_operating_points_for_classical_codecs() -> None:
    points = bench.operating_points_for_codecs(["jpeg", "jxl", "hevc"])
    configs = {(point.codec_key, point.config) for point in points}

    assert [p.config for p in points if p.codec_key == "jpeg"] == [
        "q=30",
        "q=50",
        "q=70",
        "q=85",
        "q=95",
    ]
    assert ("jxl", "distance=1") in configs
    assert ("jxl", "distance=8") in configs
    assert ("hevc", "crf=18") in configs
    assert ("hevc", "crf=38") in configs
    assert len(points) == 15


def test_csv_schema_contains_router_ready_columns() -> None:
    required = {
        "dataset",
        "image_id",
        "image_path",
        "width",
        "height",
        "codec",
        "config",
        "rate_bpp",
        "quality_metric",
        "ssimulacra2",
        "energy_j_per_image",
        "time_ms",
        "compressed_size_bytes",
        "status",
        "error",
        "bpp",
        "energy_per_image_j",
    }

    assert required.issubset(set(bench.CSV_COLUMNS))


def test_report_json_contains_boundaries(tmp_path: Path) -> None:
    kodak_dir = tmp_path / "kodak"
    out_dir = tmp_path / "out"
    kodak_dir.mkdir()

    rc = bench.main(
        [
            "--dry-run",
            "--kodak-dir",
            str(kodak_dir),
            "--out-dir",
            str(out_dir),
            "--codecs",
            "jpeg",
        ]
    )

    assert rc == 0
    report = json.loads((out_dir / "kodak_image_rde_mini_report.json").read_text())
    assert bench.BOUNDARIES == report["boundaries"]
    assert "not the main thesis benchmark" in report["boundaries"][0]


def test_missing_codec_reported_unavailable_not_crash(tmp_path: Path, monkeypatch) -> None:
    kodak_dir = tmp_path / "kodak"
    out_dir = tmp_path / "out"
    kodak_dir.mkdir()
    (kodak_dir / "kodim01.png").write_text("fake", encoding="utf-8")

    monkeypatch.setattr(
        bench,
        "codec_availability_summary",
        lambda codecs: {"jxl": {"available": False, "backend": "mock_missing"}},
    )

    rc = bench.main(
        [
            "--dry-run",
            "--kodak-dir",
            str(kodak_dir),
            "--out-dir",
            str(out_dir),
            "--codecs",
            "jxl",
        ]
    )

    assert rc == 0
    report = json.loads((out_dir / "kodak_image_rde_mini_report.json").read_text())
    assert report["codec_availability"]["jxl"]["available"] is False


def test_config_column_is_config_not_param() -> None:
    assert "config" in bench.CSV_COLUMNS
    assert "param" not in bench.CSV_COLUMNS


def test_script_does_not_write_results_by_default(tmp_path: Path) -> None:
    kodak_dir = tmp_path / "kodak"
    out_dir = tmp_path / "validation_runs" / "mini"
    kodak_dir.mkdir()

    rc = bench.main(
        [
            "--dry-run",
            "--kodak-dir",
            str(kodak_dir),
            "--out-dir",
            str(out_dir),
            "--codecs",
            "jpeg",
        ]
    )

    assert rc == 0
    assert (out_dir / "kodak_image_rde_mini_report.json").exists()
    assert "results" not in out_dir.parts


def test_mocked_end_to_end_writes_router_ready_csv(tmp_path: Path, monkeypatch) -> None:
    kodak_dir = tmp_path / "kodak"
    out_dir = tmp_path / "out"
    kodak_dir.mkdir()
    image = kodak_dir / "kodim01.png"
    image.write_text("fake image placeholder", encoding="utf-8")

    def fake_operation(*args, **kwargs):
        return bench.Measurement(
            width=2,
            height=2,
            bpp=1.0,
            ssimulacra2=80.0,
            compressed_size_bytes=1,
            time_ms=1.5,
        )

    monkeypatch.setattr(bench, "encode_decode_jpeg", fake_operation)
    rc = bench.main(
        [
            "--kodak-dir",
            str(kodak_dir),
            "--out-dir",
            str(out_dir),
            "--codecs",
            "jpeg",
            "--max-images",
            "1",
            "--image-glob",
            "*.png",
            "--energy-backend",
            "none",
            "--warmup",
            "0",
            "--skip-plots",
        ]
    )

    assert rc == 0
    csv_path = out_dir / "kodak_image_rde_mini_router_ready.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["config"] == "q=30"
    assert rows[0]["bpp"] == "1.0"
    assert rows[0]["status"] == "ok"
