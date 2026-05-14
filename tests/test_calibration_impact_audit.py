import contextlib
import csv
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.calibration.calibration_bundle import sha256_file
from src.router.calibration.calibration_impact_audit import run_calibration_impact_audit
from src.router.rde_router import main as router_main
from src.router.version import ROUTER_VERSION


def _tmp_dir(name: str) -> Path:
    root = (
        Path(__file__).with_name("_tmp")
        / "calibration_impact_audit"
        / f"{name}_{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_rde_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["codec", "config", "rate", "quality", "energy", "time_ms"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _baseline_rows() -> list[dict[str, object]]:
    return [
        {
            "codec": "JPEG",
            "config": "q=85",
            "rate": 0.40,
            "quality": 95.0,
            "energy": 1.0,
            "time_ms": 10.0,
        },
        {
            "codec": "JXL",
            "config": "d=1.0",
            "rate": 0.80,
            "quality": 95.0,
            "energy": 2.0,
            "time_ms": 20.0,
        },
    ]


def _decision_change_rows() -> list[dict[str, object]]:
    return [
        {
            "codec": "JPEG",
            "config": "q=85",
            "rate": 0.40,
            "quality": 95.0,
            "energy": 1.0,
            "time_ms": 10.0,
        },
        {
            "codec": "JXL",
            "config": "d=1.0",
            "rate": 0.10,
            "quality": 99.0,
            "energy": 0.10,
            "time_ms": 2.0,
        },
    ]


def _write_manifest(
    path: Path,
    calibrated_csv: Path,
    *,
    output_hash: str | None = None,
) -> None:
    payload = {
        "artifact_type": "promoted_calibration_bundle",
        "router_version": "0.18.0",
        "mode": "explicit_opt_in_calibration_apply",
        "source_benchmark": "benchmark.csv",
        "source_calibration": "calibration.json",
        "promotion_profile": "promotion.json",
        "output_csv": str(calibrated_csv),
        "created_at_utc": "2026-05-12T00:00:00+00:00",
        "accepted_scales": [
            {
                "codec": "JXL",
                "config": "d=1.0",
                "axis": "rate",
                "scale": 0.5,
            }
        ],
        "rejected_scales_count": 2,
        "energy_policy": {
            "requires_energy_usable_for_total": True,
            "gpu_only_energy_excluded": True,
        },
        "hashes": {
            "output_csv_sha256": output_hash or sha256_file(calibrated_csv),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _router_args() -> list[str]:
    return [
        "--codec-col",
        "codec",
        "--config-col",
        "config",
        "--rate-col",
        "rate",
        "--quality-col",
        "quality",
        "--energy-col",
        "energy",
        "--time-col",
        "time_ms",
        "--quality-target",
        "preview",
        "--quality-floor",
        "50",
        "--normalization-mode",
        "runtime",
    ]


def _run_router(csv_path: Path, out_path: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        router_main(
            [
                "--csv",
                str(csv_path),
                *_router_args(),
                "--out",
                str(out_path),
            ]
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


def _run_audit(root: Path, calibrated_rows: list[dict[str, object]]) -> dict:
    benchmark_csv = root / "benchmark.csv"
    calibrated_csv = root / "calibrated.csv"
    manifest = root / "manifest.json"
    out = root / "impact.json"
    summary = root / "impact.csv"

    _write_rde_csv(benchmark_csv, _baseline_rows())
    _write_rde_csv(calibrated_csv, calibrated_rows)
    _write_manifest(manifest, calibrated_csv)

    return run_calibration_impact_audit(
        csv_path=str(benchmark_csv),
        calibration_bundle_manifest=str(manifest),
        out_path=str(out),
        summary_out=str(summary),
        router_args=_router_args(),
    )


def test_impact_audit_no_decision_change():
    root = _tmp_dir("no_change")

    report = _run_audit(root, _baseline_rows())

    assert report["router_version"] == ROUTER_VERSION
    assert report["impact"]["decision_changed"] is False
    assert report["impact"]["codec_changed"] is False
    assert report["impact"]["config_changed"] is False
    assert report["baseline"]["selected_codec"] == report["calibrated"]["selected_codec"]
    assert report["baseline"]["selected_config"] == report["calibrated"]["selected_config"]
    assert report["impact"]["delta_j_rde"] == pytest.approx(0.0)


def test_impact_audit_detects_decision_change():
    root = _tmp_dir("decision_change")

    report = _run_audit(root, _decision_change_rows())

    assert report["impact"]["decision_changed"] is True
    assert report["impact"]["codec_changed"] is True
    assert report["baseline"]["selected_codec"] == "JPEG"
    assert report["calibrated"]["selected_codec"] == "JXL"
    assert report["impact"]["delta_j_rde"] is not None


def test_impact_audit_requires_valid_bundle_manifest():
    root = _tmp_dir("missing_manifest")
    benchmark_csv = root / "benchmark.csv"
    _write_rde_csv(benchmark_csv, _baseline_rows())

    with pytest.raises(ValueError, match="manifest not found"):
        run_calibration_impact_audit(
            csv_path=str(benchmark_csv),
            calibration_bundle_manifest=str(root / "missing.json"),
            out_path=str(root / "impact.json"),
            summary_out=str(root / "impact.csv"),
            router_args=_router_args(),
        )


def test_impact_audit_rejects_hash_mismatch():
    root = _tmp_dir("hash_mismatch")
    benchmark_csv = root / "benchmark.csv"
    calibrated_csv = root / "calibrated.csv"
    manifest = root / "manifest.json"

    _write_rde_csv(benchmark_csv, _baseline_rows())
    _write_rde_csv(calibrated_csv, _baseline_rows())
    _write_manifest(manifest, calibrated_csv, output_hash="0" * 64)

    with pytest.raises(ValueError, match="hash mismatch"):
        run_calibration_impact_audit(
            csv_path=str(benchmark_csv),
            calibration_bundle_manifest=str(manifest),
            out_path=str(root / "impact.json"),
            summary_out=str(root / "impact.csv"),
            router_args=_router_args(),
        )


def test_impact_audit_outputs_json_and_csv():
    root = _tmp_dir("outputs")
    benchmark_csv = root / "benchmark.csv"
    calibrated_csv = root / "calibrated.csv"
    manifest = root / "manifest.json"
    out = root / "impact.json"
    summary = root / "impact.csv"

    _write_rde_csv(benchmark_csv, _baseline_rows())
    _write_rde_csv(calibrated_csv, _decision_change_rows())
    _write_manifest(manifest, calibrated_csv)

    run_calibration_impact_audit(
        csv_path=str(benchmark_csv),
        calibration_bundle_manifest=str(manifest),
        out_path=str(out),
        summary_out=str(summary),
        router_args=_router_args(),
    )

    json_report = json.loads(out.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(summary.open("r", encoding="utf-8")))

    assert json_report["mode"] == "calibration_impact_audit"
    assert json_report["calibration_bundle"]["validated"] is True
    assert json_report["calibration_bundle"]["calibrated_csv_sha256"] == sha256_file(
        calibrated_csv
    )
    assert len(csv_rows) == 1
    assert csv_rows[0]["mode"] == "calibration_impact_audit"
    assert csv_rows[0]["bundle_validated"] == "True"


def test_impact_audit_does_not_modify_router_defaults():
    root = _tmp_dir("router_defaults")
    benchmark_csv = root / "benchmark.csv"
    default_report_path = root / "router_default.json"
    _write_rde_csv(benchmark_csv, _baseline_rows())

    audit = _run_audit(root, _decision_change_rows())
    default_report = _run_router(benchmark_csv, default_report_path)
    default_selected = default_report["decision"]["selected"]

    assert default_report["calibration_bundle"] == {"enabled": False}
    assert default_selected["codec"] == audit["baseline"]["selected_codec"]
    assert default_selected["config"] == audit["baseline"]["selected_config"]
