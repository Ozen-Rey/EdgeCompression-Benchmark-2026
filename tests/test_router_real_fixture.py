import json
import hashlib
from pathlib import Path

import pytest

from src.router.rde_router import main
from src.router.version import DOMAIN_SUPPORT, FEATURE_LEVEL, ROUTER_VERSION


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "router_real_fixture"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle_manifest(path: Path, calibrated_csv: Path, *, hash_value: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
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
                        "codec": "HEVC",
                        "config": "crf=15",
                        "axis": "rate",
                        "scale": 1.0,
                    }
                ],
                "rejected_scales_count": 2,
                "energy_policy": {
                    "requires_energy_usable_for_total": True,
                    "gpu_only_energy_excluded": True,
                },
                "hashes": {
                    "output_csv_sha256": hash_value or _sha256(calibrated_csv),
                },
            }
        ),
        encoding="utf-8",
    )


def test_router_end_to_end_on_real_small_image_fixture():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )

    out_path = _tmp_path("real_fixture_report.json")

    main(
        [
            "--csv",
            str(fixture),
            "--codec-col",
            "codec",
            "--config-col",
            "param",
            "--rate-col",
            "bpp",
            "--quality-col",
            "ssimulacra2",
            "--energy-col",
            "energy_per_image_j",
            "--time-col",
            "time_ms",
            "--available-codecs",
            "JPEG,JXL,HEVC",
            "--quality-target",
            "very-high",
            "--quality-floor",
            "90",
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    selected = report["decision"]["selected"]

    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
    assert report["decision"]["decision_mode"] == "safe"
    assert report["router_version"] == ROUTER_VERSION
    assert report["feature_level"] == FEATURE_LEVEL
    assert report["domain_support"] == DOMAIN_SUPPORT
    assert (
        report["energy_provenance"]["local_energy_measurement"]
        == "hardware_backend_or_fallback"
    )
    assert report["energy_provenance"]["current_method"] == "benchmark_energy"
    assert report["energy_provenance"]["energy_backend"] == "benchmark_csv"
    assert report["energy_provenance"]["energy_is_measured"] is False
    assert report["calibration_bundle"] == {"enabled": False}


def test_router_with_valid_bundle_reports_bundle_provenance():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("bundle_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("bundle_manifest.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    out_path = _tmp_path("bundle_report.json")

    main(
        [
            "--csv",
            str(fixture),
            "--calibration-bundle-manifest",
            str(manifest),
            "--codec-col",
            "codec",
            "--config-col",
            "param",
            "--rate-col",
            "bpp",
            "--quality-col",
            "ssimulacra2",
            "--energy-col",
            "energy_per_image_j",
            "--time-col",
            "time_ms",
            "--available-codecs",
            "JPEG,JXL,HEVC",
            "--quality-target",
            "very-high",
            "--quality-floor",
            "90",
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    bundle = report["calibration_bundle"]

    assert bundle["enabled"] is True
    assert bundle["manifest_path"] == str(manifest)
    assert bundle["calibrated_csv_path"] == str(calibrated_csv)
    assert bundle["calibrated_csv_sha256"] == _sha256(calibrated_csv)
    assert bundle["validated"] is True
    assert bundle["applied_scales_count"] == 1
    assert bundle["rejected_scales_count"] == 2
    assert bundle["energy_policy"] == "usable_total_only"
    assert bundle["source"] == "explicit_calibration_bundle_manifest"
    assert report["csv"] == str(calibrated_csv)


def test_router_with_invalid_bundle_fails_controlled():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("bundle_invalid_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("bundle_invalid_manifest.json")
    _write_bundle_manifest(manifest, calibrated_csv, hash_value="0" * 64)
    out_path = _tmp_path("bundle_invalid_report.json")

    with pytest.raises(ValueError, match="hash mismatch"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-manifest",
                str(manifest),
                "--codec-col",
                "codec",
                "--config-col",
                "param",
                "--rate-col",
                "bpp",
                "--quality-col",
                "ssimulacra2",
                "--energy-col",
                "energy_per_image_j",
                "--out",
                str(out_path),
            ]
        )
