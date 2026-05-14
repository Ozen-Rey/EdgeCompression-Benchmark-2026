import json
from pathlib import Path

import pytest

from src.router.calibration.calibration_bundle import validate_calibration_bundle_manifest
from src.router.codecs.codec_fingerprints import (
    CalibrationStalenessError,
    validate_codec_fingerprints,
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path) -> None:
    path.write_text(
        "codec,config,rate,quality,energy\nJPEG,q=85,1,90,0.1\n",
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    csv_path: Path,
    *,
    fingerprints: dict | None,
) -> None:
    payload = {
        "artifact_type": "promoted_calibration_bundle",
        "router_version": "0.32.0",
        "mode": "explicit_opt_in_calibration_apply",
        "source_benchmark": "benchmark.csv",
        "source_calibration": "calibration.json",
        "promotion_profile": "promotion.json",
        "output_csv": str(csv_path),
        "created_at_utc": "2026-05-13T00:00:00+00:00",
        "accepted_scales": [
            {
                "codec": "JXL",
                "config": "d=1.0",
                "axis": "rate",
                "scale": 1.0,
            }
        ],
        "rejected_scales_count": 0,
        "energy_policy": {
            "requires_energy_usable_for_total": True,
        },
        "hashes": {
            "output_csv_sha256": _sha256(csv_path),
        },
    }
    if fingerprints is not None:
        payload["codec_fingerprints"] = fingerprints

    path.write_text(json.dumps(payload), encoding="utf-8")


def _fingerprint(binary: Path, *, version: str = "codec 1.0") -> dict:
    return {
        "backend": "cjxl",
        "version": version,
        "binary_path": str(binary),
        "binary_sha256": _sha256(binary),
        "available": True,
    }


def test_manifest_with_matching_codec_fingerprint_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    csv_path = tmp_path / "calibrated.csv"
    binary = tmp_path / "cjxl.exe"
    manifest = tmp_path / "manifest.json"
    _write_csv(csv_path)
    binary.write_bytes(b"fake-cjxl")
    monkeypatch.setattr(
        "src.router.codecs.codec_fingerprints._version_from_binary",
        lambda path, backend: "codec 1.0",
    )
    _write_manifest(
        manifest,
        csv_path,
        fingerprints={"JXL": _fingerprint(binary)},
    )

    report = validate_calibration_bundle_manifest(manifest)

    validation = report["codec_fingerprint_validation"]
    assert validation["enabled"] is True
    assert validation["validated"] is True
    assert validation["validated_codecs"] == ["JXL"]
    assert validation["mismatches"] == []


def test_hash_mismatch_rejects_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    csv_path = tmp_path / "calibrated.csv"
    binary = tmp_path / "cjxl.exe"
    manifest = tmp_path / "manifest.json"
    _write_csv(csv_path)
    binary.write_bytes(b"fake-cjxl")
    expected = _fingerprint(binary)
    expected["binary_sha256"] = "0" * 64
    monkeypatch.setattr(
        "src.router.codecs.codec_fingerprints._version_from_binary",
        lambda path, backend: "codec 1.0",
    )
    _write_manifest(manifest, csv_path, fingerprints={"JXL": expected})

    with pytest.raises(CalibrationStalenessError, match="binary_sha256"):
        validate_calibration_bundle_manifest(manifest)


def test_version_mismatch_rejects_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    csv_path = tmp_path / "calibrated.csv"
    binary = tmp_path / "cjxl.exe"
    manifest = tmp_path / "manifest.json"
    _write_csv(csv_path)
    binary.write_bytes(b"fake-cjxl")
    monkeypatch.setattr(
        "src.router.codecs.codec_fingerprints._version_from_binary",
        lambda path, backend: "codec 2.0",
    )
    _write_manifest(
        manifest,
        csv_path,
        fingerprints={"JXL": _fingerprint(binary, version="codec 1.0")},
    )

    with pytest.raises(CalibrationStalenessError, match="version"):
        validate_calibration_bundle_manifest(manifest)


def test_missing_binary_rejects_bundle(tmp_path: Path):
    csv_path = tmp_path / "calibrated.csv"
    manifest = tmp_path / "manifest.json"
    missing_binary = tmp_path / "missing-cjxl.exe"
    _write_csv(csv_path)
    _write_manifest(
        manifest,
        csv_path,
        fingerprints={
            "JXL": {
                "backend": "cjxl",
                "version": "codec 1.0",
                "binary_path": str(missing_binary),
                "binary_sha256": "1" * 64,
                "available": True,
            }
        },
    )

    with pytest.raises(CalibrationStalenessError, match="available"):
        validate_calibration_bundle_manifest(manifest)


def test_legacy_manifest_without_fingerprints_reports_disabled(tmp_path: Path):
    csv_path = tmp_path / "calibrated.csv"
    manifest = tmp_path / "manifest.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path, fingerprints=None)

    report = validate_calibration_bundle_manifest(manifest)

    assert report["codec_fingerprint_validation"] == {
        "enabled": False,
        "validated": False,
        "reason": "manifest_without_codec_fingerprints",
        "validated_codecs": [],
        "mismatches": [],
    }


def test_empty_codec_fingerprints_report_disabled_not_validated(tmp_path: Path):
    csv_path = tmp_path / "calibrated.csv"
    manifest = tmp_path / "manifest.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path, fingerprints={})

    report = validate_calibration_bundle_manifest(manifest)

    assert report["codec_fingerprint_validation"] == {
        "enabled": False,
        "validated": False,
        "reason": "empty_codec_fingerprints",
        "validated_codecs": [],
        "mismatches": [],
    }


def test_validate_codec_fingerprints_rejects_non_object():
    with pytest.raises(ValueError, match="must be an object"):
        validate_codec_fingerprints([])
