import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.calibration_bundle import (
    sha256_file,
    validate_calibration_bundle_manifest,
)


def _tmp_dir(name: str) -> Path:
    root = Path(__file__).with_name("_tmp") / "calibration_bundle_validation" / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_csv(path: Path, *, body: str = "codec,config,rate,quality,energy\nJPEG,q=85,1,90,0.1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _write_manifest(path: Path, csv_path: Path, *, output_hash: str | None = None) -> None:
    payload = {
        "artifact_type": "promoted_calibration_bundle",
        "router_version": "0.18.0",
        "mode": "explicit_opt_in_calibration_apply",
        "source_benchmark": "benchmark.csv",
        "source_calibration": "calibration.json",
        "promotion_profile": "promotion.json",
        "output_csv": str(csv_path),
        "created_at_utc": "2026-05-12T00:00:00+00:00",
        "accepted_scales": [
            {
                "codec": "JPEG",
                "config": "q=85",
                "axis": "rate",
                "scale": 1.1,
            }
        ],
        "rejected_scales_count": 2,
        "energy_policy": {
            "requires_energy_usable_for_total": True,
            "gpu_only_energy_excluded": True,
        },
        "hashes": {
            "output_csv_sha256": output_hash or sha256_file(csv_path),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bundle_manifest_validates_output_hash():
    root = _tmp_dir("valid")
    csv_path = root / "calibrated.csv"
    manifest = root / "manifest.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path)

    report = validate_calibration_bundle_manifest(manifest)

    assert report["enabled"] is True
    assert report["validated"] is True
    assert report["calibrated_csv_path"] == str(csv_path)
    assert report["calibrated_csv_sha256"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert report["applied_scales_count"] == 1
    assert report["rejected_scales_count"] == 2
    assert report["energy_policy"] == "usable_total_only"


def test_bundle_manifest_rejects_hash_mismatch():
    root = _tmp_dir("hash_mismatch")
    csv_path = root / "calibrated.csv"
    manifest = root / "manifest.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path, output_hash="0" * 64)

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_calibration_bundle_manifest(manifest)


def test_bundle_manifest_rejects_missing_calibrated_csv():
    root = _tmp_dir("missing_csv")
    csv_path = root / "missing.csv"
    manifest = root / "manifest.json"
    payload = {
        "artifact_type": "promoted_calibration_bundle",
        "output_csv": str(csv_path),
        "accepted_scales": [],
        "hashes": {
            "output_csv_sha256": "0" * 64,
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="calibrated CSV not found"):
        validate_calibration_bundle_manifest(manifest)


def test_bundle_manifest_rejects_incomplete_manifest():
    root = _tmp_dir("incomplete")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"artifact_type": "promoted_calibration_bundle"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing"):
        validate_calibration_bundle_manifest(manifest)
