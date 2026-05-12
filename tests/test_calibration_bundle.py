import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.calibration_bundle import (
    sha256_file,
    validate_calibration_bundle_manifest,
    validate_calibration_bundle_validation,
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


def _write_validation(
    path: Path,
    *,
    bundle_manifest: Path | None = None,
    calibrated_csv: Path | None = None,
    accepted: bool = True,
    mode: str = "shadow_decision_validation_only",
    candidate_manifest_hash: str | None = None,
    include_hashes: bool = True,
) -> None:
    payload = {
        "mode": mode,
        "router_version": "0.22.0",
        "comparison": "shadow_decision_comparison.json",
        "validated_comparison_path": "shadow_decision_comparison.json",
        "validated_comparison_sha256": "c" * 64,
        "accepted": accepted,
        "decision_count": 3,
        "changed_decision_count": 1,
        "decision_churn_rate": 1 / 3,
        "relative_cost_improvement": 0.10,
        "rejection_reasons": [] if accepted else ["candidate_cost_regression"],
    }
    if include_hashes:
        payload["candidate_calibration_bundle_manifest_sha256"] = (
            candidate_manifest_hash
            or (sha256_file(bundle_manifest) if bundle_manifest else "b" * 64)
        )
        payload["candidate_calibrated_csv_sha256"] = (
            sha256_file(calibrated_csv) if calibrated_csv else "d" * 64
        )
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


def test_bundle_validation_accepts_shadow_decision_validation_report():
    root = _tmp_dir("validation_accepted")
    csv_path = root / "calibrated.csv"
    manifest = root / "manifest.json"
    validation = root / "validation.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path)
    _write_validation(
        validation,
        bundle_manifest=manifest,
        calibrated_csv=csv_path,
    )

    report = validate_calibration_bundle_validation(
        validation,
        bundle_manifest_path=manifest,
    )

    assert report["enabled"] is True
    assert report["validation_path"] == str(validation)
    assert report["validation_sha256"] == sha256_file(validation)
    assert report["mode"] == "shadow_decision_validation_only"
    assert report["accepted"] is True
    assert report["decision_count"] == 3
    assert report["changed_decision_count"] == 1
    assert report["decision_churn_rate"] == 1 / 3
    assert report["relative_cost_improvement"] == 0.10
    assert report["rejection_reasons"] == []
    assert report["validated_comparison_path"] == "shadow_decision_comparison.json"
    assert report["validated_comparison_sha256"] == "c" * 64
    assert report["bundle_manifest_sha256"] == sha256_file(manifest)
    assert report["validation_bundle_manifest_sha256"] == sha256_file(manifest)
    assert report["candidate_calibrated_csv_sha256"] == sha256_file(csv_path)
    assert report["integrity_match"] is True
    assert report["source"] == "explicit_shadow_decision_validation"


def test_bundle_validation_preserves_rejected_report_for_router_gate():
    root = _tmp_dir("validation_rejected")
    csv_path = root / "calibrated.csv"
    manifest = root / "manifest.json"
    validation = root / "validation.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path)
    _write_validation(
        validation,
        bundle_manifest=manifest,
        calibrated_csv=csv_path,
        accepted=False,
    )

    report = validate_calibration_bundle_validation(
        validation,
        bundle_manifest_path=manifest,
    )

    assert report["accepted"] is False
    assert report["rejection_reasons"] == ["candidate_cost_regression"]


def test_bundle_validation_rejects_missing_or_malformed_report():
    root = _tmp_dir("validation_missing")

    with pytest.raises(ValueError, match="validation not found"):
        validate_calibration_bundle_validation(
            root / "missing.json",
            bundle_manifest_path=root / "manifest.json",
        )

    malformed = root / "malformed.json"
    malformed.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        validate_calibration_bundle_validation(
            malformed,
            bundle_manifest_path=root / "manifest.json",
        )


def test_bundle_validation_rejects_incomplete_report():
    root = _tmp_dir("validation_incomplete")
    validation = root / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "mode": "shadow_decision_validation_only",
                "accepted": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires"):
        validate_calibration_bundle_validation(
            validation,
            bundle_manifest_path=root / "manifest.json",
        )


def test_bundle_validation_rejects_wrong_mode():
    root = _tmp_dir("validation_wrong_mode")
    csv_path = root / "calibrated.csv"
    manifest = root / "manifest.json"
    validation = root / "validation.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path)
    _write_validation(
        validation,
        bundle_manifest=manifest,
        calibrated_csv=csv_path,
        mode="shadow_decision_comparison_only",
    )

    with pytest.raises(ValueError, match="Unsupported"):
        validate_calibration_bundle_validation(
            validation,
            bundle_manifest_path=manifest,
        )


def test_bundle_validation_rejects_bundle_hash_mismatch():
    root = _tmp_dir("validation_hash_mismatch")
    csv_path = root / "calibrated.csv"
    manifest = root / "manifest.json"
    validation = root / "validation.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path)
    _write_validation(
        validation,
        calibrated_csv=csv_path,
        candidate_manifest_hash="0" * 64,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_calibration_bundle_validation(
            validation,
            bundle_manifest_path=manifest,
        )


def test_bundle_validation_rejects_legacy_report_without_hashes():
    root = _tmp_dir("validation_legacy")
    csv_path = root / "calibrated.csv"
    manifest = root / "manifest.json"
    validation = root / "validation.json"
    _write_csv(csv_path)
    _write_manifest(manifest, csv_path)
    _write_validation(validation, include_hashes=False)

    with pytest.raises(ValueError, match="v0.24"):
        validate_calibration_bundle_validation(
            validation,
            bundle_manifest_path=manifest,
        )
