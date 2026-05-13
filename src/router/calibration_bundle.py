"""Validation helpers for explicit calibration bundle manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .codec_fingerprints import validate_codec_fingerprints
except ImportError:  # pragma: no cover - direct script fallback
    from codec_fingerprints import validate_codec_fingerprints


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _require_key(data: dict[str, Any], key: str, *, label: str) -> Any:
    if key not in data:
        raise ValueError(f"Calibration bundle manifest missing {label}: {key}")
    return data[key]


def _require_validation_key(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(
            "Calibration bundle validation requires v0.24+ field: "
            f"{key}"
        )
    return data[key]


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ValueError(f"Calibration bundle manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Calibration bundle manifest must contain a JSON object.")

    return data


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise ValueError(f"{label} not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object.")

    return data


def validate_calibration_bundle_manifest(path: str | Path) -> dict[str, Any]:
    """Validate a promoted calibration bundle manifest and its output CSV hash."""

    manifest_path = Path(path)
    manifest = _load_manifest(manifest_path)

    artifact_type = _require_key(
        manifest,
        "artifact_type",
        label="artifact type",
    )
    if artifact_type != "promoted_calibration_bundle":
        raise ValueError(
            "Unsupported calibration bundle artifact_type: "
            f"{artifact_type!r}"
        )

    output_csv = _require_key(manifest, "output_csv", label="output CSV")
    hashes = _require_key(manifest, "hashes", label="hashes")
    if not isinstance(hashes, dict):
        raise ValueError("Calibration bundle manifest hashes must be an object.")

    expected_output_hash = _require_key(
        hashes,
        "output_csv_sha256",
        label="output CSV SHA256",
    )

    if not expected_output_hash:
        raise ValueError("Calibration bundle manifest output CSV hash is empty.")

    calibrated_csv_path = Path(str(output_csv))
    if not calibrated_csv_path.exists():
        raise ValueError(
            "Calibration bundle calibrated CSV not found: "
            f"{calibrated_csv_path}"
        )

    actual_output_hash = sha256_file(calibrated_csv_path)
    if actual_output_hash != str(expected_output_hash):
        raise ValueError(
            "Calibration bundle calibrated CSV hash mismatch: "
            f"expected={expected_output_hash}, actual={actual_output_hash}"
        )

    accepted_scales = manifest.get("accepted_scales", [])
    if not isinstance(accepted_scales, list):
        raise ValueError("Calibration bundle accepted_scales must be a list.")

    energy_policy = manifest.get("energy_policy", {})
    if not isinstance(energy_policy, dict):
        raise ValueError("Calibration bundle energy_policy must be an object.")

    codec_fingerprint_validation = validate_codec_fingerprints(
        manifest.get("codec_fingerprints")
    )

    return {
        "enabled": True,
        "manifest_path": str(manifest_path),
        "calibrated_csv_path": str(calibrated_csv_path),
        "calibrated_csv_sha256": actual_output_hash,
        "validated": True,
        "applied_scales_count": len(accepted_scales),
        "rejected_scales_count": int(
            manifest.get("rejected_scales_count", 0) or 0
        ),
        "energy_policy": (
            "usable_total_only"
            if energy_policy.get("requires_energy_usable_for_total") is True
            else "unspecified"
        ),
        "source": "explicit_calibration_bundle_manifest",
        "router_version": manifest.get("router_version"),
        "created_at_utc": manifest.get("created_at_utc"),
        "codec_fingerprint_validation": codec_fingerprint_validation,
        "manifest": manifest,
    }


def validate_calibration_bundle_validation(
    path: str | Path,
    *,
    bundle_manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate an explicit shadow decision validation report for bundle use."""

    validation_path = Path(path)
    manifest_path = Path(bundle_manifest_path)
    validation = _load_json_object(
        validation_path,
        label="Calibration bundle validation",
    )

    mode = _require_validation_key(validation, "mode")
    if mode != "shadow_decision_validation_only":
        raise ValueError(
            "Unsupported calibration bundle validation mode: "
            f"{mode!r}"
        )

    accepted = _require_validation_key(validation, "accepted")
    if not isinstance(accepted, bool):
        raise ValueError(
            "Calibration bundle validation accepted must be a boolean."
        )

    rejection_reasons = _require_validation_key(validation, "rejection_reasons")
    if not isinstance(rejection_reasons, list):
        raise ValueError(
            "Calibration bundle validation rejection_reasons must be a list."
        )

    for key in (
        "decision_count",
        "changed_decision_count",
        "decision_churn_rate",
        "relative_cost_improvement",
        "validated_comparison_path",
        "validated_comparison_sha256",
        "candidate_calibration_bundle_manifest_sha256",
        "candidate_calibrated_csv_sha256",
    ):
        _require_validation_key(validation, key)

    validation_sha256 = sha256_file(validation_path)
    bundle_manifest_sha256 = sha256_file(manifest_path)
    validation_bundle_manifest_sha256 = str(
        validation["candidate_calibration_bundle_manifest_sha256"]
    )

    if validation_bundle_manifest_sha256 != bundle_manifest_sha256:
        raise ValueError(
            "Calibration bundle validation bundle manifest hash mismatch: "
            f"validation={validation_bundle_manifest_sha256}, "
            f"bundle={bundle_manifest_sha256}"
        )

    return {
        "enabled": True,
        "validation_path": str(validation_path),
        "validation_sha256": validation_sha256,
        "accepted": accepted,
        "mode": mode,
        "rejection_reasons": list(rejection_reasons),
        "decision_count": validation.get("decision_count"),
        "changed_decision_count": validation.get("changed_decision_count"),
        "decision_churn_rate": validation.get("decision_churn_rate"),
        "relative_cost_improvement": validation.get(
            "relative_cost_improvement"
        ),
        "validated_comparison_path": validation.get(
            "validated_comparison_path"
        ),
        "validated_comparison_sha256": validation.get(
            "validated_comparison_sha256"
        ),
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "validation_bundle_manifest_sha256": (
            validation_bundle_manifest_sha256
        ),
        "candidate_calibrated_csv_sha256": validation.get(
            "candidate_calibrated_csv_sha256"
        ),
        "integrity_match": True,
        "source": "explicit_shadow_decision_validation",
    }
