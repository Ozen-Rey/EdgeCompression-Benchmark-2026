"""Explicit external codec R-D-E manifest validation and loading."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .external_codec_spec import load_external_codec_spec
    from .rde_database import RDEPoint, load_rde_points
except ImportError:  # pragma: no cover - direct script fallback
    from external_codec_spec import load_external_codec_spec
    from rde_database import RDEPoint, load_rde_points


MANIFEST_SCHEMA_VERSION = "external_codec_router_manifest_v1"


class ExternalCodecManifestError(ValueError):
    """Raised when an explicit external codec manifest is not safe to consume."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_external_codec_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ExternalCodecManifestError(
            f"External codec manifest not found: {manifest_path}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalCodecManifestError(
            f"External codec manifest JSON invalid: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ExternalCodecManifestError("External codec manifest must be an object.")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExternalCodecManifestError(
            "External codec manifest schema_version must be "
            f"{MANIFEST_SCHEMA_VERSION}."
        )

    codec_id = _required_string(manifest, "codec_id")
    spec_path = _resolve_manifest_path(manifest_path, _required_string(manifest, "spec_path"))
    probe_report_path = _resolve_manifest_path(
        manifest_path,
        _required_string(manifest, "probe_report"),
    )
    export_report_path = _resolve_manifest_path(
        manifest_path,
        _required_string(manifest, "rde_export_report"),
    )
    rde_csv_path = _resolve_manifest_path(manifest_path, _required_string(manifest, "rde_csv"))

    files = {
        "spec_sha256": spec_path,
        "probe_report_sha256": probe_report_path,
        "rde_export_report_sha256": export_report_path,
        "rde_csv_sha256": rde_csv_path,
    }

    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise ExternalCodecManifestError(
            "External codec manifest must contain hashes object."
        )

    file_reports: dict[str, dict[str, Any]] = {}
    for hash_key, file_path in files.items():
        if not file_path.exists():
            raise ExternalCodecManifestError(f"External codec file not found: {file_path}")
        expected = hashes.get(hash_key)
        if not isinstance(expected, str) or not expected:
            raise ExternalCodecManifestError(f"Missing manifest hash: {hash_key}")
        actual = sha256_file(file_path)
        if actual != expected:
            raise ExternalCodecManifestError(
                f"External codec manifest hash mismatch for {hash_key}: "
                f"expected={expected} actual={actual}"
            )
        file_reports[hash_key] = {
            "path": str(file_path),
            "sha256": actual,
        }

    spec = load_external_codec_spec(spec_path)
    spec_codec = spec.get("codec_id")
    if spec_codec != codec_id:
        raise ExternalCodecManifestError(
            f"External codec codec_id mismatch: manifest={codec_id} spec={spec_codec}"
        )

    probe_report = _load_json_object(probe_report_path, "probe_report")
    _validate_probe_report(probe_report, codec_id)

    export_report = _load_json_object(export_report_path, "rde_export_report")
    export_body = export_report.get("external_codec_rde_export")
    if not isinstance(export_body, dict):
        raise ExternalCodecManifestError(
            "External codec R-D-E export report missing external_codec_rde_export."
        )
    if export_body.get("codec_id") != codec_id:
        raise ExternalCodecManifestError(
            "External codec codec_id mismatch between manifest and export report."
        )
    if export_body.get("router_ready") is not True:
        raise ExternalCodecManifestError(
            "External codec R-D-E export report router_ready=false."
        )

    row_report = _validate_rde_csv(rde_csv_path, codec_id)

    return {
        "manifest_path": str(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "codec_id": codec_id,
        "spec_path": str(spec_path),
        "probe_report": str(probe_report_path),
        "rde_export_report": str(export_report_path),
        "rde_csv": str(rde_csv_path),
        "rde_csv_path": str(rde_csv_path),
        "created_at_utc": manifest.get("created_at_utc"),
        "router_ready": True,
        "hashes": {
            key: value["sha256"]
            for key, value in file_reports.items()
        },
        "files": file_reports,
        "rows": row_report,
        "source": "explicit_external_codec_manifest",
    }


def load_external_codec_points(
    manifest_paths: list[str | Path],
) -> tuple[list[RDEPoint], dict[str, Any]]:
    if not manifest_paths:
        return [], {
            "enabled": False,
        }

    manifests: list[dict[str, Any]] = []
    points: list[RDEPoint] = []

    for manifest_path in manifest_paths:
        validation = validate_external_codec_manifest(manifest_path)
        loaded = load_rde_points(
            validation["rde_csv_path"],
            codec_col="codec",
            config_col="param",
            rate_col="rate",
            quality_col="quality",
            energy_col="energy",
            time_col="time_ms",
        )
        for point in loaded:
            point.raw.update({
                "source": "external_codec_manifest",
                "external_codec_id": validation["codec_id"],
                "external_manifest_path": validation["manifest_path"],
                "external_router_ready": True,
            })
        points.extend(loaded)
        manifests.append({
            **validation,
            "loaded_rows": len(loaded),
        })

    report = {
        "enabled": True,
        "source": "explicit_external_codec_manifest",
        "manifests": manifests,
        "loaded_codecs": sorted({item["codec_id"] for item in manifests}),
        "loaded_rows": len(points),
        "validation": {
            "validated": True,
            "num_manifests": len(manifests),
        },
    }
    return points, report


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExternalCodecManifestError(
            f"External codec manifest missing required field: {key}"
        )
    return value


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExternalCodecManifestError(f"External codec {label} JSON invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise ExternalCodecManifestError(f"External codec {label} must be an object.")
    return data


def _validate_probe_report(probe_report: dict[str, Any], codec_id: str) -> None:
    body = probe_report.get("external_codec_probe")
    if not isinstance(body, dict):
        raise ExternalCodecManifestError(
            "External codec probe report missing external_codec_probe."
        )
    if body.get("codec_id") != codec_id:
        raise ExternalCodecManifestError(
            "External codec codec_id mismatch between manifest and probe report."
        )
    if body.get("valid_spec") is not True:
        raise ExternalCodecManifestError("External codec probe report valid_spec=false.")


def _validate_rde_csv(path: Path, codec_id: str) -> dict[str, Any]:
    required = {
        "codec",
        "param",
        "input_id",
        "input_path",
        "rate",
        "quality",
        "energy",
        "time_ms",
        "energy_provenance_tier",
        "measurement_provenance",
        "success",
        "error",
        "source_raw_csv",
    }
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(required - fieldnames)
            if missing:
                raise ExternalCodecManifestError(
                    "External codec R-D-E CSV missing columns: " + ",".join(missing)
                )

            rows = list(reader)
    except ExternalCodecManifestError:
        raise
    except Exception as exc:
        raise ExternalCodecManifestError(
            f"External codec R-D-E CSV invalid: {exc}"
        ) from exc

    if not rows:
        raise ExternalCodecManifestError("External codec R-D-E CSV contains no rows.")

    for index, row in enumerate(rows, start=1):
        if row.get("codec") != codec_id:
            raise ExternalCodecManifestError(
                f"External codec codec_id mismatch in R-D-E CSV row {index}."
            )
        if str(row.get("success", "")).strip().lower() not in {"true", "1", "yes"}:
            raise ExternalCodecManifestError(
                f"External codec R-D-E CSV row {index} success is not true."
            )
        for column in ("rate", "quality", "energy", "time_ms"):
            if row.get(column) in (None, ""):
                raise ExternalCodecManifestError(
                    f"External codec R-D-E CSV row {index} missing {column}."
                )

    return {
        "total_rows": len(rows),
        "codec_id": codec_id,
        "columns": sorted(fieldnames),
    }
