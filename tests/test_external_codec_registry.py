import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.router.external_codec_registry import (
    ExternalCodecManifestError,
    load_external_codec_points,
    validate_external_codec_manifest,
)
from src.router.rde_router import main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_spec(path: Path, *, codec_id: str = "my_codec") -> Path:
    return _write_json(
        path,
        {
            "schema_version": "0.36.0",
            "codec_id": codec_id,
            "display_name": "My Codec",
            "domain": "image",
            "family": "classical",
            "runtime": {
                "type": "external_command",
                "executable": "my-codec",
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
            "parameters": [{"name": "quality", "type": "integer", "values": [80]}],
            "output": {"extension": ".myc", "must_be_nonempty": True},
            "rate": {"metric": "bpp"},
            "quality": {"metric": "ssimulacra2", "direction": "higher_is_better"},
            "measurement": {"time": "external_benchmark"},
            "requirements": {"binaries": ["my-codec"]},
            "security": {"allow_shell": False},
        },
    )


def _write_probe(path: Path, *, codec_id: str = "my_codec") -> Path:
    return _write_json(
        path,
        {
            "external_codec_probe": {
                "schema_version": "external_codec_probe_v1",
                "valid_spec": True,
                "codec_id": codec_id,
                "domain": "image",
                "runtime_type": "external_command",
                "available": True,
                "safety": {
                    "shell_used": False,
                    "encode_executed": False,
                    "decode_executed": False,
                    "benchmark_executed": False,
                },
            }
        },
    )


def _write_rde_csv(
    path: Path,
    *,
    codec_id: str = "my_codec",
    success: str = "True",
    quality: str = "95",
    energy: str = "0.01",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "codec": codec_id,
                "param": '{"quality": "80"}',
                "input_id": "img001",
                "input_path": "input.png",
                "rate": "0.1",
                "quality": quality,
                "energy": energy,
                "time_ms": "1.0",
                "energy_provenance_tier": "benchmark_reference",
                "measurement_provenance": "test",
                "success": success,
                "error": "",
                "source_raw_csv": "raw.csv",
            }
        )
    return path


def _write_export_report(
    path: Path,
    rde_csv: Path,
    *,
    codec_id: str = "my_codec",
    router_ready: bool = True,
) -> Path:
    return _write_json(
        path,
        {
            "external_codec_rde_export": {
                "schema_version": "external_codec_rde_export_v1",
                "codec_id": codec_id,
                "raw_csv": "raw.csv",
                "output_csv": str(rde_csv),
                "raw_rows": 1,
                "successful_raw_rows": 1,
                "failed_raw_rows": 0,
                "exported_rows": 1,
                "structurally_valid_csv": True,
                "quality_available": True,
                "energy_available": True,
                "router_ready": router_ready,
                "rate_mode": "image_bpp",
                "warnings": [],
                "errors": [],
                "safety": {
                    "codec_executed": False,
                    "encode_executed": False,
                    "decode_executed": False,
                    "router_candidate_registered": False,
                },
            }
        },
    )


def _write_manifest(
    path: Path,
    *,
    codec_id: str = "my_codec",
    spec: Path,
    probe: Path,
    export_report: Path,
    rde_csv: Path,
    hash_overrides: dict | None = None,
) -> Path:
    hashes = {
        "spec_sha256": _sha256(spec),
        "probe_report_sha256": _sha256(probe),
        "rde_export_report_sha256": _sha256(export_report),
        "rde_csv_sha256": _sha256(rde_csv),
    }
    hashes.update(hash_overrides or {})
    return _write_json(
        path,
        {
            "schema_version": "external_codec_router_manifest_v1",
            "codec_id": codec_id,
            "spec_path": str(spec),
            "probe_report": str(probe),
            "rde_export_report": str(export_report),
            "rde_csv": str(rde_csv),
            "created_at_utc": "2026-05-13T00:00:00+00:00",
            "hashes": hashes,
        },
    )


def _valid_manifest(tmp_path: Path, *, codec_id: str = "my_codec") -> Path:
    spec = _write_spec(tmp_path / "spec.json", codec_id=codec_id)
    probe = _write_probe(tmp_path / "probe.json", codec_id=codec_id)
    rde_csv = _write_rde_csv(tmp_path / "external_rde.csv", codec_id=codec_id)
    export_report = _write_export_report(
        tmp_path / "export_report.json",
        rde_csv,
        codec_id=codec_id,
    )
    return _write_manifest(
        tmp_path / "manifest.json",
        codec_id=codec_id,
        spec=spec,
        probe=probe,
        export_report=export_report,
        rde_csv=rde_csv,
    )


def _base_csv(path: Path) -> Path:
    path.write_text(
        "codec,param,rate,quality,energy,time_ms\n"
        "JPEG,q=80,1.0,90,1.0,10\n",
        encoding="utf-8",
    )
    return path


def test_valid_manifest_loads_external_points(tmp_path: Path):
    manifest = _valid_manifest(tmp_path)

    points, report = load_external_codec_points([manifest])

    assert len(points) == 1
    assert points[0].codec == "my_codec"
    assert points[0].raw["source"] == "external_codec_manifest"
    assert report["enabled"] is True
    assert report["loaded_codecs"] == ["my_codec"]
    assert report["loaded_rows"] == 1
    assert report["validation"]["validated"] is True


def test_manifest_hash_mismatch_is_controlled_error(tmp_path: Path):
    manifest = _valid_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["hashes"]["rde_csv_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExternalCodecManifestError, match="hash mismatch"):
        validate_external_codec_manifest(manifest)


def test_manifest_router_ready_false_is_rejected(tmp_path: Path):
    spec = _write_spec(tmp_path / "spec.json")
    probe = _write_probe(tmp_path / "probe.json")
    rde_csv = _write_rde_csv(tmp_path / "external_rde.csv")
    export_report = _write_export_report(
        tmp_path / "export_report.json",
        rde_csv,
        router_ready=False,
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        spec=spec,
        probe=probe,
        export_report=export_report,
        rde_csv=rde_csv,
    )

    with pytest.raises(ExternalCodecManifestError, match="router_ready=false"):
        validate_external_codec_manifest(manifest)


def test_manifest_missing_file_is_controlled_error(tmp_path: Path):
    manifest = _valid_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rde_csv"] = str(tmp_path / "missing.csv")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExternalCodecManifestError, match="file not found"):
        validate_external_codec_manifest(manifest)


def test_manifest_codec_id_mismatch_is_rejected(tmp_path: Path):
    spec = _write_spec(tmp_path / "spec.json", codec_id="other_codec")
    probe = _write_probe(tmp_path / "probe.json")
    rde_csv = _write_rde_csv(tmp_path / "external_rde.csv")
    export_report = _write_export_report(tmp_path / "export_report.json", rde_csv)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        spec=spec,
        probe=probe,
        export_report=export_report,
        rde_csv=rde_csv,
    )

    with pytest.raises(ExternalCodecManifestError, match="codec_id mismatch"):
        validate_external_codec_manifest(manifest)


def test_router_without_external_manifest_is_unchanged_and_reports_disabled(tmp_path: Path):
    base = _base_csv(tmp_path / "base.csv")
    out = tmp_path / "report.json"

    main([
        "--csv",
        str(base),
        "--codec-col",
        "codec",
        "--config-col",
        "param",
        "--rate-col",
        "rate",
        "--quality-col",
        "quality",
        "--energy-col",
        "energy",
        "--time-col",
        "time_ms",
        "--available-codecs",
        "JPEG",
        "--quality-floor",
        "80",
        "--out",
        str(out),
    ])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["external_codecs"] == {"enabled": False}
    assert report["decision"]["selected"]["codec"] == "JPEG"
    assert report["decision"]["num_points_total"] == 1


def test_router_valid_manifest_loads_external_rows_into_candidate_pool(tmp_path: Path):
    base = _base_csv(tmp_path / "base.csv")
    manifest = _valid_manifest(tmp_path / "external")
    out = tmp_path / "report.json"

    main([
        "--csv",
        str(base),
        "--external-codec-manifest",
        str(manifest),
        "--codec-col",
        "codec",
        "--config-col",
        "param",
        "--rate-col",
        "rate",
        "--quality-col",
        "quality",
        "--energy-col",
        "energy",
        "--time-col",
        "time_ms",
        "--available-codecs",
        "JPEG,my_codec",
        "--quality-floor",
        "80",
        "--out",
        str(out),
    ])

    report = json.loads(out.read_text(encoding="utf-8"))
    pool = report["decision"]["scored_candidate_pool"]
    assert report["external_codecs"]["enabled"] is True
    assert report["external_codecs"]["source"] == "explicit_external_codec_manifest"
    assert report["external_codecs"]["loaded_codecs"] == ["my_codec"]
    assert report["external_codecs"]["loaded_rows"] == 1
    assert any(item["codec"] == "my_codec" for item in pool)
    external = next(item for item in pool if item["codec"] == "my_codec")
    assert external["raw"]["source"] == "external_codec_manifest"
    assert external["raw"]["external_router_ready"] is True


def test_router_rejects_external_manifest_with_incomplete_rde_csv(tmp_path: Path):
    spec = _write_spec(tmp_path / "spec.json")
    probe = _write_probe(tmp_path / "probe.json")
    rde_csv = _write_rde_csv(tmp_path / "external_rde.csv", energy="")
    export_report = _write_export_report(tmp_path / "export_report.json", rde_csv)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        spec=spec,
        probe=probe,
        export_report=export_report,
        rde_csv=rde_csv,
    )
    base = _base_csv(tmp_path / "base.csv")

    with pytest.raises(ExternalCodecManifestError, match="missing energy"):
        main([
            "--csv",
            str(base),
            "--external-codec-manifest",
            str(manifest),
            "--codec-col",
            "codec",
            "--config-col",
            "param",
            "--rate-col",
            "rate",
            "--quality-col",
            "quality",
            "--energy-col",
            "energy",
        ])


def test_router_does_not_auto_discover_external_manifest(tmp_path: Path):
    base = _base_csv(tmp_path / "base.csv")
    _valid_manifest(tmp_path / "adjacent_manifest_dir")
    out = tmp_path / "report.json"

    main([
        "--csv",
        str(base),
        "--codec-col",
        "codec",
        "--config-col",
        "param",
        "--rate-col",
        "rate",
        "--quality-col",
        "quality",
        "--energy-col",
        "energy",
        "--out",
        str(out),
    ])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["external_codecs"] == {"enabled": False}
    assert report["decision"]["num_points_total"] == 1
