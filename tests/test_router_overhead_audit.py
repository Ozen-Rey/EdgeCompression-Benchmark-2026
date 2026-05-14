import csv
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.calibration.calibration_bundle import sha256_file
from src.router.observability.router_overhead_audit import (
    main as overhead_main,
    run_router_overhead_audit,
)
from src.router.version import ROUTER_VERSION
from tests.conftest import scratch_root


def _tmp_dir(name: str) -> Path:
    root = (
        scratch_root()
        / "router_overhead_audit"
        / f"{name}_{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "domain": "image",
                "columns": {
                    "codec": "codec",
                    "config": "config",
                    "rate": "rate",
                    "quality": "quality",
                    "energy": "energy",
                    "time": "time_ms",
                },
                "selection": {
                    "quality_target": "preview",
                    "quality_floor": 50,
                },
                "normalization": {
                    "mode": "runtime",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_points(path: Path, *, changed: bool = False) -> None:
    rows = [
        "codec,config,rate,quality,energy,time_ms",
        "JPEG,q=85,0.4,95,1.0,10",
        (
            "JXL,d=1.0,0.1,99,0.1,2"
            if changed
            else "JXL,d=1.0,0.8,95,2.0,20"
        ),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_manifest(path: Path, calibrated_csv: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_type": "promoted_calibration_bundle",
                "router_version": "0.18.0",
                "mode": "explicit_opt_in_calibration_apply",
                "output_csv": str(calibrated_csv),
                "accepted_scales": [],
                "rejected_scales_count": 0,
                "energy_policy": {
                    "requires_energy_usable_for_total": True,
                    "gpu_only_energy_excluded": True,
                },
                "hashes": {
                    "output_csv_sha256": sha256_file(calibrated_csv),
                },
            }
        ),
        encoding="utf-8",
    )


def _write_validation(path: Path, manifest: Path, calibrated_csv: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": "shadow_decision_validation_only",
                "router_version": "0.24.0",
                "accepted": True,
                "decision_count": 3,
                "changed_decision_count": 0,
                "decision_churn_rate": 0.0,
                "relative_cost_improvement": 0.1,
                "rejection_reasons": [],
                "validated_comparison_path": "comparison.json",
                "validated_comparison_sha256": "c" * 64,
                "candidate_calibration_bundle_manifest_sha256": sha256_file(
                    manifest
                ),
                "candidate_calibrated_csv_sha256": sha256_file(calibrated_csv),
            }
        ),
        encoding="utf-8",
    )


def _base_inputs(root: Path) -> tuple[Path, Path]:
    csv_path = root / "points.csv"
    config = root / "router_config.json"
    _write_points(csv_path)
    _write_config(config)
    return csv_path, config


def _result_by_mode(report: dict, mode: str) -> dict:
    return {
        item["mode"]: item
        for item in report["results"]
    }[mode]


def test_audit_produces_json_and_csv_with_numeric_overhead_fields():
    root = _tmp_dir("outputs")
    csv_path, config = _base_inputs(root)
    out_dir = root / "audit"

    report = run_router_overhead_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(out_dir),
    )
    rows = list(
        csv.DictReader((out_dir / "router_overhead_audit.csv").open("r", encoding="utf-8"))
    )

    assert (out_dir / "router_overhead_audit.json").exists()
    assert report["mode"] == "router_overhead_audit"
    assert report["router_version"] == ROUTER_VERSION
    assert len(rows) == len(report["results"])
    for result in report["results"]:
        assert isinstance(result["wall_time_ms"], float)
        assert result["wall_time_ms"] >= 0.0
        assert isinstance(result["process_cpu_time_ms"], float)
        assert result["process_cpu_time_ms"] >= 0.0
        assert result["input_csv_rows"] == 2
        assert result["candidate_count"] == 2


def test_baseline_and_receipt_only_have_same_decision():
    root = _tmp_dir("receipt")
    csv_path, config = _base_inputs(root)

    report = run_router_overhead_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    baseline = _result_by_mode(report, "baseline")
    receipt = _result_by_mode(report, "receipt")

    assert receipt["decision_match_baseline"] is True
    assert receipt["selected_codec"] == baseline["selected_codec"]
    assert receipt["selected_config"] == baseline["selected_config"]
    assert receipt["comparison_type"] == "overhead-only"


def test_bundle_audit_uses_only_explicit_manifest():
    root = _tmp_dir("bundle_explicit")
    csv_path, config = _base_inputs(root)
    calibrated_csv = root / "calibrated.csv"
    manifest = root / "bundle_manifest.json"
    _write_points(calibrated_csv)
    _write_manifest(manifest, calibrated_csv)

    no_bundle = run_router_overhead_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit_no_bundle"),
    )
    explicit_bundle = run_router_overhead_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        bundle_manifest=str(manifest),
        out_dir=str(root / "audit_bundle"),
    )

    assert "bundle" not in {item["mode"] for item in no_bundle["results"]}
    assert "bundle" in {item["mode"] for item in explicit_bundle["results"]}


def test_validated_bundle_requires_validation_explicit():
    root = _tmp_dir("validated")
    csv_path, config = _base_inputs(root)
    calibrated_csv = root / "calibrated.csv"
    manifest = root / "bundle_manifest.json"
    validation = root / "validation.json"
    _write_points(calibrated_csv)
    _write_manifest(manifest, calibrated_csv)
    _write_validation(validation, manifest, calibrated_csv)

    bundle_only = run_router_overhead_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        bundle_manifest=str(manifest),
        out_dir=str(root / "audit_bundle_only"),
    )
    validated = run_router_overhead_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        bundle_manifest=str(manifest),
        bundle_validation=str(validation),
        out_dir=str(root / "audit_validated"),
    )

    assert "validated_bundle" not in {item["mode"] for item in bundle_only["results"]}
    assert "validated_bundle" in {item["mode"] for item in validated["results"]}
    assert "validated_bundle_receipt" in {item["mode"] for item in validated["results"]}


def test_validation_without_bundle_errors_instead_of_auto_discovery():
    root = _tmp_dir("validation_without_bundle")
    csv_path, config = _base_inputs(root)

    with pytest.raises(ValueError, match="requires --bundle-manifest"):
        run_router_overhead_audit(
            csv_path=str(csv_path),
            config_path=str(config),
            bundle_validation=str(root / "validation.json"),
            out_dir=str(root / "audit"),
        )


def test_replay_mode_is_marked_offline_and_not_runtime():
    root = _tmp_dir("replay")
    csv_path, config = _base_inputs(root)

    report = run_router_overhead_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
        include_replay=True,
    )
    replay = _result_by_mode(report, "replay")

    assert replay["runtime_path"] is False
    assert replay["mode_type"] == "offline-replay"
    assert replay["comparison_type"] == "offline-replay"
    assert "offline_replay_not_runtime_path" in replay["notes"]


def test_cli_writes_requested_outputs():
    root = _tmp_dir("cli")
    csv_path, config = _base_inputs(root)
    out = root / "custom.json"
    summary = root / "custom.csv"

    overhead_main(
        [
            "--csv",
            str(csv_path),
            "--config",
            str(config),
            "--out-dir",
            str(root / "audit"),
            "--out",
            str(out),
            "--summary-out",
            str(summary),
        ]
    )

    assert out.exists()
    assert summary.exists()
