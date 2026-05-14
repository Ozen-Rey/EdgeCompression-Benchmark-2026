import contextlib
import csv
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.calibration.calibration_bundle import sha256_file
from src.router.rde_router import main as router_main
from src.router.observability.shadow_decision_comparison import (
    main as shadow_main,
    run_shadow_decision_comparison,
)
from src.router.version import ROUTER_VERSION
from tests.conftest import scratch_root


def _tmp_dir(name: str) -> Path:
    root = (
        scratch_root()
        / "shadow_decision_comparison"
        / f"{name}_{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_config(path: Path) -> None:
    payload = {
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_rde_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["codec", "config", "rate", "quality", "energy", "time_ms"]
    path.parent.mkdir(parents=True, exist_ok=True)

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


def _changed_rows() -> list[dict[str, object]]:
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
    accepted_scales: list[dict[str, object]] | None = None,
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
        "accepted_scales": accepted_scales
        if accepted_scales is not None
        else [
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


def _setup_case(
    root: Path,
    calibrated_rows: list[dict[str, object]],
    *,
    output_hash: str | None = None,
    accepted_scales: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    baseline_csv = root / "baseline.csv"
    calibrated_csv = root / "calibrated.csv"
    manifest = root / "manifest.json"
    config = root / "router_config.json"
    out = root / "shadow.json"

    _write_rde_csv(baseline_csv, _baseline_rows())
    _write_rde_csv(calibrated_csv, calibrated_rows)
    _write_manifest(
        manifest,
        calibrated_csv,
        output_hash=output_hash,
        accepted_scales=accepted_scales,
    )
    _write_config(config)
    return baseline_csv, calibrated_csv, manifest, config, out


def _run_router(csv_path: Path, config_path: Path, out_path: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        router_main(
            [
                "--config",
                str(config_path),
                "--csv",
                str(csv_path),
                "--out",
                str(out_path),
            ]
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_bundle_valid_comparison_produces_json_and_csv():
    root = _tmp_dir("valid_outputs")
    baseline_csv, calibrated_csv, manifest, config, out = _setup_case(
        root,
        _changed_rows(),
    )
    summary = root / "shadow.csv"

    report = run_shadow_decision_comparison(
        baseline_csv=str(baseline_csv),
        bundle_manifest=str(manifest),
        config_path=str(config),
        out_path=str(out),
        summary_out=str(summary),
    )

    rows = list(csv.DictReader(summary.open("r", encoding="utf-8")))

    assert out.exists()
    assert summary.exists()
    assert report["mode"] == "shadow_decision_comparison_only"
    assert report["router_version"] == ROUTER_VERSION
    assert report["baseline_csv"] == str(baseline_csv)
    assert report["baseline_csv_sha256"] == sha256_file(baseline_csv)
    assert report["bundle_manifest"] == str(manifest)
    assert report["calibrated_csv"] == str(calibrated_csv)
    assert report["candidate_calibration_bundle_manifest_path"] == str(manifest)
    assert report["candidate_calibration_bundle_manifest_sha256"] == sha256_file(
        manifest
    )
    assert report["candidate_calibrated_csv_sha256"] == sha256_file(calibrated_csv)
    assert report["calibration_bundle"]["validated"] is True
    assert report["calibration_bundle"]["manifest_sha256"] == sha256_file(manifest)
    assert report["calibration_bundle"]["calibrated_csv_sha256"] == sha256_file(
        calibrated_csv
    )
    assert report["total_cases"] == 1
    assert len(rows) == 1
    assert rows[0]["case_id"] == report["per_case"][0]["case_id"]


def test_bundle_hash_mismatch_errors_controlled():
    root = _tmp_dir("hash_mismatch")
    baseline_csv, _, manifest, config, out = _setup_case(
        root,
        _baseline_rows(),
        output_hash="0" * 64,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        run_shadow_decision_comparison(
            baseline_csv=str(baseline_csv),
            bundle_manifest=str(manifest),
            config_path=str(config),
            out_path=str(out),
            summary_out=str(root / "shadow.csv"),
        )


def test_identical_baseline_and_calibrated_has_no_decision_change():
    root = _tmp_dir("no_change")
    baseline_csv, _, manifest, config, out = _setup_case(root, _baseline_rows())

    report = run_shadow_decision_comparison(
        baseline_csv=str(baseline_csv),
        bundle_manifest=str(manifest),
        config_path=str(config),
        out_path=str(out),
        summary_out=str(root / "shadow.csv"),
    )

    assert report["changed_decisions"] == 0
    assert report["unchanged_decisions"] == 1
    assert report["changed_decision_rate"] == pytest.approx(0.0)
    assert report["per_case"][0]["decision_changed"] is False


def test_calibrated_values_can_change_shadow_decision():
    root = _tmp_dir("decision_change")
    baseline_csv, _, manifest, config, out = _setup_case(root, _changed_rows())

    report = run_shadow_decision_comparison(
        baseline_csv=str(baseline_csv),
        bundle_manifest=str(manifest),
        config_path=str(config),
        out_path=str(out),
        summary_out=str(root / "shadow.csv"),
    )

    row = report["per_case"][0]
    assert report["changed_decisions"] == 1
    assert report["changed_decision_rate"] == pytest.approx(1.0)
    assert row["decision_changed"] is True
    assert row["baseline_codec"] == "JPEG"
    assert row["shadow_codec"] == "JXL"
    assert report["aggregate_deltas"]["mean_rate_delta"] == row["rate_delta"]


def test_no_auto_discovery_without_bundle_manifest():
    root = _tmp_dir("no_auto_discovery")
    baseline_csv = root / "baseline.csv"
    config = root / "router_config.json"
    out = root / "shadow.json"
    _write_rde_csv(baseline_csv, _baseline_rows())
    _write_config(config)

    with pytest.raises(SystemExit):
        shadow_main(
            [
                "--baseline-csv",
                str(baseline_csv),
                "--config",
                str(config),
                "--out",
                str(out),
                "--summary-out",
                str(root / "shadow.csv"),
            ]
        )

    assert not out.exists()


def test_gpu_only_non_total_energy_is_not_exposed_as_local_total_energy():
    root = _tmp_dir("gpu_only")
    accepted_scales = [
        {
            "codec": "JXL",
            "config": "d=1.0",
            "axis": "energy",
            "scale": 0.5,
            "energy_scope": "gpu",
            "energy_usable_for_total": False,
            "local_energy_j": 123.0,
        }
    ]
    baseline_csv, _, manifest, config, out = _setup_case(
        root,
        _baseline_rows(),
        accepted_scales=accepted_scales,
    )

    report = run_shadow_decision_comparison(
        baseline_csv=str(baseline_csv),
        bundle_manifest=str(manifest),
        config_path=str(config),
        out_path=str(out),
        summary_out=str(root / "shadow.csv"),
    )
    serialized = json.dumps(report)

    assert "local_energy_j" not in serialized
    assert "energy_usable_for_total" not in serialized
    assert report["per_case"][0]["shadow_energy"] == report["per_case"][0][
        "baseline_energy"
    ]
    assert report["calibration_bundle"]["energy_policy"] == "usable_total_only"


def test_normal_router_without_shadow_module_remains_unchanged():
    root = _tmp_dir("router_defaults")
    baseline_csv, _, manifest, config, out = _setup_case(root, _changed_rows())

    comparison = run_shadow_decision_comparison(
        baseline_csv=str(baseline_csv),
        bundle_manifest=str(manifest),
        config_path=str(config),
        out_path=str(out),
        summary_out=str(root / "shadow.csv"),
    )
    normal_report = _run_router(
        baseline_csv,
        config,
        root / "normal_router.json",
    )
    selected = normal_report["decision"]["selected"]

    assert normal_report["calibration_bundle"] == {"enabled": False}
    assert selected["codec"] == comparison["per_case"][0]["baseline_codec"]
    assert selected["config"] == comparison["per_case"][0]["baseline_config"]
