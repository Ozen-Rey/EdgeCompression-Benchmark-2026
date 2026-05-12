import csv
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.calibration_bundle import sha256_file
from src.router.router_effectiveness_audit import (
    main as effectiveness_main,
    run_router_effectiveness_audit,
)
from src.router.version import ROUTER_VERSION


def _tmp_dir(name: str) -> Path:
    root = (
        Path(__file__).with_name("_tmp")
        / "router_effectiveness_audit"
        / f"{name}_{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_config(path: Path, *, max_rate: float | None = None) -> None:
    selection = {
        "quality_target": "preview",
        "quality_floor": 50,
    }
    if max_rate is not None:
        selection["max_rate"] = max_rate

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
                "selection": selection,
                "normalization": {
                    "mode": "runtime",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_points(path: Path, *, calibrated: bool = False) -> None:
    rows = [
        "codec,config,rate,quality,energy,time_ms",
        "JPEG,q=85,0.4,95,1.0,10",
        (
            "JXL,d=1.0,0.1,99,0.1,2"
            if calibrated
            else "JXL,d=1.0,0.8,99,3.0,20"
        ),
        "WEBP,q=80,0.7,90,0.2,5",
        "HEVC,crf=15,0.2,40,0.1,1",
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


def _scenario(report: dict, name: str = "benchmark") -> dict:
    return {
        scenario["scenario"]: scenario
        for scenario in report["scenarios"]
    }[name]


def _policy(scenario: dict, name: str) -> dict:
    return {
        row["policy"]: row
        for row in scenario["policies"]
    }[name]


def test_audit_produces_json_and_csv_valid():
    root = _tmp_dir("outputs")
    csv_path, config = _base_inputs(root)
    out_dir = root / "audit"

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(out_dir),
    )
    summary_rows = list(
        csv.DictReader(
            (out_dir / "router_effectiveness_audit.csv").open(
                "r",
                encoding="utf-8",
            )
        )
    )
    policy_rows = list(
        csv.DictReader(
            (out_dir / "router_effectiveness_by_policy.csv").open(
                "r",
                encoding="utf-8",
            )
        )
    )

    assert (out_dir / "router_effectiveness_audit.json").exists()
    assert report["mode"] == "router_effectiveness_audit"
    assert report["router_version"] == ROUTER_VERSION
    assert report["read_only"] is True
    assert len(summary_rows) == 1
    assert len(policy_rows) == len(report["scenarios"][0]["policies"])


def test_router_decision_included_in_report():
    root = _tmp_dir("router_decision")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    router = _scenario(report)["router_decision"]

    assert router["codec"]
    assert router["config"]
    assert isinstance(router["cost"], float)


def test_baseline_policies_are_calculated_correctly():
    root = _tmp_dir("baselines")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    scenario = _scenario(report)

    assert _policy(scenario, "lowest_rate")["selected_codec"] == "JPEG"
    assert _policy(scenario, "highest_quality")["selected_codec"] == "JXL"
    assert _policy(scenario, "lowest_energy")["selected_codec"] == "WEBP"
    assert _policy(scenario, "fastest_time")["selected_codec"] == "WEBP"
    assert _policy(scenario, "lowest_rate")["selected_codec"] != "HEVC"


def test_constraint_violations_counted_and_quality_guard_not_bypassed():
    root = _tmp_dir("violations")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        fixed_codec="HEVC",
        fixed_config="crf=15",
        out_dir=str(root / "audit"),
    )
    scenario = _scenario(report)
    fixed = _policy(scenario, "fixed_codec_config")

    assert scenario["candidate_counts"]["num_quality_guard_violations"] == 1
    assert fixed["selected_codec"] == "HEVC"
    assert fixed["comparable"] is False
    assert fixed["quality_guard_violations"] == 1
    assert fixed["regret"] is None


def test_regret_is_calculated_only_for_comparable_candidates():
    root = _tmp_dir("regret")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        fixed_codec="HEVC",
        fixed_config="crf=15",
        out_dir=str(root / "audit"),
    )
    scenario = _scenario(report)

    assert _policy(scenario, "lowest_rate")["comparable"] is True
    assert _policy(scenario, "lowest_rate")["regret"] is not None
    assert _policy(scenario, "fixed_codec_config")["comparable"] is False
    assert _policy(scenario, "fixed_codec_config")["regret"] is None


def test_bundle_and_validation_are_only_explicit():
    root = _tmp_dir("bundle_explicit")
    csv_path, config = _base_inputs(root)
    calibrated_csv = root / "calibrated.csv"
    manifest = root / "bundle_manifest.json"
    validation = root / "validation.json"
    _write_points(calibrated_csv, calibrated=True)
    _write_manifest(manifest, calibrated_csv)
    _write_validation(validation, manifest, calibrated_csv)

    no_bundle = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit_no_bundle"),
    )
    explicit_bundle = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        bundle_manifest=str(manifest),
        out_dir=str(root / "audit_bundle"),
    )
    validated = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        bundle_manifest=str(manifest),
        bundle_validation=str(validation),
        out_dir=str(root / "audit_validated"),
    )

    assert [scenario["scenario"] for scenario in no_bundle["scenarios"]] == [
        "benchmark"
    ]
    assert "calibration_bundle" in {
        scenario["scenario"] for scenario in explicit_bundle["scenarios"]
    }
    assert "validated_calibration_bundle" in {
        scenario["scenario"] for scenario in validated["scenarios"]
    }


def test_validation_without_bundle_errors_instead_of_auto_discovery():
    root = _tmp_dir("validation_without_bundle")
    csv_path, config = _base_inputs(root)

    with pytest.raises(ValueError, match="requires --bundle-manifest"):
        run_router_effectiveness_audit(
            csv_path=str(csv_path),
            config_path=str(config),
            bundle_validation=str(root / "validation.json"),
            out_dir=str(root / "audit"),
        )


def test_no_side_effect_on_input_files():
    root = _tmp_dir("side_effects")
    csv_path, config = _base_inputs(root)
    csv_hash_before = sha256_file(csv_path)
    config_hash_before = sha256_file(config)

    run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
        include_random_safe=True,
    )

    assert sha256_file(csv_path) == csv_hash_before
    assert sha256_file(config) == config_hash_before


def test_module_is_marked_audit_read_only():
    root = _tmp_dir("read_only")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )

    assert report["mode"] == "router_effectiveness_audit"
    assert report["read_only"] is True
    assert report["methodology"]["router_logic_changed"] is False
    assert report["methodology"]["baseline_policies_respect_quality_guard"] is True


def test_cli_writes_requested_outputs():
    root = _tmp_dir("cli")
    csv_path, config = _base_inputs(root)
    out = root / "custom.json"
    summary = root / "custom.csv"
    by_policy = root / "custom_by_policy.csv"

    effectiveness_main(
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
            "--by-policy-out",
            str(by_policy),
        ]
    )

    assert out.exists()
    assert summary.exists()
    assert by_policy.exists()
