import csv
import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.router.calibration.calibration_bundle import sha256_file
from src.router.observability.router_effectiveness_audit import (
    main as effectiveness_main,
    run_router_effectiveness_audit,
)
from src.router.version import ROUTER_VERSION
from tests.conftest import scratch_root


def _tmp_dir(name: str) -> Path:
    root = (
        scratch_root()
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

    router_row = _policy(_scenario(report), "router")
    assert router_row["is_router_decision"] is True
    assert router_row["candidate_source"] == "router_decision"
    assert router_row["cost_status"] == "available"
    assert router_row["regret"] == 0.0
    assert router_row["router_relative_improvement_percent"] == 0.0


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


def test_baseline_matching_router_has_zero_regret():
    root = _tmp_dir("matching_router")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    scenario = _scenario(report)
    router = _policy(scenario, "router")
    global_best = _policy(scenario, "global_best_average")

    assert global_best["selected_codec"] == router["selected_codec"]
    assert global_best["selected_config"] == router["selected_config"]
    assert global_best["regret"] == 0.0
    assert global_best["router_relative_improvement_percent"] == 0.0


def test_feasible_unscored_candidate_is_not_generic_missing_cost():
    # With scored_candidate_pool, quality-guard-passing candidates always have costs.
    # Quality guard violators should still get a specific reason, not generic "missing_cost".
    root = _tmp_dir("feasible_unscored")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
        fixed_codec="HEVC",
        fixed_config="crf=15",
    )
    scenario = _scenario(report)

    # JXL passes quality guard â†’ now always in scored_candidate_pool â†’ always comparable
    highest_quality = _policy(scenario, "highest_quality")
    assert highest_quality["selected_codec"] == "JXL"
    assert highest_quality["comparable"] is True
    assert highest_quality["cost_status"] == "available"

    # HEVC fails quality guard â†’ not in scored pool â†’ specific reason, not generic
    fixed = _policy(scenario, "fixed_codec_config")
    assert fixed["selected_codec"] == "HEVC"
    assert fixed["comparable"] is False
    assert fixed["reason"] != "missing_cost"
    assert fixed["cost_status"] == "unavailable_filtered"
    assert fixed["quality_guard_violations"] == 1


def test_candidate_source_and_cost_status_are_populated():
    root = _tmp_dir("cost_status")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    scenario = _scenario(report)

    for policy in scenario["policies"]:
        assert "candidate_source" in policy
        assert "cost_status" in policy
        assert "cost_reason_detail" in policy

    router = _policy(scenario, "router")
    global_best = _policy(scenario, "global_best_average")
    assert router["candidate_source"] == "router_decision"
    assert router["cost_status"] == "available"
    assert global_best["candidate_source"] == "router_scored_pool"
    assert global_best["cost_status"] == "available"


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


def test_scored_candidate_pool_used_by_effectiveness_audit():
    root = _tmp_dir("scored_pool")
    csv_path, config = _base_inputs(root)

    report = run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    router_report_path = root / "audit" / "benchmark_router_report.json"
    with router_report_path.open("r", encoding="utf-8") as f:
        router_report = json.load(f)

    decision = router_report.get("decision", {}) or {}
    assert "scored_candidate_pool" in decision
    assert "unscored_candidate_pool" in decision

    pool = decision["scored_candidate_pool"]
    assert len(pool) > 0
    assert all(item.get("cost_provenance") == "router_scored" for item in pool)
    assert all(item.get("rank") is not None for item in pool)

    rank1 = next(item for item in pool if item["rank"] == 1)
    selected = decision.get("selected", {})
    assert rank1["codec"] == selected["codec"]
    assert rank1["config"] == selected["config"]

    scenario = _scenario(report)
    router_row = _policy(scenario, "router")
    assert router_row["cost_status"] == "available"


def test_normalization_audit_present_in_router_report():
    root = _tmp_dir("norm_audit")
    csv_path, config = _base_inputs(root)

    run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    router_report_path = root / "audit" / "benchmark_router_report.json"
    with router_report_path.open("r", encoding="utf-8") as f:
        router_report = json.load(f)

    audit = router_report.get("normalization_audit")
    assert audit is not None
    assert "mode" in audit
    assert "scales_source" in audit
    assert "computed_at_runtime" in audit
    assert "quality_direction" in audit
    assert audit["quality_direction"] == "higher_is_better"
    assert "rate_min" in audit
    assert "rate_max" in audit
    assert "energy_min" in audit
    assert "energy_max" in audit
    assert "quality_min" in audit
    assert "quality_max" in audit


def test_normalization_audit_present_in_decision_receipt():
    # The receipt is embedded in the router report under "decision_receipt".
    root = _tmp_dir("norm_audit_receipt")
    csv_path, config = _base_inputs(root)

    run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        out_dir=str(root / "audit"),
    )
    router_report_path = root / "audit" / "benchmark_router_report.json"
    with router_report_path.open("r", encoding="utf-8") as f:
        router_report = json.load(f)

    receipt = router_report.get("decision_receipt")
    assert receipt is not None, "decision_receipt must be embedded in the router report"
    assert "normalization_audit" in receipt
    assert receipt.get("receipt_schema_version") == "0.29.0"


def test_unscored_pool_quality_violations_are_not_scored():
    root = _tmp_dir("unscored_pool")
    csv_path, config = _base_inputs(root)

    run_router_effectiveness_audit(
        csv_path=str(csv_path),
        config_path=str(config),
        fixed_codec="HEVC",
        fixed_config="crf=15",
        out_dir=str(root / "audit"),
    )
    router_report_path = root / "audit" / "benchmark_router_report.json"
    with router_report_path.open("r", encoding="utf-8") as f:
        router_report = json.load(f)

    decision = router_report.get("decision", {}) or {}
    unscored = decision.get("unscored_candidate_pool", [])
    scored_keys = {(item["codec"], item["config"]) for item in decision.get("scored_candidate_pool", [])}

    hevc_unscored = [i for i in unscored if i["codec"] == "HEVC"]
    assert len(hevc_unscored) == 1
    assert hevc_unscored[0]["reason"] == "quality_guard_violation"
    assert hevc_unscored[0]["candidate_status"] == "infeasible_quality_guard"
    assert hevc_unscored[0]["cost_provenance"] == "unavailable_filtered"
    assert ("HEVC", "crf=15") not in scored_keys


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
