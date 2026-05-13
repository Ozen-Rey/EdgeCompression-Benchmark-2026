import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.router.rde_router import main


def test_thematic_subpackage_imports_keep_legacy_paths():
    from src.router.codecs.external_codec_spec import (
        validate_external_codec_spec as new_validate_external_codec_spec,
    )
    from src.router.external_codec_spec import (
        validate_external_codec_spec as legacy_validate_external_codec_spec,
    )
    from src.router.calibration.calibration_bundle import (
        validate_calibration_bundle_manifest as new_validate_calibration_bundle,
    )
    from src.router.calibration_bundle import (
        validate_calibration_bundle_manifest as legacy_validate_calibration_bundle,
    )

    assert new_validate_external_codec_spec is legacy_validate_external_codec_spec
    assert new_validate_calibration_bundle is legacy_validate_calibration_bundle


def test_observability_subpackage_imports_keep_legacy_paths():
    from src.router.observability.decision_receipt import (
        build_decision_receipt as new_build_decision_receipt,
    )
    from src.router.decision_receipt import (
        build_decision_receipt as legacy_build_decision_receipt,
    )
    from src.router.observability.feedback_analysis import (
        analyze_feedback as new_analyze_feedback,
    )
    from src.router.feedback_analysis import (
        analyze_feedback as legacy_analyze_feedback,
    )
    from src.router.observability.shadow_decision_validation import (
        run_shadow_decision_validation as new_run_shadow_decision_validation,
    )
    from src.router.shadow_decision_validation import (
        run_shadow_decision_validation as legacy_run_shadow_decision_validation,
    )

    assert new_build_decision_receipt is legacy_build_decision_receipt
    assert new_analyze_feedback is legacy_analyze_feedback
    assert new_run_shadow_decision_validation is legacy_run_shadow_decision_validation


def test_adaptation_subpackage_imports_keep_legacy_paths():
    from src.router.adaptation.content_policy import (
        build_content_policy_report as new_build_content_policy_report,
    )
    from src.router.adaptation.energy_provenance import (
        classify_energy_provenance as new_classify_energy_provenance,
    )
    from src.router.adaptation.system_policy import (
        build_system_policy as new_build_system_policy,
    )
    from src.router.content_policy import (
        build_content_policy_report as legacy_build_content_policy_report,
    )
    from src.router.energy_provenance import (
        classify_energy_provenance as legacy_classify_energy_provenance,
    )
    from src.router.system_policy import (
        build_system_policy as legacy_build_system_policy,
    )

    assert new_classify_energy_provenance is legacy_classify_energy_provenance
    assert new_build_content_policy_report is legacy_build_content_policy_report
    assert new_build_system_policy is legacy_build_system_policy


def test_analysis_subpackage_imports_keep_legacy_paths():
    from src.router.analysis.content_oracle_analysis import (
        analyze_content_oracle as new_analyze_content_oracle,
    )
    from src.router.content_oracle_analysis import (
        analyze_content_oracle as legacy_analyze_content_oracle,
    )
    from src.router.analysis.content_oracle_classifier import (
        evaluate_oracle_classifier as new_evaluate_oracle_classifier,
    )
    from src.router.content_oracle_classifier import (
        evaluate_oracle_classifier as legacy_evaluate_oracle_classifier,
    )
    from src.router.analysis.content_aware_paper_artifacts import (
        build_artifacts as new_build_artifacts,
    )
    from src.router.content_aware_paper_artifacts import (
        build_artifacts as legacy_build_artifacts,
    )

    assert new_analyze_content_oracle is legacy_analyze_content_oracle
    assert new_evaluate_oracle_classifier is legacy_evaluate_oracle_classifier
    assert new_build_artifacts is legacy_build_artifacts


def test_core_subpackage_imports_keep_legacy_paths():
    from src.router.core.normalization_profile import (
        build_normalization_profile as new_build_normalization_profile,
    )
    from src.router.core.quality_thresholds import (
        resolve_quality_floor as new_resolve_quality_floor,
    )
    from src.router.core.rde_database import RDEPoint as NewRDEPoint
    from src.router.normalization_profile import (
        build_normalization_profile as legacy_build_normalization_profile,
    )
    from src.router.quality_thresholds import (
        resolve_quality_floor as legacy_resolve_quality_floor,
    )
    from src.router.rde_database import RDEPoint as LegacyRDEPoint

    assert NewRDEPoint is LegacyRDEPoint
    assert new_build_normalization_profile is legacy_build_normalization_profile
    assert new_resolve_quality_floor is legacy_resolve_quality_floor


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "router_characterization"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _real_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "image_rde_real_small.csv"


def _run_router(args: list[str], out_name: str) -> dict:
    out_path = _tmp_path(out_name)
    main([*args, "--out", str(out_path)])
    return json.loads(out_path.read_text(encoding="utf-8"))


def _real_fixture_args() -> list[str]:
    return [
        "--csv",
        str(_real_fixture()),
        "--codec-col",
        "codec",
        "--config-col",
        "param",
        "--rate-col",
        "bpp",
        "--quality-col",
        "ssimulacra2",
        "--energy-col",
        "energy_per_image_j",
        "--time-col",
        "time_ms",
        "--available-codecs",
        "JPEG,JXL,HEVC",
        "--quality-target",
        "very-high",
        "--quality-floor",
        "90",
    ]


_VOLATILE_REPORT_KEYS = {
    "run_manifest",
    "system_state",
}


def _normalize_report_for_semantic_comparison(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_report_for_semantic_comparison(item)
            for key, item in value.items()
            if key not in _VOLATILE_REPORT_KEYS
            and not key.endswith("_at_utc")
            and not key.endswith("_timestamp")
            and key not in {"overhead_ms", "duration_ms", "elapsed_ms"}
        }

    if isinstance(value, list):
        return [_normalize_report_for_semantic_comparison(item) for item in value]

    if isinstance(value, str):
        text = value.replace(str(Path.cwd()), "<repo>")
        return re.sub(
            r"tests[\\/]+_tmp[\\/]+router_characterization[\\/]+[^\\/]+",
            "tests/_tmp/router_characterization/<tmp-file>",
            text,
        )

    return value


def test_base_router_characterization_without_external_codec():
    report = _run_router(_real_fixture_args(), "base_report.json")
    selected = report["decision"]["selected"]

    assert report["external_codecs"] == {"enabled": False}
    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
    assert selected["cost"] == pytest.approx(0.66)


def test_base_router_report_semantic_repeatability():
    first = _run_router(_real_fixture_args(), "base_semantic_first.json")
    second = _run_router(_real_fixture_args(), "base_semantic_second.json")

    assert _normalize_report_for_semantic_comparison(
        first
    ) == _normalize_report_for_semantic_comparison(second)


def test_content_policy_apply_characterization():
    csv_path = _tmp_path("content_apply_rde.csv")
    rules_path = _tmp_path("content_apply_rules.csv")
    csv_path.write_text(
        "\n".join(
            [
                "dataset,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,JPEG,q=85,1.0,95.0,1.0,20.0",
                "A,HEVC,crf=15,1.2,85.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )
    rules_path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,A,JPEG,q=85",
            ]
        ),
        encoding="utf-8",
    )

    report = _run_router(
        [
            "--csv",
            str(csv_path),
            "--codec-col",
            "codec",
            "--config-col",
            "param",
            "--rate-col",
            "bpp",
            "--quality-col",
            "ssimulacra2",
            "--energy-col",
            "energy_per_image_j",
            "--time-col",
            "time_ms",
            "--domain",
            "image",
            "--quality-target",
            "high",
            "--quality-floor",
            "80",
            "--content-policy",
            "--content-policy-mode",
            "apply",
            "--content-policy-rules-file",
            str(rules_path),
            "--content-policy-key",
            "dataset",
            "--content-source",
            "A",
            "--content-source-filter",
            "--content-filter-column",
            "dataset",
        ],
        "content_apply_report.json",
    )

    selected = report["decision"]["selected"]
    assert selected["codec"] == "JPEG"
    assert selected["config"] == "q=85"
    assert report["content_policy"]["applied"] is True
    assert (
        report["decision"]["decision_trace"]["selected_reason"]
        == "content_policy_preferred_candidate"
    )


def test_degraded_fallback_characterization():
    csv_path = _tmp_path("degraded_fallback_rde.csv")
    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=60,0.8,70.0,1.0,10.0",
                "JXL,d=1.0,1.0,85.0,2.0,20.0",
            ]
        ),
        encoding="utf-8",
    )

    report = _run_router(
        [
            "--csv",
            str(csv_path),
            "--codec-col",
            "codec",
            "--config-col",
            "param",
            "--rate-col",
            "bpp",
            "--quality-col",
            "ssimulacra2",
            "--energy-col",
            "energy_per_image_j",
            "--time-col",
            "time_ms",
            "--quality-floor",
            "90",
            "--allow-degraded-fallback",
            "--near-quality-floor",
            "80",
        ],
        "degraded_fallback_report.json",
    )

    assert report["decision"]["decision_mode"] == "degraded_fallback"
    assert report["decision"]["selected"]["codec"] == "JXL"
    assert (
        report["decision"]["decision_trace"]["selected_reason"]
        == "lowest_J_RDE_in_degraded_fallback_pool"
    )


def test_energy_provenance_report_blocks_characterization():
    report = _run_router(_real_fixture_args(), "energy_blocks_report.json")
    selected = report["decision"]["selected"]

    assert selected["energy_provenance_tier"] == "benchmark_reference"
    assert "energy_provenance_summary" in report
    assert report["energy_provenance_summary"]["selected_tier"] == "benchmark_reference"
    assert "energy_provenance_compatibility" in report
    assert report["energy_provenance_compatibility"]["severity"] == "ok"
    assert report["energy_tier_policy"]["mode"] == "report-only"


def test_normalization_audit_characterization():
    report = _run_router(_real_fixture_args(), "normalization_report.json")

    audit = report["normalization_audit"]
    assert audit["mode"] == "runtime"
    assert audit["scales_source"] == "computed_at_runtime"
    assert audit["computed_at_runtime"] is True
    assert report["normalization"]["scope"] == "runtime_global_before_codec_filtering"


def test_decision_receipt_characterization():
    report = _run_router(_real_fixture_args(), "receipt_report.json")
    selected = report["decision"]["selected"]
    receipt = report["decision_receipt"]

    assert receipt["artifact_type"] == "router_decision_receipt"
    assert receipt["receipt_schema_version"] == "0.29.0"
    assert receipt["decision"]["selected_codec"] == selected["codec"]
    assert receipt["decision"]["selected_config"] == selected["config"]
    assert receipt["decision"]["cost"] == selected["cost"]


def test_external_codec_disabled_by_default_characterization():
    report = _run_router(_real_fixture_args(), "external_disabled_report.json")

    assert report["external_codecs"] == {"enabled": False}
    assert report["decision"]["num_points_total"] == 6
    assert all(
        item["raw"].get("source") != "external_codec_manifest"
        for item in report["decision"]["scored_candidate_pool"]
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "src.router.rde_router",
        "src.router.external_codec_spec",
        "src.router.external_codec_probe",
        "src.router.external_codec_dry_run",
        "src.router.external_codec_benchmark",
        "src.router.external_codec_rde_exporter",
        "src.router.calibration_apply",
        "src.router.codecs.external_codec_spec",
        "src.router.codecs.external_codec_probe",
        "src.router.codecs.external_codec_dry_run",
        "src.router.codecs.external_codec_benchmark",
        "src.router.codecs.external_codec_rde_exporter",
        "src.router.calibration.calibration_apply",
        "src.router.decision_replay",
        "src.router.router_overhead_audit",
        "src.router.router_effectiveness_audit",
        "src.router.shadow_decision_comparison",
        "src.router.shadow_decision_validation",
        "src.router.feedback_analysis",
        "src.router.feedback_calibration_proposal",
        "src.router.feedback_proposal_validation",
        "src.router.feedback_calibration_promotion",
        "src.router.observability.decision_replay",
        "src.router.observability.router_overhead_audit",
        "src.router.observability.router_effectiveness_audit",
        "src.router.observability.shadow_decision_comparison",
        "src.router.observability.shadow_decision_validation",
        "src.router.observability.feedback_analysis",
        "src.router.observability.feedback_calibration_proposal",
        "src.router.observability.feedback_proposal_validation",
        "src.router.observability.feedback_calibration_promotion",
        "src.router.content_image_features",
        "src.router.content_image_manifest",
        "src.router.content_metadata_features",
        "src.router.content_metadata_policy",
        "src.router.content_classifier_model",
        "src.router.system_features",
        "src.router.adaptation.content_image_features",
        "src.router.adaptation.content_image_manifest",
        "src.router.adaptation.content_metadata_features",
        "src.router.adaptation.content_metadata_policy",
        "src.router.adaptation.content_classifier_model",
        "src.router.adaptation.system_features",
        "src.router.build_normalization_profile",
        "src.router.core.build_normalization_profile",
        "src.router.content_oracle_analysis",
        "src.router.content_oracle_classifier",
        "src.router.content_oracle_classifier_sweep",
        "src.router.content_oracle_classifier_sklearn_ablation",
        "src.router.content_aware_benchmark_table",
        "src.router.content_aware_overhead_analysis",
        "src.router.content_aware_paper_artifacts",
        "src.router.analysis.content_oracle_analysis",
        "src.router.analysis.content_oracle_classifier",
        "src.router.analysis.content_oracle_classifier_sweep",
        "src.router.analysis.content_oracle_classifier_sklearn_ablation",
        "src.router.analysis.content_aware_benchmark_table",
        "src.router.analysis.content_aware_overhead_analysis",
        "src.router.analysis.content_aware_paper_artifacts",
    ],
)
def test_cli_help_smoke(module_name: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
