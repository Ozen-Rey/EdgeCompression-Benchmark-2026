"""Post-removal checks for retired top-level router wrapper modules."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


RETIRED_WRAPPER_FILES = [
    "build_normalization_profile.py",
    "calibration.py",
    "calibration_apply.py",
    "calibration_bundle.py",
    "calibration_impact_audit.py",
    "codec_capabilities.py",
    "codec_fingerprints.py",
    "content_aware_benchmark_table.py",
    "content_aware_overhead_analysis.py",
    "content_aware_paper_artifacts.py",
    "content_classifier_model.py",
    "content_image_features.py",
    "content_image_manifest.py",
    "content_metadata_features.py",
    "content_metadata_policy.py",
    "content_oracle_analysis.py",
    "content_oracle_classifier.py",
    "content_oracle_classifier_sklearn_ablation.py",
    "content_oracle_classifier_sweep.py",
    "content_policy.py",
    "context_policy.py",
    "decision_receipt.py",
    "decision_replay.py",
    "energy_provenance.py",
    "energy_provenance_compatibility.py",
    "energy_tier_policy.py",
    "external_codec_benchmark.py",
    "external_codec_dry_run.py",
    "external_codec_probe.py",
    "external_codec_rde_exporter.py",
    "external_codec_registry.py",
    "external_codec_spec.py",
    "feedback_analysis.py",
    "feedback_calibration_promotion.py",
    "feedback_calibration_proposal.py",
    "feedback_logger.py",
    "feedback_proposal_validation.py",
    "normalization_consistency.py",
    "normalization_profile.py",
    "profiles.py",
    "quality_thresholds.py",
    "rde_database.py",
    "router_config.py",
    "router_effectiveness_audit.py",
    "router_overhead_audit.py",
    "run_manifest.py",
    "shadow_decision_comparison.py",
    "shadow_decision_validation.py",
    "simple_image_encoder.py",
    "system_features.py",
    "system_penalty.py",
    "system_policy.py",
    "system_probe.py",
]


@pytest.mark.parametrize("file_name", RETIRED_WRAPPER_FILES)
def test_legacy_wrapper_file_is_removed(file_name: str):
    assert not (Path("src") / "router" / file_name).exists()


@pytest.mark.parametrize(
    ("module_name", "representative"),
    [
        ("src.router.core.rde_database", "select_best_rde"),
        ("src.router.core.quality_thresholds", "resolve_quality_floor"),
        ("src.router.codecs.external_codec_spec", "validate_external_codec_spec"),
        (
            "src.router.calibration.calibration_bundle",
            "validate_calibration_bundle_manifest",
        ),
        ("src.router.adaptation.energy_provenance", "classify_energy_provenance"),
        ("src.router.observability.decision_receipt", "build_decision_receipt"),
        ("src.router.analysis.content_oracle_analysis", "analyze_content_oracle"),
    ],
)
def test_subpackage_replacement_import_still_works(
    module_name: str, representative: str
):
    module = importlib.import_module(module_name)

    assert hasattr(module, representative)
