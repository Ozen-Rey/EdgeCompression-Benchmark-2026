"""Internal router run context for orchestration/report metadata."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RouterContext:
    """Carries internal metadata that should not live on CLI args."""

    run_manifest: Optional[Dict[str, Any]] = None
    router_config_report: Optional[Dict[str, Any]] = None
    codec_registry_report: Optional[Dict[str, Any]] = None
    calibration_bundle_report: Optional[Dict[str, Any]] = None
    calibration_bundle_validation_report: Optional[Dict[str, Any]] = None
    calibration_report: Optional[Dict[str, Any]] = None
    external_codecs_report: Optional[Dict[str, Any]] = None
    quality_threshold_report: Optional[Dict[str, Any]] = None
    normalization_profile: Optional[Dict[str, Any]] = None
    normalization_report: Optional[Dict[str, Any]] = None
    normalization_consistency_report: Optional[Dict[str, Any]] = None
    energy_provenance_report: Optional[Dict[str, Any]] = None
    energy_provenance_summary: Optional[Dict[str, Any]] = None
    energy_provenance_compatibility: Optional[Dict[str, Any]] = None
    energy_tier_policy: Optional[Dict[str, Any]] = None
    system_features_report: Optional[Dict[str, Any]] = None
    system_policy_report: Optional[Dict[str, Any]] = None
    system_policy_simulation: Optional[Dict[str, Any]] = None
    system_penalty_report: Optional[Dict[str, Any]] = None
    system_penalty_weights_report: Optional[Dict[str, Any]] = None
    time_guard_report: Optional[Dict[str, Any]] = None
    content_policy_report: Optional[Dict[str, Any]] = None
    content_classifier_report: Optional[Dict[str, Any]] = None
    content_filter_report: Optional[Dict[str, Any]] = None
    csv_row_diagnostics: Optional[Dict[str, Any]] = None
    domain_spec_report: Optional[Dict[str, Any]] = None
    preferred_candidate_source: Optional[str] = None
