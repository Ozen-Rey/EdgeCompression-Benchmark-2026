"""Centralized router version metadata.

``FEATURE_GROUPS`` collects the router's feature flags by category. The
keys within each group are mutually exclusive across all groups, and
``FEATURE_LEVEL`` is the flat ``{feature_name: maturity_level}`` dict
derived from them; consumers that already read ``FEATURE_LEVEL`` (the
report, characterization tests, etc.) see the same surface they did
before v0.42.29. Adding a new feature should be done by inserting it
into the appropriate group below.
"""

from typing import Dict


# Keep ROUTER_VERSION and pyproject.toml [project].version in sync.
# They identify the same release; the dispatcher and the package
# metadata are read by different tools but must agree on the version.
ROUTER_VERSION = "0.43.6.1"


FEATURE_GROUPS: Dict[str, Dict[str, str]] = {
    "core_runtime": {
        "rde_selection": "stable",
        "safe_mode": "stable",
        "execution_plan": "partial",
        "backend_execution": "partial",
        "bounded_nvml_probe": "timeout_fallback",
    },
    "local_energy": {
        "local_calibration": "experimental",
        "local_energy": "windows_gpu_partial_provenance",
    },
    "calibration_bundle": {
        "calibration_bundle_manifest": "audit_provenance",
        "calibration_bundle_consumption": "explicit_manifest_validated",
        "calibration_bundle_codec_fingerprints": "explicit_bundle_staleness_gate",
        "calibration_impact_audit": "read_only_bundle_impact_audit",
        "validated_bundle_consumption": "explicit_validation_required",
        "cross_artifact_integrity": "validation_bound_by_artifact_hash",
    },
    "feedback_loop": {
        "online_feedback": "append_only_observational",
        "feedback_analysis": "read_only_prediction_audit",
        "feedback_calibration_proposal": "shadow_proposal_only",
        "feedback_proposal_validation": "offline_validation_only",
        "feedback_calibration_promotion": "candidate_profile_only",
        "promoted_calibration_apply": "explicit_opt_in",
    },
    "shadow_offline": {
        "shadow_decision_comparison": "read_only_offline_comparison",
        "shadow_decision_validation": "read_only_methodology_gate",
    },
    "audit_observability": {
        "decision_receipt": "audit_replay_receipt",
        "decision_replay": "offline_reproducibility_check",
        "router_overhead_audit": "read_only_performance_audit",
        "router_effectiveness_audit": "read_only_baseline_policy_audit",
        "effectiveness_cost_explainability": "read_only_audit_cost_completion",
        "decision_observability_export": "scored_pool_normalization_audit",
        "normalization_consistency_audit": "previous_receipt_report_only",
        "energy_provenance_tier_reporting": "report_only_observability",
        "energy_provenance_compatibility_audit": "report_only_observability",
        "energy_tier_policy_shadow": "report_only_observability",
    },
    "external_codecs": {
        "external_codec_spec_schema": "offline_validation_only",
        "external_codec_probe": "schema_validated_version_fingerprint_report_only",
        "external_codec_dry_run": "single_input_contract_validation_only",
        "external_codec_benchmark": "raw_measurement_report_only",
        "external_codec_rde_exporter": "offline_rde_csv_export_only",
        "external_codec_registry_integration": "explicit_manifest_router_input_only",
    },
    "refactor_infrastructure": {
        "legacy_import_audit": "read_only_repo_audit",
    },
    "adaptive_policies": {
        "system_aware": "experimental",
        "content_aware": "offline_validated",
        "content_classifier": "experimental",
    },
}


def _flatten_feature_groups(groups: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Flatten FEATURE_GROUPS into the historical FEATURE_LEVEL dict.

    Raises ``ValueError`` at import time if any feature name appears in
    more than one group, so accidental duplicates fail fast instead of
    silently shadowing.
    """
    flat: Dict[str, str] = {}
    for group_name, features in groups.items():
        for feature_name, level in features.items():
            if feature_name in flat:
                raise ValueError(
                    f"Duplicate feature flag {feature_name!r} across "
                    f"FEATURE_GROUPS (re-declared in group {group_name!r})"
                )
            flat[feature_name] = level
    return flat


FEATURE_LEVEL: Dict[str, str] = _flatten_feature_groups(FEATURE_GROUPS)


DOMAIN_SUPPORT: Dict[str, str] = {
    "image": "primary",
    "video": "benchmark_analysis",
    "audio": "benchmark_analysis",
}
