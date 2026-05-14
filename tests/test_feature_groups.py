"""Invariants for the router feature-flag groups introduced in v0.42.29.

FEATURE_GROUPS is the canonical declaration; FEATURE_LEVEL is the flat
view derived from it. These tests pin the relationship so that adding
or moving a feature flag fails fast when:

- a feature is duplicated across two groups,
- the flat FEATURE_LEVEL drifts from the sum of the group dicts,
- a group key or feature key is empty, or
- a maturity level is not a non-empty string.
"""

from src.router.version import FEATURE_GROUPS, FEATURE_LEVEL


def _all_group_items():
    for group_name, features in FEATURE_GROUPS.items():
        for feature_name, level in features.items():
            yield group_name, feature_name, level


def test_feature_level_matches_flattened_feature_groups():
    flattened = {
        feature_name: level
        for _, feature_name, level in _all_group_items()
    }

    assert flattened == FEATURE_LEVEL


def test_no_feature_appears_in_two_groups():
    seen: dict[str, str] = {}
    for group_name, feature_name, _ in _all_group_items():
        assert feature_name not in seen, (
            f"feature {feature_name!r} appears in both "
            f"{seen[feature_name]!r} and {group_name!r}"
        )
        seen[feature_name] = group_name


def test_every_feature_level_value_is_non_empty_string():
    for group_name, feature_name, level in _all_group_items():
        assert isinstance(level, str), (
            f"{group_name}.{feature_name} level is not a string: {level!r}"
        )
        assert level.strip() == level, (
            f"{group_name}.{feature_name} level has surrounding whitespace"
        )
        assert level, f"{group_name}.{feature_name} level is empty"


def test_every_group_and_feature_key_is_non_empty():
    for group_name, features in FEATURE_GROUPS.items():
        assert isinstance(group_name, str) and group_name
        assert features, f"group {group_name!r} is empty"
        for feature_name in features:
            assert isinstance(feature_name, str) and feature_name


def test_feature_level_preserves_pre_v0_42_29_flag_set():
    """Lock the historical feature-flag set so regrouping cannot silently drop one."""
    expected = {
        "rde_selection",
        "safe_mode",
        "execution_plan",
        "backend_execution",
        "local_calibration",
        "local_energy",
        "online_feedback",
        "feedback_analysis",
        "feedback_calibration_proposal",
        "feedback_proposal_validation",
        "feedback_calibration_promotion",
        "promoted_calibration_apply",
        "calibration_bundle_manifest",
        "calibration_bundle_consumption",
        "calibration_bundle_codec_fingerprints",
        "calibration_impact_audit",
        "shadow_decision_comparison",
        "shadow_decision_validation",
        "validated_bundle_consumption",
        "cross_artifact_integrity",
        "decision_receipt",
        "decision_replay",
        "router_overhead_audit",
        "router_effectiveness_audit",
        "effectiveness_cost_explainability",
        "decision_observability_export",
        "normalization_consistency_audit",
        "energy_provenance_tier_reporting",
        "energy_provenance_compatibility_audit",
        "energy_tier_policy_shadow",
        "external_codec_spec_schema",
        "external_codec_probe",
        "external_codec_dry_run",
        "external_codec_benchmark",
        "external_codec_rde_exporter",
        "external_codec_registry_integration",
        "legacy_import_audit",
        "bounded_nvml_probe",
        "system_aware",
        "content_aware",
        "content_classifier",
    }

    assert set(FEATURE_LEVEL.keys()) == expected
