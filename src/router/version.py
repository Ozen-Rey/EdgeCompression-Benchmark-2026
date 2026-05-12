"""Centralized router version metadata."""

ROUTER_VERSION = "0.28.0"

FEATURE_LEVEL = {
    "rde_selection": "stable",
    "safe_mode": "stable",
    "execution_plan": "partial",
    "backend_execution": "partial",
    "local_calibration": "experimental",
    "local_energy": "windows_gpu_partial_provenance",
    "online_feedback": "append_only_observational",
    "feedback_analysis": "read_only_prediction_audit",
    "feedback_calibration_proposal": "shadow_proposal_only",
    "feedback_proposal_validation": "offline_validation_only",
    "feedback_calibration_promotion": "candidate_profile_only",
    "promoted_calibration_apply": "explicit_opt_in",
    "calibration_bundle_manifest": "audit_provenance",
    "calibration_bundle_consumption": "explicit_manifest_validated",
    "calibration_impact_audit": "read_only_bundle_impact_audit",
    "shadow_decision_comparison": "read_only_offline_comparison",
    "shadow_decision_validation": "read_only_methodology_gate",
    "validated_bundle_consumption": "explicit_validation_required",
    "cross_artifact_integrity": "validation_bound_by_artifact_hash",
    "decision_receipt": "audit_replay_receipt",
    "decision_replay": "offline_reproducibility_check",
    "router_overhead_audit": "read_only_performance_audit",
    "router_effectiveness_audit": "read_only_baseline_policy_audit",
    "effectiveness_cost_explainability": "read_only_audit_cost_completion",
    "system_aware": "experimental",
    "content_aware": "offline_validated",
    "content_classifier": "experimental",
}

DOMAIN_SUPPORT = {
    "image": "primary",
    "video": "benchmark_analysis",
    "audio": "benchmark_analysis",
}
