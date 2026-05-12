"""Centralized router version metadata."""

ROUTER_VERSION = "0.14.0"

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
    "system_aware": "experimental",
    "content_aware": "offline_validated",
    "content_classifier": "experimental",
}

DOMAIN_SUPPORT = {
    "image": "primary",
    "video": "benchmark_analysis",
    "audio": "benchmark_analysis",
}
