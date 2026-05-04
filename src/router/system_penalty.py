from typing import Any, Dict

try:
    from .codec_capabilities import get_codec_capability
except ImportError:
    from codec_capabilities import get_codec_capability


LEVEL_SCORE = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "very-high": 4,
    "very_high": 4,
}


DEFAULT_RESOURCE_PROFILE = {
    "cpu_load": "medium",
    "memory": "medium",
    "gpu": "none",
    "latency": "medium",
    "energy": "medium",
    "batch_friendly": True,
    "interactive_ok": True,
}


def _score(value: Any) -> int:
    return LEVEL_SCORE.get(str(value or "medium").strip().lower(), 2)


def _classes_from_features(system_features_report: Dict[str, Any]) -> Dict[str, str]:
    return (
        system_features_report
        .get("derived_constraints", {})
        .get("classes", {})
    )


def build_system_penalty_context(
    *,
    enabled: bool,
    mode: str,
    lambda_sys: float,
    system_features_report: Dict[str, Any],
    latency_constrained: bool = False,
    execution_requested: bool = False,
) -> Dict[str, Any]:
    mode = str(mode).strip().lower()

    if mode not in {"report-only", "apply"}:
        raise ValueError("system penalty mode must be 'report-only' or 'apply'.")

    if lambda_sys < 0:
        raise ValueError("system penalty lambda must be non-negative.")

    if not enabled:
        return {
            "enabled": False,
            "mode": mode,
            "applied": False,
            "lambda_sys": lambda_sys,
            "reason": "system_penalty_disabled",
        }

    if not system_features_report.get("enabled", False):
        return {
            "enabled": False,
            "mode": mode,
            "applied": False,
            "lambda_sys": lambda_sys,
            "reason": "system_features_not_available",
            "warnings": [
                "system_penalty_requested_but_system_features_disabled"
            ],
        }

    return {
        "enabled": True,
        "mode": mode,
        "applied": mode == "apply",
        "lambda_sys": float(lambda_sys),
        "latency_constrained": bool(latency_constrained),
        "execution_requested": bool(execution_requested),
        "constraint_classes": dict(_classes_from_features(system_features_report)),
    }


def compute_candidate_system_penalty(
    *,
    codec_name: str,
    config: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    if not context.get("enabled", False):
        return {
            "enabled": False,
            "penalty_norm": 0.0,
            "weighted_penalty": 0.0,
            "hard_excluded": False,
            "rules_applied": [],
            "warnings": [],
        }

    cap = get_codec_capability(codec_name)
    profile = dict(DEFAULT_RESOURCE_PROFILE)
    profile.update(cap.get("resource_profile", {}) or {})

    classes = context.get("constraint_classes", {})
    lambda_sys = float(context.get("lambda_sys", 0.0))

    cpu_score = _score(profile.get("cpu_load"))
    memory_score = _score(profile.get("memory"))
    gpu_score = _score(profile.get("gpu"))
    latency_score = _score(profile.get("latency"))
    energy_score = _score(profile.get("energy"))
    interactive_ok = bool(profile.get("interactive_ok", True))

    penalty = 0.0
    rules = []
    warnings = []
    hard_reasons = []

    def add(amount: float, rule: str) -> None:
        nonlocal penalty
        penalty += amount
        rules.append(rule)

    cpu_class = classes.get("cpu", "normal")
    memory_class = classes.get("memory", "normal")
    battery_class = classes.get("battery", "unknown")
    gpu_class = classes.get("gpu", "unknown")
    thermal_class = classes.get("thermal", "unknown")
    disk_class = classes.get("disk", "normal")

    requires_cuda = bool(cap.get("requires_cuda", False))

    # Hard feasibility/risk constraints.
    if gpu_class == "unavailable" and requires_cuda:
        hard_reasons.append("requires_cuda_but_gpu_unavailable")

    if memory_class == "critical" and memory_score >= 3:
        hard_reasons.append("memory_critical_high_memory_codec")

    # Battery/energy pressure.
    if battery_class == "critical":
        add(0.10 * energy_score, f"battery_critical_energy_score={energy_score}")
        warnings.append("battery_critical_penalize_energy_heavy_codecs")
    elif battery_class == "low":
        add(0.06 * energy_score, f"battery_low_energy_score={energy_score}")
    elif battery_class == "battery":
        add(0.03 * energy_score, f"on_battery_energy_score={energy_score}")

    # CPU pressure.
    if cpu_class == "busy":
        add(0.08 * cpu_score, f"cpu_busy_cpu_score={cpu_score}")

    # Memory pressure.
    if memory_class == "critical":
        add(0.15 * memory_score, f"memory_critical_memory_score={memory_score}")
        warnings.append("memory_critical_penalize_memory_heavy_codecs")
    elif memory_class == "constrained":
        add(0.10 * memory_score, f"memory_constrained_memory_score={memory_score}")

    # GPU pressure.
    if gpu_class == "busy":
        add(0.12 * gpu_score, f"gpu_busy_gpu_score={gpu_score}")
    elif gpu_class == "memory_constrained":
        add(0.15 * gpu_score, f"gpu_memory_constrained_gpu_score={gpu_score}")
    elif gpu_class == "unknown" and requires_cuda:
        warnings.append("gpu_unknown_cuda_requirement_not_hard_excluded")

    # Thermal pressure.
    if thermal_class == "critical":
        thermal_score = max(cpu_score, gpu_score, energy_score)
        add(0.10 * thermal_score, f"thermal_critical_resource_score={thermal_score}")
    elif thermal_class == "hot":
        thermal_score = max(cpu_score, gpu_score, energy_score)
        add(0.06 * thermal_score, f"thermal_hot_resource_score={thermal_score}")

    # Latency pressure.
    if context.get("latency_constrained", False):
        add(0.08 * latency_score, f"latency_constrained_latency_score={latency_score}")

    # Interactive/operational risk for heavy backends under pressure.
    if (
        not interactive_ok
        and (battery_class in {"critical", "low"} or cpu_class == "busy")
    ):
        add(0.15, "interactive_risk_interactive_ok=false")

    # Disk pressure: execution risk, not codec ranking by itself.
    if disk_class == "low_space" and context.get("execution_requested", False):
        warnings.append("disk_low_space_execution_outputs_may_fail")

    penalty_norm = max(0.0, min(1.0, penalty))
    weighted_penalty = lambda_sys * penalty_norm

    return {
        "enabled": True,
        "codec": codec_name,
        "config": config,
        "resource_profile": profile,
        "constraint_classes": classes,
        "lambda_sys": lambda_sys,
        "penalty_raw": penalty,
        "penalty_norm": penalty_norm,
        "weighted_penalty": weighted_penalty,
        "hard_excluded": bool(hard_reasons),
        "hard_exclusion_reasons": hard_reasons,
        "rules_applied": rules,
        "warnings": warnings,
    }


def make_system_penalty_fn(context: Dict[str, Any]):
    def fn(point):
        return compute_candidate_system_penalty(
            codec_name=point.codec,
            config=point.config,
            context=context,
        )

    return fn
