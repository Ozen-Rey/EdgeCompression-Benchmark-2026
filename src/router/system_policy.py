from typing import Any, Dict


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(weights["w_R"]) + float(weights["w_E"]) + float(weights["w_D"])

    if total <= 0:
        raise ValueError("System policy produced non-positive weight sum.")

    return {
        "w_R": float(weights["w_R"]) / total,
        "w_E": float(weights["w_E"]) / total,
        "w_D": float(weights["w_D"]) / total,
    }


def build_system_policy(
    *,
    base_weights: Dict[str, float],
    system_features_report: Dict[str, Any],
    enabled: bool,
    mode: str = "report-only",
) -> Dict[str, Any]:
    mode = str(mode).strip().lower()

    if mode not in {"report-only", "apply"}:
        raise ValueError("system policy mode must be 'report-only' or 'apply'.")

    if not enabled:
        return {
            "enabled": False,
            "mode": mode,
            "reason": "system_policy_disabled",
            "base_weights": base_weights,
            "suggested_weights": base_weights,
            "effective_weights": base_weights,
            "applied": False,
            "rules_applied": [],
            "suggested_filters": [],
            "warnings": [],
        }

    if not system_features_report.get("enabled", False):
        return {
            "enabled": False,
            "mode": mode,
            "reason": "system_features_not_available",
            "base_weights": base_weights,
            "suggested_weights": base_weights,
            "effective_weights": base_weights,
            "applied": False,
            "rules_applied": [],
            "suggested_filters": [],
            "warnings": [
                "system_policy_requested_but_system_features_disabled"
            ],
        }

    constraints = system_features_report.get("derived_constraints", {})
    classes = constraints.get("classes", {})

    multipliers = {
        "w_R": 1.0,
        "w_E": 1.0,
        "w_D": 1.0,
    }

    rules_applied = []
    suggested_filters = []
    warnings = []

    battery_class = classes.get("battery")
    thermal_class = classes.get("thermal")
    cpu_class = classes.get("cpu")
    memory_class = classes.get("memory")
    gpu_class = classes.get("gpu")
    disk_class = classes.get("disk")

    # Battery policy: energy becomes more important.
    if battery_class == "critical":
        multipliers["w_E"] *= 3.0
        rules_applied.append("battery_critical_energy_multiplier=3.0")
        warnings.append("battery_critical_prefer_low_energy_configs")
    elif battery_class == "low":
        multipliers["w_E"] *= 2.0
        rules_applied.append("battery_low_energy_multiplier=2.0")
    elif battery_class == "battery":
        multipliers["w_E"] *= 1.35
        rules_applied.append("on_battery_energy_multiplier=1.35")

    # Thermal policy: avoid expensive work.
    if thermal_class == "critical":
        multipliers["w_E"] *= 2.0
        rules_applied.append("thermal_critical_energy_multiplier=2.0")
        warnings.append("thermal_critical_avoid_heavy_backends")
    elif thermal_class == "hot":
        multipliers["w_E"] *= 1.5
        rules_applied.append("thermal_hot_energy_multiplier=1.5")

    # CPU pressure: prefer cheaper/faster configurations.
    if cpu_class == "busy":
        multipliers["w_E"] *= 1.25
        rules_applied.append("cpu_busy_energy_multiplier=1.25")
        warnings.append("cpu_busy_prefer_lightweight_backends")

    # Memory pressure: no direct R-D-E axis, but warn and suggest avoiding heavy codecs.
    if memory_class == "critical":
        multipliers["w_E"] *= 1.35
        rules_applied.append("memory_critical_energy_multiplier=1.35")
        suggested_filters.append("avoid_high_memory_codecs")
        warnings.append("memory_critical_avoid_high_memory_codecs")
    elif memory_class == "constrained":
        multipliers["w_E"] *= 1.15
        rules_applied.append("memory_constrained_energy_multiplier=1.15")
        suggested_filters.append("prefer_low_memory_codecs")

    # GPU pressure: useful once neural/CUDA codecs are enabled.
    if gpu_class == "busy":
        suggested_filters.append("avoid_gpu_heavy_codecs")
        warnings.append("gpu_busy_avoid_cuda_neural_backends")
    elif gpu_class == "memory_constrained":
        suggested_filters.append("avoid_gpu_memory_heavy_codecs")
        warnings.append("gpu_memory_constrained_avoid_large_cuda_models")
    elif gpu_class == "unavailable":
        suggested_filters.append("exclude_requires_cuda_codecs")
    elif gpu_class == "unknown":
        warnings.append("gpu_status_unknown_probe_level_did_not_check_gpu")

    # Disk pressure: execution/batch warning.
    if disk_class == "low_space":
        warnings.append("disk_low_space_execution_outputs_may_fail")

    raw_suggested = {
        "w_R": float(base_weights["w_R"]) * multipliers["w_R"],
        "w_E": float(base_weights["w_E"]) * multipliers["w_E"],
        "w_D": float(base_weights["w_D"]) * multipliers["w_D"],
    }

    suggested_weights = _normalize_weights(raw_suggested)

    applied = mode == "apply"
    effective_weights = suggested_weights if applied else dict(base_weights)

    return {
        "enabled": True,
        "mode": mode,
        "applied": applied,
        "base_weights": dict(base_weights),
        "multipliers": multipliers,
        "raw_suggested_weights": raw_suggested,
        "suggested_weights": suggested_weights,
        "effective_weights": effective_weights,
        "rules_applied": rules_applied,
        "suggested_filters": suggested_filters,
        "warnings": warnings,
        "constraint_classes": classes,
    }
