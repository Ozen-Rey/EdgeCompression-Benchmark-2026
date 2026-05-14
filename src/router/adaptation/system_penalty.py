import copy
import json
from pathlib import Path
from typing import Any, Dict

from src.router.codecs.codec_capabilities import get_codec_capability


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


DEFAULT_SYSTEM_PENALTY_WEIGHTS = {
    "battery": {
        "critical_energy": 0.10,
        "low_energy": 0.06,
        "battery_energy": 0.03,
    },
    "cpu": {
        "busy_cpu": 0.08,
    },
    "memory": {
        "critical_memory": 0.15,
        "constrained_memory": 0.10,
    },
    "gpu": {
        "busy_gpu": 0.12,
        "memory_constrained_gpu": 0.15,
    },
    "thermal": {
        "critical_resource": 0.10,
        "hot_resource": 0.06,
    },
    "latency": {
        "constrained_latency": 0.08,
    },
    "interactive": {
        "risk": 0.12,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)

    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(out.get(key), dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value

    return out


def load_system_penalty_weights(path: str | None = None) -> Dict[str, Any]:
    if path is None or str(path).strip() == "":
        return {
            "source": None,
            "source_exists": False,
            "weights": copy.deepcopy(DEFAULT_SYSTEM_PENALTY_WEIGHTS),
        }

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"System penalty weights file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("System penalty weights file root must be a JSON object.")

    merged = _deep_merge(DEFAULT_SYSTEM_PENALTY_WEIGHTS, data)

    # Remove metadata keys from the active coefficient tree.
    merged.pop("version", None)
    merged.pop("description", None)

    return {
        "source": str(p),
        "source_exists": True,
        "weights": merged,
    }


def _coef(weights: Dict[str, Any], section: str, key: str) -> float:
    return float(
        weights
        .get(section, {})
        .get(key, DEFAULT_SYSTEM_PENALTY_WEIGHTS[section][key])
    )


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
    penalty_weights: Dict[str, Any] | None = None,
    penalty_weights_source: str | None = None,
) -> Dict[str, Any]:
    mode = str(mode).strip().lower()

    active_weights = (
        copy.deepcopy(penalty_weights)
        if penalty_weights is not None
        else copy.deepcopy(DEFAULT_SYSTEM_PENALTY_WEIGHTS)
    )

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
            "penalty_weights_source": penalty_weights_source,
            "penalty_weights": active_weights,
        }

    if not system_features_report.get("enabled", False):
        return {
            "enabled": False,
            "mode": mode,
            "applied": False,
            "lambda_sys": lambda_sys,
            "reason": "system_features_not_available",
            "penalty_weights_source": penalty_weights_source,
            "penalty_weights": active_weights,
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
        "penalty_weights_source": penalty_weights_source,
        "penalty_weights": active_weights,
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
    weights = context.get("penalty_weights", DEFAULT_SYSTEM_PENALTY_WEIGHTS)

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
        add(
            _coef(weights, "battery", "critical_energy") * energy_score,
            f"battery_critical_energy_score={energy_score}",
        )
        warnings.append("battery_critical_penalize_energy_heavy_codecs")
    elif battery_class == "low":
        add(
            _coef(weights, "battery", "low_energy") * energy_score,
            f"battery_low_energy_score={energy_score}",
        )
    elif battery_class == "battery":
        add(
            _coef(weights, "battery", "battery_energy") * energy_score,
            f"on_battery_energy_score={energy_score}",
        )

    # CPU pressure.
    if cpu_class == "busy":
        add(
            _coef(weights, "cpu", "busy_cpu") * cpu_score,
            f"cpu_busy_cpu_score={cpu_score}",
        )

    # Memory pressure.
    if memory_class == "critical":
        add(
            _coef(weights, "memory", "critical_memory") * memory_score,
            f"memory_critical_memory_score={memory_score}",
        )
        warnings.append("memory_critical_penalize_memory_heavy_codecs")
    elif memory_class == "constrained":
        add(
            _coef(weights, "memory", "constrained_memory") * memory_score,
            f"memory_constrained_memory_score={memory_score}",
        )

    # GPU pressure.
    if gpu_class == "busy":
        add(
            _coef(weights, "gpu", "busy_gpu") * gpu_score,
            f"gpu_busy_gpu_score={gpu_score}",
        )
    elif gpu_class == "memory_constrained":
        add(
            _coef(weights, "gpu", "memory_constrained_gpu") * gpu_score,
            f"gpu_memory_constrained_gpu_score={gpu_score}",
        )
    elif gpu_class == "unknown" and requires_cuda:
        warnings.append("gpu_unknown_cuda_requirement_not_hard_excluded")

    # Thermal pressure.
    if thermal_class == "critical":
        thermal_score = max(cpu_score, gpu_score, energy_score)
        add(
            _coef(weights, "thermal", "critical_resource") * thermal_score,
            f"thermal_critical_resource_score={thermal_score}",
        )
    elif thermal_class == "hot":
        thermal_score = max(cpu_score, gpu_score, energy_score)
        add(
            _coef(weights, "thermal", "hot_resource") * thermal_score,
            f"thermal_hot_resource_score={thermal_score}",
        )

    # Latency pressure.
    if context.get("latency_constrained", False):
        add(
            _coef(weights, "latency", "constrained_latency") * latency_score,
            f"latency_constrained_latency_score={latency_score}",
        )

    # Interactive/operational risk for heavy backends under pressure.
    if (
        not interactive_ok
        and (battery_class in {"critical", "low"} or cpu_class == "busy")
    ):
        add(
            _coef(weights, "interactive", "risk"),
            "interactive_risk_interactive_ok=false",
        )

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
