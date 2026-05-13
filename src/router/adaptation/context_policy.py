import math
from typing import Any, Dict, Optional


def _normalize_weights(w_e: float, w_r: float, w_d: float) -> Dict[str, float]:
    total = w_e + w_r + w_d

    if total <= 0:
        raise ValueError("La somma dei pesi deve essere positiva.")

    return {
        "w_E": w_e / total,
        "w_R": w_r / total,
        "w_D": w_d / total,
    }


def _battery_energy_boost(battery_percent: Optional[float]) -> float:
    """
    Boost continuo per l'energia.
    Batteria alta -> boost piccolo.
    Batteria bassa -> boost forte.
    """
    if battery_percent is None:
        battery_percent = 50.0

    battery_percent = max(0.0, min(100.0, battery_percent))

    # Sigmoide centrata intorno al 35%.
    # Sotto 35%, il peso energia cresce rapidamente.
    x = (35.0 - battery_percent) / 8.0
    return 3.0 / (1.0 + math.exp(-x))


def compute_context_policy(
    power_mode: str = "ac",
    battery_percent: Optional[float] = None,
    thermal_state: str = "nominal",
    network_profile: str = "normal",
    quality_target: str = "normal",
    system_load: str = "normal",
) -> Dict[str, Any]:
    """
    Traduce il contesto operativo in pesi R-D-E.

    La logica è volutamente esplicita e leggibile:
    - batteria bassa / temperatura alta -> più peso all'energia;
    - rete limitata -> più peso al rate;
    - target qualità alto -> più peso alla distorsione e soglia qualità più alta.
    """

    power_mode = power_mode.lower().strip()
    thermal_state = thermal_state.lower().strip()
    network_profile = network_profile.lower().strip()
    quality_target = quality_target.lower().strip()
    system_load = system_load.lower().strip()

    # Score grezzi prima della normalizzazione.
    score_e = 1.0
    score_r = 1.0
    score_d = 1.0

    suggested_min_quality: Optional[float] = 50.0
    rules_applied: list[str] = []

    # Power / battery.
    if power_mode == "battery":
        boost = _battery_energy_boost(battery_percent)
        score_e += boost
        rules_applied.append(f"battery_mode_energy_boost={boost:.3f}")

    elif power_mode == "ac":
        rules_applied.append("ac_power_no_battery_penalty")

    else:
        rules_applied.append("unknown_power_mode_no_adjustment")

    # Thermal state.
    if thermal_state == "warm":
        score_e += 0.5
        rules_applied.append("thermal_warm_energy_boost=0.5")
    elif thermal_state == "hot":
        score_e += 1.5
        rules_applied.append("thermal_hot_energy_boost=1.5")
    elif thermal_state == "critical":
        score_e += 3.0
        rules_applied.append("thermal_critical_energy_boost=3.0")
    else:
        rules_applied.append("thermal_nominal_no_adjustment")

    # Network.
    if network_profile == "limited":
        score_r += 1.5
        rules_applied.append("network_limited_rate_boost=1.5")
    elif network_profile in {"very-limited", "very_limited"}:
        score_r += 3.0
        rules_applied.append("network_very_limited_rate_boost=3.0")
    else:
        rules_applied.append("network_normal_no_adjustment")

    # Quality target.
    if quality_target == "preview":
        score_e += 1.0
        suggested_min_quality = 50.0
        rules_applied.append("quality_preview_energy_boost=1.0")
        rules_applied.append("quality_preview_min_quality=50")
    elif quality_target == "normal":
        suggested_min_quality = max(suggested_min_quality, 50.0)
        rules_applied.append("quality_normal_min_quality=50")
    elif quality_target == "high":
        score_d += 2.0
        suggested_min_quality = max(suggested_min_quality, 80.0)
        rules_applied.append("quality_high_distortion_boost=2.0")
        rules_applied.append("quality_high_min_quality=80")
    elif quality_target in {"very-high", "very_high"}:
        score_d += 4.0
        suggested_min_quality = max(suggested_min_quality, 90.0)
        rules_applied.append("quality_very_high_distortion_boost=4.0")
        rules_applied.append("quality_very_high_min_quality=90")
    else:
        rules_applied.append("unknown_quality_target_no_adjustment")

    # System load.
    if system_load == "high":
        score_e += 1.0
        rules_applied.append("system_load_high_energy_boost=1.0")
    elif system_load == "very-high" or system_load == "very_high":
        score_e += 2.0
        rules_applied.append("system_load_very_high_energy_boost=2.0")
    else:
        rules_applied.append("system_load_normal_no_adjustment")

    weights = _normalize_weights(score_e, score_r, score_d)

    return {
        "context": {
            "power_mode": power_mode,
            "battery_percent": battery_percent,
            "thermal_state": thermal_state,
            "network_profile": network_profile,
            "quality_target": quality_target,
            "system_load": system_load,
        },
        "scores_before_normalization": {
            "score_E": score_e,
            "score_R": score_r,
            "score_D": score_d,
        },
        "weights": weights,
        "suggested_min_quality": suggested_min_quality,
        "rules_applied": rules_applied,
    }
