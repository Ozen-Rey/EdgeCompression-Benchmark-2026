import pytest

from src.router.adaptation.system_policy import (
    apply_system_policy_simulation,
    build_system_policy,
    parse_system_policy_simulation,
)


def _features_with_classes(**classes):
    default = {
        "cpu": "normal",
        "memory": "normal",
        "battery": "ac",
        "gpu": "available",
        "thermal": "nominal",
        "disk": "normal",
    }
    default.update(classes)

    return {
        "enabled": True,
        "derived_constraints": {
            "classes": default,
        },
    }


def test_system_policy_disabled_keeps_weights():
    base = {
        "w_R": 0.2,
        "w_E": 0.2,
        "w_D": 0.6,
    }

    policy = build_system_policy(
        base_weights=base,
        system_features_report={},
        enabled=False,
        mode="apply",
    )

    assert policy["enabled"] is False
    assert policy["applied"] is False
    assert policy["effective_weights"] == base


def test_system_policy_report_only_does_not_apply_weights():
    base = {
        "w_R": 0.2,
        "w_E": 0.2,
        "w_D": 0.6,
    }

    features = _features_with_classes(
        battery="critical",
        thermal="hot",
    )

    policy = build_system_policy(
        base_weights=base,
        system_features_report=features,
        enabled=True,
        mode="report-only",
    )

    assert policy["enabled"] is True
    assert policy["applied"] is False
    assert policy["effective_weights"] == base
    assert policy["suggested_weights"]["w_E"] > base["w_E"]
    assert "battery_critical_energy_multiplier=3.0" in policy["rules_applied"]


def test_system_policy_apply_changes_effective_weights():
    base = {
        "w_R": 0.2,
        "w_E": 0.2,
        "w_D": 0.6,
    }

    features = _features_with_classes(
        battery="low",
        cpu="busy",
        memory="constrained",
    )

    policy = build_system_policy(
        base_weights=base,
        system_features_report=features,
        enabled=True,
        mode="apply",
    )

    assert policy["enabled"] is True
    assert policy["applied"] is True
    assert policy["effective_weights"]["w_E"] > base["w_E"]
    assert policy["effective_weights"]["w_D"] < base["w_D"]
    assert "prefer_low_memory_codecs" in policy["suggested_filters"]


def test_system_policy_warns_when_gpu_unavailable():
    base = {
        "w_R": 0.33,
        "w_E": 0.33,
        "w_D": 0.34,
    }

    features = _features_with_classes(gpu="unavailable")

    policy = build_system_policy(
        base_weights=base,
        system_features_report=features,
        enabled=True,
        mode="report-only",
    )

    assert "exclude_requires_cuda_codecs" in policy["suggested_filters"]


def test_system_policy_does_not_exclude_cuda_when_gpu_unknown():
    base = {
        "w_R": 0.2,
        "w_E": 0.2,
        "w_D": 0.6,
    }

    features = _features_with_classes(gpu="unknown")

    policy = build_system_policy(
        base_weights=base,
        system_features_report=features,
        enabled=True,
        mode="report-only",
    )

    assert "exclude_requires_cuda_codecs" not in policy["suggested_filters"]
    assert "gpu_status_unknown_probe_level_did_not_check_gpu" in policy["warnings"]


def test_parse_system_policy_simulation_parses_classes():
    parsed = parse_system_policy_simulation(
        "battery=critical,cpu=busy,memory=constrained"
    )

    assert parsed == {
        "battery": "critical",
        "cpu": "busy",
        "memory": "constrained",
    }


def test_parse_system_policy_simulation_rejects_invalid_key():
    with pytest.raises(ValueError, match="Invalid simulated system class key"):
        parse_system_policy_simulation("network=bad")


def test_apply_system_policy_simulation_overrides_classes():
    report = {
        "enabled": True,
        "derived_constraints": {
            "classes": {
                "cpu": "normal",
                "memory": "normal",
                "battery": "ac",
                "gpu": "available",
                "thermal": "nominal",
                "disk": "normal",
            }
        },
    }

    simulated = {
        "battery": "critical",
        "cpu": "busy",
    }

    out = apply_system_policy_simulation(
        system_features_report=report,
        simulated_classes=simulated,
    )

    classes = out["derived_constraints"]["classes"]

    assert classes["battery"] == "critical"
    assert classes["cpu"] == "busy"
    assert classes["memory"] == "normal"
    assert out["system_policy_simulation"]["enabled"] is True
