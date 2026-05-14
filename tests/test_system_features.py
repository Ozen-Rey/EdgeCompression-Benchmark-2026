from src.router.adaptation.system_features import (
    build_system_features,
    clear_system_feature_cache,
    derive_system_constraints,
    estimate_probe_efficiency,
)


def test_derive_system_constraints_detects_memory_battery_gpu_pressure():
    features = {
        "dynamic": {
            "cpu": {
                "cpu_percent": 90.0,
            },
            "memory": {
                "percent": 88.0,
                "available_ratio": 0.15,
            },
            "swap": {
                "percent": 25.0,
            },
            "battery": {
                "power_mode": "battery",
                "percent": 12.0,
            },
            "disk": {
                "percent": 91.0,
                "free_gb": 1.5,
            },
        },
        "gpu": {
            "cuda_available": True,
            "primary_gpu": {
                "utilization_percent": 85.0,
                "memory_free_ratio": 0.10,
                "temperature_c": 91.0,
            },
        },
    }

    constraints = derive_system_constraints(features)

    assert constraints["is_cpu_busy"] is True
    assert constraints["is_memory_constrained"] is True
    assert constraints["is_memory_critical"] is False
    assert constraints["is_swap_active"] is True
    assert constraints["is_on_battery"] is True
    assert constraints["is_battery_low"] is True
    assert constraints["is_battery_critical"] is True
    assert constraints["is_gpu_available"] is True
    assert constraints["is_gpu_busy"] is True
    assert constraints["is_gpu_memory_constrained"] is True
    assert constraints["is_thermal_constrained"] is True
    assert constraints["is_thermal_critical"] is True
    assert constraints["is_disk_constrained"] is True

    assert constraints["classes"]["cpu"] == "busy"
    assert constraints["classes"]["memory"] == "constrained"
    assert constraints["classes"]["battery"] == "critical"
    assert constraints["classes"]["gpu"] == "memory_constrained"
    assert constraints["classes"]["thermal"] == "critical"
    assert constraints["classes"]["disk"] == "low_space"


def test_estimate_probe_efficiency_classifies_overhead():
    excellent = estimate_probe_efficiency(
        probe_overhead_ms=1.0,
        reference_time_ms=200.0,
    )
    assert excellent["classification"] == "excellent"

    acceptable = estimate_probe_efficiency(
        probe_overhead_ms=5.0,
        reference_time_ms=200.0,
    )
    assert acceptable["classification"] == "acceptable"

    warning = estimate_probe_efficiency(
        probe_overhead_ms=15.0,
        reference_time_ms=200.0,
    )
    assert warning["classification"] == "warning"

    too_high = estimate_probe_efficiency(
        probe_overhead_ms=50.0,
        reference_time_ms=200.0,
    )
    assert too_high["classification"] == "too_high"


def test_derive_system_constraints_marks_gpu_unknown_when_probe_skipped():
    features = {
        "dynamic": {
            "cpu": {"cpu_percent": 10.0},
            "memory": {"percent": 40.0, "available_ratio": 0.6},
            "swap": {"percent": 0.0},
            "battery": {"power_mode": "unknown", "percent": None},
            "disk": {"percent": 50.0, "free_gb": 100.0},
        },
        "gpu": {
            "skipped": True,
            "cuda_available": None,
            "primary_gpu": None,
        },
    }

    constraints = derive_system_constraints(features)

    assert constraints["is_gpu_available"] is None
    assert constraints["classes"]["gpu"] == "unknown"


def test_estimate_probe_efficiency_handles_missing_reference():
    result = estimate_probe_efficiency(
        probe_overhead_ms=10.0,
        reference_time_ms=None,
    )

    assert result["enabled"] is False
    assert result["reason"] == "missing_reference_time_ms"


def test_build_system_features_basic_has_required_sections():
    clear_system_feature_cache()

    report = build_system_features(
        probe_level="basic",
        cache_ttl_s=5.0,
        cpu_interval_s=0.0,
    )

    assert report["enabled"] is True
    assert report["version"] == "0.8"
    assert report["probe_level"] == "basic"

    assert "probe_overhead" in report
    assert "features" in report
    assert "derived_constraints" in report

    assert report["probe_overhead"]["total_probe_ms"] >= 0.0
    assert "static" in report["features"]
    assert "dynamic" in report["features"]
    assert "gpu" in report["features"]

    assert report["features"]["gpu"]["skipped"] is True


def test_build_system_features_uses_cache_on_second_call():
    clear_system_feature_cache()

    first = build_system_features(
        probe_level="basic",
        cache_ttl_s=30.0,
        cpu_interval_s=0.0,
    )

    second = build_system_features(
        probe_level="basic",
        cache_ttl_s=30.0,
        cpu_interval_s=0.0,
    )

    assert first["cache"]["static_cache_hit"] is False
    assert first["cache"]["dynamic_cache_hit"] is False

    assert second["cache"]["static_cache_hit"] is True
    assert second["cache"]["dynamic_cache_hit"] is True
