from dataclasses import dataclass

from src.router.core.normalization_profile import (
    build_normalization_profile,
    normalize_with_profile,
)


@dataclass
class Point:
    codec: str
    config: str
    rate: float
    quality: float
    energy: float
    time_ms: float | None = None


def test_global_normalization_profile_contains_scales():
    points = [
        Point("A", "low", rate=0.1, quality=50.0, energy=1.0),
        Point("B", "high", rate=1.0, quality=90.0, energy=10.0),
    ]

    profile = build_normalization_profile(
        points=points,
        domain="image",
        mode="global",
        build_scope="all",
    )

    assert profile["mode"] == "global"
    assert profile["build_scope"] == "all"
    assert profile["scales"]["rate"]["min"] < profile["scales"]["rate"]["max"]
    assert profile["scales"]["energy"]["min"] < profile["scales"]["energy"]["max"]
    assert profile["scales"]["quality"]["min"] == 50.0
    assert profile["scales"]["quality"]["max"] == 90.0


def test_normalize_with_profile_maps_best_quality_to_low_distortion():
    points = [
        Point("A", "low", rate=0.1, quality=50.0, energy=1.0),
        Point("B", "high", rate=1.0, quality=90.0, energy=10.0),
    ]

    profile = build_normalization_profile(points=points, domain="image")

    best = normalize_with_profile(points[1], profile)
    worst = normalize_with_profile(points[0], profile)

    assert best["norm_distortion"] == 0.0
    assert worst["norm_distortion"] == 1.0
