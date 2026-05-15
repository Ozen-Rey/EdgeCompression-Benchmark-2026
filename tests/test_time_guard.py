from dataclasses import dataclass
import pytest

from src.router.profile_runner import build_time_guard_report


@dataclass
class Point:
    codec: str
    config: str
    time_ms: float | None


def test_time_guard_reports_available_and_over_limit_points():
    points = [
        Point("JPEG", "q=60", 5.0),
        Point("JXL", "d=1.0", 130.0),
        Point("HEVC", "crf=15", 300.0),
    ]

    report = build_time_guard_report(
        points=points,
        max_time_ms=150.0,
        strict_time=False,
    )

    assert report["enabled"] is True
    assert report["num_candidate_points"] == 3
    assert report["num_with_time"] == 3
    assert report["num_within_limit"] == 2
    assert report["num_over_limit"] == 1


def test_time_guard_raises_when_no_time_available():
    points = [
        Point("JPEG", "q=60", None),
        Point("JXL", "d=1.0", None),
    ]

    with pytest.raises(ValueError, match="no time data is available"):
        build_time_guard_report(
            points=points,
            max_time_ms=150.0,
            strict_time=False,
        )


def test_strict_time_raises_when_some_time_missing():
    points = [
        Point("JPEG", "q=60", 5.0),
        Point("JXL", "d=1.0", None),
    ]

    with pytest.raises(ValueError, match="Strict time guard requested"):
        build_time_guard_report(
            points=points,
            max_time_ms=150.0,
            strict_time=True,
        )
