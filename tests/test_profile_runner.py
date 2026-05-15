"""Unit tests for ``src.router.profile_runner`` helpers.

These tests exercise the side-effect-free helpers
(``apply_preferred_candidate_override``, ``build_time_guard_report``,
``build_content_classifier_router_report``) in isolation from the rest
of the router pipeline so their schemas and edge cases are verifiable
without spinning up a full ``run_profile`` flow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from src.router import profile_runner
from src.router.profile_runner import (
    apply_preferred_candidate_override,
    build_content_classifier_router_report,
    build_time_guard_report,
)


def _make_apply_report(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": True,
        "mode": "apply",
        "suggestion": candidate,
        "reasons": [],
        "warnings": [],
    }


def _make_decision(
    selected_codec: str,
    selected_config: str,
    preferred_audit: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    decision: Dict[str, Any] = {
        "selected": {"codec": selected_codec, "config": selected_config},
    }
    if preferred_audit is not None:
        decision["decision_trace"] = {"preferred_candidate": preferred_audit}
    return decision


def test_applied_true_when_selection_matches_candidate() -> None:
    report = _make_apply_report({"codec": "JPEG", "config": "q=85"})
    decision = _make_decision("JPEG", "q=85")

    apply_preferred_candidate_override(
        report=report,
        decision=decision,
        candidate_key="suggestion",
        label_prefix="content_policy",
    )

    assert report["applied"] is True
    assert report["reasons"] == ["content_policy_suggestion_selected"]
    assert report["warnings"] == []
    assert "decision_audit" not in report


def test_applied_false_when_admissible_but_not_competitive() -> None:
    report = _make_apply_report({"codec": "JPEG", "config": "q=85"})
    decision = _make_decision(
        "JXL",
        "d=1.0",
        preferred_audit={"admissible": True, "reason": "not_top_ranked"},
    )

    apply_preferred_candidate_override(
        report=report,
        decision=decision,
        candidate_key="suggestion",
        label_prefix="content_policy",
    )

    assert report["applied"] is False
    assert report["decision_audit"] == {
        "admissible": True,
        "reason": "not_top_ranked",
    }
    assert report["warnings"] == [
        "content_policy_suggestion_not_j_total_competitive_fallback_to_router"
    ]
    assert report["reasons"] == [
        "suggestion_admissible_but_not_competitive",
        "fallback_to_router_selection",
    ]


def test_applied_false_when_candidate_not_admissible() -> None:
    report = _make_apply_report({"codec": "JPEG", "config": "q=85"})
    decision = _make_decision(
        "JXL",
        "d=1.0",
        preferred_audit={"admissible": False, "reason": "below_min_quality"},
    )

    apply_preferred_candidate_override(
        report=report,
        decision=decision,
        candidate_key="suggestion",
        label_prefix="content_classifier",
    )

    assert report["applied"] is False
    assert report["warnings"] == [
        "content_classifier_suggestion_not_admissible_fallback_to_router"
    ]
    assert report["reasons"] == [
        "suggestion_not_admissible",
        "fallback_to_router_selection",
    ]


def test_noop_when_report_disabled() -> None:
    report = {
        "enabled": False,
        "mode": "apply",
        "suggestion": {"codec": "JPEG", "config": "q=85"},
        "reasons": [],
        "warnings": [],
    }

    apply_preferred_candidate_override(
        report=report,
        decision=_make_decision("JXL", "d=1.0"),
        candidate_key="suggestion",
        label_prefix="content_policy",
    )

    assert "applied" not in report
    assert report["reasons"] == []
    assert report["warnings"] == []


def test_noop_in_report_only_mode() -> None:
    report = {
        "enabled": True,
        "mode": "report-only",
        "suggestion": {"codec": "JPEG", "config": "q=85"},
        "reasons": [],
        "warnings": [],
    }

    apply_preferred_candidate_override(
        report=report,
        decision=_make_decision("JXL", "d=1.0"),
        candidate_key="suggestion",
        label_prefix="content_policy",
    )

    assert "applied" not in report
    assert report["reasons"] == []
    assert report["warnings"] == []


def test_noop_when_candidate_missing() -> None:
    report = {
        "enabled": True,
        "mode": "apply",
        "suggestion": None,
        "reasons": [],
        "warnings": [],
    }

    apply_preferred_candidate_override(
        report=report,
        decision=_make_decision("JXL", "d=1.0"),
        candidate_key="suggestion",
        label_prefix="content_policy",
    )

    assert "applied" not in report
    assert report["reasons"] == []
    assert report["warnings"] == []


@dataclass
class _TimePoint:
    codec: str
    config: str
    time_ms: Optional[float]


def test_build_time_guard_report_disabled_when_max_time_ms_is_none() -> None:
    report = build_time_guard_report(points=[], max_time_ms=None)

    assert report == {"enabled": False, "max_time_ms": None}


def test_build_time_guard_report_non_strict_allows_missing_time() -> None:
    points = [
        _TimePoint("JPEG", "q=60", 5.0),
        _TimePoint("JXL", "d=1.0", None),
    ]

    report = build_time_guard_report(
        points=points,
        max_time_ms=150.0,
        strict_time=False,
    )

    assert report["enabled"] is True
    assert report["strict_time"] is False
    assert report["num_candidate_points"] == 2
    assert report["num_with_time"] == 1
    assert report["num_missing_time"] == 1
    assert report["num_within_limit"] == 1
    assert report["num_over_limit"] == 0
    assert len(report["warnings"]) >= 1


def test_build_time_guard_report_strict_succeeds_when_all_have_time() -> None:
    points = [
        _TimePoint("JPEG", "q=60", 5.0),
        _TimePoint("JXL", "d=1.0", 80.0),
    ]

    report = build_time_guard_report(
        points=points,
        max_time_ms=150.0,
        strict_time=True,
    )

    assert report["enabled"] is True
    assert report["strict_time"] is True
    assert report["num_missing_time"] == 0
    assert report["num_within_limit"] == 2
    assert report["warnings"] == []


def _make_classifier_args(
    enabled: bool = False,
    mode: str = "report-only",
    config: Optional[str] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        content_classifier=enabled,
        content_classifier_mode=mode,
        content_classifier_config=config,
        content_classifier_image=None,
        content_classifier_width=None,
        content_classifier_height=None,
    )


def test_build_content_classifier_router_report_disabled_returns_stable_schema() -> None:
    report = build_content_classifier_router_report(_make_classifier_args(enabled=False))

    assert report == {
        "enabled": False,
        "mode": "report-only",
        "applied": False,
        "config": None,
        "prediction": None,
        "features": None,
        "warnings": [],
        "reasons": ["content_classifier_disabled"],
    }


def test_build_content_classifier_router_report_enabled_without_config_warns() -> None:
    report = build_content_classifier_router_report(_make_classifier_args(enabled=True))

    assert report["enabled"] is True
    assert report["applied"] is False
    assert report["prediction"] is None
    assert "content_classifier_enabled_but_missing_config" in report["warnings"]
    assert "missing_classifier_config" in report["reasons"]


def test_build_content_classifier_router_report_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="content classifier mode"):
        build_content_classifier_router_report(
            _make_classifier_args(enabled=True, mode="bogus")
        )


def test_run_profile_is_importable_from_profile_runner() -> None:
    assert callable(profile_runner.run_profile)
