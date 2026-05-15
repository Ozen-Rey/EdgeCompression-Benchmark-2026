"""Unit tests for ``src.router.profile_runner`` helpers.

These tests exercise ``apply_preferred_candidate_override`` in isolation
from the rest of the router pipeline so that the helper's side-effects
(mutation of ``report`` in place) are verifiable without spinning up a
full ``_run_profile`` flow.
"""

from __future__ import annotations

from typing import Any, Dict

from src.router.profile_runner import apply_preferred_candidate_override


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
