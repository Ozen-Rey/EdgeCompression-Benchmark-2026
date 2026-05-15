"""Profile runner helpers.

Scaffolding module that hosts small, pure helpers used by the
``_run_profile`` orchestration in :mod:`src.router.rde_router`.

This module exists to start breaking the lazy ``pipeline -> rde_router``
import cycle: helpers here must not import :mod:`src.router.rde_router`
or :mod:`src.router.pipeline`, and they must remain side-effect free so
they can be exercised by unit tests without spinning up the full router.
"""

from __future__ import annotations

from typing import Any, Dict


def apply_preferred_candidate_override(
    *,
    report: Dict[str, Any],
    decision: Dict[str, Any],
    candidate_key: str,
    label_prefix: str,
) -> None:
    """Resolve whether a preferred candidate (suggestion/prediction) was selected.

    Shared between the content-policy and content-classifier ``apply`` paths.
    Mutates ``report`` in place: sets ``applied``, appends to ``reasons``/
    ``warnings``, and records ``decision_audit`` when the router falls back
    from the preferred candidate to its own J_RDE-ranked choice. No effect
    when the report is disabled, in report-only mode, or has no candidate.
    """
    if not (report.get("enabled") and report.get("mode") == "apply"):
        return

    candidate = report.get(candidate_key)
    if not candidate:
        return

    candidate_codec = str(candidate.get("codec"))
    candidate_config = str(candidate.get("config"))

    selected = decision.get("selected", {})
    selected_codec = str(selected.get("codec"))
    selected_config = str(selected.get("config"))

    if selected_codec == candidate_codec and selected_config == candidate_config:
        report["applied"] = True
        report["reasons"].append(f"{label_prefix}_{candidate_key}_selected")
        return

    report["applied"] = False

    preferred_audit = (
        decision.get("decision_trace", {}).get("preferred_candidate")
    )
    report["decision_audit"] = preferred_audit

    if preferred_audit and preferred_audit.get("admissible") is True:
        report["warnings"].append(
            f"{label_prefix}_{candidate_key}_not_j_total_competitive_fallback_to_router"
        )
        report["reasons"].append(
            f"{candidate_key}_admissible_but_not_competitive"
        )
    else:
        report["warnings"].append(
            f"{label_prefix}_{candidate_key}_not_admissible_fallback_to_router"
        )
        report["reasons"].append(
            f"{candidate_key}_not_admissible"
        )

    report["reasons"].append("fallback_to_router_selection")
