"""Offline render of a router report into a human-readable decision explanation.

This module is read-only against an existing router report JSON. It does
not re-run the router, does not read benchmark CSVs, does not touch
feedback or calibration files, does not change the ranking score, and
does not change the operational report schema.

The output is a structured ``explanation`` dict (and, optionally, a
Markdown rendering of it) that makes the decision rationale explicit:

- which candidate was selected,
- why it was selected within the *evaluated admissible pool*,
- which constraints were active,
- whether a predictor (content policy or content classifier) suggested
  a different candidate and whether the router accepted or rejected it,
- whether the decision came from the safe pool, the degraded fallback
  pool, an accepted preferred candidate, a rejected preferred candidate
  with fallback to the router's J_RDE/J_total choice, or an infeasible
  request,
- the per-term cost decomposition (R, E, D, and, when applied, the
  system penalty) and the active ranking score (J_RDE or J_total).

The wording is deliberately scoped: the explanation talks about the
*evaluated admissible pool*, never about an unconditional "best"
choice. Fields absent from the input report are reported as
``unavailable`` rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


__all__ = [
    "build_decision_explanation",
    "render_decision_explanation_markdown",
    "main",
]


_PREDICTOR_FRAMING = (
    "The predictor can suggest a codec/configuration, but the router "
    "accepts it only if it remains admissible (inside the evaluated "
    "pool that passes the quality guard and other hard constraints) "
    "and competitive under the active ranking score (J_RDE, or "
    "J_total when the system penalty is applied)."
)


_SCOPE_DISCLAIMER = (
    "Selected within the evaluated admissible pool. No claim is made "
    "about candidates outside this pool: they were either excluded by "
    "hard constraints (quality guard, max rate, max energy, max time, "
    "codec availability, capability filtering) or absent from the "
    "input R-D-E data."
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _is_enabled(block: Any) -> bool:
    return isinstance(block, dict) and bool(block.get("enabled"))


def _is_applied(block: Any) -> bool:
    return isinstance(block, dict) and bool(block.get("applied"))


def _fmt_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _extract_selected(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    decision = _safe_dict(report.get("decision"))
    selected = decision.get("selected")
    if not isinstance(selected, dict) or not selected:
        return None

    trace = _safe_dict(decision.get("decision_trace"))
    ranking_key = trace.get("ranking_key")
    system_penalty_applied = bool(trace.get("system_penalty_applied"))

    j_rde = selected.get("cost")
    j_total = selected.get("J_total")
    system_penalty = selected.get("system_penalty")

    active_ranking_value = j_total if system_penalty_applied else j_rde
    active_ranking_name = "J_total" if system_penalty_applied else "J_RDE"

    return {
        "codec": selected.get("codec"),
        "config": selected.get("config"),
        "rate": selected.get("rate"),
        "quality": selected.get("quality"),
        "energy": selected.get("energy"),
        "time_ms": selected.get("time_ms"),
        "selected_reason": trace.get("selected_reason"),
        "active_pool": trace.get("active_pool"),
        "decision_mode": trace.get("decision_mode"),
        "ranking_key": ranking_key,
        "active_ranking_name": active_ranking_name,
        "active_ranking_value": active_ranking_value,
        "J_RDE": j_rde,
        "J_total": j_total,
        "system_penalty": system_penalty,
        "system_penalty_applied": system_penalty_applied,
    }


def _build_why_selected(
    report: Dict[str, Any],
    selected: Dict[str, Any],
) -> List[str]:
    decision = _safe_dict(report.get("decision"))
    trace = _safe_dict(decision.get("decision_trace"))
    bullets: List[str] = []

    quality_guard = _safe_dict(decision.get("quality_guard"))
    stat = quality_guard.get("stat") or trace.get("quality_constraint_stat")
    floor = quality_guard.get("floor")
    if floor is not None and stat:
        bullets.append(
            f"Candidate passed the quality guard "
            f"(stat={stat}, floor={_fmt_number(floor, 4)})."
        )
    elif stat:
        bullets.append(
            f"Candidate passed the quality guard (stat={stat}; "
            "no floor configured)."
        )
    else:
        bullets.append("Candidate passed the quality guard.")

    active_pool = trace.get("active_pool")
    decision_mode = trace.get("decision_mode")
    if active_pool == "safe_pool":
        bullets.append("Candidate was inside the safe admissible pool.")
    elif active_pool == "near_pool":
        bullets.append(
            "Candidate was inside the near-floor (degraded fallback) "
            "admissible pool; the safe pool was empty under the active "
            "constraints."
        )
    else:
        bullets.append(
            "Candidate was inside the evaluated admissible pool "
            f"(decision_mode={decision_mode or 'unknown'})."
        )

    active_ranking_name = selected.get("active_ranking_name")
    active_ranking_value = selected.get("active_ranking_value")
    if active_ranking_name and active_ranking_value is not None:
        bullets.append(
            f"Candidate had the lowest active ranking score "
            f"({active_ranking_name} = "
            f"{_fmt_number(active_ranking_value, 5)}) among the "
            "admissible candidates evaluated by the router."
        )

    preferred = trace.get("preferred_candidate")
    if isinstance(preferred, dict):
        if preferred.get("selected") is True:
            bullets.append(
                "The selection matches a preferred candidate "
                "(content-policy suggestion or content-classifier "
                f"prediction) under reason "
                f"'{preferred.get('reason', 'preferred_candidate')}'."
            )
        elif preferred.get("admissible") is True and not preferred.get(
            "competitive"
        ):
            bullets.append(
                "A preferred candidate was admissible but not "
                "competitive on the active ranking score, so the "
                "router fell back to its own ranked choice."
            )
        elif preferred.get("admissible") is False:
            bullets.append(
                "A preferred candidate existed but was not admissible "
                "under the active constraints; the router fell back to "
                "its own ranked choice."
            )

    return bullets


def _build_active_constraints(report: Dict[str, Any]) -> Dict[str, Any]:
    decision = _safe_dict(report.get("decision"))
    trace = _safe_dict(decision.get("decision_trace"))
    constraints = _safe_dict(report.get("constraints"))
    quality_guard = _safe_dict(decision.get("quality_guard"))

    out: Dict[str, Any] = {}

    quality_block = {
        "stat": quality_guard.get("stat") or trace.get("quality_constraint_stat"),
        "floor": quality_guard.get("floor"),
        "near_floor": quality_guard.get("near_floor"),
        "allow_degraded_fallback": quality_guard.get("allow_degraded_fallback"),
    }
    out["quality_guard"] = quality_block

    out["bounds"] = {
        "max_rate": constraints.get("max_rate"),
        "max_energy": constraints.get("max_energy"),
        "max_time_ms": constraints.get("max_time_ms"),
    }

    normalization_audit = _safe_dict(report.get("normalization_audit"))
    if normalization_audit:
        out["normalization"] = {
            "mode": normalization_audit.get("mode"),
            "scope": normalization_audit.get("scope"),
            "comparability": normalization_audit.get("comparability"),
            "quality_metric": normalization_audit.get("quality_metric"),
        }

    codec_filtering = _safe_dict(report.get("codec_filtering"))
    if codec_filtering:
        out["codec_filtering"] = {
            "available_codecs": codec_filtering.get("available_codecs"),
            "excluded_codecs": codec_filtering.get("excluded_codecs"),
            "strict_executables": codec_filtering.get("strict_executables"),
            "capability_filtering": codec_filtering.get("capability_filtering"),
        }

    system_policy = _safe_dict(report.get("system_policy"))
    if _is_enabled(system_policy):
        out["system_policy"] = {
            "enabled": True,
            "mode": system_policy.get("mode"),
            "applied": bool(system_policy.get("applied")),
            "exclude_neural": system_policy.get("exclude_neural"),
        }

    system_penalty = _safe_dict(report.get("system_penalty"))
    if _is_enabled(system_penalty):
        out["system_penalty"] = {
            "enabled": True,
            "mode": system_penalty.get("mode"),
            "applied": bool(system_penalty.get("applied")),
            "lambda_sys": system_penalty.get("lambda_sys"),
        }

    calibration = _safe_dict(report.get("calibration"))
    if _is_enabled(calibration):
        out["calibration"] = {
            "enabled": True,
            "energy_mode": calibration.get("energy_mode"),
        }

    calibration_bundle = _safe_dict(report.get("calibration_bundle"))
    if _is_enabled(calibration_bundle):
        out["calibration_bundle"] = {
            "enabled": True,
            "mode": calibration_bundle.get("mode"),
            "applied": bool(calibration_bundle.get("applied")),
        }

    energy_provenance = _safe_dict(report.get("energy_provenance"))
    if energy_provenance:
        out["energy_provenance"] = {
            "current_method": energy_provenance.get("current_method"),
            "energy_is_measured": energy_provenance.get("energy_is_measured"),
            "energy_quality": energy_provenance.get("energy_quality"),
            "energy_scope": energy_provenance.get("energy_scope"),
            "energy_usable_for_total": energy_provenance.get(
                "energy_usable_for_total"
            ),
        }

    time_guard = _safe_dict(report.get("time_guard"))
    if _is_enabled(time_guard):
        out["time_guard"] = {
            "enabled": True,
            "max_time_ms": time_guard.get("max_time_ms"),
            "num_within_limit": time_guard.get("num_within_limit"),
            "num_over_limit": time_guard.get("num_over_limit"),
            "num_missing_time": time_guard.get("num_missing_time"),
        }

    return out


def _build_predictor_role(report: Dict[str, Any]) -> Dict[str, Any]:
    content_policy = _safe_dict(report.get("content_policy"))
    content_classifier = _safe_dict(report.get("content_classifier"))

    decision = _safe_dict(report.get("decision"))
    trace = _safe_dict(decision.get("decision_trace"))
    preferred_audit = trace.get("preferred_candidate")
    if not isinstance(preferred_audit, dict):
        preferred_audit = None

    role: Dict[str, Any] = {
        "framing": _PREDICTOR_FRAMING,
        "content_policy_enabled": bool(content_policy.get("enabled")),
        "content_classifier_enabled": bool(content_classifier.get("enabled")),
        "preferred_candidate_audit": preferred_audit,
    }

    if content_policy.get("enabled"):
        role["content_policy"] = {
            "mode": content_policy.get("mode"),
            "applied": bool(content_policy.get("applied")),
            "suggestion": content_policy.get("suggestion"),
            "reasons": list(content_policy.get("reasons") or []),
            "warnings": list(content_policy.get("warnings") or []),
        }

    if content_classifier.get("enabled"):
        role["content_classifier"] = {
            "mode": content_classifier.get("mode"),
            "applied": bool(content_classifier.get("applied")),
            "prediction": content_classifier.get("prediction"),
            "reasons": list(content_classifier.get("reasons") or []),
            "warnings": list(content_classifier.get("warnings") or []),
        }

    return role


def _build_fallback_safety(
    report: Dict[str, Any],
    selected: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    decision = _safe_dict(report.get("decision"))
    trace = _safe_dict(decision.get("decision_trace"))
    preferred = trace.get("preferred_candidate")
    decision_mode = trace.get("decision_mode")

    if selected is None:
        return {
            "category": "infeasible_request",
            "explanation": (
                "The report does not carry a selected candidate; the "
                "request was infeasible under the active constraints."
            ),
            "decision_mode": decision_mode,
            "preferred_candidate_audit": (
                preferred if isinstance(preferred, dict) else None
            ),
        }

    preferred_dict = preferred if isinstance(preferred, dict) else None

    if preferred_dict and preferred_dict.get("selected") is True:
        category = "preferred_candidate_accepted"
        explanation = (
            "The router accepted the preferred candidate suggested by "
            "the predictor: it was admissible and competitive on the "
            "active ranking score."
        )
    elif preferred_dict and (
        preferred_dict.get("admissible") is False
        or preferred_dict.get("competitive") is False
    ):
        category = "preferred_candidate_rejected"
        explanation = (
            "The router rejected the preferred candidate suggested by "
            "the predictor and fell back to its own ranked choice: the "
            "candidate was either not admissible under the active "
            "constraints or not competitive on the active ranking score."
        )
    elif decision_mode == "degraded_fallback":
        category = "degraded_fallback_selection"
        explanation = (
            "The safe admissible pool was empty under the active "
            "constraints; the router selected from the near-floor "
            "(degraded fallback) admissible pool, which the operator "
            "must treat as an opt-in lower-safety choice."
        )
    elif decision_mode == "safe":
        category = "safe_pool_selection"
        explanation = (
            "The router selected from the safe admissible pool, which "
            "satisfies the quality guard and all other active hard "
            "constraints."
        )
    else:
        category = "unspecified_selection"
        explanation = (
            "The router selected a candidate; the decision_mode field "
            "is not present in the report, so the safety category cannot "
            "be inferred."
        )

    return {
        "category": category,
        "explanation": explanation,
        "decision_mode": decision_mode,
        "preferred_candidate_audit": preferred_dict,
    }


def _build_cost_decomposition(
    report: Dict[str, Any],
    selected: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if selected is None:
        return None

    decision = _safe_dict(report.get("decision"))
    selected_raw = _safe_dict(decision.get("selected"))
    decomposition = _safe_dict(selected_raw.get("cost_decomposition"))
    if not decomposition:
        return None

    system_penalty = selected.get("system_penalty")
    system_penalty_applied = bool(selected.get("system_penalty_applied"))
    system_penalty_block = _safe_dict(report.get("system_penalty"))
    lambda_sys = (
        system_penalty_block.get("lambda_sys")
        if system_penalty_block.get("enabled")
        else None
    )

    out = {
        "w_R": decomposition.get("w_R"),
        "w_E": decomposition.get("w_E"),
        "w_D": decomposition.get("w_D"),
        "norm_rate": decomposition.get("norm_rate"),
        "norm_energy": decomposition.get("norm_energy"),
        "norm_distortion": decomposition.get("norm_distortion"),
        "term_R": decomposition.get("term_R"),
        "term_E": decomposition.get("term_E"),
        "term_D": decomposition.get("term_D"),
        "J_RDE": selected.get("J_RDE"),
        "system_penalty": system_penalty,
        "system_penalty_applied": system_penalty_applied,
        "lambda_sys": lambda_sys,
        "J_total": selected.get("J_total"),
        "active_ranking": selected.get("active_ranking_name"),
        "active_ranking_value": selected.get("active_ranking_value"),
    }
    return out


def build_decision_explanation(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build a structured explanation dict from a router report.

    The function never re-runs the router and never reads files. Fields
    not present in the input report are reported as ``unavailable``
    rather than inferred.
    """
    if not isinstance(report, dict):
        raise TypeError(
            "build_decision_explanation requires a dict (parsed router "
            f"report JSON); got {type(report).__name__}"
        )

    selected = _extract_selected(report)

    explanation: Dict[str, Any] = {
        "router_version": report.get("router_version"),
        "domain": report.get("domain"),
        "profile": report.get("profile"),
        "weight_source": report.get("weight_source"),
        "weights": _safe_dict(report.get("weights")) or None,
        "selected": selected,
        "why_selected": (
            _build_why_selected(report, selected) if selected else []
        ),
        "active_constraints": _build_active_constraints(report),
        "predictor_role": _build_predictor_role(report),
        "fallback_safety": _build_fallback_safety(report, selected),
        "cost_decomposition": _build_cost_decomposition(report, selected),
        "framing_notes": [_SCOPE_DISCLAIMER, _PREDICTOR_FRAMING],
    }
    return explanation


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------


def _format_key_value(key: str, value: Any) -> str:
    if value is None:
        formatted = "unavailable"
    elif isinstance(value, bool):
        formatted = "true" if value else "false"
    elif isinstance(value, (int, float)):
        formatted = _fmt_number(value, 5)
    elif isinstance(value, dict):
        formatted = ", ".join(
            f"{k}={_format_key_value_inline(v)}" for k, v in value.items()
        ) or "{}"
    elif isinstance(value, list):
        formatted = (
            ", ".join(_format_key_value_inline(v) for v in value)
            if value
            else "[]"
        )
    else:
        formatted = str(value)
    return f"- **{key}**: {formatted}"


def _format_key_value_inline(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _fmt_number(value, 5)
    if isinstance(value, dict):
        inner = ", ".join(
            f"{k}={_format_key_value_inline(v)}" for k, v in value.items()
        )
        return "{" + inner + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_key_value_inline(v) for v in value) + "]"
    return str(value)


def render_decision_explanation_markdown(
    explanation: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# R-D-E Router Decision Explanation")
    lines.append("")

    header_items = [
        ("router_version", explanation.get("router_version")),
        ("domain", explanation.get("domain")),
        ("profile", explanation.get("profile")),
        ("weight_source", explanation.get("weight_source")),
    ]
    for k, v in header_items:
        lines.append(_format_key_value(k, v))
    lines.append("")

    selected = explanation.get("selected")
    lines.append("## Selected candidate")
    lines.append("")
    if selected is None:
        lines.append(
            "_The report does not carry a selected candidate. The "
            "request was infeasible under the active constraints._"
        )
        lines.append("")
    else:
        for k in [
            "codec",
            "config",
            "rate",
            "quality",
            "energy",
            "time_ms",
            "selected_reason",
            "active_pool",
            "decision_mode",
            "ranking_key",
            "active_ranking_name",
            "active_ranking_value",
            "J_RDE",
            "J_total",
            "system_penalty",
            "system_penalty_applied",
        ]:
            lines.append(_format_key_value(k, selected.get(k)))
        lines.append("")

    lines.append("## Why this candidate was selected")
    lines.append("")
    why = explanation.get("why_selected") or []
    if why:
        for bullet in why:
            lines.append(f"- {bullet}")
    else:
        lines.append("_unavailable_")
    lines.append("")

    lines.append("## Active constraints")
    lines.append("")
    constraints = explanation.get("active_constraints") or {}
    if not constraints:
        lines.append("_unavailable_")
    else:
        for section_name, block in constraints.items():
            lines.append(f"### {section_name}")
            if isinstance(block, dict) and block:
                for k, v in block.items():
                    lines.append(_format_key_value(k, v))
            else:
                lines.append("- _unavailable_")
            lines.append("")

    lines.append("## Predictor role")
    lines.append("")
    predictor = explanation.get("predictor_role") or {}
    lines.append(predictor.get("framing", _PREDICTOR_FRAMING))
    lines.append("")
    lines.append(
        _format_key_value(
            "content_policy_enabled",
            predictor.get("content_policy_enabled"),
        )
    )
    lines.append(
        _format_key_value(
            "content_classifier_enabled",
            predictor.get("content_classifier_enabled"),
        )
    )

    if "content_policy" in predictor:
        lines.append("")
        lines.append("### content_policy")
        for k, v in predictor["content_policy"].items():
            lines.append(_format_key_value(k, v))

    if "content_classifier" in predictor:
        lines.append("")
        lines.append("### content_classifier")
        for k, v in predictor["content_classifier"].items():
            lines.append(_format_key_value(k, v))

    preferred_audit = predictor.get("preferred_candidate_audit")
    if isinstance(preferred_audit, dict):
        lines.append("")
        lines.append("### preferred_candidate_audit")
        for k, v in preferred_audit.items():
            lines.append(_format_key_value(k, v))
    lines.append("")

    lines.append("## Fallback / safety")
    lines.append("")
    fallback = explanation.get("fallback_safety") or {}
    lines.append(
        _format_key_value("category", fallback.get("category"))
    )
    lines.append(
        _format_key_value("decision_mode", fallback.get("decision_mode"))
    )
    lines.append("")
    if fallback.get("explanation"):
        lines.append(fallback["explanation"])
        lines.append("")

    decomposition = explanation.get("cost_decomposition")
    lines.append("## Cost decomposition")
    lines.append("")
    if decomposition is None:
        lines.append("_unavailable_")
        lines.append("")
    else:
        for k in [
            "w_R",
            "w_E",
            "w_D",
            "norm_rate",
            "norm_energy",
            "norm_distortion",
            "term_R",
            "term_E",
            "term_D",
            "J_RDE",
            "system_penalty",
            "system_penalty_applied",
            "lambda_sys",
            "J_total",
            "active_ranking",
            "active_ranking_value",
        ]:
            lines.append(_format_key_value(k, decomposition.get(k)))
        lines.append("")

    lines.append("## Framing notes")
    lines.append("")
    for note in explanation.get("framing_notes") or []:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_report(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.router.observability.decision_explanation",
        description=(
            "Offline, read-only render of a router report JSON into a "
            "human-readable decision explanation (Markdown and/or "
            "structured JSON). Does not re-run the router, does not "
            "change the decision, and does not touch feedback or "
            "calibration files."
        ),
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to an existing router report JSON.",
    )
    parser.add_argument(
        "--out-md",
        default=None,
        help="Optional path to write the Markdown explanation.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the structured explanation JSON.",
    )
    parser.add_argument(
        "--print",
        dest="print_stdout",
        action="store_true",
        help=(
            "Print the Markdown explanation to stdout. Default behavior "
            "when neither --out-md nor --out-json is provided."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    report = _load_report(args.report)
    explanation = build_decision_explanation(report)
    markdown = render_decision_explanation_markdown(explanation)

    wrote_anything = False

    if args.out_md:
        path = Path(args.out_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        wrote_anything = True

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(explanation, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        wrote_anything = True

    if args.print_stdout or not wrote_anything:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
