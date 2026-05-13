"""Promotion gate for validated shadow feedback calibration proposals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from .feedback_calibration_proposal import _parse_bool, _parse_float
    from ..version import ROUTER_VERSION
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.router.observability.feedback_calibration_proposal import (
        _parse_bool,
        _parse_float,
    )
    from src.router.version import ROUTER_VERSION


SUMMARY_FIELDS = [
    "codec",
    "config",
    "axis",
    "scale",
    "status",
    "num_eval_rows",
    "mean_abs_log_error_before",
    "mean_abs_log_error_after",
    "improvement_ratio",
    "reasons",
    "warnings",
]


AXES = ("rate", "time", "energy")


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"{label} JSON not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level.")

    return data


def _proposal_axis_scale(proposal: dict[str, Any], axis: str) -> float | None:
    return _parse_float(proposal.get(f"{axis}_scale"))


def _proposal_axis_usable(proposal: dict[str, Any], axis: str) -> bool:
    return _parse_bool(proposal.get(f"usable_for_{axis}")) is True


def _proposal_index(proposal_report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}

    for item in proposal_report.get("proposals", []):
        if not isinstance(item, dict):
            continue

        codec = str(item.get("codec") or "unknown")
        config = str(item.get("config") or "unknown")
        out[(codec, config)] = item

    return out


def _validation_index(validation_report: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}

    for item in validation_report.get("results", []):
        if not isinstance(item, dict):
            continue

        codec = str(item.get("codec") or "unknown")
        config = str(item.get("config") or "unknown")
        axis = str(item.get("axis") or "")

        if axis not in AXES:
            continue

        out[(codec, config, axis)] = item

    return out


def _candidate_keys(
    proposals: dict[tuple[str, str], dict[str, Any]],
    validations: dict[tuple[str, str, str], dict[str, Any]],
) -> list[tuple[str, str, str]]:
    keys = set(validations)

    for (codec, config), proposal in proposals.items():
        for axis in AXES:
            if (
                _proposal_axis_scale(proposal, axis) is not None
                or f"usable_for_{axis}" in proposal
            ):
                keys.add((codec, config, axis))

    return sorted(keys)


def _validation_warnings(validation: dict[str, Any] | None) -> list[str]:
    if validation is None:
        return []

    warnings = validation.get("warnings", [])
    if isinstance(warnings, list):
        return [str(item) for item in warnings]
    if warnings in (None, ""):
        return []
    return [str(warnings)]


def _evaluate_candidate(
    *,
    codec: str,
    config: str,
    axis: str,
    proposal: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    min_samples: int,
    min_improvement: float,
    max_after_error: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings = _validation_warnings(validation)

    proposal_scale = _proposal_axis_scale(proposal, axis) if proposal else None
    proposal_usable = _proposal_axis_usable(proposal, axis) if proposal else False

    if validation is None:
        reasons.append("missing_validation")
        scale = proposal_scale
        num_eval_rows = 0
        mean_before = None
        mean_after = None
        improvement_ratio = None
        validation_usable = False
    else:
        scale = _parse_float(validation.get("scale"))
        if scale is None:
            scale = proposal_scale

        num_eval_rows = int(_parse_float(validation.get("num_eval_rows")) or 0)
        mean_before = _parse_float(validation.get("mean_abs_log_error_before"))
        mean_after = _parse_float(validation.get("mean_abs_log_error_after"))
        improvement_ratio = _parse_float(validation.get("improvement_ratio"))
        validation_usable = _parse_bool(validation.get("usable_proposal")) is True

        if not validation_usable:
            reasons.append("unusable_proposal")

        if not proposal_usable:
            reasons.append("proposal_not_marked_usable")

        if num_eval_rows < min_samples:
            reasons.append("insufficient_samples")

        if _parse_bool(validation.get("improved")) is not True:
            reasons.append("not_improved")

        if improvement_ratio is None or improvement_ratio < min_improvement:
            reasons.append("improvement_below_threshold")

        if mean_after is None or mean_after > max_after_error:
            reasons.append("after_error_above_threshold")

    if scale is None or not math.isfinite(scale) or scale <= 0.0:
        reasons.append("non_positive_scale")

    if axis == "energy" and (validation is None or num_eval_rows <= 0):
        reasons.append("energy_not_total_usable")

    status = "accepted" if not reasons else "rejected"

    out = {
        "codec": codec,
        "config": config,
        "axis": axis,
        "scale": scale,
        "status": status,
        "num_eval_rows": num_eval_rows,
        "mean_abs_log_error_before": mean_before,
        "mean_abs_log_error_after": mean_after,
        "improvement_ratio": improvement_ratio,
        "warnings": warnings,
    }

    if status == "rejected":
        out["reasons"] = sorted(set(reasons))

    return out


def build_feedback_calibration_promotion(
    *,
    proposal: str | Path,
    validation: str | Path,
    min_samples: int = 3,
    min_improvement: float = 0.10,
    max_after_error: float = 0.25,
) -> dict[str, Any]:
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    if min_improvement < 0.0:
        raise ValueError("min_improvement must be >= 0")
    if max_after_error < 0.0:
        raise ValueError("max_after_error must be >= 0")

    proposal_report = _load_json(proposal, label="Proposal")
    validation_report = _load_json(validation, label="Validation")

    proposals = _proposal_index(proposal_report)
    validations = _validation_index(validation_report)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for codec, config, axis in _candidate_keys(proposals, validations):
        item = _evaluate_candidate(
            codec=codec,
            config=config,
            axis=axis,
            proposal=proposals.get((codec, config)),
            validation=validations.get((codec, config, axis)),
            min_samples=min_samples,
            min_improvement=min_improvement,
            max_after_error=max_after_error,
        )

        if item["status"] == "accepted":
            accepted.append(item)
        else:
            rejected.append(item)

    return {
        "version": ROUTER_VERSION,
        "mode": "candidate_profile_only",
        "applied_by_router": False,
        "source": {
            "proposal": str(Path(proposal)),
            "validation": str(Path(validation)),
        },
        "thresholds": {
            "min_samples": min_samples,
            "min_improvement": min_improvement,
            "max_after_error": max_after_error,
        },
        "global": {
            "num_validation_rows": len(validations),
            "num_accepted_scales": len(accepted),
            "num_rejected_scales": len(rejected),
        },
        "calibration_profile": accepted,
        "rejected": rejected,
        "semantics": {
            "router_decision_impact": "none",
            "calibration_apply_impact": "none",
            "profile_status": "candidate_only",
            "energy_rule": (
                "energy scales are accepted only when validation has usable "
                "total-energy evidence; GPU-only telemetry is not promotable"
            ),
        },
    }


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def write_feedback_calibration_promotion(
    *,
    proposal: str | Path,
    validation: str | Path,
    out: str | Path,
    summary_out: str | Path,
    min_samples: int = 3,
    min_improvement: float = 0.10,
    max_after_error: float = 0.25,
) -> dict[str, Any]:
    report = build_feedback_calibration_promotion(
        proposal=proposal,
        validation=validation,
        min_samples=min_samples,
        min_improvement=min_improvement,
        max_after_error=max_after_error,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    rows = report["calibration_profile"] + report["rejected"]
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _format_csv_value(row.get(field))
                    for field in SUMMARY_FIELDS
                }
            )

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Promote validated shadow feedback scales to a candidate profile."
    )
    parser.add_argument("--proposal", required=True, help="Input proposal JSON.")
    parser.add_argument("--validation", required=True, help="Input validation JSON.")
    parser.add_argument("--out", required=True, help="Output candidate profile JSON.")
    parser.add_argument("--summary-out", required=True, help="Output summary CSV.")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.10)
    parser.add_argument("--max-after-error", type=float, default=0.25)

    args = parser.parse_args(argv)
    report = write_feedback_calibration_promotion(
        proposal=args.proposal,
        validation=args.validation,
        out=args.out,
        summary_out=args.summary_out,
        min_samples=args.min_samples,
        min_improvement=args.min_improvement,
        max_after_error=args.max_after_error,
    )

    print("\n=== R-D-E Router Feedback Calibration Promotion ===")
    print(f"Proposal:  {args.proposal}")
    print(f"Validation:{args.validation}")
    print(f"Accepted:  {report['global']['num_accepted_scales']}")
    print(f"Rejected:  {report['global']['num_rejected_scales']}")
    print(f"Mode:      {report['mode']}")
    print(f"JSON:      {args.out}")
    print(f"Summary:   {args.summary_out}")


if __name__ == "__main__":
    main()
