"""Offline validation for shadow feedback calibration proposals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from .feedback_calibration_proposal import _codec_config, _parse_bool, _parse_float
    from ..version import ROUTER_VERSION
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.router.observability.feedback_calibration_proposal import (
        _codec_config,
        _parse_bool,
        _parse_float,
    )
    from src.router.version import ROUTER_VERSION


SUMMARY_FIELDS = [
    "codec",
    "config",
    "axis",
    "num_eval_rows",
    "scale",
    "usable_proposal",
    "mean_abs_log_error_before",
    "mean_abs_log_error_after",
    "median_abs_log_error_before",
    "median_abs_log_error_after",
    "improvement_ratio",
    "improved",
    "warnings",
]


AXES = {
    "rate": {
        "predicted_key": "predicted_rate",
        "actual_key": "actual_rate",
        "scale_key": "rate_scale",
        "usable_key": "usable_for_rate",
        "require_usable_total_energy": False,
    },
    "time": {
        "predicted_key": "predicted_time_ms",
        "actual_key": "actual_time_ms",
        "scale_key": "time_scale",
        "usable_key": "usable_for_time",
        "require_usable_total_energy": False,
    },
    "energy": {
        "predicted_key": "predicted_energy",
        "actual_key": "local_energy_j",
        "scale_key": "energy_scale",
        "usable_key": "usable_for_energy",
        "require_usable_total_energy": True,
    },
}


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _median(values: Iterable[float]) -> float | None:
    items = sorted(values)
    if not items:
        return None

    midpoint = len(items) // 2
    if len(items) % 2 == 1:
        return items[midpoint]

    return (items[midpoint - 1] + items[midpoint]) / 2.0


def _load_feedback(path: str | Path) -> list[dict[str, Any]]:
    feedback_path = Path(path)
    if not feedback_path.exists():
        raise FileNotFoundError(f"Feedback CSV not found: {feedback_path}")

    if feedback_path.stat().st_size == 0:
        return []

    with feedback_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_proposal(path: str | Path) -> dict[str, Any]:
    proposal_path = Path(path)
    if not proposal_path.exists():
        raise FileNotFoundError(f"Proposal JSON not found: {proposal_path}")

    with proposal_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Proposal JSON must contain an object at the top level.")

    return data


def _proposal_index(proposal_report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}

    for item in proposal_report.get("proposals", []):
        if not isinstance(item, dict):
            continue

        codec = str(item.get("codec") or "unknown")
        config = str(item.get("config") or "unknown")
        out[(codec, config)] = item

    return out


def _valid_axis_pairs(
    rows: list[dict[str, Any]],
    *,
    predicted_key: str,
    actual_key: str,
    require_usable_total_energy: bool,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    for row in rows:
        if _parse_bool(row.get("execution_success")) is not True:
            continue

        if (
            require_usable_total_energy
            and _parse_bool(row.get("energy_usable_for_total")) is not True
        ):
            continue

        predicted = _parse_float(row.get(predicted_key))
        actual = _parse_float(row.get(actual_key))

        if predicted is None or actual is None:
            continue
        if predicted <= 0.0 or actual <= 0.0:
            continue

        pairs.append((actual, predicted))

    return pairs


def _abs_log_errors(
    pairs: list[tuple[float, float]],
    *,
    scale: float = 1.0,
) -> list[float]:
    errors: list[float] = []

    if scale <= 0.0 or not math.isfinite(scale):
        return errors

    for actual, predicted in pairs:
        denominator = predicted * scale
        if denominator <= 0.0:
            continue

        ratio = actual / denominator
        if ratio <= 0.0 or not math.isfinite(ratio):
            continue

        errors.append(abs(math.log(ratio)))

    return errors


def _validate_group_axis(
    *,
    codec: str,
    config: str,
    axis: str,
    rows: list[dict[str, Any]],
    proposal: dict[str, Any] | None,
) -> dict[str, Any]:
    axis_config = AXES[axis]
    pairs = _valid_axis_pairs(
        rows,
        predicted_key=str(axis_config["predicted_key"]),
        actual_key=str(axis_config["actual_key"]),
        require_usable_total_energy=bool(axis_config["require_usable_total_energy"]),
    )

    warnings: list[str] = []
    scale: float | None = None
    usable_proposal = False

    if proposal is None:
        warnings.append("missing_proposal")
    else:
        scale = _parse_float(proposal.get(str(axis_config["scale_key"])))
        usable_proposal = _parse_bool(proposal.get(str(axis_config["usable_key"]))) is True
        if not usable_proposal:
            warnings.append("proposal_not_usable")

    before_errors = _abs_log_errors(pairs)
    after_errors = _abs_log_errors(pairs, scale=scale) if usable_proposal and scale else []

    mean_before = _mean(before_errors)
    mean_after = _mean(after_errors) if usable_proposal else None
    median_before = _median(before_errors)
    median_after = _median(after_errors) if usable_proposal else None

    if (
        mean_before is not None
        and mean_before > 0.0
        and mean_after is not None
    ):
        improvement_ratio = 1.0 - (mean_after / mean_before)
    else:
        improvement_ratio = None

    return {
        "codec": codec,
        "config": config,
        "axis": axis,
        "num_eval_rows": len(pairs),
        "scale": scale,
        "mean_abs_log_error_before": mean_before,
        "mean_abs_log_error_after": mean_after,
        "median_abs_log_error_before": median_before,
        "median_abs_log_error_after": median_after,
        "improvement_ratio": improvement_ratio,
        "improved": improvement_ratio is not None and improvement_ratio > 0.0,
        "usable_proposal": usable_proposal,
        "warnings": warnings,
    }


def validate_feedback_proposals(
    *,
    feedback: str | Path,
    proposal: str | Path,
) -> dict[str, Any]:
    feedback_rows = _load_feedback(feedback)
    proposal_report = _load_proposal(proposal)
    proposals = _proposal_index(proposal_report)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in feedback_rows:
        grouped.setdefault(_codec_config(row), []).append(row)

    results: list[dict[str, Any]] = []
    for codec_config in sorted(grouped):
        codec, config = codec_config
        group_rows = grouped[codec_config]
        group_proposal = proposals.get(codec_config)

        for axis in ("rate", "time", "energy"):
            results.append(
                _validate_group_axis(
                    codec=codec,
                    config=config,
                    axis=axis,
                    rows=group_rows,
                    proposal=group_proposal,
                )
            )

    return {
        "version": ROUTER_VERSION,
        "mode": "offline_validation_only",
        "applied_by_router": False,
        "feedback": str(Path(feedback)),
        "proposal": str(Path(proposal)),
        "global": {
            "num_feedback_rows": len(feedback_rows),
            "num_proposal_groups": len(proposals),
            "num_feedback_groups": len(grouped),
            "num_results": len(results),
        },
        "results": results,
        "semantics": {
            "router_decision_impact": "none",
            "calibration_apply_impact": "none",
            "error_metric": "absolute_log_error",
            "energy_rule": (
                "energy validation uses only successful rows with "
                "energy_usable_for_total=true, predicted_energy>0, "
                "local_energy_j>0 and usable_for_energy=true in the proposal"
            ),
        },
    }


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def write_feedback_proposal_validation(
    *,
    feedback: str | Path,
    proposal: str | Path,
    out: str | Path,
    summary_out: str | Path,
) -> dict[str, Any]:
    report = validate_feedback_proposals(
        feedback=feedback,
        proposal=proposal,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for result in report["results"]:
            writer.writerow(
                {
                    field: _format_csv_value(result.get(field))
                    for field in SUMMARY_FIELDS
                }
            )

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline validation for shadow router feedback proposals."
    )
    parser.add_argument("--feedback", required=True, help="Input feedback CSV.")
    parser.add_argument("--proposal", required=True, help="Input proposal JSON.")
    parser.add_argument("--out", required=True, help="Output validation JSON.")
    parser.add_argument("--summary-out", required=True, help="Output summary CSV.")

    args = parser.parse_args(argv)
    report = write_feedback_proposal_validation(
        feedback=args.feedback,
        proposal=args.proposal,
        out=args.out,
        summary_out=args.summary_out,
    )

    print("\n=== R-D-E Router Feedback Proposal Validation ===")
    print(f"Feedback:  {args.feedback}")
    print(f"Proposal:  {args.proposal}")
    print(f"Rows:      {report['global']['num_feedback_rows']}")
    print(f"Groups:    {report['global']['num_feedback_groups']}")
    print(f"Mode:      {report['mode']}")
    print(f"JSON:      {args.out}")
    print(f"Summary:   {args.summary_out}")


if __name__ == "__main__":
    main()
