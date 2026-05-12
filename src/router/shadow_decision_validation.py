"""Read-only validation gate for shadow decision comparison reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

try:
    from src.router.calibration_bundle import sha256_file
    from src.router.version import ROUTER_VERSION
except ImportError:  # pragma: no cover - direct script fallback
    from calibration_bundle import sha256_file
    from version import ROUTER_VERSION


SUMMARY_FIELDS = [
    "mode",
    "router_version",
    "comparison",
    "validated_comparison_path",
    "validated_comparison_sha256",
    "candidate_calibration_bundle_manifest_sha256",
    "candidate_calibrated_csv_sha256",
    "accepted",
    "decision_count",
    "changed_decision_count",
    "decision_churn_rate",
    "mean_baseline_cost",
    "mean_candidate_cost",
    "mean_delta_cost",
    "relative_cost_improvement",
    "quality_guard_violations",
    "rate_guard_violations",
    "energy_guard_violations",
    "time_guard_violations",
    "unsafe_energy_rows",
    "insufficient_sample_groups",
    "rejection_reasons",
    "acceptance_reasons",
]


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _mean(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _relative_delta(candidate: Any, baseline: Any) -> Optional[float]:
    candidate_f = _parse_float(candidate)
    baseline_f = _parse_float(baseline)

    if candidate_f is None or baseline_f is None or baseline_f == 0:
        return None

    return (candidate_f - baseline_f) / abs(baseline_f)


def _positive_relative_regression(
    *,
    candidate: Any,
    baseline: Any,
    threshold: float,
) -> bool:
    rel = _relative_delta(candidate, baseline)
    return rel is not None and rel > float(threshold)


def _row_energy_unsafe(row: dict[str, Any]) -> bool:
    for key in (
        "energy_usable_for_total",
        "shadow_energy_usable_for_total",
        "candidate_energy_usable_for_total",
        "local_energy_usable_for_total",
    ):
        parsed = _parse_bool(row.get(key))
        if parsed is False:
            return True

    for key in ("energy_scope", "shadow_energy_scope", "candidate_energy_scope"):
        scope = str(row.get(key, "")).strip().lower()
        if scope in {"gpu", "gpu_only", "partial", "none", "disabled"}:
            return True

    provenance = row.get("energy_provenance")
    if isinstance(provenance, dict):
        parsed = _parse_bool(provenance.get("energy_usable_for_total"))
        if parsed is False:
            return True
        scope = str(provenance.get("energy_scope", "")).strip().lower()
        if scope in {"gpu", "gpu_only", "partial", "none", "disabled"}:
            return True

    notes = row.get("notes", [])
    if isinstance(notes, str):
        note_text = notes.lower()
    elif isinstance(notes, list):
        note_text = " ".join(str(item).lower() for item in notes)
    else:
        note_text = str(notes).lower()

    unsafe_markers = (
        "unsafe_energy",
        "gpu_only",
        "partial_energy",
        "energy_not_total",
        "energy_usable_for_total=false",
    )
    return any(marker in note_text for marker in unsafe_markers)


def _load_comparison(path: str | Path) -> dict[str, Any]:
    comparison_path = Path(path)
    if not comparison_path.exists():
        raise ValueError(f"Shadow decision comparison not found: {comparison_path}")

    with comparison_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Shadow decision comparison must be a JSON object.")

    if data.get("mode") != "shadow_decision_comparison_only":
        raise ValueError(
            "Unsupported shadow decision comparison mode: "
            f"{data.get('mode')!r}"
        )

    per_case = data.get("per_case", [])
    if not isinstance(per_case, list):
        raise ValueError("Shadow decision comparison per_case must be a list.")

    return data


def _cost_stats(per_case: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    baseline_costs = [_parse_float(row.get("baseline_cost")) for row in per_case]
    candidate_costs = [_parse_float(row.get("shadow_cost")) for row in per_case]
    deltas = [_parse_float(row.get("cost_delta")) for row in per_case]

    mean_baseline = _mean(baseline_costs)
    mean_candidate = _mean(candidate_costs)
    mean_delta = _mean(deltas)

    if mean_baseline is None or mean_candidate is None or mean_baseline == 0:
        relative_improvement = None
    else:
        relative_improvement = (
            mean_baseline - mean_candidate
        ) / abs(mean_baseline)

    return {
        "mean_baseline_cost": mean_baseline,
        "mean_candidate_cost": mean_candidate,
        "mean_delta_cost": mean_delta,
        "relative_cost_improvement": relative_improvement,
    }


def validate_shadow_decision_comparison(
    comparison: dict[str, Any],
    *,
    comparison_path: str | Path,
    min_decisions: int = 3,
    max_decision_churn: float = 0.50,
    max_relative_cost_regression: float = 0.0,
    allow_quality_violations: bool = False,
    allow_unsafe_energy: bool = False,
    max_rate_regression: float = 0.25,
    max_energy_regression: float = 0.25,
    max_time_regression: float = 0.25,
) -> dict[str, Any]:
    per_case_raw = comparison.get("per_case", [])
    per_case = [
        row for row in per_case_raw
        if isinstance(row, dict)
    ]

    decision_count = len(per_case)
    changed_decision_count = sum(
        1 for row in per_case
        if _parse_bool(row.get("decision_changed")) is True
    )
    decision_churn_rate = (
        changed_decision_count / decision_count
        if decision_count
        else None
    )

    cost = _cost_stats(per_case)

    quality_guard_violations = sum(
        1 for row in per_case
        if (_parse_float(row.get("quality_delta")) or 0.0) < 0.0
    )
    rate_guard_violations = sum(
        1 for row in per_case
        if _positive_relative_regression(
            candidate=row.get("shadow_rate"),
            baseline=row.get("baseline_rate"),
            threshold=max_rate_regression,
        )
    )
    energy_guard_violations = sum(
        1 for row in per_case
        if _positive_relative_regression(
            candidate=row.get("shadow_energy"),
            baseline=row.get("baseline_energy"),
            threshold=max_energy_regression,
        )
    )
    time_guard_violations = sum(
        1 for row in per_case
        if _positive_relative_regression(
            candidate=row.get("shadow_time"),
            baseline=row.get("baseline_time"),
            threshold=max_time_regression,
        )
    )
    unsafe_energy_rows = sum(
        1 for row in per_case
        if _row_energy_unsafe(row)
    )

    insufficient_sample_groups: list[str] = []
    rejection_reasons: list[str] = []
    acceptance_reasons: list[str] = []

    if decision_count < int(min_decisions):
        insufficient_sample_groups.append("global")
        rejection_reasons.append("insufficient_decisions")
    else:
        acceptance_reasons.append("min_decisions_met")

    if (
        decision_churn_rate is not None
        and decision_churn_rate > float(max_decision_churn)
    ):
        rejection_reasons.append("excessive_decision_churn")
    else:
        acceptance_reasons.append("decision_churn_within_limit")

    relative_cost_improvement = cost["relative_cost_improvement"]
    if relative_cost_improvement is None:
        rejection_reasons.append("missing_cost_evidence")
    elif relative_cost_improvement < -float(max_relative_cost_regression):
        rejection_reasons.append("candidate_cost_regression")
    else:
        acceptance_reasons.append("candidate_cost_not_regressed")

    if quality_guard_violations and not allow_quality_violations:
        rejection_reasons.append("quality_guard_violation")
    else:
        acceptance_reasons.append("quality_guard_ok")

    if rate_guard_violations:
        rejection_reasons.append("rate_guard_violation")
    else:
        acceptance_reasons.append("rate_guard_ok")

    if energy_guard_violations:
        rejection_reasons.append("energy_guard_violation")
    else:
        acceptance_reasons.append("energy_guard_ok")

    if time_guard_violations:
        rejection_reasons.append("time_guard_violation")
    else:
        acceptance_reasons.append("time_guard_ok")

    if unsafe_energy_rows and not allow_unsafe_energy:
        rejection_reasons.append("unsafe_energy_provenance")
    else:
        acceptance_reasons.append("energy_provenance_safe")

    rejection_reasons = sorted(set(rejection_reasons))
    acceptance_reasons = sorted(set(acceptance_reasons))

    return {
        "mode": "shadow_decision_validation_only",
        "router_version": ROUTER_VERSION,
        "comparison": str(comparison_path),
        "validated_comparison_path": str(comparison_path),
        "validated_comparison_sha256": sha256_file(comparison_path),
        "candidate_calibration_bundle_manifest_sha256": comparison.get(
            "candidate_calibration_bundle_manifest_sha256"
        ),
        "candidate_calibrated_csv_sha256": comparison.get(
            "candidate_calibrated_csv_sha256"
        ),
        "accepted": len(rejection_reasons) == 0,
        "thresholds": {
            "min_decisions": int(min_decisions),
            "max_decision_churn": float(max_decision_churn),
            "max_relative_cost_regression": float(max_relative_cost_regression),
            "allow_quality_violations": bool(allow_quality_violations),
            "allow_unsafe_energy": bool(allow_unsafe_energy),
            "max_rate_regression": float(max_rate_regression),
            "max_energy_regression": float(max_energy_regression),
            "max_time_regression": float(max_time_regression),
        },
        "decision_count": decision_count,
        "changed_decision_count": changed_decision_count,
        "decision_churn_rate": decision_churn_rate,
        "mean_baseline_cost": cost["mean_baseline_cost"],
        "mean_candidate_cost": cost["mean_candidate_cost"],
        "mean_delta_cost": cost["mean_delta_cost"],
        "relative_cost_improvement": relative_cost_improvement,
        "quality_guard_violations": quality_guard_violations,
        "rate_guard_violations": rate_guard_violations,
        "energy_guard_violations": energy_guard_violations,
        "time_guard_violations": time_guard_violations,
        "unsafe_energy_rows": unsafe_energy_rows,
        "insufficient_sample_groups": insufficient_sample_groups,
        "rejection_reasons": rejection_reasons,
        "acceptance_reasons": acceptance_reasons,
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _write_summary_csv(path: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                field: _csv_cell(report.get(field))
                for field in SUMMARY_FIELDS
            }
        )


def run_shadow_decision_validation(
    *,
    comparison_path: str,
    out_path: str,
    summary_out: str,
    min_decisions: int = 3,
    max_decision_churn: float = 0.50,
    max_relative_cost_regression: float = 0.0,
    allow_quality_violations: bool = False,
    allow_unsafe_energy: bool = False,
    max_rate_regression: float = 0.25,
    max_energy_regression: float = 0.25,
    max_time_regression: float = 0.25,
) -> dict[str, Any]:
    comparison = _load_comparison(comparison_path)
    report = validate_shadow_decision_comparison(
        comparison,
        comparison_path=comparison_path,
        min_decisions=min_decisions,
        max_decision_churn=max_decision_churn,
        max_relative_cost_regression=max_relative_cost_regression,
        allow_quality_violations=allow_quality_violations,
        allow_unsafe_energy=allow_unsafe_energy,
        max_rate_regression=max_rate_regression,
        max_energy_regression=max_energy_regression,
        max_time_regression=max_time_regression,
    )

    _write_json(out_path, report)
    _write_summary_csv(summary_out, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation gate for shadow decision comparisons."
    )
    parser.add_argument(
        "--comparison",
        required=True,
        help="shadow_decision_comparison.py JSON output.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON validation report.",
    )
    parser.add_argument(
        "--summary-out",
        required=True,
        help="Output one-row CSV validation summary.",
    )
    parser.add_argument("--min-decisions", type=int, default=3)
    parser.add_argument("--max-decision-churn", type=float, default=0.50)
    parser.add_argument("--max-relative-cost-regression", type=float, default=0.0)
    parser.add_argument("--allow-quality-violations", action="store_true")
    parser.add_argument("--allow-unsafe-energy", action="store_true")
    parser.add_argument("--max-rate-regression", type=float, default=0.25)
    parser.add_argument("--max-energy-regression", type=float, default=0.25)
    parser.add_argument("--max-time-regression", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_shadow_decision_validation(
        comparison_path=args.comparison,
        out_path=args.out,
        summary_out=args.summary_out,
        min_decisions=args.min_decisions,
        max_decision_churn=args.max_decision_churn,
        max_relative_cost_regression=args.max_relative_cost_regression,
        allow_quality_violations=args.allow_quality_violations,
        allow_unsafe_energy=args.allow_unsafe_energy,
        max_rate_regression=args.max_rate_regression,
        max_energy_regression=args.max_energy_regression,
        max_time_regression=args.max_time_regression,
    )


if __name__ == "__main__":
    main()
