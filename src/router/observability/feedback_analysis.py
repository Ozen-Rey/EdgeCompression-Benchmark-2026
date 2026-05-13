"""Read-only analysis for append-only router execution feedback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .feedback_logger import FEEDBACK_FIELDS
except ImportError:
    from feedback_logger import FEEDBACK_FIELDS


SUMMARY_FIELDS = [
    "num_rows",
    "num_executed",
    "num_success",
    "num_failed",
    "success_rate",
    "num_with_actual_rate",
    "num_with_actual_time",
    "num_with_usable_total_energy",
    "rate_abs_error_mean",
    "rate_rel_error_mean",
    "time_abs_error_ms_mean",
    "time_rel_error_mean",
    "energy_abs_error_mean",
    "energy_rel_error_mean",
]


BY_CODEC_FIELDS = [
    "codec",
    "num_runs",
    "num_success",
    "success_rate",
    "mean_predicted_rate",
    "mean_actual_rate",
    "mean_rate_rel_error",
    "mean_predicted_time_ms",
    "mean_actual_time_ms",
    "mean_time_rel_error",
    "num_usable_energy",
    "mean_predicted_energy",
    "mean_local_energy_j",
    "mean_energy_rel_error",
]


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if value in (None, ""):
        return None

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _rel_error(actual: float, predicted: float) -> float | None:
    if predicted == 0.0:
        return None
    return abs(actual - predicted) / abs(predicted)


def _error_pairs(
    rows: list[dict[str, Any]],
    actual_key: str,
    predicted_key: str,
    *,
    require_usable_energy: bool = False,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    for row in rows:
        if require_usable_energy and _parse_bool(row.get("energy_usable_for_total")) is not True:
            continue

        actual = _parse_float(row.get(actual_key))
        predicted = _parse_float(row.get(predicted_key))

        if actual is None or predicted is None:
            continue

        pairs.append((actual, predicted))

    return pairs


def _abs_error_mean(pairs: list[tuple[float, float]]) -> float | None:
    return _mean(abs(actual - predicted) for actual, predicted in pairs)


def _rel_error_mean(pairs: list[tuple[float, float]]) -> float | None:
    return _mean(
        rel
        for actual, predicted in pairs
        for rel in [_rel_error(actual, predicted)]
        if rel is not None
    )


def _is_error_row(row: dict[str, Any]) -> bool:
    if _parse_bool(row.get("execution_success")) is False:
        return True
    if _parse_bool(row.get("output_exists")) is False:
        return True
    if _parse_bool(row.get("output_nonempty")) is False:
        return True
    return bool(str(row.get("error") or "").strip())


def _load_feedback(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    feedback_path = Path(path)
    if not feedback_path.exists():
        raise FileNotFoundError(f"Feedback CSV not found: {feedback_path}")

    if feedback_path.stat().st_size == 0:
        return [], list(FEEDBACK_FIELDS)

    with feedback_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or FEEDBACK_FIELDS)
        return list(reader), fieldnames


def _build_global_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rate_pairs = _error_pairs(rows, "actual_rate", "predicted_rate")
    time_pairs = _error_pairs(rows, "actual_time_ms", "predicted_time_ms")
    energy_pairs = _error_pairs(
        rows,
        "local_energy_j",
        "predicted_energy",
        require_usable_energy=True,
    )

    num_executed = sum(
        1 for row in rows if _parse_bool(row.get("execution_requested")) is True
    )
    num_success = sum(
        1 for row in rows if _parse_bool(row.get("execution_success")) is True
    )
    num_failed = sum(
        1 for row in rows if _parse_bool(row.get("execution_success")) is False
    )

    return {
        "num_rows": len(rows),
        "num_executed": num_executed,
        "num_success": num_success,
        "num_failed": num_failed,
        "success_rate": (num_success / num_executed) if num_executed else 0.0,
        "num_with_actual_rate": sum(
            1 for row in rows if _parse_float(row.get("actual_rate")) is not None
        ),
        "num_with_actual_time": sum(
            1 for row in rows if _parse_float(row.get("actual_time_ms")) is not None
        ),
        "num_with_usable_total_energy": len(energy_pairs),
        "rate_abs_error_mean": _abs_error_mean(rate_pairs),
        "rate_rel_error_mean": _rel_error_mean(rate_pairs),
        "time_abs_error_ms_mean": _abs_error_mean(time_pairs),
        "time_rel_error_mean": _rel_error_mean(time_pairs),
        "energy_abs_error_mean": _abs_error_mean(energy_pairs),
        "energy_rel_error_mean": _rel_error_mean(energy_pairs),
    }


def _build_by_codec(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        codec = str(row.get("selected_codec") or "unknown")
        grouped.setdefault(codec, []).append(row)

    out: list[dict[str, Any]] = []

    for codec in sorted(grouped):
        codec_rows = grouped[codec]
        rate_pairs = _error_pairs(codec_rows, "actual_rate", "predicted_rate")
        time_pairs = _error_pairs(
            codec_rows,
            "actual_time_ms",
            "predicted_time_ms",
        )
        energy_pairs = _error_pairs(
            codec_rows,
            "local_energy_j",
            "predicted_energy",
            require_usable_energy=True,
        )
        num_success = sum(
            1
            for row in codec_rows
            if _parse_bool(row.get("execution_success")) is True
        )

        out.append(
            {
                "codec": codec,
                "num_runs": len(codec_rows),
                "num_success": num_success,
                "success_rate": num_success / len(codec_rows) if codec_rows else 0.0,
                "mean_predicted_rate": _mean(
                    value
                    for row in codec_rows
                    for value in [_parse_float(row.get("predicted_rate"))]
                    if value is not None
                ),
                "mean_actual_rate": _mean(
                    value
                    for row in codec_rows
                    for value in [_parse_float(row.get("actual_rate"))]
                    if value is not None
                ),
                "mean_rate_rel_error": _rel_error_mean(rate_pairs),
                "mean_predicted_time_ms": _mean(
                    value
                    for row in codec_rows
                    for value in [_parse_float(row.get("predicted_time_ms"))]
                    if value is not None
                ),
                "mean_actual_time_ms": _mean(
                    value
                    for row in codec_rows
                    for value in [_parse_float(row.get("actual_time_ms"))]
                    if value is not None
                ),
                "mean_time_rel_error": _rel_error_mean(time_pairs),
                "num_usable_energy": len(energy_pairs),
                "mean_predicted_energy": _mean(
                    value
                    for row in codec_rows
                    for value in [_parse_float(row.get("predicted_energy"))]
                    if value is not None
                ),
                "mean_local_energy_j": _mean(actual for actual, _ in energy_pairs),
                "mean_energy_rel_error": _rel_error_mean(energy_pairs),
            }
        )

    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze_feedback(
    *,
    feedback: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    rows, input_fields = _load_feedback(feedback)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = _build_global_summary(rows)
    by_codec = _build_by_codec(rows)
    error_rows = [row for row in rows if _is_error_row(row)]

    report = {
        "feedback": str(feedback),
        "num_rows": len(rows),
        "summary": summary,
        "by_codec": by_codec,
        "num_error_rows": len(error_rows),
        "semantics": {
            "read_only": True,
            "router_decision_impact": "none",
            "energy_error_rule": (
                "energy errors use only rows with energy_usable_for_total=true, "
                "numeric local_energy_j and numeric predicted_energy"
            ),
        },
    }

    with (out_path / "feedback_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _write_csv(
        out_path / "feedback_summary.csv",
        [summary],
        SUMMARY_FIELDS,
    )
    _write_csv(
        out_path / "feedback_by_codec.csv",
        by_codec,
        BY_CODEC_FIELDS,
    )
    _write_csv(
        out_path / "feedback_errors.csv",
        error_rows,
        input_fields or list(FEEDBACK_FIELDS),
    )

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read-only analysis for router online feedback CSV files."
    )
    parser.add_argument("--feedback", required=True, help="Input feedback CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")

    args = parser.parse_args(argv)
    report = analyze_feedback(feedback=args.feedback, out_dir=args.out_dir)

    print("\n=== R-D-E Router Feedback Analysis ===")
    print(f"Feedback: {args.feedback}")
    print(f"Rows:     {report['summary']['num_rows']}")
    print(f"Success:  {report['summary']['num_success']}")
    print(f"Failed:   {report['summary']['num_failed']}")
    print(f"Out dir:  {args.out_dir}")


if __name__ == "__main__":
    main()
