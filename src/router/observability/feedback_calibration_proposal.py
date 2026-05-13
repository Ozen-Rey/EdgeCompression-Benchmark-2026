"""Shadow calibration proposals derived from online router feedback."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from ..version import ROUTER_VERSION
except ImportError:
    from version import ROUTER_VERSION


SUMMARY_FIELDS = [
    "codec",
    "config",
    "num_samples",
    "num_success",
    "success_rate",
    "rate_scale",
    "rate_confidence",
    "usable_for_rate",
    "time_scale",
    "time_confidence",
    "usable_for_time",
    "energy_scale",
    "energy_confidence",
    "usable_for_energy",
    "warnings",
]


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(out):
        return None
    return out


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


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def _confidence(num_valid: int, min_samples: int) -> str:
    if num_valid < min_samples:
        return "insufficient"
    if num_valid < 5:
        return "low"
    if num_valid < 20:
        return "medium"
    return "high"


def _codec_config(row: dict[str, Any]) -> tuple[str, str]:
    codec = _clean_text(row.get("codec"))
    selected_codec = _clean_text(row.get("selected_codec"))
    config = _clean_text(row.get("config"))
    selected_config = _clean_text(row.get("selected_config"))

    if selected_config is not None:
        return codec or selected_codec or "unknown", selected_config

    if config is not None:
        return codec or selected_codec or "unknown", config

    return codec or selected_codec or "unknown", "unknown"


def _scale_values(
    rows: list[dict[str, Any]],
    *,
    predicted_key: str,
    actual_key: str,
    require_usable_total_energy: bool = False,
) -> tuple[float | None, int]:
    ratios: list[float] = []

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

        ratios.append(actual / predicted)

    return _mean(ratios), len(ratios)


def _usable_scale(
    *,
    scale: float | None,
    confidence: str,
    warning_name: str,
    warnings: list[str],
) -> bool:
    if confidence == "insufficient" or scale is None:
        return False

    if not math.isfinite(scale) or scale < 0.1 or scale > 10.0:
        warnings.append(warning_name)
        return False

    return True


def _load_feedback(path: str | Path) -> list[dict[str, Any]]:
    feedback_path = Path(path)
    if not feedback_path.exists():
        raise FileNotFoundError(f"Feedback CSV not found: {feedback_path}")

    if feedback_path.stat().st_size == 0:
        return []

    with feedback_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _build_proposal_for_group(
    *,
    codec: str,
    config: str,
    rows: list[dict[str, Any]],
    min_samples: int,
) -> dict[str, Any]:
    rate_scale, rate_valid = _scale_values(
        rows,
        predicted_key="predicted_rate",
        actual_key="actual_rate",
    )
    time_scale, time_valid = _scale_values(
        rows,
        predicted_key="predicted_time_ms",
        actual_key="actual_time_ms",
    )
    energy_scale, energy_valid = _scale_values(
        rows,
        predicted_key="predicted_energy",
        actual_key="local_energy_j",
        require_usable_total_energy=True,
    )

    rate_confidence = _confidence(rate_valid, min_samples)
    time_confidence = _confidence(time_valid, min_samples)
    energy_confidence = _confidence(energy_valid, min_samples)
    warnings: list[str] = []

    num_success = sum(
        1 for row in rows if _parse_bool(row.get("execution_success")) is True
    )

    proposal = {
        "codec": codec,
        "config": config,
        "num_samples": len(rows),
        "num_success": num_success,
        "success_rate": num_success / len(rows) if rows else 0.0,
        "rate_scale": rate_scale,
        "time_scale": time_scale,
        "energy_scale": energy_scale,
        "rate_confidence": rate_confidence,
        "time_confidence": time_confidence,
        "energy_confidence": energy_confidence,
        "usable_for_rate": _usable_scale(
            scale=rate_scale,
            confidence=rate_confidence,
            warning_name="rate_scale_out_of_bounds",
            warnings=warnings,
        ),
        "usable_for_time": _usable_scale(
            scale=time_scale,
            confidence=time_confidence,
            warning_name="time_scale_out_of_bounds",
            warnings=warnings,
        ),
        "usable_for_energy": _usable_scale(
            scale=energy_scale,
            confidence=energy_confidence,
            warning_name="energy_scale_out_of_bounds",
            warnings=warnings,
        ),
        "warnings": warnings,
    }

    return proposal


def build_feedback_calibration_proposal(
    *,
    feedback: str | Path,
    min_samples: int = 3,
) -> dict[str, Any]:
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    rows = _load_feedback(feedback)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(_codec_config(row), []).append(row)

    proposals = [
        _build_proposal_for_group(
            codec=codec,
            config=config,
            rows=group_rows,
            min_samples=min_samples,
        )
        for (codec, config), group_rows in sorted(grouped.items())
    ]

    global_report = {
        "num_rows": len(rows),
        "num_groups": len(grouped),
        "num_success": sum(
            1 for row in rows if _parse_bool(row.get("execution_success")) is True
        ),
        "num_rate_valid": sum(
            1
            for proposal in proposals
            if proposal["rate_confidence"] != "insufficient"
        ),
        "num_time_valid": sum(
            1
            for proposal in proposals
            if proposal["time_confidence"] != "insufficient"
        ),
        "num_energy_valid": sum(
            1
            for proposal in proposals
            if proposal["energy_confidence"] != "insufficient"
        ),
    }

    return {
        "version": ROUTER_VERSION,
        "source": Path(feedback).name,
        "source_path": str(Path(feedback)),
        "mode": "shadow_proposal_only",
        "applied_by_router": False,
        "min_samples": min_samples,
        "global": global_report,
        "proposals": proposals,
        "semantics": {
            "router_decision_impact": "none",
            "calibration_apply_impact": "none",
            "energy_rule": (
                "energy_scale uses only successful rows with "
                "energy_usable_for_total=true, numeric local_energy_j and "
                "numeric predicted_energy"
            ),
        },
    }


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def write_feedback_calibration_proposal(
    *,
    feedback: str | Path,
    out: str | Path,
    summary_out: str | Path,
    min_samples: int = 3,
) -> dict[str, Any]:
    report = build_feedback_calibration_proposal(
        feedback=feedback,
        min_samples=min_samples,
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
        for proposal in report["proposals"]:
            writer.writerow(
                {
                    field: _format_csv_value(proposal.get(field))
                    for field in SUMMARY_FIELDS
                }
            )

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build shadow calibration proposals from router feedback."
    )
    parser.add_argument("--feedback", required=True, help="Input feedback CSV.")
    parser.add_argument("--out", required=True, help="Output proposal JSON.")
    parser.add_argument("--summary-out", required=True, help="Output summary CSV.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum valid samples required before a scale is usable.",
    )

    args = parser.parse_args(argv)
    report = write_feedback_calibration_proposal(
        feedback=args.feedback,
        out=args.out,
        summary_out=args.summary_out,
        min_samples=args.min_samples,
    )

    print("\n=== R-D-E Router Feedback Calibration Proposal ===")
    print(f"Feedback:  {args.feedback}")
    print(f"Rows:      {report['global']['num_rows']}")
    print(f"Groups:    {report['global']['num_groups']}")
    print(f"Mode:      {report['mode']}")
    print(f"JSON:      {args.out}")
    print(f"Summary:   {args.summary_out}")


if __name__ == "__main__":
    main()
