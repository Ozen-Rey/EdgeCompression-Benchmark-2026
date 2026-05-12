"""Offline shadow comparison between baseline and calibrated bundle decisions."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    from src.router.calibration_bundle import (
        sha256_file,
        validate_calibration_bundle_manifest,
    )
    from src.router.rde_router import main as router_main
    from src.router.router_config import load_router_config
    from src.router.version import ROUTER_VERSION
except ImportError:  # pragma: no cover - direct script fallback
    from calibration_bundle import sha256_file, validate_calibration_bundle_manifest
    from rde_router import main as router_main
    from router_config import load_router_config
    from version import ROUTER_VERSION


SUMMARY_FIELDS = [
    "case_id",
    "baseline_codec",
    "baseline_config",
    "shadow_codec",
    "shadow_config",
    "decision_changed",
    "baseline_rate",
    "shadow_rate",
    "baseline_quality",
    "shadow_quality",
    "baseline_energy",
    "shadow_energy",
    "baseline_time",
    "shadow_time",
    "baseline_cost",
    "shadow_cost",
    "rate_delta",
    "quality_delta",
    "energy_delta",
    "time_delta",
    "cost_delta",
    "notes",
]


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(shadow: Any, baseline: Any) -> Optional[float]:
    shadow_f = _parse_float(shadow)
    baseline_f = _parse_float(baseline)

    if shadow_f is None or baseline_f is None:
        return None

    return shadow_f - baseline_f


def _mean(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]

    if not valid:
        return None

    return sum(valid) / len(valid)


def _selected(report: dict[str, Any]) -> dict[str, Any]:
    selected = report.get("decision", {}).get("selected", {})

    return {
        "codec": selected.get("codec"),
        "config": selected.get("config"),
        "rate": selected.get("rate"),
        "quality": selected.get("quality"),
        "energy": selected.get("energy"),
        "time": selected.get("time_ms"),
        "cost": selected.get("cost"),
    }


def _bundle_summary(
    *,
    manifest_path: str | Path,
    bundle_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "validated": bool(bundle_report.get("validated", False)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "calibrated_csv_path": bundle_report.get("calibrated_csv_path"),
        "calibrated_csv_sha256": bundle_report.get("calibrated_csv_sha256"),
        "applied_scales_count": bundle_report.get("applied_scales_count"),
        "rejected_scales_count": bundle_report.get("rejected_scales_count"),
        "energy_policy": bundle_report.get("energy_policy"),
        "source": bundle_report.get("source"),
    }


def _case_row(
    *,
    case_id: str,
    baseline_report: dict[str, Any],
    shadow_report: dict[str, Any],
) -> dict[str, Any]:
    baseline = _selected(baseline_report)
    shadow = _selected(shadow_report)

    decision_changed = (
        baseline["codec"] != shadow["codec"]
        or baseline["config"] != shadow["config"]
    )

    notes: list[str] = []
    if shadow_report.get("calibration_bundle", {}).get("validated") is True:
        notes.append("calibration_bundle_validated")

    return {
        "case_id": case_id,
        "baseline_codec": baseline.get("codec"),
        "baseline_config": baseline.get("config"),
        "shadow_codec": shadow.get("codec"),
        "shadow_config": shadow.get("config"),
        "decision_changed": decision_changed,
        "baseline_rate": baseline.get("rate"),
        "shadow_rate": shadow.get("rate"),
        "baseline_quality": baseline.get("quality"),
        "shadow_quality": shadow.get("quality"),
        "baseline_energy": baseline.get("energy"),
        "shadow_energy": shadow.get("energy"),
        "baseline_time": baseline.get("time"),
        "shadow_time": shadow.get("time"),
        "baseline_cost": baseline.get("cost"),
        "shadow_cost": shadow.get("cost"),
        "rate_delta": _delta(shadow.get("rate"), baseline.get("rate")),
        "quality_delta": _delta(shadow.get("quality"), baseline.get("quality")),
        "energy_delta": _delta(shadow.get("energy"), baseline.get("energy")),
        "time_delta": _delta(shadow.get("time"), baseline.get("time")),
        "cost_delta": _delta(shadow.get("cost"), baseline.get("cost")),
        "notes": notes,
    }


def _csv_row(case: dict[str, Any]) -> dict[str, Any]:
    row = dict(case)
    notes = row.get("notes")
    if isinstance(notes, list):
        row["notes"] = ";".join(str(item) for item in notes)
    return row


def build_shadow_decision_comparison(
    *,
    baseline_csv: str | Path,
    bundle_manifest: str | Path,
    bundle_report: dict[str, Any],
    baseline_report: dict[str, Any],
    shadow_report: dict[str, Any],
) -> dict[str, Any]:
    case_id = str(baseline_report.get("profile") or "case_1")
    per_case = [
        _case_row(
            case_id=case_id,
            baseline_report=baseline_report,
            shadow_report=shadow_report,
        )
    ]

    changed_decisions = sum(1 for row in per_case if row["decision_changed"])
    total_cases = len(per_case)
    unchanged_decisions = total_cases - changed_decisions

    return {
        "mode": "shadow_decision_comparison_only",
        "router_version": ROUTER_VERSION,
        "baseline_csv": str(baseline_csv),
        "baseline_csv_sha256": sha256_file(baseline_csv),
        "bundle_manifest": str(bundle_manifest),
        "calibrated_csv": bundle_report.get("calibrated_csv_path"),
        "candidate_calibration_bundle_manifest_path": str(bundle_manifest),
        "candidate_calibration_bundle_manifest_sha256": sha256_file(
            bundle_manifest
        ),
        "candidate_calibrated_csv_sha256": bundle_report.get(
            "calibrated_csv_sha256"
        ),
        "calibration_bundle": _bundle_summary(
            manifest_path=bundle_manifest,
            bundle_report=bundle_report,
        ),
        "total_cases": total_cases,
        "changed_decisions": changed_decisions,
        "unchanged_decisions": unchanged_decisions,
        "changed_decision_rate": (
            changed_decisions / total_cases if total_cases else None
        ),
        "aggregate_deltas": {
            "mean_rate_delta": _mean([row.get("rate_delta") for row in per_case]),
            "mean_quality_delta": _mean(
                [row.get("quality_delta") for row in per_case]
            ),
            "mean_energy_delta": _mean(
                [row.get("energy_delta") for row in per_case]
            ),
            "mean_time_delta": _mean([row.get("time_delta") for row in per_case]),
            "mean_cost_delta": _mean([row.get("cost_delta") for row in per_case]),
        },
        "per_case": per_case,
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_summary_csv(path: str | Path, per_case: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in per_case:
            writer.writerow(_csv_row(row))


def _validate_decision_only_config(config_path: str | Path) -> None:
    config = load_router_config(str(config_path))

    if (
        isinstance(config.get("execution"), dict)
        and config["execution"].get("execute") is True
    ):
        raise ValueError(
            "Shadow decision comparison is read-only and cannot use a config "
            "with execution.execute=true."
        )

    if (
        isinstance(config.get("selection"), dict)
        and config["selection"].get("all_profiles") is True
    ):
        raise ValueError(
            "Shadow decision comparison currently supports one router decision; "
            "selection.all_profiles=true is not supported."
        )


def _run_router_report(args: list[str], out_path: Path) -> dict[str, Any]:
    argv = list(args) + ["--out", str(out_path)]

    with contextlib.redirect_stdout(io.StringIO()):
        router_main(argv)

    with out_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_shadow_decision_comparison(
    *,
    baseline_csv: str,
    bundle_manifest: str,
    config_path: str,
    out_path: str,
    summary_out: str,
) -> dict[str, Any]:
    _validate_decision_only_config(config_path)
    bundle_report = validate_calibration_bundle_manifest(bundle_manifest)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base_router_args = [
        "--config",
        str(config_path),
        "--csv",
        str(baseline_csv),
    ]

    with tempfile.TemporaryDirectory(
        prefix="shadow_decision_comparison_",
        dir=str(out.parent),
    ) as tmp:
        tmp_dir = Path(tmp)
        baseline_report = _run_router_report(
            base_router_args,
            tmp_dir / "baseline_router_report.json",
        )
        shadow_report = _run_router_report(
            base_router_args
            + [
                "--calibration-bundle-manifest",
                str(bundle_manifest),
            ],
            tmp_dir / "shadow_router_report.json",
        )

    report = build_shadow_decision_comparison(
        baseline_csv=baseline_csv,
        bundle_manifest=bundle_manifest,
        bundle_report=bundle_report,
        baseline_report=baseline_report,
        shadow_report=shadow_report,
    )

    _write_json(out, report)
    _write_summary_csv(summary_out, report["per_case"])
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only shadow comparison of baseline and calibrated bundle "
            "router decisions."
        )
    )
    parser.add_argument(
        "--baseline-csv",
        required=True,
        help="Original baseline R-D-E CSV.",
    )
    parser.add_argument(
        "--bundle-manifest",
        required=True,
        help="Validated calibration bundle manifest.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Router config JSON used for both decisions.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON shadow comparison report.",
    )
    parser.add_argument(
        "--summary-out",
        required=True,
        help="Output per-case CSV summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_shadow_decision_comparison(
        baseline_csv=args.baseline_csv,
        bundle_manifest=args.bundle_manifest,
        config_path=args.config,
        out_path=args.out,
        summary_out=args.summary_out,
    )


if __name__ == "__main__":
    main()
