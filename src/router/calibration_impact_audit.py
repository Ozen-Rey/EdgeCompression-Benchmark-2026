"""Read-only audit of calibration bundle impact on router decisions."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from src.router.calibration_bundle import validate_calibration_bundle_manifest
    from src.router.rde_router import main as router_main
    from src.router.router_config import load_router_config
    from src.router.version import ROUTER_VERSION
except ImportError:  # pragma: no cover - direct script fallback
    from calibration_bundle import validate_calibration_bundle_manifest
    from rde_router import main as router_main
    from router_config import load_router_config
    from version import ROUTER_VERSION


SUMMARY_FIELDS = [
    "mode",
    "router_version",
    "baseline_selected_codec",
    "baseline_selected_config",
    "baseline_j_rde",
    "baseline_rate",
    "baseline_quality",
    "baseline_energy",
    "baseline_time",
    "calibrated_selected_codec",
    "calibrated_selected_config",
    "calibrated_j_rde",
    "calibrated_rate",
    "calibrated_quality",
    "calibrated_energy",
    "calibrated_time",
    "decision_changed",
    "codec_changed",
    "config_changed",
    "delta_j_rde",
    "delta_rate",
    "delta_quality",
    "delta_energy",
    "delta_time",
    "bundle_validated",
    "manifest_path",
    "calibrated_csv_path",
    "calibrated_csv_sha256",
    "applied_scales_count",
    "rejected_scales_count",
    "energy_policy",
]


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(after: Any, before: Any) -> Optional[float]:
    after_f = _parse_float(after)
    before_f = _parse_float(before)

    if after_f is None or before_f is None:
        return None

    return after_f - before_f


def _decision_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = report.get("decision", {}).get("selected", {})

    return {
        "selected_codec": selected.get("codec"),
        "selected_config": selected.get("config"),
        "j_rde": selected.get("cost"),
        "rate": selected.get("rate"),
        "quality": selected.get("quality"),
        "energy": selected.get("energy"),
        "time": selected.get("time_ms"),
    }


def _bundle_summary(
    manifest_path: str | Path,
    bundle_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "validated": bool(bundle_report.get("validated", False)),
        "manifest_path": str(manifest_path),
        "calibrated_csv_path": bundle_report.get("calibrated_csv_path"),
        "calibrated_csv_sha256": bundle_report.get("calibrated_csv_sha256"),
        "applied_scales_count": bundle_report.get("applied_scales_count"),
        "rejected_scales_count": bundle_report.get("rejected_scales_count"),
        "energy_policy": bundle_report.get("energy_policy"),
        "source": bundle_report.get("source"),
    }


def build_impact_report(
    *,
    baseline_report: dict[str, Any],
    calibrated_report: dict[str, Any],
    calibration_bundle: dict[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    baseline = _decision_summary(baseline_report)
    calibrated = _decision_summary(calibrated_report)

    codec_changed = baseline["selected_codec"] != calibrated["selected_codec"]
    config_changed = baseline["selected_config"] != calibrated["selected_config"]

    return {
        "mode": "calibration_impact_audit",
        "router_version": ROUTER_VERSION,
        "baseline": baseline,
        "calibrated": calibrated,
        "impact": {
            "decision_changed": codec_changed or config_changed,
            "codec_changed": codec_changed,
            "config_changed": config_changed,
            "delta_j_rde": _delta(calibrated["j_rde"], baseline["j_rde"]),
            "delta_energy": _delta(calibrated["energy"], baseline["energy"]),
            "delta_time": _delta(calibrated["time"], baseline["time"]),
            "delta_rate": _delta(calibrated["rate"], baseline["rate"]),
            "delta_quality": _delta(calibrated["quality"], baseline["quality"]),
        },
        "calibration_bundle": _bundle_summary(
            manifest_path=manifest_path,
            bundle_report=calibration_bundle,
        ),
    }


def _flatten_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    baseline = report["baseline"]
    calibrated = report["calibrated"]
    impact = report["impact"]
    bundle = report["calibration_bundle"]

    return {
        "mode": report["mode"],
        "router_version": report["router_version"],
        "baseline_selected_codec": baseline.get("selected_codec"),
        "baseline_selected_config": baseline.get("selected_config"),
        "baseline_j_rde": baseline.get("j_rde"),
        "baseline_rate": baseline.get("rate"),
        "baseline_quality": baseline.get("quality"),
        "baseline_energy": baseline.get("energy"),
        "baseline_time": baseline.get("time"),
        "calibrated_selected_codec": calibrated.get("selected_codec"),
        "calibrated_selected_config": calibrated.get("selected_config"),
        "calibrated_j_rde": calibrated.get("j_rde"),
        "calibrated_rate": calibrated.get("rate"),
        "calibrated_quality": calibrated.get("quality"),
        "calibrated_energy": calibrated.get("energy"),
        "calibrated_time": calibrated.get("time"),
        "decision_changed": impact.get("decision_changed"),
        "codec_changed": impact.get("codec_changed"),
        "config_changed": impact.get("config_changed"),
        "delta_j_rde": impact.get("delta_j_rde"),
        "delta_rate": impact.get("delta_rate"),
        "delta_quality": impact.get("delta_quality"),
        "delta_energy": impact.get("delta_energy"),
        "delta_time": impact.get("delta_time"),
        "bundle_validated": bundle.get("validated"),
        "manifest_path": bundle.get("manifest_path"),
        "calibrated_csv_path": bundle.get("calibrated_csv_path"),
        "calibrated_csv_sha256": bundle.get("calibrated_csv_sha256"),
        "applied_scales_count": bundle.get("applied_scales_count"),
        "rejected_scales_count": bundle.get("rejected_scales_count"),
        "energy_policy": bundle.get("energy_policy"),
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_summary_csv(path: str | Path, report: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(_flatten_summary_row(report))


def _contains_flag(argv: Iterable[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _validate_decision_only_request(
    *,
    config_path: str | None,
    router_args: list[str],
) -> None:
    if _contains_flag(router_args, "--execute"):
        raise ValueError("Calibration impact audit is read-only and cannot use --execute.")

    if _contains_flag(router_args, "--all-profiles"):
        raise ValueError(
            "Calibration impact audit requires a single router decision; "
            "--all-profiles is not supported."
        )

    if _contains_flag(router_args, "--calibration-bundle-manifest"):
        raise ValueError(
            "Pass the bundle through --calibration-bundle-manifest on the audit "
            "command, not as a forwarded router argument."
        )

    if config_path is None:
        return

    config = load_router_config(config_path)

    if (
        isinstance(config.get("execution"), dict)
        and config["execution"].get("execute") is True
    ):
        raise ValueError(
            "Calibration impact audit is read-only and cannot use a config "
            "with execution.execute=true."
        )

    if (
        isinstance(config.get("selection"), dict)
        and config["selection"].get("all_profiles") is True
    ):
        raise ValueError(
            "Calibration impact audit requires a single router decision; "
            "selection.all_profiles=true is not supported."
        )


def _router_args(
    *,
    config_path: str | None,
    csv_path: str,
    router_args: list[str],
) -> list[str]:
    args: list[str] = []

    if config_path:
        args.extend(["--config", config_path])

    args.extend(["--csv", csv_path])
    args.extend(router_args)
    return args


def _run_router_report(router_args: list[str], out_path: Path) -> dict[str, Any]:
    argv = list(router_args) + ["--out", str(out_path)]

    with contextlib.redirect_stdout(io.StringIO()):
        router_main(argv)

    with out_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_calibration_impact_audit(
    *,
    csv_path: str,
    calibration_bundle_manifest: str,
    out_path: str,
    summary_out: str,
    config_path: str | None = None,
    router_args: list[str] | None = None,
) -> dict[str, Any]:
    router_args = list(router_args or [])
    _validate_decision_only_request(
        config_path=config_path,
        router_args=router_args,
    )

    bundle_report = validate_calibration_bundle_manifest(
        calibration_bundle_manifest
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base_args = _router_args(
        config_path=config_path,
        csv_path=csv_path,
        router_args=router_args,
    )

    with tempfile.TemporaryDirectory(
        prefix="calibration_impact_audit_",
        dir=str(out.parent),
    ) as tmp:
        tmp_dir = Path(tmp)
        baseline_report = _run_router_report(
            base_args,
            tmp_dir / "baseline_router_report.json",
        )
        calibrated_report = _run_router_report(
            base_args
            + [
                "--calibration-bundle-manifest",
                calibration_bundle_manifest,
            ],
            tmp_dir / "calibrated_router_report.json",
        )

    report = build_impact_report(
        baseline_report=baseline_report,
        calibrated_report=calibrated_report,
        calibration_bundle=bundle_report,
        manifest_path=calibration_bundle_manifest,
    )

    _write_json(out, report)
    _write_summary_csv(summary_out, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit comparing baseline router decisions with "
            "explicit calibration-bundle decisions."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional router config JSON. Forwarded to rde_router.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Baseline benchmark CSV used by the router.",
    )
    parser.add_argument(
        "--calibration-bundle-manifest",
        required=True,
        help="Validated v0.18+ calibration bundle manifest.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON audit report.",
    )
    parser.add_argument(
        "--summary-out",
        required=True,
        help="Output one-row CSV summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args, router_args = parser.parse_known_args(argv)

    run_calibration_impact_audit(
        csv_path=args.csv,
        calibration_bundle_manifest=args.calibration_bundle_manifest,
        out_path=args.out,
        summary_out=args.summary_out,
        config_path=args.config,
        router_args=router_args,
    )


if __name__ == "__main__":
    main()
