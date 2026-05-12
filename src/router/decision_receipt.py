"""Auditable router decision receipts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from src.router.calibration_bundle import sha256_file
    from src.router.version import ROUTER_VERSION
except ImportError:  # pragma: no cover - direct script fallback
    from calibration_bundle import sha256_file
    from version import ROUTER_VERSION


FLAGS_WITH_VALUE_TO_REMOVE = {
    "--out",
    "--out-dir",
    "--summary-out",
    "--feedback-out",
}

FLAGS_WITHOUT_VALUE_TO_REMOVE = {
    "--all-profiles",
    "--execute",
    "--export-topk",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path_sha256(path: Any) -> Optional[str]:
    if path is None:
        return None

    p = Path(str(path))
    if not p.exists() or not p.is_file():
        return None

    return sha256_file(p)


def _add_artifact(
    artifacts: list[dict[str, Any]],
    *,
    name: str,
    path: Any,
    sha256: str | None = None,
) -> None:
    if path is None or str(path).strip() == "":
        return

    artifact = {
        "name": name,
        "path": str(path),
        "sha256": sha256 or _path_sha256(path),
    }

    artifacts.append(artifact)


def sanitize_replay_argv(argv: Iterable[str]) -> list[str]:
    """Remove output/execution side-effect flags from router argv."""

    items = list(argv)
    out: list[str] = []
    i = 0

    while i < len(items):
        arg = str(items[i])

        if arg in FLAGS_WITH_VALUE_TO_REMOVE:
            i += 2
            continue

        if any(
            arg.startswith(f"{flag}=")
            for flag in FLAGS_WITH_VALUE_TO_REMOVE
        ):
            i += 1
            continue

        if arg in FLAGS_WITHOUT_VALUE_TO_REMOVE:
            i += 1
            continue

        out.append(arg)
        i += 1

    return out


def _selected_decision(report: dict[str, Any]) -> dict[str, Any]:
    decision = report.get("decision", {}) or {}
    selected = decision.get("selected", {}) or {}

    return {
        "profile": report.get("profile"),
        "decision_mode": decision.get("decision_mode"),
        "selected_codec": selected.get("codec"),
        "selected_config": selected.get("config"),
        "rate": selected.get("rate"),
        "quality": selected.get("quality"),
        "energy": selected.get("energy"),
        "time_ms": selected.get("time_ms"),
        "cost": selected.get("cost"),
        "quality_constraint_stat": selected.get("quality_constraint_stat"),
        "quality_constraint_value": selected.get("quality_constraint_value"),
    }


def _input_artifacts(report: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    resolved_args = report.get("resolved_args", {}) or {}
    router_config = report.get("router_config", {}) or {}
    bundle = report.get("calibration_bundle", {}) or {}
    bundle_validation = report.get("calibration_bundle_validation", {}) or {}

    _add_artifact(
        artifacts,
        name="effective_csv",
        path=report.get("csv"),
    )
    _add_artifact(
        artifacts,
        name="requested_csv",
        path=resolved_args.get("csv"),
    )
    _add_artifact(
        artifacts,
        name="router_config",
        path=router_config.get("source"),
    )
    _add_artifact(
        artifacts,
        name="normalization_file",
        path=resolved_args.get("normalization_file"),
    )
    _add_artifact(
        artifacts,
        name="quality_thresholds_file",
        path=resolved_args.get("quality_thresholds_file"),
    )
    _add_artifact(
        artifacts,
        name="codec_registry_file",
        path=resolved_args.get("codec_registry_file"),
    )
    _add_artifact(
        artifacts,
        name="calibration_bundle_manifest",
        path=bundle.get("manifest_path"),
        sha256=bundle.get("bundle_manifest_sha256")
        or bundle_validation.get("bundle_manifest_sha256"),
    )
    _add_artifact(
        artifacts,
        name="calibrated_csv",
        path=bundle.get("calibrated_csv_path"),
        sha256=bundle.get("calibrated_csv_sha256"),
    )
    _add_artifact(
        artifacts,
        name="calibration_bundle_validation",
        path=bundle_validation.get("validation_path"),
        sha256=bundle_validation.get("validation_sha256"),
    )

    return artifacts


def build_decision_receipt(report: dict[str, Any]) -> dict[str, Any]:
    """Build a replayable receipt for a single router decision report."""

    run_manifest = report.get("run_manifest", {}) or {}
    argv = (run_manifest.get("argv", {}) or {}).get("expanded", [])

    return {
        "artifact_type": "router_decision_receipt",
        "mode": "decision_receipt",
        "router_version": ROUTER_VERSION,
        "receipt_schema_version": "0.25.0",
        "created_at_utc": _utc_now(),
        "replay": {
            "argv": sanitize_replay_argv(argv),
            "side_effects_removed": sorted(
                FLAGS_WITH_VALUE_TO_REMOVE | FLAGS_WITHOUT_VALUE_TO_REMOVE
            ),
        },
        "input_artifacts": _input_artifacts(report),
        "decision": _selected_decision(report),
        "weights": report.get("weights"),
        "constraints": report.get("constraints"),
        "normalization": report.get("normalization"),
        "calibration_bundle": {
            key: value
            for key, value in (report.get("calibration_bundle", {}) or {}).items()
            if key != "manifest"
        },
        "calibration_bundle_validation": report.get(
            "calibration_bundle_validation",
            {
                "enabled": False,
            },
        ),
        "counts": {
            "num_rows_loaded_before_aggregation": report.get(
                "num_rows_loaded_before_aggregation"
            ),
            "num_candidate_points": (
                report.get("decision", {}) or {}
            ).get("num_points_total"),
            "num_admissible_points": (
                report.get("decision", {}) or {}
            ).get("num_points_admissible"),
            "num_safe_points": (
                report.get("decision", {}) or {}
            ).get("num_points_safe"),
            "num_near_points": (
                report.get("decision", {}) or {}
            ).get("num_points_near"),
        },
    }
