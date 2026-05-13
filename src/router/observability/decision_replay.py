"""Offline replay validation for router decision receipts."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from src.router.calibration.calibration_bundle import sha256_file
    from src.router.rde_router import main as router_main
    from src.router.version import ROUTER_VERSION
except ImportError:  # pragma: no cover - direct script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.router.calibration.calibration_bundle import sha256_file
    from src.router.rde_router import main as router_main
    from src.router.version import ROUTER_VERSION


DECISION_FIELDS = [
    "profile",
    "decision_mode",
    "selected_codec",
    "selected_config",
    "rate",
    "quality",
    "energy",
    "time_ms",
    "cost",
    "quality_constraint_stat",
    "quality_constraint_value",
]


def _load_json_object(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Decision receipt not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Decision receipt must be a JSON object.")

    return data


def load_decision_receipt(path: str | Path) -> dict[str, Any]:
    data = _load_json_object(path)
    receipt = data.get("decision_receipt", data)

    if not isinstance(receipt, dict):
        raise ValueError("decision_receipt must be a JSON object.")

    if receipt.get("artifact_type") != "router_decision_receipt":
        raise ValueError(
            "Unsupported decision receipt artifact_type: "
            f"{receipt.get('artifact_type')!r}"
        )

    return receipt


def _verify_input_artifacts(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for artifact in receipt.get("input_artifacts", []) or []:
        if not isinstance(artifact, dict):
            continue

        path = artifact.get("path")
        expected = artifact.get("sha256")
        p = Path(str(path)) if path else None

        if p is None or not p.exists() or not p.is_file():
            actual = None
            match = False
        else:
            actual = sha256_file(p)
            match = expected is None or actual == expected

        results.append(
            {
                "name": artifact.get("name"),
                "path": path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": match,
            }
        )

    return results


def _selected_from_report(report: dict[str, Any]) -> dict[str, Any]:
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


def _numbers_equal(a: Any, b: Any, *, tolerance: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return a == b


def _compare_decisions(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    for field in DECISION_FIELDS:
        expected_value = expected.get(field)
        actual_value = actual.get(field)

        if not _numbers_equal(
            expected_value,
            actual_value,
            tolerance=tolerance,
        ):
            mismatches.append(
                {
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )

    return mismatches


def replay_decision_receipt(
    *,
    receipt_path: str,
    out_path: str,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    receipt = load_decision_receipt(receipt_path)
    artifact_checks = _verify_input_artifacts(receipt)
    input_hashes_match = all(item["match"] for item in artifact_checks)

    replay_success = False
    actual_decision: dict[str, Any] | None = None
    replay_error: str | None = None
    mismatches: list[dict[str, Any]] = []

    if input_hashes_match:
        argv = list((receipt.get("replay", {}) or {}).get("argv", []))

        if not argv:
            replay_error = "receipt_missing_replay_argv"
        else:
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            try:
                with tempfile.TemporaryDirectory(
                    prefix="decision_replay_",
                    dir=str(out.parent),
                ) as tmp:
                    replay_report_path = Path(tmp) / "router_replay_report.json"
                    with contextlib.redirect_stdout(io.StringIO()):
                        router_main(argv + ["--out", str(replay_report_path)])

                    replay_report = _load_json_object(replay_report_path)

                actual_decision = _selected_from_report(replay_report)
                mismatches = _compare_decisions(
                    receipt.get("decision", {}) or {},
                    actual_decision,
                    tolerance=tolerance,
                )
                replay_success = True
            except Exception as exc:
                replay_error = str(exc)

    report = {
        "mode": "decision_replay_validation",
        "router_version": ROUTER_VERSION,
        "receipt_path": str(receipt_path),
        "receipt_router_version": receipt.get("router_version"),
        "input_hashes_match": input_hashes_match,
        "artifact_checks": artifact_checks,
        "replay_success": replay_success,
        "decision_reproduced": replay_success and not mismatches,
        "tolerance": float(tolerance),
        "expected_decision": receipt.get("decision"),
        "actual_decision": actual_decision,
        "mismatches": mismatches,
        "error": replay_error,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay and validate a router decision receipt."
    )
    parser.add_argument(
        "--receipt",
        required=True,
        help="Router report containing decision_receipt, or receipt JSON.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output replay validation JSON.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Numeric comparison tolerance.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    replay_decision_receipt(
        receipt_path=args.receipt,
        out_path=args.out,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
