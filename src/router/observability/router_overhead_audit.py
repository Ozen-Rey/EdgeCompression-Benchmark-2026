"""Read-only performance overhead audit for router modes."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from src.router.observability.decision_replay import replay_decision_receipt
    from src.router.rde_router import main as router_main
    from src.router.core.router_config import load_router_config
    from src.router.version import ROUTER_VERSION
except ImportError:  # pragma: no cover - direct script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.router.observability.decision_replay import replay_decision_receipt
    from src.router.rde_router import main as router_main
    from src.router.core.router_config import load_router_config
    from src.router.version import ROUTER_VERSION


CSV_FIELDS = [
    "mode",
    "mode_type",
    "runtime_path",
    "wall_time_ms",
    "process_cpu_time_ms",
    "peak_memory_mb",
    "input_csv_rows",
    "candidate_count",
    "selected_codec",
    "selected_config",
    "decision_match_baseline",
    "comparison_type",
    "overhead_absolute_ms",
    "overhead_relative_percent",
    "report_path",
    "notes",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _contains_flag(argv: Iterable[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _validate_read_only_request(
    *,
    config_path: str | None,
    router_args: list[str],
) -> None:
    forbidden_flags = [
        "--execute",
        "--all-profiles",
        "--feedback-out",
    ]
    for flag in forbidden_flags:
        if _contains_flag(router_args, flag):
            raise ValueError(
                f"router_overhead_audit is read-only and does not allow {flag}."
            )

    if not config_path:
        return

    config = load_router_config(config_path)

    if (
        isinstance(config.get("execution"), dict)
        and config["execution"].get("execute") is True
    ):
        raise ValueError(
            "router_overhead_audit is read-only and cannot use a config "
            "with execution.execute=true."
        )

    if (
        isinstance(config.get("selection"), dict)
        and config["selection"].get("all_profiles") is True
    ):
        raise ValueError(
            "router_overhead_audit requires single-decision runs; "
            "selection.all_profiles=true is not supported."
        )


def _count_csv_rows(path: str | Path) -> Optional[int]:
    p = Path(path)
    if not p.exists():
        return None

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _selected(report: dict[str, Any]) -> dict[str, Any]:
    selected = (report.get("decision", {}) or {}).get("selected", {}) or {}
    return {
        "codec": selected.get("codec"),
        "config": selected.get("config"),
        "rate": selected.get("rate"),
        "quality": selected.get("quality"),
        "energy": selected.get("energy"),
        "time_ms": selected.get("time_ms"),
        "cost": selected.get("cost"),
    }


def _decision_key(report: dict[str, Any]) -> tuple[Any, Any]:
    selected = _selected(report)
    return selected.get("codec"), selected.get("config")


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _run_router_once(argv: list[str], report_path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    tracemalloc_was_running = tracemalloc.is_tracing()
    if not tracemalloc_was_running:
        tracemalloc.start()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    with contextlib.redirect_stdout(io.StringIO()):
        router_main(argv + ["--out", str(report_path)])

    process_cpu_time_ms = (time.process_time() - cpu_start) * 1000.0
    wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()

    if not tracemalloc_was_running:
        tracemalloc.stop()

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    metrics = {
        "wall_time_ms": wall_time_ms,
        "process_cpu_time_ms": process_cpu_time_ms,
        "peak_memory_mb": peak_bytes / (1024.0 * 1024.0),
    }
    return report, metrics


def _base_router_args(
    *,
    csv_path: str,
    config_path: str | None,
    router_args: list[str],
) -> list[str]:
    args: list[str] = []
    if config_path:
        args.extend(["--config", config_path])
    args.extend(["--csv", csv_path])
    args.extend(router_args)
    return args


def _mode_router_args(
    *,
    base_args: list[str],
    bundle_manifest: str | None = None,
    bundle_validation: str | None = None,
) -> list[str]:
    args = list(base_args)
    if bundle_manifest:
        args.extend(["--calibration-bundle-manifest", bundle_manifest])
    if bundle_validation:
        args.extend(["--calibration-bundle-validation", bundle_validation])
    return args


def _summarize_router_mode(
    *,
    mode: str,
    mode_type: str,
    runtime_path: bool,
    reports: list[dict[str, Any]],
    measurements: list[dict[str, float]],
    report_path: Path,
    baseline_report: dict[str, Any],
    baseline_wall_time_ms: float,
    input_csv_rows: Optional[int],
) -> dict[str, Any]:
    report = reports[-1]
    selected = _selected(report)
    wall_time_ms = _mean([m["wall_time_ms"] for m in measurements])
    cpu_time_ms = _mean([m["process_cpu_time_ms"] for m in measurements])
    peak_memory_mb = max(m["peak_memory_mb"] for m in measurements)
    decision_match = _decision_key(report) == _decision_key(baseline_report)
    overhead = wall_time_ms - baseline_wall_time_ms

    if baseline_wall_time_ms > 0:
        overhead_relative = (overhead / baseline_wall_time_ms) * 100.0
    else:
        overhead_relative = None

    uses_bundle = bool(
        (report.get("calibration_bundle", {}) or {}).get("enabled", False)
    )
    comparison_type = (
        "calibrated-data"
        if uses_bundle and not decision_match
        else "overhead-only"
    )

    return {
        "mode": mode,
        "mode_type": mode_type,
        "runtime_path": runtime_path,
        "wall_time_ms": wall_time_ms,
        "process_cpu_time_ms": cpu_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "input_csv_rows": input_csv_rows,
        "candidate_count": (
            report.get("decision", {}) or {}
        ).get("num_points_total"),
        "selected_codec": selected.get("codec"),
        "selected_config": selected.get("config"),
        "selected": selected,
        "decision_match_baseline": decision_match,
        "comparison_type": comparison_type,
        "overhead_absolute_ms": overhead,
        "overhead_relative_percent": overhead_relative,
        "report_path": str(report_path),
        "notes": [],
    }


def _run_router_mode(
    *,
    mode: str,
    mode_type: str,
    runtime_path: bool,
    argv: list[str],
    out_dir: Path,
    repeat: int,
    baseline_report: dict[str, Any],
    baseline_wall_time_ms: float,
    input_csv_rows: Optional[int],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    measurements: list[dict[str, float]] = []
    last_report_path = out_dir / f"{mode}_router_report.json"

    for index in range(repeat):
        report_path = (
            last_report_path
            if index == repeat - 1
            else out_dir / f"{mode}_router_report_iter_{index + 1}.json"
        )
        report, metrics = _run_router_once(argv, report_path)
        reports.append(report)
        measurements.append(metrics)

    result = _summarize_router_mode(
        mode=mode,
        mode_type=mode_type,
        runtime_path=runtime_path,
        reports=reports,
        measurements=measurements,
        report_path=last_report_path,
        baseline_report=baseline_report,
        baseline_wall_time_ms=baseline_wall_time_ms,
        input_csv_rows=input_csv_rows,
    )
    result["iterations"] = measurements
    return result


def _run_replay_mode(
    *,
    baseline_report_path: Path,
    out_dir: Path,
    baseline_report: dict[str, Any],
    baseline_wall_time_ms: float,
    input_csv_rows: Optional[int],
) -> dict[str, Any]:
    replay_out = out_dir / "replay_validation.json"
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    replay_report = replay_decision_receipt(
        receipt_path=str(baseline_report_path),
        out_path=str(replay_out),
    )
    cpu_ms = (time.process_time() - cpu_start) * 1000.0
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    actual = replay_report.get("actual_decision") or {}
    decision_match = (
        actual.get("selected_codec"),
        actual.get("selected_config"),
    ) == _decision_key(baseline_report)
    overhead = wall_ms - baseline_wall_time_ms

    return {
        "mode": "replay",
        "mode_type": "offline-replay",
        "runtime_path": False,
        "wall_time_ms": wall_ms,
        "process_cpu_time_ms": cpu_ms,
        "peak_memory_mb": None,
        "input_csv_rows": input_csv_rows,
        "candidate_count": None,
        "selected_codec": actual.get("selected_codec"),
        "selected_config": actual.get("selected_config"),
        "selected": actual,
        "decision_match_baseline": decision_match,
        "comparison_type": "offline-replay",
        "overhead_absolute_ms": overhead,
        "overhead_relative_percent": (
            (overhead / baseline_wall_time_ms) * 100.0
            if baseline_wall_time_ms > 0
            else None
        ),
        "report_path": str(replay_out),
        "notes": ["offline_replay_not_runtime_path"],
        "replay_report": replay_report,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_cell(row.get(field))
                    for field in CSV_FIELDS
                }
            )


def run_router_overhead_audit(
    *,
    csv_path: str,
    config_path: str | None = None,
    bundle_manifest: str | None = None,
    bundle_validation: str | None = None,
    out_dir: str = "results/routing_context/overhead_audit",
    out_path: str | None = None,
    summary_out: str | None = None,
    repeat: int = 1,
    include_replay: bool = False,
    router_args: list[str] | None = None,
) -> dict[str, Any]:
    router_args = list(router_args or [])
    if bundle_validation and not bundle_manifest:
        raise ValueError(
            "--bundle-validation requires --bundle-manifest. "
            "No automatic discovery is performed."
        )

    _validate_read_only_request(
        config_path=config_path,
        router_args=router_args,
    )

    repeat = max(int(repeat), 1)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    json_out = (
        Path(out_path)
        if out_path is not None
        else out_root / "router_overhead_audit.json"
    )
    csv_out = (
        Path(summary_out)
        if summary_out is not None
        else out_root / "router_overhead_audit.csv"
    )

    input_csv_rows = _count_csv_rows(csv_path)
    base_args = _base_router_args(
        csv_path=csv_path,
        config_path=config_path,
        router_args=router_args,
    )

    baseline_report_path = out_root / "baseline_router_report.json"
    baseline_report, baseline_metrics = _run_router_once(
        base_args,
        baseline_report_path,
    )
    baseline_wall = baseline_metrics["wall_time_ms"]

    results = [
        _summarize_router_mode(
            mode="baseline",
            mode_type="runtime",
            runtime_path=True,
            reports=[baseline_report],
            measurements=[baseline_metrics],
            report_path=baseline_report_path,
            baseline_report=baseline_report,
            baseline_wall_time_ms=baseline_wall,
            input_csv_rows=input_csv_rows,
        )
    ]

    results.append(
        _run_router_mode(
            mode="receipt",
            mode_type="runtime-receipt-observation",
            runtime_path=True,
            argv=base_args,
            out_dir=out_root,
            repeat=repeat,
            baseline_report=baseline_report,
            baseline_wall_time_ms=baseline_wall,
            input_csv_rows=input_csv_rows,
        )
    )

    if bundle_manifest:
        bundle_args = _mode_router_args(
            base_args=base_args,
            bundle_manifest=bundle_manifest,
        )
        results.append(
            _run_router_mode(
                mode="bundle",
                mode_type="runtime-bundle",
                runtime_path=True,
                argv=bundle_args,
                out_dir=out_root,
                repeat=repeat,
                baseline_report=baseline_report,
                baseline_wall_time_ms=baseline_wall,
                input_csv_rows=input_csv_rows,
            )
        )

        if bundle_validation:
            validated_args = _mode_router_args(
                base_args=base_args,
                bundle_manifest=bundle_manifest,
                bundle_validation=bundle_validation,
            )
            results.append(
                _run_router_mode(
                    mode="validated_bundle",
                    mode_type="runtime-validated-bundle",
                    runtime_path=True,
                    argv=validated_args,
                    out_dir=out_root,
                    repeat=repeat,
                    baseline_report=baseline_report,
                    baseline_wall_time_ms=baseline_wall,
                    input_csv_rows=input_csv_rows,
                )
            )
            results.append(
                _run_router_mode(
                    mode="validated_bundle_receipt",
                    mode_type="runtime-validated-bundle-receipt-observation",
                    runtime_path=True,
                    argv=validated_args,
                    out_dir=out_root,
                    repeat=repeat,
                    baseline_report=baseline_report,
                    baseline_wall_time_ms=baseline_wall,
                    input_csv_rows=input_csv_rows,
                )
            )

    if include_replay:
        results.append(
            _run_replay_mode(
                baseline_report_path=baseline_report_path,
                out_dir=out_root,
                baseline_report=baseline_report,
                baseline_wall_time_ms=baseline_wall,
                input_csv_rows=input_csv_rows,
            )
        )

    report = {
        "mode": "router_overhead_audit",
        "router_version": ROUTER_VERSION,
        "created_at_utc": _utc_now(),
        "read_only": True,
        "repeat": repeat,
        "inputs": {
            "csv": str(csv_path),
            "config": config_path,
            "bundle_manifest": bundle_manifest,
            "bundle_validation": bundle_validation,
            "include_replay": include_replay,
        },
        "baseline_decision": _selected(baseline_report),
        "results": results,
    }

    _write_json(json_out, report)
    _write_csv(csv_out, results)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of router runtime overhead by mode."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bundle-manifest", default=None)
    parser.add_argument("--bundle-validation", default=None)
    parser.add_argument(
        "--out-dir",
        default="results/routing_context/overhead_audit",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--include-replay", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args, router_args = parser.parse_known_args(argv)

    run_router_overhead_audit(
        csv_path=args.csv,
        config_path=args.config,
        bundle_manifest=args.bundle_manifest,
        bundle_validation=args.bundle_validation,
        out_dir=args.out_dir,
        out_path=args.out,
        summary_out=args.summary_out,
        repeat=args.repeat,
        include_replay=args.include_replay,
        router_args=router_args,
    )


if __name__ == "__main__":
    main()
