import copy
import json
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_calibration_file(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Calibration file non trovato: {p}")

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_calibration_lookup(calibration: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    summary = calibration.get("summary", {})

    for codec, config_map in summary.items():
        for config, stats in config_map.items():
            key = (str(codec), str(config))
            lookup[key] = stats

    return lookup


def _get_nested_mean(stats: Dict[str, Any], field: str):
    value = stats.get(field, {})
    if isinstance(value, dict):
        return value.get("mean")
    return None


def _clone_point_with_updates(point, updates: Dict[str, Any]):
    clean_updates = {
        k: v for k, v in updates.items()
        if v is not None and hasattr(point, k)
    }

    if not clean_updates:
        return point

    if is_dataclass(point):
        return replace(point, **clean_updates)

    new_point = copy.copy(point)
    for key, value in clean_updates.items():
        try:
            setattr(new_point, key, value)
        except Exception:
            pass

    return new_point


def apply_local_calibration(points: List[Any], calibration_file: str):
    calibration = _load_calibration_file(calibration_file)
    lookup = _build_calibration_lookup(calibration)

    calibrated_points = []
    applied = []
    skipped = []

    for point in points:
        key = (str(point.codec), str(point.config))
        stats = lookup.get(key)

        if stats is None:
            calibrated_points.append(point)
            skipped.append(
                {
                    "codec": point.codec,
                    "config": point.config,
                    "reason": "no_local_calibration_for_point",
                }
            )
            continue

        success_rate = float(stats.get("success_rate", 0.0) or 0.0)
        if success_rate <= 0.0:
            calibrated_points.append(point)
            skipped.append(
                {
                    "codec": point.codec,
                    "config": point.config,
                    "reason": "local_calibration_has_no_successful_runs",
                }
            )
            continue

        local_time_ms = _get_nested_mean(stats, "time_ms")
        local_bpp = _get_nested_mean(stats, "local_bpp")

        updates: Dict[str, Any] = {}

        if local_bpp is not None:
            updates["rate"] = float(local_bpp)

        if local_time_ms is not None:
            updates["time_ms"] = float(local_time_ms)

        calibrated_energy = None
        benchmark_rate = getattr(point, "rate", None)
        benchmark_time_ms = getattr(point, "time_ms", None)
        benchmark_energy = getattr(point, "energy", None)
        time_scale = None

        if (
            local_time_ms is not None
            and benchmark_time_ms is not None
            and float(benchmark_time_ms) > 0.0
        ):
            time_scale = float(local_time_ms) / float(benchmark_time_ms)

        if benchmark_energy is not None and time_scale is not None:
            calibrated_energy = float(benchmark_energy) * time_scale
            updates["energy"] = calibrated_energy

        new_point = _clone_point_with_updates(point, updates)
        calibrated_points.append(new_point)

        calibrated_rate = updates.get("rate", benchmark_rate)
        calibrated_time_ms = updates.get("time_ms", benchmark_time_ms)
        calibrated_energy = updates.get("energy", benchmark_energy)

        applied.append(
            {
                "codec": point.codec,
                "config": point.config,
                "success_rate": success_rate,

                "rate_before": benchmark_rate,
                "rate_after": calibrated_rate,

                "time_ms_before": benchmark_time_ms,
                "time_ms_after": calibrated_time_ms,

                "energy_before": benchmark_energy,
                "energy_after": calibrated_energy,

                "local_time_ms": local_time_ms,
                "local_bpp": local_bpp,
                "time_scale": time_scale,

                "energy_scaling_method": (
                    "benchmark_energy_scaled_by_time_ratio"
                    if calibrated_energy is not None and time_scale is not None
                    else None
                ),

                "updated_fields": sorted(list(updates.keys())),
            }
        )

    report = {
        "enabled": True,
        "source": calibration_file,
        "version": calibration.get("version"),
        "level": calibration.get("level"),
        "created_at": calibration.get("created_at"),
        "measured": calibration.get("measured", []),
        "estimated": ["energy_by_time_scaling"],
        "not_calibrated": ["quality"],
        "num_points_before": len(points),
        "num_points_after": len(calibrated_points),
        "num_applied": len(applied),
        "num_skipped": len(skipped),
        "applied": applied,
        "skipped_preview": skipped[:20],
    }

    return calibrated_points, report
