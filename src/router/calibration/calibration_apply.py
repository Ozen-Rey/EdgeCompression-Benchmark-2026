import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from ..codecs.codec_fingerprints import build_codec_fingerprints_for_manifest
    from ..version import ROUTER_VERSION
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.router.codecs.codec_fingerprints import build_codec_fingerprints_for_manifest
    from src.router.version import ROUTER_VERSION


ENERGY_MODES = {"auto", "require-measured-total", "benchmark-only"}
PROMOTION_AXES = {"rate", "time", "energy"}
PROMOTION_ACCEPTED_STATUSES = {"accepted", "promoted", "usable"}


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


def _get_optional_stat_value(stats: Dict[str, Any], field: str):
    value = stats.get(field)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _parse_float_or_none(value: Any):
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json_file(path: str | Path, *, label: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"{label} file non trovato: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{label} file must contain a JSON object.")

    return data


def _promotion_entries(profile: Dict[str, Any]) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []

    for section in ("calibration_profile", "promoted", "rejected"):
        section_entries = profile.get(section, [])
        if not isinstance(section_entries, list):
            continue

        for item in section_entries:
            if isinstance(item, dict):
                entries.append(item)

    return entries


def _is_energy_scale_total_usable(item: Dict[str, Any]) -> bool:
    explicit_total = item.get(
        "energy_usable_for_total",
        item.get("energy_total_usable", None),
    )
    if explicit_total is not None:
        return _parse_bool(explicit_total)

    scope = str(item.get("energy_scope", "") or "").strip().lower()
    if scope and "gpu" in scope and "cpu" not in scope and "total" not in scope:
        return False

    num_eval_rows = _parse_float_or_none(item.get("num_eval_rows"))
    return num_eval_rows is not None and num_eval_rows > 0.0


def _load_promotion_profile(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {
            "enabled": False,
            "source": None,
            "lookup": {},
            "ignored": [],
        }

    profile = _load_json_file(path, label="Promotion profile")
    lookup: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    ignored: list[Dict[str, Any]] = []

    for item in _promotion_entries(profile):
        codec = str(item.get("codec") or "unknown")
        config = str(item.get("config") or "unknown")
        axis = str(item.get("axis") or "").strip().lower()
        status = str(item.get("status") or "").strip().lower()
        scale = _parse_float_or_none(item.get("scale"))

        ignore_reason = None
        if axis not in PROMOTION_AXES:
            ignore_reason = "unsupported_axis"
        elif status not in PROMOTION_ACCEPTED_STATUSES:
            ignore_reason = "status_not_promotable"
        elif scale is None or not math.isfinite(scale) or scale <= 0.0:
            ignore_reason = "non_positive_scale"
        elif axis == "energy" and not _is_energy_scale_total_usable(item):
            ignore_reason = "energy_not_total_usable"

        if ignore_reason is not None:
            ignored.append(
                {
                    "codec": codec,
                    "config": config,
                    "axis": axis,
                    "status": status,
                    "scale": scale,
                    "reason": ignore_reason,
                }
            )
            continue

        lookup.setdefault((codec, config), {})[axis] = {
            "scale": float(scale),
            "status": status,
            "num_eval_rows": item.get("num_eval_rows"),
            "mean_abs_log_error_before": item.get("mean_abs_log_error_before"),
            "mean_abs_log_error_after": item.get("mean_abs_log_error_after"),
            "improvement_ratio": item.get("improvement_ratio"),
            "warnings": item.get("warnings", []),
            "energy_usable_for_total": (
                _is_energy_scale_total_usable(item) if axis == "energy" else None
            ),
        }

    return {
        "enabled": True,
        "source": str(Path(path)),
        "version": profile.get("version"),
        "mode": profile.get("mode"),
        "lookup": lookup,
        "ignored": ignored,
        "num_loaded_scales": sum(len(v) for v in lookup.values()),
        "num_ignored_scales": len(ignored),
    }


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


def _apply_promoted_scales(
    *,
    point: Any,
    updates: Dict[str, Any],
    promotion_scales: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    applied: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    methods: list[str] = []

    field_map = {
        "rate": "rate",
        "time": "time_ms",
        "energy": "energy",
    }

    for axis in ("rate", "time", "energy"):
        scale_info = promotion_scales.get(axis)
        if scale_info is None:
            continue

        if axis == "energy" and scale_info.get("energy_usable_for_total") is not True:
            skipped.append(
                {
                    "axis": axis,
                    "reason": "energy_not_total_usable",
                    "scale": scale_info.get("scale"),
                }
            )
            continue

        field = field_map[axis]
        current_value = updates.get(field, getattr(point, field, None))
        current_float = _parse_float_or_none(current_value)

        if current_float is None:
            skipped.append(
                {
                    "axis": axis,
                    "reason": "missing_base_value",
                    "scale": scale_info.get("scale"),
                }
            )
            continue

        scale = float(scale_info["scale"])
        updates[field] = current_float * scale
        method = f"feedback_promoted_scale_{axis}"
        methods.append(method)
        applied.append(
            {
                "axis": axis,
                "field": field,
                "scale": scale,
                "before": current_float,
                "after": updates[field],
                "method": method,
                "status": scale_info.get("status"),
                "num_eval_rows": scale_info.get("num_eval_rows"),
                "mean_abs_log_error_before": scale_info.get(
                    "mean_abs_log_error_before"
                ),
                "mean_abs_log_error_after": scale_info.get(
                    "mean_abs_log_error_after"
                ),
                "improvement_ratio": scale_info.get("improvement_ratio"),
                "warnings": scale_info.get("warnings", []),
            }
        )

    return {
        "applied": applied,
        "skipped": skipped,
        "methods": methods,
    }


def apply_local_calibration(
    points: List[Any],
    calibration_file: str,
    promotion_profile: str | Path | None = None,
):
    calibration = _load_calibration_file(calibration_file)
    energy_mode = str(calibration.get("energy_mode", "auto") or "auto")

    if energy_mode not in ENERGY_MODES:
        raise ValueError(
            "Unsupported calibration energy_mode: "
            f"{energy_mode}. Expected one of: {', '.join(sorted(ENERGY_MODES))}"
        )

    lookup = _build_calibration_lookup(calibration)
    promotion = _load_promotion_profile(promotion_profile)
    promotion_lookup = promotion["lookup"]

    calibrated_points = []
    applied = []
    skipped = []

    for point in points:
        key = (str(point.codec), str(point.config))
        stats = lookup.get(key)
        point_promotion_scales = promotion_lookup.get(key, {})

        if stats is None:
            updates: Dict[str, Any] = {}
            promotion_result = (
                _apply_promoted_scales(
                    point=point,
                    updates=updates,
                    promotion_scales=point_promotion_scales,
                )
                if promotion["enabled"] and point_promotion_scales
                else {"applied": [], "skipped": [], "methods": []}
            )

            if promotion_result["applied"]:
                new_point = _clone_point_with_updates(point, updates)
                calibrated_points.append(new_point)
                applied.append(
                    {
                        "codec": point.codec,
                        "config": point.config,
                        "local_calibration_applied": False,
                        "local_calibration_skip_reason": "no_local_calibration_for_point",
                        "promotion_profile_applied": True,
                        "feedback_promotion": promotion_result["applied"],
                        "feedback_promotion_skipped": promotion_result["skipped"],
                        "feedback_promotion_methods": promotion_result["methods"],
                        "current_method": promotion_result["methods"],
                        "rate_before": getattr(point, "rate", None),
                        "rate_after": updates.get("rate", getattr(point, "rate", None)),
                        "time_ms_before": getattr(point, "time_ms", None),
                        "time_ms_after": updates.get(
                            "time_ms",
                            getattr(point, "time_ms", None),
                        ),
                        "energy_before": getattr(point, "energy", None),
                        "energy_after": updates.get(
                            "energy",
                            getattr(point, "energy", None),
                        ),
                        "updated_fields": sorted(list(updates.keys())),
                    }
                )
            else:
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
            updates = {}
            promotion_result = (
                _apply_promoted_scales(
                    point=point,
                    updates=updates,
                    promotion_scales=point_promotion_scales,
                )
                if promotion["enabled"] and point_promotion_scales
                else {"applied": [], "skipped": [], "methods": []}
            )

            if promotion_result["applied"]:
                new_point = _clone_point_with_updates(point, updates)
                calibrated_points.append(new_point)
                applied.append(
                    {
                        "codec": point.codec,
                        "config": point.config,
                        "local_calibration_applied": False,
                        "local_calibration_skip_reason": (
                            "local_calibration_has_no_successful_runs"
                        ),
                        "promotion_profile_applied": True,
                        "feedback_promotion": promotion_result["applied"],
                        "feedback_promotion_skipped": promotion_result["skipped"],
                        "feedback_promotion_methods": promotion_result["methods"],
                        "current_method": promotion_result["methods"],
                        "rate_before": getattr(point, "rate", None),
                        "rate_after": updates.get("rate", getattr(point, "rate", None)),
                        "time_ms_before": getattr(point, "time_ms", None),
                        "time_ms_after": updates.get(
                            "time_ms",
                            getattr(point, "time_ms", None),
                        ),
                        "energy_before": getattr(point, "energy", None),
                        "energy_after": updates.get(
                            "energy",
                            getattr(point, "energy", None),
                        ),
                        "updated_fields": sorted(list(updates.keys())),
                    }
                )
            else:
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
        local_energy_j = _parse_float_or_none(
            _get_optional_stat_value(stats, "local_energy_j")
        )
        energy_is_measured = _parse_bool(stats.get("energy_is_measured", False))
        energy_scope = stats.get("energy_scope")
        energy_usable_for_total = _parse_bool(
            stats.get("energy_usable_for_total", False)
        )
        energy_backend = stats.get("energy_backend")
        energy_method = stats.get("energy_method")
        energy_quality = stats.get("energy_quality")
        energy_scaling_method = None

        if (
            local_time_ms is not None
            and benchmark_time_ms is not None
            and float(benchmark_time_ms) > 0.0
        ):
            time_scale = float(local_time_ms) / float(benchmark_time_ms)

        has_usable_local_energy = (
            local_energy_j is not None
            and energy_is_measured
            and energy_usable_for_total
        )

        if energy_mode == "require-measured-total" and not has_usable_local_energy:
            calibrated_points.append(point)
            skipped.append(
                {
                    "codec": point.codec,
                    "config": point.config,
                    "reason": "strict_energy_missing_usable_total",
                    "energy_mode": energy_mode,
                    "local_energy_j": local_energy_j,
                    "energy_is_measured": energy_is_measured,
                    "energy_scope": energy_scope,
                    "energy_usable_for_total": energy_usable_for_total,
                    "energy_backend": energy_backend,
                    "energy_method": energy_method,
                    "energy_quality": energy_quality,
                }
            )
            continue

        if energy_mode == "benchmark-only":
            if benchmark_energy is not None and time_scale is not None:
                calibrated_energy = float(benchmark_energy) * time_scale
                updates["energy"] = calibrated_energy
                energy_scaling_method = "benchmark_only_energy_mode"
            else:
                calibrated_energy = benchmark_energy
        elif has_usable_local_energy:
            calibrated_energy = local_energy_j
            updates["energy"] = calibrated_energy
            energy_scaling_method = "local_hardware_energy_total"
        elif benchmark_energy is not None and time_scale is not None:
            calibrated_energy = float(benchmark_energy) * time_scale
            updates["energy"] = calibrated_energy
            if local_energy_j is not None and energy_is_measured:
                energy_scaling_method = (
                    "benchmark_energy_scaled_by_time_ratio_"
                    "local_measurement_partial_not_comparable"
                )
            else:
                energy_scaling_method = "benchmark_energy_scaled_by_time_ratio"
        else:
            calibrated_energy = benchmark_energy

        promotion_result = (
            _apply_promoted_scales(
                point=point,
                updates=updates,
                promotion_scales=point_promotion_scales,
            )
            if promotion["enabled"] and point_promotion_scales
            else {"applied": [], "skipped": [], "methods": []}
        )

        if any(item["axis"] == "energy" for item in promotion_result["applied"]):
            if energy_scaling_method:
                energy_scaling_method = (
                    f"{energy_scaling_method}+feedback_promoted_scale_energy"
                )
            else:
                energy_scaling_method = "feedback_promoted_scale_energy"

        new_point = _clone_point_with_updates(point, updates)
        calibrated_points.append(new_point)

        calibrated_rate = updates.get("rate", benchmark_rate)
        calibrated_time_ms = updates.get("time_ms", benchmark_time_ms)
        calibrated_energy = updates.get("energy", benchmark_energy)

        applied_item = {
                "codec": point.codec,
                "config": point.config,
                "energy_mode": energy_mode,
                "success_rate": success_rate,

                "rate_before": benchmark_rate,
                "rate_after": calibrated_rate,

                "time_ms_before": benchmark_time_ms,
                "time_ms_after": calibrated_time_ms,

                "energy_before": benchmark_energy,
                "energy_after": calibrated_energy,

                "local_time_ms": local_time_ms,
                "local_bpp": local_bpp,
                "local_energy_j": local_energy_j,
                "energy_is_measured": energy_is_measured,
                "energy_scope": energy_scope,
                "energy_usable_for_total": energy_usable_for_total,
                "energy_backend": energy_backend,
                "energy_method": energy_method,
                "energy_quality": energy_quality,
                "time_scale": time_scale,

                "energy_scaling_method": energy_scaling_method,

                "updated_fields": sorted(list(updates.keys())),
            }

        if promotion["enabled"]:
            promotion_methods = list(promotion_result["methods"])
            current_methods = []
            if energy_scaling_method:
                current_methods.append(energy_scaling_method)
            current_methods.extend(
                method
                for method in promotion_methods
                if method != "feedback_promoted_scale_energy"
            )

            applied_item.update(
                {
                    "promotion_profile_applied": bool(promotion_result["applied"]),
                    "feedback_promotion": promotion_result["applied"],
                    "feedback_promotion_skipped": promotion_result["skipped"],
                    "feedback_promotion_methods": promotion_methods,
                    "current_method": current_methods,
                }
            )

        applied.append(applied_item)

    estimated = sorted(
        {
            "energy_by_time_scaling"
            for item in applied
            if (
                str(item.get("energy_scaling_method", "")).startswith(
                    "benchmark_energy_scaled_by_time_ratio"
                )
                or item.get("energy_scaling_method") == "benchmark_only_energy_mode"
            )
        }
    )

    if any(
        "feedback_promoted_scale_energy" in item.get("feedback_promotion_methods", [])
        for item in applied
    ):
        estimated.append("energy_by_feedback_promoted_scale")

    report = {
        "enabled": True,
        "source": calibration_file,
        "version": calibration.get("version"),
        "level": calibration.get("level"),
        "energy_mode": energy_mode,
        "created_at": calibration.get("created_at"),
        "measured": calibration.get("measured", []),
        "estimated": estimated,
        "not_calibrated": ["quality"],
        "energy_measurement": calibration.get("energy_measurement", {}),
        "num_points_before": len(points),
        "num_points_after": len(calibrated_points),
        "num_applied": len(applied),
        "num_skipped": len(skipped),
        "applied": applied,
        "skipped_preview": skipped[:20],
    }

    if promotion["enabled"]:
        report["promotion_profile"] = {
            "enabled": True,
            "source": promotion["source"],
            "version": promotion.get("version"),
            "mode": promotion.get("mode"),
            "num_loaded_scales": promotion.get("num_loaded_scales", 0),
            "num_ignored_scales": promotion.get("num_ignored_scales", 0),
            "ignored_preview": promotion.get("ignored", [])[:20],
            "num_applied_scales": sum(
                len(item.get("feedback_promotion", [])) for item in applied
            ),
        }

    return calibrated_points, report


def _write_points_csv(points: List[Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["codec", "config", "rate", "quality", "energy", "time_ms"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "codec": getattr(point, "codec", None),
                    "config": getattr(point, "config", None),
                    "rate": getattr(point, "rate", None),
                    "quality": getattr(point, "quality", None),
                    "energy": getattr(point, "energy", None),
                    "time_ms": getattr(point, "time_ms", None),
                }
            )


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None

    p = Path(path)
    if not p.exists():
        return None

    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _accepted_scales_from_report(report: Dict[str, Any]) -> list[Dict[str, Any]]:
    accepted: list[Dict[str, Any]] = []

    for item in report.get("applied", []):
        codec = item.get("codec")
        config = item.get("config")

        for scale in item.get("feedback_promotion", []):
            if not isinstance(scale, dict):
                continue

            out = {
                "codec": codec,
                "config": config,
                "axis": scale.get("axis"),
                "scale": scale.get("scale"),
                "method": scale.get("method"),
                "status": scale.get("status"),
                "num_eval_rows": scale.get("num_eval_rows"),
                "mean_abs_log_error_before": scale.get(
                    "mean_abs_log_error_before"
                ),
                "mean_abs_log_error_after": scale.get(
                    "mean_abs_log_error_after"
                ),
                "improvement_ratio": scale.get("improvement_ratio"),
                "warnings": scale.get("warnings", []),
            }

            if scale.get("axis") == "energy":
                out["energy_usable_for_total"] = True

            accepted.append(out)

    return accepted


def _count_unapplied_promotion_scales(report: Dict[str, Any]) -> int:
    promotion = report.get("promotion_profile", {})
    loaded = int(promotion.get("num_loaded_scales", 0) or 0)
    ignored = int(promotion.get("num_ignored_scales", 0) or 0)
    applied = int(promotion.get("num_applied_scales", 0) or 0)
    return max(ignored + loaded - applied, 0)


def build_calibration_bundle_manifest(
    *,
    report: Dict[str, Any],
    source_benchmark: str | Path,
    source_calibration: str | Path,
    promotion_profile: str | Path | None,
    output_csv: str | Path,
) -> Dict[str, Any]:
    accepted_scales = _accepted_scales_from_report(report)
    return {
        "artifact_type": "promoted_calibration_bundle",
        "router_version": ROUTER_VERSION,
        "mode": "explicit_opt_in_calibration_apply",
        "source_benchmark": str(Path(source_benchmark)),
        "source_calibration": str(Path(source_calibration)),
        "promotion_profile": (
            str(Path(promotion_profile)) if promotion_profile is not None else None
        ),
        "output_csv": str(Path(output_csv)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_scales": accepted_scales,
        "rejected_scales_count": _count_unapplied_promotion_scales(report),
        "codec_fingerprints": build_codec_fingerprints_for_manifest(
            accepted_scales,
            applied_items=report.get("applied", []),
        ),
        "energy_policy": {
            "requires_energy_usable_for_total": True,
            "gpu_only_energy_excluded": True,
        },
        "hashes": {
            "source_benchmark_sha256": _sha256_file(source_benchmark),
            "source_calibration_sha256": _sha256_file(source_calibration),
            "promotion_profile_sha256": _sha256_file(promotion_profile),
            "output_csv_sha256": _sha256_file(output_csv),
        },
        "calibration_report_summary": {
            "num_points_before": report.get("num_points_before"),
            "num_points_after": report.get("num_points_after"),
            "num_applied": report.get("num_applied"),
            "num_skipped": report.get("num_skipped"),
            "promotion_profile": report.get("promotion_profile", {}),
        },
        "semantics": {
            "router_decision_impact": "none",
            "online_learning": False,
            "requires_explicit_opt_in": True,
        },
    }


def write_calibration_bundle_manifest(
    *,
    path: str | Path,
    report: Dict[str, Any],
    source_benchmark: str | Path,
    source_calibration: str | Path,
    promotion_profile: str | Path | None,
    output_csv: str | Path,
) -> Dict[str, Any]:
    manifest = build_calibration_bundle_manifest(
        report=report,
        source_benchmark=source_benchmark,
        source_calibration=source_calibration,
        promotion_profile=promotion_profile,
        output_csv=output_csv,
    )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main(argv: list[str] | None = None) -> None:
    try:
        from ..core.rde_database import load_rde_points
    except ImportError:
        from src.router.core.rde_database import load_rde_points

    parser = argparse.ArgumentParser(
        description="Apply local calibration and optional promoted feedback scales."
    )
    parser.add_argument("--benchmark", required=True, help="Input benchmark CSV.")
    parser.add_argument("--calibration", required=True, help="Local calibration JSON.")
    parser.add_argument(
        "--promotion-profile",
        default=None,
        help="Optional promoted feedback calibration profile JSON.",
    )
    parser.add_argument("--out", required=True, help="Output calibrated CSV.")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional provenance manifest for the calibrated CSV bundle.",
    )
    parser.add_argument("--codec-col", default=None)
    parser.add_argument("--config-col", default=None)
    parser.add_argument("--rate-col", default=None)
    parser.add_argument("--quality-col", default=None)
    parser.add_argument("--energy-col", default=None)
    parser.add_argument("--time-col", default=None)

    args = parser.parse_args(argv)

    points = load_rde_points(
        csv_path=args.benchmark,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        time_col=args.time_col,
    )
    calibrated_points, report = apply_local_calibration(
        points=points,
        calibration_file=args.calibration,
        promotion_profile=args.promotion_profile,
    )
    _write_points_csv(calibrated_points, args.out)

    manifest = None
    if args.manifest_out:
        manifest = write_calibration_bundle_manifest(
            path=args.manifest_out,
            report=report,
            source_benchmark=args.benchmark,
            source_calibration=args.calibration,
            promotion_profile=args.promotion_profile,
            output_csv=args.out,
        )

    print("\n=== R-D-E Calibration Apply ===")
    print(f"Benchmark:          {args.benchmark}")
    print(f"Calibration:        {args.calibration}")
    print(f"Promotion profile:  {args.promotion_profile}")
    print(f"Output:             {args.out}")
    print(f"Applied points:     {report['num_applied']}")
    print(f"Skipped points:     {report['num_skipped']}")
    if report.get("promotion_profile", {}).get("enabled", False):
        print(
            "Promoted scales:    "
            f"{report['promotion_profile'].get('num_applied_scales')} applied"
        )
    if manifest is not None:
        print(f"Manifest:           {args.manifest_out}")


if __name__ == "__main__":
    main()
