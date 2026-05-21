"""Real measured audio/video R-D-E router validation artifacts.

This module is intentionally read-only with respect to benchmark sources: it
turns existing measured rows into router-facing CSV/JSON artifacts and computes
offline oracle/baseline/regret summaries through ``DomainSpec``. It does not
run codecs, synthesize missing measurements, or change runtime ranking logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.router.core.dataset_manifest import validate_dataset_manifest
from src.router.core.domain_spec import (
    DomainSpec,
    domain_spec_to_dict,
    resolve_domain_spec,
    validate_rde_dataframe_against_domain_spec,
)
from src.router.core.profiles import available_profiles, get_profile
from src.router.core.rde_database import RDEPoint, select_best_rde
from src.router.rde_router import main as router_main


DEFAULT_OUT_DIR = Path("results/routing_context/audio_video_real_validation")
DEFAULT_PROFILES = ["balanced", "bandwidth-limited", "energy-limited", "quality-first"]
CODEC_TERMS = {
    "Opus",
    "EnCodec",
    "DAC",
    "SNAC",
    "WavTokenizer",
    "x264",
    "x265",
    "SVT",
    "VVenC",
    "DCVC",
    "VMAF",
    "ViSQOL",
    "FAD",
}
CLASSICAL_CODECS = {
    "opus",
    "x264",
    "x265",
    "svtav1",
    "svt-av1",
    "vvenc",
}
NEURAL_CODECS = {
    "encodec",
    "dac",
    "snac",
    "wavtokenizer",
    "dcvc_dc",
    "dcvc_fm",
    "dcvc_rt",
    "dcvc_rt_cuda",
}


@dataclass(frozen=True)
class BuildResult:
    rows: list[dict[str, Any]]
    report: dict[str, Any]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        value_f = float(text)
    except ValueError:
        return None
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def _format_float(value: float | None, digits: int = 10) -> str:
    if value is None:
        return ""
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _canonical_param(value: Any) -> str:
    text = str(value).strip()
    value_f = _maybe_float(text)
    if value_f is None:
        return text
    return f"{value_f:.8f}".rstrip("0").rstrip(".")


def _item_stem(value: Any) -> str:
    return Path(str(value).strip()).stem


def _source(path: Path) -> str:
    return path.as_posix()


def _rows_from_csv_for_domain_spec(csv_path: Path) -> list[dict[str, str]]:
    return _read_csv(csv_path)


def _count_rows_and_columns(path: Path) -> tuple[int | None, list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = sum(1 for _ in reader)
            return rows, list(reader.fieldnames or [])
    except Exception:
        return None, []


def _detect_domain(path: Path, columns: Iterable[str], sample: str) -> str | None:
    haystack = " ".join([path.as_posix(), *columns, sample]).lower()
    audio_hits = ["audio", "visqol", "fad", "opus", "encodec", "wavtokenizer", "snac"]
    video_hits = ["video", "vmaf", "psnr_y", "x264", "x265", "svt", "vvenc", "dcvc"]
    audio_score = sum(term in haystack for term in audio_hits)
    video_score = sum(term in haystack for term in video_hits)
    if audio_score == 0 and video_score == 0:
        return None
    if audio_score > video_score:
        return "audio"
    if video_score > audio_score:
        return "video"
    return "audio/video"


def _detect_columns(columns: Iterable[str], terms: Iterable[str]) -> list[str]:
    out: list[str] = []
    lower_terms = [term.lower() for term in terms]
    for column in columns:
        lower = column.lower()
        if any(term in lower for term in lower_terms):
            out.append(column)
    return out


def _detect_codecs(rows: list[dict[str, str]]) -> list[str]:
    values: set[str] = set()
    for row in rows:
        for key in ("codec", "codec_label", "method", "model"):
            value = row.get(key)
            if value:
                values.add(str(value).strip())
    return sorted(values)


def _detect_items(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("dataset", "file", "item_id", "sequence", "seq"):
        values = {str(row.get(key, "")).strip() for row in rows if row.get(key)}
        if values:
            summary[key] = {
                "count": len(values),
                "examples": sorted(values)[:10],
            }
    return summary


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _candidate_files(root: Path, exclude_dir: Path | None = None) -> list[Path]:
    roots = [root / "data", root / "results", root / "docs"]
    files: set[Path] = set()
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if exclude_dir is not None and _is_relative_to(path, exclude_dir):
                continue
            if path.suffix.lower() not in {".csv", ".json"}:
                continue
            name = path.name.lower()
            if any(
                term in name
                for term in ("audio", "video", "vmaf", "visqol", "fad", "energy", "rde")
            ):
                files.add(path)
                continue
            try:
                sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:65536]
            except Exception:
                sample = ""
            if any(term.lower() in sample.lower() for term in CODEC_TERMS):
                files.add(path)
    return sorted(files)


def build_source_inventory(root: Path, out_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in _candidate_files(root, exclude_dir=out_dir):
        sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:65536]
        rows_count: int | None
        columns: list[str]
        sample_rows: list[dict[str, str]] = []
        if path.suffix.lower() == ".csv":
            rows_count, columns = _count_rows_and_columns(path)
            try:
                sample_rows = _read_csv(path)[:200]
            except Exception:
                sample_rows = []
        else:
            rows_count, columns = None, []

        domain = _detect_domain(path, columns, sample)
        if domain is None:
            continue

        metrics = _detect_columns(
            columns,
            ["visqol", "fad", "vmaf", "psnr", "pesq", "stoi", "sdr", "ssim"],
        )
        energy_columns = _detect_columns(columns, ["energy", "joule", "j_per_s", "j/s", "kj"])
        rate_columns = _detect_columns(columns, ["bitrate", "kbps", "mbps", "actual_kbps", "actual_mbps"])
        time_columns = _detect_columns(columns, ["time", "rtf", "ms", "_s"])

        usable, reason = _inventory_usability(path, columns)
        entries.append(
            {
                "path": _source(path),
                "domain": domain,
                "rows": rows_count,
                "columns": columns,
                "codecs_detected": _detect_codecs(sample_rows),
                "datasets_items_detected": _detect_items(sample_rows),
                "metrics_detected": metrics,
                "energy_columns": energy_columns,
                "rate_columns": rate_columns,
                "time_columns": time_columns,
                "usable_for_router": usable,
                "reason_if_false": "" if usable else reason,
            }
        )

    payload = {
        "valid": True,
        "scope": "audio/video real validation source inventory",
        "root": str(root),
        "num_candidates": len(entries),
        "candidates": entries,
    }
    _write_json(out_dir / "source_inventory.json", payload)
    return payload


def _inventory_usability(path: Path, columns: list[str]) -> tuple[bool, str]:
    cols = {column.lower() for column in columns}
    if path.as_posix().endswith("results/audio/visqol_audio_mode_benchmark.csv"):
        return True, ""
    if path.as_posix().endswith("results/audio/full_pipeline_energy_benchmark.csv"):
        return True, ""
    if path.as_posix().endswith("results/audio/audio_summary_full.csv"):
        return True, ""
    if path.as_posix().endswith("results/video/video_LDP_reference_preset_paper_ready.csv"):
        return True, ""
    has_codec_config = "codec" in cols and ("param" in cols or "preset" in cols)
    has_audio_quality = "visqol" in cols or "visqol_audio" in cols
    has_video_quality = "vmaf" in cols or "vmaf_mean" in cols
    has_rate = any(c in cols for c in ("bitrate_kbps", "actual_kbps", "actual_mbps", "kbps"))
    has_energy = any("energy" in c or c == "j_per_s" for c in cols)
    if has_codec_config and (has_audio_quality or has_video_quality) and has_rate and has_energy:
        return True, ""
    missing: list[str] = []
    if not has_codec_config:
        missing.append("codec/config")
    if not (has_audio_quality or has_video_quality):
        missing.append("per-row primary quality")
    if not has_rate:
        missing.append("rate")
    if not has_energy:
        missing.append("energy")
    return False, "not router-ready alone: missing " + ", ".join(missing)


def build_audio_router_ready(root: Path, out_dir: Path) -> BuildResult:
    quality_path = root / "results/audio/visqol_audio_mode_benchmark.csv"
    energy_path = root / "results/audio/full_pipeline_energy_benchmark.csv"
    summary_path = root / "results/audio/audio_summary_full.csv"

    quality_rows = _read_csv(quality_path)
    energy_rows = _read_csv(energy_path)
    summary_rows = _read_csv(summary_path)

    rate_by_config: dict[tuple[str, str], float] = {}
    for row in summary_rows:
        rate = _maybe_float(row.get("actual_kbps"))
        if rate is not None:
            rate_by_config[(row["codec"].strip(), _canonical_param(row["param"]))] = rate

    energy_by_item: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in energy_rows:
        key = (
            row.get("dataset", "").strip(),
            _item_stem(row.get("file", "")),
            row.get("codec", "").strip(),
            _canonical_param(row.get("param", "")),
        )
        energy_by_item[key] = row

    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for row in quality_rows:
        dataset = row.get("dataset", "").strip()
        item_id = _item_stem(row.get("file", ""))
        codec = row.get("codec", "").strip()
        param = _canonical_param(row.get("param", ""))
        visqol = _maybe_float(row.get("visqol_audio") or row.get("visqol"))
        rate = rate_by_config.get((codec, param))
        energy_row = energy_by_item.get((dataset, item_id, codec, param))
        if visqol is None:
            dropped["missing_visqol"] += 1
            continue
        if rate is None:
            dropped["missing_measured_rate"] += 1
            continue
        if energy_row is None:
            dropped["missing_measured_energy_row"] += 1
            continue
        total_energy = _maybe_float(energy_row.get("energy_total_net_j"))
        duration_s = _maybe_float(energy_row.get("duration_s"))
        if total_energy is None:
            dropped["missing_energy_total_net_j"] += 1
            continue
        if duration_s is None or duration_s <= 0:
            dropped["missing_or_invalid_duration_s"] += 1
            continue
        time_ms = _maybe_float(energy_row.get("time_total_ms"))
        energy_j_per_second = total_energy / duration_s
        rows.append(
            {
                "dataset": dataset,
                "item_id": item_id,
                "codec": codec,
                "param": param,
                "bitrate_kbps": _format_float(rate),
                "visqol": _format_float(visqol),
                "energy_j_per_second": _format_float(energy_j_per_second),
                "time_ms": _format_float(time_ms),
                "measurement_provenance": (
                    f"quality={_source(quality_path)}:visqol_audio;"
                    f"rate={_source(summary_path)}:actual_kbps_by_codec_param;"
                    f"energy={_source(energy_path)}:energy_total_net_j/duration_s;"
                    f"time={_source(energy_path)}:time_total_ms"
                ),
                "duration_s": _format_float(duration_s),
                "energy_total_net_j": _format_float(total_energy),
                "is_neural": energy_row.get("is_neural", ""),
            }
        )

    out_csv = out_dir / "audio_rde_router_ready.csv"
    fieldnames = [
        "dataset",
        "item_id",
        "codec",
        "param",
        "bitrate_kbps",
        "visqol",
        "energy_j_per_second",
        "time_ms",
        "measurement_provenance",
        "duration_s",
        "energy_total_net_j",
        "is_neural",
    ]
    _write_csv(out_csv, rows, fieldnames)

    report = {
        "valid": bool(rows),
        "domain": "audio",
        "domain_spec": "audio_visqol",
        "output_csv": _source(out_csv),
        "source_files": [_source(quality_path), _source(summary_path), _source(energy_path)],
        "num_source_quality_rows": len(quality_rows),
        "num_output_rows": len(rows),
        "dropped_rows": dict(sorted(dropped.items())),
        "columns": fieldnames,
        "codecs": sorted({str(row["codec"]) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "quality_metric": "visqol",
        "rate_mapping": "bitrate_kbps = audio_summary_full.actual_kbps by codec/param",
        "energy_mapping": "energy_j_per_second = full_pipeline_energy_benchmark.energy_total_net_j / duration_s per item/config",
        "time_mapping": "time_ms = full_pipeline_energy_benchmark.time_total_ms when present",
        "warnings": [
            "ViSQOL is used as the per-item router quality metric.",
            "FAD is not used as router quality because available FAD rows are aggregate codec/config statistics, not per-item reliable router quality rows.",
        ],
    }
    _write_json(out_dir / "audio_rde_build_report.json", report)
    return BuildResult(rows=rows, report=report)


def build_video_router_ready(root: Path, out_dir: Path) -> BuildResult:
    source_path = root / "results/video/video_LDP_reference_preset_paper_ready.csv"
    source_rows = _read_csv(source_path)
    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()

    for row in source_rows:
        rate_mbps = _maybe_float(row.get("actual_mbps"))
        vmaf = _maybe_float(row.get("vmaf_mean"))
        energy_kj = _maybe_float(row.get("energy_total_kj"))
        encode_s = _maybe_float(row.get("time_encode_s"))
        decode_s = _maybe_float(row.get("time_decode_s"))
        if rate_mbps is None:
            dropped["missing_actual_mbps"] += 1
            continue
        if vmaf is None:
            dropped["missing_vmaf_mean"] += 1
            continue
        if energy_kj is None:
            dropped["missing_energy_total_kj"] += 1
            continue
        measured_time_s = sum(value for value in (encode_s, decode_s) if value is not None)
        time_ms = measured_time_s * 1000.0 if measured_time_s > 0 else None
        rows.append(
            {
                "dataset": "uvg_ldp_reference_preset",
                "sequence": row.get("seq", "").strip(),
                "codec": row.get("codec", "").strip(),
                "param": row.get("param", "").strip(),
                "bitrate_kbps": _format_float(rate_mbps * 1000.0),
                "vmaf": _format_float(vmaf),
                "energy_kj_per_sequence": _format_float(energy_kj),
                "time_ms": _format_float(time_ms),
                "measurement_provenance": (
                    f"quality_rate_energy={_source(source_path)}:"
                    "actual_mbps,vmaf_mean,energy_total_kj;"
                    "time=(time_encode_s+time_decode_s where measured)"
                ),
                "family": row.get("family", ""),
                "profile": row.get("profile", ""),
                "preset": row.get("preset", ""),
                "n_frames": row.get("n_frames", ""),
                "is_neural": row.get("is_neural", ""),
                "time_encode_s": row.get("time_encode_s", ""),
                "time_decode_s": row.get("time_decode_s", ""),
                "energy_total_j": row.get("energy_total_j", ""),
                "psnr_y": row.get("psnr_y", ""),
            }
        )

    out_csv = out_dir / "video_rde_router_ready.csv"
    fieldnames = [
        "dataset",
        "sequence",
        "codec",
        "param",
        "bitrate_kbps",
        "vmaf",
        "energy_kj_per_sequence",
        "time_ms",
        "measurement_provenance",
        "family",
        "profile",
        "preset",
        "n_frames",
        "is_neural",
        "time_encode_s",
        "time_decode_s",
        "energy_total_j",
        "psnr_y",
    ]
    _write_csv(out_csv, rows, fieldnames)

    report = {
        "valid": bool(rows),
        "domain": "video",
        "domain_spec": "video_vmaf",
        "output_csv": _source(out_csv),
        "source_files": [_source(source_path)],
        "num_source_rows": len(source_rows),
        "num_output_rows": len(rows),
        "dropped_rows": dict(sorted(dropped.items())),
        "columns": fieldnames,
        "codecs": sorted({str(row["codec"]) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "quality_metric": "vmaf",
        "rate_mapping": "bitrate_kbps = actual_mbps * 1000",
        "energy_mapping": "energy_kj_per_sequence = energy_total_kj",
        "time_mapping": "time_ms = (time_encode_s + measured time_decode_s) * 1000; missing decode time is not invented",
        "warnings": [],
    }
    _write_json(out_dir / "video_rde_build_report.json", report)
    return BuildResult(rows=rows, report=report)


def build_audio_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row["item_id"])
        by_item.setdefault(
            item_id,
            {
                "item_id": item_id,
                "path": item_id,
                "duration_s": _maybe_float(row.get("duration_s")),
                "sample_rate": None,
                "channels": None,
                "metadata": {"dataset": row.get("dataset")},
            },
        )
    item_ids = sorted(by_item)
    manifest = {
        "schema_version": "dataset_manifest_v1",
        "dataset_id": "audio_real_rde_validation",
        "display_name": "Audio real R-D-E validation rows",
        "domain": "audio",
        "root": ".",
        "items": [by_item[item_id] for item_id in item_ids],
        "splits": {"all": item_ids, "test": item_ids},
        "metadata": {
            "source": "results/audio/visqol_audio_mode_benchmark.csv + full_pipeline_energy_benchmark.csv",
            "sample_rate": "not available in source rows",
            "channels": "not available in source rows",
        },
        "license": None,
        "source_url": None,
        "notes": "Router-facing manifest for measured audio validation rows; raw audio paths are not required for this offline validation.",
    }
    _write_json(out_dir / "audio_dataset_manifest.json", manifest)
    return manifest


def build_video_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, dict[str, Any]] = {}
    for row in rows:
        sequence = str(row["sequence"])
        n_frames = _maybe_float(row.get("n_frames"))
        by_item.setdefault(
            sequence,
            {
                "item_id": sequence,
                "path": sequence,
                "width": None,
                "height": None,
                "fps": None,
                "num_frames": int(n_frames) if n_frames is not None else None,
                "metadata": {"dataset": row.get("dataset")},
            },
        )
    item_ids = sorted(by_item)
    manifest = {
        "schema_version": "dataset_manifest_v1",
        "dataset_id": "video_real_rde_validation",
        "display_name": "Video real R-D-E validation rows",
        "domain": "video",
        "root": ".",
        "items": [by_item[item_id] for item_id in item_ids],
        "splits": {"all": item_ids, "test": item_ids},
        "metadata": {
            "source": "results/video/video_LDP_reference_preset_paper_ready.csv",
            "width": "not available in source rows",
            "height": "not available in source rows",
            "fps": "not available in source rows",
        },
        "license": None,
        "source_url": None,
        "notes": "Router-facing manifest for measured video validation rows; raw video paths are not required for this offline validation.",
    }
    _write_json(out_dir / "video_dataset_manifest.json", manifest)
    return manifest


def _domain_validation(csv_path: Path, domain_spec_name: str) -> dict[str, Any]:
    spec = resolve_domain_spec(domain_spec_name)
    return validate_rde_dataframe_against_domain_spec(
        _rows_from_csv_for_domain_spec(csv_path),
        spec,
    )


def _manifest_validation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return validate_dataset_manifest(manifest, check_files=False)


def _weights(profile_name: str) -> dict[str, float]:
    profile = get_profile(profile_name)
    total = profile.w_e + profile.w_r + profile.w_d
    return {
        "w_E": profile.w_e / total,
        "w_R": profile.w_r / total,
        "w_D": profile.w_d / total,
    }


def _item_id(row: Mapping[str, Any], spec: DomainSpec) -> str:
    for column in spec.item_id_columns:
        value = row.get(column)
        if value not in (None, ""):
            return str(value)
    return ""


def _point_from_row(row: Mapping[str, Any], spec: DomainSpec) -> RDEPoint | None:
    rate = _maybe_float(row.get(spec.rate_column))
    quality = _maybe_float(row.get(spec.quality_column))
    energy = _maybe_float(row.get(spec.energy_column))
    if rate is None or quality is None or energy is None:
        return None
    time_ms = _maybe_float(row.get("time_ms"))
    return RDEPoint(
        codec=str(row.get(spec.codec_column, "")),
        config=str(row.get(spec.config_column, "")),
        rate=rate,
        quality=quality,
        energy=energy,
        raw=dict(row),
        time_ms=time_ms,
    )


def _safe_select(points: list[RDEPoint], weights: dict[str, float], floor: float | None) -> dict[str, Any] | None:
    try:
        return select_best_rde(
            points,
            weights=weights,
            min_quality=floor,
            quality_constraint_stat="mean",
            top_k=max(1, len(points)),
        )
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _codec_family(codec: str) -> str:
    token = codec.strip().lower()
    if token in NEURAL_CODECS:
        return "neural"
    if token in CLASSICAL_CODECS:
        return "classical"
    return "unknown"


def policy_validation(
    csv_path: Path,
    domain_spec_name: str,
    out_prefix: Path,
    *,
    profiles: list[str] | None = None,
    router_reports: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    spec = resolve_domain_spec(domain_spec_name)
    rows = _read_csv(csv_path)
    profiles = profiles or DEFAULT_PROFILES
    by_item: dict[str, list[RDEPoint]] = defaultdict(list)
    all_points: list[RDEPoint] = []
    for row in rows:
        point = _point_from_row(row, spec)
        item_id = _item_id(row, spec)
        if point is None or not item_id:
            continue
        by_item[item_id].append(point)
        all_points.append(point)

    comparison_rows: list[dict[str, Any]] = []
    selected_codec_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    rqe_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "valid": True,
        "csv": _source(csv_path),
        "domain_spec": domain_spec_name,
        "domain_spec_details": domain_spec_to_dict(spec),
        "num_rows": len(rows),
        "num_items": len(by_item),
        "profiles": {},
        "warnings": [],
    }

    for profile_name in profiles:
        weights = _weights(profile_name)
        quality_floor = _quality_floor_for(spec.domain, spec.quality_column)
        item_oracles: dict[str, dict[str, Any]] = {}
        for item, points in by_item.items():
            decision = _safe_select(points, weights, quality_floor)
            if decision is not None:
                item_oracles[item] = decision["selected"]

        config_costs: dict[tuple[str, str], list[float]] = defaultdict(list)
        for item, points in by_item.items():
            decision = _safe_select(points, weights, quality_floor)
            if decision is None:
                continue
            for candidate in decision["scored_candidate_pool"]:
                config_costs[
                    (str(candidate["codec"]), str(candidate["config"]))
                ].append(float(candidate["cost"]))

        robust_config: tuple[str, str] | None = None
        if config_costs:
            robust_config = min(
                config_costs,
                key=lambda key: _mean(config_costs[key]) if _mean(config_costs[key]) is not None else float("inf"),
            )

        router_selected: tuple[str, str] | None = None
        router_report_path = None
        if router_reports and profile_name in router_reports:
            router_report_path = router_reports[profile_name]
            if router_report_path.exists():
                router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
                selected = router_report.get("decision", {}).get("selected", {})
                if selected.get("codec") and selected.get("config"):
                    router_selected = (str(selected["codec"]), str(selected["config"]))

        oracle_costs: list[float] = []
        robust_costs: list[float] = []
        router_costs: list[float] = []
        robust_regrets: list[float] = []
        router_regrets: list[float] = []
        oracle_codecs: Counter[str] = Counter()
        robust_codecs: Counter[str] = Counter()
        router_codecs: Counter[str] = Counter()
        feasible_items = 0
        for item, points in by_item.items():
            decision = _safe_select(points, weights, quality_floor)
            if decision is None:
                continue
            feasible_items += 1
            selected = decision["selected"]
            oracle_cost = float(selected["cost"])
            oracle_costs.append(oracle_cost)
            oracle_codecs[str(selected["codec"])] += 1
            scored = {
                (str(candidate["codec"]), str(candidate["config"])): candidate
                for candidate in decision["scored_candidate_pool"]
            }
            if robust_config in scored:
                robust_cost = float(scored[robust_config]["cost"])  # type: ignore[index]
                robust_costs.append(robust_cost)
                robust_regrets.append(robust_cost - oracle_cost)
                robust_codecs[str(robust_config[0])] += 1  # type: ignore[index]
            if router_selected in scored:
                router_cost = float(scored[router_selected]["cost"])  # type: ignore[index]
                router_costs.append(router_cost)
                router_regrets.append(router_cost - oracle_cost)
                router_codecs[str(router_selected[0])] += 1  # type: ignore[index]

        mean_oracle = _mean(oracle_costs)
        mean_robust = _mean(robust_costs)
        mean_router = _mean(router_costs)
        robust_regret = _mean(robust_regrets)
        router_regret = _mean(router_regrets)
        profile_summary = {
            "profile": profile_name,
            "weights": weights,
            "quality_floor": quality_floor,
            "feasible_items": feasible_items,
            "oracle_mean_cost": mean_oracle,
            "robust_global_baseline": {
                "codec": robust_config[0] if robust_config else None,
                "config": robust_config[1] if robust_config else None,
                "mean_cost": mean_robust,
                "mean_regret": robust_regret,
                "evaluable_items": len(robust_costs),
                "selected_codec_distribution": dict(sorted(robust_codecs.items())),
            },
            "router_selected_policy": {
                "codec": router_selected[0] if router_selected else None,
                "config": router_selected[1] if router_selected else None,
                "router_report": _source(router_report_path) if router_report_path else None,
                "mean_cost": mean_router,
                "mean_regret": router_regret,
                "evaluable_items": len(router_costs),
                "selected_codec_distribution": dict(sorted(router_codecs.items())),
            },
            "oracle_selected_codec_distribution": dict(sorted(oracle_codecs.items())),
        }
        report["profiles"][profile_name] = profile_summary
        for policy_name, mean_cost, regret, codec, config, evaluable_items in (
            (
                "oracle",
                mean_oracle,
                0.0 if mean_oracle is not None else None,
                None,
                None,
                feasible_items,
            ),
            (
                "robust_global_baseline",
                mean_robust,
                robust_regret,
                robust_config[0] if robust_config else None,
                robust_config[1] if robust_config else None,
                len(robust_costs),
            ),
            (
                "router_selected_policy",
                mean_router,
                router_regret,
                router_selected[0] if router_selected else None,
                router_selected[1] if router_selected else None,
                len(router_costs),
            ),
        ):
            comparison_rows.append(
                {
                    "profile": profile_name,
                    "policy": policy_name,
                    "selected_codec": codec or "",
                    "selected_config": config or "",
                    "mean_cost": _format_float(mean_cost),
                    "mean_regret": _format_float(regret),
                    "feasible_items": feasible_items,
                    "evaluable_items": evaluable_items,
                    "quality_floor": _format_float(quality_floor),
                }
            )
        for codec, count in sorted(oracle_codecs.items()):
            selected_codec_rows.append(
                {
                    "profile": profile_name,
                    "policy": "oracle",
                    "codec": codec,
                    "family": _codec_family(codec),
                    "count": count,
                    "share": _format_float(count / feasible_items if feasible_items else None),
                }
            )
        for policy_name, dist in (
            ("robust_global_baseline", robust_codecs),
            ("router_selected_policy", router_codecs),
        ):
            for codec, count in sorted(dist.items()):
                selected_codec_rows.append(
                    {
                        "profile": profile_name,
                        "policy": policy_name,
                        "codec": codec,
                        "family": _codec_family(codec),
                        "count": count,
                        "share": _format_float(count / feasible_items if feasible_items else None),
                    }
                )
        regret_rows.extend(
            row
            for row in comparison_rows
            if row["profile"] == profile_name and row["policy"] != "oracle"
        )
        for policy_name, selected_key in (
            ("oracle", None),
            ("robust_global_baseline", robust_config),
            ("router_selected_policy", router_selected),
        ):
            selected_points: list[RDEPoint] = []
            for item, points in by_item.items():
                decision = _safe_select(points, weights, quality_floor)
                if decision is None:
                    continue
                if selected_key is None:
                    raw = decision["selected"]
                    selected_points.append(
                        RDEPoint(
                            codec=str(raw["codec"]),
                            config=str(raw["config"]),
                            rate=float(raw["rate"]),
                            quality=float(raw["quality"]),
                            energy=float(raw["energy"]),
                            raw={},
                            time_ms=raw.get("time_ms"),
                        )
                    )
                    continue
                for point in points:
                    if (point.codec, point.config) == selected_key:
                        selected_points.append(point)
                        break
            rqe_rows.append(
                {
                    "profile": profile_name,
                    "policy": policy_name,
                    "mean_rate": _format_float(_mean(p.rate for p in selected_points)),
                    "mean_quality": _format_float(_mean(p.quality for p in selected_points)),
                    "mean_energy": _format_float(_mean(p.energy for p in selected_points)),
                    "num_items": len(selected_points),
                    "quality_floor": _format_float(quality_floor),
                }
            )

    _write_csv(
        out_prefix.with_name(out_prefix.name + "_comparison.csv"),
        comparison_rows,
        [
            "profile",
            "policy",
            "selected_codec",
            "selected_config",
            "mean_cost",
            "mean_regret",
            "feasible_items",
            "evaluable_items",
            "quality_floor",
        ],
    )
    _write_json(out_prefix.with_name(out_prefix.name + "_comparison.json"), report)
    prefix_name = out_prefix.name.removesuffix("_policy")
    _write_csv(
        out_prefix.with_name(prefix_name + "_selected_codec_by_profile.csv"),
        selected_codec_rows,
        ["profile", "policy", "codec", "family", "count", "share"],
    )
    _write_csv(
        out_prefix.with_name(prefix_name + "_regret_by_profile.csv"),
        regret_rows,
        [
            "profile",
            "policy",
            "selected_codec",
            "selected_config",
            "mean_cost",
            "mean_regret",
            "feasible_items",
            "evaluable_items",
            "quality_floor",
        ],
    )
    _write_csv(
        out_prefix.with_name(prefix_name + "_rate_quality_energy_summary.csv"),
        rqe_rows,
        ["profile", "policy", "mean_rate", "mean_quality", "mean_energy", "num_items", "quality_floor"],
    )
    return report


def _quality_floor_for(domain: str, quality_col: str) -> float | None:
    if domain == "audio" and quality_col == "visqol":
        return 3.5
    if domain == "video" and quality_col == "vmaf":
        return 80.0
    return None


def run_router_profiles(
    csv_path: Path,
    domain_spec_name: str,
    out_dir: Path,
    prefix: str,
    profiles: list[str] | None = None,
) -> dict[str, Path]:
    profiles = profiles or DEFAULT_PROFILES
    report_paths: dict[str, Path] = {}
    summary_rows: list[dict[str, Any]] = []
    for profile in profiles:
        safe = profile.replace("-", "_")
        report_path = out_dir / f"{prefix}_router_{safe}_report.json"
        summary_path = out_dir / f"{prefix}_router_{safe}_summary.csv"
        router_main(
            [
                "--csv",
                str(csv_path),
                "--domain-spec",
                domain_spec_name,
                "--profile",
                profile,
                "--out",
                str(report_path),
                "--summary-out",
                str(summary_path),
            ]
        )
        report_paths[profile] = report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = report.get("decision", {}).get("selected", {})
        decision = report.get("decision", {})
        summary_rows.append(
            {
                "profile": profile,
                "selected_codec": selected.get("codec", ""),
                "selected_config": selected.get("config", ""),
                "selected_family": _codec_family(str(selected.get("codec", ""))),
                "mean_rate": _format_float(_maybe_float(selected.get("rate"))),
                "mean_quality": _format_float(_maybe_float(selected.get("quality"))),
                "mean_energy": _format_float(_maybe_float(selected.get("energy"))),
                "feasible_rows": decision.get("num_points_admissible", ""),
                "quality_floor": report.get("constraints", {}).get("quality_floor", ""),
                "warnings": ";".join(str(w) for w in report.get("warnings", [])),
                "report_path": _source(report_path),
                "summary_path": _source(summary_path),
            }
        )
    _write_csv(
        out_dir / f"{prefix}_router_profile_summary.csv",
        summary_rows,
        [
            "profile",
            "selected_codec",
            "selected_config",
            "selected_family",
            "mean_rate",
            "mean_quality",
            "mean_energy",
            "feasible_rows",
            "quality_floor",
            "warnings",
            "report_path",
            "summary_path",
        ],
    )
    return report_paths


def _optional_plots(out_dir: Path, prefix: str) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    written: list[str] = []
    try:
        selected_path = out_dir / f"{prefix}_selected_codec_by_profile.csv"
        if selected_path.exists():
            rows = _read_csv(selected_path)
            labels = [f"{row['profile']}\n{row['codec']}" for row in rows if row.get("policy") == "oracle"]
            values = [float(row["count"]) for row in rows if row.get("policy") == "oracle"]
            if labels and values:
                plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
                plt.bar(labels, values)
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("count")
                plt.tight_layout()
                out = out_dir / f"{prefix}_codec_selection_by_profile.png"
                plt.savefig(out, dpi=160)
                plt.close()
                written.append(_source(out))

        csv_path = out_dir / f"{prefix}_rde_router_ready.csv"
        if csv_path.exists():
            rows = _read_csv(csv_path)
            rate_col = "bitrate_kbps"
            quality_col = "visqol" if prefix == "audio" else "vmaf"
            energy_col = "energy_j_per_second" if prefix == "audio" else "energy_kj_per_sequence"
            rates = [_maybe_float(row.get(rate_col)) for row in rows]
            qualities = [_maybe_float(row.get(quality_col)) for row in rows]
            energies = [_maybe_float(row.get(energy_col)) for row in rows]
            triples = [
                (r, q, e)
                for r, q, e in zip(rates, qualities, energies)
                if r is not None and q is not None and e is not None
            ]
            if triples:
                plt.figure(figsize=(6, 4))
                plt.scatter(
                    [r for r, _, _ in triples],
                    [q for _, q, _ in triples],
                    c=[e for _, _, e in triples],
                    s=14,
                    alpha=0.75,
                )
                plt.xlabel(rate_col)
                plt.ylabel(quality_col)
                plt.colorbar(label=energy_col)
                plt.tight_layout()
                out = out_dir / f"{prefix}_rde_scatter.png"
                plt.savefig(out, dpi=160)
                plt.close()
                written.append(_source(out))
    except Exception:
        return written
    return written


def finalize_report(out_dir: Path, audio_policy: Mapping[str, Any], video_policy: Mapping[str, Any]) -> dict[str, Any]:
    audio_build = json.loads((out_dir / "audio_rde_build_report.json").read_text(encoding="utf-8"))
    video_build = json.loads((out_dir / "video_rde_build_report.json").read_text(encoding="utf-8"))
    audio_domain = json.loads((out_dir / "audio_domain_validation.json").read_text(encoding="utf-8"))
    video_domain = json.loads((out_dir / "video_domain_validation.json").read_text(encoding="utf-8"))
    audio_manifest = json.loads((out_dir / "audio_manifest_validation.json").read_text(encoding="utf-8"))
    video_manifest = json.loads((out_dir / "video_manifest_validation.json").read_text(encoding="utf-8"))
    payload = {
        "valid": all(
            bool(item.get("valid"))
            for item in (audio_build, video_build, audio_domain, video_domain, audio_manifest, video_manifest)
        ),
        "scope": "real measured audio/video R-D-E router validation",
        "not_a_new_benchmark": True,
        "audio": {
            "source_files": audio_build["source_files"],
            "domain_spec": "audio_visqol",
            "num_rows": audio_build["num_output_rows"],
            "codecs": audio_build["codecs"],
            "datasets": audio_build["datasets"],
            "quality_metric": "visqol",
            "rate_unit": "kbps",
            "energy_unit": "J/s",
            "router_runs": _router_runs_summary(out_dir, "audio"),
            "policy_comparison": {
                "path": _source(out_dir / "audio_policy_comparison.json"),
                "profiles": audio_policy.get("profiles", {}),
            },
            "warnings": audio_build.get("warnings", []),
        },
        "video": {
            "source_files": video_build["source_files"],
            "domain_spec": "video_vmaf",
            "num_rows": video_build["num_output_rows"],
            "codecs": video_build["codecs"],
            "datasets": video_build["datasets"],
            "quality_metric": "vmaf",
            "rate_unit": "kbps",
            "energy_unit": "kJ/sequence",
            "router_runs": _router_runs_summary(out_dir, "video"),
            "policy_comparison": {
                "path": _source(out_dir / "video_policy_comparison.json"),
                "profiles": video_policy.get("profiles", {}),
            },
            "warnings": video_build.get("warnings", []),
        },
        "boundaries": [
            "uses existing measured benchmark rows",
            "does not validate new codec execution backends",
            "does not introduce content-aware audio/video predictors",
            "does not replace the domain-specific discussion of Chapter 4",
            "does not invent missing quality or energy measurements",
        ],
    }
    _write_json(out_dir / "audio_video_router_validation_report.json", payload)
    return payload


def _router_runs_summary(out_dir: Path, prefix: str) -> dict[str, Any]:
    path = out_dir / f"{prefix}_router_profile_summary.csv"
    if not path.exists():
        return {"summary_csv": _source(path), "profiles": {}}
    rows = _read_csv(path)
    return {
        "summary_csv": _source(path),
        "profiles": {row["profile"]: row for row in rows},
    }


def build_all(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_source_inventory(root, out_dir)
    audio = build_audio_router_ready(root, out_dir)
    video = build_video_router_ready(root, out_dir)

    audio_manifest = build_audio_manifest(out_dir, audio.rows)
    video_manifest = build_video_manifest(out_dir, video.rows)
    _write_json(out_dir / "audio_domain_validation.json", _domain_validation(out_dir / "audio_rde_router_ready.csv", "audio_visqol"))
    _write_json(out_dir / "video_domain_validation.json", _domain_validation(out_dir / "video_rde_router_ready.csv", "video_vmaf"))
    _write_json(out_dir / "audio_manifest_validation.json", _manifest_validation(audio_manifest))
    _write_json(out_dir / "video_manifest_validation.json", _manifest_validation(video_manifest))

    audio_router_reports = run_router_profiles(
        out_dir / "audio_rde_router_ready.csv",
        "audio_visqol",
        out_dir,
        "audio",
    )
    video_router_reports = run_router_profiles(
        out_dir / "video_rde_router_ready.csv",
        "video_vmaf",
        out_dir,
        "video",
    )

    audio_policy = policy_validation(
        out_dir / "audio_rde_router_ready.csv",
        "audio_visqol",
        out_dir / "audio_policy",
        router_reports=audio_router_reports,
    )
    video_policy = policy_validation(
        out_dir / "video_rde_router_ready.csv",
        "video_vmaf",
        out_dir / "video_policy",
        router_reports=video_router_reports,
    )
    plots = _optional_plots(out_dir, "audio") + _optional_plots(out_dir, "video")
    final = finalize_report(out_dir, audio_policy, video_policy)
    final["source_inventory_candidates"] = inventory["num_candidates"]
    final["derived_figures"] = plots
    _write_json(out_dir / "audio_video_router_validation_report.json", final)
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build real measured audio/video R-D-E router validation artifacts."
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument(
        "--mode",
        choices=["all", "inventory", "build-csv", "policy"],
        default="all",
        help="Artifact generation mode.",
    )
    parser.add_argument("--csv", default=None, help="Router-ready CSV for --mode policy.")
    parser.add_argument("--domain-spec", default=None, help="DomainSpec for --mode policy.")
    parser.add_argument("--out-prefix", default=None, help="Output prefix path for --mode policy.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if args.mode == "inventory":
        return build_source_inventory(root, out_dir)
    if args.mode == "build-csv":
        audio = build_audio_router_ready(root, out_dir)
        video = build_video_router_ready(root, out_dir)
        return {"valid": audio.report["valid"] and video.report["valid"]}
    if args.mode == "policy":
        if not args.csv or not args.domain_spec or not args.out_prefix:
            parser.error("--mode policy requires --csv, --domain-spec and --out-prefix")
        return policy_validation(Path(args.csv), args.domain_spec, Path(args.out_prefix))

    return build_all(root, out_dir)


if __name__ == "__main__":
    main()
