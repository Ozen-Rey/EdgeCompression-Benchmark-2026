"""Export raw external codec benchmark measurements to R-D-E-shaped CSV."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path
from typing import Any

from src.router.codecs.external_codec_spec import (
    load_external_codec_spec,
    validate_external_codec_spec,
)


SCHEMA_VERSION = "external_codec_rde_export_v1"

RAW_REQUIRED_COLUMNS = {
    "codec_id",
    "domain",
    "input_path",
    "input_id",
    "param_set_id",
    "param_json",
    "encode_success",
    "decode_success",
    "output_size_bytes",
    "encode_time_ms",
    "decode_time_ms",
    "energy_j",
    "energy_provenance_tier",
    "error",
}

RDE_COLUMNS = [
    "codec",
    "param",
    "input_id",
    "input_path",
    "rate",
    "quality",
    "energy",
    "time_ms",
    "energy_provenance_tier",
    "measurement_provenance",
    "success",
    "error",
    "source_raw_csv",
]


def export_external_codec_rde(
    *,
    spec_path: str | Path,
    raw_csv: str | Path,
    out: str | Path,
    report_out: str | Path,
    quality_csv: str | Path | None = None,
    quality_column: str | None = None,
    rate_mode: str = "bytes",
    time_mode: str = "encode",
) -> dict[str, Any]:
    spec_file = Path(spec_path)
    raw_path = Path(raw_csv)
    out_path = Path(out)
    report_path = Path(report_out)
    errors: list[str] = []
    warnings: list[str] = []

    report = {
        "external_codec_rde_export": {
            "schema_version": SCHEMA_VERSION,
            "codec_id": None,
            "raw_csv": str(raw_path),
            "output_csv": str(out_path),
            "raw_rows": 0,
            "successful_raw_rows": 0,
            "failed_raw_rows": 0,
            "exported_rows": 0,
            "structurally_valid_csv": False,
            "quality_available": False,
            "energy_available": False,
            "router_ready": False,
            "rate_mode": rate_mode,
            "warnings": warnings,
            "errors": errors,
            "safety": {
                "codec_executed": False,
                "encode_executed": False,
                "decode_executed": False,
                "router_candidate_registered": False,
            },
        }
    }
    body = report["external_codec_rde_export"]

    try:
        spec = load_external_codec_spec(spec_file)
    except Exception as exc:
        errors.append(f"spec_load_error:{exc}")
        _write_outputs([], out_path, report, report_path)
        return report

    body["codec_id"] = spec.get("codec_id")
    validation = validate_external_codec_spec(spec)
    if not validation.get("valid", False):
        errors.extend(validation.get("errors", []))
        warnings.extend(validation.get("warnings", []))
        _write_outputs([], out_path, report, report_path)
        return report

    if rate_mode not in {"image_bpp", "bytes"}:
        errors.append("invalid_rate_mode")
        _write_outputs([], out_path, report, report_path)
        return report
    if time_mode not in {"encode", "encode_decode"}:
        errors.append("invalid_time_mode")
        _write_outputs([], out_path, report, report_path)
        return report

    raw_rows, fieldnames = _read_csv(raw_path, errors)
    if raw_rows is None:
        _write_outputs([], out_path, report, report_path)
        return report

    missing = sorted(RAW_REQUIRED_COLUMNS - set(fieldnames or []))
    if missing:
        errors.append(f"raw_csv_missing_columns:{','.join(missing)}")
        _write_outputs([], out_path, report, report_path)
        return report

    body["structurally_valid_csv"] = True
    body["raw_rows"] = len(raw_rows)
    quality_map = _read_quality_map(
        quality_csv,
        quality_column,
        errors,
        warnings,
    )
    if errors:
        _write_outputs([], out_path, report, report_path)
        return report

    exported: list[dict[str, Any]] = []
    successful_raw = 0
    failed_raw = 0
    quality_values: list[str] = []
    energy_values: list[str] = []

    for raw in raw_rows:
        if not _raw_success(raw):
            failed_raw += 1
            continue
        successful_raw += 1
        row = _export_row(
            raw,
            spec=spec,
            raw_path=raw_path,
            quality_map=quality_map,
            quality_column=quality_column,
            rate_mode=rate_mode,
            time_mode=time_mode,
            warnings=warnings,
        )
        exported.append(row)
        if row["quality"] not in ("", None):
            quality_values.append(str(row["quality"]))
        if row["energy"] not in ("", None):
            energy_values.append(str(row["energy"]))

    body["successful_raw_rows"] = successful_raw
    body["failed_raw_rows"] = failed_raw
    body["exported_rows"] = len(exported)
    if failed_raw:
        warnings.append("failed_raw_rows_not_exported")

    body["quality_available"] = bool(exported) and len(quality_values) == len(exported)
    body["energy_available"] = bool(exported) and len(energy_values) == len(exported)
    body["router_ready"] = (
        body["structurally_valid_csv"]
        and body["exported_rows"] > 0
        and body["quality_available"]
        and body["energy_available"]
        and not errors
    )

    _write_outputs(exported, out_path, report, report_path)
    return report


def _read_csv(path: Path, errors: list[str]) -> tuple[list[dict[str, str]] | None, list[str] | None]:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader), reader.fieldnames
    except Exception as exc:
        errors.append(f"csv_read_error:{exc}")
        return None, None


def _read_quality_map(
    quality_csv: str | Path | None,
    quality_column: str | None,
    errors: list[str],
    warnings: list[str],
) -> dict[tuple[str, str], str]:
    if quality_csv is None:
        warnings.append("quality_csv_not_provided")
        return {}
    if not quality_column:
        errors.append("quality_column_required_with_quality_csv")
        return {}

    path = Path(quality_csv)
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            required = {"input_id", "param_set_id", quality_column}
            missing = sorted(required - fieldnames)
            if missing:
                errors.append(f"quality_csv_missing_columns:{','.join(missing)}")
                return {}
            return {
                (row.get("input_id", ""), row.get("param_set_id", "")): row.get(quality_column, "")
                for row in reader
            }
    except Exception as exc:
        errors.append(f"quality_csv_read_error:{exc}")
        return {}


def _raw_success(row: dict[str, str]) -> bool:
    encode_success = _as_bool(row.get("encode_success"))
    decode_value = row.get("decode_success", "")
    decode_success = True if decode_value == "" else _as_bool(decode_value)
    return encode_success and decode_success


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _export_row(
    raw: dict[str, str],
    *,
    spec: dict[str, Any],
    raw_path: Path,
    quality_map: dict[tuple[str, str], str],
    quality_column: str | None,
    rate_mode: str,
    time_mode: str,
    warnings: list[str],
) -> dict[str, Any]:
    input_id = raw.get("input_id", "")
    param_set_id = raw.get("param_set_id", "")
    quality = ""
    if quality_column:
        quality = quality_map.get((input_id, param_set_id), "")

    energy = raw.get("energy_j", "")
    tier = raw.get("energy_provenance_tier") or "unknown"
    if not energy:
        tier = "unknown"

    return {
        "codec": spec.get("codec_id") or raw.get("codec_id", ""),
        "param": raw.get("param_json", ""),
        "input_id": input_id,
        "input_path": raw.get("input_path", ""),
        "rate": _rate_value(raw, rate_mode, warnings),
        "quality": quality,
        "energy": energy,
        "time_ms": _time_ms(raw, time_mode),
        "energy_provenance_tier": tier,
        "measurement_provenance": _measurement_provenance(
            rate_mode,
            quality_available=bool(quality),
            energy_available=bool(energy),
            time_mode=time_mode,
        ),
        "success": True,
        "error": "",
        "source_raw_csv": str(raw_path),
    }


def _rate_value(row: dict[str, str], rate_mode: str, warnings: list[str]) -> str:
    size = _float_or_none(row.get("output_size_bytes"))
    if size is None:
        return ""
    if rate_mode == "bytes":
        return _format_number(size)

    dimensions = _image_dimensions(Path(row.get("input_path", "")))
    if dimensions is None:
        warnings.append("image_bpp_dimension_read_failed_fell_back_to_bytes")
        return _format_number(size)
    width, height = dimensions
    if width <= 0 or height <= 0:
        warnings.append("image_bpp_invalid_dimensions_fell_back_to_bytes")
        return _format_number(size)
    return _format_number(size * 8.0 / float(width * height))


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(24)
    except Exception:
        return None
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        return struct.unpack(">II", header[16:24])
    return None


def _time_ms(row: dict[str, str], time_mode: str) -> str:
    encode = _float_or_none(row.get("encode_time_ms"))
    if encode is None:
        return ""
    if time_mode == "encode":
        return _format_number(encode)
    decode = _float_or_none(row.get("decode_time_ms"))
    if decode is None:
        decode = 0.0
    return _format_number(encode + decode)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def _measurement_provenance(
    rate_mode: str,
    *,
    quality_available: bool,
    energy_available: bool,
    time_mode: str,
) -> str:
    parts = [
        f"rate:{rate_mode}",
        f"time:{time_mode}",
        "quality:external_csv" if quality_available else "quality:missing",
        "energy:raw_csv" if energy_available else "energy:not_measured",
    ]
    return ";".join(parts)


def _write_outputs(
    rows: list[dict[str, Any]],
    out_path: Path,
    report: dict[str, Any],
    report_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RDE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RDE_COLUMNS})

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export external codec raw benchmark CSV to R-D-E-shaped CSV."
    )
    parser.add_argument("--spec", required=True, help="External codec spec JSON.")
    parser.add_argument("--raw-csv", required=True, help="Raw benchmark CSV.")
    parser.add_argument("--out", required=True, help="R-D-E output CSV.")
    parser.add_argument("--report-out", required=True, help="Export report JSON.")
    parser.add_argument("--quality-csv", help="Optional quality CSV.")
    parser.add_argument("--quality-column", help="Quality value column in quality CSV.")
    parser.add_argument(
        "--rate-mode",
        choices=["image_bpp", "bytes"],
        default="bytes",
        help="Rate export mode.",
    )
    parser.add_argument(
        "--time-mode",
        choices=["encode", "encode_decode"],
        default="encode",
        help="Time export mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = export_external_codec_rde(
        spec_path=args.spec,
        raw_csv=args.raw_csv,
        out=args.out,
        report_out=args.report_out,
        quality_csv=args.quality_csv,
        quality_column=args.quality_column,
        rate_mode=args.rate_mode,
        time_mode=args.time_mode,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
