"""Manifest-driven ingestion from raw measurements to router-ready R-D-E CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from src.router.core.dataset_manifest import (
    DatasetManifest,
    dataset_manifest_to_dict,
    load_dataset_manifest,
    normalize_dataset_manifest,
    validate_dataset_manifest,
    validate_manifest_against_domain_spec,
)
from src.router.core.domain_spec import (
    DomainSpec,
    domain_spec_to_dict,
    resolve_domain_spec,
    validate_rde_dataframe_against_domain_spec,
)

STANDARD_METADATA_COLUMNS = [
    "width",
    "height",
    "pixels",
    "duration_s",
    "sample_rate",
    "channels",
    "fps",
    "num_frames",
]


def load_measurements_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _numeric_value(value: Any) -> float:
    if value is None:
        raise ValueError("missing")
    text = str(value).strip()
    if text == "":
        raise ValueError("missing")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"not_numeric:{value}")
    return float(match.group(0))


def _csv_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return columns


def join_manifest_measurements(
    manifest: DatasetManifest | Mapping[str, Any],
    measurements_df: list[dict[str, Any]],
    *,
    item_id_col: str,
) -> dict[str, Any]:
    normalized = normalize_dataset_manifest(manifest)
    manifest_items = {item.item_id: item for item in normalized.items}

    rows: list[dict[str, Any]] = []
    measured_item_ids: set[str] = set()
    unknown_items: list[str] = []

    for index, measurement in enumerate(measurements_df):
        item_id = str(measurement.get(item_id_col, "")).strip()
        item = manifest_items.get(item_id)
        if item is None:
            if item_id and item_id not in unknown_items:
                unknown_items.append(item_id)
            rows.append(
                {
                    "measurement_row_index": index,
                    "measurement": dict(measurement),
                    "manifest_item": None,
                    "item_id": item_id,
                    "join_status": "unknown_measurement_item",
                }
            )
            continue

        measured_item_ids.add(item_id)
        rows.append(
            {
                "measurement_row_index": index,
                "measurement": dict(measurement),
                "manifest_item": item,
                "item_id": item_id,
                "join_status": "matched",
            }
        )

    missing_manifest_items = [
        item.item_id
        for item in normalized.items
        if item.item_id not in measured_item_ids
    ]

    return {
        "manifest": normalized,
        "rows": rows,
        "item_id_col": item_id_col,
        "unknown_measurement_items": unknown_items,
        "missing_manifest_items": missing_manifest_items,
    }


def _mapped_value(
    measurement: Mapping[str, Any],
    column_mapping: Mapping[str, str | None],
    role: str,
) -> Any:
    column = column_mapping.get(role)
    if column is None:
        return None
    return measurement.get(column)


def build_rde_csv(
    joined_df: Mapping[str, Any],
    domain_spec: DomainSpec | Mapping[str, Any] | str,
    column_mapping: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain_spec) if isinstance(domain_spec, str) else domain_spec
    if not isinstance(spec, DomainSpec):
        spec = DomainSpec(**dict(spec))  # type: ignore[arg-type]

    manifest = normalize_dataset_manifest(joined_df["manifest"])
    output_rows: list[dict[str, Any]] = []
    item_column = spec.item_id_columns[0] if spec.item_id_columns else "item_id"

    for joined in joined_df["rows"]:
        if joined.get("join_status") != "matched":
            continue

        item = joined["manifest_item"]
        measurement = joined["measurement"]
        dataset_value = _mapped_value(measurement, column_mapping, "dataset")
        if dataset_value in (None, ""):
            dataset_value = manifest.dataset_id

        row: dict[str, Any] = {
            spec.dataset_column: dataset_value,
            spec.codec_column: _mapped_value(measurement, column_mapping, "codec"),
            spec.config_column: _mapped_value(measurement, column_mapping, "config"),
            item_column: item.item_id,
            spec.rate_column: _mapped_value(measurement, column_mapping, "rate"),
            spec.quality_column: _mapped_value(measurement, column_mapping, "quality"),
            spec.energy_column: _mapped_value(measurement, column_mapping, "energy"),
        }

        time_column = column_mapping.get("time")
        if time_column:
            row["time_ms"] = measurement.get(time_column)
        elif spec.time_column is not None:
            row[spec.time_column] = None

        for column in STANDARD_METADATA_COLUMNS:
            value = getattr(item, column)
            if value is not None:
                row[column] = value

        for key, value in item.metadata.items():
            row[f"metadata_{key}"] = value

        output_rows.append(row)

    return output_rows


def _invalid_rde_row_reasons(
    row: Mapping[str, Any],
    domain_spec: DomainSpec,
) -> list[str]:
    reasons: list[str] = []

    required_text_columns = [
        domain_spec.dataset_column,
        domain_spec.codec_column,
        domain_spec.config_column,
        *domain_spec.item_id_columns,
    ]
    for column in required_text_columns:
        if row.get(column) in (None, ""):
            reasons.append(f"missing_required_column_value:{column}")

    for role, column in (
        ("rate", domain_spec.rate_column),
        ("quality", domain_spec.quality_column),
        ("energy", domain_spec.energy_column),
    ):
        try:
            _numeric_value(row.get(column))
        except ValueError as exc:
            reasons.append(f"invalid_{role}:{column}:{exc}")

    return reasons


def validate_ingested_rde(
    df: list[dict[str, Any]],
    domain_spec: DomainSpec | Mapping[str, Any] | str,
    manifest: DatasetManifest | Mapping[str, Any],
) -> dict[str, Any]:
    spec = resolve_domain_spec(domain_spec) if isinstance(domain_spec, str) else domain_spec
    if not isinstance(spec, DomainSpec):
        spec = DomainSpec(**dict(spec))  # type: ignore[arg-type]

    manifest_report = validate_dataset_manifest(manifest)
    domain_report = validate_manifest_against_domain_spec(manifest, spec)
    if df:
        rde_report = validate_rde_dataframe_against_domain_spec(df, spec)
    else:
        rde_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "numeric_validity_summary": {},
        }

    invalid_rows = []
    for index, row in enumerate(df):
        reasons = _invalid_rde_row_reasons(row, spec)
        if reasons:
            invalid_rows.append({"row_index": index, "reasons": reasons})

    errors = list(manifest_report["errors"]) + list(domain_report["errors"])
    errors.extend(rde_report["errors"])
    warnings = list(manifest_report["warnings"]) + list(domain_report["warnings"])
    warnings.extend(rde_report["warnings"])

    return {
        "valid": len(errors) == 0 and len(invalid_rows) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_valid_rde_rows": len(df) - len(invalid_rows),
        "num_invalid_rde_rows": len(invalid_rows),
        "invalid_rde_rows": invalid_rows,
        "numeric_validity_summary": rde_report.get("numeric_validity_summary", {}),
        "domain_spec_validation": rde_report,
    }


def _write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    columns = _csv_columns(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ingestion_report(report: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_ingestion_report(
    *,
    manifest: DatasetManifest,
    domain_spec: DomainSpec,
    measurements: list[dict[str, Any]],
    joined: Mapping[str, Any],
    rde_rows: list[dict[str, Any]],
    validation_report: Mapping[str, Any],
    column_mapping: Mapping[str, str | None],
    measurements_csv: str | Path,
    output_csv: str | Path,
    strict: bool,
    allow_missing_items: bool,
) -> dict[str, Any]:
    unknown_items = list(joined.get("unknown_measurement_items", []))
    missing_items = list(joined.get("missing_manifest_items", []))
    errors = list(validation_report["errors"])
    warnings = list(validation_report["warnings"])

    if unknown_items:
        message = "unknown_measurement_items:" + ",".join(unknown_items)
        if strict:
            errors.append(message)
        else:
            warnings.append(message + ":dropped")

    if missing_items:
        warning = "manifest_items_without_measurements:" + ",".join(missing_items)
        warnings.append(warning)
        if not allow_missing_items:
            warnings.append("allow_missing_items_false_missing_items_retained_as_warning")

    return {
        "valid": len(errors) == 0 and validation_report["num_invalid_rde_rows"] == 0,
        "manifest": dataset_manifest_to_dict(manifest),
        "domain_spec": domain_spec_to_dict(domain_spec),
        "input_measurements": {
            "csv": str(measurements_csv),
            "columns": _csv_columns(measurements),
        },
        "output_csv": str(output_csv),
        "num_manifest_items": len(manifest.items),
        "num_measurement_rows": len(measurements),
        "num_joined_rows": len(rde_rows),
        "num_valid_rde_rows": validation_report["num_valid_rde_rows"],
        "num_invalid_rde_rows": validation_report["num_invalid_rde_rows"],
        "missing_manifest_items": missing_items,
        "unknown_measurement_items": unknown_items,
        "column_mapping": dict(column_mapping),
        "numeric_validity_summary": validation_report["numeric_validity_summary"],
        "warnings": warnings,
        "errors": errors,
        "invalid_rde_rows": validation_report["invalid_rde_rows"],
        "strict": strict,
        "allow_missing_items": allow_missing_items,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest DatasetManifest + measurements CSV into router-ready R-D-E CSV."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--measurements-csv", required=True)
    parser.add_argument("--domain-spec", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--item-id-col", required=True)
    parser.add_argument("--dataset-col", default=None)
    parser.add_argument("--codec-col", required=True)
    parser.add_argument("--config-col", required=True)
    parser.add_argument("--rate-col", required=True)
    parser.add_argument("--quality-col", required=True)
    parser.add_argument("--energy-col", required=True)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--allow-missing-items", type=_parse_bool, default=False)
    parser.add_argument("--strict", type=_parse_bool, default=True)
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    manifest = load_dataset_manifest(args.manifest)
    domain_spec = resolve_domain_spec(args.domain_spec)
    measurements = load_measurements_csv(args.measurements_csv)
    column_mapping = {
        "dataset": args.dataset_col,
        "codec": args.codec_col,
        "config": args.config_col,
        "rate": args.rate_col,
        "quality": args.quality_col,
        "energy": args.energy_col,
        "time": args.time_col,
    }

    joined = join_manifest_measurements(
        manifest,
        measurements,
        item_id_col=args.item_id_col,
    )
    rde_rows = build_rde_csv(joined, domain_spec, column_mapping)
    validation = validate_ingested_rde(rde_rows, domain_spec, manifest)
    report = _build_ingestion_report(
        manifest=manifest,
        domain_spec=domain_spec,
        measurements=measurements,
        joined=joined,
        rde_rows=rde_rows,
        validation_report=validation,
        column_mapping=column_mapping,
        measurements_csv=args.measurements_csv,
        output_csv=args.out_csv,
        strict=args.strict,
        allow_missing_items=args.allow_missing_items,
    )

    _write_rows_csv(rde_rows, args.out_csv)
    write_ingestion_report(report, args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
