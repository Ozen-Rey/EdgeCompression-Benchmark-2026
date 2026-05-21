"""Codec onboarding checks for pluggable R-D-E measurements."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from src.router.codecs.external_codec_spec import (
    load_external_codec_spec,
    validate_external_codec_spec,
)
from src.router.core.contracts import ONBOARDING_CONTRACT_ID
from src.router.core.dataset_ingestion import (
    build_measurements_template,
    load_measurements_csv,
)
from src.router.core.domain_spec import (
    DomainSpec,
    domain_spec_to_dict,
    resolve_domain_spec,
)


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


def _write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    columns = _csv_columns(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _numeric_summary(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    valid = 0
    missing = 0
    invalid = 0
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        value = row.get(column)
        try:
            _numeric_value(value)
            valid += 1
        except ValueError as exc:
            if str(exc) == "missing":
                missing += 1
            else:
                invalid += 1
            if len(examples) < 5:
                examples.append(
                    {"row_index": index, "value": value, "reason": str(exc)}
                )
    total = len(rows)
    return {
        "column": column,
        "total": total,
        "valid": valid,
        "missing": missing,
        "invalid": invalid,
        "all_valid": valid == total and missing == 0 and invalid == 0,
        "examples": examples,
    }


def validate_codec_domain_compatibility(
    codec_spec: Mapping[str, Any],
    domain_spec: DomainSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    spec = resolve_domain_spec(domain_spec) if isinstance(domain_spec, str) else domain_spec
    spec_domain = spec.domain if isinstance(spec, DomainSpec) else str(spec.get("domain", ""))
    codec_domain = str(codec_spec.get("domain", ""))
    errors: list[str] = []
    warnings: list[str] = []

    if not codec_spec.get("codec_id"):
        errors.append("missing_codec_id")
    if not codec_domain:
        errors.append("missing_codec_domain")
    elif codec_domain != spec_domain:
        errors.append(
            f"domain_mismatch:codec_spec={codec_domain}:domain_spec={spec_domain}"
        )

    return {
        "valid": len(errors) == 0,
        "domain_compatible": len(errors) == 0,
        "codec_domain": codec_domain or None,
        "domain_spec_domain": spec_domain,
        "errors": errors,
        "warnings": warnings,
    }


def validate_codec_measurements_compatibility(
    measurements_df: list[dict[str, Any]],
    domain_spec: DomainSpec | Mapping[str, Any] | str,
    codec_id: str | None = None,
    column_mapping: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    spec = resolve_domain_spec(domain_spec) if isinstance(domain_spec, str) else domain_spec
    if not isinstance(spec, DomainSpec):
        spec = DomainSpec(**dict(spec))  # type: ignore[arg-type]

    mapping = dict(column_mapping or {})
    codec_col = mapping.get("codec") or spec.codec_column
    config_col = mapping.get("config") or spec.config_column
    rate_col = mapping.get("rate") or spec.rate_column
    quality_col = mapping.get("quality") or spec.quality_column
    energy_col = mapping.get("energy") or spec.energy_column

    required = [codec_col, config_col, rate_col, quality_col, energy_col]
    columns = set(_csv_columns(measurements_df))
    missing = [column for column in required if column not in columns]

    numeric_validity_summary: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []

    if missing:
        errors.append("missing_required_columns:" + ",".join(missing))

    for role, column in (
        ("rate", rate_col),
        ("quality", quality_col),
        ("energy", energy_col),
    ):
        if column not in columns:
            continue
        summary = _numeric_summary(measurements_df, column)
        numeric_validity_summary[role] = summary
        if not summary["all_valid"]:
            errors.append(
                f"non_numeric_{role}:{column}:"
                f"invalid={summary['invalid']}:missing={summary['missing']}"
            )

    codec_ids = sorted(
        {
            str(row.get(codec_col)).strip()
            for row in measurements_df
            if str(row.get(codec_col, "")).strip()
        }
    )
    if codec_id is not None and codec_ids and codec_id not in codec_ids:
        warnings.append(f"codec_id_not_found_in_measurements:{codec_id}")

    return {
        "valid": len(errors) == 0,
        "required_columns_present": len(missing) == 0,
        "missing_required_columns": missing,
        "codec_ids_detected": codec_ids,
        "numeric_validity_summary": numeric_validity_summary,
        "errors": errors,
        "warnings": warnings,
        "column_mapping": {
            "codec": codec_col,
            "config": config_col,
            "rate": rate_col,
            "quality": quality_col,
            "energy": energy_col,
        },
    }


def build_codec_onboarding_summary(
    *,
    domain_spec: DomainSpec,
    codec_spec: Mapping[str, Any] | None = None,
    codec_spec_report: Mapping[str, Any] | None = None,
    domain_report: Mapping[str, Any] | None = None,
    measurements_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    codec_spec_report = dict(codec_spec_report or {"enabled": False})
    domain_report = dict(domain_report or {"enabled": False})
    measurements_report = dict(measurements_report or {"enabled": False})

    for section in (codec_spec_report, domain_report, measurements_report):
        errors.extend(str(error) for error in section.get("errors", []))
        warnings.extend(str(warning) for warning in section.get("warnings", []))

    return {
        "valid": len(errors) == 0,
        "contract_id": ONBOARDING_CONTRACT_ID,
        "domain_spec": domain_spec_to_dict(domain_spec),
        "codec_spec": codec_spec,
        "codec_spec_validation": codec_spec_report,
        "measurements": measurements_report,
        "codec_ids_detected": measurements_report.get("codec_ids_detected", []),
        "domain_compatible": bool(domain_report.get("domain_compatible", False)),
        "required_columns_present": bool(
            measurements_report.get("required_columns_present", False)
        ),
        "numeric_validity_summary": measurements_report.get(
            "numeric_validity_summary",
            {},
        ),
        "warnings": warnings,
        "errors": errors,
    }


def _write_report(report: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate codec onboarding inputs for R-D-E router use."
    )
    parser.add_argument("--codec-spec", default=None)
    parser.add_argument("--measurements-csv", default=None)
    parser.add_argument("--domain-spec", default=None)
    parser.add_argument("--validate-domain", action="store_true")
    parser.add_argument("--codec-col", default=None)
    parser.add_argument("--config-col", default=None)
    parser.add_argument("--rate-col", default=None)
    parser.add_argument("--quality-col", default=None)
    parser.add_argument("--energy-col", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--new-codec-measurements-template", default=None)
    parser.add_argument("--out", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.new_codec_measurements_template is not None:
        if args.out is None:
            parser.error("--new-codec-measurements-template requires --out")
        rows = build_measurements_template(args.new_codec_measurements_template)
        _write_rows_csv(rows, args.out)
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "template_out": str(args.out),
            "columns": _csv_columns(rows),
            "domain_spec": domain_spec_to_dict(
                resolve_domain_spec(args.new_codec_measurements_template)
            ),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    if args.domain_spec is None:
        parser.error("--domain-spec is required")
    if args.report_out is None:
        parser.error("--report-out is required")

    domain_spec = resolve_domain_spec(args.domain_spec)
    codec_spec = None
    codec_spec_report: dict[str, Any] = {"enabled": False, "errors": [], "warnings": []}
    domain_report: dict[str, Any] = {"enabled": False, "domain_compatible": False}
    measurements_report: dict[str, Any] = {
        "enabled": False,
        "required_columns_present": False,
        "errors": [],
        "warnings": [],
    }

    if args.codec_spec is not None:
        codec_spec = load_external_codec_spec(args.codec_spec)
        codec_spec_report = validate_external_codec_spec(codec_spec)
        codec_spec_report["enabled"] = True
        domain_report = validate_codec_domain_compatibility(codec_spec, domain_spec)
        domain_report["enabled"] = True

    if args.measurements_csv is not None:
        measurements = load_measurements_csv(args.measurements_csv)
        measurements_report = validate_codec_measurements_compatibility(
            measurements,
            domain_spec,
            codec_id=codec_spec.get("codec_id") if codec_spec else None,
            column_mapping={
                "codec": args.codec_col,
                "config": args.config_col,
                "rate": args.rate_col,
                "quality": args.quality_col,
                "energy": args.energy_col,
            },
        )
        measurements_report["enabled"] = True
        measurements_report["csv"] = str(args.measurements_csv)

    report = build_codec_onboarding_summary(
        domain_spec=domain_spec,
        codec_spec=codec_spec,
        codec_spec_report=codec_spec_report,
        domain_report=domain_report,
        measurements_report=measurements_report,
    )
    _write_report(report, args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
