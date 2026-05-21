"""End-to-end onboarding workflow for pluggable router datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from src.router.codecs.external_codec_spec import (
    load_external_codec_spec,
    validate_external_codec_spec,
)
from src.router.core.codec_onboarding import (
    build_codec_onboarding_summary,
    validate_codec_domain_compatibility,
    validate_codec_measurements_compatibility,
)
from src.router.core.contracts import ONBOARDING_CONTRACT_ID
from src.router.core.dataset_ingestion import (
    build_rde_csv,
    join_manifest_measurements,
    load_measurements_csv,
    validate_ingested_rde,
)
from src.router.core.dataset_manifest import (
    load_dataset_manifest,
    validate_dataset_manifest,
)
from src.router.core.domain_spec import (
    resolve_domain_spec,
    validate_rde_dataframe_against_domain_spec,
)
from src.router.rde_router import main as router_main


def _csv_columns(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _rows_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return columns


def _write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    columns = _rows_columns(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(report: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _missing_columns(path: str | Path, columns: list[str | None]) -> list[str]:
    available = set(_csv_columns(path))
    return [str(column) for column in columns if column and column not in available]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the dataset onboarding workflow end to end."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--measurements-csv", required=True)
    parser.add_argument("--domain-spec", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--item-id-col", required=True)
    parser.add_argument("--codec-col", required=True)
    parser.add_argument("--config-col", required=True)
    parser.add_argument("--rate-col", required=True)
    parser.add_argument("--quality-col", required=True)
    parser.add_argument("--energy-col", required=True)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--router-profile", default="balanced")
    parser.add_argument("--codec-spec", default=None)
    return parser


def run_onboarding(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ingested_csv = work_dir / "ingested_rde.csv"
    router_report_path = work_dir / "router_report.json"
    router_summary_path = work_dir / "router_summary.csv"
    onboarding_report_path = work_dir / "onboarding_report.json"

    warnings: list[str] = []
    errors: list[str] = []
    selected_codec = None
    selected_config = None
    codec_onboarding_report: dict[str, Any] = {
        "enabled": False,
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    manifest = load_dataset_manifest(args.manifest)
    domain_spec = resolve_domain_spec(args.domain_spec)
    manifest_report = validate_dataset_manifest(manifest)
    manifest_valid = bool(manifest_report["valid"])
    warnings.extend(manifest_report.get("warnings", []))
    errors.extend(manifest_report.get("errors", []))

    required_measurement_columns = [
        args.item_id_col,
        args.codec_col,
        args.config_col,
        args.rate_col,
        args.quality_col,
        args.energy_col,
        args.time_col,
    ]
    missing_measurement_columns = _missing_columns(
        args.measurements_csv,
        required_measurement_columns,
    )
    measurements_valid = len(missing_measurement_columns) == 0
    if missing_measurement_columns:
        errors.append(
            "missing_measurement_columns:" + ",".join(missing_measurement_columns)
        )

    measurements = load_measurements_csv(args.measurements_csv)
    column_mapping = {
        "dataset": None,
        "codec": args.codec_col,
        "config": args.config_col,
        "rate": args.rate_col,
        "quality": args.quality_col,
        "energy": args.energy_col,
        "time": args.time_col,
    }

    if args.codec_spec is not None:
        codec_spec = load_external_codec_spec(args.codec_spec)
        codec_spec_report = validate_external_codec_spec(codec_spec)
        codec_spec_report["enabled"] = True
        codec_domain_report = validate_codec_domain_compatibility(
            codec_spec,
            domain_spec,
        )
        codec_domain_report["enabled"] = True
        codec_measurements_report = validate_codec_measurements_compatibility(
            measurements,
            domain_spec,
            codec_id=codec_spec.get("codec_id"),
            column_mapping={
                "codec": args.codec_col,
                "config": args.config_col,
                "rate": args.rate_col,
                "quality": args.quality_col,
                "energy": args.energy_col,
            },
        )
        codec_measurements_report["enabled"] = True
        codec_onboarding_report = build_codec_onboarding_summary(
            domain_spec=domain_spec,
            codec_spec=codec_spec,
            codec_spec_report=codec_spec_report,
            domain_report=codec_domain_report,
            measurements_report=codec_measurements_report,
        )
        codec_onboarding_report["enabled"] = True
        warnings.extend(codec_onboarding_report.get("warnings", []))
        errors.extend(codec_onboarding_report.get("errors", []))

    ingestion_valid = False
    domain_spec_valid = False
    router_decision_valid = False
    ingestion_report: dict[str, Any] = {
        "valid": False,
        "errors": [],
        "warnings": [],
    }
    domain_spec_report: dict[str, Any] = {
        "valid": False,
        "errors": [],
        "warnings": [],
    }

    if manifest_valid and measurements_valid:
        joined = join_manifest_measurements(
            manifest,
            measurements,
            item_id_col=args.item_id_col,
        )
        unknown_items = list(joined.get("unknown_measurement_items", []))
        if unknown_items:
            errors.append("unknown_measurement_items:" + ",".join(unknown_items))

        rde_rows = build_rde_csv(joined, domain_spec, column_mapping)
        _write_rows_csv(rde_rows, ingested_csv)
        ingestion_report = validate_ingested_rde(rde_rows, domain_spec, manifest)
        ingestion_valid = (
            bool(ingestion_report["valid"])
            and not unknown_items
        )
        warnings.extend(ingestion_report.get("warnings", []))
        errors.extend(ingestion_report.get("errors", []))

        domain_spec_report = validate_rde_dataframe_against_domain_spec(
            rde_rows,
            domain_spec,
        )
        domain_spec_valid = bool(domain_spec_report["valid"])
        warnings.extend(domain_spec_report.get("warnings", []))
        errors.extend(domain_spec_report.get("errors", []))

        if ingestion_valid and domain_spec_valid:
            router_main(
                [
                    "--csv",
                    str(ingested_csv),
                    "--domain-spec",
                    args.domain_spec,
                    "--profile",
                    args.router_profile,
                    "--out",
                    str(router_report_path),
                    "--summary-out",
                    str(router_summary_path),
                ]
            )
            router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
            selected = router_report.get("decision", {}).get("selected", {})
            selected_codec = selected.get("codec")
            selected_config = selected.get("config")
            router_decision_valid = bool(selected_codec and selected_config)
            if not router_decision_valid:
                errors.append("router_decision_missing_selected_codec_or_config")

    report = {
        "valid": (
            manifest_valid
            and measurements_valid
            and ingestion_valid
            and domain_spec_valid
            and router_decision_valid
            and not errors
        ),
        "contract_id": ONBOARDING_CONTRACT_ID,
        "manifest_valid": manifest_valid,
        "measurements_valid": measurements_valid,
        "ingestion_valid": ingestion_valid,
        "domain_spec_valid": domain_spec_valid,
        "router_decision_valid": router_decision_valid,
        "selected_codec": selected_codec,
        "selected_config": selected_config,
        "outputs": {
            "onboarding_report": str(onboarding_report_path),
            "ingested_rde_csv": str(ingested_csv),
            "router_report": str(router_report_path),
            "router_summary": str(router_summary_path),
        },
        "steps": {
            "manifest": manifest_report,
            "measurements": {
                "valid": measurements_valid,
                "columns": _csv_columns(args.measurements_csv),
                "missing_columns": missing_measurement_columns,
            },
            "ingestion": ingestion_report,
            "domain_spec": domain_spec_report,
            "router_decision": {
                "valid": router_decision_valid,
                "selected_codec": selected_codec,
                "selected_config": selected_config,
            },
            "codec_onboarding": codec_onboarding_report,
        },
        "codec_onboarding": codec_onboarding_report,
        "warnings": sorted(set(str(warning) for warning in warnings)),
        "errors": sorted(set(str(error) for error in errors)),
    }
    _write_json(report, onboarding_report_path)
    return report


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = run_onboarding(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
