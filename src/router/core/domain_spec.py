"""Domain-specific R-D-E schema specifications and validation CLI."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

QualityDirection = Literal["higher_is_better", "lower_is_better"]
DistortionTransform = Literal[
    "negate_quality",
    "identity",
    "invert",
    "lower_is_better",
]

VALID_QUALITY_DIRECTIONS = {"higher_is_better", "lower_is_better"}
VALID_DISTORTION_TRANSFORMS = {
    "negate_quality",
    "identity",
    "invert",
    "lower_is_better",
}


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    item_id_columns: list[str]
    dataset_column: str
    codec_column: str
    config_column: str
    rate_column: str
    quality_column: str
    energy_column: str
    time_column: str | None
    rate_unit: str
    quality_metric: str
    quality_direction: QualityDirection
    energy_unit: str
    default_quality_floor: float | None
    default_near_quality_floor: float | None
    distortion_transform: DistortionTransform
    notes: str | None


BUILTIN_DOMAIN_SPECS: dict[str, DomainSpec] = {
    "image_ssimulacra2": DomainSpec(
        domain="image",
        item_id_columns=["image_id"],
        dataset_column="dataset",
        codec_column="codec",
        config_column="config",
        rate_column="bpp",
        quality_column="ssimulacra2",
        energy_column="energy_per_image_j",
        time_column=None,
        rate_unit="bpp",
        quality_metric="SSIMULACRA2",
        quality_direction="higher_is_better",
        energy_unit="J/image",
        default_quality_floor=80.0,
        default_near_quality_floor=60.0,
        distortion_transform="negate_quality",
        notes="Default image-domain schema for SSIMULACRA2-based R-D-E CSVs.",
    ),
    "image_psnr": DomainSpec(
        domain="image",
        item_id_columns=["image_id"],
        dataset_column="dataset",
        codec_column="codec",
        config_column="config",
        rate_column="bpp",
        quality_column="psnr",
        energy_column="energy_per_image_j",
        time_column=None,
        rate_unit="bpp",
        quality_metric="PSNR",
        quality_direction="higher_is_better",
        energy_unit="J/image",
        default_quality_floor=30.0,
        default_near_quality_floor=None,
        distortion_transform="negate_quality",
        notes="Image-domain schema for PSNR-oriented analyses.",
    ),
    "video_vmaf": DomainSpec(
        domain="video",
        item_id_columns=["sequence"],
        dataset_column="dataset",
        codec_column="codec",
        config_column="param",
        rate_column="bitrate_kbps",
        quality_column="vmaf",
        energy_column="energy_kj_per_sequence",
        time_column=None,
        rate_unit="kbps",
        quality_metric="VMAF",
        quality_direction="higher_is_better",
        energy_unit="kJ/sequence",
        default_quality_floor=None,
        default_near_quality_floor=None,
        distortion_transform="negate_quality",
        notes="Video-domain schema for VMAF and sequence-level energy.",
    ),
    "video_psnr_y": DomainSpec(
        domain="video",
        item_id_columns=["sequence"],
        dataset_column="dataset",
        codec_column="codec",
        config_column="param",
        rate_column="bitrate_kbps",
        quality_column="psnr_y",
        energy_column="energy_j_per_frame",
        time_column=None,
        rate_unit="kbps",
        quality_metric="PSNR-Y",
        quality_direction="higher_is_better",
        energy_unit="J/frame",
        default_quality_floor=None,
        default_near_quality_floor=None,
        distortion_transform="negate_quality",
        notes="Video-domain schema for luma PSNR and frame-level energy.",
    ),
    "audio_visqol": DomainSpec(
        domain="audio",
        item_id_columns=["item_id"],
        dataset_column="dataset",
        codec_column="codec",
        config_column="param",
        rate_column="bitrate_kbps",
        quality_column="visqol",
        energy_column="energy_j_per_second",
        time_column=None,
        rate_unit="kbps",
        quality_metric="ViSQOL",
        quality_direction="higher_is_better",
        energy_unit="J/s",
        default_quality_floor=None,
        default_near_quality_floor=None,
        distortion_transform="negate_quality",
        notes="Audio-domain schema for ViSQOL and steady-state energy.",
    ),
    "audio_fad": DomainSpec(
        domain="audio",
        item_id_columns=["item_id"],
        dataset_column="dataset",
        codec_column="codec",
        config_column="param",
        rate_column="bitrate_kbps",
        quality_column="fad",
        energy_column="energy_j_per_second",
        time_column=None,
        rate_unit="kbps",
        quality_metric="FAD",
        quality_direction="lower_is_better",
        energy_unit="J/s",
        default_quality_floor=None,
        default_near_quality_floor=None,
        distortion_transform="lower_is_better",
        notes="Audio-domain schema for Frechet Audio Distance; lower is better.",
    ),
}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _numeric_value(value: Any) -> float:
    if value is None:
        raise ValueError("missing")

    text = str(value).strip()
    if text == "":
        raise ValueError("missing")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    text = text.replace("\u2212", "-").replace("\u00e2\u02c6\u2019", "-")

    match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"not_numeric:{value}")
    return float(match.group(0))


def _coerce_domain_spec(data: DomainSpec | Mapping[str, Any]) -> DomainSpec:
    if isinstance(data, DomainSpec):
        return data

    allowed = set(DomainSpec.__dataclass_fields__)
    payload = {key: value for key, value in data.items() if key in allowed}
    missing = sorted(allowed - set(payload))
    if missing:
        raise ValueError("missing_domain_spec_fields: " + ", ".join(missing))

    if payload.get("item_id_columns") is None:
        payload["item_id_columns"] = []
    else:
        payload["item_id_columns"] = list(payload["item_id_columns"])

    for key in ("default_quality_floor", "default_near_quality_floor"):
        if payload.get(key) is not None:
            payload[key] = float(payload[key])

    return DomainSpec(**payload)  # type: ignore[arg-type]


def domain_spec_to_dict(spec: DomainSpec) -> dict[str, Any]:
    return asdict(spec)


def validate_domain_spec(spec: DomainSpec | Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        normalized = _coerce_domain_spec(spec)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": warnings,
            "normalized_spec": None,
        }

    if normalized.quality_direction not in VALID_QUALITY_DIRECTIONS:
        errors.append(f"invalid_quality_direction:{normalized.quality_direction}")
    if normalized.distortion_transform not in VALID_DISTORTION_TRANSFORMS:
        errors.append(f"invalid_distortion_transform:{normalized.distortion_transform}")

    required_text_fields = [
        "domain",
        "dataset_column",
        "codec_column",
        "config_column",
        "rate_column",
        "quality_column",
        "energy_column",
        "rate_unit",
        "quality_metric",
        "energy_unit",
    ]
    spec_dict = domain_spec_to_dict(normalized)
    for field_name in required_text_fields:
        if str(spec_dict.get(field_name) or "").strip() == "":
            errors.append(f"empty_required_field:{field_name}")

    if not isinstance(normalized.item_id_columns, list):
        errors.append("item_id_columns_must_be_list")
    else:
        for index, column in enumerate(normalized.item_id_columns):
            if str(column or "").strip() == "":
                errors.append(f"empty_item_id_column:{index}")

    if (
        normalized.quality_direction == "lower_is_better"
        and normalized.distortion_transform == "negate_quality"
    ):
        warnings.append("lower_is_better_quality_with_negate_quality_transform")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_spec": domain_spec_to_dict(normalized),
    }


def load_domain_spec_json(path: str | Path) -> DomainSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _coerce_domain_spec(data)


def resolve_domain_spec(name_or_path: str | Path) -> DomainSpec:
    name = str(name_or_path)
    if name in BUILTIN_DOMAIN_SPECS:
        return BUILTIN_DOMAIN_SPECS[name]

    path = Path(name)
    if path.exists():
        return load_domain_spec_json(path)

    raise ValueError(
        f"Unknown domain spec {name!r}. "
        f"Builtins: {', '.join(sorted(BUILTIN_DOMAIN_SPECS))}"
    )


def resolve_builtin_domain_spec_for_metric(
    domain: str | None,
    quality_metric: str | None,
) -> DomainSpec | None:
    if not domain or not quality_metric:
        return None

    metric_norm = _normalize_token(quality_metric)
    domain_norm = _normalize_token(domain)

    if quality_metric in BUILTIN_DOMAIN_SPECS:
        spec = BUILTIN_DOMAIN_SPECS[quality_metric]
        return spec if _normalize_token(spec.domain) == domain_norm else None

    for spec in BUILTIN_DOMAIN_SPECS.values():
        if _normalize_token(spec.domain) != domain_norm:
            continue
        if metric_norm in {
            _normalize_token(spec.quality_metric),
            _normalize_token(spec.quality_column),
        }:
            return spec

    return None


def _rows_from_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _columns_from_dataframe(df: Any) -> list[str]:
    if hasattr(df, "columns"):
        return [str(column) for column in list(df.columns)]
    if isinstance(df, list):
        columns: list[str] = []
        seen: set[str] = set()
        for row in df:
            if not isinstance(row, Mapping):
                continue
            for column in row.keys():
                column_text = str(column)
                if column_text not in seen:
                    columns.append(column_text)
                    seen.add(column_text)
        return columns
    raise TypeError("df must be a pandas-like object with columns or a list of row dicts")


def _num_rows(df: Any) -> int:
    try:
        return len(df)
    except TypeError:
        return 0


def _column_values(df: Any, column: str) -> list[Any]:
    if hasattr(df, "columns"):
        values = df[column]
        if hasattr(values, "tolist"):
            return list(values.tolist())
        return list(values)
    return [row.get(column) for row in df if isinstance(row, Mapping)]


def _numeric_summary(df: Any, column: str) -> dict[str, Any]:
    values = _column_values(df, column)
    valid = 0
    missing = 0
    invalid = 0
    examples: list[dict[str, Any]] = []

    for index, value in enumerate(values):
        try:
            _numeric_value(value)
            valid += 1
        except ValueError as exc:
            if str(exc) == "missing":
                missing += 1
                reason = "missing"
            else:
                invalid += 1
                reason = str(exc)
            if len(examples) < 5:
                examples.append({"row_index": index, "value": value, "reason": reason})

    total = len(values)
    return {
        "column": column,
        "total": total,
        "valid": valid,
        "missing": missing,
        "invalid": invalid,
        "all_valid": valid == total and missing == 0 and invalid == 0,
        "examples": examples,
    }


def validate_rde_dataframe_against_domain_spec(
    df: Any,
    spec: DomainSpec | Mapping[str, Any],
) -> dict[str, Any]:
    spec_report = validate_domain_spec(spec)
    errors = list(spec_report["errors"])
    warnings = list(spec_report["warnings"])
    normalized_dict = spec_report["normalized_spec"]

    if normalized_dict is None:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "normalized_spec": None,
            "detected_columns": [],
            "num_rows": _num_rows(df),
            "numeric_validity_summary": {},
        }

    normalized = _coerce_domain_spec(normalized_dict)
    detected_columns = _columns_from_dataframe(df)
    detected = set(detected_columns)

    required_columns = {
        "dataset_column": normalized.dataset_column,
        "codec_column": normalized.codec_column,
        "config_column": normalized.config_column,
        "rate_column": normalized.rate_column,
        "quality_column": normalized.quality_column,
        "energy_column": normalized.energy_column,
    }
    if normalized.time_column is not None:
        required_columns["time_column"] = normalized.time_column

    for role, column in required_columns.items():
        if column not in detected:
            errors.append(f"missing_column:{role}:{column}")

    for column in normalized.item_id_columns:
        if column not in detected:
            errors.append(f"missing_item_id_column:{column}")

    numeric_validity_summary: dict[str, Any] = {}
    for role, column in (
        ("rate", normalized.rate_column),
        ("quality", normalized.quality_column),
        ("energy", normalized.energy_column),
    ):
        if column not in detected:
            continue
        summary = _numeric_summary(df, column)
        numeric_validity_summary[role] = summary
        if not summary["all_valid"]:
            errors.append(
                f"non_numeric_{role}:{column}:"
                f"invalid={summary['invalid']}:missing={summary['missing']}"
            )

    if "domain" in detected:
        values = {
            str(value).strip().lower()
            for value in _column_values(df, "domain")
            if str(value).strip() != ""
        }
        expected = normalized.domain.strip().lower()
        mismatches = sorted(value for value in values if value != expected)
        if mismatches:
            errors.append(
                f"domain_mismatch:expected={expected}:found={','.join(mismatches)}"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_spec": domain_spec_to_dict(normalized),
        "detected_columns": detected_columns,
        "num_rows": _num_rows(df),
        "numeric_validity_summary": numeric_validity_summary,
    }


def _json_ready_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def _builtin_listing() -> dict[str, Any]:
    return {
        "valid": True,
        "builtins": sorted(BUILTIN_DOMAIN_SPECS),
        "domains": sorted({spec.domain for spec in BUILTIN_DOMAIN_SPECS.values()}),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate domain-specific R-D-E CSV schemas."
    )
    parser.add_argument("--list-builtins", action="store_true")
    parser.add_argument("--builtin", default=None, help="Built-in domain spec name.")
    parser.add_argument("--spec", default=None, help="Path to a DomainSpec JSON file.")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--csv", default=None, help="CSV file to validate.")
    parser.add_argument("--validate-csv", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_builtins:
        report = _builtin_listing()
        print(_json_ready_report(report))
        return report

    spec: DomainSpec | None = None
    if args.builtin:
        spec = resolve_domain_spec(args.builtin)
    if args.spec:
        if spec is not None:
            parser.error("use either --builtin or --spec, not both")
        spec = resolve_domain_spec(args.spec)

    if args.print_json:
        if spec is None:
            parser.error("--print-json requires --builtin or --spec")
        report = domain_spec_to_dict(spec)
        print(_json_ready_report(report))
        return report

    if args.validate_csv:
        if spec is None:
            parser.error("--validate-csv requires --builtin or --spec")
        if args.csv is None:
            parser.error("--validate-csv requires --csv")
        report = validate_rde_dataframe_against_domain_spec(_rows_from_csv(args.csv), spec)
        print(_json_ready_report(report))
        return report

    if args.validate:
        if spec is None:
            parser.error("--validate requires --builtin or --spec")
        report = validate_domain_spec(spec)
        print(_json_ready_report(report))
        return report

    parser.print_help()
    return {"valid": True, "errors": [], "warnings": []}


if __name__ == "__main__":
    main()
