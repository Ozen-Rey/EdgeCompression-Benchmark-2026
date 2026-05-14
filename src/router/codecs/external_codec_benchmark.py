"""Small external codec benchmark runner producing raw measurements only."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any

from src.router.codecs.external_codec_dry_run import dry_run_external_codec_spec
from src.router.codecs.external_codec_spec import (
    load_external_codec_spec,
    validate_external_codec_spec,
)


SCHEMA_VERSION = "external_codec_benchmark_v1"

CSV_COLUMNS = [
    "codec_id",
    "domain",
    "input_path",
    "input_id",
    "param_set_id",
    "param_json",
    "encode_success",
    "decode_success",
    "output_path",
    "reconstruction_path",
    "output_size_bytes",
    "encode_time_ms",
    "decode_time_ms",
    "rate_metric",
    "rate_value",
    "energy_j",
    "energy_provenance_tier",
    "error",
]


def run_external_codec_benchmark(
    spec_path: str | Path,
    *,
    input_paths: list[str | Path],
    out_dir: str | Path,
    report_out: str | Path | None = None,
    csv_out: str | Path | None = None,
    param_sets: list[dict[str, str]] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    spec_file = Path(spec_path)
    output_dir = Path(out_dir)
    errors: list[str] = []
    warnings: list[str] = []

    report = {
        "external_codec_benchmark": {
            "schema_version": SCHEMA_VERSION,
            "spec_path": str(spec_file),
            "codec_id": None,
            "domain": None,
            "valid_spec": False,
            "out_dir": str(output_dir),
            "csv_path": str(csv_out) if csv_out is not None else None,
            "num_inputs": len(input_paths),
            "num_param_sets": 0,
            "num_runs": 0,
            "num_successful_runs": 0,
            "rows": [],
            "success": False,
            "errors": errors,
            "warnings": warnings,
            "safety": {
                "shell_used": False,
                "output_confined_to_out_dir": False,
                "benchmark_dataset_auto_discovered": False,
                "router_candidate_registered": False,
            },
        }
    }
    body = report["external_codec_benchmark"]

    try:
        spec = load_external_codec_spec(spec_file)
    except Exception as exc:
        errors.append(f"spec_load_error:{exc}")
        _write_outputs(report, report_out, csv_out)
        return report

    body["codec_id"] = spec.get("codec_id")
    body["domain"] = spec.get("domain")
    validation = validate_external_codec_spec(spec)
    body["valid_spec"] = bool(validation.get("valid", False))
    if not body["valid_spec"]:
        errors.extend(validation.get("errors", []))
        warnings.extend(validation.get("warnings", []))
        _write_outputs(report, report_out, csv_out)
        return report

    if timeout_s <= 0:
        errors.append("timeout_must_be_positive")
        _write_outputs(report, report_out, csv_out)
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir.resolve()
    if not output_dir_resolved.exists() or not output_dir_resolved.is_dir():
        errors.append("out_dir_not_available")
        _write_outputs(report, report_out, csv_out)
        return report
    body["safety"]["output_confined_to_out_dir"] = True

    resolved_inputs = [Path(path) for path in input_paths]
    if not resolved_inputs:
        errors.append("no_inputs")
        _write_outputs(report, report_out, csv_out)
        return report

    if param_sets is None:
        param_sets = _default_param_grid(spec)
    param_sets = [_stringify_params(item) for item in param_sets]
    declared = _declared_parameters(spec)
    undeclared = sorted({
        name
        for params in param_sets
        for name in params
        if name not in declared
    })
    for name in undeclared:
        errors.append(f"undeclared_parameter:{name}")
    if errors:
        _write_outputs(report, report_out, csv_out)
        return report

    body["num_param_sets"] = len(param_sets)
    body["num_runs"] = len(resolved_inputs) * len(param_sets)

    rows: list[dict[str, Any]] = []
    for input_index, input_file in enumerate(resolved_inputs):
        input_id = _input_id(input_file, input_index)
        for param_index, params in enumerate(param_sets):
            param_set_id = f"p{param_index:03d}"
            run_dir = _confined_run_dir(
                output_dir_resolved,
                input_id=input_id,
                param_set_id=param_set_id,
            )
            row = _run_one(
                spec_file,
                spec,
                input_file=input_file,
                input_id=input_id,
                param_set_id=param_set_id,
                params=params,
                run_dir=run_dir,
                timeout_s=timeout_s,
            )
            rows.append(row)

    body["rows"] = rows
    body["num_successful_runs"] = sum(
        1
        for row in rows
        if row["encode_success"] and (row["decode_success"] or row["decode_success"] == "")
    )
    body["success"] = body["num_successful_runs"] == body["num_runs"] and not errors
    _write_outputs(report, report_out, csv_out)
    return report


def _declared_parameters(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        str(parameter.get("name")): parameter
        for parameter in spec.get("parameters", [])
        if isinstance(parameter, dict) and parameter.get("name") is not None
    }


def _default_param_grid(spec: dict[str, Any]) -> list[dict[str, str]]:
    declared = _declared_parameters(spec)
    if not declared:
        return [{}]

    names = sorted(declared)
    value_lists = []
    for name in names:
        values = declared[name].get("values", [])
        value_lists.append([str(value) for value in values])

    return [
        dict(zip(names, values))
        for values in itertools.product(*value_lists)
    ]


def _stringify_params(params: dict[str, Any]) -> dict[str, str]:
    return {str(name): str(value) for name, value in params.items()}


def _input_id(input_file: Path, index: int) -> str:
    stem = input_file.stem or f"input_{index}"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem).strip("._")
    return safe or f"input_{index}"


def _confined_run_dir(
    out_dir: Path,
    *,
    input_id: str,
    param_set_id: str,
) -> Path:
    path = (out_dir / input_id / param_set_id).resolve()
    path.relative_to(out_dir)
    return path


def _run_one(
    spec_file: Path,
    spec: dict[str, Any],
    *,
    input_file: Path,
    input_id: str,
    param_set_id: str,
    params: dict[str, str],
    run_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    result = dry_run_external_codec_spec(
        spec_file,
        input_path=input_file,
        out_dir=run_dir,
        params=params,
        timeout_s=timeout_s,
    )["external_codec_dry_run"]

    encode = result["encode"]
    decode = result["decode"]
    encode_success = bool(encode.get("success", False))
    decode_success: bool | str
    if decode.get("requested", False):
        decode_success = bool(decode.get("success", False))
    else:
        decode_success = ""

    output_size = encode.get("output_size_bytes")
    error = ";".join(result.get("errors", []))
    return {
        "codec_id": spec.get("codec_id"),
        "domain": spec.get("domain"),
        "input_path": str(input_file),
        "input_id": input_id,
        "param_set_id": param_set_id,
        "param_json": json.dumps(params, sort_keys=True),
        "encode_success": encode_success,
        "decode_success": decode_success,
        "output_path": encode.get("output_path") or "",
        "reconstruction_path": decode.get("reconstruction_path") or "",
        "output_size_bytes": output_size if output_size is not None else "",
        "encode_time_ms": encode.get("time_ms") if encode.get("time_ms") is not None else "",
        "decode_time_ms": decode.get("time_ms") if decode.get("time_ms") is not None else "",
        "rate_metric": _rate_metric(spec),
        "rate_value": output_size if encode_success and output_size is not None else "",
        "energy_j": "",
        "energy_provenance_tier": "unknown",
        "error": error,
    }


def _rate_metric(spec: dict[str, Any]) -> str:
    rate = spec.get("rate", {})
    if isinstance(rate, dict):
        return str(rate.get("metric") or "output_size_bytes")
    return "output_size_bytes"


def _write_outputs(
    report: dict[str, Any],
    report_out: str | Path | None,
    csv_out: str | Path | None,
) -> None:
    body = report["external_codec_benchmark"]
    rows = body.get("rows", [])
    if csv_out is not None:
        csv_path = Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
        body["csv_path"] = str(csv_path)

    if report_out is not None:
        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _parse_param_sets(values: list[str]) -> list[dict[str, str]] | None:
    if not values:
        return None
    parsed: list[dict[str, str]] = []
    for value in values:
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("--param-set must be a JSON object")
        parsed.append(_stringify_params(data))
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small raw benchmark for a validated external codec spec."
    )
    parser.add_argument("--spec", required=True, help="External codec spec JSON.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input file. Repeat for a small explicit dataset.",
    )
    parser.add_argument("--out-dir", required=True, help="Controlled output directory.")
    parser.add_argument(
        "--param-set",
        action="append",
        default=[],
        help="Parameter set as JSON object. Repeat for an explicit grid.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Timeout per encode/decode command.",
    )
    parser.add_argument("--out", required=True, help="Benchmark report JSON path.")
    parser.add_argument("--csv", required=True, help="Raw measurements CSV path.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = run_external_codec_benchmark(
        args.spec,
        input_paths=[Path(item) for item in args.input],
        out_dir=args.out_dir,
        report_out=args.out,
        csv_out=args.csv,
        param_sets=_parse_param_sets(args.param_set),
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
