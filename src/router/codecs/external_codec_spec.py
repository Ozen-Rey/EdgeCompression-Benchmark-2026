"""Declarative external codec specification validation."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


SUPPORTED_DOMAINS = {"image", "video", "audio"}
SUPPORTED_FAMILIES = {"classical", "neural", "hybrid", "unknown"}
SUPPORTED_RUNTIME_TYPES = {"external_command", "python_module"}
CODEC_ID_RE = re.compile(r"^[a-z0-9_-]+$")


REQUIRED_TOP_LEVEL_FIELDS = [
    "schema_version",
    "codec_id",
    "display_name",
    "domain",
    "family",
    "runtime",
    "version_probe",
    "encode",
    "decode",
    "parameters",
    "output",
    "rate",
    "quality",
    "measurement",
    "requirements",
    "security",
]


def load_external_codec_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("External codec spec must be a JSON object.")

    return data


def validate_external_codec_spec(spec: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(spec, dict):
        return {
            "valid": False,
            "errors": ["spec_must_be_object"],
            "warnings": [],
        }

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in spec:
            errors.append(f"missing_required_field:{field}")

    codec_id = spec.get("codec_id")
    if not isinstance(codec_id, str) or not CODEC_ID_RE.fullmatch(codec_id):
        errors.append("invalid_codec_id")

    domain = spec.get("domain")
    if domain not in SUPPORTED_DOMAINS:
        errors.append("invalid_domain")

    family = spec.get("family")
    if family is not None and family not in SUPPORTED_FAMILIES:
        errors.append("invalid_family")

    runtime = spec.get("runtime")
    runtime_type = None
    if not isinstance(runtime, dict):
        errors.append("runtime_must_be_object")
    else:
        runtime_type = runtime.get("type")
        if runtime_type not in SUPPORTED_RUNTIME_TYPES:
            errors.append("unsupported_runtime_type")
        max_runtime = runtime.get("max_runtime_seconds")
        if max_runtime is not None:
            if not isinstance(max_runtime, (int, float)) or max_runtime <= 0:
                errors.append("invalid_max_runtime_seconds")

    _validate_operation_template(spec.get("encode"), "encode", errors)
    _validate_operation_template(spec.get("decode"), "decode", errors)
    _validate_parameters(spec.get("parameters"), errors)
    _validate_output(spec.get("output"), errors)
    _validate_security(spec.get("security"), errors)

    normalized_spec = deepcopy(spec)
    if isinstance(codec_id, str):
        normalized_spec["codec_id"] = codec_id.strip()
    if isinstance(runtime, dict) and runtime_type is not None:
        normalized_spec.setdefault("runtime", {})["type"] = runtime_type

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized_spec": normalized_spec if not errors else None,
    }


def _validate_operation_template(
    operation: Any,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(operation, dict):
        errors.append(f"{label}_must_be_object")
        return

    template = operation.get("command_template")
    if isinstance(template, str):
        errors.append(f"{label}_command_template_must_be_argv_list")
        return

    if not isinstance(template, list):
        errors.append(f"{label}_command_template_must_be_argv_list")
        return

    if not all(isinstance(item, str) for item in template):
        errors.append(f"{label}_command_template_items_must_be_strings")
        return

    joined = " ".join(template)
    if "{input}" not in joined:
        errors.append(f"{label}_command_template_missing_input_placeholder")
    if "{output}" not in joined:
        errors.append(f"{label}_command_template_missing_output_placeholder")


def _validate_parameters(parameters: Any, errors: list[str]) -> None:
    if not isinstance(parameters, list):
        errors.append("parameters_must_be_list")
        return

    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, dict):
            errors.append(f"parameter_{index}_must_be_object")
            continue

        for field in ("name", "type", "values"):
            if field not in parameter:
                errors.append(f"parameter_{index}_missing_{field}")

        values = parameter.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"parameter_{index}_values_must_be_nonempty_list")


def _validate_output(output: Any, errors: list[str]) -> None:
    if not isinstance(output, dict):
        errors.append("output_must_be_object")
        return

    extension = output.get("extension")
    if not isinstance(extension, str) or not extension.strip():
        errors.append("output_extension_required")


def _validate_security(security: Any, errors: list[str]) -> None:
    if security is None:
        return

    if not isinstance(security, dict):
        errors.append("security_must_be_object")
        return

    if security.get("allow_shell") is True:
        errors.append("security_allow_shell_must_be_false")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate external codec specification JSON without execution."
    )
    parser.add_argument("--spec", required=True, help="External codec spec JSON.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the spec and print a JSON validation report.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.validate:
        raise ValueError("--validate is required; this module does not execute codecs.")

    spec = load_external_codec_spec(args.spec)
    report = validate_external_codec_spec(spec)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
