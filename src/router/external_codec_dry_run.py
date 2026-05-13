"""Controlled dry-run contract validation for external codec specifications."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from .external_codec_spec import (
        load_external_codec_spec,
        validate_external_codec_spec,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from external_codec_spec import (
        load_external_codec_spec,
        validate_external_codec_spec,
    )


SCHEMA_VERSION = "external_codec_dry_run_v1"
PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def dry_run_external_codec_spec(
    spec_path: str | Path,
    *,
    input_path: str | Path,
    out_dir: str | Path,
    params: dict[str, str] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    spec_file = Path(spec_path)
    input_file = Path(input_path)
    output_dir = Path(out_dir)
    params = params or {}
    errors: list[str] = []
    warnings: list[str] = []

    report = _base_report(
        spec_path=spec_file,
        valid_spec=False,
        errors=errors,
        warnings=warnings,
    )

    try:
        spec = load_external_codec_spec(spec_file)
    except Exception as exc:
        report["errors"].append(f"spec_load_error:{exc}")
        return {"external_codec_dry_run": report}

    validation = validate_external_codec_spec(spec)
    report["codec_id"] = spec.get("codec_id")
    report["domain"] = spec.get("domain")
    report["valid_spec"] = bool(validation.get("valid", False))
    if not report["valid_spec"]:
        report["errors"].extend(validation.get("errors", []))
        report["warnings"].extend(validation.get("warnings", []))
        return {"external_codec_dry_run": report}

    if timeout_s <= 0:
        report["errors"].append("timeout_must_be_positive")
        return {"external_codec_dry_run": report}

    if not input_file.exists() or not input_file.is_file():
        report["errors"].append("input_not_found")
        return {"external_codec_dry_run": report}

    runtime = spec.get("runtime", {})
    if not isinstance(runtime, dict) or runtime.get("type") != "external_command":
        report["errors"].append("dry_run_requires_external_command_runtime")
        return {"external_codec_dry_run": report}

    executable = runtime.get("executable")
    if not isinstance(executable, str) or not executable.strip():
        report["errors"].append("runtime_executable_required")
        return {"external_codec_dry_run": report}

    executable_path = _path_from_spec(executable, spec_file)
    report["executable_checked"] = True
    if not executable_path.exists() or not executable_path.is_file():
        report["errors"].append("executable_not_found")
        return {"external_codec_dry_run": report}

    working_dir = _working_dir(spec, spec_file, report)
    if working_dir is False:
        return {"external_codec_dry_run": report}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir.resolve()
    if not output_dir_resolved.exists() or not output_dir_resolved.is_dir():
        report["errors"].append("out_dir_not_available")
        return {"external_codec_dry_run": report}

    codec_id = str(spec.get("codec_id"))
    encoded_path = _encoded_output_path(spec, output_dir_resolved, codec_id, report)
    if encoded_path is None:
        return {"external_codec_dry_run": report}

    declared_params = _declared_parameters(spec)
    _validate_params(params, declared_params, report)
    if report["errors"]:
        return {"external_codec_dry_run": report}

    encode_command = _build_command(
        spec.get("encode"),
        label="encode",
        executable_path=executable_path,
        input_path=input_file,
        output_path=encoded_path,
        params=params,
        declared_params=declared_params,
        report=report,
    )
    if encode_command is None:
        return {"external_codec_dry_run": report}

    report["encode"]["command_redacted_or_argv"] = encode_command
    _run_step(
        report,
        step="encode",
        command=encode_command,
        timeout_s=timeout_s,
        working_dir=working_dir,
    )
    _validate_encoded_output(spec, encoded_path, report)

    decode = spec.get("decode", {})
    decode_requested = bool(isinstance(decode, dict) and decode.get("available", True))
    report["decode"]["requested"] = decode_requested
    if decode_requested and report["encode"]["success"]:
        reconstruction_path = _reconstruction_output_path(
            spec,
            output_dir_resolved,
            codec_id,
            input_file,
            report,
        )
        if reconstruction_path is not None:
            decode_command = _build_command(
                decode,
                label="decode",
                executable_path=executable_path,
                input_path=encoded_path,
                output_path=reconstruction_path,
                params=params,
                declared_params=declared_params,
                report=report,
            )
            if decode_command is not None:
                report["decode"]["command_redacted_or_argv"] = decode_command
                _run_step(
                    report,
                    step="decode",
                    command=decode_command,
                    timeout_s=timeout_s,
                    working_dir=working_dir,
                )
                _validate_decode_output(decode, reconstruction_path, report)

    report["success"] = (
        not report["errors"]
        and report["encode"]["success"]
        and (not report["decode"]["requested"] or report["decode"]["success"])
    )
    return {"external_codec_dry_run": report}


def _base_report(
    *,
    spec_path: Path,
    valid_spec: bool,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_path": str(spec_path),
        "codec_id": None,
        "domain": None,
        "valid_spec": valid_spec,
        "executable_checked": False,
        "encode": {
            "executed": False,
            "command_redacted_or_argv": [],
            "returncode": None,
            "timeout": False,
            "output_path": None,
            "output_exists": False,
            "output_size_bytes": None,
            "output_extension_valid": False,
            "success": False,
        },
        "decode": {
            "requested": False,
            "executed": False,
            "command_redacted_or_argv": [],
            "returncode": None,
            "timeout": False,
            "reconstruction_path": None,
            "reconstruction_exists": False,
            "reconstruction_size_bytes": None,
            "success": False,
        },
        "success": False,
        "errors": errors,
        "warnings": warnings,
        "safety": {
            "shell_used": False,
            "benchmark_executed": False,
            "router_candidate_registered": False,
            "output_confined_to_out_dir": False,
        },
    }


def _path_from_spec(path_value: str, spec_file: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return spec_file.parent / path


def _working_dir(
    spec: dict[str, Any],
    spec_file: Path,
    report: dict[str, Any],
) -> Path | None | bool:
    runtime = spec.get("runtime", {})
    working_dir = runtime.get("working_dir") if isinstance(runtime, dict) else None
    if working_dir is None:
        return None
    if not isinstance(working_dir, str) or not working_dir.strip():
        report["errors"].append("working_dir_invalid")
        return False

    path = _path_from_spec(working_dir, spec_file)
    if not path.exists() or not path.is_dir():
        report["errors"].append("working_dir_not_found")
        return False
    return path


def _declared_parameters(spec: dict[str, Any]) -> dict[str, Any]:
    parameters = spec.get("parameters", [])
    return {
        str(parameter.get("name")): parameter
        for parameter in parameters
        if isinstance(parameter, dict) and parameter.get("name") is not None
    }


def _validate_params(
    params: dict[str, str],
    declared_params: dict[str, Any],
    report: dict[str, Any],
) -> None:
    for name in params:
        if name not in declared_params:
            report["errors"].append(f"undeclared_parameter:{name}")


def _safe_extension(extension: Any) -> str | None:
    if not isinstance(extension, str) or not extension.strip():
        return None
    if "/" in extension or "\\" in extension:
        return None
    if ".." in extension:
        return None
    return extension if extension.startswith(".") else f".{extension}"


def _safe_filename(filename: Any) -> str | None:
    if filename is None:
        return None
    if not isinstance(filename, str) or not filename.strip():
        return None
    path = Path(filename)
    if path.name != filename or path.is_absolute() or ".." in path.parts:
        return None
    return filename


def _encoded_output_path(
    spec: dict[str, Any],
    out_dir: Path,
    codec_id: str,
    report: dict[str, Any],
) -> Path | None:
    output = spec.get("output", {})
    extension = _safe_extension(output.get("extension") if isinstance(output, dict) else None)
    if extension is None:
        report["errors"].append("output_extension_unsafe")
        return None

    filename = _safe_filename(output.get("filename") if isinstance(output, dict) else None)
    if isinstance(output, dict) and output.get("filename") is not None and filename is None:
        report["errors"].append("output_filename_unsafe")
        return None

    path = out_dir / (filename or f"{codec_id}_encoded{extension}")
    return _confined_path(path, out_dir, report)


def _reconstruction_output_path(
    spec: dict[str, Any],
    out_dir: Path,
    codec_id: str,
    input_path: Path,
    report: dict[str, Any],
) -> Path | None:
    decode = spec.get("decode", {})
    extension = None
    filename = None
    if isinstance(decode, dict):
        extension = _safe_extension(decode.get("reconstruction_extension"))
        filename = _safe_filename(decode.get("reconstruction_filename"))
        if decode.get("reconstruction_filename") is not None and filename is None:
            report["errors"].append("decode_reconstruction_filename_unsafe")
            return None
    if extension is None:
        extension = input_path.suffix or ".raw"

    path = out_dir / (filename or f"{codec_id}_reconstruction{extension}")
    return _confined_path(path, out_dir, report)


def _confined_path(path: Path, out_dir: Path, report: dict[str, Any]) -> Path | None:
    resolved = path.resolve()
    try:
        resolved.relative_to(out_dir)
    except ValueError:
        report["errors"].append("output_path_outside_out_dir")
        report["safety"]["output_confined_to_out_dir"] = False
        return None
    report["safety"]["output_confined_to_out_dir"] = True
    return resolved


def _build_command(
    operation: Any,
    *,
    label: str,
    executable_path: Path,
    input_path: Path,
    output_path: Path,
    params: dict[str, str],
    declared_params: dict[str, Any],
    report: dict[str, Any],
) -> list[str] | None:
    if not isinstance(operation, dict):
        report["errors"].append(f"{label}_must_be_object")
        return None

    template = operation.get("command_template")
    if isinstance(template, str) or not isinstance(template, list):
        report["errors"].append(f"{label}_command_template_must_be_argv_list")
        return None
    if not all(isinstance(item, str) for item in template):
        report["errors"].append(f"{label}_command_template_items_must_be_strings")
        return None

    values = {
        "executable": str(executable_path),
        "binary": str(executable_path),
        "input": str(input_path),
        "output": str(output_path),
    }
    values.update(params)

    command: list[str] = []
    unresolved: list[str] = []
    for item in template:
        rendered = item
        for placeholder in PLACEHOLDER_RE.findall(item):
            if placeholder in values:
                rendered = rendered.replace(f"{{{placeholder}}}", values[placeholder])
            elif placeholder in declared_params:
                unresolved.append(placeholder)
            else:
                unresolved.append(placeholder)
        command.append(rendered)

    if unresolved:
        for placeholder in sorted(set(unresolved)):
            report["errors"].append(f"unresolved_placeholder:{placeholder}")
        return None

    return command


def _run_step(
    report: dict[str, Any],
    *,
    step: str,
    command: list[str],
    timeout_s: float,
    working_dir: Path | None,
) -> None:
    step_report = report[step]
    step_report["executed"] = True
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
            cwd=str(working_dir) if working_dir is not None else None,
        )
    except subprocess.TimeoutExpired:
        step_report["timeout"] = True
        report["errors"].append(f"{step}_timeout")
        return
    except OSError as exc:
        report["errors"].append(f"{step}_os_error:{exc}")
        return

    step_report["returncode"] = result.returncode
    if result.returncode != 0:
        report["errors"].append(f"{step}_nonzero_returncode")


def _validate_encoded_output(
    spec: dict[str, Any],
    output_path: Path,
    report: dict[str, Any],
) -> None:
    output_report = report["encode"]
    output_report["output_path"] = str(output_path)
    output_report["output_exists"] = output_path.exists()
    output_report["output_extension_valid"] = _extension_matches(
        output_path,
        spec.get("output", {}).get("extension"),
    )
    if output_path.exists():
        output_report["output_size_bytes"] = output_path.stat().st_size
    else:
        report["errors"].append("encode_output_missing")

    if not output_report["output_extension_valid"]:
        report["errors"].append("encode_output_extension_invalid")

    must_be_nonempty = bool(spec.get("output", {}).get("must_be_nonempty", False))
    if must_be_nonempty and output_report["output_size_bytes"] == 0:
        report["errors"].append("encode_output_empty")

    output_report["success"] = (
        output_report["executed"]
        and output_report["returncode"] == 0
        and not output_report["timeout"]
        and output_report["output_exists"]
        and output_report["output_extension_valid"]
        and (not must_be_nonempty or output_report["output_size_bytes"] > 0)
    )


def _validate_decode_output(
    decode: dict[str, Any],
    output_path: Path,
    report: dict[str, Any],
) -> None:
    output_report = report["decode"]
    output_report["reconstruction_path"] = str(output_path)
    output_report["reconstruction_exists"] = output_path.exists()
    if output_path.exists():
        output_report["reconstruction_size_bytes"] = output_path.stat().st_size
    else:
        report["errors"].append("decode_output_missing")

    must_be_nonempty = bool(decode.get("must_be_nonempty", False))
    if must_be_nonempty and output_report["reconstruction_size_bytes"] == 0:
        report["errors"].append("decode_output_empty")

    output_report["success"] = (
        output_report["executed"]
        and output_report["returncode"] == 0
        and not output_report["timeout"]
        and output_report["reconstruction_exists"]
        and (not must_be_nonempty or output_report["reconstruction_size_bytes"] > 0)
    )


def _extension_matches(path: Path, extension: Any) -> bool:
    expected = _safe_extension(extension)
    if expected is None:
        return False
    return path.suffix.lower() == expected.lower()


def _parse_params(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --param value {value!r}; expected name=value")
        name, param_value = value.split("=", 1)
        if not name:
            raise ValueError("Invalid --param value with empty name.")
        parsed[name] = param_value
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controlled one-input dry-run for an external codec spec."
    )
    parser.add_argument("--spec", required=True, help="External codec spec JSON.")
    parser.add_argument("--input", required=True, help="Single input file.")
    parser.add_argument("--out-dir", required=True, help="Controlled output directory.")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Declared parameter override as name=value.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Encode/decode timeout in seconds.",
    )
    parser.add_argument("--out", required=True, help="Dry-run report JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    params = _parse_params(args.param)
    report = dry_run_external_codec_spec(
        args.spec,
        input_path=args.input,
        out_dir=args.out_dir,
        params=params,
        timeout_s=args.timeout_s,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
