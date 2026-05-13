"""Controlled probe for declarative external codec specifications."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from .codec_fingerprints import _sha256_file
    from .external_codec_spec import (
        load_external_codec_spec,
        validate_external_codec_spec,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from codec_fingerprints import _sha256_file
    from external_codec_spec import (
        load_external_codec_spec,
        validate_external_codec_spec,
    )


SCHEMA_VERSION = "external_codec_probe_v1"


def probe_external_codec_spec(
    spec_path: str | Path,
    *,
    timeout_s: float = 5.0,
    run_version_probe: bool = True,
) -> dict[str, Any]:
    spec_file = Path(spec_path)
    errors: list[str] = []
    warnings: list[str] = []

    try:
        spec = load_external_codec_spec(spec_file)
    except Exception as exc:
        errors.append(f"spec_load_error:{exc}")
        return {"external_codec_probe": _base_report(
            spec_path=spec_file,
            valid_spec=False,
            errors=errors,
            warnings=warnings,
        )}

    validation = validate_external_codec_spec(spec)
    if not validation.get("valid", False):
        return {"external_codec_probe": _base_report(
            spec_path=spec_file,
            valid_spec=False,
            codec_id=spec.get("codec_id"),
            domain=spec.get("domain"),
            runtime_type=_runtime_type(spec),
            errors=list(validation.get("errors", [])),
            warnings=list(validation.get("warnings", [])),
        )}

    report = _base_report(
        spec_path=spec_file,
        valid_spec=True,
        codec_id=spec.get("codec_id"),
        domain=spec.get("domain"),
        runtime_type=_runtime_type(spec),
        errors=errors,
        warnings=warnings,
    )

    runtime = spec.get("runtime", {})
    runtime_type = runtime.get("type") if isinstance(runtime, dict) else None
    if runtime_type == "external_command":
        _probe_external_command(
            report,
            spec=spec,
            spec_file=spec_file,
            timeout_s=timeout_s,
            run_version_probe=run_version_probe,
        )
    elif runtime_type == "python_module":
        _probe_python_module(report, spec)
    else:
        report["errors"].append("unsupported_runtime_type")
        report["available"] = False

    return {"external_codec_probe": report}


def _base_report(
    *,
    spec_path: Path,
    valid_spec: bool,
    errors: list[str],
    warnings: list[str],
    codec_id: Any = None,
    domain: Any = None,
    runtime_type: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_path": str(spec_path),
        "valid_spec": valid_spec,
        "codec_id": codec_id,
        "domain": domain,
        "runtime_type": runtime_type,
        "executable_path": None,
        "executable_exists": None,
        "binary_sha256": None,
        "version_probe_executed": False,
        "version_string": None,
        "version_returncode": None,
        "available": False,
        "errors": errors,
        "warnings": warnings,
        "safety": {
            "shell_used": False,
            "encode_executed": False,
            "decode_executed": False,
            "benchmark_executed": False,
        },
    }


def _runtime_type(spec: dict[str, Any]) -> Any:
    runtime = spec.get("runtime")
    if not isinstance(runtime, dict):
        return None
    return runtime.get("type")


def _probe_external_command(
    report: dict[str, Any],
    *,
    spec: dict[str, Any],
    spec_file: Path,
    timeout_s: float,
    run_version_probe: bool,
) -> None:
    runtime = spec.get("runtime", {})
    executable = runtime.get("executable")
    if not isinstance(executable, str) or not executable.strip():
        report["errors"].append("runtime_executable_required")
        report["executable_exists"] = False
        report["available"] = False
        return

    executable_path = _path_from_spec(executable, spec_file)
    report["executable_path"] = str(executable_path)
    executable_exists = executable_path.exists() and executable_path.is_file()
    report["executable_exists"] = executable_exists
    if not executable_exists:
        report["errors"].append("executable_not_found")
        report["available"] = False
        return

    report["binary_sha256"] = _sha256_file(executable_path)

    working_dir = _working_dir(spec, spec_file, report)
    if working_dir is False:
        report["available"] = False
        return

    if run_version_probe:
        command = _version_probe_command(spec, executable_path, report)
        if command is None:
            if report["errors"]:
                report["available"] = False
                return
            report["warnings"].append("version_probe_not_declared")
        else:
            _run_version_probe(
                report,
                command=command,
                timeout_s=timeout_s,
                working_dir=working_dir,
            )
            if report["errors"]:
                report["available"] = False
                return
    else:
        report["warnings"].append("version_probe_skipped")

    report["available"] = True


def _probe_python_module(report: dict[str, Any], spec: dict[str, Any]) -> None:
    runtime = spec.get("runtime", {})
    module = runtime.get("module") if isinstance(runtime, dict) else None
    if not isinstance(module, str) or not module.strip():
        report["errors"].append("runtime_module_required")
        report["available"] = False
        return

    report["available"] = "unknown"
    report["warnings"].append("python_module_not_imported")


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


def _version_probe_command(
    spec: dict[str, Any],
    executable_path: Path,
    report: dict[str, Any],
) -> list[str] | None:
    version_probe = spec.get("version_probe")
    if not isinstance(version_probe, dict):
        report["errors"].append("version_probe_must_be_object")
        return None

    command = version_probe.get("command")
    if command is None:
        command = version_probe.get("command_template")
    if command is None:
        return None
    if isinstance(command, str):
        report["errors"].append("version_probe_command_must_be_argv_list")
        return None
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        report["errors"].append("version_probe_command_must_be_argv_list")
        return None

    executable = str(executable_path)
    return [
        item.replace("{executable}", executable).replace("{binary}", executable)
        for item in command
    ]


def _run_version_probe(
    report: dict[str, Any],
    *,
    command: list[str],
    timeout_s: float,
    working_dir: Path | None,
) -> None:
    report["version_probe_executed"] = True
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
        report["errors"].append("version_probe_timeout")
        return
    except OSError as exc:
        report["errors"].append(f"version_probe_os_error:{exc}")
        return

    report["version_returncode"] = result.returncode
    output = (result.stdout or result.stderr or "").strip()
    if output:
        report["version_string"] = output.splitlines()[0].strip()
    if result.returncode != 0:
        report["errors"].append("version_probe_nonzero_returncode")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe an external codec spec without encode/decode execution."
    )
    parser.add_argument("--spec", required=True, help="External codec spec JSON.")
    parser.add_argument("--out", required=True, help="Probe report JSON output path.")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=5.0,
        help="Timeout in seconds for the version probe.",
    )
    parser.add_argument(
        "--no-version-probe",
        action="store_true",
        help="Only check executable existence and SHA256.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")

    report = probe_external_codec_spec(
        args.spec,
        timeout_s=args.timeout_s,
        run_version_probe=not args.no_version_probe,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
