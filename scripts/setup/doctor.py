"""Read-only router environment doctor.

The doctor inspects the Python/router development environment and optional
benchmark execution tools. It never installs packages, downloads artifacts, or
changes repository files except when the caller explicitly requests a JSON
report path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLI_MODULES = [
    "src.router.rde_router",
    "src.router.core.domain_spec",
    "src.router.core.dataset_manifest",
    "src.router.core.dataset_ingestion",
    "src.router.core.codec_onboarding",
    "src.router.analysis.audio_video_policy_validation",
]
DEPENDENCIES = [
    "pytest",
]
OPTIONAL_EXTERNAL_TOOLS = [
    "ffmpeg",
    "ffprobe",
    "cjxl",
    "djxl",
    "nvidia-smi",
]


def _run_command(command: List[str], timeout_s: float = 10.0) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {timeout_s}s",
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _first_output_line(result: Mapping[str, Any]) -> str | None:
    text = str(result.get("stdout") or result.get("stderr") or "").strip()
    if not text:
        return None
    return text.splitlines()[0]


def _python_info() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
    }


def _venv_info() -> Dict[str, Any]:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return {
        "active": in_venv,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
    }


def _dependency_report(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        entry: Dict[str, Any] = {"available": spec is not None}
        try:
            entry["version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            entry["version"] = None
        report[name] = entry
    return report


def _router_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "import_ok": False,
        "version": None,
        "cli_help": {},
    }

    try:
        from src.router.version import ROUTER_VERSION

        report["import_ok"] = True
        report["version"] = ROUTER_VERSION
    except Exception as exc:  # pragma: no cover - defensive reporting
        report["import_error"] = repr(exc)
        return report

    for module in CLI_MODULES:
        result = _run_command([sys.executable, "-m", module, "--help"])
        report["cli_help"][module] = {
            "ok": result["ok"],
            "returncode": result["returncode"],
            "summary": _first_output_line(result),
        }
    return report


def _external_tool_report(names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for name in names:
        path = shutil.which(name)
        entry: Dict[str, Any] = {
            "available": path is not None,
            "path": path,
            "optional_for_benchmark_or_execution": True,
        }
        if path:
            version_result = _run_command([path, "--version"], timeout_s=5.0)
            entry["version_probe_ok"] = version_result["ok"]
            entry["version_summary"] = _first_output_line(version_result)
        report[name] = entry
    return report


def build_report() -> Dict[str, Any]:
    router = _router_report()
    warnings: List[str] = []
    errors: List[str] = []

    if not router.get("import_ok"):
        errors.append("Router import failed.")
    for module, result in router.get("cli_help", {}).items():
        if not result.get("ok"):
            warnings.append(f"CLI help failed for {module}.")

    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": _python_info(),
        "venv": _venv_info(),
        "router": router,
        "python_dependencies": _dependency_report(DEPENDENCIES),
        "optional_external_tools": _external_tool_report(OPTIONAL_EXTERNAL_TOOLS),
        "notes": [
            "External tools are optional and only needed for benchmark/execution workflows.",
            "Datasets, checkpoints and benchmark outputs are not installed by setup.",
        ],
        "warnings": warnings,
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only doctor for the router development environment."
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        help="Write the environment doctor report as JSON.",
    )
    return parser


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = build_report()

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report_out:
        args.report_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
