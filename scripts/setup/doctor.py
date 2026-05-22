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
    {
        "package": "Pillow",
        "import_name": "PIL",
        "required": True,
        "required_for": "router CLI import path / content-aware image support",
    },
    {
        "package": "pytest",
        "import_name": "pytest",
        "required": False,
        "required_for": "router-only test subset",
    },
]
OPTIONAL_EXTERNAL_TOOLS = [
    "ffmpeg",
    "ffprobe",
    "cjxl",
    "djxl",
    "nvidia-smi",
]


def _platform_id() -> Dict[str, Any]:
    system = platform.system()
    info: Dict[str, Any] = {
        "system": system,
        "id": system.lower(),
        "id_like": [],
        "pretty_name": platform.platform(),
    }
    if system == "Linux":
        try:
            release = platform.freedesktop_os_release()
        except OSError:
            release = {}
        info.update(
            {
                "id": str(release.get("ID") or "linux").lower(),
                "id_like": [
                    value.lower()
                    for value in str(release.get("ID_LIKE") or "").split()
                    if value
                ],
                "pretty_name": release.get("PRETTY_NAME") or platform.platform(),
            }
        )
    elif system == "Darwin":
        info["id"] = "macos"
        info["pretty_name"] = f"macOS {platform.mac_ver()[0]}".strip()
    elif system == "Windows":
        info["id"] = "windows"
    return info


def _system_package_hints(platform_info: Mapping[str, Any]) -> List[str]:
    distro_id = str(platform_info.get("id") or "").lower()
    id_like = {str(value).lower() for value in platform_info.get("id_like", [])}

    if distro_id in {"arch", "manjaro", "endeavouros"} or "arch" in id_like:
        return ["sudo pacman -S --needed git python python-pip unzip"]
    if distro_id in {"ubuntu", "debian"} or {"ubuntu", "debian"} & id_like:
        return [
            "sudo apt update",
            "sudo apt install git python3 python3-pip python3-venv unzip",
        ]
    if distro_id == "fedora" or "fedora" in id_like:
        return ["sudo dnf install git python3 python3-pip unzip"]
    if distro_id == "macos":
        return [
            "Install Python 3 and Git via the official installers, or via Homebrew if you already use Homebrew."
        ]
    if distro_id == "windows":
        return [
            "Install Git and Python 3; use setup.ps1 or python scripts/setup/setup_router.py."
        ]
    return [
        "Install Git, Python 3, pip, venv support, and unzip using your OS package manager."
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


def _dependency_report(dependencies: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for dependency in dependencies:
        package = str(dependency["package"])
        import_name = str(dependency["import_name"])
        spec = importlib.util.find_spec(import_name)
        entry: Dict[str, Any] = {
            "import_name": import_name,
            "install_scope": "python_package",
            "installed_via": "pyproject.toml / pip",
            "required": bool(dependency.get("required", False)),
            "required_for": dependency.get("required_for"),
            "available": spec is not None,
        }
        try:
            entry["version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            entry["version"] = None
        report[package] = entry
    return report


def _bootstrap_prerequisite_report() -> Dict[str, Dict[str, Any]]:
    pip_probe = _run_command([sys.executable, "-m", "pip", "--version"], timeout_s=5.0)
    venv_spec = importlib.util.find_spec("venv")
    git_path = shutil.which("git")
    unzip_path = shutil.which("unzip")

    report: Dict[str, Dict[str, Any]] = {
        "git": {
            "available": git_path is not None,
            "path": git_path,
            "required": True,
            "required_for": "source checkout and version metadata",
            "install_scope": "system/bootstrap prerequisite",
            "managed_by_setup": False,
        },
        "python": {
            "available": True,
            "path": sys.executable,
            "version": platform.python_version(),
            "required": True,
            "required_for": "router runtime and setup helper",
            "install_scope": "system/bootstrap prerequisite",
            "managed_by_setup": False,
        },
        "pip": {
            "available": pip_probe["ok"],
            "path": sys.executable,
            "version_summary": _first_output_line(pip_probe),
            "required": True,
            "required_for": "installing project Python dependencies from pyproject.toml",
            "install_scope": "system/bootstrap prerequisite",
            "managed_by_setup": False,
        },
        "venv": {
            "available": venv_spec is not None,
            "path": None,
            "required": True,
            "required_for": "creating an isolated router environment",
            "install_scope": "system/bootstrap prerequisite",
            "managed_by_setup": False,
        },
        "unzip": {
            "available": unzip_path is not None,
            "path": unzip_path,
            "required": False,
            "required_for": "extracting source archives",
            "optional_for_source_archives": True,
            "install_scope": "system/bootstrap prerequisite",
            "managed_by_setup": False,
        },
    }

    if git_path:
        version_probe = _run_command([git_path, "--version"], timeout_s=5.0)
        report["git"]["version_summary"] = _first_output_line(version_probe)
    if unzip_path:
        version_probe = _run_command([unzip_path, "-v"], timeout_s=5.0)
        report["unzip"]["version_summary"] = _first_output_line(version_probe)

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
    python_dependencies = _dependency_report(DEPENDENCIES)
    platform_info = _platform_id()
    bootstrap_prerequisites = _bootstrap_prerequisite_report()
    warnings: List[str] = []
    errors: List[str] = []

    if not router.get("import_ok"):
        errors.append("Router import failed.")
    for module, result in router.get("cli_help", {}).items():
        if not result.get("ok"):
            warnings.append(f"CLI help failed for {module}.")
    for package, dependency in python_dependencies.items():
        if dependency.get("required") and not dependency.get("available"):
            errors.append(
                "Missing required Python dependency "
                f"{package} (import {dependency.get('import_name')}). "
                "Re-run scripts/setup/setup_router.py or install the project "
                "dependencies with python -m pip install -e \".[test]\"."
            )
    for name, prerequisite in bootstrap_prerequisites.items():
        if prerequisite.get("required") and not prerequisite.get("available"):
            warnings.append(
                f"Missing bootstrap prerequisite {name}. See system_package_hints."
            )

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
        "python_dependencies": python_dependencies,
        "bootstrap_prerequisites": bootstrap_prerequisites,
        "system_package_hints": _system_package_hints(platform_info),
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
