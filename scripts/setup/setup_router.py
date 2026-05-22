"""Router-only cross-platform setup helper.

This script prepares the Python environment used to develop and run the router.
It does not install external codecs, datasets, checkpoints, system packages, or
benchmark execution stacks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
MIN_PYTHON = (3, 10)
SMOKE_MODULES = [
    "src.router.rde_router",
    "src.router.core.domain_spec",
    "src.router.core.dataset_manifest",
    "src.router.core.dataset_ingestion",
    "src.router.core.codec_onboarding",
    "src.router.analysis.audio_video_policy_validation",
]
LIGHTWEIGHT_TESTS = [
    "tests/test_domain_spec.py",
    "tests/test_dataset_manifest.py",
    "tests/test_dataset_ingestion.py",
    "tests/test_codec_onboarding.py",
    "tests/test_multidomain_router_smoke.py",
    "tests/test_audio_video_policy_validation.py",
]
ROUTER_CORE_DEPENDENCIES = [
    {
        "module": "src.router.version",
        "expected_package": "edgecompression-benchmark-2026",
        "required_for": "router version metadata",
    },
    {
        "module": "PIL",
        "expected_package": "Pillow",
        "required_for": "router CLI import path / content-aware image support",
    },
    {
        "module": "src.router.rde_router",
        "expected_package": "edgecompression-benchmark-2026",
        "required_for": "router CLI entrypoint",
    },
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


def _run_probe(command: Sequence[str], timeout_s: float = 5.0) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
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


def _bootstrap_prerequisite_report() -> Dict[str, Dict[str, Any]]:
    pip_probe = _run_probe([sys.executable, "-m", "pip", "--version"])
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
        version_probe = _run_probe([git_path, "--version"])
        report["git"]["version_summary"] = _first_output_line(version_probe)
    if unzip_path:
        version_probe = _run_probe([unzip_path, "-v"])
        report["unzip"]["version_summary"] = _first_output_line(version_probe)

    return report


def _venv_failure_hint(platform_info: Mapping[str, Any]) -> str:
    distro_id = str(platform_info.get("id") or "").lower()
    id_like = {str(value).lower() for value in platform_info.get("id_like", [])}

    if distro_id in {"ubuntu", "debian"} or {"ubuntu", "debian"} & id_like:
        return "Install python3-venv and re-run setup."
    if distro_id in {"arch", "manjaro", "endeavouros"} or "arch" in id_like:
        return "Verify that the python package is installed, then re-run setup."
    return "Verify that Python venv support is installed for this Python interpreter."


def _venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _is_active_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _prompt_yes_no(question: str, assume_yes: bool = False) -> bool:
    if assume_yes:
        print(f"{question} [y/N] y")
        return True
    if not sys.stdin.isatty():
        print(f"{question} [y/N] n")
        return False
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        print("n")
        return False
    return answer in {"y", "yes"}


def _run(command: Sequence[str], dry_run: bool, timeout_s: float | None = None) -> Dict[str, Any]:
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(f"[dry-run] {printable}")
        return {"ok": True, "returncode": None, "command": list(command), "dry_run": True}

    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": list(command),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _check_router_import(python_executable: Path | str, dry_run: bool) -> Dict[str, Any]:
    return _run(
        [
            str(python_executable),
            "-c",
            "from src.router.version import ROUTER_VERSION; print(ROUTER_VERSION)",
        ],
        dry_run=dry_run,
        timeout_s=10,
    )


def _probe_router_core_dependencies(
    python_executable: Path | str,
    dry_run: bool,
) -> Dict[str, Any]:
    probe: Dict[str, Any] = {
        "ok": True,
        "checks": {},
        "missing": [],
        "suggested_action": (
            "Re-run scripts/setup/setup_router.py or install the project "
            "dependencies with python -m pip install -e \".[test]\"."
        ),
    }

    for dependency in ROUTER_CORE_DEPENDENCIES:
        module = dependency["module"]
        result = _run(
            [
                str(python_executable),
                "-c",
                f"import importlib; importlib.import_module({module!r}); print('ok')",
            ],
            dry_run=dry_run,
            timeout_s=10,
        )
        entry = {
            "module": module,
            "expected_package": dependency["expected_package"],
            "required_for": dependency["required_for"],
            "install_scope": "python_package",
            "installed_via": "pyproject.toml / pip",
            "available": None if dry_run else result["ok"],
            "result": result,
        }
        probe["checks"][module] = entry
        if not result["ok"]:
            probe["ok"] = False
            probe["missing"].append(
                {
                    "module": module,
                    "expected_package": dependency["expected_package"],
                    "required_for": dependency["required_for"],
                    "suggested_action": probe["suggested_action"],
                }
            )

    return probe


def _run_smoke_checks(python_executable: Path | str, dry_run: bool) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for module in SMOKE_MODULES:
        checks[module] = _run(
            [str(python_executable), "-m", module, "--help"],
            dry_run=dry_run,
            timeout_s=20,
        )
    return checks


def _run_lightweight_tests(python_executable: Path | str, dry_run: bool) -> Dict[str, Any]:
    return _run(
        [str(python_executable), "-m", "pytest", "-q", *LIGHTWEIGHT_TESTS],
        dry_run=dry_run,
        timeout_s=120,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the router Python development environment. This does not "
            "install codecs, datasets, checkpoints, system packages, or run benchmarks."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing anything.")
    parser.add_argument("--yes", action="store_true", help="Auto-accept router-only actions.")
    parser.add_argument("--venv", type=Path, default=Path(".venv"), help="Virtual environment path.")
    parser.add_argument("--no-venv", action="store_true", help="Use the current Python environment.")
    parser.add_argument("--with-tests", action="store_true", help="Offer/run the lightweight router-only test subset.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when base router checks fail.")
    parser.add_argument("--report-out", type=Path, help="Write a setup report as JSON.")
    return parser


def run_setup(args: argparse.Namespace) -> Dict[str, Any]:
    python_ok = sys.version_info >= MIN_PYTHON
    platform_info = _platform_id()
    report: Dict[str, Any] = {
        "scope": "router-only setup",
        "dry_run": args.dry_run,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "minimum_supported": ".".join(str(part) for part in MIN_PYTHON),
            "supported": python_ok,
        },
        "bootstrap_prerequisites": _bootstrap_prerequisite_report(),
        "system_package_hints": _system_package_hints(platform_info),
        "dependency_boundaries": {
            "router_python_dependencies": "pyproject.toml / pip",
            "system_bootstrap_prerequisites": (
                "manual OS bootstrap only; setup_router.py reports hints but "
                "does not install system packages"
            ),
        },
        "venv": {
            "requested": not args.no_venv,
            "path": None if args.no_venv else str(args.venv),
            "active": _is_active_venv(),
            "exists": False if args.no_venv else args.venv.exists(),
        },
        "actions": [],
        "smoke_checks": {},
        "tests": None,
        "warnings": [],
        "errors": [],
        "non_goals": [
            "does not install system packages",
            "does not install external codecs",
            "does not download datasets",
            "does not download checkpoints",
            "does not run benchmarks",
            "does not write results/",
        ],
    }

    if not python_ok:
        report["errors"].append("Python version is below the router minimum.")

    selected_python: Path | str = sys.executable
    if not args.no_venv:
        venv_path = args.venv
        venv_python = _venv_python(venv_path)
        create_requested = False
        if not venv_path.exists():
            create = _prompt_yes_no(
                f"Virtual environment {venv_path} not found. Create it?",
                assume_yes=args.yes,
            )
            if create:
                create_requested = True
                result = _run([sys.executable, "-m", "venv", str(venv_path)], args.dry_run)
                report["actions"].append({"name": "create_venv", "result": result})
                if not result["ok"]:
                    hint = _venv_failure_hint(platform_info)
                    print(f"Virtual environment creation failed. {hint}", file=sys.stderr)
                    report["errors"].append(
                        f"Virtual environment creation failed. {hint}"
                    )
            else:
                report["actions"].append({"name": "create_venv", "skipped": True})
        if venv_python.exists() or (args.dry_run and create_requested):
            selected_python = venv_python
        elif venv_path.exists():
            report["warnings"].append(f"Virtual environment exists but {venv_python} was not found.")

    install = _prompt_yes_no(
        "Install Python package in editable mode with test dependencies?",
        assume_yes=args.yes,
    )
    if install:
        pip_upgrade = _run([str(selected_python), "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)
        editable = _run([str(selected_python), "-m", "pip", "install", "-e", ".[test]"], args.dry_run)
        report["actions"].extend(
            [
                {"name": "upgrade_pip", "result": pip_upgrade},
                {"name": "install_editable_test_extra", "result": editable},
            ]
        )
        if not pip_upgrade["ok"] or not editable["ok"]:
            report["errors"].append("Python dependency installation failed.")
    else:
        report["actions"].append({"name": "install_editable_test_extra", "skipped": True})

    import_result = _check_router_import(selected_python, args.dry_run)
    report["router_import"] = import_result
    if not import_result["ok"]:
        report["errors"].append("Router import check failed.")

    dependency_probe = _probe_router_core_dependencies(selected_python, args.dry_run)
    report["router_core_dependency_probe"] = dependency_probe
    if not dependency_probe["ok"]:
        for missing in dependency_probe["missing"]:
            report["errors"].append(
                "Missing required router dependency: "
                f"module {missing['module']} "
                f"(package {missing['expected_package']}). "
                f"{missing['suggested_action']}"
            )

    run_smoke = _prompt_yes_no("Run router smoke checks?", assume_yes=args.yes)
    if run_smoke:
        report["smoke_checks"] = _run_smoke_checks(selected_python, args.dry_run)
        if any(not result["ok"] for result in report["smoke_checks"].values()):
            report["errors"].append("One or more router smoke checks failed.")
    else:
        report["actions"].append({"name": "router_smoke_checks", "skipped": True})

    if args.with_tests:
        run_tests = _prompt_yes_no("Run lightweight test subset?", assume_yes=args.yes)
        if run_tests:
            report["tests"] = _run_lightweight_tests(selected_python, args.dry_run)
            if not report["tests"]["ok"]:
                report["errors"].append("Lightweight router test subset failed.")
        else:
            report["actions"].append({"name": "lightweight_tests", "skipped": True})

    return report


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = run_setup(args)

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.report_out and not args.dry_run:
        args.report_out.write_text(payload + "\n", encoding="utf-8")
    elif args.report_out and args.dry_run:
        print(f"[dry-run] Would write setup report to {args.report_out}")

    if args.strict and report["errors"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":  # pragma: no cover
    main()
