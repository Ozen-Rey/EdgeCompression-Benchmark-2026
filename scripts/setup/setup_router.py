"""Router-only cross-platform setup helper.

This script prepares the Python environment used to develop and run the router.
It does not install external codecs, datasets, checkpoints, system packages, or
benchmark execution stacks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


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
                    report["errors"].append("Virtual environment creation failed.")
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
