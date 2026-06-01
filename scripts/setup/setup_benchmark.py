"""Self-contained image mini-benchmark setup helper.

This is the benchmark counterpart of ``setup_router.py`` and is intentionally
kept separate from it: the router setup stays router-only, while this script
prepares everything needed to run the image mini-benchmark on the spot and
produce a router-ready CSV.

With confirmation (default ``No``; pass ``--yes`` to auto-accept) it can:

- create / reuse a Python virtual environment;
- install the project with the ``benchmark`` extra (numpy, imagecodecs,
  matplotlib) via ``pip install -e ".[benchmark]"``;
- download the 24 Kodak PNG images locally (cross-platform, stdlib only);
- run the Kodak image mini-benchmark with the numpy-only PSNR metric and
  replay the R-D-E router on the produced CSV so the result can be inspected
  immediately.

Non-goals (kept consistent with the router-only boundary):

- it does not install torch / DCAE checkpoints (DCAE rows degrade gracefully);
- it does not install ffmpeg, cjxl/djxl or other system binaries (HEVC / the
  JPEG XL CLI fallback are reported as optional);
- it does not claim hardware-invariant energy measurements.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_PYTHON = (3, 10)
DEFAULT_KODAK_DIR = ROOT / "datasets" / "images" / "kodak"
DEFAULT_OUT_DIR = ROOT / "validation_runs" / "benchmark_quickstart"
MINI_BENCHMARK = "scripts.validation.image_kodak_mini_benchmark"

# Python dependencies pulled in by the ".[benchmark]" extra.
BENCHMARK_DEPENDENCIES = [
    {"module": "numpy", "package": "numpy", "required_for": "PSNR metric and array handling"},
    {"module": "imagecodecs", "package": "imagecodecs", "required_for": "JPEG / JPEG XL encode-decode"},
    {"module": "matplotlib", "package": "matplotlib", "required_for": "diagnostic R-D-E plots"},
    {"module": "ssimulacra2", "package": "ssimulacra2", "required_for": "SSIMULACRA2 quality metric (--quality-metric ssimulacra2)"},
    {"module": "PIL", "package": "Pillow", "required_for": "image I/O"},
]
# System binaries that unlock extra codecs but are never installed here.
OPTIONAL_EXTERNAL_TOOLS = {
    "ffmpeg": "HEVC intra (libx265) operating points",
    "cjxl": "JPEG XL CLI fallback when imagecodecs lacks JPEG XL",
    "djxl": "JPEG XL CLI fallback when imagecodecs lacks JPEG XL",
}


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


def _benchmark_dependency_report() -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for dependency in BENCHMARK_DEPENDENCIES:
        module = str(dependency["module"])
        report[dependency["package"]] = {
            "import_name": module,
            "required_for": dependency["required_for"],
            "available": importlib.util.find_spec(module) is not None,
            "install_scope": "python_package",
            "installed_via": 'pip install -e ".[benchmark]"',
        }
    return report


def _optional_tool_report() -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for name, purpose in OPTIONAL_EXTERNAL_TOOLS.items():
        path = shutil.which(name)
        report[name] = {
            "available": path is not None,
            "path": path,
            "unlocks": purpose,
            "managed_by_setup": False,
        }
    return report


def _codec_plan(tool_report: Dict[str, Dict[str, Any]], deps: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    imagecodecs_ok = deps.get("imagecodecs", {}).get("available", False)
    jxl_cli = tool_report.get("cjxl", {}).get("available") and tool_report.get("djxl", {}).get("available")
    return {
        "jpeg": "ready (imagecodecs or Pillow fallback)",
        "jxl": "ready" if imagecodecs_ok or jxl_cli else "needs imagecodecs or cjxl/djxl",
        "hevc": "ready" if tool_report.get("ffmpeg", {}).get("available") else "needs ffmpeg (skipped otherwise)",
        "dcae": "needs torch + DCAE checkpoints (skipped otherwise)",
    }


def _check_router_import(python_executable: Path | str, dry_run: bool) -> Dict[str, Any]:
    return _run(
        [
            str(python_executable),
            "-c",
            "from src.router.version import ROUTER_VERSION; print(ROUTER_VERSION)",
        ],
        dry_run=dry_run,
        timeout_s=15,
    )


def run_setup(args: argparse.Namespace) -> Dict[str, Any]:
    python_ok = sys.version_info >= MIN_PYTHON
    deps = _benchmark_dependency_report()
    tools = _optional_tool_report()

    report: Dict[str, Any] = {
        "scope": "image mini-benchmark setup (separate from router-only setup)",
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
        "quality_metric": args.quality_metric,
        "energy_proxy": args.energy_proxy,
        "benchmark_dependencies": deps,
        "optional_external_tools": tools,
        "codec_plan": _codec_plan(tools, deps),
        "actions": [],
        "warnings": [],
        "errors": [],
        "non_goals": [
            "does not install torch or DCAE checkpoints",
            "does not install ffmpeg / cjxl / djxl system binaries",
            "does not claim hardware-invariant energy measurements",
            "does not redistribute Kodak images, checkpoints, or codec binaries",
        ],
    }

    if not python_ok:
        report["errors"].append("Python version is below the minimum.")

    # 1) Virtual environment.
    selected_python: Path | str = sys.executable
    if not args.no_venv:
        venv_path = args.venv
        venv_python = _venv_python(venv_path)
        create_requested = False
        if not venv_path.exists():
            if _prompt_yes_no(f"Virtual environment {venv_path} not found. Create it?", assume_yes=args.yes):
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

    # 2) Install the benchmark extra.
    if _prompt_yes_no('Install project with the benchmark extra (pip install -e ".[benchmark]")?', assume_yes=args.yes):
        pip_upgrade = _run([str(selected_python), "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)
        editable = _run([str(selected_python), "-m", "pip", "install", "-e", ".[benchmark]"], args.dry_run)
        report["actions"].extend(
            [
                {"name": "upgrade_pip", "result": pip_upgrade},
                {"name": "install_editable_benchmark_extra", "result": editable},
            ]
        )
        if not pip_upgrade["ok"] or not editable["ok"]:
            report["errors"].append("Benchmark dependency installation failed.")
    else:
        report["actions"].append({"name": "install_editable_benchmark_extra", "skipped": True})

    import_result = _check_router_import(selected_python, args.dry_run)
    report["router_import"] = import_result
    if not import_result["ok"]:
        report["warnings"].append("Router import check failed; router replay may not work.")

    # 3) Download Kodak.
    kodak_dir = args.kodak_dir.expanduser()
    kodak_ready = kodak_dir.exists() and len(list(kodak_dir.glob("*.png"))) >= 24
    if kodak_ready:
        report["actions"].append({"name": "download_kodak", "skipped": "already_present", "path": str(kodak_dir)})
    elif _prompt_yes_no(f"Download the 24 Kodak PNG images into {kodak_dir}?", assume_yes=args.yes):
        result = _run(
            [str(selected_python), "-m", "scripts.setup.download_kodak", "--out-dir", str(kodak_dir)],
            args.dry_run,
            timeout_s=600,
        )
        report["actions"].append({"name": "download_kodak", "result": result})
        kodak_ready = args.dry_run or result["ok"]
        if not kodak_ready:
            report["errors"].append("Kodak download failed.")
    else:
        report["actions"].append({"name": "download_kodak", "skipped": True})

    # 4) Run the mini-benchmark and replay the router.
    if not args.skip_benchmark and (kodak_ready or args.dry_run):
        if _prompt_yes_no("Run the Kodak mini-benchmark and replay the router now?", assume_yes=args.yes):
            cmd = [
                str(selected_python),
                "-m",
                MINI_BENCHMARK,
                "--kodak-dir",
                str(kodak_dir),
                "--out-dir",
                str(args.out_dir),
                "--codecs",
                args.codecs,
                "--quality-metric",
                args.quality_metric,
                "--max-images",
                str(args.max_images),
                "--energy-backend",
                args.energy_backend,
                "--energy-proxy",
                args.energy_proxy,
            ]
            if args.run_router:
                cmd.append("--run-router")
            result = _run(cmd, args.dry_run, timeout_s=1800)
            report["actions"].append({"name": "run_mini_benchmark", "result": result})
            report["benchmark_outputs"] = {
                "out_dir": str(args.out_dir),
                "router_ready_csv": str(args.out_dir / "kodak_image_rde_mini_router_ready.csv"),
                "report_json": str(args.out_dir / "kodak_image_rde_mini_report.json"),
            }
            if not result["ok"]:
                report["errors"].append("Mini-benchmark run failed.")
        else:
            report["actions"].append({"name": "run_mini_benchmark", "skipped": True})
    elif not args.skip_benchmark:
        report["actions"].append({"name": "run_mini_benchmark", "skipped": "kodak_not_ready"})

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and run the image mini-benchmark, producing a router-ready CSV. "
            "Separate from the router-only setup."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing anything.")
    parser.add_argument("--yes", action="store_true", help="Auto-accept all confirmations.")
    parser.add_argument("--venv", type=Path, default=Path(".venv"), help="Virtual environment path.")
    parser.add_argument("--no-venv", action="store_true", help="Use the current Python environment.")
    parser.add_argument("--kodak-dir", type=Path, default=DEFAULT_KODAK_DIR, help="Kodak image directory.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Mini-benchmark output directory.")
    parser.add_argument("--codecs", default="jpeg,jxl", help="Codecs to run. Default: jpeg,jxl (pure pip).")
    parser.add_argument("--quality-metric", choices=["psnr", "ssimulacra2"], default="psnr", help="Quality metric.")
    parser.add_argument("--max-images", type=int, default=4, help="Image limit for the smoke run.")
    parser.add_argument("--energy-backend", choices=["auto", "cpu", "gpu", "both", "none"], default="auto")
    parser.add_argument(
        "--energy-proxy",
        choices=["off", "time"],
        default="time",
        help=(
            "When energy telemetry is unavailable, fill missing energy with a "
            "labeled time proxy so the router replay works everywhere. Default: time. "
            "Use 'off' on a real measurement host to keep energy strictly measured."
        ),
    )
    parser.add_argument("--no-router", dest="run_router", action="store_false", help="Skip router replay.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Set up only; do not run the benchmark.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any step fails.")
    parser.add_argument("--report-out", type=Path, help="Write a setup report as JSON.")
    parser.set_defaults(run_router=True)
    return parser


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
