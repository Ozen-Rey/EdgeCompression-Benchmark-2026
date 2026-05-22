"""Read-only router environment validation report generator.

This script records cross-platform router setup evidence. It does not install
packages, create virtual environments, download datasets/checkpoints, run
benchmarks, require external codecs, or write under ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_SET = "basic"
BOUNDARIES = [
    "No benchmark was executed.",
    "No datasets or checkpoints were downloaded.",
    "No external codec binaries are required for fixture routing.",
    "This validates router setup and deterministic replay on fixed R-D-E rows.",
    "It does not validate hardware-invariant energy measurements.",
]
HELP_MODULES = [
    "src.router.rde_router",
    "src.router.core.domain_spec",
    "src.router.core.dataset_manifest",
    "src.router.core.dataset_ingestion",
    "src.router.core.codec_onboarding",
    "src.router.analysis.audio_video_policy_validation",
]
EXTERNAL_TOOLS = [
    "ffmpeg",
    "ffprobe",
    "cjxl",
    "djxl",
    "nvidia-smi",
]
FIXTURES = {
    "video_vmaf": {
        "domain": "video",
        "csv": "tests/fixtures/rde_video_vmaf.csv",
        "domain_spec": "video_vmaf",
    },
    "audio_visqol": {
        "domain": "audio",
        "csv": "tests/fixtures/rde_audio_visqol.csv",
        "domain_spec": "audio_visqol",
    },
    "image_ssimulacra2": {
        "domain": "image",
        "csv": "tests/fixtures/image_rde_real_small.csv",
        "domain_spec": "image_ssimulacra2",
    },
}


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned.strip("._-") or "router_environment"


def _default_out_dir(label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("validation_runs") / f"{_safe_label(label)}_{stamp}"


def _resolve_out_dir(path: Path) -> Path:
    resolved = (ROOT / path if not path.is_absolute() else path).resolve()
    results_dir = (ROOT / "results").resolve()
    if resolved == results_dir or results_dir in resolved.parents:
        raise ValueError("Refusing to write validation outputs under results/.")
    return resolved


def run_command(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Run a command and return a JSON-serializable result without raising."""
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": None,
            "command": list(command),
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "command": list(command),
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {timeout_s}s",
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": list(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _relative(path: Path, out_dir: Path) -> str:
    try:
        return str(path.relative_to(out_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_os_release() -> Dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    return data


def _windows_version_info() -> Dict[str, Any] | None:
    if platform.system() != "Windows":
        return None
    version = sys.getwindowsversion()
    return {
        "major": version.major,
        "minor": version.minor,
        "build": version.build,
        "platform": version.platform,
        "service_pack": version.service_pack,
    }


def _macos_version_info() -> Dict[str, Any] | None:
    if platform.system() != "Darwin":
        return None
    version, version_info, machine = platform.mac_ver()
    return {
        "version": version,
        "version_info": list(version_info),
        "machine": machine,
    }


def _linux_cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def _psutil_snapshot() -> tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return (
            {
                "available": False,
                "cpu": None,
                "ram": {
                    "total_bytes": None,
                    "available_bytes": None,
                },
            },
            ["psutil_unavailable: physical CPU cores, frequency and RAM unavailable"],
        )

    cpu_freq = psutil.cpu_freq()
    memory = psutil.virtual_memory()
    return (
        {
            "available": True,
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency_mhz": None
                if cpu_freq is None
                else {
                    "current": cpu_freq.current,
                    "min": cpu_freq.min,
                    "max": cpu_freq.max,
                },
            },
            "ram": {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
            },
        },
        warnings,
    )


def _tool_version_command(tool: str, path: str) -> List[str]:
    if tool in {"ffmpeg", "ffprobe"}:
        return [path, "-version"]
    if tool in {"cjxl", "djxl"}:
        return [path, "--version"]
    if tool == "nvidia-smi":
        return [path, "--version"]
    return [path, "--version"]


def _first_line(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped.splitlines()[0]


def _external_tools_snapshot() -> Dict[str, Dict[str, Any]]:
    tools: Dict[str, Dict[str, Any]] = {}
    for tool in EXTERNAL_TOOLS:
        path = shutil.which(tool)
        entry: Dict[str, Any] = {
            "available": path is not None,
            "path": path,
            "optional": True,
        }
        if path is not None:
            result = run_command(_tool_version_command(tool, path), timeout_s=10.0)
            entry.update(
                {
                    "version_probe_ok": result["ok"],
                    "version_summary": _first_line(result.get("stdout"))
                    or _first_line(result.get("stderr")),
                }
            )
        tools[tool] = entry
    return tools


def _parse_nvidia_smi_csv(stdout: str, include_cuda: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if include_cuda and len(parts) >= 5:
            name, driver, total, free, cuda = parts[:5]
        elif len(parts) >= 4:
            name, driver, total, free = parts[:4]
            cuda = None
        else:
            continue
        rows.append(
            {
                "name": name,
                "driver_version": driver,
                "memory_total": total,
                "memory_free": free,
                "cuda_version": cuda,
            }
        )
    return rows


def _gpu_snapshot() -> tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return (
            {
                "optional": True,
                "nvidia_smi_available": False,
                "gpus": [],
            },
            ["nvidia_smi_unavailable: GPU details not collected"],
        )

    query = [
        nvidia_smi,
        "--query-gpu=name,driver_version,memory.total,memory.free,cuda_version",
        "--format=csv,noheader",
    ]
    result = run_command(query, timeout_s=10.0)
    include_cuda = True
    if not result["ok"]:
        warnings.append("nvidia_smi_cuda_query_failed: retrying without cuda_version")
        result = run_command(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            timeout_s=10.0,
        )
        include_cuda = False

    return (
        {
            "optional": True,
            "nvidia_smi_available": True,
            "nvidia_smi_path": nvidia_smi,
            "query_ok": result["ok"],
            "gpus": _parse_nvidia_smi_csv(result.get("stdout", ""), include_cuda)
            if result["ok"]
            else [],
        },
        warnings if result["ok"] else [*warnings, "nvidia_smi_query_failed"],
    )


def _sanitize_string(value: str) -> str:
    sanitized = value
    replacements = [
        (str(ROOT.resolve()), "<repo-root>"),
        (str(Path.home()), "<home>"),
    ]
    try:
        replacements.append((str(ROOT.resolve()).replace("\\", "/"), "<repo-root>"))
        replacements.append((str(Path.home()).replace("\\", "/"), "<home>"))
    except RuntimeError:
        pass
    for old, new in replacements:
        if old:
            sanitized = sanitized.replace(old, new)
    home_name = Path.home().name
    if home_name:
        sanitized = re.sub(re.escape(home_name), "<user>", sanitized, flags=re.IGNORECASE)
    return sanitized


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    return value


def _fingerprint_summary(fingerprint: Mapping[str, Any]) -> Dict[str, Any]:
    os_info = fingerprint.get("os", {})
    cpu_info = fingerprint.get("cpu", {})
    ram_info = fingerprint.get("ram", {})
    gpu_info = fingerprint.get("gpu", {})
    python_info = fingerprint.get("python", {})
    gpus = gpu_info.get("gpus") or []
    first_gpu = gpus[0].get("name") if gpus and isinstance(gpus[0], dict) else None
    ram_total = ram_info.get("total_bytes")
    return {
        "os": os_info.get("platform"),
        "kernel": os_info.get("release"),
        "cpu_model": cpu_info.get("linux_model") or cpu_info.get("processor"),
        "logical_cpus": cpu_info.get("logical_cpus"),
        "ram_gb": None if ram_total is None else round(float(ram_total) / (1024**3), 2),
        "gpu": first_gpu,
        "python": str(python_info.get("version") or "").splitlines()[0] or None,
    }


def collect_platform_fingerprint(
    python: str,
    out_dir: Path,
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    warnings: List[str] = []
    psutil_data, psutil_warnings = _psutil_snapshot()
    warnings.extend(psutil_warnings)
    gpu_data, gpu_warnings = _gpu_snapshot()
    warnings.extend(gpu_warnings)
    python_info = _python_metadata(python)
    git_info = _git_metadata()

    cpu_psutil = psutil_data.get("cpu") or {}
    ram_psutil = psutil_data.get("ram") or {}
    fingerprint: Dict[str, Any] = {
        "schema": "rde_platform_fingerprint_v1",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "linux_os_release": _read_os_release() if platform.system() == "Linux" else None,
            "windows_version": _windows_version_info(),
            "macos_version": _macos_version_info(),
        },
        "cpu": {
            "processor": platform.processor() or None,
            "logical_cpus": os.cpu_count(),
            "physical_cores": cpu_psutil.get("physical_cores"),
            "psutil_logical_cores": cpu_psutil.get("logical_cores"),
            "frequency_mhz": cpu_psutil.get("frequency_mhz"),
            "linux_model": _linux_cpu_model() if platform.system() == "Linux" else None,
        },
        "ram": {
            "total_bytes": ram_psutil.get("total_bytes"),
            "available_bytes": ram_psutil.get("available_bytes"),
            "source": "psutil" if psutil_data.get("available") else None,
        },
        "gpu": gpu_data,
        "python": {
            "version": python_info.get("version"),
            "executable": python_info.get("executable"),
            "prefix": python_info.get("prefix"),
            "base_prefix": python_info.get("base_prefix"),
            "venv_active": python_info.get("prefix") != python_info.get("base_prefix"),
            "pip_freeze_path": environment.get("pip_freeze", {}).get("path"),
        },
        "git": {
            "commit": git_info.get("commit"),
            "tag_or_describe": git_info.get("tag_or_describe"),
            "dirty": git_info.get("dirty"),
        },
        "external_tools": _external_tools_snapshot(),
        "warnings": warnings,
    }

    full_path = out_dir / "platform_fingerprint_full.json"
    sanitized_path = out_dir / "platform_fingerprint_sanitized.json"
    sanitized = _sanitize_payload(fingerprint)
    sanitized_sha256 = hashlib.sha256(_canonical_json_bytes(sanitized)).hexdigest()
    _write_json(full_path, fingerprint)
    _write_json(sanitized_path, sanitized)

    return {
        "schema": "rde_platform_fingerprint_v1",
        "full_path": _relative(full_path, out_dir),
        "sanitized_path": _relative(sanitized_path, out_dir),
        "sanitized_sha256": sanitized_sha256,
        "summary": _fingerprint_summary(fingerprint),
        "warnings": warnings,
    }


def _python_metadata(python: str) -> Dict[str, Any]:
    probe = run_command(
        [
            python,
            "-c",
            (
                "import json, sys; "
                "print(json.dumps({"
                "'version': sys.version, "
                "'executable': sys.executable, "
                "'prefix': sys.prefix, "
                "'base_prefix': getattr(sys, 'base_prefix', sys.prefix), "
                "'venv_active': sys.prefix != getattr(sys, 'base_prefix', sys.prefix)"
                "}))"
            ),
        ],
        timeout_s=10.0,
    )
    if probe["ok"]:
        try:
            data = json.loads(probe["stdout"])
            return {
                "version": data["version"],
                "executable": data["executable"],
                "prefix": data["prefix"],
                "base_prefix": data["base_prefix"],
                "probe_ok": True,
            }
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "version": sys.version,
        "executable": python,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "probe_ok": False,
        "probe_error": probe.get("stderr") or probe.get("stdout"),
    }


def _git_metadata() -> Dict[str, Any]:
    warnings: List[str] = []
    commit = run_command(["git", "rev-parse", "HEAD"], timeout_s=10.0)
    describe = run_command(["git", "describe", "--tags", "--always", "--dirty"], timeout_s=10.0)
    status = run_command(["git", "status", "--short", "--branch"], timeout_s=10.0)

    for name, result in (
        ("commit", commit),
        ("describe", describe),
        ("status", status),
    ):
        if not result["ok"]:
            warnings.append(f"git_{name}_failed: {result.get('stderr') or result.get('stdout')}")

    status_text = status.get("stdout", "") if status["ok"] else ""
    dirty = any(
        line and not line.startswith("##")
        for line in status_text.splitlines()
    )
    return {
        "commit": commit["stdout"].strip() if commit["ok"] else None,
        "tag_or_describe": describe["stdout"].strip() if describe["ok"] else None,
        "dirty": dirty if status["ok"] else None,
        "status_short_branch": status_text,
        "warnings": warnings,
    }


def collect_environment_metadata(python: str, out_dir: Path, timestamp: str) -> Dict[str, Any]:
    python_info = _python_metadata(python)
    git_info = _git_metadata()
    pip_freeze = run_command([python, "-m", "pip", "freeze"], timeout_s=60.0)
    _write_text(out_dir / "pip_freeze.txt", pip_freeze.get("stdout", ""))

    return {
        "timestamp_utc": timestamp,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "distro": _read_os_release() if platform.system() == "Linux" else {},
        },
        "python": {
            "version": python_info["version"],
            "executable": python_info["executable"],
            "prefix": python_info["prefix"],
            "base_prefix": python_info["base_prefix"],
        },
        "venv": {
            "active": python_info["prefix"] != python_info["base_prefix"],
            "prefix": python_info["prefix"],
            "base_prefix": python_info["base_prefix"],
        },
        "git": git_info,
        "pip_freeze": {
            "ok": pip_freeze["ok"],
            "returncode": pip_freeze["returncode"],
            "path": "pip_freeze.txt",
            "stderr": pip_freeze.get("stderr", ""),
        },
        "warnings": [*git_info["warnings"]]
        + ([] if pip_freeze["ok"] else ["pip_freeze_failed"]),
    }


def run_doctor(python: str, out_dir: Path) -> Dict[str, Any]:
    report_path = out_dir / "doctor_report.json"
    result = run_command(
        [
            python,
            "scripts/setup/doctor.py",
            "--report-out",
            str(report_path),
        ],
        timeout_s=120.0,
    )
    _write_text(out_dir / "doctor_stdout.txt", result.get("stdout", ""))
    _write_text(out_dir / "doctor_stderr.txt", result.get("stderr", ""))

    summary: Dict[str, Any] = {
        "ok": result["ok"] and report_path.exists(),
        "path": _relative(report_path, out_dir),
        "returncode": result["returncode"],
        "stdout": "doctor_stdout.txt",
        "stderr": "doctor_stderr.txt",
        "warnings": [],
        "errors": [],
    }
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            router = report.get("router", {})
            summary.update(
                {
                    "router_import_ok": router.get("import_ok"),
                    "router_version": router.get("version"),
                    "python_dependencies": report.get("python_dependencies", {}),
                    "optional_external_tools": report.get("optional_external_tools", {}),
                    "warnings": report.get("warnings", []),
                    "errors": report.get("errors", []),
                }
            )
        except json.JSONDecodeError as exc:
            summary["ok"] = False
            summary["errors"] = [f"doctor_report_json_decode_failed:{exc}"]
    return summary


def run_cli_help_checks(python: str, out_dir: Path) -> Dict[str, Dict[str, Any]]:
    help_dir = out_dir / "cli_help"
    help_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    for module in HELP_MODULES:
        stem = module.replace(".", "_")
        stdout_path = help_dir / f"{stem}_stdout.txt"
        stderr_path = help_dir / f"{stem}_stderr.txt"
        result = run_command([python, "-m", module, "--help"], timeout_s=60.0)
        _write_text(stdout_path, result.get("stdout", ""))
        _write_text(stderr_path, result.get("stderr", ""))
        results[module] = {
            "ok": result["ok"],
            "returncode": result["returncode"],
            "stdout": _relative(stdout_path, out_dir),
            "stderr": _relative(stderr_path, out_dir),
        }
    return results


def _read_summary_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_fixture_selection(report_path: Path, summary_path: Path) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary_rows = _read_summary_rows(summary_path) if summary_path.exists() else []
    selected = report.get("decision", {}).get("selected", {})
    return {
        "selected_codec": selected.get("codec"),
        "selected_config": selected.get("config"),
        "rate": selected.get("rate"),
        "quality": selected.get("quality"),
        "energy": selected.get("energy"),
        "J_RDE": selected.get("cost"),
        "summary_rows": len(summary_rows),
    }


def run_fixture_checks(python: str, out_dir: Path, fixture_set: str) -> Dict[str, Dict[str, Any]]:
    if fixture_set != DEFAULT_FIXTURE_SET:
        raise ValueError(f"Unsupported fixture set {fixture_set!r}; expected 'basic'.")

    results: Dict[str, Dict[str, Any]] = {}
    for name, fixture in FIXTURES.items():
        domain = fixture["domain"]
        report_path = out_dir / f"fixture_{domain}_report.json"
        summary_path = out_dir / f"fixture_{domain}_summary.csv"
        stdout_path = out_dir / f"fixture_{domain}_stdout.txt"
        stderr_path = out_dir / f"fixture_{domain}_stderr.txt"
        result = run_command(
            [
                python,
                "-m",
                "src.router.rde_router",
                "--csv",
                fixture["csv"],
                "--domain-spec",
                fixture["domain_spec"],
                "--profile",
                "balanced",
                "--out",
                str(report_path),
                "--summary-out",
                str(summary_path),
            ],
            timeout_s=120.0,
        )
        _write_text(stdout_path, result.get("stdout", ""))
        _write_text(stderr_path, result.get("stderr", ""))

        entry: Dict[str, Any] = {
            "ok": result["ok"] and report_path.exists() and summary_path.exists(),
            "returncode": result["returncode"],
            "summary_path": _relative(summary_path, out_dir),
            "report_path": _relative(report_path, out_dir),
            "stdout": _relative(stdout_path, out_dir),
            "stderr": _relative(stderr_path, out_dir),
        }
        if entry["ok"]:
            try:
                entry.update(_extract_fixture_selection(report_path, summary_path))
            except (json.JSONDecodeError, OSError) as exc:
                entry["ok"] = False
                entry["error"] = f"fixture_output_parse_failed:{exc}"
        else:
            entry["error"] = result.get("stderr") or result.get("stdout") or "fixture run failed"
        results[name] = entry
    return results


def _build_summary_markdown(manifest: Mapping[str, Any]) -> str:
    environment = manifest["environment"]
    repo = manifest["repo"]
    doctor = manifest["doctor"]
    rows = [
        "# Router Environment Validation",
        "",
        f"- Label: `{manifest['label']}`",
        f"- OS: `{environment['os'].get('platform')}`",
        f"- Python: `{environment['python'].get('version').splitlines()[0]}`",
        f"- Commit: `{repo.get('commit')}`",
        f"- Tag/describe: `{repo.get('tag_or_describe')}`",
        f"- Doctor: `{'OK' if doctor.get('ok') else 'FAIL/SKIPPED'}`",
        "",
        "| Domain | OK | Selected codec | Selected config | J_RDE |",
        "|---|---:|---|---|---:|",
    ]
    for name, result in manifest.get("fixture_runs", {}).items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    "yes" if result.get("ok") else "no",
                    str(result.get("selected_codec", "")),
                    str(result.get("selected_config", "")),
                    str(result.get("J_RDE", "")),
                ]
            )
            + " |"
        )
    if not manifest.get("fixture_runs"):
        rows.append("| skipped | no |  |  |  |")

    rows.extend(["", "## Boundaries", ""])
    rows.extend(f"- {item}" for item in manifest["boundaries"])
    rows.append("")
    return "\n".join(rows)


def build_manifest(
    *,
    label: str,
    timestamp: str,
    environment: Mapping[str, Any],
    platform_fingerprint: Mapping[str, Any],
    doctor: Mapping[str, Any],
    cli_help: Mapping[str, Any],
    fixture_runs: Mapping[str, Any],
) -> Dict[str, Any]:
    repo = environment.get("git", {})
    required_results = [doctor.get("ok", True), *[item.get("ok") for item in cli_help.values()]]
    required_results.extend(item.get("ok") for item in fixture_runs.values())
    overall_ok = all(bool(item) for item in required_results)

    return {
        "validation_scope": "router environment validation",
        "not_benchmark_reproduction": True,
        "label": label,
        "timestamp_utc": timestamp,
        "repo": {
            "commit": repo.get("commit"),
            "tag_or_describe": repo.get("tag_or_describe"),
            "dirty": repo.get("dirty"),
        },
        "environment": {
            "os": environment.get("os", {}),
            "python": environment.get("python", {}),
            "venv": environment.get("venv", {}),
        },
        "platform_fingerprint": dict(platform_fingerprint),
        "doctor": dict(doctor),
        "cli_help": dict(cli_help),
        "fixture_runs": dict(fixture_runs),
        "overall_ok": overall_ok,
        "boundaries": BOUNDARIES,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a read-only router environment validation report."
    )
    parser.add_argument("--label", default=platform.node() or "router_environment")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to validation_runs/<label>_<timestamp>.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if required checks fail.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip fixture router runs.")
    parser.add_argument("--no-doctor", action="store_true", help="Skip doctor execution.")
    parser.add_argument(
        "--fixture-set",
        default=DEFAULT_FIXTURE_SET,
        choices=[DEFAULT_FIXTURE_SET],
        help="Fixture set to run.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    return parser


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    timestamp = _timestamp_utc()
    out_dir = _resolve_out_dir(args.out_dir or _default_out_dir(args.label))
    out_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment_metadata(args.python, out_dir, timestamp)
    _write_json(out_dir / "environment_metadata.json", environment)
    platform_fingerprint = collect_platform_fingerprint(args.python, out_dir, environment)

    if args.no_doctor:
        doctor = {
            "ok": True,
            "skipped": True,
            "path": None,
            "warnings": [],
            "errors": [],
        }
    else:
        doctor = run_doctor(args.python, out_dir)

    cli_help = run_cli_help_checks(args.python, out_dir)
    fixture_runs = (
        {}
        if args.skip_tests
        else run_fixture_checks(args.python, out_dir, args.fixture_set)
    )

    manifest = build_manifest(
        label=args.label,
        timestamp=timestamp,
        environment=environment,
        platform_fingerprint=platform_fingerprint,
        doctor=doctor,
        cli_help=cli_help,
        fixture_runs=fixture_runs,
    )
    manifest_path = out_dir / "router_environment_validation_manifest.json"
    _write_json(manifest_path, manifest)
    _write_text(out_dir / "VALIDATION_SUMMARY.md", _build_summary_markdown(manifest))

    print(json.dumps({"overall_ok": manifest["overall_ok"], "manifest": str(manifest_path)}, indent=2))
    if args.strict and not manifest["overall_ok"]:
        raise SystemExit(1)
    return manifest


if __name__ == "__main__":  # pragma: no cover
    main()
