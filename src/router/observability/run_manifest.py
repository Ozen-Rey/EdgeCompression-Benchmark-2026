import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_git_command(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _args_to_json_safe_dict(args: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for key, value in vars(args).items():
        if key.startswith("_"):
            continue

        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [_json_safe(v) for v in value]
        elif isinstance(value, dict):
            out[key] = {
                str(k): _json_safe(v)
                for k, v in value.items()
            }
        else:
            out[key] = _json_safe(value)

    return out


def build_run_manifest(
    *,
    original_argv: List[str],
    expanded_argv: List[str],
    args: Any,
    router_config_report: Dict[str, Any],
) -> Dict[str, Any]:
    commit = _run_git_command(["rev-parse", "HEAD"])
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git_command(["status", "--porcelain"])

    dirty_worktree = None
    status_preview: List[str] = []

    if status is not None:
        dirty_worktree = bool(status.strip())
        status_preview = status.splitlines()[:20]

    commit_short = commit[:12] if commit else None

    return {
        "enabled": True,
        "version": "0.6",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": {
            "commit": commit,
            "commit_short": commit_short,
            "branch": branch,
            "dirty_worktree": dirty_worktree,
            "status_preview": status_preview,
        },
        "argv": {
            "original": list(original_argv),
            "expanded": list(expanded_argv),
        },
        "inputs": {
            "router_config_file": router_config_report.get("source"),
            "router_config_experiment": router_config_report.get("experiment_name"),
            "csv": getattr(args, "csv", None),
            "codec_registry_file": getattr(args, "codec_registry_file", None),
            "normalization_file": getattr(args, "normalization_file", None),
            "calibration_file": getattr(args, "calibration_file", None),
            "quality_thresholds_file": getattr(args, "quality_thresholds_file", None),
            "input": getattr(args, "input", None),
            "output": getattr(args, "output", None),
        },
        "resolved_args": _args_to_json_safe_dict(args),
    }
