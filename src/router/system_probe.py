import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _to_float_or_none(value: str):
    try:
        return float(value)
    except Exception:
        return None


def _candidate_executable_names(name: str) -> list[str]:
    if platform.system().lower() == "windows" and not name.lower().endswith(".exe"):
        return [name, f"{name}.exe"]
    return [name]


def _find_executable(name: str) -> Dict[str, Any]:
    """
    Cerca un eseguibile prima nel PATH, poi nelle cartelle locali del progetto.

    Esempi:
      tools/jxl/cjxl.exe
      tools/jxl/bin/cjxl.exe
      tools/ffmpeg/ffmpeg.exe
      tools/ffmpeg/bin/ffmpeg.exe
    """

    path_hit = shutil.which(name)
    if path_hit is not None:
        return {
            "available": True,
            "path": path_hit,
            "source": "PATH",
        }

    local_search_dirs = [
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "tools" / "jxl",
        PROJECT_ROOT / "tools" / "jxl" / "bin",
        PROJECT_ROOT / "tools" / "ffmpeg",
        PROJECT_ROOT / "tools" / "ffmpeg" / "bin",
        PROJECT_ROOT / "tools" / "opus",
        PROJECT_ROOT / "tools" / "opus" / "bin",
        PROJECT_ROOT / "tools" / "svt-av1",
        PROJECT_ROOT / "tools" / "svt-av1" / "bin",
        PROJECT_ROOT / "tools" / "vvenc",
        PROJECT_ROOT / "tools" / "vvenc" / "bin",
    ]

    for directory in local_search_dirs:
        for candidate_name in _candidate_executable_names(name):
            candidate = directory / candidate_name
            if candidate.exists() and candidate.is_file():
                return {
                    "available": True,
                    "path": str(candidate),
                    "source": "project_tools",
                }

    return {
        "available": False,
        "path": None,
        "source": None,
    }


def _probe_psutil() -> Dict[str, Any]:
    try:
        import psutil
    except Exception:
        return {
            "available": False,
            "cpu_percent": None,
            "memory_total_gb": None,
            "memory_available_gb": None,
            "battery": None,
        }

    vm = _safe_call(psutil.virtual_memory)
    battery = _safe_call(psutil.sensors_battery)

    battery_info = None
    if battery is not None:
        battery_info = {
            "percent": battery.percent,
            "power_plugged": battery.power_plugged,
            "secsleft": battery.secsleft,
        }

    return {
        "available": True,
        "cpu_percent": _safe_call(lambda: psutil.cpu_percent(interval=0.1)),
        "memory_total_gb": round(vm.total / (1024**3), 3) if vm else None,
        "memory_available_gb": round(vm.available / (1024**3), 3) if vm else None,
        "battery": battery_info,
    }


def _probe_torch_cuda() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "torch_available": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
        }

    cuda_available = bool(_safe_call(lambda: torch.cuda.is_available(), False))
    device_count = int(_safe_call(lambda: torch.cuda.device_count(), 0) or 0)

    devices = []
    if cuda_available:
        for i in range(device_count):
            props = _safe_call(lambda i=i: torch.cuda.get_device_properties(i))
            if props is not None:
                devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    }
                )

    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
    }


def _probe_nvidia_smi() -> Dict[str, Any]:
    nvidia_smi_info = _find_executable("nvidia-smi")
    nvidia_smi = nvidia_smi_info.get("path")

    if nvidia_smi is None:
        return {
            "available": False,
            "path": None,
            "source": None,
            "gpus": [],
        }

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

        gpus = []
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(
                        {
                            "name": parts[0],
                            "memory_total_mb": _to_float_or_none(parts[1]),
                            "memory_free_mb": _to_float_or_none(parts[2]),
                            "temperature_c": _to_float_or_none(parts[3]),
                            "utilization_percent": _to_float_or_none(parts[4]),
                        }
                    )

        return {
            "available": True,
            "path": nvidia_smi,
            "source": nvidia_smi_info.get("source"),
            "gpus": gpus,
        }

    except Exception:
        return {
            "available": True,
            "path": nvidia_smi,
            "source": nvidia_smi_info.get("source"),
            "gpus": [],
        }


def _probe_executables() -> Dict[str, Any]:
    names = [
        "ffmpeg",
        "python",
        "x264",
        "x265",
        "cjxl",
        "djxl",
        "vvencapp",
        "SvtAv1EncApp",
        "opusenc",
        "opusdec",
        "nvidia-smi",
    ]

    return {
        name: _find_executable(name)
        for name in names
    }


def probe_system() -> Dict[str, Any]:
    psutil_info = _probe_psutil()
    torch_cuda = _probe_torch_cuda()
    nvidia_smi = _probe_nvidia_smi()
    executables = _probe_executables()

    cuda_available = bool(
        torch_cuda.get("cuda_available", False)
        or nvidia_smi.get("available", False)
    )

    return {
        "project_root": str(PROJECT_ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
        },
        "cpu": {
            "logical_cores": _safe_call(lambda: os.cpu_count()),
            "cpu_percent": psutil_info.get("cpu_percent"),
        },
        "memory": {
            "total_gb": psutil_info.get("memory_total_gb"),
            "available_gb": psutil_info.get("memory_available_gb"),
        },
        "battery": psutil_info.get("battery"),
        "cuda": {
            "available": cuda_available,
            "torch": torch_cuda,
            "nvidia_smi": nvidia_smi,
        },
        "executables": executables,
    }