import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CACHE: Dict[str, Any] = {
    "static": None,
    "dynamic": None,
    "gpu": None,
}

_CACHE_TS: Dict[str, float] = {
    "static": 0.0,
    "dynamic": 0.0,
    "gpu": 0.0,
}


EXECUTABLE_NAMES = [
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


def clear_system_feature_cache() -> None:
    for key in _CACHE:
        _CACHE[key] = None
        _CACHE_TS[key] = 0.0


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(t0: float) -> float:
    return (_now() - t0) * 1000.0


def _cache_valid(key: str, ttl_s: float) -> bool:
    if _CACHE.get(key) is None:
        return False
    if ttl_s <= 0:
        return False
    return (_now() - _CACHE_TS.get(key, 0.0)) <= ttl_s


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _gb(value_bytes: Optional[float]) -> Optional[float]:
    if value_bytes is None:
        return None
    return round(float(value_bytes) / (1024 ** 3), 3)


def _find_executable(name: str) -> Dict[str, Any]:
    hit = shutil.which(name)
    if hit:
        return {
            "available": True,
            "path": hit,
            "source": "PATH",
        }

    local_dirs = [
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

    candidate_names = [name]
    if platform.system().lower() == "windows" and not name.lower().endswith(".exe"):
        candidate_names.append(f"{name}.exe")

    for directory in local_dirs:
        for candidate_name in candidate_names:
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


def _probe_static_features() -> Dict[str, Any]:
    executables = {
        name: _find_executable(name)
        for name in EXECUTABLE_NAMES
    }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "is_windows": platform.system().lower() == "windows",
            "is_linux": platform.system().lower() == "linux",
            "is_darwin": platform.system().lower() == "darwin",
        },
        "cpu_static": {
            "logical_cores": os.cpu_count(),
        },
        "executables": executables,
    }


def _probe_dynamic_features(cpu_interval_s: float = 0.0) -> Dict[str, Any]:
    try:
        import psutil
    except Exception:
        return {
            "psutil_available": False,
            "cpu": {
                "cpu_percent": None,
            },
            "memory": {
                "total_gb": None,
                "available_gb": None,
                "used_gb": None,
                "percent": None,
                "available_ratio": None,
            },
            "swap": {
                "total_gb": None,
                "used_gb": None,
                "percent": None,
            },
            "battery": {
                "available": False,
                "percent": None,
                "power_plugged": None,
                "secsleft": None,
                "power_mode": "unknown",
            },
            "disk": _probe_disk_features(),
        }

    cpu_percent = _safe_call(
        lambda: psutil.cpu_percent(
            interval=cpu_interval_s if cpu_interval_s > 0 else None
        )
    )

    cpu_freq = _safe_call(psutil.cpu_freq)
    vm = _safe_call(psutil.virtual_memory)
    sm = _safe_call(psutil.swap_memory)
    battery = _safe_call(psutil.sensors_battery)

    if battery is None:
        battery_info = {
            "available": False,
            "percent": None,
            "power_plugged": None,
            "secsleft": None,
            "power_mode": "unknown",
        }
    else:
        power_plugged = bool(battery.power_plugged)
        battery_info = {
            "available": True,
            "percent": battery.percent,
            "power_plugged": power_plugged,
            "secsleft": battery.secsleft,
            "power_mode": "ac" if power_plugged else "battery",
        }

    return {
        "psutil_available": True,
        "cpu": {
            "cpu_percent": cpu_percent,
            "cpu_freq_current_mhz": getattr(cpu_freq, "current", None) if cpu_freq else None,
            "cpu_freq_max_mhz": getattr(cpu_freq, "max", None) if cpu_freq else None,
        },
        "memory": {
            "total_gb": _gb(vm.total) if vm else None,
            "available_gb": _gb(vm.available) if vm else None,
            "used_gb": _gb(vm.used) if vm else None,
            "percent": vm.percent if vm else None,
            "available_ratio": (
                float(vm.available) / float(vm.total)
                if vm and vm.total
                else None
            ),
        },
        "swap": {
            "total_gb": _gb(sm.total) if sm else None,
            "used_gb": _gb(sm.used) if sm else None,
            "percent": sm.percent if sm else None,
        },
        "battery": battery_info,
        "disk": _probe_disk_features(),
    }


def _probe_disk_features() -> Dict[str, Any]:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        return {
            "path": str(PROJECT_ROOT),
            "total_gb": _gb(usage.total),
            "used_gb": _gb(usage.used),
            "free_gb": _gb(usage.free),
            "percent": (
                float(usage.used) / float(usage.total) * 100.0
                if usage.total
                else None
            ),
        }
    except Exception:
        return {
            "path": str(PROJECT_ROOT),
            "total_gb": None,
            "used_gb": None,
            "free_gb": None,
            "percent": None,
        }


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _probe_nvidia_smi() -> Dict[str, Any]:
    nvidia_smi_info = _find_executable("nvidia-smi")
    nvidia_smi = nvidia_smi_info.get("path")

    if not nvidia_smi:
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
                "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
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
                if len(parts) >= 6:
                    total = _to_float_or_none(parts[1])
                    free = _to_float_or_none(parts[2])
                    used = _to_float_or_none(parts[3])

                    gpus.append(
                        {
                            "name": parts[0],
                            "memory_total_mb": total,
                            "memory_free_mb": free,
                            "memory_used_mb": used,
                            "memory_free_ratio": (
                                free / total
                                if total and free is not None and total > 0
                                else None
                            ),
                            "temperature_c": _to_float_or_none(parts[4]),
                            "utilization_percent": _to_float_or_none(parts[5]),
                        }
                    )

        return {
            "available": True,
            "path": nvidia_smi,
            "source": nvidia_smi_info.get("source"),
            "gpus": gpus,
        }

    except Exception as exc:
        return {
            "available": True,
            "path": nvidia_smi,
            "source": nvidia_smi_info.get("source"),
            "gpus": [],
            "error": str(exc),
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
        for index in range(device_count):
            props = _safe_call(lambda index=index: torch.cuda.get_device_properties(index))
            if props is not None:
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_mb": round(props.total_memory / (1024 ** 2), 3),
                    }
                )

    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
    }


def _probe_gpu_features(include_torch: bool = False) -> Dict[str, Any]:
    nvidia_smi = _probe_nvidia_smi()
    torch_cuda = _probe_torch_cuda() if include_torch else {
        "torch_available": None,
        "cuda_available": None,
        "cuda_device_count": None,
        "cuda_devices": [],
        "skipped": True,
    }

    cuda_available = bool(
        nvidia_smi.get("available", False)
        or torch_cuda.get("cuda_available", False)
    )

    primary_gpu = None
    gpus = nvidia_smi.get("gpus", [])
    if gpus:
        primary_gpu = gpus[0]

    return {
        "cuda_available": cuda_available,
        "nvidia_smi": nvidia_smi,
        "torch_cuda": torch_cuda,
        "primary_gpu": primary_gpu,
    }


def derive_system_constraints(features: Dict[str, Any]) -> Dict[str, Any]:
    cpu = features.get("dynamic", {}).get("cpu", {})
    memory = features.get("dynamic", {}).get("memory", {})
    swap = features.get("dynamic", {}).get("swap", {})
    battery = features.get("dynamic", {}).get("battery", {})
    disk = features.get("dynamic", {}).get("disk", {})
    gpu = features.get("gpu", {})
    primary_gpu = gpu.get("primary_gpu") or {}

    cpu_percent = _to_float_or_none(cpu.get("cpu_percent"))
    memory_percent = _to_float_or_none(memory.get("percent"))
    memory_available_ratio = _to_float_or_none(memory.get("available_ratio"))
    swap_percent = _to_float_or_none(swap.get("percent"))
    battery_percent = _to_float_or_none(battery.get("percent"))
    power_mode = battery.get("power_mode", "unknown")
    disk_percent = _to_float_or_none(disk.get("percent"))
    disk_free_gb = _to_float_or_none(disk.get("free_gb"))

    gpu_util = _to_float_or_none(primary_gpu.get("utilization_percent"))
    gpu_free_ratio = _to_float_or_none(primary_gpu.get("memory_free_ratio"))
    gpu_temp = _to_float_or_none(primary_gpu.get("temperature_c"))

    is_cpu_busy = cpu_percent is not None and cpu_percent >= 85.0

    is_memory_constrained = (
        (memory_percent is not None and memory_percent >= 85.0)
        or (memory_available_ratio is not None and memory_available_ratio <= 0.20)
    )

    is_memory_critical = (
        (memory_percent is not None and memory_percent >= 95.0)
        or (memory_available_ratio is not None and memory_available_ratio <= 0.10)
    )

    is_swap_active = swap_percent is not None and swap_percent >= 20.0

    is_on_battery = power_mode == "battery"
    is_battery_low = is_on_battery and battery_percent is not None and battery_percent <= 30.0
    is_battery_critical = is_on_battery and battery_percent is not None and battery_percent <= 15.0

    is_gpu_available = bool(gpu.get("cuda_available", False))
    is_gpu_busy = gpu_util is not None and gpu_util >= 80.0
    is_gpu_memory_constrained = gpu_free_ratio is not None and gpu_free_ratio <= 0.15

    is_thermal_constrained = gpu_temp is not None and gpu_temp >= 80.0
    is_thermal_critical = gpu_temp is not None and gpu_temp >= 90.0

    is_disk_constrained = (
        (disk_percent is not None and disk_percent >= 90.0)
        or (disk_free_gb is not None and disk_free_gb <= 2.0)
    )

    return {
        "is_cpu_busy": is_cpu_busy,
        "is_memory_constrained": is_memory_constrained,
        "is_memory_critical": is_memory_critical,
        "is_swap_active": is_swap_active,
        "is_on_battery": is_on_battery,
        "is_battery_low": is_battery_low,
        "is_battery_critical": is_battery_critical,
        "is_gpu_available": is_gpu_available,
        "is_gpu_busy": is_gpu_busy,
        "is_gpu_memory_constrained": is_gpu_memory_constrained,
        "is_thermal_constrained": is_thermal_constrained,
        "is_thermal_critical": is_thermal_critical,
        "is_disk_constrained": is_disk_constrained,
        "classes": {
            "cpu": "busy" if is_cpu_busy else "normal",
            "memory": (
                "critical"
                if is_memory_critical
                else "constrained"
                if is_memory_constrained
                else "normal"
            ),
            "battery": (
                "critical"
                if is_battery_critical
                else "low"
                if is_battery_low
                else power_mode
            ),
            "gpu": (
                "unavailable"
                if not is_gpu_available
                else "memory_constrained"
                if is_gpu_memory_constrained
                else "busy"
                if is_gpu_busy
                else "available"
            ),
            "thermal": (
                "critical"
                if is_thermal_critical
                else "hot"
                if is_thermal_constrained
                else "unknown"
                if gpu_temp is None
                else "nominal"
            ),
            "disk": "low_space" if is_disk_constrained else "normal",
        },
    }


def estimate_probe_efficiency(
    *,
    probe_overhead_ms: float,
    reference_time_ms: Optional[float],
) -> Dict[str, Any]:
    if reference_time_ms is None or reference_time_ms <= 0:
        return {
            "enabled": False,
            "reason": "missing_reference_time_ms",
            "probe_overhead_ms": probe_overhead_ms,
            "reference_time_ms": reference_time_ms,
            "overhead_ratio": None,
            "classification": None,
        }

    ratio = probe_overhead_ms / reference_time_ms

    if ratio < 0.01:
        classification = "excellent"
    elif ratio < 0.05:
        classification = "acceptable"
    elif ratio < 0.10:
        classification = "warning"
    else:
        classification = "too_high"

    return {
        "enabled": True,
        "probe_overhead_ms": probe_overhead_ms,
        "reference_time_ms": reference_time_ms,
        "overhead_ratio": ratio,
        "classification": classification,
    }


def build_system_features(
    *,
    probe_level: str = "basic",
    cache_ttl_s: float = 5.0,
    cpu_interval_s: float = 0.0,
) -> Dict[str, Any]:
    probe_level = probe_level.strip().lower()

    if probe_level not in {"basic", "gpu", "full"}:
        raise ValueError("probe_level must be one of: basic, gpu, full")

    cache_report = {
        "static_cache_hit": False,
        "dynamic_cache_hit": False,
        "gpu_cache_hit": None,
    }

    overhead = {
        "static_probe_ms": 0.0,
        "dynamic_probe_ms": 0.0,
        "gpu_probe_ms": None,
        "total_probe_ms": 0.0,
    }

    total_t0 = _now()

    if _cache_valid("static", cache_ttl_s):
        static_features = _CACHE["static"]
        cache_report["static_cache_hit"] = True
    else:
        t0 = _now()
        static_features = _probe_static_features()
        overhead["static_probe_ms"] = _elapsed_ms(t0)
        _CACHE["static"] = static_features
        _CACHE_TS["static"] = _now()

    if _cache_valid("dynamic", cache_ttl_s):
        dynamic_features = _CACHE["dynamic"]
        cache_report["dynamic_cache_hit"] = True
    else:
        t0 = _now()
        dynamic_features = _probe_dynamic_features(cpu_interval_s=cpu_interval_s)
        overhead["dynamic_probe_ms"] = _elapsed_ms(t0)
        _CACHE["dynamic"] = dynamic_features
        _CACHE_TS["dynamic"] = _now()

    gpu_features = {
        "cuda_available": None,
        "nvidia_smi": {
            "available": None,
            "skipped": True,
        },
        "torch_cuda": {
            "skipped": True,
        },
        "primary_gpu": None,
        "skipped": True,
    }

    if probe_level in {"gpu", "full"}:
        cache_report["gpu_cache_hit"] = False

        if _cache_valid("gpu", cache_ttl_s):
            gpu_features = _CACHE["gpu"]
            cache_report["gpu_cache_hit"] = True
        else:
            t0 = _now()
            gpu_features = _probe_gpu_features(include_torch=probe_level == "full")
            overhead["gpu_probe_ms"] = _elapsed_ms(t0)
            _CACHE["gpu"] = gpu_features
            _CACHE_TS["gpu"] = _now()

    features = {
        "static": static_features,
        "dynamic": dynamic_features,
        "gpu": gpu_features,
    }

    overhead["total_probe_ms"] = _elapsed_ms(total_t0)

    constraints = derive_system_constraints(features)

    return {
        "enabled": True,
        "version": "0.8",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "probe_level": probe_level,
        "cache_ttl_s": cache_ttl_s,
        "cpu_interval_s": cpu_interval_s,
        "cache": cache_report,
        "probe_overhead": overhead,
        "features": features,
        "derived_constraints": constraints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cheap system-aware features for the R-D-E router."
    )

    parser.add_argument(
        "--probe-level",
        default="basic",
        choices=["basic", "gpu", "full"],
    )

    parser.add_argument(
        "--cache-ttl-s",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--cpu-interval-s",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--out",
        default=None,
    )

    args = parser.parse_args()

    report = build_system_features(
        probe_level=args.probe_level,
        cache_ttl_s=args.cache_ttl_s,
        cpu_interval_s=args.cpu_interval_s,
    )

    print("\n=== R-D-E System Features ===")
    print(f"Probe level: {report['probe_level']}")
    print(f"Total probe overhead ms: {report['probe_overhead']['total_probe_ms']:.3f}")
    print(f"CPU class: {report['derived_constraints']['classes']['cpu']}")
    print(f"Memory class: {report['derived_constraints']['classes']['memory']}")
    print(f"Battery class: {report['derived_constraints']['classes']['battery']}")
    print(f"GPU class: {report['derived_constraints']['classes']['gpu']}")
    print(f"Thermal class: {report['derived_constraints']['classes']['thermal']}")
    print(f"Disk class: {report['derived_constraints']['classes']['disk']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()