"""Hardware energy measurement backends.

Design goals:
- Prefer real hardware counters when available.
- Never silently pretend that estimated energy is measured energy.
- Always return explicit provenance.
- Support Linux RAPL + NVIDIA NVML first.
- Support Windows NVIDIA NVML first; Intel PCM / AMD uProf as external-tool
  backends in a later patch if not already installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import glob
import platform
import shutil
import threading
import time
from typing import Any, Callable, Optional


@dataclass
class EnergyReading:
    cpu_j: Optional[float]
    gpu_j: Optional[float]
    total_j: Optional[float]
    time_s: float
    energy_backend: str
    energy_method: str
    energy_is_measured: bool
    energy_quality: str
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    energy_scope: str = "none"
    energy_usable_for_total: bool = False


class EnergyBackend:
    name = "base"
    method = "unknown"
    quality = "unknown"

    def available(self) -> bool:
        raise NotImplementedError

    def start(self) -> Any:
        raise NotImplementedError

    def stop(self, state: Any) -> EnergyReading:
        raise NotImplementedError


class LinuxRaplBackend(EnergyBackend):
    name = "linux_rapl"
    method = "rapl_package_energy_uj_delta"
    quality = "hardware_counter"

    def __init__(self) -> None:
        self.zones = self._discover_package_zones()

    @staticmethod
    def _discover_package_zones() -> list[Path]:
        zones: list[Path] = []
        for p in glob.glob("/sys/class/powercap/*-rapl:*"):
            path = Path(p)
            if path.name.count(":") == 1 and (path / "energy_uj").exists():
                zones.append(path)
        return sorted(zones)

    def available(self) -> bool:
        return platform.system().lower() == "linux" and bool(self.zones)

    @staticmethod
    def _read_zone_uj(zone: Path) -> int:
        return int((zone / "energy_uj").read_text().strip())

    @staticmethod
    def _read_zone_max_uj(zone: Path) -> Optional[int]:
        p = zone / "max_energy_range_uj"
        if not p.exists():
            return None
        return int(p.read_text().strip())

    def _read_all(self) -> dict[str, int]:
        return {str(z): self._read_zone_uj(z) for z in self.zones}

    def start(self) -> dict[str, Any]:
        return {
            "t0": time.perf_counter(),
            "energy_uj": self._read_all(),
            "zones": [str(z) for z in self.zones],
        }

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        t1 = time.perf_counter()
        after = self._read_all()
        total_j = 0.0
        warnings: list[str] = []

        for zone_str, before_uj in state["energy_uj"].items():
            zone = Path(zone_str)
            after_uj = after.get(zone_str)

            if after_uj is None:
                warnings.append(f"rapl_zone_missing_after:{zone_str}")
                continue

            delta = after_uj - before_uj
            if delta < 0:
                max_uj = self._read_zone_max_uj(zone)
                if max_uj is None:
                    warnings.append(f"rapl_wrap_without_max:{zone_str}")
                    continue
                delta = (max_uj - before_uj) + after_uj

            total_j += delta / 1e6

        return EnergyReading(
            cpu_j=total_j,
            gpu_j=None,
            total_j=total_j,
            time_s=t1 - float(state["t0"]),
            energy_backend=self.name,
            energy_method=self.method,
            energy_is_measured=True,
            energy_quality=self.quality,
            warnings=warnings,
            raw={"zones": state["zones"]},
            energy_scope="cpu",
            energy_usable_for_total=True,
        )


class NvidiaNvmlEnergyBackend(EnergyBackend):
    name = "nvidia_nvml_total_energy"
    method = "nvml_total_energy_counter_delta"
    quality = "hardware_counter"

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self.nvml = None
        self.handle = None
        self._init_error: Optional[str] = None

        try:
            import pynvml  # type: ignore

            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception as exc:
            self._init_error = repr(exc)
            self.nvml = None
            self.handle = None

    def available(self) -> bool:
        return self.nvml is not None and self.handle is not None

    def start(self) -> dict[str, Any]:
        assert self.nvml is not None and self.handle is not None
        return {
            "t0": time.perf_counter(),
            "gpu_mj": float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)),
            "gpu_index": self.gpu_index,
        }

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        assert self.nvml is not None and self.handle is not None
        t1 = time.perf_counter()
        after_mj = float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        delta_j = max((after_mj - float(state["gpu_mj"])) / 1000.0, 0.0)

        return EnergyReading(
            cpu_j=None,
            gpu_j=delta_j,
            total_j=delta_j,
            time_s=t1 - float(state["t0"]),
            energy_backend=self.name,
            energy_method=self.method,
            energy_is_measured=True,
            energy_quality=self.quality,
            warnings=[],
            raw={"gpu_index": state["gpu_index"]},
            energy_scope="gpu",
            energy_usable_for_total=False,
        )


class NvidiaNvmlPowerSamplerBackend(EnergyBackend):
    name = "nvidia_nvml_power_sampler"
    method = "nvml_power_usage_sampling_integral"
    quality = "sampled_power_estimate"

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.05) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.nvml = None
        self.handle = None
        self._init_error: Optional[str] = None

        try:
            import pynvml  # type: ignore

            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.nvml.nvmlDeviceGetPowerUsage(self.handle)
        except Exception as exc:
            self._init_error = repr(exc)
            self.nvml = None
            self.handle = None

    def available(self) -> bool:
        return self.nvml is not None and self.handle is not None

    def _sample_loop(self, state: dict[str, Any]) -> None:
        assert self.nvml is not None and self.handle is not None

        while not state["stop"]:
            now = time.perf_counter()
            try:
                power_w = float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
                state["samples"].append((now, power_w))
            except Exception as exc:
                state["warnings"].append(f"nvml_power_sample_failed:{repr(exc)}")
            time.sleep(self.interval_s)

    def start(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "t0": time.perf_counter(),
            "samples": [],
            "warnings": [],
            "stop": False,
            "gpu_index": self.gpu_index,
            "interval_s": self.interval_s,
        }

        thread = threading.Thread(target=self._sample_loop, args=(state,), daemon=True)
        state["thread"] = thread
        thread.start()
        return state

    @staticmethod
    def _integrate_samples(samples: list[tuple[float, float]]) -> float:
        if len(samples) < 2:
            return 0.0

        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(samples[:-1], samples[1:]):
            dt = max(t1 - t0, 0.0)
            energy_j += 0.5 * (p0 + p1) * dt
        return energy_j

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        t1 = time.perf_counter()
        state["stop"] = True
        state["thread"].join(timeout=1.0)

        samples = list(state["samples"])
        gpu_j = self._integrate_samples(samples)

        warnings = list(state["warnings"])
        if len(samples) < 2:
            warnings.append("nvml_power_sampler_too_few_samples")

        return EnergyReading(
            cpu_j=None,
            gpu_j=gpu_j,
            total_j=gpu_j,
            time_s=t1 - float(state["t0"]),
            energy_backend=self.name,
            energy_method=self.method,
            energy_is_measured=True,
            energy_quality=self.quality,
            warnings=warnings,
            raw={
                "gpu_index": state["gpu_index"],
                "interval_s": state["interval_s"],
                "num_samples": len(samples),
            },
            energy_scope="gpu",
            energy_usable_for_total=False,
        )


class WindowsNvidiaNvmlTotalEnergyBackend(EnergyBackend):
    name = "nvml_total_counter"
    method = "nvml_total_energy_counter_delta"
    quality = "hardware_counter"

    def __init__(
        self,
        gpu_index: int = 0,
        nvml: Any = None,
        system_name: Optional[str] = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.nvml = None
        self.handle = None
        self._init_error: Optional[str] = None
        self.system_name = system_name or platform.system()

        if self.system_name.lower() != "windows":
            self._init_error = "windows_nvml_backend_not_on_windows"
            return

        try:
            if nvml is None:
                import pynvml  # type: ignore

                nvml = pynvml

            self.nvml = nvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception as exc:
            self._init_error = repr(exc)
            self.nvml = None
            self.handle = None

    def available(self) -> bool:
        return self.nvml is not None and self.handle is not None

    def start(self) -> dict[str, Any]:
        assert self.nvml is not None and self.handle is not None
        return {
            "t0": time.perf_counter(),
            "gpu_mj": float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)),
            "gpu_index": self.gpu_index,
        }

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        assert self.nvml is not None and self.handle is not None
        t1 = time.perf_counter()
        after_mj = float(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        delta_j = max((after_mj - float(state["gpu_mj"])) / 1000.0, 0.0)

        return EnergyReading(
            cpu_j=None,
            gpu_j=delta_j,
            total_j=delta_j,
            time_s=t1 - float(state["t0"]),
            energy_backend=self.name,
            energy_method=self.method,
            energy_is_measured=True,
            energy_quality=self.quality,
            warnings=["gpu_energy_partial_not_total_pipeline_energy"],
            raw={"gpu_index": state["gpu_index"], "platform": "windows"},
            energy_scope="gpu",
            energy_usable_for_total=False,
        )


class WindowsNvidiaNvmlPowerSamplerBackend(EnergyBackend):
    name = "nvml_power_sampler"
    method = "nvml_power_usage_sampling_integral"
    quality = "sampled_power_integral"

    def __init__(
        self,
        gpu_index: int = 0,
        interval_s: float = 0.05,
        nvml: Any = None,
        system_name: Optional[str] = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.nvml = None
        self.handle = None
        self._init_error: Optional[str] = None
        self.system_name = system_name or platform.system()

        if self.system_name.lower() != "windows":
            self._init_error = "windows_nvml_backend_not_on_windows"
            return

        try:
            if nvml is None:
                import pynvml  # type: ignore

                nvml = pynvml

            self.nvml = nvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.nvml.nvmlDeviceGetPowerUsage(self.handle)
        except Exception as exc:
            self._init_error = repr(exc)
            self.nvml = None
            self.handle = None

    def available(self) -> bool:
        return self.nvml is not None and self.handle is not None

    def _sample_loop(self, state: dict[str, Any]) -> None:
        assert self.nvml is not None and self.handle is not None

        while not state["stop"]:
            now = time.perf_counter()
            try:
                power_w = float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
                state["samples"].append((now, power_w))
            except Exception as exc:
                state["warnings"].append(f"nvml_power_sample_failed:{repr(exc)}")
            time.sleep(self.interval_s)

    def start(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "t0": time.perf_counter(),
            "samples": [],
            "warnings": ["gpu_energy_partial_not_total_pipeline_energy"],
            "stop": False,
            "gpu_index": self.gpu_index,
            "interval_s": self.interval_s,
        }

        thread = threading.Thread(target=self._sample_loop, args=(state,), daemon=True)
        state["thread"] = thread
        thread.start()
        return state

    @staticmethod
    def _integrate_samples(samples: list[tuple[float, float]]) -> float:
        return NvidiaNvmlPowerSamplerBackend._integrate_samples(samples)

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        t1 = time.perf_counter()
        state["stop"] = True
        state["thread"].join(timeout=1.0)

        samples = list(state["samples"])
        gpu_j = self._integrate_samples(samples)

        warnings = list(state["warnings"])
        if len(samples) < 2:
            warnings.append("nvml_power_sampler_too_few_samples")

        return EnergyReading(
            cpu_j=None,
            gpu_j=gpu_j,
            total_j=gpu_j,
            time_s=t1 - float(state["t0"]),
            energy_backend=self.name,
            energy_method=self.method,
            energy_is_measured=True,
            energy_quality=self.quality,
            warnings=warnings,
            raw={
                "gpu_index": state["gpu_index"],
                "interval_s": state["interval_s"],
                "num_samples": len(samples),
                "platform": "windows",
            },
            energy_scope="gpu",
            energy_usable_for_total=False,
        )


class WindowsIntelPcmPackageBackend(EnergyBackend):
    name = "windows_intel_pcm"
    method = "pcm_external_tool_unavailable"
    quality = "unavailable"

    def __init__(
        self,
        system_name: Optional[str] = None,
        which: Callable[[str], Optional[str]] = shutil.which,
    ) -> None:
        self.system_name = system_name or platform.system()
        self.executable = None

        if self.system_name.lower() == "windows":
            self.executable = which("pcm-power.exe") or which("pcm.exe")

    def available(self) -> bool:
        return False

    def start(self) -> dict[str, Any]:
        raise RuntimeError(
            "Intel PCM Windows backend is detected only. Per-command package "
            "energy parsing is not enabled in router v0.11.0."
        )

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        raise RuntimeError("Intel PCM Windows backend is not active.")

    def diagnostic(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "platform": self.system_name,
            "executable": self.executable,
            "available_for_measurement": False,
            "warning": (
                "Intel PCM executable detection is diagnostic only in v0.11.0; "
                "it is not used for per-command energy calibration yet."
            ),
        }


class WindowsPowercfgDiagnosticBackend(EnergyBackend):
    name = "windows_powercfg_diagnostic"
    method = "powercfg_diagnostic_only"
    quality = "diagnostic_only"

    def __init__(
        self,
        system_name: Optional[str] = None,
        which: Callable[[str], Optional[str]] = shutil.which,
    ) -> None:
        self.system_name = system_name or platform.system()
        self.executable = None

        if self.system_name.lower() == "windows":
            self.executable = which("powercfg")

    def available(self) -> bool:
        return self.system_name.lower() == "windows" and self.executable is not None

    def start(self) -> dict[str, Any]:
        return {"t0": time.perf_counter(), "executable": self.executable}

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        t1 = time.perf_counter()
        return EnergyReading(
            cpu_j=None,
            gpu_j=None,
            total_j=None,
            time_s=t1 - float(state["t0"]),
            energy_backend=self.name,
            energy_method=self.method,
            energy_is_measured=False,
            energy_quality=self.quality,
            warnings=["powercfg_diagnostic_only_not_per_command_energy"],
            raw={"executable": state.get("executable"), "platform": "windows"},
            energy_scope="diagnostic",
            energy_usable_for_total=False,
        )


class NoEnergyBackend(EnergyBackend):
    name = "none"
    method = "unavailable"
    quality = "not_measured"

    def available(self) -> bool:
        return True

    def start(self) -> dict[str, Any]:
        return {"t0": time.perf_counter()}

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        t1 = time.perf_counter()
        return EnergyReading(
            cpu_j=None,
            gpu_j=None,
            total_j=None,
            time_s=t1 - float(state["t0"]),
            energy_backend=self.name,
            energy_method=self.method,
            energy_is_measured=False,
            energy_quality=self.quality,
            warnings=["no_hardware_energy_backend_available"],
            raw={},
            energy_scope="none",
            energy_usable_for_total=False,
        )


class CompositeEnergyMeter:
    """Measure CPU and GPU energy with the best available independent backends."""

    def __init__(
        self,
        cpu_backend: Optional[EnergyBackend] = None,
        gpu_backend: Optional[EnergyBackend] = None,
    ) -> None:
        self.cpu_backend: EnergyBackend = cpu_backend or self._select_cpu_backend()
        self.gpu_backend: EnergyBackend = gpu_backend or self._select_gpu_backend()

    @staticmethod
    def _select_cpu_backend() -> EnergyBackend:
        candidates: list[EnergyBackend] = [
            LinuxRaplBackend(),
        ]

        for backend in candidates:
            if backend.available():
                return backend

        return NoEnergyBackend()

    @staticmethod
    def _select_gpu_backend() -> EnergyBackend:
        if platform.system().lower() == "windows":
            for factory in (
                WindowsNvidiaNvmlTotalEnergyBackend,
                WindowsNvidiaNvmlPowerSamplerBackend,
            ):
                backend = factory()
                if backend.available():
                    return backend

            return NoEnergyBackend()

        for factory in (NvidiaNvmlEnergyBackend, NvidiaNvmlPowerSamplerBackend):
            backend = factory()
            if backend.available():
                return backend

        return NoEnergyBackend()

    def start(self) -> dict[str, Any]:
        return {
            "cpu": self.cpu_backend.start(),
            "gpu": self.gpu_backend.start(),
        }

    def stop(self, state: dict[str, Any]) -> EnergyReading:
        cpu = self.cpu_backend.stop(state["cpu"])
        gpu = self.gpu_backend.stop(state["gpu"])

        cpu_j = cpu.cpu_j
        gpu_j = gpu.gpu_j

        measured_parts = []
        if cpu_j is not None:
            measured_parts.append(cpu_j)
        if gpu_j is not None:
            measured_parts.append(gpu_j)

        total_j = sum(measured_parts) if measured_parts else None
        energy_is_measured = cpu.energy_is_measured or gpu.energy_is_measured

        warnings = list(dict.fromkeys(cpu.warnings + gpu.warnings))
        scope_parts = []
        if cpu.cpu_j is not None:
            scope_parts.extend(
                part
                for part in cpu.energy_scope.split("+")
                if part and part != "none"
            )
        if gpu.gpu_j is not None:
            scope_parts.extend(
                part
                for part in gpu.energy_scope.split("+")
                if part and part != "none"
            )

        energy_scope = "+".join(dict.fromkeys(scope_parts)) if scope_parts else "none"
        energy_usable_for_total = bool(cpu.energy_usable_for_total)

        return EnergyReading(
            cpu_j=cpu_j,
            gpu_j=gpu_j,
            total_j=total_j,
            time_s=max(cpu.time_s, gpu.time_s),
            energy_backend=f"cpu={cpu.energy_backend};gpu={gpu.energy_backend}",
            energy_method=f"cpu={cpu.energy_method};gpu={gpu.energy_method}",
            energy_is_measured=energy_is_measured,
            energy_quality=f"cpu={cpu.energy_quality};gpu={gpu.energy_quality}",
            warnings=warnings,
            raw={
                "cpu": cpu.raw,
                "gpu": gpu.raw,
            },
            energy_scope=energy_scope,
            energy_usable_for_total=energy_usable_for_total,
        )

    def measure_callable(self, fn: Callable[[], Any]) -> tuple[Any, EnergyReading]:
        state = self.start()
        result = fn()
        reading = self.stop(state)
        return result, reading


def measure_callable_energy(fn: Callable[[], Any]) -> tuple[Any, EnergyReading]:
    return CompositeEnergyMeter().measure_callable(fn)


def collect_energy_backend_diagnostics(
    system_name: Optional[str] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> dict[str, Any]:
    system = system_name or platform.system()
    diagnostics: dict[str, Any] = {
        "platform": system,
        "powercfg": {
            "diagnostic_only": True,
            "available": False,
            "used_for_per_command_energy": False,
        },
        "intel_pcm": WindowsIntelPcmPackageBackend(
            system_name=system,
            which=which,
        ).diagnostic(),
    }

    powercfg = WindowsPowercfgDiagnosticBackend(system_name=system, which=which)
    diagnostics["powercfg"]["available"] = powercfg.available()
    diagnostics["powercfg"]["executable"] = powercfg.executable
    diagnostics["powercfg"]["warning"] = (
        "powercfg is diagnostic only and is not used as Joule telemetry."
    )

    return diagnostics
