import time

from src.utils.energy_backends import (
    CompositeEnergyMeter,
    EnergyReading,
    NoEnergyBackend,
    WindowsNvidiaNvmlPowerSamplerBackend,
    WindowsNvidiaNvmlTotalEnergyBackend,
    WindowsPowercfgDiagnosticBackend,
)


class FakeNvmlTotalEnergy:
    def __init__(self):
        self.values = [1000.0, 1000.0, 4500.0]

    def nvmlInit(self):
        return None

    def nvmlDeviceGetHandleByIndex(self, gpu_index):
        return f"gpu-{gpu_index}"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        return self.values.pop(0)


class FakeNvmlPowerUsage:
    def nvmlInit(self):
        return None

    def nvmlDeviceGetHandleByIndex(self, gpu_index):
        return f"gpu-{gpu_index}"

    def nvmlDeviceGetPowerUsage(self, handle):
        return 50_000.0


def test_composite_energy_meter_returns_provenance():
    meter = CompositeEnergyMeter()

    def work():
        return 123

    result, reading = meter.measure_callable(work)

    assert result == 123
    assert isinstance(reading, EnergyReading)
    assert reading.time_s >= 0.0
    assert isinstance(reading.energy_backend, str)
    assert isinstance(reading.energy_method, str)
    assert isinstance(reading.energy_is_measured, bool)
    assert isinstance(reading.energy_quality, str)
    assert isinstance(reading.warnings, list)
    assert isinstance(reading.energy_scope, str)
    assert isinstance(reading.energy_usable_for_total, bool)


def test_windows_nvml_gpu_only_is_partial_not_total():
    gpu_backend = WindowsNvidiaNvmlTotalEnergyBackend(
        nvml=FakeNvmlTotalEnergy(),
        system_name="Windows",
    )
    meter = CompositeEnergyMeter(
        cpu_backend=NoEnergyBackend(),
        gpu_backend=gpu_backend,
    )

    result, reading = meter.measure_callable(lambda: 123)

    assert result == 123
    assert reading.energy_backend == "cpu=none;gpu=nvml_total_counter"
    assert reading.energy_method == (
        "cpu=unavailable;gpu=nvml_total_energy_counter_delta"
    )
    assert reading.energy_quality == "cpu=not_measured;gpu=hardware_counter"
    assert reading.energy_is_measured is True
    assert reading.cpu_j is None
    assert reading.gpu_j == 3.5
    assert reading.energy_scope == "gpu"
    assert reading.energy_usable_for_total is False
    assert "gpu_energy_partial_not_total_pipeline_energy" in reading.warnings


def test_windows_nvml_sampled_power_is_partial_not_total():
    gpu_backend = WindowsNvidiaNvmlPowerSamplerBackend(
        interval_s=0.001,
        nvml=FakeNvmlPowerUsage(),
        system_name="Windows",
    )
    meter = CompositeEnergyMeter(
        cpu_backend=NoEnergyBackend(),
        gpu_backend=gpu_backend,
    )

    state = meter.start()
    time.sleep(0.01)
    reading = meter.stop(state)

    assert reading.energy_backend == "cpu=none;gpu=nvml_power_sampler"
    assert reading.energy_method == (
        "cpu=unavailable;gpu=nvml_power_usage_sampling_integral"
    )
    assert reading.energy_quality == "cpu=not_measured;gpu=sampled_power_integral"
    assert reading.energy_is_measured is True
    assert reading.cpu_j is None
    assert reading.gpu_j is not None
    assert reading.gpu_j > 0.0
    assert reading.energy_scope == "gpu"
    assert reading.energy_usable_for_total is False
    assert "gpu_energy_partial_not_total_pipeline_energy" in reading.warnings


def test_windows_powercfg_is_diagnostic_only():
    backend = WindowsPowercfgDiagnosticBackend(
        system_name="Windows",
        which=lambda name: r"C:\Windows\System32\powercfg.exe"
        if name == "powercfg"
        else None,
    )

    assert backend.available() is True
    reading = backend.stop(backend.start())

    assert reading.energy_backend == "windows_powercfg_diagnostic"
    assert reading.energy_method == "powercfg_diagnostic_only"
    assert reading.energy_quality == "diagnostic_only"
    assert reading.energy_is_measured is False
    assert reading.energy_scope == "diagnostic"
    assert reading.energy_usable_for_total is False
    assert reading.total_j is None
    assert "powercfg_diagnostic_only_not_per_command_energy" in reading.warnings


def test_windows_no_backend_falls_back_cleanly():
    meter = CompositeEnergyMeter(
        cpu_backend=NoEnergyBackend(),
        gpu_backend=NoEnergyBackend(),
    )

    result, reading = meter.measure_callable(lambda: 123)

    assert result == 123
    assert reading.energy_backend == "cpu=none;gpu=none"
    assert reading.energy_method == "cpu=unavailable;gpu=unavailable"
    assert reading.energy_is_measured is False
    assert reading.energy_quality == "cpu=not_measured;gpu=not_measured"
    assert reading.energy_scope == "none"
    assert reading.energy_usable_for_total is False
    assert reading.total_j is None
