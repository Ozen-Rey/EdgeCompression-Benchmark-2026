from dataclasses import dataclass
import json
from pathlib import Path

from src.router.calibration_apply import apply_local_calibration


@dataclass(frozen=True)
class Point:
    codec: str
    config: str
    rate: float
    quality: float
    energy: float
    time_ms: float


def test_apply_local_calibration_updates_rate_time_and_energy():
    calibration = {
        "version": "0.3",
        "level": "quick",
        "created_at": "test",
        "measured": ["time_ms", "output_bytes", "local_bpp"],
        "summary": {
            "JXL": {
                "d=1.0": {
                    "success_rate": 1.0,
                    "time_ms": {"mean": 300.0},
                    "local_bpp": {"mean": 0.8},
                    "output_bytes": {"mean": 1000.0},
                }
            }
        },
    }

    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    points = [
        Point("JXL", "d=1.0", rate=1.2, quality=85.0, energy=2.0, time_ms=100.0),
        Point("JPEG", "q=60", rate=0.9, quality=67.0, energy=0.1, time_ms=5.0),
    ]

    calibrated, report = apply_local_calibration(points, str(path))

    jxl = calibrated[0]
    jpeg = calibrated[1]

    assert jxl.rate == 0.8
    assert jxl.time_ms == 300.0
    assert jxl.energy == 6.0

    assert jpeg.rate == 0.9
    assert jpeg.time_ms == 5.0
    assert jpeg.energy == 0.1

    assert report["enabled"] is True
    assert report["num_applied"] == 1
    assert report["applied"][0]["rate_before"] == 1.2
    assert report["applied"][0]["rate_after"] == 0.8
    assert report["applied"][0]["time_scale"] == 3.0
    assert (
        report["applied"][0]["energy_scaling_method"]
        == "benchmark_energy_scaled_by_time_ratio"
    )
    assert report["estimated"] == ["energy_by_time_scaling"]


def test_auto_uses_local_energy_only_when_usable_for_total():
    calibration = {
        "version": "0.10.0",
        "level": "quick",
        "energy_mode": "auto",
        "created_at": "test",
        "measured": [
            "time_ms",
            "output_bytes",
            "local_bpp",
            "local_energy_j_if_backend_available",
        ],
        "energy_measurement": {
            "enabled": True,
            "backend_dependent": True,
        },
        "summary": {
            "JXL": {
                "d=1.0": {
                    "success_rate": 1.0,
                    "time_ms": {"mean": 300.0},
                    "local_bpp": {"mean": 0.8},
                    "local_energy_j": {"mean": 4.2},
                    "energy_is_measured": True,
                    "energy_scope": "cpu",
                    "energy_usable_for_total": True,
                    "energy_backend": "cpu=linux_rapl;gpu=none",
                    "energy_method": "cpu=rapl_package_energy_uj_delta;gpu=unavailable",
                    "energy_quality": "cpu=hardware_counter;gpu=not_measured",
                }
            }
        },
    }

    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / "calibration_measured_energy.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    points = [
        Point("JXL", "d=1.0", rate=1.2, quality=85.0, energy=2.0, time_ms=100.0),
    ]

    calibrated, report = apply_local_calibration(points, str(path))

    assert calibrated[0].energy == 4.2
    assert report["estimated"] == []
    assert report["applied"][0]["local_energy_j"] == 4.2
    assert report["applied"][0]["energy_is_measured"] is True
    assert report["applied"][0]["energy_scope"] == "cpu"
    assert report["applied"][0]["energy_usable_for_total"] is True
    assert report["applied"][0]["energy_backend"] == "cpu=linux_rapl;gpu=none"
    assert (
        report["applied"][0]["energy_scaling_method"]
        == "local_hardware_energy_total"
    )


def test_partial_gpu_only_energy_is_not_used_as_total_energy():
    calibration = {
        "version": "0.10.0",
        "level": "quick",
        "energy_mode": "auto",
        "created_at": "test",
        "measured": [
            "time_ms",
            "output_bytes",
            "local_bpp",
            "local_energy_j_if_backend_available",
        ],
        "summary": {
            "JXL": {
                "d=1.0": {
                    "success_rate": 1.0,
                    "time_ms": {"mean": 300.0},
                    "local_bpp": {"mean": 0.8},
                    "local_energy_j": {"mean": 4.2},
                    "energy_is_measured": True,
                    "energy_scope": "gpu",
                    "energy_usable_for_total": False,
                    "energy_backend": "cpu=none;gpu=nvidia_nvml_total_energy",
                    "energy_method": "cpu=unavailable;gpu=nvml_total_energy_counter_delta",
                    "energy_quality": "cpu=not_measured;gpu=hardware_counter",
                }
            }
        },
    }

    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / "calibration_gpu_only_energy.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    points = [
        Point("JXL", "d=1.0", rate=1.2, quality=85.0, energy=2.0, time_ms=100.0),
    ]

    calibrated, report = apply_local_calibration(points, str(path))

    assert calibrated[0].energy == 6.0
    assert calibrated[0].energy != 4.2
    assert report["estimated"] == ["energy_by_time_scaling"]
    assert report["applied"][0]["local_energy_j"] == 4.2
    assert report["applied"][0]["energy_is_measured"] is True
    assert report["applied"][0]["energy_scope"] == "gpu"
    assert report["applied"][0]["energy_usable_for_total"] is False
    assert (
        report["applied"][0]["energy_scaling_method"]
        == "benchmark_energy_scaled_by_time_ratio_"
        "local_measurement_partial_not_comparable"
    )


def test_benchmark_only_ignores_usable_local_energy():
    calibration = {
        "version": "0.10.1",
        "level": "quick",
        "energy_mode": "benchmark-only",
        "created_at": "test",
        "measured": [
            "time_ms",
            "output_bytes",
            "local_bpp",
            "local_energy_j_if_backend_available",
        ],
        "summary": {
            "JXL": {
                "d=1.0": {
                    "success_rate": 1.0,
                    "time_ms": {"mean": 300.0},
                    "local_bpp": {"mean": 0.8},
                    "local_energy_j": {"mean": 4.2},
                    "energy_is_measured": True,
                    "energy_scope": "cpu",
                    "energy_usable_for_total": True,
                    "energy_backend": "cpu=linux_rapl;gpu=none",
                    "energy_method": "cpu=rapl_package_energy_uj_delta;gpu=unavailable",
                    "energy_quality": "cpu=hardware_counter;gpu=not_measured",
                }
            }
        },
    }

    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / "calibration_benchmark_only_energy.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    points = [
        Point("JXL", "d=1.0", rate=1.2, quality=85.0, energy=2.0, time_ms=100.0),
    ]

    calibrated, report = apply_local_calibration(points, str(path))

    assert calibrated[0].energy == 6.0
    assert calibrated[0].energy != 4.2
    assert report["energy_mode"] == "benchmark-only"
    assert report["estimated"] == ["energy_by_time_scaling"]
    assert report["applied"][0]["energy_mode"] == "benchmark-only"
    assert (
        report["applied"][0]["energy_scaling_method"]
        == "benchmark_only_energy_mode"
    )


def test_require_measured_total_rejects_partial_gpu_only_energy():
    calibration = {
        "version": "0.10.1",
        "level": "quick",
        "energy_mode": "require-measured-total",
        "created_at": "test",
        "measured": [
            "time_ms",
            "output_bytes",
            "local_bpp",
            "local_energy_j_if_backend_available",
        ],
        "summary": {
            "JXL": {
                "d=1.0": {
                    "success_rate": 1.0,
                    "time_ms": {"mean": 300.0},
                    "local_bpp": {"mean": 0.8},
                    "local_energy_j": {"mean": 4.2},
                    "energy_is_measured": True,
                    "energy_scope": "gpu",
                    "energy_usable_for_total": False,
                    "energy_backend": "cpu=none;gpu=nvidia_nvml_total_energy",
                    "energy_method": "cpu=unavailable;gpu=nvml_total_energy_counter_delta",
                    "energy_quality": "cpu=not_measured;gpu=hardware_counter",
                }
            }
        },
    }

    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / "calibration_strict_gpu_only_energy.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    points = [
        Point("JXL", "d=1.0", rate=1.2, quality=85.0, energy=2.0, time_ms=100.0),
    ]

    calibrated, report = apply_local_calibration(points, str(path))

    assert calibrated[0].rate == 1.2
    assert calibrated[0].time_ms == 100.0
    assert calibrated[0].energy == 2.0
    assert report["energy_mode"] == "require-measured-total"
    assert report["num_applied"] == 0
    assert report["num_skipped"] == 1
    assert report["skipped_preview"][0]["reason"] == "strict_energy_missing_usable_total"
    assert report["skipped_preview"][0]["energy_scope"] == "gpu"
    assert report["skipped_preview"][0]["energy_usable_for_total"] is False


def test_strict_energy_mode_rejects_windows_gpu_only():
    calibration = {
        "version": "0.11.0",
        "level": "quick",
        "energy_mode": "require-measured-total",
        "created_at": "test",
        "measured": [
            "time_ms",
            "output_bytes",
            "local_bpp",
            "local_energy_j_if_backend_available",
        ],
        "summary": {
            "JPEG": {
                "q=60": {
                    "success_rate": 1.0,
                    "time_ms": {"mean": 200.0},
                    "local_bpp": {"mean": 0.7},
                    "local_energy_j": {"mean": 1.5},
                    "energy_is_measured": True,
                    "energy_scope": "gpu",
                    "energy_usable_for_total": False,
                    "energy_backend": "cpu=none;gpu=nvml_total_counter",
                    "energy_method": (
                        "cpu=unavailable;"
                        "gpu=nvml_total_energy_counter_delta"
                    ),
                    "energy_quality": "cpu=not_measured;gpu=hardware_counter",
                }
            }
        },
    }

    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / "calibration_strict_windows_gpu_only_energy.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    points = [
        Point("JPEG", "q=60", rate=1.1, quality=80.0, energy=3.0, time_ms=100.0),
    ]

    calibrated, report = apply_local_calibration(points, str(path))

    assert calibrated[0].rate == 1.1
    assert calibrated[0].time_ms == 100.0
    assert calibrated[0].energy == 3.0
    assert report["num_applied"] == 0
    assert report["num_skipped"] == 1
    assert report["skipped_preview"][0]["reason"] == "strict_energy_missing_usable_total"
    assert report["skipped_preview"][0]["energy_backend"] == (
        "cpu=none;gpu=nvml_total_counter"
    )
