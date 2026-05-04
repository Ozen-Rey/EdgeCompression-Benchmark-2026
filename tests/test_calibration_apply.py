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
