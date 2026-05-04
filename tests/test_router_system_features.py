import json
from pathlib import Path

from src.router.rde_router import main


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "router_system_features"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_router_report_includes_system_features():
    csv_path = _tmp_path("points.csv")
    out_path = _tmp_path("report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=85,1.0,80.0,1.0,10.0",
            ]
        ),
        encoding="utf-8",
    )

    main(
        [
            "--csv",
            str(csv_path),
            "--codec-col",
            "codec",
            "--config-col",
            "param",
            "--rate-col",
            "bpp",
            "--quality-col",
            "ssimulacra2",
            "--energy-col",
            "energy_per_image_j",
            "--time-col",
            "time_ms",
            "--available-codecs",
            "JPEG",
            "--quality-target",
            "normal",
            "--system-features",
            "--system-probe-level",
            "basic",
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert report["system_features"]["enabled"] is True
    assert report["system_features"]["probe_level"] == "basic"
    assert "probe_overhead" in report["system_features"]
    assert "derived_constraints" in report["system_features"]

    assert report["system_probe_efficiency"]["enabled"] is True
    assert report["system_probe_efficiency"]["reference_time_ms"] == 10.0
    assert report["system_probe_efficiency"]["probe_overhead_ms"] >= 0.0
