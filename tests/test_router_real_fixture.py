import json
from pathlib import Path

from src.router.rde_router import main


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "router_real_fixture"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_router_end_to_end_on_real_small_image_fixture():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )

    out_path = _tmp_path("real_fixture_report.json")

    main(
        [
            "--csv",
            str(fixture),
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
            "JPEG,JXL,HEVC",
            "--quality-target",
            "very-high",
            "--quality-floor",
            "90",
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    selected = report["decision"]["selected"]

    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
    assert report["decision"]["decision_mode"] == "safe"
