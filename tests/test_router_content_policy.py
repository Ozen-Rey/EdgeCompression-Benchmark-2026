import json
from pathlib import Path

from src.router.rde_router import main


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "router_content_policy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_router_reports_content_policy_suggestion():
    csv_path = _tmp_path("rde.csv")
    rules_path = _tmp_path("rules.csv")
    report_path = _tmp_path("report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=85,1.2,85.0,1.0,20.0",
                "HEVC,crf=15,1.0,95.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    rules_path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,tecnick,JPEG,q=85",
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
            "--domain",
            "image",
            "--quality-target",
            "high",
            "--quality-floor",
            "80",
            "--content-policy",
            "--content-policy-mode",
            "report-only",
            "--content-policy-rules-file",
            str(rules_path),
            "--content-policy-key",
            "dataset",
            "--content-source",
            "tecnick",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["content_policy"]["enabled"] is True
    assert report["content_policy"]["mode"] == "report-only"
    assert report["content_policy"]["policy_key"] == "dataset"
    assert report["content_policy"]["policy_value"] == "tecnick"
    assert report["content_policy"]["suggestion"]["codec"] == "JPEG"
    assert report["content_policy"]["suggestion"]["config"] == "q=85"
