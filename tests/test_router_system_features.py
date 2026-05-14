import json
from pathlib import Path

from src.router.rde_router import main
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "router_system_features"
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


def test_router_report_includes_system_policy_simulation():
    csv_path = _tmp_path("simulation_points.csv")
    out_path = _tmp_path("simulation_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=85,1.0,80.0,1.0,10.0",
                "JXL,d=1.0,0.8,75.0,0.5,20.0",
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
            "JPEG,JXL",
            "--quality-target",
            "normal",
            "--system-features",
            "--system-policy",
            "--system-policy-mode",
            "apply",
            "--system-policy-simulate",
            "battery=critical,cpu=busy,memory=constrained",
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))

    simulation = report["system_policy_simulation"]
    policy = report["system_policy"]

    assert simulation["enabled"] is True
    assert simulation["classes"]["battery"] == "critical"
    assert simulation["classes"]["cpu"] == "busy"
    assert simulation["classes"]["memory"] == "constrained"

    assert policy["enabled"] is True
    assert policy["applied"] is True
    assert policy["effective_weights"]["w_E"] > policy["base_weights"]["w_E"]
    assert "battery_critical_energy_multiplier=3.0" in policy["rules_applied"]


def test_system_policy_apply_can_change_selected_candidate():
    csv_path = _tmp_path("policy_change.csv")
    report_only_path = _tmp_path("report_only.json")
    apply_path = _tmp_path("apply.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "QualityHeavy,mode=quality,1.0,95.0,100.0,200.0",
                "EnergyLight,mode=energy,1.2,85.0,1.0,20.0",
            ]
        ),
        encoding="utf-8",
    )

    common_args = [
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
        "--auto-weights",
        "--quality-target",
        "high",
        "--quality-floor",
        "80",
        "--system-policy",
        "--system-policy-simulate",
        "battery=critical,cpu=busy,memory=constrained",
    ]

    main(
        [
            *common_args,
            "--system-policy-mode",
            "report-only",
            "--out",
            str(report_only_path),
        ]
    )

    main(
        [
            *common_args,
            "--system-policy-mode",
            "apply",
            "--out",
            str(apply_path),
        ]
    )

    report_only = json.loads(report_only_path.read_text(encoding="utf-8"))
    applied = json.loads(apply_path.read_text(encoding="utf-8"))

    assert report_only["system_policy"]["applied"] is False
    assert applied["system_policy"]["applied"] is True

    assert report_only["decision"]["selected"]["codec"] == "QualityHeavy"
    assert applied["decision"]["selected"]["codec"] == "EnergyLight"

    assert (
        applied["system_policy"]["effective_weights"]["w_E"]
        > applied["system_policy"]["base_weights"]["w_E"]
    )


def test_system_penalty_apply_can_change_selected_candidate():
    csv_path = _tmp_path("penalty_change.csv")
    report_only_path = _tmp_path("penalty_report_only.json")
    apply_path = _tmp_path("penalty_apply.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "HEVC,crf=15,1.0,95.0,100.0,200.0",
                "JPEG,q=85,1.2,85.0,1.0,20.0",
            ]
        ),
        encoding="utf-8",
    )

    common_args = [
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
        "--auto-weights",
        "--quality-target",
        "high",
        "--quality-floor",
        "80",
        "--system-policy-simulate",
        "battery=critical,cpu=busy,memory=constrained",
        "--system-penalty",
        "--system-penalty-lambda",
        "1.1",
    ]

    main(
        [
            *common_args,
            "--system-penalty-mode",
            "report-only",
            "--out",
            str(report_only_path),
        ]
    )

    main(
        [
            *common_args,
            "--system-penalty-mode",
            "apply",
            "--out",
            str(apply_path),
        ]
    )

    report_only = json.loads(report_only_path.read_text(encoding="utf-8"))
    applied = json.loads(apply_path.read_text(encoding="utf-8"))

    assert report_only["decision"]["selected"]["codec"] == "HEVC"
    assert applied["decision"]["selected"]["codec"] == "JPEG"

    selected = applied["decision"]["selected"]

    assert selected["system_penalty"]["enabled"] is True
    assert selected["J_total"] >= selected["cost"]
    assert applied["decision"]["decision_trace"]["ranking_key"] == "minimize_J_total"
    assert (
        applied["decision"]["decision_trace"]["selected_reason"]
        == "lowest_J_total_in_safe_pool"
    )
