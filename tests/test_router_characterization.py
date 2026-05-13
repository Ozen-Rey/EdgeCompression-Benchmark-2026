import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.router.rde_router import main


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "router_characterization"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _real_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "image_rde_real_small.csv"


def _run_router(args: list[str], out_name: str) -> dict:
    out_path = _tmp_path(out_name)
    main([*args, "--out", str(out_path)])
    return json.loads(out_path.read_text(encoding="utf-8"))


def _real_fixture_args() -> list[str]:
    return [
        "--csv",
        str(_real_fixture()),
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
    ]


def test_base_router_characterization_without_external_codec():
    report = _run_router(_real_fixture_args(), "base_report.json")
    selected = report["decision"]["selected"]

    assert report["external_codecs"] == {"enabled": False}
    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
    assert selected["cost"] == pytest.approx(0.66)


def test_content_policy_apply_characterization():
    csv_path = _tmp_path("content_apply_rde.csv")
    rules_path = _tmp_path("content_apply_rules.csv")
    csv_path.write_text(
        "\n".join(
            [
                "dataset,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,JPEG,q=85,1.0,95.0,1.0,20.0",
                "A,HEVC,crf=15,1.2,85.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )
    rules_path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,A,JPEG,q=85",
            ]
        ),
        encoding="utf-8",
    )

    report = _run_router(
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
            "apply",
            "--content-policy-rules-file",
            str(rules_path),
            "--content-policy-key",
            "dataset",
            "--content-source",
            "A",
            "--content-source-filter",
            "--content-filter-column",
            "dataset",
        ],
        "content_apply_report.json",
    )

    selected = report["decision"]["selected"]
    assert selected["codec"] == "JPEG"
    assert selected["config"] == "q=85"
    assert report["content_policy"]["applied"] is True
    assert (
        report["decision"]["decision_trace"]["selected_reason"]
        == "content_policy_preferred_candidate"
    )


def test_degraded_fallback_characterization():
    csv_path = _tmp_path("degraded_fallback_rde.csv")
    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=60,0.8,70.0,1.0,10.0",
                "JXL,d=1.0,1.0,85.0,2.0,20.0",
            ]
        ),
        encoding="utf-8",
    )

    report = _run_router(
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
            "--quality-floor",
            "90",
            "--allow-degraded-fallback",
            "--near-quality-floor",
            "80",
        ],
        "degraded_fallback_report.json",
    )

    assert report["decision"]["decision_mode"] == "degraded_fallback"
    assert report["decision"]["selected"]["codec"] == "JXL"
    assert (
        report["decision"]["decision_trace"]["selected_reason"]
        == "lowest_J_RDE_in_degraded_fallback_pool"
    )


def test_energy_provenance_report_blocks_characterization():
    report = _run_router(_real_fixture_args(), "energy_blocks_report.json")
    selected = report["decision"]["selected"]

    assert selected["energy_provenance_tier"] == "benchmark_reference"
    assert "energy_provenance_summary" in report
    assert report["energy_provenance_summary"]["selected_tier"] == "benchmark_reference"
    assert "energy_provenance_compatibility" in report
    assert report["energy_provenance_compatibility"]["severity"] == "ok"
    assert report["energy_tier_policy"]["mode"] == "report-only"


def test_normalization_audit_characterization():
    report = _run_router(_real_fixture_args(), "normalization_report.json")

    audit = report["normalization_audit"]
    assert audit["mode"] == "runtime"
    assert audit["scales_source"] == "computed_at_runtime"
    assert audit["computed_at_runtime"] is True
    assert report["normalization"]["scope"] == "runtime_global_before_codec_filtering"


def test_decision_receipt_characterization():
    report = _run_router(_real_fixture_args(), "receipt_report.json")
    selected = report["decision"]["selected"]
    receipt = report["decision_receipt"]

    assert receipt["artifact_type"] == "router_decision_receipt"
    assert receipt["receipt_schema_version"] == "0.29.0"
    assert receipt["decision"]["selected_codec"] == selected["codec"]
    assert receipt["decision"]["selected_config"] == selected["config"]
    assert receipt["decision"]["cost"] == selected["cost"]


def test_external_codec_disabled_by_default_characterization():
    report = _run_router(_real_fixture_args(), "external_disabled_report.json")

    assert report["external_codecs"] == {"enabled": False}
    assert report["decision"]["num_points_total"] == 6
    assert all(
        item["raw"].get("source") != "external_codec_manifest"
        for item in report["decision"]["scored_candidate_pool"]
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "src.router.rde_router",
        "src.router.external_codec_spec",
        "src.router.external_codec_probe",
        "src.router.external_codec_dry_run",
        "src.router.external_codec_benchmark",
        "src.router.external_codec_rde_exporter",
    ],
)
def test_cli_help_smoke(module_name: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
