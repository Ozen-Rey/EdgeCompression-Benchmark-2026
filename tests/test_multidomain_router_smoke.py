import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.router.core.domain_spec import (
    BUILTIN_DOMAIN_SPECS,
    main as domain_spec_main,
)
from src.router.rde_router import main as router_main


ROOT = Path(__file__).resolve().parents[1]
AUDIO_VISQOL = ROOT / "tests" / "fixtures" / "rde_audio_visqol.csv"
VIDEO_VMAF = ROOT / "tests" / "fixtures" / "rde_video_vmaf.csv"
AUDIO_FAD = ROOT / "tests" / "fixtures" / "rde_audio_fad.csv"
IMAGE_SSIMULACRA2 = ROOT / "tests" / "fixtures" / "image_rde_real_small.csv"


def test_audio_visqol_fixture_validates_with_domain_spec(capsys) -> None:
    report = domain_spec_main(
        ["--csv", str(AUDIO_VISQOL), "--builtin", "audio_visqol", "--validate-csv"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert report["valid"] is True
    assert payload["valid"] is True
    assert payload["num_rows"] == 6
    assert "bitrate_kbps" in payload["detected_columns"]
    assert payload["numeric_validity_summary"]["rate"]["valid"] == 6
    assert payload["normalized_spec"]["domain"] == "audio"


def test_video_vmaf_fixture_validates_with_domain_spec(capsys) -> None:
    report = domain_spec_main(
        ["--csv", str(VIDEO_VMAF), "--builtin", "video_vmaf", "--validate-csv"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert report["valid"] is True
    assert payload["valid"] is True
    assert payload["num_rows"] == 6
    assert "bitrate_kbps" in payload["detected_columns"]
    assert payload["numeric_validity_summary"]["quality"]["valid"] == 6
    assert payload["normalized_spec"]["domain"] == "video"


def _run_router(csv_path: Path, domain_spec: str, tmp_path: Path) -> tuple[dict, Path]:
    report_path = tmp_path / f"{domain_spec}_report.json"
    summary_path = tmp_path / f"{domain_spec}_summary.csv"

    router_main(
        [
            "--csv",
            str(csv_path),
            "--domain-spec",
            domain_spec,
            "--profile",
            "balanced",
            "--out",
            str(report_path),
            "--summary-out",
            str(summary_path),
        ]
    )

    return json.loads(report_path.read_text(encoding="utf-8")), summary_path


def test_router_audio_produces_report_and_summary(tmp_path: Path) -> None:
    report, summary_path = _run_router(AUDIO_VISQOL, "audio_visqol", tmp_path)

    assert report["domain"] == "audio"
    assert report["domain_spec"]["enabled"] is True
    assert report["domain_spec"]["quality_metric"] == "ViSQOL"
    assert report["domain_spec"]["rate_unit"] == "kbps"
    assert report["domain_spec"]["energy_unit"] == "J/s"
    assert report["decision"]["selected"]["codec"]
    assert report["decision"]["selected"]["config"]
    assert report["decision"]["decision_trace"]["quality_guard_applied"] is True
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["selected_codec"]


def test_router_video_produces_report_and_summary(tmp_path: Path) -> None:
    report, summary_path = _run_router(VIDEO_VMAF, "video_vmaf", tmp_path)

    assert report["domain"] == "video"
    assert report["domain_spec"]["enabled"] is True
    assert report["domain_spec"]["quality_metric"] == "VMAF"
    assert report["domain_spec"]["rate_unit"] == "kbps"
    assert report["domain_spec"]["energy_unit"] == "kJ/sequence"
    assert report["decision"]["selected"]["codec"]
    assert report["decision"]["selected"]["config"]
    assert report["decision"]["decision_trace"]["quality_guard_applied"] is True
    assert summary_path.exists()


def test_router_image_produces_report_and_summary_with_domain_spec(
    tmp_path: Path,
) -> None:
    report, summary_path = _run_router(
        IMAGE_SSIMULACRA2,
        "image_ssimulacra2",
        tmp_path,
    )

    assert report["domain"] == "image"
    assert report["domain_spec"]["enabled"] is True
    assert report["domain_spec"]["quality_metric"] == "SSIMULACRA2"
    assert report["resolved_args"]["config_col"] == "config"
    assert report["decision"]["selected"]["codec"]
    assert report["decision"]["selected"]["config"]
    assert summary_path.exists()


def test_audio_video_report_uses_domain_spec_columns_not_image_defaults(
    tmp_path: Path,
) -> None:
    audio_report, _ = _run_router(AUDIO_VISQOL, "audio_visqol", tmp_path)
    video_report, _ = _run_router(VIDEO_VMAF, "video_vmaf", tmp_path)

    assert audio_report["resolved_args"]["rate_col"] == "bitrate_kbps"
    assert audio_report["resolved_args"]["quality_col"] == "visqol"
    assert audio_report["resolved_args"]["energy_col"] == "energy_j_per_second"
    assert video_report["resolved_args"]["rate_col"] == "bitrate_kbps"
    assert video_report["resolved_args"]["quality_col"] == "vmaf"
    assert video_report["resolved_args"]["energy_col"] == "energy_kj_per_sequence"
    assert audio_report["domain_spec"]["rate_column"] != "bpp"
    assert video_report["domain_spec"]["quality_column"] != "ssimulacra2"
    assert audio_report["domain_spec"]["energy_column"] != "energy_per_image_j"


def test_missing_domain_spec_column_produces_clear_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "audio_missing_energy.csv"
    bad_csv.write_text(
        "dataset,codec,param,item_id,bitrate_kbps,visqol,time_ms\n"
        "speech_demo,OPUS,opus_64,aud001,64,4.25,12.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="DomainSpec CSV column mismatch.*energy"):
        router_main(
            [
                "--csv",
                str(bad_csv),
                "--domain-spec",
                "audio_visqol",
                "--out",
                str(tmp_path / "report.json"),
            ]
        )


def test_explicit_cli_column_override_wins_over_domain_spec_default(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "audio_override.csv"
    csv_path.write_text(
        "dataset,codec,param,item_id,bitrate_kbps,visqol,visqol_override,"
        "energy_j_per_second,time_ms\n"
        "speech_demo,OPUS,opus_64,aud001,64,3.60,4.80,0.090,12.0\n"
        "speech_demo,AAC,aac_96,aud001,96,3.70,4.10,0.060,10.0\n",
        encoding="utf-8",
    )

    report_path = tmp_path / "override_report.json"
    router_main(
        [
            "--csv",
            str(csv_path),
            "--domain-spec",
            "audio_visqol",
            "--quality-col",
            "visqol_override",
            "--quality-floor",
            "4.0",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["resolved_args"]["quality_col"] == "visqol_override"
    assert report["decision"]["selected"]["codec"] in {"OPUS", "AAC"}
    assert all(
        candidate["quality"] >= 4.0
        for candidate in report["decision"]["scored_candidate_pool"]
    )


def test_explicit_config_column_override_wins_over_domain_spec_default(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "image_override.csv"
    csv_path.write_text(
        "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms\n"
        "JPEG,q=85,1.60,81.1,0.10,6.3\n"
        "JXL,d=1.0,1.37,85.2,2.55,134.0\n",
        encoding="utf-8",
    )

    report_path = tmp_path / "image_override_report.json"
    router_main(
        [
            "--csv",
            str(csv_path),
            "--domain-spec",
            "image_ssimulacra2",
            "--config-col",
            "param",
            "--profile",
            "balanced",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["resolved_args"]["config_col"] == "param"
    assert report["decision"]["selected"]["config"]


def test_audio_fad_lower_is_better_validated(capsys) -> None:
    report = domain_spec_main(
        ["--csv", str(AUDIO_FAD), "--builtin", "audio_fad", "--validate-csv"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert report["valid"] is True
    assert payload["normalized_spec"]["quality_direction"] == "lower_is_better"
    assert payload["normalized_spec"]["distortion_transform"] == "lower_is_better"
    assert BUILTIN_DOMAIN_SPECS["audio_fad"].quality_direction == "lower_is_better"


def test_router_help_shows_domain_spec_flag() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.router.rde_router", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--domain-spec" in completed.stdout
