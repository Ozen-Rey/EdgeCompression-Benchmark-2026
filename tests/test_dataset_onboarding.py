import csv
import json
import subprocess
import sys
from pathlib import Path

from src.router.core.dataset_manifest import (
    load_dataset_manifest,
    main as manifest_main,
    validate_dataset_manifest,
)
from src.router.core.dataset_ingestion import main as ingestion_main
from src.router.core.dataset_onboarding import main as onboarding_main


ROOT = Path(__file__).resolve().parents[1]
IMAGE_MANIFEST = ROOT / "configs" / "datasets" / "example_image_dataset.json"
AUDIO_MANIFEST = ROOT / "configs" / "datasets" / "example_audio_dataset.json"
VIDEO_MANIFEST = ROOT / "configs" / "datasets" / "example_video_dataset.json"
IMAGE_MEASUREMENTS = ROOT / "tests" / "fixtures" / "measurements_image_manifest_example.csv"
AUDIO_MEASUREMENTS = ROOT / "tests" / "fixtures" / "measurements_audio_manifest_example.csv"
VIDEO_MEASUREMENTS = ROOT / "tests" / "fixtures" / "measurements_video_manifest_example.csv"


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _onboard(
    *,
    manifest: Path,
    measurements: Path,
    domain_spec: str,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> dict:
    quality_col = (
        "visqol"
        if domain_spec == "audio_visqol"
        else "vmaf"
        if domain_spec == "video_vmaf"
        else "ssimulacra2"
    )
    args = [
        "--manifest",
        str(manifest),
        "--measurements-csv",
        str(measurements),
        "--domain-spec",
        domain_spec,
        "--work-dir",
        str(tmp_path),
        "--item-id-col",
        "item_id",
        "--codec-col",
        "codec",
        "--config-col",
        "param",
        "--rate-col",
        "bitrate_kbps" if domain_spec.startswith(("audio", "video")) else "bpp",
        "--quality-col",
        quality_col,
        "--energy-col",
        "energy_j_per_second"
        if domain_spec.startswith("audio")
        else "energy_kj_per_sequence"
        if domain_spec.startswith("video")
        else "energy_per_image_j",
        "--time-col",
        "time_ms",
    ]
    if extra_args:
        args.extend(extra_args)
    return onboarding_main(args)


def test_new_template_image_audio_video_produce_valid_json(tmp_path: Path) -> None:
    for domain in ("image", "audio", "video"):
        out = tmp_path / f"{domain}.json"
        report = manifest_main(["--new-template", domain, "--out", str(out)])
        manifest = load_dataset_manifest(out)
        validation = validate_dataset_manifest(manifest)

        assert report["valid"] is True
        assert validation["valid"] is True
        assert manifest.domain == domain


def test_new_measurements_template_has_expected_columns(tmp_path: Path) -> None:
    cases = {
        "image_ssimulacra2": ["dataset", "image_id", "codec", "config", "bpp"],
        "audio_visqol": ["dataset", "item_id", "codec", "param", "bitrate_kbps"],
        "video_vmaf": ["dataset", "sequence", "codec", "param", "bitrate_kbps"],
    }
    for spec, expected_columns in cases.items():
        out = tmp_path / f"{spec}.csv"
        report = ingestion_main(["--new-measurements-template", spec, "--out", str(out)])

        assert report["valid"] is True
        header = _read_header(out)
        for column in expected_columns:
            assert column in header


def test_onboarding_end_to_end_image_fixture_passes(tmp_path: Path) -> None:
    report = _onboard(
        manifest=IMAGE_MANIFEST,
        measurements=IMAGE_MEASUREMENTS,
        domain_spec="image_ssimulacra2",
        tmp_path=tmp_path,
    )

    assert report["valid"] is True
    assert report["selected_codec"]
    assert report["selected_config"]


def test_onboarding_end_to_end_audio_fixture_passes(tmp_path: Path) -> None:
    report = _onboard(
        manifest=AUDIO_MANIFEST,
        measurements=AUDIO_MEASUREMENTS,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )

    assert report["valid"] is True
    assert report["selected_codec"]
    assert report["selected_config"]


def test_onboarding_end_to_end_video_fixture_passes(tmp_path: Path) -> None:
    report = _onboard(
        manifest=VIDEO_MANIFEST,
        measurements=VIDEO_MEASUREMENTS,
        domain_spec="video_vmaf",
        tmp_path=tmp_path,
    )

    assert report["valid"] is True
    assert report["selected_codec"]
    assert report["selected_config"]


def test_missing_measurement_column_produces_clear_error(tmp_path: Path) -> None:
    bad_measurements = tmp_path / "missing_quality.csv"
    bad_measurements.write_text(
        "item_id,codec,param,bitrate_kbps,energy_j_per_second,time_ms\n"
        "aud001,OPUS,opus_64,64,0.090,12.0\n",
        encoding="utf-8",
    )

    report = _onboard(
        manifest=AUDIO_MANIFEST,
        measurements=bad_measurements,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )

    assert report["valid"] is False
    assert "missing_measurement_columns:visqol" in report["errors"]


def test_router_report_is_produced(tmp_path: Path) -> None:
    report = _onboard(
        manifest=AUDIO_MANIFEST,
        measurements=AUDIO_MEASUREMENTS,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )
    router_report = Path(report["outputs"]["router_report"])

    assert router_report.exists()
    payload = json.loads(router_report.read_text(encoding="utf-8"))
    assert payload["decision"]["selected"]["codec"]


def test_selected_codec_and_config_are_non_empty(tmp_path: Path) -> None:
    report = _onboard(
        manifest=VIDEO_MANIFEST,
        measurements=VIDEO_MEASUREMENTS,
        domain_spec="video_vmaf",
        tmp_path=tmp_path,
    )

    assert str(report["selected_codec"]).strip()
    assert str(report["selected_config"]).strip()


def test_onboarding_report_contains_all_steps(tmp_path: Path) -> None:
    report = _onboard(
        manifest=IMAGE_MANIFEST,
        measurements=IMAGE_MEASUREMENTS,
        domain_spec="image_ssimulacra2",
        tmp_path=tmp_path,
    )
    persisted = json.loads(
        Path(report["outputs"]["onboarding_report"]).read_text(encoding="utf-8")
    )

    assert set(persisted["steps"]) == {
        "manifest",
        "measurements",
        "ingestion",
        "domain_spec",
        "router_decision",
        "codec_onboarding",
    }
    assert persisted["manifest_valid"] is True
    assert persisted["ingestion_valid"] is True
    assert persisted["domain_spec_valid"] is True
    assert persisted["router_decision_valid"] is True


def test_cli_help_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.router.core.dataset_onboarding", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--measurements-csv" in completed.stdout
