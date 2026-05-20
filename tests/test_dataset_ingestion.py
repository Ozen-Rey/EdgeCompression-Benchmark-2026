import csv
import json
import subprocess
import sys
from pathlib import Path

from src.router.core.dataset_ingestion import (
    load_measurements_csv,
    main as ingestion_main,
)
from src.router.core.domain_spec import (
    BUILTIN_DOMAIN_SPECS,
    validate_rde_dataframe_against_domain_spec,
)
from src.router.rde_router import main as router_main


ROOT = Path(__file__).resolve().parents[1]
IMAGE_MANIFEST = ROOT / "configs" / "datasets" / "example_image_dataset.json"
AUDIO_MANIFEST = ROOT / "configs" / "datasets" / "example_audio_dataset.json"
VIDEO_MANIFEST = ROOT / "configs" / "datasets" / "example_video_dataset.json"
IMAGE_MEASUREMENTS = ROOT / "tests" / "fixtures" / "measurements_image_manifest_example.csv"
AUDIO_MEASUREMENTS = ROOT / "tests" / "fixtures" / "measurements_audio_manifest_example.csv"
VIDEO_MEASUREMENTS = ROOT / "tests" / "fixtures" / "measurements_video_manifest_example.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ingest(
    *,
    manifest: Path,
    measurements: Path,
    domain_spec: str,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> tuple[list[dict[str, str]], dict, Path]:
    out_csv = tmp_path / f"{domain_spec}_rde.csv"
    report_out = tmp_path / f"{domain_spec}_report.json"
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
        "--out-csv",
        str(out_csv),
        "--report-out",
        str(report_out),
    ]
    if extra_args:
        args.extend(extra_args)

    report = ingestion_main(args)
    return _read_csv(out_csv), json.loads(report_out.read_text(encoding="utf-8")), out_csv


def test_image_manifest_measurements_to_valid_rde_csv(tmp_path: Path) -> None:
    rows, report, _ = _ingest(
        manifest=IMAGE_MANIFEST,
        measurements=IMAGE_MEASUREMENTS,
        domain_spec="image_ssimulacra2",
        tmp_path=tmp_path,
    )

    assert report["valid"] is True
    assert report["num_valid_rde_rows"] == 3
    assert rows[0]["dataset"] == "example_images"
    assert rows[0]["image_id"] == "img001"
    assert rows[0]["width"] == "768"
    assert rows[0]["height"] == "512"


def test_audio_manifest_measurements_to_valid_rde_csv(tmp_path: Path) -> None:
    rows, report, _ = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=AUDIO_MEASUREMENTS,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )

    assert report["valid"] is True
    assert report["num_valid_rde_rows"] == 3
    assert rows[0]["dataset"] == "example_audio"
    assert rows[0]["item_id"] == "aud001"
    assert rows[0]["duration_s"] == "3.5"
    assert rows[0]["sample_rate"] == "48000"


def test_video_manifest_measurements_to_valid_rde_csv(tmp_path: Path) -> None:
    rows, report, _ = _ingest(
        manifest=VIDEO_MANIFEST,
        measurements=VIDEO_MEASUREMENTS,
        domain_spec="video_vmaf",
        tmp_path=tmp_path,
    )

    assert report["valid"] is True
    assert report["num_valid_rde_rows"] == 3
    assert rows[0]["dataset"] == "example_video"
    assert rows[0]["sequence"] == "vid001"
    assert rows[0]["fps"] == "30.0"
    assert rows[0]["num_frames"] == "120"


def test_output_validates_with_domain_spec(tmp_path: Path) -> None:
    rows, _, _ = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=AUDIO_MEASUREMENTS,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )

    validation = validate_rde_dataframe_against_domain_spec(
        rows,
        BUILTIN_DOMAIN_SPECS["audio_visqol"],
    )

    assert validation["valid"] is True


def test_ingested_output_is_usable_by_router(tmp_path: Path) -> None:
    _, _, out_csv = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=AUDIO_MEASUREMENTS,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )
    router_report = tmp_path / "audio_router_report.json"

    router_main(
        [
            "--csv",
            str(out_csv),
            "--domain-spec",
            "audio_visqol",
            "--out",
            str(router_report),
        ]
    )
    report = json.loads(router_report.read_text(encoding="utf-8"))

    assert report["domain"] == "audio"
    assert report["decision"]["selected"]["codec"]
    assert report["decision"]["selected"]["config"]


def test_unknown_measurement_item_errors_in_strict_mode(tmp_path: Path) -> None:
    measurements = tmp_path / "unknown_item.csv"
    measurements.write_text(
        "item_id,codec,param,bitrate_kbps,visqol,energy_j_per_second\n"
        "aud999,OPUS,opus_64,64,4.25,0.090\n",
        encoding="utf-8",
    )

    _, report, _ = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=measurements,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )

    assert report["valid"] is False
    assert "unknown_measurement_items:aud999" in report["errors"]


def test_unknown_measurement_item_warns_and_drops_in_non_strict_mode(
    tmp_path: Path,
) -> None:
    measurements = tmp_path / "mixed_unknown_item.csv"
    measurements.write_text(
        "item_id,codec,param,bitrate_kbps,visqol,energy_j_per_second\n"
        "aud001,OPUS,opus_64,64,4.25,0.090\n"
        "aud999,AAC,aac_96,96,4.55,0.060\n",
        encoding="utf-8",
    )

    rows, report, _ = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=measurements,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
        extra_args=["--strict", "false"],
    )

    assert report["valid"] is True
    assert report["unknown_measurement_items"] == ["aud999"]
    assert "unknown_measurement_items:aud999:dropped" in report["warnings"]
    assert len(rows) == 1


def test_manifest_item_without_measurements_produces_warning(tmp_path: Path) -> None:
    measurements = tmp_path / "empty_measurements.csv"
    measurements.write_text(
        "item_id,codec,param,bitrate_kbps,visqol,energy_j_per_second\n",
        encoding="utf-8",
    )

    _, report, _ = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=measurements,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
        extra_args=["--strict", "false"],
    )

    assert report["valid"] is True
    assert report["missing_manifest_items"] == ["aud001"]
    assert "manifest_items_without_measurements:aud001" in report["warnings"]


def test_missing_numeric_quality_produces_invalid_row(tmp_path: Path) -> None:
    measurements = tmp_path / "bad_quality.csv"
    measurements.write_text(
        "item_id,codec,param,bitrate_kbps,visqol,energy_j_per_second\n"
        "aud001,OPUS,opus_64,64,,0.090\n",
        encoding="utf-8",
    )

    _, report, _ = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=measurements,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )

    assert report["valid"] is False
    assert report["num_invalid_rde_rows"] == 1
    assert report["numeric_validity_summary"]["quality"]["missing"] == 1


def test_cli_help_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.router.core.dataset_ingestion", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--measurements-csv" in completed.stdout


def test_cli_end_to_end_writes_out_csv_and_report(tmp_path: Path) -> None:
    rows, report, out_csv = _ingest(
        manifest=IMAGE_MANIFEST,
        measurements=IMAGE_MEASUREMENTS,
        domain_spec="image_ssimulacra2",
        tmp_path=tmp_path,
    )

    assert out_csv.exists()
    assert report["output_csv"] == str(out_csv)
    assert rows[0]["metadata_source"] == "example"


def test_metadata_is_propagated_for_image_audio_and_video(tmp_path: Path) -> None:
    image_rows, _, _ = _ingest(
        manifest=IMAGE_MANIFEST,
        measurements=IMAGE_MEASUREMENTS,
        domain_spec="image_ssimulacra2",
        tmp_path=tmp_path,
    )
    audio_rows, _, _ = _ingest(
        manifest=AUDIO_MANIFEST,
        measurements=AUDIO_MEASUREMENTS,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )
    video_rows, _, _ = _ingest(
        manifest=VIDEO_MANIFEST,
        measurements=VIDEO_MEASUREMENTS,
        domain_spec="video_vmaf",
        tmp_path=tmp_path,
    )

    assert image_rows[0]["pixels"] == str(768 * 512)
    assert audio_rows[0]["channels"] == "2"
    assert video_rows[0]["fps"] == "30.0"


def test_dataset_can_be_added_with_manifest_and_csv_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "new_audio_manifest.json"
    measurements_path = tmp_path / "new_audio_measurements.csv"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "dataset_manifest_v1",
                "dataset_id": "new_audio",
                "display_name": "New Audio",
                "domain": "audio",
                "root": "datasets/new_audio",
                "items": [
                    {
                        "item_id": "clip001",
                        "path": "clip001.wav",
                        "duration_s": 2.0,
                        "sample_rate": 48000,
                        "channels": 1,
                    }
                ],
                "splits": {"all": ["clip001"]},
                "metadata": {"created_in_test": True},
                "license": None,
                "source_url": None,
                "notes": None,
            }
        ),
        encoding="utf-8",
    )
    measurements_path.write_text(
        "item_id,codec,param,bitrate_kbps,visqol,energy_j_per_second\n"
        "clip001,OPUS,opus_48,48,4.1,0.04\n",
        encoding="utf-8",
    )

    rows, report, _ = _ingest(
        manifest=manifest_path,
        measurements=measurements_path,
        domain_spec="audio_visqol",
        tmp_path=tmp_path,
    )

    assert report["valid"] is True
    assert rows[0]["dataset"] == "new_audio"
    assert rows[0]["item_id"] == "clip001"


def test_ingested_video_output_is_usable_by_router(tmp_path: Path) -> None:
    _, _, out_csv = _ingest(
        manifest=VIDEO_MANIFEST,
        measurements=VIDEO_MEASUREMENTS,
        domain_spec="video_vmaf",
        tmp_path=tmp_path,
    )
    router_report = tmp_path / "video_router_report.json"

    router_main(
        [
            "--csv",
            str(out_csv),
            "--domain-spec",
            "video_vmaf",
            "--out",
            str(router_report),
        ]
    )
    report = json.loads(router_report.read_text(encoding="utf-8"))

    assert report["domain"] == "video"
    assert report["decision"]["selected"]["codec"]
    assert report["decision"]["selected"]["config"]


def test_load_measurements_csv_reads_rows() -> None:
    rows = load_measurements_csv(AUDIO_MEASUREMENTS)

    assert rows[0]["item_id"] == "aud001"
    assert len(rows) == 3
