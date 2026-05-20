import csv
import json
import subprocess
import sys
from pathlib import Path

from src.router.core.codec_onboarding import main as codec_onboarding_main
from src.router.core.dataset_ingestion import main as ingestion_main
from src.router.core.dataset_onboarding import main as dataset_onboarding_main
from src.router.rde_router import main as router_main


ROOT = Path(__file__).resolve().parents[1]
IMAGE_MANIFEST = ROOT / "configs" / "datasets" / "example_image_dataset.json"
AUDIO_MANIFEST = ROOT / "configs" / "datasets" / "example_audio_dataset.json"
VIDEO_MANIFEST = ROOT / "configs" / "datasets" / "example_video_dataset.json"
NEW_IMAGE = ROOT / "tests" / "fixtures" / "measurements_new_codec_image.csv"
NEW_AUDIO = ROOT / "tests" / "fixtures" / "measurements_new_codec_audio.csv"
NEW_VIDEO = ROOT / "tests" / "fixtures" / "measurements_new_codec_video.csv"
IMAGE_CODEC_SPEC = ROOT / "tests" / "fixtures" / "external_codec_image_spec.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _codec_report(
    tmp_path: Path,
    *,
    measurements: Path | None,
    domain_spec: str,
    codec_spec: Path | None = None,
    config_col: str = "param",
    rate_col: str = "bitrate_kbps",
    quality_col: str = "visqol",
    energy_col: str = "energy_j_per_second",
) -> dict:
    report_out = tmp_path / f"{domain_spec}_codec_report.json"
    args = ["--domain-spec", domain_spec, "--report-out", str(report_out)]
    if measurements is not None:
        args.extend(
            [
                "--measurements-csv",
                str(measurements),
                "--codec-col",
                "codec",
                "--config-col",
                config_col,
                "--rate-col",
                rate_col,
                "--quality-col",
                quality_col,
                "--energy-col",
                energy_col,
            ]
        )
    if codec_spec is not None:
        args.extend(["--codec-spec", str(codec_spec), "--validate-domain"])
    report = codec_onboarding_main(args)
    persisted = json.loads(report_out.read_text(encoding="utf-8"))
    assert persisted == report
    return report


def _ingest_new_codec(
    tmp_path: Path,
    *,
    manifest: Path,
    measurements: Path,
    domain_spec: str,
    item_id_col: str,
    config_col: str,
    rate_col: str,
    quality_col: str,
    energy_col: str,
) -> Path:
    out_csv = tmp_path / f"{domain_spec}_rde.csv"
    report_out = tmp_path / f"{domain_spec}_ingestion_report.json"
    ingestion_main(
        [
            "--manifest",
            str(manifest),
            "--measurements-csv",
            str(measurements),
            "--domain-spec",
            domain_spec,
            "--item-id-col",
            item_id_col,
            "--codec-col",
            "codec",
            "--config-col",
            config_col,
            "--rate-col",
            rate_col,
            "--quality-col",
            quality_col,
            "--energy-col",
            energy_col,
            "--time-col",
            "time_ms",
            "--out-csv",
            str(out_csv),
            "--report-out",
            str(report_out),
        ]
    )
    return out_csv


def test_cli_help_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.router.core.codec_onboarding", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--measurements-csv" in completed.stdout


def test_measured_only_image_valid(tmp_path: Path) -> None:
    report = _codec_report(
        tmp_path,
        measurements=NEW_IMAGE,
        domain_spec="image_ssimulacra2",
        config_col="config",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
    )

    assert report["valid"] is True
    assert "example_neural_image_codec" in report["codec_ids_detected"]


def test_measured_only_audio_valid(tmp_path: Path) -> None:
    report = _codec_report(tmp_path, measurements=NEW_AUDIO, domain_spec="audio_visqol")

    assert report["valid"] is True
    assert "example_audio_codec" in report["codec_ids_detected"]


def test_measured_only_video_valid(tmp_path: Path) -> None:
    report = _codec_report(
        tmp_path,
        measurements=NEW_VIDEO,
        domain_spec="video_vmaf",
        quality_col="vmaf",
        energy_col="energy_kj_per_sequence",
    )

    assert report["valid"] is True
    assert "example_video_codec" in report["codec_ids_detected"]


def test_missing_required_column_produces_clear_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "missing_energy.csv"
    bad_csv.write_text(
        "item_id,codec,param,bitrate_kbps,visqol\n"
        "aud001,example_audio_codec,mode_a,56,4.35\n",
        encoding="utf-8",
    )

    report = _codec_report(tmp_path, measurements=bad_csv, domain_spec="audio_visqol")

    assert report["valid"] is False
    assert "missing_required_columns:energy_j_per_second" in report["errors"]


def test_non_numeric_energy_produces_numeric_validity_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_energy.csv"
    bad_csv.write_text(
        "item_id,codec,param,bitrate_kbps,visqol,energy_j_per_second\n"
        "aud001,example_audio_codec,mode_a,56,4.35,not-a-number\n",
        encoding="utf-8",
    )

    report = _codec_report(tmp_path, measurements=bad_csv, domain_spec="audio_visqol")

    assert report["valid"] is False
    assert "non_numeric_energy:energy_j_per_second:invalid=1:missing=0" in report["errors"]
    assert report["numeric_validity_summary"]["energy"]["invalid"] == 1


def test_codec_spec_domain_compatible_passes(tmp_path: Path) -> None:
    report = _codec_report(
        tmp_path,
        measurements=None,
        domain_spec="image_ssimulacra2",
        codec_spec=IMAGE_CODEC_SPEC,
    )

    assert report["valid"] is True
    assert report["domain_compatible"] is True


def test_codec_spec_domain_mismatch_fails(tmp_path: Path) -> None:
    report = _codec_report(
        tmp_path,
        measurements=None,
        domain_spec="audio_visqol",
        codec_spec=IMAGE_CODEC_SPEC,
    )

    assert report["valid"] is False
    assert "domain_mismatch:codec_spec=image:domain_spec=audio" in report["errors"]


def test_missing_codec_id_or_domain_in_spec_errors(tmp_path: Path) -> None:
    bad_spec = tmp_path / "bad_spec.json"
    data = json.loads(IMAGE_CODEC_SPEC.read_text(encoding="utf-8"))
    data.pop("codec_id")
    data.pop("domain")
    bad_spec.write_text(json.dumps(data), encoding="utf-8")

    report = _codec_report(
        tmp_path,
        measurements=None,
        domain_spec="image_ssimulacra2",
        codec_spec=bad_spec,
    )

    assert report["valid"] is False
    assert "missing_codec_id" in report["errors"]
    assert "missing_codec_domain" in report["errors"]


def test_new_codec_appears_in_ingested_rde_csv(tmp_path: Path) -> None:
    out_csv = _ingest_new_codec(
        tmp_path,
        manifest=AUDIO_MANIFEST,
        measurements=NEW_AUDIO,
        domain_spec="audio_visqol",
        item_id_col="item_id",
        config_col="param",
        rate_col="bitrate_kbps",
        quality_col="visqol",
        energy_col="energy_j_per_second",
    )

    codecs = {row["codec"] for row in _read_csv(out_csv)}
    assert "example_audio_codec" in codecs


def test_router_can_consume_ingested_csv_containing_new_codec(tmp_path: Path) -> None:
    out_csv = _ingest_new_codec(
        tmp_path,
        manifest=VIDEO_MANIFEST,
        measurements=NEW_VIDEO,
        domain_spec="video_vmaf",
        item_id_col="sequence",
        config_col="param",
        rate_col="bitrate_kbps",
        quality_col="vmaf",
        energy_col="energy_kj_per_sequence",
    )
    router_report = tmp_path / "router_report.json"

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
    codecs = {row["codec"] for row in report["decision"]["scored_candidate_pool"]}

    assert "example_video_codec" in codecs
    assert report["decision"]["selected"]["codec"]


def test_dataset_onboarding_report_includes_codec_block(tmp_path: Path) -> None:
    report = dataset_onboarding_main(
        [
            "--manifest",
            str(IMAGE_MANIFEST),
            "--measurements-csv",
            str(NEW_IMAGE),
            "--domain-spec",
            "image_ssimulacra2",
            "--work-dir",
            str(tmp_path),
            "--item-id-col",
            "image_id",
            "--codec-col",
            "codec",
            "--config-col",
            "config",
            "--rate-col",
            "bpp",
            "--quality-col",
            "ssimulacra2",
            "--energy-col",
            "energy_per_image_j",
            "--time-col",
            "time_ms",
            "--codec-spec",
            str(IMAGE_CODEC_SPEC),
        ]
    )

    assert report["valid"] is True
    assert report["codec_onboarding"]["enabled"] is True
    assert report["codec_onboarding"]["domain_compatible"] is True
    assert "codec_onboarding" in report["steps"]


def test_new_codec_measurements_template_writes_expected_columns(tmp_path: Path) -> None:
    out = tmp_path / "codec_measurements_template.csv"
    report = codec_onboarding_main(
        [
            "--new-codec-measurements-template",
            "audio_visqol",
            "--out",
            str(out),
        ]
    )

    assert report["valid"] is True
    header = _read_csv(out)[0].keys()
    assert {"dataset", "item_id", "codec", "param", "bitrate_kbps", "visqol"} <= set(header)
