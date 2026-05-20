import json
import subprocess
import sys
from pathlib import Path

from src.router.core.domain_spec import (
    BUILTIN_DOMAIN_SPECS,
    main,
    validate_rde_dataframe_against_domain_spec,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _valid_image_rows() -> list[dict[str, object]]:
    return [
        {
            "domain": "image",
            "dataset": "kodak",
            "image_id": "kodim01",
            "codec": "JPEG",
            "config": "q=85",
            "bpp": "1.60",
            "ssimulacra2": "81.1",
            "energy_per_image_j": "0.10",
        }
    ]


def test_builtins_cover_image_audio_video() -> None:
    domains = {spec.domain for spec in BUILTIN_DOMAIN_SPECS.values()}

    assert {"image", "audio", "video"} <= domains
    assert "image_ssimulacra2" in BUILTIN_DOMAIN_SPECS
    assert "video_vmaf" in BUILTIN_DOMAIN_SPECS
    assert "audio_visqol" in BUILTIN_DOMAIN_SPECS


def test_image_ssimulacra2_validates_image_fixture() -> None:
    report = validate_rde_dataframe_against_domain_spec(
        _valid_image_rows(),
        BUILTIN_DOMAIN_SPECS["image_ssimulacra2"],
    )

    assert report["valid"] is True
    assert report["num_rows"] == 1


def test_video_vmaf_validates_video_fixture() -> None:
    rows = [
        {
            "domain": "video",
            "dataset": "ugc",
            "sequence": "seq001",
            "codec": "AV1",
            "param": "crf=30",
            "bitrate_kbps": "850",
            "vmaf": "94.2",
            "energy_kj_per_sequence": "1.7",
        }
    ]

    report = validate_rde_dataframe_against_domain_spec(
        rows,
        BUILTIN_DOMAIN_SPECS["video_vmaf"],
    )

    assert report["valid"] is True


def test_audio_visqol_validates_audio_fixture() -> None:
    rows = [
        {
            "domain": "audio",
            "dataset": "speech",
            "item_id": "clip001",
            "codec": "OPUS",
            "param": "64k",
            "bitrate_kbps": "64",
            "visqol": "4.5",
            "energy_j_per_second": "0.04",
        }
    ]

    report = validate_rde_dataframe_against_domain_spec(
        rows,
        BUILTIN_DOMAIN_SPECS["audio_visqol"],
    )

    assert report["valid"] is True


def test_audio_fad_is_lower_is_better() -> None:
    spec = BUILTIN_DOMAIN_SPECS["audio_fad"]
    rows = [
        {
            "domain": "audio",
            "dataset": "music",
            "item_id": "clip002",
            "codec": "AAC",
            "param": "96k",
            "bitrate_kbps": "96",
            "fad": "1.2",
            "energy_j_per_second": "0.05",
        }
    ]

    report = validate_rde_dataframe_against_domain_spec(rows, spec)

    assert report["valid"] is True
    assert report["normalized_spec"]["quality_direction"] == "lower_is_better"
    assert report["normalized_spec"]["distortion_transform"] == "lower_is_better"


def test_missing_column_produces_clear_error() -> None:
    rows = _valid_image_rows()
    rows[0].pop("energy_per_image_j")

    report = validate_rde_dataframe_against_domain_spec(
        rows,
        BUILTIN_DOMAIN_SPECS["image_ssimulacra2"],
    )

    assert report["valid"] is False
    assert "missing_column:energy_column:energy_per_image_j" in report["errors"]


def test_non_numeric_rate_quality_energy_are_diagnosed() -> None:
    rows = [
        {
            "domain": "image",
            "dataset": "kodak",
            "image_id": "kodim01",
            "codec": "JPEG",
            "config": "q=85",
            "bpp": "fast",
            "ssimulacra2": "good",
            "energy_per_image_j": "low",
        }
    ]

    report = validate_rde_dataframe_against_domain_spec(
        rows,
        BUILTIN_DOMAIN_SPECS["image_ssimulacra2"],
    )

    assert report["valid"] is False
    assert report["numeric_validity_summary"]["rate"]["invalid"] == 1
    assert report["numeric_validity_summary"]["quality"]["invalid"] == 1
    assert report["numeric_validity_summary"]["energy"]["invalid"] == 1


def test_cli_help_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.router.core.domain_spec", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--list-builtins" in completed.stdout


def test_list_builtins_includes_all_domains(capsys) -> None:
    report = main(["--list-builtins"])
    captured = capsys.readouterr()

    assert set(report["domains"]) == {"audio", "image", "video"}
    assert "image_ssimulacra2" in json.loads(captured.out)["builtins"]


def test_print_json_produces_valid_json(capsys) -> None:
    main(["--builtin", "image_ssimulacra2", "--print-json"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert payload["domain"] == "image"
    assert payload["quality_column"] == "ssimulacra2"


def test_validate_csv_reports_rows_and_numeric_summary(tmp_path: Path, capsys) -> None:
    csv_path = _write_csv(
        tmp_path / "image.csv",
        [
            "domain",
            "dataset",
            "image_id",
            "codec",
            "config",
            "bpp",
            "ssimulacra2",
            "energy_per_image_j",
        ],
        [["image", "kodak", "kodim01", "JPEG", "q=85", 1.6, 81.1, 0.10]],
    )

    report = main(
        [
            "--csv",
            str(csv_path),
            "--builtin",
            "image_ssimulacra2",
            "--validate-csv",
        ]
    )
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert report["valid"] is True
    assert payload["num_rows"] == 1
    assert "numeric_validity_summary" in payload
    assert payload["numeric_validity_summary"]["rate"]["all_valid"] is True
