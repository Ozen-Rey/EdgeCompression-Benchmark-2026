import csv
import json
import subprocess
import sys
from pathlib import Path

from src.router.core.dataset_manifest import (
    load_dataset_manifest,
    main,
    manifest_to_item_table,
    validate_dataset_manifest,
    validate_manifest_against_domain_spec,
)
from src.router.core.domain_spec import BUILTIN_DOMAIN_SPECS


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGE = ROOT / "configs" / "datasets" / "example_image_dataset.json"
EXAMPLE_AUDIO = ROOT / "configs" / "datasets" / "example_audio_dataset.json"
EXAMPLE_VIDEO = ROOT / "configs" / "datasets" / "example_video_dataset.json"


def _image_manifest(**overrides):
    manifest = {
        "schema_version": "dataset_manifest_v1",
        "dataset_id": "example_images",
        "display_name": "Example Image Dataset",
        "domain": "image",
        "root": "datasets/example_images",
        "items": [
            {
                "item_id": "img001",
                "path": "img001.png",
                "width": 768,
                "height": 512,
                "metadata": {"source": "example"},
            }
        ],
        "splits": {"all": ["img001"], "test": ["img001"]},
        "metadata": {"source": "example"},
        "license": None,
        "source_url": None,
        "notes": None,
    }
    manifest.update(overrides)
    return manifest


def test_valid_image_manifest() -> None:
    report = validate_dataset_manifest(load_dataset_manifest(EXAMPLE_IMAGE))

    assert report["valid"] is True
    assert report["domain"] == "image"
    assert report["normalized_manifest"]["items"][0]["pixels"] == 768 * 512


def test_valid_audio_manifest() -> None:
    report = validate_dataset_manifest(load_dataset_manifest(EXAMPLE_AUDIO))

    assert report["valid"] is True
    assert report["domain"] == "audio"


def test_valid_video_manifest() -> None:
    report = validate_dataset_manifest(load_dataset_manifest(EXAMPLE_VIDEO))

    assert report["valid"] is True
    assert report["domain"] == "video"


def test_duplicate_item_id_produces_error() -> None:
    manifest = _image_manifest(
        items=[
            {"item_id": "img001", "path": "a.png", "width": 10, "height": 10},
            {"item_id": "img001", "path": "b.png", "width": 10, "height": 10},
        ]
    )

    report = validate_dataset_manifest(manifest)

    assert report["valid"] is False
    assert "duplicate_item_id:img001" in report["errors"]


def test_split_with_missing_item_produces_error() -> None:
    manifest = _image_manifest(splits={"test": ["missing"]})

    report = validate_dataset_manifest(manifest)

    assert report["valid"] is False
    assert "split_item_id_not_found:test:missing" in report["errors"]


def test_missing_root_and_path_with_check_files_false_does_not_fail() -> None:
    report = validate_dataset_manifest(_image_manifest(root="does/not/exist"))

    assert report["valid"] is True
    assert not any(error.startswith("missing_root_path") for error in report["errors"])
    assert report["missing_files"] == 0


def test_missing_file_with_check_files_true_produces_error(tmp_path: Path) -> None:
    manifest = _image_manifest(root=str(tmp_path))

    report = validate_dataset_manifest(manifest, check_files=True)

    assert report["valid"] is False
    assert report["missing_files"] == 1
    assert "missing_item_file:img001:img001.png" in report["errors"]


def test_manifest_to_item_table_produces_expected_rows() -> None:
    rows = manifest_to_item_table(_image_manifest())

    assert rows == [
        {
            "dataset_id": "example_images",
            "domain": "image",
            "root": "datasets/example_images",
            "item_id": "img001",
            "path": "img001.png",
            "width": 768,
            "height": 512,
            "pixels": 768 * 512,
            "duration_s": None,
            "sample_rate": None,
            "channels": None,
            "fps": None,
            "num_frames": None,
            "metadata_json": json.dumps({"source": "example"}, sort_keys=True),
            "metadata_source": "example",
        }
    ]


def test_validate_manifest_against_domain_spec_accepts_matching_domain() -> None:
    report = validate_manifest_against_domain_spec(
        _image_manifest(),
        BUILTIN_DOMAIN_SPECS["image_ssimulacra2"],
    )

    assert report["valid"] is True


def test_validate_manifest_against_domain_spec_rejects_domain_mismatch() -> None:
    report = validate_manifest_against_domain_spec(
        _image_manifest(domain="audio"),
        BUILTIN_DOMAIN_SPECS["image_ssimulacra2"],
    )

    assert report["valid"] is False
    assert "domain_mismatch:manifest=audio:domain_spec=image" in report["errors"]


def test_cli_help_exit_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.router.core.dataset_manifest", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--manifest" in completed.stdout


def test_cli_validate_example_manifests(capsys) -> None:
    for path in (EXAMPLE_IMAGE, EXAMPLE_AUDIO, EXAMPLE_VIDEO):
        report = main(["--manifest", str(path), "--validate"])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert report["valid"] is True
        assert payload["valid"] is True
        assert payload["num_items"] == 1


def test_cli_to_csv_writes_item_table(tmp_path: Path, capsys) -> None:
    out_csv = tmp_path / "items.csv"

    report = main(["--manifest", str(EXAMPLE_IMAGE), "--to-csv", str(out_csv)])
    captured = capsys.readouterr()

    assert report["valid"] is True
    assert json.loads(captured.out)["item_table_csv"] == str(out_csv)
    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["item_id"] == "img001"
    assert rows[0]["metadata_source"] == "example"


def test_missing_recommended_metadata_produces_warnings() -> None:
    manifest = _image_manifest(
        items=[{"item_id": "img001", "path": "img001.png", "metadata": {}}],
        metadata={},
    )

    report = validate_dataset_manifest(manifest)

    assert report["valid"] is True
    assert "item_recommended_image_dimensions_missing:img001" in report["warnings"]
    assert "dataset_metadata_empty" in report["warnings"]
