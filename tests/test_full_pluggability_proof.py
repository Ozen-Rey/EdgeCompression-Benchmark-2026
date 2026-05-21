import csv
import json
from pathlib import Path

from src.router.core.codec_onboarding import main as codec_onboarding_main
from src.router.core.contracts import ONBOARDING_CONTRACT_ID
from src.router.core.dataset_onboarding import main as dataset_onboarding_main


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REPORT = (
    ROOT / "docs" / "examples" / "full_pluggability_proof_report.example.json"
)
IMAGE_MANIFEST = ROOT / "configs" / "datasets" / "example_image_dataset.json"
IMAGE_REAL_CODEC_MEASUREMENTS = (
    ROOT / "tests" / "fixtures" / "measurements_image_manifest_example.csv"
)


CASES = [
    {
        "domain": "image",
        "manifest": ROOT / "configs" / "datasets" / "example_image_dataset.json",
        "measurements": ROOT / "tests" / "fixtures" / "measurements_new_codec_image.csv",
        "domain_spec": "image_ssimulacra2",
        "new_codec": "example_neural_image_codec",
        "item_id_col": "image_id",
        "config_col": "config",
        "rate_col": "bpp",
        "quality_col": "ssimulacra2",
        "energy_col": "energy_per_image_j",
    },
    {
        "domain": "audio",
        "manifest": ROOT / "configs" / "datasets" / "example_audio_dataset.json",
        "measurements": ROOT / "tests" / "fixtures" / "measurements_new_codec_audio.csv",
        "domain_spec": "audio_visqol",
        "new_codec": "example_audio_codec",
        "item_id_col": "item_id",
        "config_col": "param",
        "rate_col": "bitrate_kbps",
        "quality_col": "visqol",
        "energy_col": "energy_j_per_second",
    },
    {
        "domain": "video",
        "manifest": ROOT / "configs" / "datasets" / "example_video_dataset.json",
        "measurements": ROOT / "tests" / "fixtures" / "measurements_new_codec_video.csv",
        "domain_spec": "video_vmaf",
        "new_codec": "example_video_codec",
        "item_id_col": "sequence",
        "config_col": "param",
        "rate_col": "bitrate_kbps",
        "quality_col": "vmaf",
        "energy_col": "energy_kj_per_sequence",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _candidate_codecs(router_report_path: Path) -> set[str]:
    router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
    return {
        row["codec"]
        for row in router_report["decision"]["scored_candidate_pool"]
    }


def test_full_dataset_codec_pluggability_proof_image_audio_video(
    tmp_path: Path,
) -> None:
    proof_cases = []

    for case in CASES:
        domain_work_dir = tmp_path / case["domain"]
        codec_report_out = domain_work_dir / "codec_measurements_report.json"

        codec_report = codec_onboarding_main(
            [
                "--measurements-csv",
                str(case["measurements"]),
                "--domain-spec",
                str(case["domain_spec"]),
                "--codec-col",
                "codec",
                "--config-col",
                str(case["config_col"]),
                "--rate-col",
                str(case["rate_col"]),
                "--quality-col",
                str(case["quality_col"]),
                "--energy-col",
                str(case["energy_col"]),
                "--report-out",
                str(codec_report_out),
            ]
        )

        onboarding_report = dataset_onboarding_main(
            [
                "--manifest",
                str(case["manifest"]),
                "--measurements-csv",
                str(case["measurements"]),
                "--domain-spec",
                str(case["domain_spec"]),
                "--work-dir",
                str(domain_work_dir),
                "--item-id-col",
                str(case["item_id_col"]),
                "--codec-col",
                "codec",
                "--config-col",
                str(case["config_col"]),
                "--rate-col",
                str(case["rate_col"]),
                "--quality-col",
                str(case["quality_col"]),
                "--energy-col",
                str(case["energy_col"]),
                "--time-col",
                "time_ms",
                "--router-profile",
                "balanced",
            ]
        )

        onboarding_report_path = Path(onboarding_report["outputs"]["onboarding_report"])
        ingested_csv = Path(onboarding_report["outputs"]["ingested_rde_csv"])
        router_report_path = Path(onboarding_report["outputs"]["router_report"])
        router_summary = Path(onboarding_report["outputs"]["router_summary"])
        ingested_rows = _read_csv(ingested_csv)
        candidate_codecs = _candidate_codecs(router_report_path)

        assert codec_report["valid"] is True
        assert codec_report["contract_id"] == ONBOARDING_CONTRACT_ID
        assert onboarding_report["valid"] is True
        assert onboarding_report["contract_id"] == ONBOARDING_CONTRACT_ID
        assert onboarding_report["manifest_valid"] is True
        assert onboarding_report["measurements_valid"] is True
        assert onboarding_report["ingestion_valid"] is True
        assert onboarding_report["domain_spec_valid"] is True
        assert onboarding_report["router_decision_valid"] is True
        assert onboarding_report["selected_codec"]
        assert onboarding_report["selected_config"]
        assert case["new_codec"] in {row["codec"] for row in ingested_rows}
        assert case["new_codec"] in candidate_codecs
        assert codec_report_out.exists()
        assert onboarding_report_path.exists()
        assert router_report_path.exists()
        assert router_summary.exists()

        proof_cases.append(
            {
                "domain": case["domain"],
                "domain_spec": case["domain_spec"],
                "new_codec": case["new_codec"],
                "manifest_valid": onboarding_report["manifest_valid"],
                "codec_measurements_valid": codec_report["valid"],
                "ingestion_valid": onboarding_report["ingestion_valid"],
                "domain_spec_validation_valid": onboarding_report[
                    "domain_spec_valid"
                ],
                "router_decision_valid": onboarding_report[
                    "router_decision_valid"
                ],
                "selected_codec": onboarding_report["selected_codec"],
                "selected_config": onboarding_report["selected_config"],
                "new_codec_in_candidate_pool": case["new_codec"] in candidate_codecs,
                "outputs": onboarding_report["outputs"]
                | {"codec_measurements_report": str(codec_report_out)},
            }
        )

    valid = all(
        case["manifest_valid"]
        and case["codec_measurements_valid"]
        and case["ingestion_valid"]
        and case["domain_spec_validation_valid"]
        and case["router_decision_valid"]
        and case["selected_codec"]
        and case["selected_config"]
        and case["new_codec_in_candidate_pool"]
        for case in proof_cases
    )
    proof_report = {
        "valid": valid,
        "contract_id": ONBOARDING_CONTRACT_ID,
        "proof_mode": "pytest_tmp_generated",
        "pluggability_status": "proven" if valid else "failed",
        "claim": (
            "new dataset + new codec/model measurements + DomainSpec -> "
            "R-D-E CSV -> router decision without router code changes"
        ),
        "manifest_valid": all(case["manifest_valid"] for case in proof_cases),
        "codec_measurements_valid": all(
            case["codec_measurements_valid"] for case in proof_cases
        ),
        "ingestion_valid": all(case["ingestion_valid"] for case in proof_cases),
        "domain_spec_valid": all(
            case["domain_spec_validation_valid"] for case in proof_cases
        ),
        "router_decision_valid": all(
            case["router_decision_valid"] for case in proof_cases
        ),
        "selected_codec": {
            case["domain"]: case["selected_codec"] for case in proof_cases
        },
        "selected_config": {
            case["domain"]: case["selected_config"] for case in proof_cases
        },
        "warnings": [],
        "errors": [],
        "cases": proof_cases,
    }
    proof_report_path = tmp_path / "full_pluggability_proof_report.json"
    proof_report_path.write_text(
        json.dumps(proof_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    persisted = json.loads(proof_report_path.read_text(encoding="utf-8"))
    assert persisted["valid"] is True
    assert persisted["contract_id"] == ONBOARDING_CONTRACT_ID
    assert {case["domain"] for case in persisted["cases"]} == {
        "image",
        "audio",
        "video",
    }


def test_full_pluggability_proof_with_real_jpeg_codec(tmp_path: Path) -> None:
    work_dir = tmp_path / "real_jpeg"
    report = dataset_onboarding_main(
        [
            "--manifest",
            str(IMAGE_MANIFEST),
            "--measurements-csv",
            str(IMAGE_REAL_CODEC_MEASUREMENTS),
            "--domain-spec",
            "image_ssimulacra2",
            "--work-dir",
            str(work_dir),
            "--item-id-col",
            "item_id",
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
            "--router-profile",
            "balanced",
        ]
    )
    ingested_csv = Path(report["outputs"]["ingested_rde_csv"])
    router_report_path = Path(report["outputs"]["router_report"])
    router_summary = Path(report["outputs"]["router_summary"])
    ingested_codecs = {row["codec"] for row in _read_csv(ingested_csv)}

    assert report["valid"] is True
    assert report["manifest_valid"] is True
    assert report["ingestion_valid"] is True
    assert report["domain_spec_valid"] is True
    assert report["router_decision_valid"] is True
    assert report["selected_codec"]
    assert report["selected_config"]
    assert "JPEG" in ingested_codecs
    assert "JPEG" in _candidate_codecs(router_report_path)
    assert router_summary.exists()


def test_full_pluggability_sample_report_shape_is_consistent() -> None:
    payload = json.loads(SAMPLE_REPORT.read_text(encoding="utf-8"))

    assert payload["contract_id"] == ONBOARDING_CONTRACT_ID
    assert payload["manifest_valid"] is True
    assert payload["ingestion_valid"] is True
    assert payload["domain_spec_valid"] is True
    assert payload["router_decision_valid"] is True
    assert payload["selected_codec"]
    assert payload["selected_config"]
    assert payload["proof_mode"] == "static_documentation_example"
    assert payload["pluggability_status"] == "proven"
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["errors"], list)
    assert {case["domain"] for case in payload["cases"]} == {
        "image",
        "audio",
        "video",
    }
