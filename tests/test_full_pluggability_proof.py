import csv
import json
from pathlib import Path

from src.router.core.codec_onboarding import main as codec_onboarding_main
from src.router.core.dataset_onboarding import main as dataset_onboarding_main


ROOT = Path(__file__).resolve().parents[1]


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
        router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
        ingested_rows = _read_csv(ingested_csv)
        candidate_codecs = {
            row["codec"]
            for row in router_report["decision"]["scored_candidate_pool"]
        }

        assert codec_report["valid"] is True
        assert onboarding_report["valid"] is True
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

    proof_report = {
        "valid": all(
            case["manifest_valid"]
            and case["codec_measurements_valid"]
            and case["ingestion_valid"]
            and case["domain_spec_validation_valid"]
            and case["router_decision_valid"]
            and case["selected_codec"]
            and case["selected_config"]
            and case["new_codec_in_candidate_pool"]
            for case in proof_cases
        ),
        "claim": (
            "new dataset + new codec/model measurements + DomainSpec -> "
            "R-D-E CSV -> router decision without router code changes"
        ),
        "cases": proof_cases,
    }
    proof_report_path = tmp_path / "full_pluggability_proof_report.json"
    proof_report_path.write_text(
        json.dumps(proof_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    persisted = json.loads(proof_report_path.read_text(encoding="utf-8"))
    assert persisted["valid"] is True
    assert {case["domain"] for case in persisted["cases"]} == {
        "image",
        "audio",
        "video",
    }
