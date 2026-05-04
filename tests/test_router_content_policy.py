import json
from pathlib import Path

from src.router.rde_router import main


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "router_content_policy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_router_reports_content_policy_suggestion():
    csv_path = _tmp_path("rde.csv")
    rules_path = _tmp_path("rules.csv")
    report_path = _tmp_path("report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=85,1.2,85.0,1.0,20.0",
                "HEVC,crf=15,1.0,95.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    rules_path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,tecnick,JPEG,q=85",
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
            "--domain",
            "image",
            "--quality-target",
            "high",
            "--quality-floor",
            "80",
            "--content-policy",
            "--content-policy-mode",
            "report-only",
            "--content-policy-rules-file",
            str(rules_path),
            "--content-policy-key",
            "dataset",
            "--content-source",
            "tecnick",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["content_policy"]["enabled"] is True
    assert report["content_policy"]["mode"] == "report-only"
    assert report["content_policy"]["policy_key"] == "dataset"
    assert report["content_policy"]["policy_value"] == "tecnick"
    assert report["content_policy"]["suggestion"]["codec"] == "JPEG"
    assert report["content_policy"]["suggestion"]["config"] == "q=85"


def test_router_content_source_filter_restricts_candidate_pool():
    csv_path = _tmp_path("source_filter_rde.csv")
    report_path = _tmp_path("source_filter_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "dataset,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,JPEG,q=85,1.2,85.0,1.0,20.0",
                "A,HEVC,crf=15,1.0,95.0,10.0,100.0",
                "B,JPEG,q=85,1.2,60.0,1.0,20.0",
                "B,HEVC,crf=15,1.0,95.0,10.0,100.0",
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
            "--domain",
            "image",
            "--quality-target",
            "high",
            "--quality-floor",
            "80",
            "--content-source",
            "A",
            "--content-source-filter",
            "--content-filter-column",
            "dataset",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["content_filter"]["enabled"] is True
    assert report["content_filter"]["applied"] is True
    assert report["content_filter"]["column"] == "dataset"
    assert report["content_filter"]["value"] == "A"
    assert report["content_filter"]["before_count"] == 4
    assert report["content_filter"]["after_count"] == 2


def test_router_content_policy_apply_selects_feasible_preferred_candidate():
    csv_path = _tmp_path("apply_feasible_rde.csv")
    rules_path = _tmp_path("apply_feasible_rules.csv")
    report_path = _tmp_path("apply_feasible_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "dataset,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,JPEG,q=85,1.2,85.0,1.0,20.0",
                "A,HEVC,crf=15,1.0,95.0,10.0,100.0",
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
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["decision"]["selected"]

    assert selected["codec"] == "JPEG"
    assert selected["config"] == "q=85"
    assert report["content_policy"]["applied"] is True
    assert (
        report["decision"]["decision_trace"]["selected_reason"]
        == "content_policy_preferred_candidate"
    )


def test_router_content_policy_apply_falls_back_when_preferred_not_safe():
    csv_path = _tmp_path("apply_fallback_rde.csv")
    rules_path = _tmp_path("apply_fallback_rules.csv")
    report_path = _tmp_path("apply_fallback_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "dataset,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,JPEG,q=85,1.2,60.0,1.0,20.0",
                "A,HEVC,crf=15,1.0,95.0,10.0,100.0",
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
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["decision"]["selected"]

    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
    assert report["content_policy"]["applied"] is False
    assert (
        "content_policy_suggestion_not_admissible_fallback_to_router"
        in report["content_policy"]["warnings"]
    )


def test_router_reports_content_classifier_prediction():
    csv_path = _tmp_path("classifier_rde.csv")
    training_path = _tmp_path("classifier_training.csv")
    config_path = _tmp_path("classifier.json")
    report_path = _tmp_path("classifier_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=85,1.2,85.0,1.0,20.0",
                "HEVC,crf=15,1.0,95.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    training_path.write_text(
        "\n".join(
            [
                "dataset,image,image_id,oracle_codec,oracle_config,megapixels,aspect_ratio,resolution_class,orientation_class",
                "A,img1,A::img1,JPEG,q=85,1.0,1.0,medium,squareish",
                "B,img2,B::img2,HEVC,crf=15,8.0,1.7,huge,landscape",
            ]
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "model_type": "knn_oracle_classifier",
                "feature_set": "metadata_no_source",
                "k": 1,
                "training_rows": str(training_path),
                "pixel_features": None,
                "quality_floor": 80.0,
                "fallback": "router",
                "selection_reason": "content_classifier_preferred_candidate",
            }
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
            "--domain",
            "image",
            "--quality-target",
            "high",
            "--quality-floor",
            "80",
            "--content-classifier",
            "--content-classifier-mode",
            "report-only",
            "--content-classifier-config",
            str(config_path),
            "--content-classifier-width",
            "1000",
            "--content-classifier-height",
            "1000",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["content_classifier"]["enabled"] is True
    assert report["content_classifier"]["mode"] == "report-only"
    assert report["content_classifier"]["applied"] is False
    assert report["content_classifier"]["prediction"]["codec"] == "JPEG"
    assert report["content_classifier"]["prediction"]["config"] == "q=85"
    assert report["content_classifier"]["features"]["resolution_class"] == "medium"
    assert report["content_classifier"]["features"]["orientation_class"] == "squareish"


def test_router_content_classifier_apply_selects_feasible_prediction():
    csv_path = _tmp_path("classifier_apply_feasible_rde.csv")
    training_path = _tmp_path("classifier_apply_feasible_training.csv")
    config_path = _tmp_path("classifier_apply_feasible.json")
    report_path = _tmp_path("classifier_apply_feasible_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=85,1.2,85.0,1.0,20.0",
                "HEVC,crf=15,1.0,95.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    training_path.write_text(
        "\n".join(
            [
                "dataset,image,image_id,oracle_codec,oracle_config,megapixels,aspect_ratio,resolution_class,orientation_class",
                "A,img1,A::img1,JPEG,q=85,1.0,1.0,medium,squareish",
                "B,img2,B::img2,HEVC,crf=15,8.0,1.7,huge,landscape",
            ]
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "model_type": "knn_oracle_classifier",
                "feature_set": "metadata_no_source",
                "k": 1,
                "training_rows": str(training_path),
                "pixel_features": None,
                "quality_floor": 80.0,
                "fallback": "router",
                "selection_reason": "content_classifier_preferred_candidate",
            }
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
            "--domain",
            "image",
            "--quality-target",
            "high",
            "--quality-floor",
            "80",
            "--content-classifier",
            "--content-classifier-mode",
            "apply",
            "--content-classifier-config",
            str(config_path),
            "--content-classifier-width",
            "1000",
            "--content-classifier-height",
            "1000",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["content_classifier"]["applied"] is True
    assert report["decision"]["selected"]["codec"] == "JPEG"
    assert report["decision"]["selected"]["config"] == "q=85"
    assert (
        report["decision"]["decision_trace"]["selected_reason"]
        == "content_classifier_preferred_candidate"
    )


def test_router_content_classifier_apply_falls_back_when_prediction_not_safe():
    csv_path = _tmp_path("classifier_apply_fallback_rde.csv")
    training_path = _tmp_path("classifier_apply_fallback_training.csv")
    config_path = _tmp_path("classifier_apply_fallback.json")
    report_path = _tmp_path("classifier_apply_fallback_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "JPEG,q=85,1.2,60.0,1.0,20.0",
                "HEVC,crf=15,1.0,95.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    training_path.write_text(
        "\n".join(
            [
                "dataset,image,image_id,oracle_codec,oracle_config,megapixels,aspect_ratio,resolution_class,orientation_class",
                "A,img1,A::img1,JPEG,q=85,1.0,1.0,medium,squareish",
                "B,img2,B::img2,HEVC,crf=15,8.0,1.7,huge,landscape",
            ]
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "model_type": "knn_oracle_classifier",
                "feature_set": "metadata_no_source",
                "k": 1,
                "training_rows": str(training_path),
                "pixel_features": None,
                "quality_floor": 80.0,
                "fallback": "router",
                "selection_reason": "content_classifier_preferred_candidate",
            }
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
            "--domain",
            "image",
            "--quality-target",
            "high",
            "--quality-floor",
            "80",
            "--content-classifier",
            "--content-classifier-mode",
            "apply",
            "--content-classifier-config",
            str(config_path),
            "--content-classifier-width",
            "1000",
            "--content-classifier-height",
            "1000",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["content_classifier"]["applied"] is False
    assert report["decision"]["selected"]["codec"] == "HEVC"
    assert report["decision"]["selected"]["config"] == "crf=15"
    assert (
        "content_classifier_prediction_not_admissible_fallback_to_router"
        in report["content_classifier"]["warnings"]
    )
