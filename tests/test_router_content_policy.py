import json
from pathlib import Path

from src.router.rde_router import main
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "router_content_policy"
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
                "JPEG,q=85,1.0,95.0,1.0,20.0",
                "HEVC,crf=15,1.2,85.0,10.0,100.0",
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


def test_router_content_classifier_apply_rejects_admissible_but_noncompetitive_prediction():
    csv_path = _tmp_path("classifier_apply_noncompetitive_rde.csv")
    training_path = _tmp_path("classifier_apply_noncompetitive_training.csv")
    config_path = _tmp_path("classifier_apply_noncompetitive.json")
    report_path = _tmp_path("classifier_apply_noncompetitive_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,config,rate,quality,energy,time_ms,quality_mean,quality_min,quality_p10,quality_p25",
                "JPEG,q=85,10.0,90.0,10.0,10.0,90.0,90.0,90.0,90.0",
                "HEVC,crf=15,1.0,95.0,1.0,10.0,95.0,95.0,95.0,95.0",
            ]
        ),
        encoding="utf-8",
    )

    training_path.write_text(
        "\n".join(
            [
                "dataset,image,image_id,width,height,megapixels,aspect_ratio,resolution_class,orientation_class,oracle_label",
                "A,img1,A::img1,1000,1000,1.0,1.0,medium,squareish,JPEG|q=85",
                "A,img2,A::img2,1000,1000,1.0,1.0,medium,squareish,JPEG|q=85",
                "A,img3,A::img3,1000,1000,1.0,1.0,medium,squareish,JPEG|q=85",
            ]
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        json.dumps(
            {
                "training_csv": str(training_path),
                "feature_set": "metadata_no_source",
                "k": 3,
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
            "--domain",
            "image",
            "--quality-metric",
            "ssimulacra2",
            "--w-r",
            "0.2",
            "--w-e",
            "0.2",
            "--w-d",
            "0.6",
            "--quality-floor",
            "80",
            "--quality-constraint-stat",
            "min",
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

    assert report["content_classifier"]["prediction"]["codec"] == "JPEG"
    assert report["content_classifier"]["prediction"]["config"] == "q=85"

    assert report["content_classifier"]["applied"] is False
    assert report["decision"]["selected"]["codec"] == "HEVC"
    assert report["decision"]["selected"]["config"] == "crf=15"

    audit = report["decision"]["decision_trace"]["preferred_candidate"]

    assert audit["admissible"] is True
    assert audit["competitive"] is False
    assert audit["selected"] is False
    assert audit["preferred_ranking_cost"] > audit["best_ranking_cost"]

    assert (
        "content_classifier_prediction_not_j_total_competitive_fallback_to_router"
        in report["content_classifier"]["warnings"]
    )


def test_router_content_classifier_apply_does_not_override_better_j_total():
    csv_path = _tmp_path("classifier_system_fusion_rde.csv")
    training_path = _tmp_path("classifier_system_fusion_training.csv")
    config_path = _tmp_path("classifier_system_fusion.json")
    report_path = _tmp_path("classifier_system_fusion_report.json")

    csv_path.write_text(
        "\n".join(
            [
                "dataset,codec,config,rate,quality,energy,time_ms,quality_mean,quality_min,quality_p10,quality_p25,quality_std",
                "tecnick,JPEG,q=85,1.2,85.0,0.07,5.0,85.0,82.0,83.0,84.0,1.0",
                "tecnick,HEVC,crf=15,1.0,92.0,15.0,350.0,92.0,88.0,89.0,90.0,1.0",
            ]
        ),
        encoding="utf-8",
    )

    training_path.write_text(
        "\n".join(
            [
                "dataset,image,image_id,oracle_label,width,height,megapixels,aspect_ratio,resolution_class,orientation_class",
                "tecnick,img.png,tecnick::img.png,JPEG|q=85,1200,1200,1.44,1.0,medium,squareish",
            ]
        ),
        encoding="utf-8",
    )

    config_path.write_text(
        json.dumps(
            {
                "feature_set": "metadata_no_source",
                "k": 1,
                "training_csv": str(training_path),
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
            "--auto-weights",
            "--quality-floor",
            "80",
            "--quality-constraint-stat",
            "min",
            "--content-classifier",
            "--content-classifier-mode",
            "apply",
            "--content-classifier-config",
            str(config_path),
            "--content-classifier-width",
            "1200",
            "--content-classifier-height",
            "1200",
            "--system-features",
            "--system-policy-simulate",
            "battery=critical,cpu=busy,memory=constrained",
            "--system-penalty",
            "--system-penalty-mode",
            "apply",
            "--system-penalty-lambda",
            "0.25",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    selected = report["decision"]["selected"]
    trace = report["decision"]["decision_trace"]

    assert report["content_classifier"]["enabled"] is True
    assert report["content_classifier"]["mode"] == "apply"
    assert report["content_classifier"]["prediction"]["codec"] == "JPEG"
    assert report["content_classifier"]["prediction"]["config"] == "q=85"

    assert trace["ranking_key"] == "minimize_J_total"
    assert trace["system_penalty_applied"] is True

    assert selected["ranking_cost"] <= selected["J_total"] + 1e-12
    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
    assert trace["preferred_candidate"]["codec"] == "JPEG"
    assert trace["preferred_candidate"]["config"] == "q=85"
    assert trace["preferred_candidate"]["selected"] is False
    assert trace["preferred_candidate"]["competitive"] is False
    assert report["content_classifier"]["applied"] is False
    assert (
        "content_classifier_prediction_not_j_total_competitive_fallback_to_router"
        in report["content_classifier"]["warnings"]
    )
