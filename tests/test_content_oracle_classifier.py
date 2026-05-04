import shutil
from pathlib import Path

from src.router.content_oracle_classifier import (
    evaluate_oracle_classifier,
    load_classifier_rows,
)


TEST_DIR = Path("tests/_tmp/content_oracle_classifier")


def _reset_test_dir() -> Path:
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR


def test_load_classifier_rows_joins_metadata_and_pixel_features():
    root = _reset_test_dir()
    metadata = root / "metadata.csv"
    pixel = root / "pixel.csv"

    metadata.write_text(
        "\n".join(
            [
                "dataset,image,image_id,oracle_codec,oracle_config,oracle_cost,global_codec,global_config,regret,megapixels,aspect_ratio,resolution_class,orientation_class",
                "A,img1,A::img1,JPEG,q=85,0.0,HEVC,crf=15,1.0,1.0,1.0,medium,squareish",
            ]
        ),
        encoding="utf-8",
    )

    pixel.write_text(
        "\n".join(
            [
                "dataset,image,image_id,luminance_std,luminance_entropy_norm,gradient_mean,gradient_std,edge_density,flat_area_ratio,colorfulness,entropy_class,edge_class,texture_class,color_class",
                "A,img1,A::img1,10.0,0.5,3.0,5.0,0.1,0.8,20.0,medium_entropy,medium_edge,moderate,normal_color",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_classifier_rows(
        metadata_oracle_csv=str(metadata),
        pixel_features_csv=str(pixel),
    )

    assert len(rows) == 1
    assert rows[0]["image_id"] == "A::img1"
    assert rows[0]["oracle_label"] == "JPEG|q=85"
    assert rows[0]["luminance_std"] == "10.0"


def test_oracle_classifier_predicts_by_source_when_enabled():
    rows = [
        {
            "dataset": "A",
            "image": "img1",
            "image_id": "A::img1",
            "oracle_label": "JPEG|q=85",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "1.0",
            "megapixels": "1.0",
            "aspect_ratio": "1.0",
            "resolution_class": "medium",
            "orientation_class": "squareish",
            "luminance_std": "10.0",
            "luminance_entropy_norm": "0.5",
            "gradient_mean": "3.0",
            "gradient_std": "5.0",
            "edge_density": "0.1",
            "flat_area_ratio": "0.8",
            "colorfulness": "20.0",
            "entropy_class": "medium_entropy",
            "edge_class": "medium_edge",
            "texture_class": "moderate",
            "color_class": "normal_color",
        },
        {
            "dataset": "A",
            "image": "img2",
            "image_id": "A::img2",
            "oracle_label": "JPEG|q=85",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "1.0",
            "megapixels": "1.0",
            "aspect_ratio": "1.0",
            "resolution_class": "medium",
            "orientation_class": "squareish",
            "luminance_std": "11.0",
            "luminance_entropy_norm": "0.5",
            "gradient_mean": "3.0",
            "gradient_std": "5.0",
            "edge_density": "0.1",
            "flat_area_ratio": "0.8",
            "colorfulness": "20.0",
            "entropy_class": "medium_entropy",
            "edge_class": "medium_edge",
            "texture_class": "moderate",
            "color_class": "normal_color",
        },
    ]

    candidate_lookup = {
        ("A::img1", "JPEG", "q=85"): {"quality": 90.0, "J_RDE": 0.0},
        ("A::img1", "HEVC", "crf=15"): {"quality": 95.0, "J_RDE": 1.0},
        ("A::img2", "JPEG", "q=85"): {"quality": 90.0, "J_RDE": 0.0},
        ("A::img2", "HEVC", "crf=15"): {"quality": 95.0, "J_RDE": 1.0},
    }

    evaluation = evaluate_oracle_classifier(
        rows=rows,
        candidate_lookup=candidate_lookup,
        feature_set="all_with_source",
        k=1,
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
    )

    summary = {(row["section"], row["key"]): row["value"] for row in evaluation["summary"]}

    assert summary[("accuracy", "oracle_match_rate")] == 1.0
    assert summary[("regret", "mean")] == 0.0
    assert summary[("regret_reduction", "mean")] == 1.0


def test_oracle_classifier_falls_back_when_prediction_not_feasible():
    rows = [
        {
            "dataset": "A",
            "image": "img1",
            "image_id": "A::img1",
            "oracle_label": "JPEG|q=85",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "1.0",
            "megapixels": "1.0",
            "aspect_ratio": "1.0",
            "resolution_class": "medium",
            "orientation_class": "squareish",
            "luminance_std": "10.0",
            "luminance_entropy_norm": "0.5",
            "gradient_mean": "3.0",
            "gradient_std": "5.0",
            "edge_density": "0.1",
            "flat_area_ratio": "0.8",
            "colorfulness": "20.0",
            "entropy_class": "medium_entropy",
            "edge_class": "medium_edge",
            "texture_class": "moderate",
            "color_class": "normal_color",
        },
        {
            "dataset": "A",
            "image": "img2",
            "image_id": "A::img2",
            "oracle_label": "JPEG|q=85",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "1.0",
            "megapixels": "1.0",
            "aspect_ratio": "1.0",
            "resolution_class": "medium",
            "orientation_class": "squareish",
            "luminance_std": "11.0",
            "luminance_entropy_norm": "0.5",
            "gradient_mean": "3.0",
            "gradient_std": "5.0",
            "edge_density": "0.1",
            "flat_area_ratio": "0.8",
            "colorfulness": "20.0",
            "entropy_class": "medium_entropy",
            "edge_class": "medium_edge",
            "texture_class": "moderate",
            "color_class": "normal_color",
        },
    ]

    candidate_lookup = {
        ("A::img1", "JPEG", "q=85"): {"quality": 60.0, "J_RDE": 0.0},
        ("A::img1", "HEVC", "crf=15"): {"quality": 95.0, "J_RDE": 1.0},
        ("A::img2", "JPEG", "q=85"): {"quality": 60.0, "J_RDE": 0.0},
        ("A::img2", "HEVC", "crf=15"): {"quality": 95.0, "J_RDE": 1.0},
    }

    evaluation = evaluate_oracle_classifier(
        rows=rows,
        candidate_lookup=candidate_lookup,
        feature_set="all_with_source",
        k=1,
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
    )

    assert all(row["fallback_used"] for row in evaluation["decisions"])
    assert all(row["selected_codec"] == "HEVC" for row in evaluation["decisions"])
