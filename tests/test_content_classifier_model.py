import json
import shutil
from pathlib import Path

from PIL import Image

from src.router.adaptation.content_classifier_model import (
    build_metadata_no_source_features,
    extract_metadata_features_from_image,
    load_content_classifier_config,
    load_training_rows_from_config,
    predict_content_classifier,
)
from tests.conftest import scratch_root


def _reset():
    root = scratch_root() / "content_classifier_model"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_build_metadata_no_source_features_classifies_geometry():
    features = build_metadata_no_source_features(width=1200, height=1200)

    assert features["pixels"] == 1440000
    assert features["megapixels"] == 1.44
    assert features["aspect_ratio"] == 1.0
    assert features["resolution_class"] == "medium"
    assert features["orientation_class"] == "squareish"


def test_extract_metadata_features_from_image_reads_dimensions():
    root = _reset()
    path = root / "image.png"

    Image.new("RGB", (64, 32), color=(255, 0, 0)).save(path)

    features = extract_metadata_features_from_image(str(path))

    assert features["width"] == 64
    assert features["height"] == 32
    assert features["orientation_class"] == "landscape"
    assert features["mode"] == "RGB"


def test_load_content_classifier_config_and_training_rows():
    root = _reset()

    training = root / "training.csv"
    config_path = root / "classifier.json"

    training.write_text(
        "\n".join(
            [
                "dataset,image,image_id,oracle_codec,oracle_config,megapixels,aspect_ratio,resolution_class,orientation_class",
                "A,img1,A::img1,JPEG,q=85,1.0,1.0,medium,squareish",
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
                "training_rows": str(training),
                "pixel_features": None,
                "fallback": "router",
            }
        ),
        encoding="utf-8",
    )

    config = load_content_classifier_config(str(config_path))
    rows = load_training_rows_from_config(config)

    assert config["feature_set"] == "metadata_no_source"
    assert rows[0]["oracle_label"] == "JPEG|q=85"


def test_predict_content_classifier_returns_codec_config():
    training_rows = [
        {
            "dataset": "A",
            "image": "img1",
            "image_id": "A::img1",
            "oracle_label": "JPEG|q=85",
            "megapixels": "1.0",
            "aspect_ratio": "1.0",
            "resolution_class": "medium",
            "orientation_class": "squareish",
        },
        {
            "dataset": "B",
            "image": "img2",
            "image_id": "B::img2",
            "oracle_label": "HEVC|crf=15",
            "megapixels": "8.0",
            "aspect_ratio": "1.7",
            "resolution_class": "huge",
            "orientation_class": "landscape",
        },
    ]

    config = {
        "enabled": True,
        "model_type": "knn_oracle_classifier",
        "feature_set": "metadata_no_source",
        "k": 1,
        "training_rows": "dummy.csv",
        "pixel_features": None,
        "fallback": "router",
    }

    features = build_metadata_no_source_features(width=1000, height=1000)

    report = predict_content_classifier(
        config=config,
        content_features=features,
        training_rows=training_rows,
    )

    assert report["prediction"]["codec"] == "JPEG"
    assert report["prediction"]["config"] == "q=85"
