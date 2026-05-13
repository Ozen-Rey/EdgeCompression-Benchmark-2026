import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from .content_metadata_features import (
        _orientation_class,
        _resolution_class,
    )
    from .content_metadata_policy import load_metadata_oracle_rows
    from ..content_oracle_classifier import (
        FEATURE_SETS,
        _predict_knn_label,
        _split_label,
        load_classifier_rows,
    )
except ImportError:
    from content_metadata_features import (
        _orientation_class,
        _resolution_class,
    )
    from content_metadata_policy import load_metadata_oracle_rows
    from content_oracle_classifier import (
        FEATURE_SETS,
        _predict_knn_label,
        _split_label,
        load_classifier_rows,
    )


def load_content_classifier_config(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Content classifier config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("model_type", "knn_oracle_classifier") != "knn_oracle_classifier":
        raise ValueError("Only model_type='knn_oracle_classifier' is supported.")

    feature_set = str(config.get("feature_set", ""))

    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set={feature_set}. "
            f"Expected one of: {', '.join(sorted(FEATURE_SETS))}"
        )

    k = int(config.get("k", 0))
    if k <= 0:
        raise ValueError("Classifier k must be positive.")

    training_rows = config.get("training_rows") or config.get("training_csv")
    if not training_rows:
        raise ValueError("Classifier config requires training_rows.")

    config["training_rows"] = training_rows

    fallback = str(config.get("fallback", "router"))
    if fallback != "router":
        raise ValueError("Only fallback='router' is currently supported.")

    return config


def _oracle_label_from_row(row: Dict[str, Any]) -> str:
    codec = str(row.get("oracle_codec", "")).strip()
    config = str(row.get("oracle_config", "")).strip()

    if not codec or not config:
        raise ValueError("Training row is missing oracle_codec/oracle_config.")

    return f"{codec}|{config}"


def load_training_rows_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    training_rows = str(config.get("training_rows") or config.get("training_csv"))
    pixel_features = config.get("pixel_features")

    if pixel_features:
        rows = load_classifier_rows(
            metadata_oracle_csv=training_rows,
            pixel_features_csv=str(pixel_features),
        )
    else:
        rows = load_metadata_oracle_rows(training_rows)

        for row in rows:
            if row.get("oracle_label"):
                continue

            row["oracle_label"] = _oracle_label_from_row(row)

    if not rows:
        raise ValueError("No classifier training rows loaded.")

    return rows


def build_metadata_no_source_features(
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    pixels = width * height
    megapixels = pixels / 1_000_000.0
    aspect_ratio = width / height

    return {
        "width": width,
        "height": height,
        "pixels": pixels,
        "megapixels": megapixels,
        "aspect_ratio": aspect_ratio,
        "resolution_class": _resolution_class(megapixels),
        "orientation_class": _orientation_class(width, height),
    }


def extract_metadata_features_from_image(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    with Image.open(p) as im:
        width = int(im.width)
        height = int(im.height)
        mode = str(im.mode)

    features = build_metadata_no_source_features(
        width=width,
        height=height,
    )

    features["path"] = str(p)
    features["mode"] = mode

    return features


def predict_content_classifier(
    *,
    config: Dict[str, Any],
    content_features: Dict[str, Any],
    training_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    enabled = bool(config.get("enabled", True))
    feature_set = str(config.get("feature_set", "metadata_no_source"))
    k = int(config.get("k", 7))

    report: Dict[str, Any] = {
        "enabled": enabled,
        "model_type": config.get("model_type", "knn_oracle_classifier"),
        "feature_set": feature_set,
        "k": k,
        "training_rows": config.get("training_rows"),
        "pixel_features": config.get("pixel_features"),
        "quality_floor": config.get("quality_floor"),
        "fallback": config.get("fallback", "router"),
        "selection_reason": config.get(
            "selection_reason",
            "content_classifier_preferred_candidate",
        ),
        "features": dict(content_features),
        "prediction": None,
        "warnings": [],
        "reasons": [],
    }

    if not enabled:
        report["reasons"].append("content_classifier_disabled")
        return report

    if training_rows is None:
        training_rows = load_training_rows_from_config(config)

    if not training_rows:
        raise ValueError("No training rows available for content classifier.")

    predicted_label = _predict_knn_label(
        train_rows=training_rows,
        test_row=content_features,
        feature_set=feature_set,
        k=k,
    )

    codec, selected_config = _split_label(predicted_label)

    report["prediction"] = {
        "codec": codec,
        "config": selected_config,
        "label": predicted_label,
        "source": "knn_oracle_classifier",
    }

    report["reasons"].append("content_classifier_prediction_available")

    return report


def write_json(path: str, data: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run source-agnostic content classifier prediction for an image."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--out",
        default="results/routing_context/v09_content_classifier_prediction.json",
    )

    args = parser.parse_args()

    config = load_content_classifier_config(args.config)

    if args.image:
        features = extract_metadata_features_from_image(args.image)
    else:
        if args.width is None or args.height is None:
            raise ValueError("Provide either --image or both --width and --height.")

        features = build_metadata_no_source_features(
            width=args.width,
            height=args.height,
        )

    report = predict_content_classifier(
        config=config,
        content_features=features,
    )

    write_json(args.out, report)

    prediction = report.get("prediction") or {}

    print("\n=== R-D-E Content Classifier Prediction ===")
    print(f"Config:        {args.config}")
    print(f"Feature set:   {report.get('feature_set')}")
    print(f"k:             {report.get('k')}")
    print(f"Features:      {features}")
    print(
        "Prediction:    "
        f"{prediction.get('codec')} {prediction.get('config')}"
    )
    print(f"Report:        {args.out}")


if __name__ == "__main__":
    main()
