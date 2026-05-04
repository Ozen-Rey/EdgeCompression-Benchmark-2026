import shutil
from pathlib import Path

from PIL import Image

from src.router.content_image_features import (
    extract_features_from_manifest,
    extract_image_features,
    summarize_feature_rows,
)


TEST_DIR = Path("tests/_tmp/content_image_features")


def _reset_test_dir() -> Path:
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR


def test_extract_image_features_detects_flat_rgb_image():
    root = _reset_test_dir()
    path = root / "flat.png"

    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(path)

    features = extract_image_features(str(path), resize_long_side=32)

    assert features["width"] == 64
    assert features["height"] == 64
    assert features["original_mode"] == "RGB"
    assert features["has_alpha"] is False
    assert features["luminance_std"] == 0.0
    assert features["edge_density"] == 0.0
    assert features["flat_area_ratio"] > 0.95
    assert features["texture_class"] == "flat"
    assert features["color_class"] == "grayscale_like"


def test_extract_image_features_detects_alpha_channel():
    root = _reset_test_dir()
    path = root / "alpha.png"

    Image.new("RGBA", (32, 32), color=(255, 0, 0, 128)).save(path)

    features = extract_image_features(str(path), resize_long_side=32)

    assert features["original_mode"] == "RGBA"
    assert features["has_alpha"] is True


def test_extract_features_from_manifest_reads_rows():
    root = _reset_test_dir()
    path = root / "img.png"
    manifest = root / "manifest.csv"

    Image.new("RGB", (20, 10), color=(0, 255, 0)).save(path)

    manifest.write_text(
        "\n".join(
            [
                "dataset,image,path,width,height,pixels,mode,status",
                f"toy,img.png,{path.as_posix()},20,10,200,RGB,matched",
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_features_from_manifest(str(manifest), resize_long_side=16)

    assert len(rows) == 1
    assert rows[0]["dataset"] == "toy"
    assert rows[0]["image"] == "img.png"
    assert rows[0]["image_id"] == "toy::img.png"
    assert rows[0]["width"] == 20
    assert rows[0]["height"] == 10


def test_summarize_feature_rows_counts_classes():
    rows = [
        {
            "dataset": "A",
            "resolution_class": "small",
            "orientation_class": "landscape",
            "entropy_class": "low_entropy",
            "edge_class": "low_edge",
            "texture_class": "flat",
            "color_class": "grayscale_like",
            "feature_overhead_ms": 1.0,
            "luminance_std": 0.0,
            "luminance_entropy_norm": 0.1,
            "edge_density": 0.0,
            "colorfulness": 0.0,
        },
        {
            "dataset": "A",
            "resolution_class": "small",
            "orientation_class": "portrait",
            "entropy_class": "high_entropy",
            "edge_class": "high_edge",
            "texture_class": "textured",
            "color_class": "colorful",
            "feature_overhead_ms": 3.0,
            "luminance_std": 20.0,
            "luminance_entropy_norm": 0.9,
            "edge_density": 0.2,
            "colorfulness": 30.0,
        },
    ]

    summary = summarize_feature_rows(rows)
    lookup = {(r["section"], r["key"]): r["value"] for r in summary}

    assert lookup[("summary", "num_images")] == 2
    assert lookup[("overhead", "mean_ms")] == 2.0
    assert lookup[("dataset_count", "A")] == 2
    assert lookup[("resolution_class_count", "small")] == 2
