from pathlib import Path

from src.router.content_metadata_features import (
    extract_metadata_features,
    join_metadata_with_oracle,
    summarize_metadata_oracle,
)


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "content_metadata_features"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_extract_metadata_features_deduplicates_images():
    csv_path = _tmp_path("benchmark.csv")

    csv_path.write_text(
        "\n".join(
            [
                "dataset,image,codec,param,width,height,pixels",
                "A,img1,JPEG,q=85,100,50,5000",
                "A,img1,JXL,d=1.0,100,50,5000",
                "A,img2,JPEG,q=85,40,80,3200",
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_metadata_features(str(csv_path))

    assert len(rows) == 2

    by_id = {r["image_id"]: r for r in rows}

    assert by_id["A::img1"]["aspect_ratio"] == 2.0
    assert by_id["A::img1"]["orientation_class"] == "landscape"
    assert by_id["A::img2"]["orientation_class"] == "portrait"


def test_join_metadata_with_oracle_adds_labels():
    metadata = [
        {
            "dataset": "A",
            "image": "img1",
            "image_id": "A::img1",
            "width": 100,
            "height": 50,
            "pixels": 5000,
            "megapixels": 0.005,
            "aspect_ratio": 2.0,
            "orientation_class": "landscape",
            "resolution_class": "small",
        }
    ]

    oracle = {
        "A::img1": {
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
            "regret": "0.1",
        }
    }

    joined = join_metadata_with_oracle(metadata, oracle)

    assert len(joined) == 1
    assert joined[0]["oracle_codec"] == "JPEG"
    assert joined[0]["oracle_config"] == "q=85"
    assert joined[0]["regret"] == "0.1"


def test_summarize_metadata_oracle_counts_oracles_by_group():
    rows = [
        {
            "dataset": "A",
            "resolution_class": "small",
            "orientation_class": "landscape",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
        },
        {
            "dataset": "A",
            "resolution_class": "small",
            "orientation_class": "portrait",
            "oracle_codec": "JXL",
            "oracle_config": "d=1.0",
        },
        {
            "dataset": "B",
            "resolution_class": "large",
            "orientation_class": "landscape",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
        },
    ]

    summary = summarize_metadata_oracle(rows)

    lookup = {
        (r["section"], r["key"]): r["value"]
        for r in summary
    }

    assert lookup[("summary", "num_images")] == 3
    assert lookup[("oracle", "num_distinct_oracle_configs")] == 2
    assert lookup[("oracle_count", "JPEG|q=85")] == 2
    assert lookup[("dataset_count", "A")] == 2
    assert lookup[("resolution_class_count", "small")] == 2
