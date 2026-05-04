import pytest

pytest.importorskip("sklearn")

from src.router.content_oracle_classifier_sklearn_ablation import (
    run_sklearn_ablation,
)


def _rows():
    return [
        {
            "image_id": "A::img1",
            "dataset": "A",
            "image": "img1",
            "oracle_label": "JPEG|q=85",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
            "oracle_cost": 0.10,
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "megapixels": 0.50,
            "aspect_ratio": 1.0,
            "resolution_class": "small",
            "orientation_class": "squareish",
        },
        {
            "image_id": "A::img2",
            "dataset": "A",
            "image": "img2",
            "oracle_label": "JPEG|q=85",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
            "oracle_cost": 0.11,
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "megapixels": 0.60,
            "aspect_ratio": 1.1,
            "resolution_class": "small",
            "orientation_class": "landscape",
        },
        {
            "image_id": "B::img3",
            "dataset": "B",
            "image": "img3",
            "oracle_label": "HEVC|crf=15",
            "oracle_codec": "HEVC",
            "oracle_config": "crf=15",
            "oracle_cost": 0.10,
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "megapixels": 4.20,
            "aspect_ratio": 1.5,
            "resolution_class": "large",
            "orientation_class": "landscape",
        },
        {
            "image_id": "B::img4",
            "dataset": "B",
            "image": "img4",
            "oracle_label": "HEVC|crf=15",
            "oracle_codec": "HEVC",
            "oracle_config": "crf=15",
            "oracle_cost": 0.12,
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "megapixels": 3.90,
            "aspect_ratio": 1.4,
            "resolution_class": "large",
            "orientation_class": "landscape",
        },
        {
            "image_id": "C::img5",
            "dataset": "C",
            "image": "img5",
            "oracle_label": "JXL|d=1.0",
            "oracle_codec": "JXL",
            "oracle_config": "d=1.0",
            "oracle_cost": 0.10,
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "megapixels": 1.20,
            "aspect_ratio": 0.75,
            "resolution_class": "medium",
            "orientation_class": "portrait",
        },
        {
            "image_id": "C::img6",
            "dataset": "C",
            "image": "img6",
            "oracle_label": "JXL|d=1.0",
            "oracle_codec": "JXL",
            "oracle_config": "d=1.0",
            "oracle_cost": 0.11,
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "megapixels": 1.10,
            "aspect_ratio": 0.70,
            "resolution_class": "medium",
            "orientation_class": "portrait",
        },
    ]


def _candidate_lookup(rows):
    labels = [
        ("JPEG", "q=85"),
        ("HEVC", "crf=15"),
        ("JXL", "d=1.0"),
    ]

    lookup = {}

    for row in rows:
        oracle = (row["oracle_codec"], row["oracle_config"])

        for codec, config in labels:
            if (codec, config) == oracle:
                cost = float(row["oracle_cost"])
            elif (codec, config) == ("HEVC", "crf=15"):
                cost = 0.30
            else:
                cost = 0.45

            lookup[(row["image_id"], codec, config)] = {
                "quality": 90.0,
                "J_RDE": cost,
            }

    return lookup


def test_run_sklearn_ablation_compares_optional_models():
    rows = _rows()

    ablation = run_sklearn_ablation(
        rows=rows,
        candidate_lookup=_candidate_lookup(rows),
        feature_sets=["metadata_no_source"],
        models=["knn", "decision_tree"],
        k_values=[1, 3],
        evaluation_modes=["leave_one_image_out", "leave_one_dataset_out"],
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
    )

    summary = ablation["summary"]
    decisions = ablation["decisions"]

    assert len(summary) == 6
    assert decisions

    model_ids = {row["model_id"] for row in summary}
    assert model_ids == {"knn", "decision_tree"}

    knn_rows = [row for row in summary if row["model_id"] == "knn"]
    tree_rows = [row for row in summary if row["model_id"] == "decision_tree"]

    assert {row["k"] for row in knn_rows} == {1, 3}
    assert all(row["k"] == "" for row in tree_rows)
    assert all("mean_regret" in row for row in summary)
