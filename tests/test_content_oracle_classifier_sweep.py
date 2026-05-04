from src.router.content_oracle_classifier_sweep import (
    evaluate_leave_one_dataset_out,
    evaluate_leave_one_image_out,
    run_sweep,
)


def _toy_rows():
    return [
        {
            "dataset": "A",
            "image": "img1",
            "image_id": "A::img1",
            "oracle_label": "JPEG|q=85",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
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
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
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
        {
            "dataset": "B",
            "image": "img1",
            "image_id": "B::img1",
            "oracle_label": "HEVC|crf=15",
            "oracle_codec": "HEVC",
            "oracle_config": "crf=15",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "0.0",
            "megapixels": "2.0",
            "aspect_ratio": "1.0",
            "resolution_class": "large",
            "orientation_class": "squareish",
            "luminance_std": "40.0",
            "luminance_entropy_norm": "0.9",
            "gradient_mean": "30.0",
            "gradient_std": "20.0",
            "edge_density": "0.4",
            "flat_area_ratio": "0.1",
            "colorfulness": "40.0",
            "entropy_class": "high_entropy",
            "edge_class": "high_edge",
            "texture_class": "textured",
            "color_class": "colorful",
        },
        {
            "dataset": "B",
            "image": "img2",
            "image_id": "B::img2",
            "oracle_label": "HEVC|crf=15",
            "oracle_codec": "HEVC",
            "oracle_config": "crf=15",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "0.0",
            "megapixels": "2.0",
            "aspect_ratio": "1.0",
            "resolution_class": "large",
            "orientation_class": "squareish",
            "luminance_std": "41.0",
            "luminance_entropy_norm": "0.9",
            "gradient_mean": "30.0",
            "gradient_std": "20.0",
            "edge_density": "0.4",
            "flat_area_ratio": "0.1",
            "colorfulness": "40.0",
            "entropy_class": "high_entropy",
            "edge_class": "high_edge",
            "texture_class": "textured",
            "color_class": "colorful",
        },
    ]


def _toy_lookup():
    lookup = {}

    for row in _toy_rows():
        image_id = row["image_id"]
        lookup[(image_id, "JPEG", "q=85")] = {"quality": 90.0, "J_RDE": 0.0}
        lookup[(image_id, "HEVC", "crf=15")] = {"quality": 95.0, "J_RDE": 1.0}

    return lookup


def test_evaluate_leave_one_image_out_returns_one_decision_per_row():
    rows = _toy_rows()

    decisions = evaluate_leave_one_image_out(
        rows=rows,
        candidate_lookup=_toy_lookup(),
        feature_set="pixel_no_source",
        k=1,
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
    )

    assert len(decisions) == len(rows)
    assert {d["evaluation_mode"] for d in decisions} == {"leave_one_image_out"}


def test_evaluate_leave_one_dataset_out_returns_one_decision_per_row():
    rows = _toy_rows()

    decisions = evaluate_leave_one_dataset_out(
        rows=rows,
        candidate_lookup=_toy_lookup(),
        feature_set="pixel_no_source",
        k=1,
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
    )

    assert len(decisions) == len(rows)
    assert {d["evaluation_mode"] for d in decisions} == {"leave_one_dataset_out"}
    assert {d["fold_id"] for d in decisions} == {"A", "B"}


def test_run_sweep_produces_flat_summary_rows():
    sweep = run_sweep(
        rows=_toy_rows(),
        candidate_lookup=_toy_lookup(),
        feature_sets=["pixel_no_source"],
        k_values=[1, 3],
        evaluation_modes=["leave_one_image_out", "leave_one_dataset_out"],
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
    )

    assert len(sweep["summary"]) == 4
    assert len(sweep["decisions"]) == 16

    keys = {
        (row["evaluation_mode"], row["feature_set"], row["k"])
        for row in sweep["summary"]
    }

    assert ("leave_one_image_out", "pixel_no_source", 1) in keys
    assert ("leave_one_image_out", "pixel_no_source", 3) in keys
    assert ("leave_one_dataset_out", "pixel_no_source", 1) in keys
    assert ("leave_one_dataset_out", "pixel_no_source", 3) in keys
