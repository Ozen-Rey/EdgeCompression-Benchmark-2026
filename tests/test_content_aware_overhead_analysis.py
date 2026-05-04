import shutil
from pathlib import Path

from src.router.content_aware_overhead_analysis import (
    build_knn_sensitivity_table,
    build_overhead_table,
)


TEST_DIR = Path("tests/_tmp/content_aware_overhead_analysis")


def _reset():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR


def test_build_overhead_table_computes_pixel_over_encoding_ratio():
    root = _reset()

    pixel = root / "pixel.csv"
    benchmark = root / "benchmark.csv"

    pixel.write_text(
        "\n".join(
            [
                "dataset,image,image_id,feature_overhead_ms",
                "tecnick,a,tecnick::a,10.0",
                "tecnick,b,tecnick::b,20.0",
            ]
        ),
        encoding="utf-8",
    )

    benchmark.write_text(
        "\n".join(
            [
                "dataset,codec,param,time_ms",
                "tecnick,JPEG,q=85,5.0",
                "tecnick,JPEG,q=85,10.0",
                "kodak,HEVC,crf=15,100.0",
                "tecnick,HEVC,crf=15,200.0",
                "kodak,JXL,d=1.0,50.0",
            ]
        ),
        encoding="utf-8",
    )

    table = build_overhead_table(
        pixel_features_csv=str(pixel),
        benchmark_csv=str(benchmark),
    )

    by_case = {row["case_id"]: row for row in table}

    assert by_case["pixel_features_long_side_256"]["mean_ms"] == 15.0
    assert by_case["jpeg_q85_tecnick"]["mean_ms"] == 7.5
    assert by_case["jpeg_q85_tecnick"]["pixel_feature_mean_over_this_mean"] == 2.0
    assert by_case["hevc_crf15_global"]["mean_ms"] == 150.0


def test_build_knn_sensitivity_table_marks_best_rows():
    root = _reset()
    sweep = root / "sweep.csv"

    sweep.write_text(
        "\n".join(
            [
                "evaluation_mode,feature_set,k,accuracy,fallback_rate,infeasible_rate,mean_regret,median_regret,p90_regret,max_regret,relative_regret_reduction",
                "leave_one_image_out,metadata_no_source,1,0.6,0.2,0.0,0.04,0.0,0.1,0.2,0.6",
                "leave_one_image_out,metadata_no_source,3,0.7,0.1,0.0,0.03,0.0,0.08,0.15,0.7",
                "leave_one_image_out,pixel_no_source,1,0.5,0.2,0.0,0.05,0.0,0.1,0.2,0.5",
                "leave_one_dataset_out,metadata_no_source,7,0.5,0.3,0.0,0.06,0.0,0.12,0.25,0.4",
                "leave_one_dataset_out,metadata_no_source,9,0.6,0.2,0.0,0.05,0.0,0.10,0.20,0.5",
            ]
        ),
        encoding="utf-8",
    )

    table = build_knn_sensitivity_table(
        classifier_sweep_summary_csv=str(sweep),
    )

    best_eval = [
        row for row in table
        if row["best_for_evaluation_mode"]
    ]

    assert len(best_eval) == 2

    by_key = {
        (row["evaluation_mode"], row["feature_set"], row["k"]): row
        for row in table
    }

    assert by_key[("leave_one_image_out", "metadata_no_source", "3")]["best_for_evaluation_mode"] is True
    assert by_key[("leave_one_dataset_out", "metadata_no_source", "9")]["best_for_evaluation_mode"] is True