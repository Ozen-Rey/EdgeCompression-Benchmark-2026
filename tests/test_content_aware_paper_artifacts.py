import shutil
from pathlib import Path

from src.router.content_aware_paper_artifacts import (
    build_artifacts,
    build_best_k_table,
    build_main_paper_table,
    build_overhead_paper_table,
)


TEST_DIR = Path("tests/_tmp/content_aware_paper_artifacts")


def _reset():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR


def test_build_main_paper_table_formats_core_methods():
    rows = [
        {
            "method_id": "robust_global_baseline",
            "evaluation_protocol": "global_coverage_oracle_analysis",
            "deployment_setting": "source_agnostic",
            "feature_set": "none",
            "selected_policy": "HEVC crf=15",
            "mean_regret": "0.090474",
            "relative_regret_reduction": "0.0",
            "accuracy": "0.03125",
            "fallback_rate": "0.0",
        },
        {
            "method_id": "per_image_oracle",
            "evaluation_protocol": "oracle",
            "deployment_setting": "not_deployable",
            "feature_set": "oracle",
            "selected_policy": "oracle",
            "mean_regret": "0.0",
            "relative_regret_reduction": "1.0",
            "accuracy": "1.0",
            "fallback_rate": "0.0",
        },
    ]

    table = build_main_paper_table(rows)

    assert len(table) == 2
    assert table[0]["method"] == "Robust global baseline"
    assert table[0]["mean_regret"] == "0.09047"
    assert table[1]["regret_reduction_percent"] == "100.0"


def test_build_overhead_paper_table_computes_ratios():
    rows = [
        {
            "case_id": "pixel_features_long_side_256",
            "num_samples": "10",
            "mean_ms": "30",
            "p90_ms": "40",
        },
        {"case_id": "metadata_no_source", "num_samples": "10", "mean_ms": "", "p90_ms": ""},
        {"case_id": "jpeg_q85_tecnick", "num_samples": "10", "mean_ms": "5", "p90_ms": "7"},
        {"case_id": "jpeg_q85_global", "num_samples": "10", "mean_ms": "6", "p90_ms": "8"},
        {"case_id": "jxl_d1_global", "num_samples": "10", "mean_ms": "120", "p90_ms": "130"},
        {"case_id": "hevc_crf15_global", "num_samples": "10", "mean_ms": "300", "p90_ms": "320"},
    ]

    table = build_overhead_paper_table(rows)
    by_component = {row["component"]: row for row in table}

    assert by_component["Pixel feature extraction"]["mean_ms"] == "30.00"
    assert by_component["Pixel feature extraction"]["ratio_vs_jpeg_global"] == "5.00"
    assert by_component["Pixel feature extraction"]["ratio_vs_hevc_global"] == "0.10"


def test_build_best_k_table_keeps_best_rows():
    rows = [
        {
            "evaluation_mode": "leave_one_image_out",
            "feature_set": "metadata_no_source",
            "k": "11",
            "mean_regret": "0.01166",
            "relative_regret_reduction": "0.87",
            "accuracy": "0.76",
            "fallback_rate": "0.20",
            "best_for_evaluation_mode": "True",
        },
        {
            "evaluation_mode": "leave_one_image_out",
            "feature_set": "metadata_no_source",
            "k": "7",
            "mean_regret": "0.012",
            "relative_regret_reduction": "0.86",
            "accuracy": "0.75",
            "fallback_rate": "0.20",
            "best_for_evaluation_mode": "False",
        },
    ]

    table = build_best_k_table(rows)

    assert len(table) == 1
    assert table[0]["protocol"] == "LOIO"
    assert table[0]["k"] == "11"


def test_build_artifacts_writes_csv_and_tex_without_plots():
    root = _reset()

    benchmark = root / "benchmark.csv"
    overhead = root / "overhead.csv"
    sensitivity = root / "sensitivity.csv"
    oracle = root / "oracle.csv"
    out = root / "out"

    benchmark.write_text(
        "\n".join(
            [
                "method_id,evaluation_protocol,deployment_setting,feature_set,k,selected_policy,mean_regret,relative_regret_reduction,accuracy,fallback_rate",
                "robust_global_baseline,global_coverage_oracle_analysis,source_agnostic,none,,HEVC crf=15,0.09,0.0,0.03,0.0",
                "per_image_oracle,oracle,not_deployable,oracle,,oracle,0.0,1.0,1.0,0.0",
            ]
        ),
        encoding="utf-8",
    )

    overhead.write_text(
        "\n".join(
            [
                "case_id,num_samples,mean_ms,p90_ms",
                "pixel_features_long_side_256,10,30,40",
                "metadata_no_source,10,,",
                "jpeg_q85_tecnick,10,5,7",
                "jpeg_q85_global,10,6,8",
                "jxl_d1_global,10,120,130",
                "hevc_crf15_global,10,300,320",
            ]
        ),
        encoding="utf-8",
    )

    sensitivity.write_text(
        "\n".join(
            [
                "evaluation_mode,feature_set,k,mean_regret,relative_regret_reduction,accuracy,fallback_rate,best_for_evaluation_mode",
                "leave_one_image_out,metadata_no_source,11,0.011,0.87,0.76,0.20,True",
            ]
        ),
        encoding="utf-8",
    )

    oracle.write_text(
        "\n".join(
            [
                "section,key,value",
                "oracle_count,JPEG|q=85,10",
            ]
        ),
        encoding="utf-8",
    )

    paths = build_artifacts(
        benchmark_table_csv=str(benchmark),
        overhead_table_csv=str(overhead),
        sensitivity_table_csv=str(sensitivity),
        oracle_summary_csv=str(oracle),
        out_dir=str(out),
        make_plots=False,
    )

    assert Path(paths["main_csv"]).exists()
    assert Path(paths["main_tex"]).exists()
    assert Path(paths["overhead_csv"]).exists()
    assert Path(paths["best_k_tex"]).exists()
