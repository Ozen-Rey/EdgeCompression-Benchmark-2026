import shutil
from pathlib import Path

from src.router.analysis.content_aware_paper_artifacts import (
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
    assert table[0]["method"] == "Baseline globale robusta"
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

    assert by_component["Feature pixel"]["mean_ms"] == "30.00"
    assert by_component["Feature pixel"]["ratio_vs_jpeg_global"] == "5.00"
    assert by_component["Feature pixel"]["ratio_vs_hevc_global"] == "0.10"


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


def test_build_artifacts_writes_real_plot_files():
    root = _reset()

    benchmark = root / "benchmark_plots.csv"
    overhead = root / "overhead_plots.csv"
    sensitivity = root / "sensitivity_plots.csv"
    oracle = root / "oracle_plots.csv"
    out = root / "out_plots"

    benchmark.write_text(
        "\n".join(
            [
                "method_id,evaluation_protocol,deployment_setting,feature_set,k,selected_policy,mean_regret,relative_regret_reduction,accuracy,fallback_rate",
                "robust_global_baseline,global_coverage_oracle_analysis,source_agnostic,none,,HEVC|crf=15,0.09047,0.0,0.03125,0.0",
                "source_aware_dataset_majority_policy,leave-one-out,batch_known_source,dataset,,dataset_majority,0.01177,0.86995,0.77083,0.1875",
                "best_source_agnostic_knn_leave_one_image_out,leave_one_image_out,source_agnostic,metadata_no_source,11,knn,0.01167,0.87105,0.76042,0.20833",
                "best_source_agnostic_knn_leave_one_dataset_out,leave_one_dataset_out,source_agnostic,metadata_no_source,7,knn,0.02197,0.75716,0.57292,0.28125",
                "per_image_oracle,oracle,not_deployable,oracle,,oracle,0.0,1.0,1.0,0.0",
            ]
        ),
        encoding="utf-8",
    )

    overhead.write_text(
        "\n".join(
            [
                "component,case_id,scope,dataset,codec,config,num_samples,mean_ms,p90_ms,ratio_vs_jpeg_global,ratio_vs_hevc_global",
                "content_feature_extraction,pixel_features_long_side_256,all_images,,,96,30.52,52.36,4.84,0.09",
                "content_feature_extraction,metadata_no_source,image_header_or_known_metadata,,,96,,,,",
                "encoding_time,jpeg_q85_tecnick,source_filtered_tecnick,tecnick,JPEG,q=85,24,4.52,5.17,0.72,0.01",
                "encoding_time,jpeg_q85_global,global_all_datasets,,JPEG,q=85,96,6.31,10.84,1.00,0.02",
                "encoding_time,jxl_d1_global,global_all_datasets,,JXL,d=1.0,96,134.01,223.10,21.25,0.39",
                "encoding_time,hevc_crf15_global,global_all_datasets,,HEVC,crf=15,96,342.22,548.32,54.26,1.00",
            ]
        ),
        encoding="utf-8",
    )

    sensitivity.write_text(
        "\n".join(
            [
                "evaluation_mode,feature_set,k,accuracy,mean_regret,relative_regret_reduction,fallback_rate,best_for_evaluation_mode",
                "leave_one_image_out,metadata_no_source,1,0.625,0.02099,0.76795,0.19792,False",
                "leave_one_image_out,metadata_no_source,5,0.76042,0.01213,0.86588,0.20833,False",
                "leave_one_image_out,metadata_no_source,11,0.76042,0.01167,0.87105,0.20833,True",
                "leave_one_dataset_out,metadata_no_source,1,0.59375,0.02245,0.75181,0.22917,False",
                "leave_one_dataset_out,metadata_no_source,7,0.57292,0.02197,0.75716,0.28125,True",
                "leave_one_dataset_out,metadata_no_source,11,0.57292,0.02197,0.75716,0.28125,False",
                "leave_one_dataset_out,pixel_no_source,1,0.46875,0.02904,0.67904,0.25,False",
                "leave_one_dataset_out,pixel_no_source,3,0.47917,0.02792,0.69142,0.23958,False",
            ]
        ),
        encoding="utf-8",
    )

    oracle.write_text(
        "\n".join(
            [
                "section,label,count",
                "oracle_count,JPEG q=85,63",
                "oracle_count,JXL d=1.0,30",
                "oracle_count,HEVC crf=15,3",
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
        make_plots=True,
    )

    fig_dir = Path(paths["figures_dir"])
    expected_figures = [
        "v09_mean_regret_methods.png",
        "v09_relative_regret_reduction_methods.png",
        "v09_k_sensitivity_leave_one_image_out.png",
        "v09_k_sensitivity_leave_one_dataset_out.png",
        "v09_overhead_vs_encoding.png",
        "v09_oracle_distribution.png",
    ]

    for name in expected_figures:
        path = fig_dir / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
