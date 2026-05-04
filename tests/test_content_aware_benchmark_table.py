import shutil
from pathlib import Path

from src.router.content_aware_benchmark_table import (
    build_content_aware_benchmark_tables,
)


TEST_DIR = Path("tests/_tmp/content_aware_benchmark_table")


def _reset():
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR


def test_build_content_aware_benchmark_table_selects_best_rows():
    root = _reset()

    oracle = root / "oracle.csv"
    policy = root / "policy.csv"
    sweep = root / "sweep.csv"

    oracle.write_text(
        "\n".join(
            [
                "section,key,value",
                "summary,num_images_analyzed,10",
                "global_best,codec,HEVC",
                "global_best,config,crf=15",
                "oracle_diversity,oracle_matches_global_count,2",
                "regret,mean,0.10",
                "regret,median,0.09",
                "regret,p90,0.15",
                "regret,max,0.20",
            ]
        ),
        encoding="utf-8",
    )

    policy.write_text(
        "\n".join(
            [
                "section,key,value",
                "summary,evaluation_mode,leave-one-out",
                "accuracy,oracle_match_rate,0.8",
                "regret,mean,0.02",
                "regret,median,0.0",
                "regret,p90,0.05",
                "regret,max,0.07",
                "global_regret,mean,0.10",
                "regret_reduction,mean,0.08",
                "regret_reduction,relative_mean,0.8",
                "fallback,fallback_rate,0.1",
                "feasibility,infeasible_rate,0.0",
            ]
        ),
        encoding="utf-8",
    )

    sweep.write_text(
        "\n".join(
            [
                "evaluation_mode,feature_set,k,num_images,accuracy,fallback_rate,infeasible_rate,mean_regret,median_regret,p90_regret,max_regret,global_mean_regret,mean_regret_reduction,relative_regret_reduction",
                "leave_one_image_out,metadata_no_source,1,10,0.6,0.2,0.0,0.04,0.0,0.1,0.2,0.10,0.06,0.6",
                "leave_one_image_out,metadata_no_source,3,10,0.7,0.1,0.0,0.03,0.0,0.08,0.15,0.10,0.07,0.7",
                "leave_one_dataset_out,metadata_no_source,1,10,0.5,0.3,0.0,0.06,0.0,0.12,0.25,0.10,0.04,0.4",
                "leave_one_dataset_out,metadata_no_source,3,10,0.6,0.2,0.0,0.05,0.0,0.10,0.20,0.10,0.05,0.5",
            ]
        ),
        encoding="utf-8",
    )

    tables = build_content_aware_benchmark_tables(
        oracle_summary_csv=str(oracle),
        dataset_policy_summary_csv=str(policy),
        classifier_sweep_summary_csv=str(sweep),
    )

    paper = tables["paper_table"]
    by_id = {row["method_id"]: row for row in paper}

    assert len(paper) == 5

    assert by_id["robust_global_baseline"]["mean_regret"] == 0.10
    assert by_id["robust_global_baseline"]["accuracy"] == 0.2

    assert by_id["source_aware_dataset_majority_policy"]["mean_regret"] == 0.02
    assert by_id["source_aware_dataset_majority_policy"]["requires_source_label"] is True

    assert by_id["best_source_agnostic_knn_leave_one_image_out"]["k"] == "3"
    assert by_id["best_source_agnostic_knn_leave_one_image_out"]["mean_regret"] == 0.03

    assert by_id["best_source_agnostic_knn_leave_one_dataset_out"]["k"] == "3"
    assert by_id["best_source_agnostic_knn_leave_one_dataset_out"]["mean_regret"] == 0.05

    assert by_id["per_image_oracle"]["mean_regret"] == 0.0
    assert by_id["per_image_oracle"]["uses_oracle"] is True
