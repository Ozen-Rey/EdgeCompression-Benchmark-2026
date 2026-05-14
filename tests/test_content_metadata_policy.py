from pathlib import Path

import pytest

from src.router.adaptation.content_metadata_policy import (
    build_candidate_lookup,
    build_majority_rules,
    evaluate_metadata_policy,
    infer_global_baseline,
    resolve_global_baseline,
)


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "content_metadata_policy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_build_majority_rules_selects_group_majority():
    rows = [
        {
            "image_id": "A::1",
            "dataset": "A",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
        },
        {
            "image_id": "A::2",
            "dataset": "A",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
        },
        {
            "image_id": "A::3",
            "dataset": "A",
            "oracle_codec": "JXL",
            "oracle_config": "d=1.0",
        },
        {
            "image_id": "B::1",
            "dataset": "B",
            "oracle_codec": "HEVC",
            "oracle_config": "crf=15",
        },
    ]

    rules = build_majority_rules(rows, policy_key="dataset")

    assert rules["A"] == ("JPEG", "q=85")
    assert rules["B"] == ("HEVC", "crf=15")


def test_infer_global_baseline_accepts_unique_pair():
    rows = [
        {"global_codec": "HEVC", "global_config": "crf=15"},
        {"global_codec": "HEVC", "global_config": "crf=15"},
    ]

    assert infer_global_baseline(rows) == ("HEVC", "crf=15")


def test_infer_global_baseline_rejects_ambiguous_pairs_by_default():
    rows = [
        {
            "global_codec": "HEVC",
            "global_config": "crf=15",
        },
        {
            "global_codec": "HEVC",
            "global_config": "crf=15",
        },
        {
            "global_codec": "JPEG",
            "global_config": "q=85",
        },
    ]

    with pytest.raises(ValueError, match="Ambiguous global baseline"):
        infer_global_baseline(rows)


def test_infer_global_baseline_can_return_modal_pair_when_requested():
    rows = [
        {"global_codec": "HEVC", "global_config": "crf=15"},
        {"global_codec": "JPEG", "global_config": "q=85"},
        {"global_codec": "HEVC", "global_config": "crf=15"},
    ]

    assert infer_global_baseline(rows, require_unique=False) == ("HEVC", "crf=15")


def test_resolve_global_baseline_uses_explicit_pair_even_if_rows_are_ambiguous():
    rows = [
        {"global_codec": "HEVC", "global_config": "crf=15"},
        {"global_codec": "JPEG", "global_config": "q=85"},
    ]

    assert resolve_global_baseline(
        rows,
        global_baseline_codec="JXL",
        global_baseline_config="d=1.0",
    ) == ("JXL", "d=1.0")


def test_resolve_global_baseline_requires_complete_explicit_pair():
    rows = [
        {"global_codec": "HEVC", "global_config": "crf=15"},
    ]

    with pytest.raises(ValueError, match="must be provided together"):
        resolve_global_baseline(
            rows,
            global_baseline_codec="HEVC",
            global_baseline_config=None,
        )


def test_evaluate_metadata_policy_falls_back_when_group_choice_infeasible():
    benchmark = _tmp_path("fallback_benchmark.csv")

    benchmark.write_text(
        "\n".join(
            [
                "dataset,image,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,img1,JPEG,q=85,1.0,90.0,1.0,10.0",
                "A,img1,HEVC,crf=15,2.0,95.0,10.0,100.0",
                "A,img2,JPEG,q=85,1.0,60.0,1.0,10.0",
                "A,img2,HEVC,crf=15,2.0,95.0,10.0,100.0",
                "A,img3,JPEG,q=85,1.0,90.0,1.0,10.0",
                "A,img3,HEVC,crf=15,2.0,95.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    candidate_lookup = build_candidate_lookup(
        str(benchmark),
        available_codecs="JPEG,HEVC",
    )

    metadata_rows = [
        {
            "dataset": "A",
            "image": "img1",
            "image_id": "A::img1",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
            "oracle_cost": "0.6",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "0.2",
        },
        {
            "dataset": "A",
            "image": "img2",
            "image_id": "A::img2",
            "oracle_codec": "HEVC",
            "oracle_config": "crf=15",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "0.0",
        },
        {
            "dataset": "A",
            "image": "img3",
            "image_id": "A::img3",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
            "oracle_cost": "0.0",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "0.2",
        },
    ]

    evaluation = evaluate_metadata_policy(
        metadata_oracle_rows=metadata_rows,
        candidate_lookup=candidate_lookup,
        policy_key="dataset",
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
        evaluation_mode="in-sample",
    )

    img2 = [
        row for row in evaluation["decisions"]
        if row["image"] == "img2"
    ][0]

    assert img2["proposed_codec"] == "JPEG"
    assert img2["proposed_feasible"] is False
    assert img2["fallback_used"] is True
    assert img2["selected_codec"] == "HEVC"
    assert img2["selected_config"] == "crf=15"


def test_evaluate_metadata_policy_reports_regret_reduction():
    benchmark = _tmp_path("reduction_benchmark.csv")

    benchmark.write_text(
        "\n".join(
            [
                "dataset,image,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,img1,JPEG,q=85,1.0,95.0,1.0,10.0",
                "A,img1,HEVC,crf=15,2.0,95.0,10.0,100.0",
                "A,img2,JPEG,q=85,1.0,95.0,1.0,10.0",
                "A,img2,HEVC,crf=15,2.0,95.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    candidate_lookup = build_candidate_lookup(
        str(benchmark),
        available_codecs="JPEG,HEVC",
    )

    metadata_rows = [
        {
            "dataset": "A",
            "image": "img1",
            "image_id": "A::img1",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
            "oracle_cost": "0.6",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "1.0",
        },
        {
            "dataset": "A",
            "image": "img2",
            "image_id": "A::img2",
            "oracle_codec": "JPEG",
            "oracle_config": "q=85",
            "oracle_cost": "0.6",
            "global_codec": "HEVC",
            "global_config": "crf=15",
            "regret": "1.0",
        },
    ]

    evaluation = evaluate_metadata_policy(
        metadata_oracle_rows=metadata_rows,
        candidate_lookup=candidate_lookup,
        policy_key="dataset",
        quality_floor=80.0,
        global_baseline=("HEVC", "crf=15"),
        evaluation_mode="in-sample",
    )

    summary = {
        (row["section"], row["key"]): row["value"]
        for row in evaluation["summary"]
    }

    assert summary[("accuracy", "oracle_match_rate")] == 1.0
    assert summary[("regret", "mean")] == 0.0
    assert summary[("regret_reduction", "mean")] == 1.0
