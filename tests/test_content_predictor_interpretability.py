"""Tests for ``src.router.analysis.content_predictor_interpretability``.

The module is sklearn-only (decision tree surrogate, logistic regression,
permutation/leave-one-feature-out attribution). Tests use a tiny synthetic
fixture: 8 rows across 2 datasets, with 4 JPEG + 3 JXL + 1 HEVC labels,
so the binary filter has 7 rows and HEVC exercises the minority case.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("sklearn")

from src.router.analysis import content_predictor_interpretability as cpi
from src.router.analysis.content_predictor_interpretability import (
    audit_class_balance,
    build_hevc_case_study,
    build_interpretation,
    detect_lodo_missing_classes,
    filter_binary_jpeg_jxl,
    load_unified_rows,
    run_feature_attribution,
    run_logistic_pairwise_interactions,
    run_surrogate_tree_sweep,
)
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    base = scratch_root() / "content_predictor_interpretability"
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _unified_fixture() -> List[Dict[str, Any]]:
    """4 JPEG (small), 3 JXL (large), 1 HEVC (across 2 datasets)."""
    return [
        # Dataset A — JPEG-heavy (small images)
        {
            "image_id": "A::img1", "dataset": "A", "image": "img1",
            "oracle_codec": "JPEG", "oracle_config": "q=85",
            "oracle_label": "JPEG|q=85",
            "megapixels": 0.30, "aspect_ratio": 1.0,
            "resolution_class": "small", "orientation_class": "squareish",
            "loio_predicted_label": "JPEG|q=85",
            "lodo_predicted_label": "JPEG|q=85",
        },
        {
            "image_id": "A::img2", "dataset": "A", "image": "img2",
            "oracle_codec": "JPEG", "oracle_config": "q=85",
            "oracle_label": "JPEG|q=85",
            "megapixels": 0.40, "aspect_ratio": 1.1,
            "resolution_class": "small", "orientation_class": "landscape",
            "loio_predicted_label": "JPEG|q=85",
            "lodo_predicted_label": "JPEG|q=85",
        },
        {
            "image_id": "A::img3", "dataset": "A", "image": "img3",
            "oracle_codec": "JPEG", "oracle_config": "q=85",
            "oracle_label": "JPEG|q=85",
            "megapixels": 0.45, "aspect_ratio": 0.9,
            "resolution_class": "small", "orientation_class": "portrait",
            "loio_predicted_label": "JPEG|q=85",
            "lodo_predicted_label": "JXL|d=1.0",
        },
        {
            "image_id": "A::img4", "dataset": "A", "image": "img4",
            "oracle_codec": "JPEG", "oracle_config": "q=85",
            "oracle_label": "JPEG|q=85",
            "megapixels": 0.35, "aspect_ratio": 1.2,
            "resolution_class": "small", "orientation_class": "landscape",
            "loio_predicted_label": "JPEG|q=85",
            "lodo_predicted_label": "JPEG|q=85",
        },
        # Dataset B — JXL-heavy (large images)
        {
            "image_id": "B::img5", "dataset": "B", "image": "img5",
            "oracle_codec": "JXL", "oracle_config": "d=1.0",
            "oracle_label": "JXL|d=1.0",
            "megapixels": 6.5, "aspect_ratio": 1.5,
            "resolution_class": "large", "orientation_class": "landscape",
            "loio_predicted_label": "JXL|d=1.0",
            "lodo_predicted_label": "JXL|d=1.0",
        },
        {
            "image_id": "B::img6", "dataset": "B", "image": "img6",
            "oracle_codec": "JXL", "oracle_config": "d=1.0",
            "oracle_label": "JXL|d=1.0",
            "megapixels": 7.2, "aspect_ratio": 1.4,
            "resolution_class": "large", "orientation_class": "landscape",
            "loio_predicted_label": "JXL|d=1.0",
            "lodo_predicted_label": "JPEG|q=85",
        },
        {
            "image_id": "B::img7", "dataset": "B", "image": "img7",
            "oracle_codec": "JXL", "oracle_config": "d=1.0",
            "oracle_label": "JXL|d=1.0",
            "megapixels": 5.8, "aspect_ratio": 1.6,
            "resolution_class": "large", "orientation_class": "landscape",
            "loio_predicted_label": "JXL|d=1.0",
            "lodo_predicted_label": "JXL|d=1.0",
        },
        # Dataset B — single HEVC case (minority class)
        {
            "image_id": "B::img8", "dataset": "B", "image": "img8",
            "oracle_codec": "HEVC", "oracle_config": "crf=15",
            "oracle_label": "HEVC|crf=15",
            "megapixels": 8.5, "aspect_ratio": 1.8,
            "resolution_class": "huge", "orientation_class": "landscape",
            "loio_predicted_label": "JXL|d=1.0",
            "lodo_predicted_label": "JXL|d=1.0",
        },
    ]


def test_module_is_importable():
    assert hasattr(cpi, "main")
    assert hasattr(cpi, "build_interpretation")
    assert hasattr(cpi, "run_surrogate_tree_sweep")


def test_class_balance_audit_aggregates_distribution():
    rows = _unified_fixture()
    audit = audit_class_balance(rows)

    assert audit["global_oracle_distribution"] == {
        "JPEG|q=85": 4,
        "JXL|d=1.0": 3,
        "HEVC|crf=15": 1,
    }
    assert audit["per_dataset_oracle_distribution"]["A"] == {"JPEG|q=85": 4}
    assert audit["per_dataset_oracle_distribution"]["B"] == {
        "JXL|d=1.0": 3,
        "HEVC|crf=15": 1,
    }
    assert audit["hevc_count"] == 1
    # HEVC and JXL both <=5 trigger minority warnings on this fixture.
    minority_warnings = [
        w for w in audit["warnings"]
        if w.startswith("minority_class_too_small_for_structural_claim")
    ]
    assert any("HEVC|crf=15" in w for w in minority_warnings)


def test_lodo_missing_classes_detection():
    rows = _unified_fixture()
    missing = detect_lodo_missing_classes(rows)

    # Dropping dataset A leaves no JPEG in training (only JXL + HEVC).
    a_entry = next(e for e in missing if e["left_out_dataset"] == "A")
    assert "JPEG|q=85" in a_entry["missing_classes"]

    # Dropping dataset B leaves no JXL and no HEVC in training (only JPEG).
    b_entry = next(e for e in missing if e["left_out_dataset"] == "B")
    assert set(b_entry["missing_classes"]) == {"JXL|d=1.0", "HEVC|crf=15"}


def test_binary_filter_excludes_hevc_but_keeps_case_study():
    rows = _unified_fixture()
    binary, hevc = filter_binary_jpeg_jxl(rows)

    assert len(binary) == 7
    assert all(r["oracle_label"] != "HEVC|crf=15" for r in binary)
    assert len(hevc) == 1

    case_study = build_hevc_case_study(hevc)
    assert case_study["num_samples"] == 1
    assert case_study["rows"][0]["image_id"] == "B::img8"
    assert "qualitative case study" in case_study["note"]


def test_surrogate_tree_sweep_produces_rows_per_depth():
    rows = _unified_fixture()
    binary, _ = filter_binary_jpeg_jxl(rows)
    sweep = run_surrogate_tree_sweep(binary, max_depths=[1, 2, 3, 4], seed=42)

    assert {row["max_depth"] for row in sweep["depth_rows"]} == {1, 2, 3, 4}
    for row in sweep["depth_rows"]:
        assert row["num_leaves"] >= 1
        assert 0.0 <= row["fidelity_to_knn"] <= 1.0
        assert 0.0 <= row["fidelity_to_oracle"] <= 1.0

    # On this fixture, megapixels alone separates JPEG/JXL perfectly, so even
    # a depth-1 surrogate should match the kNN target on every binary row.
    depth1 = next(r for r in sweep["depth_rows"] if r["max_depth"] == 1)
    assert depth1["fidelity_to_knn"] == pytest.approx(1.0)

    assert 1 in sweep["rules_by_depth"]
    rules_text = sweep["rules_by_depth"][1]
    # The depth-1 rule on this synthetic fixture splits on a single feature
    # that separates JPEG-small from JXL-large cleanly; we don't pin the
    # exact feature name, only that some recognisable column is used.
    assert any(
        column in rules_text
        for column in (
            "megapixels",
            "resolution_class",
            "aspect_ratio",
            "orientation_class",
        )
    )


def test_logistic_pairwise_interactions_returns_sorted_coefficients():
    rows = _unified_fixture()
    binary, _ = filter_binary_jpeg_jxl(rows)
    report = run_logistic_pairwise_interactions(binary, seed=42)

    assert report["coefficients"], "expected at least one coefficient row"
    abs_values = [c["absolute_coefficient"] for c in report["coefficients"]]
    assert abs_values == sorted(abs_values, reverse=True)
    assert report["model_score"] is not None
    # Interaction columns should appear in the expanded feature list.
    assert any(" * " in c["feature"] for c in report["coefficients"])


def test_feature_attribution_produces_sortable_deltas():
    rows = _unified_fixture()
    binary, _ = filter_binary_jpeg_jxl(rows)
    attribution = run_feature_attribution(binary, seed=42)

    rows_out = attribution["rows"]
    assert rows_out, "expected non-empty attribution rows"

    deltas = [r["delta"] for r in rows_out]
    assert deltas == sorted(deltas, reverse=True)
    ranks = [r["rank"] for r in rows_out]
    assert ranks == list(range(1, len(rows_out) + 1))

    methods = {r["method"] for r in rows_out}
    assert methods == {"leave_one_feature_out", "permutation"}


def test_build_interpretation_emits_prudent_strings():
    rows = _unified_fixture()
    binary, hevc = filter_binary_jpeg_jxl(rows)
    notes = build_interpretation(
        class_balance=audit_class_balance(rows),
        surrogate_sweep=run_surrogate_tree_sweep(
            binary, max_depths=[1, 2, 3, 4], seed=42
        ),
        logistic_report=run_logistic_pairwise_interactions(binary, seed=42),
        attribution_report=run_feature_attribution(binary, seed=42),
        hevc_case_study=build_hevc_case_study(hevc),
        lodo_missing=detect_lodo_missing_classes(rows),
    )

    text = "\n".join(notes)
    # HEVC minority warning is mandatory on this fixture.
    assert "HEVC oracle-optimal cases are too few" in text
    # LODO missing class warning is mandatory on this fixture.
    assert "LODO fold" in text
    # No statistical-significance claims, no "best possible"/"proves".
    forbidden = ["statistically significant", "best possible", "proves "]
    for word in forbidden:
        assert word.lower() not in text.lower()
    # Scope disclaimer must be present (the exact phrasing uses "do not"
    # because the subject is "These observations", which is plural).
    assert "establish universal generalization" in text


def test_load_unified_rows_from_csv_files():
    rows = _unified_fixture()

    oracle_csv = _tmp_path("unified_oracle.csv")
    metadata_csv = _tmp_path("unified_metadata.csv")
    decisions_csv = _tmp_path("unified_decisions.csv")

    _write_csv(
        oracle_csv,
        [
            {
                "image_id": r["image_id"],
                "dataset": r["dataset"],
                "image": r["image"],
                "oracle_codec": r["oracle_codec"],
                "oracle_config": r["oracle_config"],
                "regret": 0.0,
                "global_feasible": "True",
            }
            for r in rows
        ],
    )
    _write_csv(
        metadata_csv,
        [
            {
                "image_id": r["image_id"],
                "dataset": r["dataset"],
                "image": r["image"],
                "megapixels": r["megapixels"],
                "aspect_ratio": r["aspect_ratio"],
                "resolution_class": r["resolution_class"],
                "orientation_class": r["orientation_class"],
            }
            for r in rows
        ],
    )
    _write_csv(
        decisions_csv,
        [
            {
                "image_id": r["image_id"],
                "feature_set": "metadata_no_source",
                "k": 7,
                "evaluation_mode": "leave_one_image_out",
                "predicted_codec": r["loio_predicted_label"].split("|", 1)[0],
                "predicted_config": r["loio_predicted_label"].split("|", 1)[1],
            }
            for r in rows
        ],
    )

    loaded = load_unified_rows(
        oracle_by_image_path=str(oracle_csv),
        metadata_features_path=str(metadata_csv),
        classifier_decisions_path=str(decisions_csv),
        feature_set="metadata_no_source",
        k=7,
    )
    assert len(loaded["rows"]) == 8
    assert loaded["has_loio"] is True
    assert all(r["megapixels"] is not None for r in loaded["rows"])
    assert all(
        r["loio_predicted_label"] for r in loaded["rows"]
    ), "every row should carry an LOIO prediction after the join"


def test_cli_help_exits_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.router.analysis.content_predictor_interpretability",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "interpretability" in result.stdout.lower()


def test_cli_end_to_end_writes_all_outputs():
    rows = _unified_fixture()

    oracle_csv = _tmp_path("cli_oracle.csv")
    metadata_csv = _tmp_path("cli_metadata.csv")
    decisions_csv = _tmp_path("cli_decisions.csv")

    _write_csv(
        oracle_csv,
        [
            {
                "image_id": r["image_id"],
                "dataset": r["dataset"],
                "image": r["image"],
                "oracle_codec": r["oracle_codec"],
                "oracle_config": r["oracle_config"],
                "regret": 0.0,
                "global_feasible": "True",
            }
            for r in rows
        ],
    )
    _write_csv(
        metadata_csv,
        [
            {
                "image_id": r["image_id"],
                "dataset": r["dataset"],
                "image": r["image"],
                "megapixels": r["megapixels"],
                "aspect_ratio": r["aspect_ratio"],
                "resolution_class": r["resolution_class"],
                "orientation_class": r["orientation_class"],
            }
            for r in rows
        ],
    )
    _write_csv(
        decisions_csv,
        [
            {
                "image_id": r["image_id"],
                "feature_set": "metadata_no_source",
                "k": 7,
                "evaluation_mode": "leave_one_image_out",
                "predicted_codec": r["loio_predicted_label"].split("|", 1)[0],
                "predicted_config": r["loio_predicted_label"].split("|", 1)[1],
            }
            for r in rows
        ],
    )

    out_dir = _tmp_path("cli_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cpi.main(
        [
            "--oracle-by-image",
            str(oracle_csv),
            "--metadata-features",
            str(metadata_csv),
            "--classifier-decisions",
            str(decisions_csv),
            "--feature-set",
            "metadata_no_source",
            "--k",
            "7",
            "--out-dir",
            str(out_dir),
            "--tree-depths",
            "1,2,3,4",
            "--seed",
            "42",
        ]
    )

    for filename in [
        "content_predictor_class_balance.csv",
        "content_predictor_tree_surrogates.csv",
        "content_predictor_logistic_interactions.csv",
        "content_predictor_feature_attribution.csv",
        "content_predictor_interpretability.json",
        "content_predictor_tree_rules_depth_1.txt",
        "content_predictor_tree_rules_depth_4.txt",
    ]:
        assert (out_dir / filename).is_file(), f"missing output: {filename}"

    payload = json.loads(
        (out_dir / "content_predictor_interpretability.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["num_rows"] == 8
    assert payload["binary_jpeg_jxl"]["num_samples"] == 7
    assert payload["hevc_case_study"]["num_samples"] == 1
    assert payload["provenance"]["surrogate_target"] == "knn_loio"
    assert payload["provenance"]["shap_used"] is False
    assert isinstance(payload["interpretation"], list)
    assert payload["interpretation"], "interpretation list must be non-empty"
