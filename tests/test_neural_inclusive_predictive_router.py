"""Tests for ``src.router.analysis.neural_inclusive_predictive_router``.

The fixture has 6 images across 2 datasets (3 + 3) and 4 codecs (2
classical + 2 neural). Each image has measurements for every codec,
crafted so that under bandwidth-limited the neural pool dominates on
some images and under energy-limited the classical pool dominates on
every image. The dataset structure also makes one dataset
neural-favourable and the other classical-favourable, so source-aware
majority and kNN both have a real chance to discriminate.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.router.analysis import neural_inclusive_predictive_router as nipr
from src.router.analysis.neural_inclusive_oracle import (
    classify_codec_family,
    load_pool_rows,
    normalize_full_pool,
)
from src.router.analysis.neural_inclusive_predictive_router import (
    _loio_training_rows,
    _lodo_training_rows,
    bootstrap_paired_ci,
    build_interpretation,
    build_per_image_oracle_labels,
    evaluate_predictive_router,
    main,
    policy_knn_metadata,
    policy_robust_global,
    policy_source_aware_majority,
)
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    base = scratch_root() / "neural_inclusive_predictive_router"
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


def _make_row(
    *,
    dataset: str,
    image: str,
    codec: str,
    config: str,
    bpp: float,
    quality: float,
    energy: float,
    width: int = 1280,
    height: int = 1024,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "image": image,
        "codec": codec,
        "param": config,
        "bpp": bpp,
        "ssimulacra2": quality,
        "energy_per_image_j": energy,
        "width": width,
        "height": height,
    }


def _fixture_rows() -> List[Dict[str, Any]]:
    """6 images x 4 codecs.

    Dataset 'small' (3 images, small megapixels) — neural Balle dominates
    on bandwidth-limited; JXL dominates on energy-limited.

    Dataset 'large' (3 images, larger megapixels) — classical JXL
    dominates everywhere because the neural rate advantage is smaller
    on these images and the energy cost remains high.
    """
    rows: List[Dict[str, Any]] = []
    for idx in (1, 2, 3):
        rows.extend(
            [
                _make_row(
                    dataset="small",
                    image=f"sml{idx}",
                    codec="JPEG",
                    config="q=85",
                    bpp=1.2,
                    quality=72.0,
                    energy=0.10,
                    width=640,
                    height=512,
                ),
                _make_row(
                    dataset="small",
                    image=f"sml{idx}",
                    codec="JXL",
                    config="d=1.0",
                    bpp=0.9,
                    quality=75.0,
                    energy=0.05,
                    width=640,
                    height=512,
                ),
                _make_row(
                    dataset="small",
                    image=f"sml{idx}",
                    codec="JPEG_AI",
                    config="lambda=0.01",
                    bpp=0.35,
                    quality=74.0,
                    energy=50.0,
                    width=640,
                    height=512,
                ),
                _make_row(
                    dataset="small",
                    image=f"sml{idx}",
                    codec="Balle",
                    config="lambda=0.005",
                    bpp=0.20,
                    quality=72.0,
                    energy=120.0,
                    width=640,
                    height=512,
                ),
            ]
        )
    for idx in (1, 2, 3):
        rows.extend(
            [
                _make_row(
                    dataset="large",
                    image=f"lrg{idx}",
                    codec="JPEG",
                    config="q=85",
                    bpp=1.0,
                    quality=75.0,
                    energy=0.30,
                    width=3840,
                    height=2160,
                ),
                _make_row(
                    dataset="large",
                    image=f"lrg{idx}",
                    codec="JXL",
                    config="d=1.0",
                    bpp=0.7,
                    quality=78.0,
                    energy=0.20,
                    width=3840,
                    height=2160,
                ),
                _make_row(
                    dataset="large",
                    image=f"lrg{idx}",
                    codec="JPEG_AI",
                    config="lambda=0.01",
                    bpp=0.5,
                    quality=76.0,
                    energy=200.0,
                    width=3840,
                    height=2160,
                ),
                _make_row(
                    dataset="large",
                    image=f"lrg{idx}",
                    codec="Balle",
                    config="lambda=0.005",
                    bpp=0.40,
                    quality=74.0,
                    energy=300.0,
                    width=3840,
                    height=2160,
                ),
            ]
        )
    return rows


def _write_fixture_csv() -> Path:
    path = _tmp_path("predictive_router_fixture.csv")
    _write_csv(path, _fixture_rows())
    return path


def _load_normalised_fixture() -> List[Dict[str, Any]]:
    csv_path = _write_fixture_csv()
    rows = load_pool_rows(
        str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        image_id_col="dataset,image",
        dataset_col="dataset",
    )
    normalize_full_pool(rows)
    return rows


def test_module_is_importable():
    assert hasattr(nipr, "main")
    assert hasattr(nipr, "evaluate_predictive_router")


def test_loio_training_excludes_test_image_only():
    rows = _load_normalised_fixture()
    image_ids = sorted({r["image_id"] for r in rows})
    target = image_ids[0]
    training = _loio_training_rows(rows, test_image_id=target)
    training_images = {r["image_id"] for r in training}
    assert target not in training_images
    # All other images must still be present.
    assert training_images == set(image_ids) - {target}


def test_lodo_training_excludes_entire_target_dataset():
    rows = _load_normalised_fixture()
    target_dataset = "small"
    training = _lodo_training_rows(rows, test_dataset=target_dataset)
    training_datasets = {str(r.get("dataset")) for r in training}
    assert target_dataset not in training_datasets
    # The other dataset must still be present.
    assert "large" in training_datasets


def test_full_pool_oracle_labels_match_per_image_argmin():
    rows = _load_normalised_fixture()
    weights = {"w_R": 0.6, "w_E": 0.2, "w_D": 0.2}  # bandwidth-limited
    oracle = build_per_image_oracle_labels(
        rows,
        pool="full_pool",
        weights=weights,
        quality_floor=70.0,
    )

    # On the 'small' dataset the neural Balle row has the lowest bpp
    # (0.20) and meets the quality floor; under bandwidth-limited it
    # should beat JXL.
    sml1 = oracle["small::sml1"]
    assert sml1 is not None
    assert sml1["family"] == "neural"

    # On the 'large' dataset, the rate advantage of Balle (0.40) over
    # JXL (0.70) is smaller and the energy cost is much higher; the
    # classical JXL should remain oracle-optimal.
    lrg1 = oracle["large::lrg1"]
    assert lrg1 is not None
    assert lrg1["family"] == "classical"


def test_robust_global_baseline_does_not_pick_minority_codec():
    rows = _load_normalised_fixture()
    weights = {"w_R": 0.2, "w_E": 0.6, "w_D": 0.2}  # energy-limited
    pair = policy_robust_global(
        training_rows=rows,
        pool="full_pool",
        weights=weights,
        quality_floor=70.0,
    )
    # Under energy-limited, classical codecs (energy 0.05–0.30 J/img)
    # must dominate over neural codecs (50–300 J/img).
    assert pair is not None
    assert classify_codec_family(pair[0]) == "classical"


def test_source_aware_majority_uses_per_dataset_oracle():
    rows = _load_normalised_fixture()
    weights = {"w_R": 0.6, "w_E": 0.2, "w_D": 0.2}  # bandwidth-limited
    rules, fallback = policy_source_aware_majority(
        training_rows=rows,
        pool="full_pool",
        weights=weights,
        quality_floor=70.0,
    )
    # Small dataset prefers a neural codec; large dataset stays classical.
    assert classify_codec_family(rules["small"][0]) == "neural"
    assert classify_codec_family(rules["large"][0]) == "classical"
    # Fallback must be a real (codec, config) pair.
    assert fallback is not None


def test_knn_metadata_predicts_neural_when_training_supports_it():
    rows = _load_normalised_fixture()
    weights = {"w_R": 0.6, "w_E": 0.2, "w_D": 0.2}  # bandwidth-limited
    # Test image: a small image; training: every other image.
    target_image = "small::sml1"
    training_rows = _loio_training_rows(rows, test_image_id=target_image)

    metadata = nipr._derive_metadata_from_rde_rows(rows)
    predicted, confidence, fallback = policy_knn_metadata(
        training_rows=training_rows,
        metadata_by_image=metadata,
        pool="full_pool",
        weights=weights,
        quality_floor=70.0,
        test_meta=metadata[target_image],
        k=3,
    )
    assert predicted is not None
    assert classify_codec_family(predicted[0]) == "neural"
    assert confidence is not None
    assert 0.0 < confidence <= 1.0


def test_bootstrap_paired_ci_is_deterministic():
    a = bootstrap_paired_ci(
        policy_regrets=[0.01, 0.02, 0.05, 0.00, 0.03, 0.02],
        baseline_regrets=[0.10, 0.05, 0.20, 0.00, 0.15, 0.08],
        iterations=200,
        seed=99,
    )
    b = bootstrap_paired_ci(
        policy_regrets=[0.01, 0.02, 0.05, 0.00, 0.03, 0.02],
        baseline_regrets=[0.10, 0.05, 0.20, 0.00, 0.15, 0.08],
        iterations=200,
        seed=99,
    )
    assert a == b
    c = bootstrap_paired_ci(
        policy_regrets=[0.01, 0.02, 0.05, 0.00, 0.03, 0.02],
        baseline_regrets=[0.10, 0.05, 0.20, 0.00, 0.15, 0.08],
        iterations=200,
        seed=100,
    )
    assert a != c


def test_evaluate_predictive_router_returns_expected_structure():
    rows = _load_normalised_fixture()
    metadata = nipr._derive_metadata_from_rde_rows(rows)
    evaluation = evaluate_predictive_router(
        rows=rows,
        metadata_by_image=metadata,
        profile_names=["bandwidth-limited", "energy-limited"],
        quality_floors=[70.0],
        protocols=["loio", "lodo"],
        k=3,
        bootstrap_iterations=100,
        seed=7,
    )
    # 5 policies x 2 protocols x 2 profiles x 1 floor = 20 summary rows.
    assert len(evaluation["summaries"]) == 20

    # Every summary has neural_family precision/recall computed when oracle has neural.
    bandwidth_summaries = [
        s for s in evaluation["summaries"]
        if s["profile"] == "bandwidth-limited"
    ]
    full_pool_oracle = next(
        s for s in bandwidth_summaries
        if s["policy_name"] == "full_pool_oracle"
    )
    assert full_pool_oracle["mean_regret"] == pytest.approx(0.0)
    assert full_pool_oracle["family_match_rate"] == pytest.approx(1.0)


def test_classic_only_predictive_never_predicts_neural():
    rows = _load_normalised_fixture()
    metadata = nipr._derive_metadata_from_rde_rows(rows)
    evaluation = evaluate_predictive_router(
        rows=rows,
        metadata_by_image=metadata,
        profile_names=["bandwidth-limited"],
        quality_floors=[70.0],
        protocols=["loio"],
        k=3,
        bootstrap_iterations=50,
        seed=1,
    )
    classic = next(
        s for s in evaluation["summaries"]
        if s["policy_name"] == "classic_only_predictive_baseline"
    )
    assert classic["neural_selection_rate"] == pytest.approx(0.0)


def test_decision_rows_do_not_leak_test_rde_for_selection():
    """The policy's predicted_codec must not depend on the test image's
    own measured candidates: predictions for the same image_id under
    LOIO must come from training of *other* images only."""
    rows = _load_normalised_fixture()
    metadata = nipr._derive_metadata_from_rde_rows(rows)
    evaluation = evaluate_predictive_router(
        rows=rows,
        metadata_by_image=metadata,
        profile_names=["bandwidth-limited"],
        quality_floors=[70.0],
        protocols=["loio"],
        k=3,
        bootstrap_iterations=50,
        seed=1,
    )
    # For the kNN policy: if we remove a single image (LOIO), the
    # predicted pair for that image must equal the kNN prediction
    # using only the remaining images as training. Re-compute it
    # explicitly here and compare.
    target_image = "small::sml1"
    target_meta = metadata[target_image]
    training_rows = _loio_training_rows(rows, test_image_id=target_image)
    weights = {"w_R": 0.6, "w_E": 0.2, "w_D": 0.2}
    predicted, _, _ = policy_knn_metadata(
        training_rows=training_rows,
        metadata_by_image=metadata,
        pool="full_pool",
        weights=weights,
        quality_floor=70.0,
        test_meta=target_meta,
        k=3,
    )
    decision = next(
        d for d in evaluation["decisions"]
        if d["image_id"] == target_image
        and d["policy_name"] == "knn_metadata_full_pool"
        and d["protocol"] == "loio"
    )
    assert decision["predicted_codec"] == predicted[0]
    assert decision["predicted_config"] == predicted[1]


def test_build_interpretation_uses_prudent_wording():
    rows = _load_normalised_fixture()
    metadata = nipr._derive_metadata_from_rde_rows(rows)
    evaluation = evaluate_predictive_router(
        rows=rows,
        metadata_by_image=metadata,
        profile_names=["bandwidth-limited"],
        quality_floors=[70.0],
        protocols=["loio", "lodo"],
        k=3,
        bootstrap_iterations=50,
        seed=1,
    )
    notes = build_interpretation(
        summaries=evaluation["summaries"],
        profiles=["bandwidth-limited"],
        quality_floors=[70.0],
        protocols=["loio", "lodo"],
    )
    text = "\n".join(notes)
    for forbidden in (
        "neural codecs are better",
        "classical codecs are obsolete",
        "best possible",
        "proves",
    ):
        assert forbidden.lower() not in text.lower()
    assert (
        "does not establish universal generalization" in text
        or "do not establish universal generalization" in text
        or "universal neural/classical dominance" in text
    )
    # LODO disclaimer must be present when lodo is among protocols.
    assert "LODO is the stricter protocol" in text


def test_cli_help_exits_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.router.analysis.neural_inclusive_predictive_router",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "predictive" in result.stdout.lower()


def test_cli_end_to_end_writes_all_outputs_under_tmp_path():
    csv_path = _write_fixture_csv()
    out_dir = _tmp_path("cli_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    main(
        [
            "--rde-csv",
            str(csv_path),
            "--image-id-col",
            "dataset,image",
            "--dataset-col",
            "dataset",
            "--quality-floors",
            "70",
            "--profiles",
            "bandwidth-limited,energy-limited",
            "--protocols",
            "loio,lodo",
            "--k",
            "3",
            "--bootstrap-iterations",
            "50",
            "--seed",
            "7",
            "--out-dir",
            str(out_dir),
        ]
    )

    for filename in [
        "neural_inclusive_predictive_router_decisions.csv",
        "neural_inclusive_predictive_router_summary.csv",
        "neural_inclusive_predictive_router_report.json",
    ]:
        assert (out_dir / filename).is_file(), f"missing output: {filename}"

    payload = json.loads(
        (out_dir / "neural_inclusive_predictive_router_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["num_images"] == 6
    assert payload["provenance"]["policy_does_not_see_test_image_rde"] is True
    assert (
        payload["provenance"]["test_image_rde_used_only_for_realisation_and_oracle"]
        is True
    )
    assert payload["normalization"]["scope"] == "full_pool_global"
    assert isinstance(payload["interpretation"], list)
    assert payload["interpretation"]
    # Internal underscore-prefixed payload must not be serialised.
    assert "_decisions" not in payload

    # Decisions CSV must carry every required column.
    with (out_dir / "neural_inclusive_predictive_router_decisions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        required = {
            "image_id",
            "dataset",
            "protocol",
            "profile",
            "quality_floor",
            "policy_name",
            "predicted_codec",
            "predicted_config",
            "predicted_family",
            "oracle_codec",
            "oracle_config",
            "oracle_family",
            "selected_J_on_test",
            "oracle_J_on_test",
            "regret",
            "family_match",
            "exact_match",
            "confidence",
            "fallback_used",
            "fallback_reason",
            "quality_violation",
            "provenance",
        }
        assert required <= set(reader.fieldnames or [])
