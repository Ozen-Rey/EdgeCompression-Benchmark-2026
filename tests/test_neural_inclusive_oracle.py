"""Tests for ``src.router.analysis.neural_inclusive_oracle``.

Synthetic fixture: 2 images × 4 codecs (2 classical + 2 neural) × 1
config each, with rate/quality/energy crafted so that:

- under ``bandwidth-limited``, the neural codec wins on at least one
  image (lowest bpp dominates the weighted cost),
- under ``energy-limited``, the classical codec wins on every image
  (lowest energy dominates the weighted cost),
- under ``quality-first``, the highest-quality codec is selected,
- the classic-only pool incurs a positive regret vs the full pool
  under ``bandwidth-limited``.

These invariants drive every assertion in the file.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.router.analysis import neural_inclusive_oracle as nio
from src.router.analysis.neural_inclusive_oracle import (
    build_interpretation,
    classify_codec_family,
    compute_pool_comparison,
    evaluate_pool,
    load_pool_rows,
    normalize_full_pool,
)
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    base = scratch_root() / "neural_inclusive_oracle"
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


def _fixture_rows() -> List[Dict[str, Any]]:
    """2 images x 4 codecs (JPEG, JXL, JPEG_AI, Balle).

    For each image:
    - JPEG: high bpp, high energy(low), high quality(moderate)
    - JXL:  moderate bpp, low energy(lower), high quality
    - JPEG_AI: low bpp, high energy, high quality
    - Balle:   very low bpp, very high energy, high quality
    """
    rows: List[Dict[str, Any]] = []
    for image_idx in (1, 2):
        rows.extend(
            [
                # classical
                {
                    "image_id": f"img{image_idx}",
                    "dataset": "synthetic",
                    "codec": "JPEG",
                    "param": "q=85",
                    "bpp": 1.2,
                    "ssimulacra2": 70.0,
                    "energy_per_image_j": 0.10,
                },
                {
                    "image_id": f"img{image_idx}",
                    "dataset": "synthetic",
                    "codec": "JXL",
                    "param": "d=1.0",
                    "bpp": 0.9,
                    "ssimulacra2": 75.0,
                    "energy_per_image_j": 0.05,
                },
                # neural
                {
                    "image_id": f"img{image_idx}",
                    "dataset": "synthetic",
                    "codec": "JPEG_AI",
                    "param": "lambda=0.01",
                    "bpp": 0.30,
                    "ssimulacra2": 73.0,
                    "energy_per_image_j": 50.0,
                },
                {
                    "image_id": f"img{image_idx}",
                    "dataset": "synthetic",
                    "codec": "Balle",
                    "param": "lambda=0.005",
                    "bpp": 0.20,
                    "ssimulacra2": 72.0,
                    "energy_per_image_j": 120.0,
                },
            ]
        )
    return rows


def _write_fixture_csv() -> Path:
    path = _tmp_path("fixture_image_rde.csv")
    _write_csv(path, _fixture_rows())
    return path


def test_module_is_importable():
    assert hasattr(nio, "main")
    assert hasattr(nio, "build_interpretation")
    assert hasattr(nio, "compute_pool_comparison")


def test_classify_codec_family_handles_known_spellings():
    assert classify_codec_family("JPEG") == "classical"
    assert classify_codec_family("JXL") == "classical"
    assert classify_codec_family("HEVC") == "classical"
    assert classify_codec_family("JPEG_AI") == "neural"
    assert classify_codec_family("JPEGAI") == "neural"
    # Accent variants of Ballé must classify as neural.
    assert classify_codec_family("Balle") == "neural"
    assert classify_codec_family("Ballé") == "neural"
    assert classify_codec_family("Cheng") == "neural"
    assert classify_codec_family("ELIC") == "neural"
    assert classify_codec_family("TCM") == "neural"
    assert classify_codec_family("DCAE") == "neural"
    # Unknown codec must not be coerced.
    assert classify_codec_family("FooCodec") == "unknown"


def test_load_pool_rows_uses_global_group_when_image_id_absent():
    path = _tmp_path("load_no_image_id.csv")
    _write_csv(
        path,
        [
            {
                "codec": "JPEG",
                "param": "q=85",
                "bpp": 1.0,
                "ssimulacra2": 70.0,
                "energy_per_image_j": 0.1,
            },
            {
                "codec": "JXL",
                "param": "d=1.0",
                "bpp": 0.8,
                "ssimulacra2": 75.0,
                "energy_per_image_j": 0.05,
            },
        ],
    )
    rows = load_pool_rows(
        str(path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        image_id_col="image_id",
    )
    assert {r["image_id"] for r in rows} == {"__global__"}


def test_load_pool_rows_supports_energy_join_two_files():
    metrics_path = _tmp_path("two_file_metrics.csv")
    energy_path = _tmp_path("two_file_energy.csv")

    _write_csv(
        metrics_path,
        [
            {
                "image_id": "g1",
                "codec": "JPEG",
                "param": "q=85",
                "bpp": 1.0,
                "ssimulacra2": 75.0,
            },
            {
                "image_id": "g1",
                "codec": "JPEG_AI",
                "param": "lambda=0.01",
                "bpp": 0.3,
                "ssimulacra2": 73.0,
            },
        ],
    )
    # Two rows per (codec, param) — the loader must average them.
    _write_csv(
        energy_path,
        [
            {"codec": "JPEG", "param": "q=85", "energy_per_image_j": 0.10},
            {"codec": "JPEG", "param": "q=85", "energy_per_image_j": 0.20},
            {"codec": "JPEG_AI", "param": "lambda=0.01",
             "energy_per_image_j": 40.0},
            {"codec": "JPEG_AI", "param": "lambda=0.01",
             "energy_per_image_j": 60.0},
        ],
    )

    rows = load_pool_rows(
        str(metrics_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        image_id_col="image_id",
        energy_csv_path=str(energy_path),
    )

    by_codec = {r["codec"]: r for r in rows}
    assert by_codec["JPEG"]["energy"] == pytest.approx(0.15)
    assert by_codec["JPEG_AI"]["energy"] == pytest.approx(50.0)


def test_normalization_is_global_not_per_pool():
    rows = _fixture_rows()
    rows = [dict(r) for r in rows]
    # Mutate keys to match what load_pool_rows would produce.
    parsed = []
    for r in rows:
        parsed.append(
            {
                "image_id": r["image_id"],
                "dataset": r["dataset"],
                "codec": r["codec"],
                "config": r["param"],
                "codec_family": classify_codec_family(r["codec"]),
                "rate": r["bpp"],
                "quality": r["ssimulacra2"],
                "energy": r["energy_per_image_j"],
            }
        )

    stats = normalize_full_pool(parsed)
    # The energy max comes from the neural Balle row at 120.0 J/img.
    # If normalization were computed per pool, the classical pool's
    # energy_max would be 0.10 J/img and the JPEG row would map to 1.0.
    # Global normalization must instead place 0.10 J/img near the
    # *bottom* of the global range.
    jpeg = next(r for r in parsed if r["codec"] == "JPEG")
    balle = next(r for r in parsed if r["codec"] == "Balle")
    assert jpeg["norm_energy"] < 0.1
    assert balle["norm_energy"] == pytest.approx(1.0)
    assert stats["energy_log10_max"] > stats["energy_log10_min"]


def test_oracle_selects_neural_under_bandwidth_limited():
    csv_path = _write_fixture_csv()
    rows = load_pool_rows(
        str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        image_id_col="image_id",
    )
    normalize_full_pool(rows)

    # Bandwidth-limited: w_R is largest. Balle has the lowest bpp by far.
    weights = {"w_R": 0.6, "w_E": 0.2, "w_D": 0.2}
    selections = evaluate_pool(
        rows,
        pool="full_pool",
        profile_name="bandwidth-limited",
        weights=weights,
        quality_floor=70.0,
    )
    families = {s["selected_family"] for s in selections if s["has_safe_selection"]}
    assert "neural" in families


def test_oracle_selects_classical_under_energy_limited():
    csv_path = _write_fixture_csv()
    rows = load_pool_rows(
        str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        image_id_col="image_id",
    )
    normalize_full_pool(rows)

    # Energy-limited: w_E is largest. Neural codecs cost 50-120 J vs
    # 0.05-0.1 J for classical: the classical pool must dominate.
    weights = {"w_R": 0.2, "w_E": 0.6, "w_D": 0.2}
    selections = evaluate_pool(
        rows,
        pool="full_pool",
        profile_name="energy-limited",
        weights=weights,
        quality_floor=70.0,
    )
    families = {s["selected_family"] for s in selections if s["has_safe_selection"]}
    assert families == {"classical"}


def test_pool_comparison_reports_regret_when_neural_wins():
    csv_path = _write_fixture_csv()
    rows = load_pool_rows(
        str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        image_id_col="image_id",
    )
    normalize_full_pool(rows)

    weights = {"w_R": 0.6, "w_E": 0.2, "w_D": 0.2}
    classic_sels = evaluate_pool(
        rows,
        pool="classic_pool",
        profile_name="bandwidth-limited",
        weights=weights,
        quality_floor=70.0,
    )
    full_sels = evaluate_pool(
        rows,
        pool="full_pool",
        profile_name="bandwidth-limited",
        weights=weights,
        quality_floor=70.0,
    )
    cmp = compute_pool_comparison(
        classic_sels,
        full_sels,
        profile_name="bandwidth-limited",
        quality_floor=70.0,
    )
    # Neural wins under this profile, so full-pool oracle has lower J_RDE.
    assert cmp["mean_regret_classic_vs_full"] > 0
    assert cmp["neural_selection_count_in_full"] >= 1
    assert cmp["mean_rate_gain_when_neural_selected"] is not None
    assert cmp["mean_rate_gain_when_neural_selected"] > 0
    # Energy penalty must be positive: neural costs more J/img than classical.
    assert cmp["mean_energy_penalty_when_neural_selected"] > 0


def test_build_interpretation_uses_prudent_wording():
    csv_path = _write_fixture_csv()
    rows = load_pool_rows(
        str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        image_id_col="image_id",
    )
    normalize_full_pool(rows)

    summaries = []
    comparisons = []
    for profile_name, weights in [
        ("bandwidth-limited", {"w_R": 0.6, "w_E": 0.2, "w_D": 0.2}),
        ("energy-limited", {"w_R": 0.2, "w_E": 0.6, "w_D": 0.2}),
    ]:
        for floor in (70.0,):
            classic_sels = evaluate_pool(
                rows,
                pool="classic_pool",
                profile_name=profile_name,
                weights=weights,
                quality_floor=floor,
            )
            full_sels = evaluate_pool(
                rows,
                pool="full_pool",
                profile_name=profile_name,
                weights=weights,
                quality_floor=floor,
            )
            summaries.append(
                nio.build_pool_summary(
                    classic_sels,
                    pool="classic_pool",
                    profile_name=profile_name,
                    quality_floor=floor,
                )
            )
            summaries.append(
                nio.build_pool_summary(
                    full_sels,
                    pool="full_pool",
                    profile_name=profile_name,
                    quality_floor=floor,
                )
            )
            comparisons.append(
                compute_pool_comparison(
                    classic_sels,
                    full_sels,
                    profile_name=profile_name,
                    quality_floor=floor,
                )
            )

    notes = build_interpretation(
        summaries=summaries,
        comparisons=comparisons,
        profiles=["bandwidth-limited", "energy-limited"],
        quality_floors=[70.0],
    )
    text = "\n".join(notes)

    forbidden = [
        "neural codecs are better",
        "classical codecs are obsolete",
        "best possible",
        "proves",
    ]
    for word in forbidden:
        assert word.lower() not in text.lower(), f"Forbidden phrase: {word}"

    # At least one prudent marker must be present.
    assert any(
        marker in text
        for marker in (
            "within this benchmark",
            "suggests",
            "is consistent with",
        )
    )
    assert (
        "does not establish universal generalization" in text
        or "does not imply universal dominance" in text
    )


def test_cli_help_exits_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.router.analysis.neural_inclusive_oracle",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "neural" in result.stdout.lower()


def test_cli_end_to_end_writes_all_outputs_under_tmp_path():
    csv_path = _write_fixture_csv()
    out_dir = _tmp_path("cli_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    nio.main(
        [
            "--csv",
            str(csv_path),
            "--image-id-col",
            "image_id",
            "--quality-col",
            "ssimulacra2",
            "--rate-col",
            "bpp",
            "--energy-col",
            "energy_per_image_j",
            "--quality-floors",
            "70",
            "--profiles",
            "bandwidth-limited,energy-limited",
            "--out-dir",
            str(out_dir),
        ]
    )

    for filename in [
        "neural_inclusive_oracle_summary.csv",
        "neural_inclusive_oracle_by_image.csv",
        "neural_inclusive_pool_comparison.csv",
        "neural_inclusive_per_group_labels.csv",
        "neural_inclusive_oracle_report.json",
    ]:
        assert (out_dir / filename).is_file(), f"missing output: {filename}"

    payload = json.loads(
        (out_dir / "neural_inclusive_oracle_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["num_rows_loaded"] == 8
    assert payload["normalization"]["scope"] == "full_pool_global"
    assert payload["provenance"]["uses_router_profiles"] is True
    # Internal underscore-prefixed selections payload must not be serialised.
    assert "_selections_by_pool" not in payload
    assert "_per_group_labels" not in payload
    # Interpretation must be present and non-empty.
    assert isinstance(payload["interpretation"], list)
    assert payload["interpretation"]
