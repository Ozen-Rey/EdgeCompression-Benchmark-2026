import json
from pathlib import Path

import pytest

from src.router.core.rde_database import (
    CSV_ROW_DIAGNOSTICS_EXAMPLE_LIMIT,
    load_rde_points,
    load_rde_points_with_diagnostics,
)
from src.router.rde_router import main
from tests.conftest import scratch_root


_VALID_HEADER = "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms"


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "csv_row_diagnostics"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _real_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "image_rde_real_small.csv"


def _common_csv_args(csv_path: Path) -> list[str]:
    return [
        "--csv",
        str(csv_path),
        "--codec-col",
        "codec",
        "--config-col",
        "param",
        "--rate-col",
        "bpp",
        "--quality-col",
        "ssimulacra2",
        "--energy-col",
        "energy_per_image_j",
        "--time-col",
        "time_ms",
        "--quality-target",
        "high",
        "--quality-floor",
        "70",
    ]


def _run_router(args: list[str], out_name: str) -> dict:
    out_path = _tmp_path(out_name)
    main([*args, "--out", str(out_path)])
    return json.loads(out_path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join([_VALID_HEADER, *rows]) + "\n", encoding="utf-8")


def test_loader_clean_csv_has_zero_dropped_rows():
    points, diagnostics = load_rde_points_with_diagnostics(
        csv_path=str(_real_fixture()),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        time_col="time_ms",
    )

    assert len(points) > 0
    assert diagnostics["enabled"] is True
    assert diagnostics["dropped_rows"] == 0
    assert diagnostics["reasons"] == {}
    assert diagnostics["examples"] == []


def test_loader_drops_row_with_invalid_rate_and_records_diagnostics():
    csv_path = _tmp_path("invalid_rate.csv")
    _write_csv(
        csv_path,
        [
            "JPEG,q=85,1.6,81.0,0.1,6.3",
            "JXL,d=1.0,not_a_number,85.0,2.5,134.0",
            "HEVC,crf=15,0.2,40.0,0.1,1.0",
        ],
    )

    points, diagnostics = load_rde_points_with_diagnostics(
        csv_path=str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        time_col="time_ms",
    )

    assert {p.codec for p in points} == {"JPEG", "HEVC"}
    assert diagnostics["dropped_rows"] == 1
    assert diagnostics["reasons"] == {"invalid_rate": 1}

    [example] = diagnostics["examples"]
    assert example["reason"] == "invalid_rate"
    assert example["row_index"] == 1
    assert example["csv_line_number"] == 3
    assert example["raw_values"]["codec"] == "JXL"
    assert example["raw_values"]["bpp"] == "not_a_number"
    assert "detail" in example


def test_loader_drops_row_with_missing_codec():
    csv_path = _tmp_path("missing_codec.csv")
    _write_csv(
        csv_path,
        [
            "JPEG,q=85,1.0,90.0,0.1,5.0",
            ",q=60,0.9,67.0,0.08,4.9",
            "JXL,d=1.0,1.37,85.0,2.55,134.0",
        ],
    )

    points, diagnostics = load_rde_points_with_diagnostics(
        csv_path=str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        time_col="time_ms",
    )

    assert {p.codec for p in points} == {"JPEG", "JXL"}
    assert diagnostics["reasons"] == {"missing_codec": 1}
    [example] = diagnostics["examples"]
    assert example["reason"] == "missing_codec"
    assert example["row_index"] == 1


def test_loader_aggregates_reasons_and_caps_examples():
    csv_path = _tmp_path("many_dropped.csv")
    bad_rows = [f"JPEG,q={i},notafloat,80.0,0.1,5.0" for i in range(25)]
    rows = ["JPEG,q=85,1.6,81.0,0.1,6.3", *bad_rows]
    _write_csv(csv_path, rows)

    points, diagnostics = load_rde_points_with_diagnostics(
        csv_path=str(csv_path),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        time_col="time_ms",
    )

    assert len(points) == 1
    assert diagnostics["dropped_rows"] == 25
    assert diagnostics["reasons"] == {"invalid_rate": 25}
    assert len(diagnostics["examples"]) == CSV_ROW_DIAGNOSTICS_EXAMPLE_LIMIT


def test_loader_raises_when_all_rows_invalid():
    csv_path = _tmp_path("all_invalid.csv")
    _write_csv(
        csv_path,
        [
            "JPEG,q=85,notafloat,90.0,0.1,5.0",
            ",q=60,0.9,67.0,0.08,4.9",
        ],
    )

    with pytest.raises(ValueError, match="dropped_rows=2"):
        load_rde_points_with_diagnostics(
            csv_path=str(csv_path),
            codec_col="codec",
            config_col="param",
            rate_col="bpp",
            quality_col="ssimulacra2",
            energy_col="energy_per_image_j",
            time_col="time_ms",
        )


def test_load_rde_points_wrapper_returns_only_points_list():
    points = load_rde_points(
        csv_path=str(_real_fixture()),
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        time_col="time_ms",
    )

    assert isinstance(points, list)
    assert all(hasattr(p, "codec") for p in points)


def test_router_report_clean_csv_has_diagnostics_block_with_zero_drops():
    report = _run_router(_common_csv_args(_real_fixture()), "clean_report.json")

    block = report["csv_row_diagnostics"]
    assert block["enabled"] is True
    assert block["dropped_rows"] == 0
    assert block["reasons"] == {}
    assert block["examples"] == []


def test_router_report_includes_diagnostics_for_dropped_rows():
    csv_path = _tmp_path("router_drops.csv")
    _write_csv(
        csv_path,
        [
            "JPEG,q=85,1.6,81.0,0.1,6.3",
            "JXL,d=1.0,not_a_number,85.0,2.5,134.0",
            "HEVC,crf=15,0.2,80.0,0.1,1.0",
            ",q=60,0.9,67.0,0.08,4.9",
        ],
    )

    report = _run_router(_common_csv_args(csv_path), "router_drops_report.json")

    block = report["csv_row_diagnostics"]
    assert block["enabled"] is True
    assert block["dropped_rows"] == 2
    assert block["reasons"] == {"invalid_rate": 1, "missing_codec": 1}
    assert len(block["examples"]) == 2
    reasons_in_examples = {ex["reason"] for ex in block["examples"]}
    assert reasons_in_examples == {"invalid_rate", "missing_codec"}


def test_router_report_csv_diagnostics_does_not_change_selection_for_real_fixture():
    baseline = _run_router(_common_csv_args(_real_fixture()), "baseline_selection.json")
    selected = baseline["decision"]["selected"]

    csv_path = _tmp_path("real_plus_bad_row.csv")
    fixture_rows = _real_fixture().read_text(encoding="utf-8").strip().splitlines()
    assert fixture_rows[0] == _VALID_HEADER
    rows_with_drop = [*fixture_rows[1:], "JPEG,q=85,not_a_number,80.0,0.1,5.0"]
    _write_csv(csv_path, rows_with_drop)

    augmented = _run_router(_common_csv_args(csv_path), "augmented_selection.json")
    aug_selected = augmented["decision"]["selected"]

    assert aug_selected["codec"] == selected["codec"]
    assert aug_selected["config"] == selected["config"]
    assert aug_selected["cost"] == pytest.approx(selected["cost"])
    assert augmented["csv_row_diagnostics"]["dropped_rows"] == 1
