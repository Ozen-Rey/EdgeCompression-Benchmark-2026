import json
from pathlib import Path

import pytest

from src.router.observability.normalization_consistency import (
    NormalizationAuditLoadError,
    compare_normalization_audits,
    load_previous_normalization_audit,
)
from src.router.rde_router import main


BASE_AUDIT = {
    "mode": "runtime",
    "scales_source": "computed_at_runtime",
    "computed_at_runtime": True,
    "scope": "runtime_global_before_codec_filtering",
    "comparability": "run_local",
    "quality_metric": "ssimulacra2",
    "quality_direction": "higher_is_better",
    "num_reference_points": 3,
    "rate_scale": "log10",
    "rate_min": 0.1,
    "rate_max": 1.0,
    "energy_scale": "log10",
    "energy_min": 1.0,
    "energy_max": 10.0,
    "quality_scale": "linear",
    "quality_min": 50.0,
    "quality_max": 90.0,
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _router_args(out_path: Path, *extra: str) -> list[str]:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    return [
        "--csv",
        str(fixture),
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
        "--available-codecs",
        "JPEG,JXL,HEVC",
        "--quality-target",
        "very-high",
        "--quality-floor",
        "90",
        *extra,
        "--out",
        str(out_path),
    ]


def _run_router(out_path: Path, *extra: str) -> dict:
    main(_router_args(out_path, *extra))
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_same_audit_is_comparable_without_warnings():
    result = compare_normalization_audits(BASE_AUDIT, dict(BASE_AUDIT))

    assert result["comparable"] is True
    assert result["warnings"] == []
    assert all(value is False for value in result["differences"].values())


def test_mode_change_is_not_comparable():
    previous = dict(BASE_AUDIT)
    current = dict(BASE_AUDIT, mode="global")

    result = compare_normalization_audits(current, previous)

    assert result["comparable"] is False
    assert result["differences"]["mode_changed"] is True
    assert any("mode_changed" in warning for warning in result["warnings"])


def test_quality_metric_change_is_not_comparable():
    previous = dict(BASE_AUDIT)
    current = dict(BASE_AUDIT, quality_metric="vmaf")

    result = compare_normalization_audits(current, previous)

    assert result["comparable"] is False
    assert result["differences"]["quality_metric_changed"] is True


def test_scales_source_change_with_global_mode_is_not_comparable():
    previous = dict(BASE_AUDIT, mode="global", scales_source="profile_a.json")
    current = dict(BASE_AUDIT, mode="global", scales_source="profile_b.json")

    result = compare_normalization_audits(current, previous)

    assert result["comparable"] is False
    assert result["differences"]["scales_source_changed"] is True
    assert any("scales_source_changed" in warning for warning in result["warnings"])


def test_numeric_range_change_beyond_tolerance_warns():
    previous = dict(BASE_AUDIT)
    current = dict(BASE_AUDIT, rate_max=1.1)

    result = compare_normalization_audits(current, previous)

    assert result["comparable"] is True
    assert result["differences"]["rate_range_changed"] is True
    assert any("rate_range_changed" in warning for warning in result["warnings"])


def test_loads_audit_from_router_report_decision_receipt(tmp_path: Path):
    path = tmp_path / "report.json"
    _write_json(path, {"decision_receipt": {"normalization_audit": BASE_AUDIT}})

    assert load_previous_normalization_audit(path) == BASE_AUDIT


def test_loads_audit_from_top_level_router_report(tmp_path: Path):
    path = tmp_path / "report.json"
    _write_json(path, {"normalization_audit": BASE_AUDIT})

    assert load_previous_normalization_audit(path) == BASE_AUDIT


def test_missing_previous_audit_is_controlled_warning(tmp_path: Path):
    path = tmp_path / "report.json"
    _write_json(path, {"decision_receipt": {}})

    previous = load_previous_normalization_audit(path)
    result = compare_normalization_audits(BASE_AUDIT, previous)

    assert previous is None
    assert result["comparable"] is False
    assert result["warnings"] == ["previous_normalization_audit_missing"]


def test_missing_previous_file_is_controlled_error(tmp_path: Path):
    path = tmp_path / "missing.json"

    with pytest.raises(NormalizationAuditLoadError) as exc:
        load_previous_normalization_audit(path)

    assert "does not exist" in str(exc.value)


def test_invalid_previous_json_is_controlled_error(tmp_path: Path):
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(NormalizationAuditLoadError) as exc:
        load_previous_normalization_audit(path)

    assert "not valid JSON" in str(exc.value)


def test_router_without_previous_receipt_reports_disabled_and_same_decision(tmp_path: Path):
    first = _run_router(tmp_path / "first.json")
    second = _run_router(tmp_path / "second.json")

    assert first["normalization_consistency"] == {"enabled": False}
    assert second["normalization_consistency"] == {"enabled": False}
    assert second["decision"]["selected"] == first["decision"]["selected"]


def test_router_with_compatible_previous_receipt_is_comparable(tmp_path: Path):
    baseline = _run_router(tmp_path / "baseline.json")
    compatible = tmp_path / "compatible_receipt.json"
    _write_json(
        compatible,
        {"normalization_audit": baseline["normalization_audit"]},
    )

    report = _run_router(
        tmp_path / "compatible.json",
        "--previous-decision-receipt",
        str(compatible),
    )

    consistency = report["normalization_consistency"]
    assert consistency["enabled"] is True
    assert consistency["previous_receipt_loaded"] is True
    assert consistency["comparable"] is True
    assert consistency["warnings"] == []


def test_router_with_incompatible_previous_receipt_does_not_change_decision(
    tmp_path: Path,
):
    baseline = _run_router(tmp_path / "baseline.json")
    previous_audit = dict(baseline["normalization_audit"], mode="global")
    incompatible = tmp_path / "incompatible_receipt.json"
    _write_json(incompatible, {"normalization_audit": previous_audit})

    report = _run_router(
        tmp_path / "incompatible.json",
        "--previous-decision-receipt",
        str(incompatible),
    )

    consistency = report["normalization_consistency"]
    assert consistency["enabled"] is True
    assert consistency["previous_receipt_loaded"] is True
    assert consistency["comparable"] is False
    assert consistency["differences"]["mode_changed"] is True
    assert any("mode_changed" in warning for warning in consistency["warnings"])
    assert report["decision"]["selected"] == baseline["decision"]["selected"]
