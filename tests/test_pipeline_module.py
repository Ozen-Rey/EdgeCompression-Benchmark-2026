"""Smoke checks for the router pipeline module.

The module hosts side-effect-free helpers extracted from rde_router.py
plus the run_router orchestration entry point. These tests cover its
import surface, the small pure helpers, and a minimal end-to-end wiring
exercise of run_router against the real fixture.
"""

import json
from pathlib import Path

import pytest

from src.router import pipeline
from src.router.cli import build_router_arg_parser
from src.router.context import RouterContext
from src.router.core.rde_database import RDEPoint
from src.router.core.router_config import expand_argv_with_config
from src.router.pipeline import (
    apply_system_aware_policy,
    filter_points_by_codec_availability,
    is_neural_codec,
    normalize_token,
    parse_codec_list,
    run_router,
)


def test_pipeline_module_exports_extracted_helpers():
    for name in (
        "annotate_points_with_calibration_provenance",
        "apply_system_aware_policy",
        "build_weights_for_profile",
        "filter_points_by_codec_availability",
        "is_neural_codec",
        "normalize_token",
        "parse_codec_list",
        "run_router",
        "summary_row_from_report",
        "topk_rows_from_report",
    ):
        assert callable(getattr(pipeline, name)), f"missing pipeline helper: {name}"


def test_parse_codec_list_returns_none_for_missing_value():
    assert parse_codec_list(None) is None
    assert parse_codec_list("") is None
    assert parse_codec_list("   ") is None


def test_parse_codec_list_normalizes_tokens_and_trims_whitespace():
    parsed = parse_codec_list(" JPEG, jxl ,Hevc")

    assert parsed == {"jpeg", "jxl", "hevc"}


def test_normalize_token_strips_accents_and_punctuation():
    assert normalize_token("JPEG") == "jpeg"
    assert normalize_token(" jxl ") == "jxl"
    assert normalize_token("Ballé") == "balle"


def test_is_neural_codec_recognizes_known_neural_codec():
    assert is_neural_codec("JPEG_AI") is True
    assert is_neural_codec("jpegai") is True
    assert is_neural_codec("JPEG") is False
    assert is_neural_codec("JXL") is False


def _make_point(codec: str, config: str = "q=50") -> RDEPoint:
    return RDEPoint(
        codec=codec,
        config=config,
        rate=1.0,
        quality=80.0,
        energy=1.0,
        raw={},
    )


def test_filter_points_by_codec_availability_keeps_only_allowed_codecs():
    points = [
        _make_point("JPEG"),
        _make_point("JXL"),
        _make_point("HEVC"),
        _make_point("JPEG_AI"),
    ]

    filtered, report = filter_points_by_codec_availability(
        points=points,
        available_codecs={"jpeg", "jxl"},
        exclude_codecs=None,
        exclude_neural=False,
    )

    assert [p.codec for p in filtered] == ["JPEG", "JXL"]
    assert report["num_before_codec_filtering"] == 4
    assert report["num_after_codec_filtering"] == 2
    assert report["excluded_by_available_codecs"] == 2
    assert report["excluded_by_exclude_codecs"] == 0
    assert report["excluded_by_exclude_neural"] == 0


def test_filter_points_by_codec_availability_drops_excluded_and_neural():
    points = [
        _make_point("JPEG"),
        _make_point("JXL"),
        _make_point("JPEG_AI"),
    ]

    filtered, report = filter_points_by_codec_availability(
        points=points,
        available_codecs=None,
        exclude_codecs={"jxl"},
        exclude_neural=True,
    )

    assert [p.codec for p in filtered] == ["JPEG"]
    assert report["excluded_by_exclude_codecs"] == 1
    assert report["excluded_by_exclude_neural"] == 1


def test_filter_points_by_codec_availability_raises_on_empty_pool():
    points = [_make_point("JPEG_AI")]

    with pytest.raises(ValueError, match="Pool vuoto dopo i filtri codec"):
        filter_points_by_codec_availability(
            points=points,
            available_codecs=None,
            exclude_codecs=None,
            exclude_neural=True,
        )


def _state(cuda_available: bool) -> dict:
    return {"cuda": {"available": cuda_available}}


def test_apply_system_aware_policy_disabled_passes_request_through():
    effective, report = apply_system_aware_policy(
        system_state=_state(cuda_available=False),
        enabled=False,
        simulate_no_cuda=False,
        exclude_neural_requested=False,
        capability_aware_enabled=False,
    )

    assert effective is False
    assert report["enabled"] is False
    assert report["cuda_available"] is False
    assert report["effective_exclude_neural"] is False
    assert report["rules_applied"] == []


def test_apply_system_aware_policy_keeps_neural_when_cuda_available():
    effective, report = apply_system_aware_policy(
        system_state=_state(cuda_available=True),
        enabled=True,
        simulate_no_cuda=False,
        exclude_neural_requested=False,
        capability_aware_enabled=False,
    )

    assert effective is False
    assert report["cuda_available"] is True
    assert report["rules_applied"] == ["cuda_available_keep_neural_candidates"]


def test_apply_system_aware_policy_excludes_neural_when_cuda_missing():
    effective, report = apply_system_aware_policy(
        system_state=_state(cuda_available=False),
        enabled=True,
        simulate_no_cuda=False,
        exclude_neural_requested=False,
        capability_aware_enabled=False,
    )

    assert effective is True
    assert report["effective_exclude_neural"] is True
    assert report["rules_applied"] == ["cuda_unavailable_exclude_neural_candidates"]


def test_apply_system_aware_policy_defers_to_capability_aware_when_set():
    effective, report = apply_system_aware_policy(
        system_state=_state(cuda_available=False),
        enabled=True,
        simulate_no_cuda=False,
        exclude_neural_requested=False,
        capability_aware_enabled=True,
    )

    assert effective is False
    assert report["capability_aware_enabled"] is True
    assert report["rules_applied"] == [
        "cuda_unavailable_defer_neural_filtering_to_codec_capabilities"
    ]


def test_apply_system_aware_policy_manual_exclude_neural_is_recorded():
    effective, report = apply_system_aware_policy(
        system_state=_state(cuda_available=True),
        enabled=False,
        simulate_no_cuda=False,
        exclude_neural_requested=True,
        capability_aware_enabled=False,
    )

    assert effective is True
    assert report["exclude_neural_requested"] is True
    assert report["rules_applied"] == ["manual_exclude_neural"]


def test_apply_system_aware_policy_simulate_no_cuda_overrides_state():
    effective, report = apply_system_aware_policy(
        system_state=_state(cuda_available=True),
        enabled=True,
        simulate_no_cuda=True,
        exclude_neural_requested=False,
        capability_aware_enabled=False,
    )

    assert effective is True
    assert report["simulate_no_cuda"] is True
    assert report["cuda_available"] is False
    assert report["rules_applied"] == ["cuda_unavailable_exclude_neural_candidates"]


def test_normalize_weights_returns_normalized_components():
    weights = pipeline._normalize_weights(1.0, 2.0, 1.0)

    assert set(weights.keys()) == {"w_E", "w_R", "w_D"}
    assert weights["w_E"] == 0.25
    assert weights["w_R"] == 0.5
    assert weights["w_D"] == 0.25
    assert abs(sum(weights.values()) - 1.0) < 1e-12


def _real_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "image_rde_real_small.csv"


def test_run_router_end_to_end_on_real_fixture_writes_report(tmp_path: Path):
    out_path = tmp_path / "router_decision_report.json"
    argv = [
        "--csv",
        str(_real_fixture()),
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
        "--out",
        str(out_path),
    ]

    original_argv = list(argv)
    expanded_argv, router_config_report = expand_argv_with_config(argv)
    parser = build_router_arg_parser()
    args = parser.parse_args(expanded_argv)
    router_context = RouterContext()

    run_router(
        args=args,
        router_context=router_context,
        original_argv=original_argv,
        expanded_argv=list(expanded_argv),
        router_config_report=router_config_report,
    )

    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    selected = report["decision"]["selected"]
    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
