"""Smoke checks for the router pipeline module.

The module hosts side-effect-free helpers extracted from rde_router.py
plus the run_router orchestration entry point. These tests cover its
import surface, the small pure helpers, and a minimal end-to-end wiring
exercise of run_router against the real fixture.
"""

import json
from pathlib import Path

from src.router import pipeline
from src.router.cli import build_router_arg_parser
from src.router.context import RouterContext
from src.router.core.router_config import expand_argv_with_config
from src.router.pipeline import run_router


def test_pipeline_module_exports_extracted_helpers():
    for name in (
        "annotate_points_with_calibration_provenance",
        "build_weights_for_profile",
        "run_router",
        "summary_row_from_report",
        "topk_rows_from_report",
    ):
        assert callable(getattr(pipeline, name)), f"missing pipeline helper: {name}"


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
