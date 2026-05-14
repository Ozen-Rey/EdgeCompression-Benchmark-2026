"""Smoke checks for the router pipeline module.

The module hosts side-effect-free helpers extracted from rde_router.py.
These tests cover its import surface and the small pure helpers that are
not already exercised end-to-end by the router characterization tests.
"""

from src.router import pipeline


def test_pipeline_module_exports_extracted_helpers():
    for name in (
        "annotate_points_with_calibration_provenance",
        "build_weights_for_profile",
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
