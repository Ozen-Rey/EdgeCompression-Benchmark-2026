"""Intentional smoke coverage for legacy router wrapper module paths."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("legacy_name", "subpackage_name", "representative"),
    [
        ("src.router.rde_database", "src.router.core.rde_database", "select_best_rde"),
        (
            "src.router.external_codec_spec",
            "src.router.codecs.external_codec_spec",
            "validate_external_codec_spec",
        ),
        (
            "src.router.calibration_bundle",
            "src.router.calibration.calibration_bundle",
            "validate_calibration_bundle_manifest",
        ),
        (
            "src.router.energy_provenance",
            "src.router.adaptation.energy_provenance",
            "classify_energy_provenance",
        ),
        (
            "src.router.decision_receipt",
            "src.router.observability.decision_receipt",
            "build_decision_receipt",
        ),
        (
            "src.router.content_oracle_analysis",
            "src.router.analysis.content_oracle_analysis",
            "analyze_content_oracle",
        ),
    ],
)
def test_legacy_wrapper_import_matches_subpackage(
    legacy_name: str, subpackage_name: str, representative: str
):
    legacy_module = importlib.import_module(legacy_name)
    subpackage_module = importlib.import_module(subpackage_name)

    assert legacy_module is subpackage_module or getattr(
        legacy_module, representative
    ) is getattr(subpackage_module, representative)


@pytest.mark.parametrize(
    "module_name",
    [
        "src.router.external_codec_spec",
        "src.router.calibration_apply",
        "src.router.content_oracle_analysis",
    ],
)
def test_representative_legacy_cli_help_still_works(module_name: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
