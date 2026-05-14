"""Smoke checks for the router outputs module.

Covers the public symbol surface and the small pure helper
``safe_profile_filename`` end-to-end on a few characteristic inputs.
The CSV/JSON writers are already exercised indirectly by the router
characterization tests, so we only verify their import path here.
"""

import json
from pathlib import Path

from src.router import outputs


def test_outputs_module_exports_writers_and_helper():
    for name in (
        "safe_profile_filename",
        "write_json_report",
        "write_summary_csv",
        "write_topk_csv",
    ):
        assert callable(getattr(outputs, name)), f"missing output helper: {name}"


def test_safe_profile_filename_normalizes_separators_and_case():
    assert outputs.safe_profile_filename("balanced") == "balanced"
    assert outputs.safe_profile_filename("Energy-Limited") == "energy_limited"
    assert outputs.safe_profile_filename("  Bandwidth Limited  ") == "bandwidth_limited"
    assert outputs.safe_profile_filename("quality-first") == "quality_first"


def test_write_json_report_creates_parent_and_writes_unicode(tmp_path: Path):
    out = tmp_path / "nested" / "report.json"
    payload = {"profile": "balanced", "note": "città"}

    outputs.write_json_report(payload, out)

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload


def test_write_summary_csv_skips_empty_rows(tmp_path: Path):
    out = tmp_path / "summary.csv"
    outputs.write_summary_csv([], out)
    assert not out.exists()


def test_write_summary_csv_writes_header_and_row(tmp_path: Path):
    out = tmp_path / "nested" / "summary.csv"
    outputs.write_summary_csv(
        [{"profile": "balanced", "selected_codec": "HEVC"}],
        out,
    )

    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "profile,selected_codec"
    assert text[1] == "balanced,HEVC"
