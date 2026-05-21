import inspect
import json
from pathlib import Path

import pytest

from src.router.analysis import audio_video_policy_validation as avpv
from src.router.core.domain_spec import main as domain_spec_main


ROOT = Path(__file__).resolve().parents[1]
AUDIO_FIXTURE = ROOT / "tests" / "fixtures" / "rde_audio_visqol.csv"
VIDEO_FIXTURE = ROOT / "tests" / "fixtures" / "rde_video_vmaf.csv"


def test_audio_router_ready_fixture_validates_with_audio_visqol(capsys) -> None:
    report = domain_spec_main(
        ["--csv", str(AUDIO_FIXTURE), "--builtin", "audio_visqol", "--validate-csv"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert report["valid"] is True
    assert payload["valid"] is True
    assert payload["normalized_spec"]["quality_column"] == "visqol"


def test_video_router_ready_fixture_validates_with_video_vmaf(capsys) -> None:
    report = domain_spec_main(
        ["--csv", str(VIDEO_FIXTURE), "--builtin", "video_vmaf", "--validate-csv"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert report["valid"] is True
    assert payload["valid"] is True
    assert payload["normalized_spec"]["quality_column"] == "vmaf"


def test_policy_validation_uses_domain_spec_columns(tmp_path: Path) -> None:
    out_prefix = tmp_path / "audio_policy"

    report = avpv.policy_validation(AUDIO_FIXTURE, "audio_visqol", out_prefix)

    assert report["valid"] is True
    assert report["domain_spec_details"]["rate_column"] == "bitrate_kbps"
    assert report["domain_spec_details"]["quality_column"] == "visqol"
    assert report["domain_spec_details"]["energy_column"] == "energy_j_per_second"
    assert (tmp_path / "audio_policy_comparison.csv").exists()
    assert (tmp_path / "audio_policy_comparison.json").exists()


def test_policy_validation_module_has_no_image_column_hardcoding() -> None:
    source = inspect.getsource(avpv)

    assert "ssimulacra2" not in source
    assert "energy_per_image_j" not in source
    assert "bpp" not in source


def test_audio_video_policy_validation_cli_help() -> None:
    with pytest.raises(SystemExit) as excinfo:
        avpv.main(["--help"])

    assert excinfo.value.code == 0
