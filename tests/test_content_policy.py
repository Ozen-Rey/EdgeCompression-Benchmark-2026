from pathlib import Path

import pytest

from src.router.content_policy import (
    build_content_policy_report,
    get_content_policy_preferred_candidate,
    load_content_policy_rules,
)


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "content_policy"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_load_content_policy_rules_reads_metadata_policy_csv():
    path = _tmp_path("rules_load.csv")

    path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,tecnick,JPEG,q=85",
                "dataset,clic2020,JXL,d=1.0",
            ]
        ),
        encoding="utf-8",
    )

    rules = load_content_policy_rules(str(path))

    assert rules[("dataset", "tecnick")] == ("JPEG", "q=85")
    assert rules[("dataset", "clic2020")] == ("JXL", "d=1.0")


def test_content_policy_report_only_suggests_without_apply():
    path = _tmp_path("rules_report_only.csv")

    path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,tecnick,JPEG,q=85",
            ]
        ),
        encoding="utf-8",
    )

    report = build_content_policy_report(
        enabled=True,
        mode="report-only",
        rules_file=str(path),
        policy_key="dataset",
        policy_value="tecnick",
    )

    assert report["enabled"] is True
    assert report["mode"] == "report-only"
    assert report["suggestion"]["codec"] == "JPEG"
    assert report["suggestion"]["config"] == "q=85"
    assert get_content_policy_preferred_candidate(report) is None


def test_content_policy_apply_returns_preferred_candidate():
    path = _tmp_path("rules_apply.csv")

    path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,clic2020,JXL,d=1.0",
            ]
        ),
        encoding="utf-8",
    )

    report = build_content_policy_report(
        enabled=True,
        mode="apply",
        rules_file=str(path),
        policy_key="dataset",
        policy_value="clic2020",
    )

    assert get_content_policy_preferred_candidate(report) == ("JXL", "d=1.0")


def test_content_policy_missing_source_warns_without_suggestion():
    path = _tmp_path("rules_missing_source.csv")

    path.write_text(
        "\n".join(
            [
                "policy_key,policy_value,selected_codec,selected_config",
                "dataset,tecnick,JPEG,q=85",
            ]
        ),
        encoding="utf-8",
    )

    report = build_content_policy_report(
        enabled=True,
        mode="apply",
        rules_file=str(path),
        policy_key="dataset",
        policy_value=None,
    )

    assert report["suggestion"] is None
    assert "missing_policy_value" in report["reasons"]
    assert get_content_policy_preferred_candidate(report) is None


def test_content_policy_rejects_invalid_mode():
    with pytest.raises(ValueError, match="content policy mode"):
        build_content_policy_report(
            enabled=True,
            mode="bad",
            rules_file=None,
            policy_key="dataset",
            policy_value="tecnick",
        )
