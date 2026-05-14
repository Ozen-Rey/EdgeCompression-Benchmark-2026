import json
import hashlib
from pathlib import Path

import pytest

from src.router.rde_router import main
from src.router.version import DOMAIN_SUPPORT, FEATURE_LEVEL, ROUTER_VERSION
from src.router.codecs.codec_fingerprints import fingerprint_codec
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "router_real_fixture"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle_manifest(
    path: Path,
    calibrated_csv: Path,
    *,
    hash_value: str | None = None,
    codec_fingerprints: dict | None = None,
) -> None:
    payload = {
        "artifact_type": "promoted_calibration_bundle",
        "router_version": "0.18.0",
        "mode": "explicit_opt_in_calibration_apply",
        "source_benchmark": "benchmark.csv",
        "source_calibration": "calibration.json",
        "promotion_profile": "promotion.json",
        "output_csv": str(calibrated_csv),
        "created_at_utc": "2026-05-12T00:00:00+00:00",
        "accepted_scales": [
            {
                "codec": "HEVC",
                "config": "crf=15",
                "axis": "rate",
                "scale": 1.0,
            }
        ],
        "rejected_scales_count": 2,
        "energy_policy": {
            "requires_energy_usable_for_total": True,
            "gpu_only_energy_excluded": True,
        },
        "hashes": {
            "output_csv_sha256": hash_value or _sha256(calibrated_csv),
        },
    }
    if codec_fingerprints is not None:
        payload["codec_fingerprints"] = codec_fingerprints

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bundle_validation(
    path: Path,
    *,
    bundle_manifest: Path | None = None,
    calibrated_csv: Path | None = None,
    accepted: bool = True,
    candidate_manifest_hash: str | None = None,
    include_hashes: bool = True,
) -> None:
    payload = {
        "mode": "shadow_decision_validation_only",
        "router_version": "0.22.0",
        "comparison": "shadow_decision_comparison.json",
        "validated_comparison_path": "shadow_decision_comparison.json",
        "validated_comparison_sha256": "c" * 64,
        "accepted": accepted,
        "decision_count": 3,
        "changed_decision_count": 1,
        "decision_churn_rate": 1 / 3,
        "mean_baseline_cost": 1.0,
        "mean_candidate_cost": 0.9,
        "mean_delta_cost": -0.1,
        "relative_cost_improvement": 0.1,
        "rejection_reasons": []
        if accepted
        else ["candidate_cost_regression"],
        "acceptance_reasons": ["candidate_cost_not_regressed"]
        if accepted
        else [],
    }
    if include_hashes:
        payload["candidate_calibration_bundle_manifest_sha256"] = (
            candidate_manifest_hash
            or (_sha256(bundle_manifest) if bundle_manifest else "b" * 64)
        )
        payload["candidate_calibrated_csv_sha256"] = (
            _sha256(calibrated_csv) if calibrated_csv else "d" * 64
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_router_end_to_end_on_real_small_image_fixture():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )

    out_path = _tmp_path("real_fixture_report.json")

    main(
        [
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
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    selected = report["decision"]["selected"]

    assert selected["codec"] == "HEVC"
    assert selected["config"] == "crf=15"
    assert selected["energy_provenance_tier"] == "benchmark_reference"
    assert all(
        item["energy_provenance_tier"] == "benchmark_reference"
        for item in report["decision"]["scored_candidate_pool"]
    )
    assert report["energy_provenance_summary"]["selected_tier"] == (
        "benchmark_reference"
    )
    assert report["energy_provenance_summary"]["counts"]["benchmark_reference"] == (
        len(report["decision"]["scored_candidate_pool"])
        + len(report["decision"]["unscored_candidate_pool"])
    )
    compatibility = report["energy_provenance_compatibility"]
    assert compatibility["enabled"] is True
    assert compatibility["compatible"] is True
    assert compatibility["selected_tier"] == "benchmark_reference"
    assert compatibility["mixed_tiers"] is False
    assert compatibility["severity"] == "ok"
    assert compatibility["warnings"] == []
    policy = report["energy_tier_policy"]
    assert policy["enabled"] is True
    assert policy["mode"] == "report-only"
    assert policy["policy"] == "strict-compatible"
    assert policy["status"] == "no_action_single_tier_pool"
    assert policy["would_change_decision"] is False
    assert policy["shadow_selected"]["codec"] == selected["codec"]
    assert policy["shadow_selected"]["config"] == selected["config"]
    assert policy["shadow_selected"]["cost"] == selected["cost"]
    assert report["decision"]["decision_mode"] == "safe"
    assert report["router_version"] == ROUTER_VERSION
    assert report["feature_level"] == FEATURE_LEVEL
    assert report["domain_support"] == DOMAIN_SUPPORT
    assert (
        report["energy_provenance"]["local_energy_measurement"]
        == "hardware_backend_or_fallback"
    )
    assert report["energy_provenance"]["current_method"] == "benchmark_energy"
    assert report["energy_provenance"]["energy_backend"] == "benchmark_csv"
    assert report["energy_provenance"]["energy_is_measured"] is False
    assert report["calibration_bundle"] == {"enabled": False}
    assert report["calibration_bundle_validation"] == {"enabled": False}
    assert report["external_codecs"] == {"enabled": False}
    receipt = report["decision_receipt"]
    assert receipt["artifact_type"] == "router_decision_receipt"
    assert receipt["router_version"] == ROUTER_VERSION
    assert receipt["decision"]["selected_codec"] == selected["codec"]
    assert receipt["decision"]["selected_config"] == selected["config"]
    assert receipt["decision"]["cost"] == selected["cost"]
    assert "--out" not in receipt["replay"]["argv"]


def test_energy_tier_reporting_does_not_change_fixture_decision_or_ranking():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    first_out = _tmp_path("energy_tier_invariance_first.json")
    second_out = _tmp_path("energy_tier_invariance_second.json")
    args = [
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
    ]

    main([*args, "--out", str(first_out)])
    main([*args, "--out", str(second_out)])

    first = json.loads(first_out.read_text(encoding="utf-8"))
    second = json.loads(second_out.read_text(encoding="utf-8"))
    first_selected = first["decision"]["selected"]
    second_selected = second["decision"]["selected"]

    assert second_selected["codec"] == first_selected["codec"]
    assert second_selected["config"] == first_selected["config"]
    assert second_selected["cost"] == first_selected["cost"]
    assert second["energy_provenance_compatibility"]["compatible"] is True
    assert second["energy_provenance_compatibility"]["severity"] == "ok"
    assert second["energy_tier_policy"]["mode"] == "report-only"
    assert second["energy_tier_policy"]["would_change_decision"] is False
    assert [
        (item["rank"], item["codec"], item["config"], item["cost"])
        for item in second["decision"]["scored_candidate_pool"]
    ] == [
        (item["rank"], item["codec"], item["config"], item["cost"])
        for item in first["decision"]["scored_candidate_pool"]
    ]


def test_router_with_valid_bundle_reports_bundle_provenance():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("bundle_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("bundle_manifest.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    out_path = _tmp_path("bundle_report.json")

    main(
        [
            "--csv",
            str(fixture),
            "--calibration-bundle-manifest",
            str(manifest),
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
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    bundle = report["calibration_bundle"]

    assert bundle["enabled"] is True
    assert bundle["manifest_path"] == str(manifest)
    assert bundle["calibrated_csv_path"] == str(calibrated_csv)
    assert bundle["calibrated_csv_sha256"] == _sha256(calibrated_csv)
    assert bundle["validated"] is True
    assert bundle["applied_scales_count"] == 1
    assert bundle["rejected_scales_count"] == 2
    assert bundle["energy_policy"] == "usable_total_only"
    assert bundle["source"] == "explicit_calibration_bundle_manifest"
    assert bundle["codec_fingerprint_validation"] == {
        "enabled": False,
        "validated": False,
        "reason": "manifest_without_codec_fingerprints",
        "validated_codecs": [],
        "mismatches": [],
    }
    assert report["csv"] == str(calibrated_csv)
    assert report["calibration_bundle_validation"] == {"enabled": False}


def test_router_with_modern_bundle_reports_nonempty_fingerprint_validation():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("fingerprint_bundle_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    legacy_manifest = _tmp_path("fingerprint_legacy_manifest.json")
    modern_manifest = _tmp_path("fingerprint_modern_manifest.json")
    legacy_out = _tmp_path("fingerprint_legacy_report.json")
    modern_out = _tmp_path("fingerprint_modern_report.json")
    _write_bundle_manifest(legacy_manifest, calibrated_csv)
    _write_bundle_manifest(
        modern_manifest,
        calibrated_csv,
        codec_fingerprints={"JPEG": fingerprint_codec("JPEG")},
    )

    base_args = [
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
    ]

    main(
        [
            *base_args,
            "--calibration-bundle-manifest",
            str(legacy_manifest),
            "--out",
            str(legacy_out),
        ]
    )
    main(
        [
            *base_args,
            "--calibration-bundle-manifest",
            str(modern_manifest),
            "--out",
            str(modern_out),
        ]
    )

    legacy_report = json.loads(legacy_out.read_text(encoding="utf-8"))
    modern_report = json.loads(modern_out.read_text(encoding="utf-8"))
    validation = modern_report["calibration_bundle"][
        "codec_fingerprint_validation"
    ]

    assert validation["enabled"] is True
    assert validation["validated"] is True
    assert validation["validated_codecs"] == ["JPEG"]
    assert validation["mismatches"] == []
    assert (
        modern_report["decision"]["selected"]
        == legacy_report["decision"]["selected"]
    )


def test_router_with_validated_bundle_reports_validation_provenance():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("validated_bundle_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("validated_bundle_manifest.json")
    validation = _tmp_path("validated_bundle_validation.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    _write_bundle_validation(
        validation,
        bundle_manifest=manifest,
        calibrated_csv=calibrated_csv,
        accepted=True,
    )
    out_path = _tmp_path("validated_bundle_report.json")

    main(
        [
            "--csv",
            str(fixture),
            "--calibration-bundle-manifest",
            str(manifest),
            "--calibration-bundle-validation",
            str(validation),
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
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    validation_report = report["calibration_bundle_validation"]

    assert validation_report["enabled"] is True
    assert validation_report["accepted"] is True
    assert validation_report["mode"] == "shadow_decision_validation_only"
    assert validation_report["validation_path"] == str(validation)
    assert validation_report["validation_sha256"] == _sha256(validation)
    assert validation_report["bundle_manifest_sha256"] == _sha256(manifest)
    assert validation_report["validation_bundle_manifest_sha256"] == _sha256(
        manifest
    )
    assert validation_report["candidate_calibrated_csv_sha256"] == _sha256(
        calibrated_csv
    )
    assert validation_report["integrity_match"] is True
    assert validation_report["rejection_reasons"] == []
    assert validation_report["decision_count"] == 3
    assert validation_report["changed_decision_count"] == 1
    assert validation_report["decision_churn_rate"] == 1 / 3
    assert validation_report["relative_cost_improvement"] == 0.1
    assert report["csv"] == str(calibrated_csv)


def test_router_rejects_bundle_with_rejected_validation():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("rejected_validation_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("rejected_validation_manifest.json")
    validation = _tmp_path("rejected_validation.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    _write_bundle_validation(
        validation,
        bundle_manifest=manifest,
        calibrated_csv=calibrated_csv,
        accepted=False,
    )

    with pytest.raises(ValueError, match="validation was not accepted"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-manifest",
                str(manifest),
                "--calibration-bundle-validation",
                str(validation),
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
            ]
        )


def test_router_rejects_validation_bound_to_different_bundle():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("mismatch_validation_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("mismatch_validation_manifest.json")
    validation = _tmp_path("mismatch_validation.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    _write_bundle_validation(
        validation,
        calibrated_csv=calibrated_csv,
        accepted=True,
        candidate_manifest_hash="0" * 64,
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-manifest",
                str(manifest),
                "--calibration-bundle-validation",
                str(validation),
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
            ]
        )


def test_router_rejects_legacy_validation_without_bundle_hash():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("legacy_validation_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("legacy_validation_manifest.json")
    validation = _tmp_path("legacy_validation.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    _write_bundle_validation(
        validation,
        accepted=True,
        include_hashes=False,
    )

    with pytest.raises(ValueError, match="v0.24"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-manifest",
                str(manifest),
                "--calibration-bundle-validation",
                str(validation),
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
            ]
        )


def test_router_rejects_missing_or_malformed_bundle_validation():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("malformed_validation_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("malformed_validation_manifest.json")
    _write_bundle_manifest(manifest, calibrated_csv)

    with pytest.raises(ValueError, match="validation not found"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-manifest",
                str(manifest),
                "--calibration-bundle-validation",
                str(_tmp_path("missing_validation.json")),
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
            ]
        )

    malformed = _tmp_path("malformed_validation.json")
    malformed.write_text(json.dumps({"accepted": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="requires"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-manifest",
                str(manifest),
                "--calibration-bundle-validation",
                str(malformed),
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
            ]
        )


def test_router_rejects_validation_without_bundle_manifest():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    validation = _tmp_path("validation_without_bundle.json")
    _write_bundle_validation(validation, accepted=True)

    with pytest.raises(ValueError, match="requires --calibration-bundle-manifest"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-validation",
                str(validation),
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
            ]
        )


def test_bundle_validation_flag_only_gates_and_does_not_change_ranking():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("ranking_gate_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("ranking_gate_manifest.json")
    validation = _tmp_path("ranking_gate_validation.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    _write_bundle_validation(
        validation,
        bundle_manifest=manifest,
        calibrated_csv=calibrated_csv,
        accepted=True,
    )
    out_without_validation = _tmp_path("ranking_gate_without_validation.json")
    out_with_validation = _tmp_path("ranking_gate_with_validation.json")

    common_args = [
        "--csv",
        str(fixture),
        "--calibration-bundle-manifest",
        str(manifest),
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
    ]

    main(common_args + ["--out", str(out_without_validation)])
    main(
        common_args
        + [
            "--calibration-bundle-validation",
            str(validation),
            "--out",
            str(out_with_validation),
        ]
    )

    without_validation = json.loads(
        out_without_validation.read_text(encoding="utf-8")
    )
    with_validation = json.loads(
        out_with_validation.read_text(encoding="utf-8")
    )

    assert without_validation["decision"]["selected"] == with_validation[
        "decision"
    ]["selected"]
    assert without_validation["decision"]["decision_mode"] == with_validation[
        "decision"
    ]["decision_mode"]


def test_router_does_not_auto_discover_bundle_validation():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("no_auto_validation_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("no_auto_validation_manifest.json")
    adjacent_rejected = manifest.with_name("shadow_decision_validation.json")
    _write_bundle_manifest(manifest, calibrated_csv)
    _write_bundle_validation(
        adjacent_rejected,
        bundle_manifest=manifest,
        calibrated_csv=calibrated_csv,
        accepted=False,
    )
    out_path = _tmp_path("no_auto_validation_report.json")

    main(
        [
            "--csv",
            str(fixture),
            "--calibration-bundle-manifest",
            str(manifest),
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
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["calibration_bundle"]["enabled"] is True
    assert report["calibration_bundle_validation"] == {"enabled": False}


def test_router_with_invalid_bundle_fails_controlled():
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_rde_real_small.csv"
    )
    calibrated_csv = _tmp_path("bundle_invalid_calibrated.csv")
    calibrated_csv.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = _tmp_path("bundle_invalid_manifest.json")
    _write_bundle_manifest(manifest, calibrated_csv, hash_value="0" * 64)
    out_path = _tmp_path("bundle_invalid_report.json")

    with pytest.raises(ValueError, match="hash mismatch"):
        main(
            [
                "--csv",
                str(fixture),
                "--calibration-bundle-manifest",
                str(manifest),
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
                "--out",
                str(out_path),
            ]
        )
