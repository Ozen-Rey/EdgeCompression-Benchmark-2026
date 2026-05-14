from pathlib import Path

from src.router.adaptation.system_penalty import (
    build_system_penalty_context,
    compute_candidate_system_penalty,
    load_system_penalty_weights,
)
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "system_penalty"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _features_with_classes(**classes):
    base = {
        "cpu": "normal",
        "memory": "normal",
        "battery": "ac",
        "gpu": "available",
        "thermal": "nominal",
        "disk": "normal",
    }
    base.update(classes)

    return {
        "enabled": True,
        "derived_constraints": {
            "classes": base,
        },
    }


def test_battery_critical_penalizes_hevc_more_than_jpeg():
    context = build_system_penalty_context(
        enabled=True,
        mode="apply",
        lambda_sys=0.5,
        system_features_report=_features_with_classes(
            battery="critical",
        ),
    )

    jpeg = compute_candidate_system_penalty(
        codec_name="JPEG",
        config="q=85",
        context=context,
    )

    hevc = compute_candidate_system_penalty(
        codec_name="HEVC",
        config="crf=15",
        context=context,
    )

    assert hevc["penalty_norm"] > jpeg["penalty_norm"]
    assert hevc["weighted_penalty"] > jpeg["weighted_penalty"]


def test_gpu_unavailable_hard_excludes_cuda_codec():
    context = build_system_penalty_context(
        enabled=True,
        mode="apply",
        lambda_sys=0.5,
        system_features_report=_features_with_classes(
            gpu="unavailable",
        ),
    )

    penalty = compute_candidate_system_penalty(
        codec_name="DCAE",
        config="lam=0.0035",
        context=context,
    )

    assert penalty["hard_excluded"] is True
    assert "requires_cuda_but_gpu_unavailable" in penalty["hard_exclusion_reasons"]


def test_gpu_unknown_does_not_hard_exclude_cuda_codec():
    context = build_system_penalty_context(
        enabled=True,
        mode="apply",
        lambda_sys=0.5,
        system_features_report=_features_with_classes(
            gpu="unknown",
        ),
    )

    penalty = compute_candidate_system_penalty(
        codec_name="DCAE",
        config="lam=0.0035",
        context=context,
    )

    assert penalty["hard_excluded"] is False
    assert "gpu_unknown_cuda_requirement_not_hard_excluded" in penalty["warnings"]


def test_memory_critical_hard_excludes_high_memory_codec():
    context = build_system_penalty_context(
        enabled=True,
        mode="apply",
        lambda_sys=0.5,
        system_features_report=_features_with_classes(
            memory="critical",
        ),
    )

    penalty = compute_candidate_system_penalty(
        codec_name="DCAE",
        config="lam=0.0035",
        context=context,
    )

    assert penalty["hard_excluded"] is True
    assert "memory_critical_high_memory_codec" in penalty["hard_exclusion_reasons"]


def test_system_penalty_battery_critical_coefficient_regression_for_jpeg():
    context = build_system_penalty_context(
        enabled=True,
        mode="apply",
        lambda_sys=0.5,
        system_features_report=_features_with_classes(
            battery="critical",
        ),
    )

    penalty = compute_candidate_system_penalty(
        codec_name="JPEG",
        config="q=85",
        context=context,
    )

    # JPEG has resource_profile.energy = low -> energy_score = 1.
    # battery critical coefficient is currently 0.10 * energy_score.
    assert penalty["penalty_norm"] == 0.10
    assert penalty["weighted_penalty"] == 0.05
    assert "battery_critical_energy_score=1" in penalty["rules_applied"]


def test_load_system_penalty_weights_uses_defaults_without_file():
    report = load_system_penalty_weights(None)

    assert report["source"] is None
    assert report["source_exists"] is False
    assert report["weights"]["battery"]["critical_energy"] == 0.10
    assert report["weights"]["cpu"]["busy_cpu"] == 0.08


def test_load_system_penalty_weights_merges_custom_file():
    path = _tmp_path("weights.json")

    path.write_text(
        """
        {
          "battery": {
            "critical_energy": 0.20
          }
        }
        """,
        encoding="utf-8",
    )

    report = load_system_penalty_weights(str(path))

    assert report["source"] == str(path)
    assert report["source_exists"] is True
    assert report["weights"]["battery"]["critical_energy"] == 0.20

    # Unspecified values must still come from defaults.
    assert report["weights"]["cpu"]["busy_cpu"] == 0.08


def test_custom_system_penalty_weights_change_penalty_value():
    path = _tmp_path("custom_weights.json")

    path.write_text(
        """
        {
          "battery": {
            "critical_energy": 0.20
          }
        }
        """,
        encoding="utf-8",
    )

    weights_report = load_system_penalty_weights(str(path))

    context = build_system_penalty_context(
        enabled=True,
        mode="apply",
        lambda_sys=0.5,
        system_features_report=_features_with_classes(
            battery="critical",
        ),
        penalty_weights=weights_report["weights"],
        penalty_weights_source=weights_report["source"],
    )

    penalty = compute_candidate_system_penalty(
        codec_name="JPEG",
        config="q=85",
        context=context,
    )

    # JPEG energy_score = 1, custom coefficient = 0.20.
    assert penalty["penalty_norm"] == 0.20
    assert penalty["weighted_penalty"] == 0.10
