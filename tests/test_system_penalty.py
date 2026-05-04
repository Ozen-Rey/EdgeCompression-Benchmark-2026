from src.router.system_penalty import (
    build_system_penalty_context,
    compute_candidate_system_penalty,
)


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
