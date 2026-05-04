import argparse

from src.router.run_manifest import build_run_manifest


def test_build_run_manifest_contains_reproducibility_fields():
    args = argparse.Namespace(
        csv="results/images/image_4dataset_RDE_paper_ready.csv",
        codec_registry_file="configs/codecs_image_v05.json",
        normalization_file="results/routing_context/normalization_image.json",
        calibration_file=None,
        quality_thresholds_file="configs/quality_thresholds.json",
        input="test_images/input.png",
        output="test_images/output.jxl",
    )

    router_config_report = {
        "enabled": True,
        "source": "configs/router_image_v05.json",
        "experiment_name": "image_v06_manifest_test",
    }

    manifest = build_run_manifest(
        original_argv=["--config", "configs/router_image_v05.json"],
        expanded_argv=["--csv", "results/images/image_4dataset_RDE_paper_ready.csv"],
        args=args,
        router_config_report=router_config_report,
    )

    assert manifest["enabled"] is True
    assert manifest["version"] == "0.6"

    assert "created_at" in manifest
    assert "python" in manifest
    assert "platform" in manifest
    assert "git" in manifest
    assert "argv" in manifest
    assert "inputs" in manifest
    assert "resolved_args" in manifest

    assert manifest["inputs"]["router_config_file"] == "configs/router_image_v05.json"
    assert manifest["inputs"]["router_config_experiment"] == "image_v06_manifest_test"
    assert manifest["inputs"]["codec_registry_file"] == "configs/codecs_image_v05.json"
    assert manifest["inputs"]["normalization_file"] == "results/routing_context/normalization_image.json"

    assert manifest["argv"]["original"] == ["--config", "configs/router_image_v05.json"]
    assert manifest["argv"]["expanded"] == [
        "--csv",
        "results/images/image_4dataset_RDE_paper_ready.csv",
    ]