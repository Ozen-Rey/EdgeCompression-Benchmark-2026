from pathlib import Path

from src.router.router_config import config_to_cli_args, expand_argv_with_config


def test_config_to_cli_args_maps_core_fields():
    config = {
        "domain": "image",
        "data": {
            "csv": "results/images/image_4dataset_RDE_paper_ready.csv",
            "out_report": "results/routing_context/test.json",
        },
        "columns": {
            "codec": "codec",
            "config": "param",
            "rate": "bpp",
            "quality": "ssimulacra2",
            "energy": "energy_per_image_j",
            "time": "time_ms",
        },
        "selection": {
            "available_codecs": ["JPEG", "JXL", "HEVC"],
            "aggregate_by_config": True,
            "auto_weights": True,
            "quality_target": "high",
            "allow_degraded_fallback": True,
        },
        "system": {
            "system_aware": True,
            "capability_aware": True,
            "strict_executables": True,
        },
        "normalization": {
            "mode": "global",
            "file": "results/routing_context/normalization_image.json",
        },
    }

    args = config_to_cli_args(config)

    assert "--domain" in args
    assert "image" in args
    assert "--csv" in args
    assert "results/images/image_4dataset_RDE_paper_ready.csv" in args
    assert "--codec-col" in args
    assert "codec" in args
    assert "--available-codecs" in args
    assert "JPEG,JXL,HEVC" in args
    assert "--aggregate-by-config" in args
    assert "--auto-weights" in args
    assert "--quality-target" in args
    assert "high" in args
    assert "--system-aware" in args
    assert "--capability-aware" in args
    assert "--strict-executables" in args
    assert "--normalization-mode" in args
    assert "global" in args


def test_expand_argv_with_config_places_cli_overrides_after_config():
    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    config_path = tmp_dir / "router_config.json"

    config_path.write_text(
        """
        {
          "experiment_name": "test_exp",
          "domain": "image",
          "data": {
            "csv": "from_config.csv"
          },
          "selection": {
            "available_codecs": ["JPEG", "HEVC"]
          }
        }
        """,
        encoding="utf-8",
    )

    expanded, report = expand_argv_with_config(
        [
            "--config",
            str(config_path),
            "--available-codecs",
            "JXL",
            "--out",
            "override.json",
        ]
    )

    assert report["enabled"] is True
    assert report["source"] == str(config_path)
    assert report["experiment_name"] == "test_exp"

    config_available_idx = expanded.index("--available-codecs")
    override_available_idx = len(expanded) - 4

    assert expanded[config_available_idx + 1] == "JPEG,HEVC"
    assert expanded[override_available_idx] == "--available-codecs"
    assert expanded[override_available_idx + 1] == "JXL"

    assert expanded[-2] == "--out"
    assert expanded[-1] == "override.json"
