import json
from pathlib import Path

from src.router.codec_capabilities import (
    build_execution_plan,
    get_codec_capability,
    load_external_codec_registry,
)


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "codec_registry"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_external_codec_registry_overrides_capability():
    registry_path = _tmp_path("codecs_override.json")

    registry_path.write_text(
        json.dumps(
            {
                "version": "0.5",
                "domain": "image",
                "codecs": {
                    "JXL": {
                        "canonical": "JXL",
                        "aliases": ["jpegxl"],
                        "family": "classical",
                        "requires_cuda": False,
                        "requires_torch": False,
                        "requires_executables": ["cjxl"],
                        "execution_supported": True,
                        "execution_backend": "external_command",
                        "output_extension": ".jxl",
                        "command_template": [
                            "{cjxl}",
                            "{input}",
                            "{output}",
                            "-d",
                            "{distance}",
                        ],
                        "config_map": {
                            "d=1.0": {
                                "distance": "1.0",
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = load_external_codec_registry(str(registry_path))

    assert report["enabled"] is True
    assert report["num_codecs"] == 1

    cap = get_codec_capability("JXL")

    assert cap["canonical"] == "JXL"
    assert cap["execution_backend"] == "external_command"
    assert cap["_external_registry"] is True


def test_external_command_plan_builds_template():
    registry_path = _tmp_path("codecs_command.json")
    input_path = _tmp_path("input.png")
    output_path = _tmp_path("output.mp4")

    input_path.write_bytes(b"fake")

    registry_path.write_text(
        json.dumps(
            {
                "version": "0.5",
                "domain": "image",
                "codecs": {
                    "JXL": {
                        "canonical": "JXL",
                        "family": "classical",
                        "requires_executables": ["cjxl"],
                        "execution_supported": True,
                        "execution_backend": "external_command",
                        "output_extension": ".jxl",
                        "command_template": [
                            "{cjxl}",
                            "{input}",
                            "{output}",
                            "-d",
                            "{distance}",
                        ],
                        "config_map": {
                            "d=1.0": {
                                "distance": "1.0",
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    load_external_codec_registry(str(registry_path))

    system_state = {
        "executables": {
            "cjxl": {
                "available": True,
                "path": "C:/tools/jxl/cjxl.exe",
            },
        },
    }

    plan = build_execution_plan(
        codec_name="JXL",
        config="d=1.0",
        input_path=str(input_path),
        output_path=str(output_path),
        system_state=system_state,
        requested=True,
    )

    assert plan["can_execute"] is True
    assert plan["execution_backend"] == "external_command"
    assert plan["output"].endswith(".jxl")
    assert plan["command"] == [
        "C:/tools/jxl/cjxl.exe",
        str(input_path),
        str(output_path.with_suffix(".jxl")),
        "-d",
        "1.0",
    ]
