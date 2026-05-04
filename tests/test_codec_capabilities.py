from src.router.codec_capabilities import (
    build_execution_plan,
    get_codec_capability,
    is_neural_codec,
)


def test_codec_registry_knows_classical_and_neural_codecs():
    jxl = get_codec_capability("JXL")
    dcae = get_codec_capability("DCAE")

    assert jxl["family"] == "classical"
    assert jxl["execution_supported"] is True

    assert dcae["family"] == "neural"
    assert is_neural_codec("DCAE") is True


def test_unknown_codec_is_not_assumed_neural():
    cap = get_codec_capability("MyUnknownCodec")

    assert cap["family"] == "unknown"
    assert is_neural_codec("MyUnknownCodec") is False
    assert cap["benchmark_only"] is True


def test_jxl_output_extension_is_corrected():
    system_state = {
        "executables": {
            "cjxl": {
                "available": True,
                "path": "C:\\fake\\cjxl.exe",
                "source": "test",
            }
        }
    }

    plan = build_execution_plan(
        codec_name="JXL",
        config="d=1.0",
        input_path="input.png",
        output_path="wrong.mp4",
        system_state=system_state,
        requested=True,
    )

    assert plan["output"].endswith(".jxl")
    assert "output_extension_corrected:.mp4->.jxl" in plan["warnings"]
