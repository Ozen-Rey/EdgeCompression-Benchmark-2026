import json
from pathlib import Path

import pytest

from src.router.codecs.external_codec_spec import (
    load_external_codec_spec,
    main,
    validate_external_codec_spec,
)


def _valid_spec(**overrides):
    spec = {
        "schema_version": "0.36.0",
        "codec_id": "example_codec",
        "display_name": "Example Codec",
        "domain": "image",
        "family": "classical",
        "runtime": {
            "type": "external_command",
            "max_runtime_seconds": 30,
        },
        "version_probe": {
            "command_template": ["{binary}", "--version"],
        },
        "encode": {
            "command_template": [
                "{binary}",
                "--input",
                "{input}",
                "--output",
                "{output}",
                "--quality",
                "{quality}",
            ],
        },
        "decode": {
            "command_template": [
                "{binary}",
                "--decode",
                "{input}",
                "--output",
                "{output}",
            ],
        },
        "parameters": [
            {
                "name": "quality",
                "type": "integer",
                "values": [60, 75, 90],
            }
        ],
        "output": {
            "extension": ".exi",
        },
        "rate": {
            "metric": "bpp",
        },
        "quality": {
            "metric": "ssimulacra2",
            "direction": "higher_is_better",
        },
        "measurement": {
            "time": "wall_clock",
        },
        "requirements": {
            "binaries": ["example-codec"],
        },
        "security": {
            "allow_shell": False,
        },
    }
    spec.update(overrides)
    return spec


def test_minimal_valid_image_external_command_spec():
    report = validate_external_codec_spec(_valid_spec())

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["normalized_spec"]["codec_id"] == "example_codec"


def test_valid_python_module_spec():
    spec = _valid_spec(
        codec_id="python_codec",
        runtime={
            "type": "python_module",
            "module": "example_codec",
            "max_runtime_seconds": 10,
        },
    )

    report = validate_external_codec_spec(spec)

    assert report["valid"] is True
    assert report["errors"] == []


def test_invalid_codec_id_is_rejected():
    report = validate_external_codec_spec(_valid_spec(codec_id="Bad Codec!"))

    assert report["valid"] is False
    assert "invalid_codec_id" in report["errors"]


def test_invalid_domain_is_rejected():
    report = validate_external_codec_spec(_valid_spec(domain="text"))

    assert report["valid"] is False
    assert "invalid_domain" in report["errors"]


def test_shell_string_command_template_is_rejected():
    spec = _valid_spec(
        encode={
            "command_template": "codec --input {input} --output {output}",
        }
    )

    report = validate_external_codec_spec(spec)

    assert report["valid"] is False
    assert "encode_command_template_must_be_argv_list" in report["errors"]


def test_command_template_without_input_output_is_rejected():
    spec = _valid_spec(
        encode={
            "command_template": ["{binary}", "--quality", "{quality}"],
        }
    )

    report = validate_external_codec_spec(spec)

    assert report["valid"] is False
    assert "encode_command_template_missing_input_placeholder" in report["errors"]
    assert "encode_command_template_missing_output_placeholder" in report["errors"]


def test_allow_shell_true_is_rejected():
    report = validate_external_codec_spec(
        _valid_spec(security={"allow_shell": True})
    )

    assert report["valid"] is False
    assert "security_allow_shell_must_be_false" in report["errors"]


def test_parameter_without_values_is_rejected():
    report = validate_external_codec_spec(
        _valid_spec(parameters=[{"name": "quality", "type": "integer"}])
    )

    assert report["valid"] is False
    assert "parameter_0_missing_values" in report["errors"]
    assert "parameter_0_values_must_be_nonempty_list" in report["errors"]


def test_negative_max_runtime_seconds_is_rejected():
    report = validate_external_codec_spec(
        _valid_spec(runtime={"type": "external_command", "max_runtime_seconds": -1})
    )

    assert report["valid"] is False
    assert "invalid_max_runtime_seconds" in report["errors"]


def test_validation_does_not_execute_version_probe(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("validator must not execute commands")

    monkeypatch.setattr("subprocess.run", fail_if_called)

    report = validate_external_codec_spec(_valid_spec())

    assert report["valid"] is True


def test_cli_validate_reads_spec_without_execution(tmp_path: Path, capsys):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_valid_spec()), encoding="utf-8")

    report = main(["--spec", str(spec_path), "--validate"])
    captured = capsys.readouterr()

    assert report["valid"] is True
    assert json.loads(captured.out)["valid"] is True
    assert load_external_codec_spec(spec_path)["codec_id"] == "example_codec"


def test_cli_requires_validate(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_valid_spec()), encoding="utf-8")

    with pytest.raises(ValueError, match="--validate is required"):
        main(["--spec", str(spec_path)])
