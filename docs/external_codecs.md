# External Codec Specifications

Router v0.36.0 introduces a declarative JSON schema for describing external
codecs. This is validation-only: specs are not registered with the router, no
codec commands are executed, no version probes are run, and no ranking or
backend execution behavior changes.

Validate a spec offline:

```powershell
python -m src.router.external_codec_spec `
  --spec configs/external_codecs/example_image_codec.json `
  --validate
```

The validator returns:

```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "normalized_spec": {}
}
```

## Required Fields

An external codec spec is a JSON object with these top-level fields:

- `schema_version`
- `codec_id`
- `display_name`
- `domain`: `image`, `video`, or `audio`
- `family`: `classical`, `neural`, `hybrid`, or `unknown`
- `runtime`
- `version_probe`
- `encode`
- `decode`
- `parameters`
- `output`
- `rate`
- `quality`
- `measurement`
- `requirements`
- `security`

`codec_id` must be a lowercase slug containing only letters, numbers,
underscore, and dash.

## Runtime

Supported `runtime.type` values in v0.36.0:

- `external_command`
- `python_module`

`runtime.max_runtime_seconds`, when present, must be positive.

## Command Templates

`encode.command_template` and `decode.command_template` must be argv lists, not
shell strings:

```json
{
  "encode": {
    "command_template": [
      "{binary}",
      "--input",
      "{input}",
      "--output",
      "{output}"
    ]
  }
}
```

Both templates must include `{input}` and `{output}` placeholders.
`version_probe` is declared but not executed by the validator.

## Parameters

`parameters` must be a list. Each parameter must include:

- `name`
- `type`
- `values`

`values` must be a non-empty list.

## Security

`security.allow_shell` must be `false` or absent. v0.36.0 intentionally rejects
shell-string command templates and does not check whether binaries exist.

## Minimal Example

```json
{
  "schema_version": "0.36.0",
  "codec_id": "example_codec",
  "display_name": "Example Codec",
  "domain": "image",
  "family": "classical",
  "runtime": {
    "type": "external_command",
    "max_runtime_seconds": 30
  },
  "version_probe": {
    "command_template": ["{binary}", "--version"]
  },
  "encode": {
    "command_template": [
      "{binary}",
      "--input",
      "{input}",
      "--output",
      "{output}",
      "--quality",
      "{quality}"
    ]
  },
  "decode": {
    "command_template": ["{binary}", "--decode", "{input}", "--output", "{output}"]
  },
  "parameters": [
    {
      "name": "quality",
      "type": "integer",
      "values": [60, 75, 90]
    }
  ],
  "output": {
    "extension": ".exi"
  },
  "rate": {
    "metric": "bpp"
  },
  "quality": {
    "metric": "ssimulacra2",
    "direction": "higher_is_better"
  },
  "measurement": {
    "time": "wall_clock"
  },
  "requirements": {
    "binaries": ["example-codec"]
  },
  "security": {
    "allow_shell": false
  }
}
```
