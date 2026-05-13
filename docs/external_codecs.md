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

Probe a valid spec without running encode/decode:

```powershell
python -m src.router.external_codec_probe `
  --spec configs/external_codecs/example_image_codec.json `
  --out results/routing_context/external_codec_probe.json
```

The v0.37.0 probe performs only this sequence:

```text
external codec spec -> schema validation -> executable/version/fingerprint probe -> report
```

It does not benchmark the codec, read datasets, write codec outputs, register
the codec with the router, or change any router candidate pool.

Run a controlled one-input contract dry-run:

```powershell
python -m src.router.external_codec_dry_run `
  --spec configs/external_codecs/example_image_codec.json `
  --input test_images/input.png `
  --out-dir results/external_codec_dry_runs/example_codec `
  --param quality=50 `
  --timeout-s 30 `
  --out results/external_codec_dry_runs/example_codec_dry_run.json
```

The v0.38.0 dry-run performs only contract validation on one declared input:
it validates the spec, checks the external executable, builds argv commands
from `encode.command_template` and, only when `decode.available=true`,
`decode.command_template`, confines generated outputs to `--out-dir`, enforces
timeouts, and verifies that expected files exist with the declared extension
and optional non-empty contract.

It does not read benchmark datasets, compute quality metrics, compute energy,
generate R-D-E CSV rows, register the codec, or affect router decisions.

Run a small raw benchmark over explicitly named inputs:

```powershell
python -m src.router.external_codec_benchmark `
  --spec configs/external_codecs/example_image_codec.json `
  --input test_images/input_a.png `
  --input test_images/input_b.png `
  --out-dir results/external_codec_benchmarks/example_codec `
  --param-set '{"quality":"50"}' `
  --param-set '{"quality":"75"}' `
  --timeout-s 30 `
  --out results/external_codec_benchmarks/example_codec_benchmark.json `
  --csv results/external_codec_benchmarks/example_codec_measurements.csv
```

Router v0.39.0's benchmark runner repeats the dry-run contract over a small,
explicit input list and a declared parameter grid. It writes a JSON report and
a raw CSV with encode/decode success, output paths, output byte size, wall-time
measurements and a simple rate value. Failed runs produce failed rows instead
of disappearing.

This CSV is not an R-D-E database and is not consumed by the router. The runner
does not compute quality metrics, does not require real energy measurement,
does not score candidates, does not register codecs and does not change router
ranking.

Export raw measurements to an R-D-E-shaped CSV:

```powershell
python -m src.router.external_codec_rde_exporter `
  --spec configs/external_codecs/example_image_codec.json `
  --raw-csv results/external_codec_benchmarks/example_codec_measurements.csv `
  --out results/external_codec_benchmarks/example_codec_rde.csv `
  --report-out results/external_codec_benchmarks/example_codec_rde_report.json `
  --quality-csv results/external_codec_benchmarks/example_codec_quality.csv `
  --quality-column ssimulacra2 `
  --rate-mode image_bpp
```

Router v0.40.0's exporter is offline only. It does not execute codecs. It reads
the v0.39 raw CSV, excludes failed raw rows from valid R-D-E output while
counting them in the report, optionally joins quality values by
`input_id,param_set_id`, and writes a standardized CSV with:

```text
codec,param,input_id,input_path,rate,quality,energy,time_ms,
energy_provenance_tier,measurement_provenance,success,error,source_raw_csv
```

If quality is missing, the exporter leaves `quality` empty and reports
`quality_available=false`. If energy is missing, it leaves `energy` empty,
uses `energy_provenance_tier=unknown`, and reports `energy_available=false`.
`router_ready` is true only when the exported CSV is structurally valid and
has the needed quality and energy fields populated.

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

Router v0.37.0 adds `external_codec_probe`, a controlled availability probe for
specs that have already passed schema validation. For `runtime.type =
external_command`, the probe checks the explicitly declared
`runtime.executable`, computes its SHA256 when present, and may run only the
declared `version_probe.command` or `version_probe.command_template` with
`shell=False` and a timeout. Use `--no-version-probe` to restrict the probe to
existence plus fingerprinting.

For `runtime.type = python_module`, the probe does not import the module by
default. It reports availability as `unknown` with a provenance warning until a
future release introduces an explicit import/probe mode.

The probe report includes safety flags:

```json
{
  "safety": {
    "shell_used": false,
    "encode_executed": false,
    "decode_executed": false,
    "benchmark_executed": false
  }
}
```

The probe never executes `encode.command_template` or `decode.command_template`.

Router v0.38.0 adds `external_codec_dry_run`, a single-input contract check.
It may execute encode and optional decode commands, but only from argv
templates, only with `shell=False`, only with a required timeout, and only with
outputs generated inside the explicitly provided `--out-dir`. It rejects
undeclared parameters, unresolved placeholders, unsafe output filenames,
unsafe output extensions and missing input files.

The dry-run report includes safety flags:

```json
{
  "safety": {
    "shell_used": false,
    "benchmark_executed": false,
    "router_candidate_registered": false,
    "output_confined_to_out_dir": true
  }
}
```

This is still not a benchmark or a router integration path.

Router v0.39.0 adds `external_codec_benchmark`, a small raw-measurement runner.
It uses the same argv-only, `shell=False`, timeout-bound and output-confined
execution model as the dry-run. Inputs must be named explicitly with repeated
`--input`; there is no automatic dataset or codec discovery. Parameters must
either come from the spec's declared values or be passed as repeated
`--param-set` JSON objects containing only declared parameter names.

Router v0.40.0 adds `external_codec_rde_exporter`, an offline transformer from
the raw benchmark CSV to a future-router-compatible R-D-E-shaped CSV. It never
executes subprocesses, encode, decode or probes; it only reads declared files
and writes export artifacts.

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
    "executable": "example-codec",
    "max_runtime_seconds": 30
  },
  "version_probe": {
    "command": ["{executable}", "--version"]
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
    "available": false,
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
    "extension": ".exi",
    "must_be_nonempty": true
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
