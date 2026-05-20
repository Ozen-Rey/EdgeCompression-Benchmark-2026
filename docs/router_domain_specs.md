# Router Domain Specifications

v0.44.0 introduces `DomainSpec`, a small schema layer for describing the
columns and units used by R-D-E CSVs across image, audio, and video domains.
It is a foundation release: it does not change `J_RDE`, the core ranking
logic, benchmark data, execution backends, external codec handling, or
historical reports.

## Why Domain Specs Exist

The router started with image experiments, where the common shape was:

- quality: SSIMULACRA2 or PSNR
- rate: bpp
- energy: J/image

Audio and video use the same R-D-E idea but different units and quality
contracts. A domain specification makes those assumptions explicit before the
CSV enters the router or an offline analysis. The core still sees normalized
rate, distortion, and energy values as before; `DomainSpec` only documents and
validates which input columns mean those things.

## Built-In Domains

The built-in registry lives in `src/router/core/domain_spec.py`:

- `image_ssimulacra2`: `bpp`, `ssimulacra2`, `energy_per_image_j`, `J/image`
- `image_psnr`: `bpp`, `psnr`, `energy_per_image_j`, `J/image`
- `video_vmaf`: `bitrate_kbps`, `vmaf`, `energy_kj_per_sequence`, `kJ/sequence`
- `video_psnr_y`: `bitrate_kbps`, `psnr_y`, `energy_j_per_frame`, `J/frame`
- `audio_visqol`: `bitrate_kbps`, `visqol`, `energy_j_per_second`, `J/s`
- `audio_fad`: `bitrate_kbps`, `fad`, `energy_j_per_second`, `J/s`

These names are intentionally metric-specific. A video CSV with VMAF and a
video CSV with PSNR-Y may both be video-domain artifacts, but they have
different quality columns and may use different quality floors.

## Units By Domain

Image schemas normally use `bpp` for rate and `J/image` for energy. The item
identity is usually an image id, optionally grouped by dataset.

Audio schemas normally use `kbps` or `bps` for rate and `J/s` for energy.
The item identity is usually a clip id. Metrics such as ViSQOL and PESQ are
typically higher-is-better, while FAD is lower-is-better.

Video schemas normally use `kbps` or `Mbps` for rate. Energy can be reported
as `J/sequence`, `kJ/sequence`, or `J/frame`, depending on the measurement
protocol. The item identity is usually a sequence id.

## Quality Direction

`quality_direction` tells validation and future adapters how to interpret the
metric:

- `higher_is_better`: SSIMULACRA2, PSNR, VMAF, ViSQOL, PESQ.
- `lower_is_better`: FAD and other distance/error metrics.

`distortion_transform` records how a quality metric should be treated when
converted into a distortion-like quantity. For the current v0.44.0 foundation,
this is metadata and validation context only; the router ranking formula is
unchanged.

## Example JSON

```json
{
  "domain": "audio",
  "item_id_columns": ["clip_id"],
  "dataset_column": "dataset",
  "codec_column": "codec",
  "config_column": "config",
  "rate_column": "bitrate_kbps",
  "quality_column": "visqol",
  "energy_column": "energy_j_per_second",
  "time_column": null,
  "rate_unit": "kbps",
  "quality_metric": "ViSQOL",
  "quality_direction": "higher_is_better",
  "energy_unit": "J/s",
  "default_quality_floor": null,
  "default_near_quality_floor": null,
  "distortion_transform": "negate_quality",
  "notes": "Example audio schema."
}
```

For an alternate column name, copy a built-in spec, change the relevant
`*_column` field, and validate the JSON before using it for CSV validation.

## CLI

List built-ins:

```bash
python -m src.router.core.domain_spec --list-builtins
```

Print a built-in as JSON:

```bash
python -m src.router.core.domain_spec --builtin image_ssimulacra2 --print-json
```

Validate a custom spec:

```bash
python -m src.router.core.domain_spec --spec path/to/spec.json --validate
```

Validate a CSV against a built-in:

```bash
python -m src.router.core.domain_spec \
  --csv path/to/rde.csv \
  --builtin video_vmaf \
  --validate-csv
```

The CSV validation report is JSON and includes:

- `valid`
- `errors`
- `warnings`
- `normalized_spec`
- `detected_columns`
- `num_rows`
- `numeric_validity_summary`

## Adding A New Domain Or Metric

To add a new schema:

1. Choose the domain and metric-specific name, for example `audio_pesq`.
2. Define item identity columns, dataset, codec, config, rate, quality, energy,
   and optional time columns.
3. Specify units, especially rate and energy units.
4. Set `quality_direction`.
5. Choose a `distortion_transform` that matches the direction.
6. Add a fixture CSV and validation tests before wiring it into runtime paths.

Runtime integration should stay conservative: validate the schema offline
first, then use it to supply column defaults only when the CSV actually
contains those columns.

## Relationship To Dataset Manifests

v0.44.1 adds `DatasetManifest` in `src/router/core/dataset_manifest.py`.
Use `DatasetManifest` to describe source media files, item ids, splits, and
media metadata. Use `DomainSpec` to describe the resulting R-D-E table columns,
metrics, directions, and units. A dataset manifest does not contain codec
results; a domain spec does not enumerate source files.

## Multi-Domain Router Smoke

v0.44.2 adds fixture-level router smoke coverage for audio and video R-D-E
CSVs. This does not run real audio/video benchmarks and does not compress
media. It demonstrates that already measured CSVs can be interpreted through
`DomainSpec` without assuming image-only columns such as `bpp`,
`ssimulacra2`, or `energy_per_image_j`.

Validate audio and video fixture CSVs:

```bash
python -m src.router.core.domain_spec \
  --csv tests/fixtures/rde_audio_visqol.csv \
  --builtin audio_visqol \
  --validate-csv

python -m src.router.core.domain_spec \
  --csv tests/fixtures/rde_video_vmaf.csv \
  --builtin video_vmaf \
  --validate-csv
```

Run offline router decisions on measured fixture rows:

```bash
python -m src.router.rde_router \
  --csv tests/fixtures/rde_audio_visqol.csv \
  --domain-spec audio_visqol \
  --profile balanced \
  --out tmp/audio_report.json \
  --summary-out tmp/audio_summary.csv

python -m src.router.rde_router \
  --csv tests/fixtures/rde_video_vmaf.csv \
  --domain-spec video_vmaf \
  --profile balanced \
  --out tmp/video_report.json \
  --summary-out tmp/video_summary.csv
```

Lower-is-better metrics such as `audio_fad` are validated at the schema level
in v0.44.2. Runtime ranking remains higher-is-better and FAD runtime support is
left for a future compatibility pass.
