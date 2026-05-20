# Router Dataset Ingestion

v0.44.3 adds a manifest-driven ingestion step that turns measured rows into a
router-ready R-D-E CSV without changing Python code.

This remains an offline data preparation tool. It does not benchmark media,
compress files, probe codecs, measure energy, change `J_RDE`, or alter router
decision logic.

## Three Layers

`DomainSpec` describes how to interpret R-D-E columns: dataset, codec, config,
rate, quality, energy, item id, units, and quality direction.

`DatasetManifest` describes the dataset itself: media item ids, paths, splits,
and media metadata such as image dimensions, audio duration, or video frame
rate.

`DatasetIngestion` joins a manifest with a measurements CSV and writes a
normalized R-D-E CSV that the router can read with `--domain-spec`.

## Workflow

1. Write a dataset manifest.
2. Validate the manifest.
3. Produce or collect a measurements CSV.
4. Ingest manifest + measurements into an R-D-E CSV.
5. Validate the output with `DomainSpec`.
6. Run the router on the output CSV.

## Example

```bash
python -m src.router.core.dataset_ingestion \
  --manifest configs/datasets/example_audio_dataset.json \
  --measurements-csv tests/fixtures/measurements_audio_manifest_example.csv \
  --domain-spec audio_visqol \
  --item-id-col item_id \
  --codec-col codec \
  --config-col param \
  --rate-col bitrate_kbps \
  --quality-col visqol \
  --energy-col energy_j_per_second \
  --time-col time_ms \
  --out-csv .pytest_tmp_ingest/example_audio_rde.csv \
  --report-out .pytest_tmp_ingest/example_audio_report.json
```

Then validate and route:

```bash
python -m src.router.core.domain_spec \
  --csv .pytest_tmp_ingest/example_audio_rde.csv \
  --builtin audio_visqol \
  --validate-csv

python -m src.router.rde_router \
  --csv .pytest_tmp_ingest/example_audio_rde.csv \
  --domain-spec audio_visqol \
  --out .pytest_tmp_ingest/audio_router_report.json
```

## Join Semantics

`--item-id-col` links measurements to manifest items. Unknown measurement
items are errors when `--strict true` is used. With `--strict false`, they are
reported as warnings and dropped before the R-D-E CSV is written.

Manifest items without measurements are warnings, not errors. This allows a
researcher to ingest partial measurement campaigns while keeping the manifest
complete.

If `--dataset-col` is not provided, the output dataset column is filled with
`manifest.dataset_id`.

## Output

The output CSV contains the columns required by the selected `DomainSpec`, plus
available manifest metadata:

- image/video: `width`, `height`, `pixels`
- audio: `duration_s`, `sample_rate`, `channels`
- video: `fps`, `num_frames`
- item metadata as `metadata_*` columns

Missing or non-numeric rate, quality, or energy values are not invented. They
are kept in the output and reported as invalid rows in the JSON report.

## Report

The ingestion report records:

- manifest and domain spec used
- input measurement columns
- output CSV path
- manifest item count and measurement row count
- joined, valid, and invalid R-D-E row counts
- missing manifest items
- unknown measurement items
- column mapping
- numeric validity summary
- warnings and errors

This report is intended to make dataset ingestion auditable before any router
decision is made.

## Onboarding Workflow

v0.44.4 adds `src.router.core.dataset_onboarding`, which runs the practical
researcher workflow in one command: manifest validation, measurement column
checks, ingestion, DomainSpec validation, and a router dry decision. It writes
`onboarding_report.json`, `ingested_rde.csv`, `router_report.json`, and
`router_summary.csv` to a chosen work directory.

See `docs/router_dataset_onboarding.md` for the step-by-step guide.

## Codec Measurements

v0.44.5 adds `src.router.core.codec_onboarding` for measured-only codec rows
and optional external codec specs. Dataset ingestion does not need to know how
the codec is implemented; it only maps codec/config/rate/quality/energy
measurements into the selected `DomainSpec`. See
`docs/router_codec_onboarding.md` for the codec workflow.
