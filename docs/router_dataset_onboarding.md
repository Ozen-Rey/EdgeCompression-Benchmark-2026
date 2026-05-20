# How to add a new dataset

This guide shows the practical v0.44.4 workflow for adding a dataset without
editing Python code. It uses three small tools:

- `rde-dataset-manifest`: create and validate a dataset manifest.
- `rde-dataset-ingest`: create measurement templates and ingest measured rows.
- `rde-dataset-onboard`: run the full validation + ingestion + router check.

The workflow is offline. It does not run benchmarks, compress media, probe
codecs, or change router ranking.

## 1. Choose a DomainSpec

Pick the R-D-E schema that matches your measured quality metric:

- images with SSIMULACRA2: `image_ssimulacra2`
- audio with ViSQOL: `audio_visqol`
- video with VMAF: `video_vmaf`

Inspect a spec:

```bash
python -m src.router.core.domain_spec --builtin audio_visqol --print-json
```

## 2. Create a DatasetManifest

Start from a template:

```bash
python -m src.router.core.dataset_manifest \
  --new-template image \
  --out configs/datasets/my_images.json
```

Audio and video:

```bash
python -m src.router.core.dataset_manifest \
  --new-template audio \
  --out configs/datasets/my_audio.json

python -m src.router.core.dataset_manifest \
  --new-template video \
  --out configs/datasets/my_video.json
```

Edit item ids, paths, splits, and metadata. Validate without requiring files:

```bash
python -m src.router.core.dataset_manifest \
  --manifest configs/datasets/my_images.json \
  --validate
```

Use `--check-files` only when files exist locally.

## 3. Create or export measurements CSV

Generate a starter CSV:

```bash
python -m src.router.core.dataset_ingestion \
  --new-measurements-template image_ssimulacra2 \
  --out measurements_template.csv
```

Audio and video:

```bash
python -m src.router.core.dataset_ingestion \
  --new-measurements-template audio_visqol \
  --out audio_measurements_template.csv

python -m src.router.core.dataset_ingestion \
  --new-measurements-template video_vmaf \
  --out video_measurements_template.csv
```

Replace example rows with measured codec rows. The item id column must match
the manifest item ids.

## 4. Run ingestion

Image example:

```bash
python -m src.router.core.dataset_ingestion \
  --manifest configs/datasets/my_images.json \
  --measurements-csv measurements_template.csv \
  --domain-spec image_ssimulacra2 \
  --item-id-col image_id \
  --codec-col codec \
  --config-col config \
  --rate-col bpp \
  --quality-col ssimulacra2 \
  --energy-col energy_per_image_j \
  --time-col time_ms \
  --out-csv .pytest_tmp_onboard/my_images_rde.csv \
  --report-out .pytest_tmp_onboard/my_images_ingestion_report.json
```

Audio example:

```bash
python -m src.router.core.dataset_ingestion \
  --manifest configs/datasets/my_audio.json \
  --measurements-csv audio_measurements_template.csv \
  --domain-spec audio_visqol \
  --item-id-col item_id \
  --codec-col codec \
  --config-col param \
  --rate-col bitrate_kbps \
  --quality-col visqol \
  --energy-col energy_j_per_second \
  --time-col time_ms \
  --out-csv .pytest_tmp_onboard/my_audio_rde.csv \
  --report-out .pytest_tmp_onboard/my_audio_ingestion_report.json
```

Video example:

```bash
python -m src.router.core.dataset_ingestion \
  --manifest configs/datasets/my_video.json \
  --measurements-csv video_measurements_template.csv \
  --domain-spec video_vmaf \
  --item-id-col sequence \
  --codec-col codec \
  --config-col param \
  --rate-col bitrate_kbps \
  --quality-col vmaf \
  --energy-col energy_kj_per_sequence \
  --time-col time_ms \
  --out-csv .pytest_tmp_onboard/my_video_rde.csv \
  --report-out .pytest_tmp_onboard/my_video_ingestion_report.json
```

## 5. Validate the R-D-E CSV

```bash
python -m src.router.core.domain_spec \
  --csv .pytest_tmp_onboard/my_audio_rde.csv \
  --builtin audio_visqol \
  --validate-csv
```

## 6. Run the router

```bash
python -m src.router.rde_router \
  --csv .pytest_tmp_onboard/my_audio_rde.csv \
  --domain-spec audio_visqol \
  --profile balanced \
  --out .pytest_tmp_onboard/router_report.json \
  --summary-out .pytest_tmp_onboard/router_summary.csv
```

## 7. One-command onboarding check

Use the onboarding CLI to run validation, ingestion, DomainSpec validation, and
a dry router decision in sequence:

```bash
python -m src.router.core.dataset_onboarding \
  --manifest configs/datasets/my_audio.json \
  --measurements-csv audio_measurements_template.csv \
  --domain-spec audio_visqol \
  --work-dir .pytest_tmp_onboard \
  --item-id-col item_id \
  --codec-col codec \
  --config-col param \
  --rate-col bitrate_kbps \
  --quality-col visqol \
  --energy-col energy_j_per_second \
  --time-col time_ms \
  --router-profile balanced
```

The work directory receives:

- `onboarding_report.json`
- `ingested_rde.csv`
- `router_report.json`
- `router_summary.csv`

## 8. Interpret the report

`onboarding_report.json` contains:

- `manifest_valid`
- `measurements_valid`
- `ingestion_valid`
- `domain_spec_valid`
- `router_decision_valid`
- `selected_codec`
- `selected_config`
- `warnings`
- `errors`

The workflow is ready when all step booleans are true and `errors` is empty.

## Adding a new codec to a new dataset

For a measured-only codec, add the codec rows directly to the measurements CSV.
The router does not need codec-specific Python code; it needs valid R-D-E rows.

Validate the codec measurements first:

```bash
python -m src.router.core.codec_onboarding \
  --measurements-csv audio_measurements_template.csv \
  --domain-spec audio_visqol \
  --codec-col codec \
  --config-col param \
  --rate-col bitrate_kbps \
  --quality-col visqol \
  --energy-col energy_j_per_second \
  --report-out .pytest_tmp_onboard/codec_report.json
```

Then run ingestion and the onboarding workflow as usual. If you also have an
external codec spec, pass it to the one-command workflow:

```bash
python -m src.router.core.dataset_onboarding \
  --manifest configs/datasets/my_images.json \
  --measurements-csv measurements_template.csv \
  --domain-spec image_ssimulacra2 \
  --work-dir .pytest_tmp_onboard \
  --item-id-col image_id \
  --codec-col codec \
  --config-col config \
  --rate-col bpp \
  --quality-col ssimulacra2 \
  --energy-col energy_per_image_j \
  --time-col time_ms \
  --codec-spec tests/fixtures/external_codec_image_spec.json
```

The onboarding report includes a `codec_onboarding` block when `--codec-spec`
is provided. See `docs/router_codec_onboarding.md` for the codec-specific
workflow.

## Common errors

`missing_measurement_columns:<name>` means the CSV does not contain a column
named by the CLI mapping. Fix the mapping or rename the CSV column.

`unknown_measurement_items:<id>` means a measured item id is not present in the
manifest. Add the item to the manifest or remove the measurement row.

`manifest_items_without_measurements:<id>` means a manifest item has no
measured rows. This is a warning during ingestion, but it may indicate an
incomplete campaign.

`non_numeric_quality`, `non_numeric_rate`, or `non_numeric_energy` means the
router cannot use one or more measured values. Do not invent missing metrics;
repair the measurements or remove the invalid row.
