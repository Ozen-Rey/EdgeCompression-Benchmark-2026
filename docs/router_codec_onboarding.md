# Router Codec Onboarding

v0.44.5 closes the codec side of the pluggable dataset workflow. A researcher
can introduce a codec by providing valid R-D-E rows; the router does not need
to know the codec internals.

## Three Levels

### 1. Codec measured-only

This is the minimum path. Provide a measurements CSV with codec, config, rate,
quality, and energy columns. The codec can be a new implementation, a manual
measurement, or an external tool. If the rows match a `DomainSpec`, ingestion
can turn them into router-ready R-D-E CSV rows.

Validate measured rows:

```bash
python -m src.router.core.codec_onboarding \
  --measurements-csv tests/fixtures/measurements_new_codec_audio.csv \
  --domain-spec audio_visqol \
  --codec-col codec \
  --config-col param \
  --rate-col bitrate_kbps \
  --quality-col visqol \
  --energy-col energy_j_per_second \
  --report-out .pytest_tmp_codec/audio_codec_report.json
```

### 2. External codec declared

If the codec can be executed by the external codec pipeline, provide an
external codec spec JSON. The existing probe, dry-run, benchmark, and export
tools can use that spec when compatible. v0.44.5 adds a lightweight domain
compatibility check between the external spec and `DomainSpec`.

```bash
python -m src.router.core.codec_onboarding \
  --codec-spec tests/fixtures/external_codec_image_spec.json \
  --domain-spec image_ssimulacra2 \
  --validate-domain \
  --report-out .pytest_tmp_codec/image_codec_spec_report.json
```

### 3. Router-consumable codec

The router consumes R-D-E rows, not codec source code. A codec becomes
router-consumable when measured or exported rows are:

1. mapped through `DatasetManifest` + measurements ingestion;
2. validated against `DomainSpec`;
3. passed to `src.router.rde_router --domain-spec`.

The selected codec/config is then decided by the existing `J_RDE` ranking. The
ranking formula and runtime decision logic are unchanged.

## Template

Create a measurements template for a new codec:

```bash
python -m src.router.core.codec_onboarding \
  --new-codec-measurements-template video_vmaf \
  --out new_video_codec_measurements.csv
```

The template uses the same structure as dataset ingestion measurements:
dataset, item id, codec, config, rate, quality, energy, and `time_ms`.

## Adding a new codec to a new dataset

1. Choose a `DomainSpec`.
2. Write or generate a `DatasetManifest`.
3. Add measured rows for your codec to the measurements CSV.
4. Validate codec measurements with `codec_onboarding`.
5. Ingest manifest + measurements into R-D-E CSV.
6. Validate the R-D-E CSV with `domain_spec`.
7. Run the router with `--domain-spec`.
8. Optionally provide an external codec spec for provenance and future
   probe/benchmark/export workflows.

The key invariant is simple: the router only needs valid, domain-consistent
R-D-E rows.
