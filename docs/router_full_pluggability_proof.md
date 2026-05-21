# Router Full Pluggability Proof

v0.44.6 adds a documented proof that the multi-domain onboarding pieces fit
together without requiring router code changes. v0.44.6.1 hardens that proof
with a static sample report, an explicit boundary statement, and an image smoke
test that uses the real benchmark codec name `JPEG`. v0.44.6.2 identifies the
onboarding proof contract as `rde_manifest_v1` and propagates that identifier to
the generated onboarding reports and the committed sample report.

The proof covers this path:

1. A dataset is described by a `DatasetManifest`.
2. A new codec or model appears only as rows in a measurements CSV.
3. A `DomainSpec` explains how to interpret rate, quality, energy and identity
   columns.
4. Dataset ingestion writes a router-ready R-D-E CSV.
5. The router consumes that CSV with `--domain-spec` and produces a decision.

This is intentionally a proof, not a new routing feature. It does not change
`J_RDE`, ranking, admissibility, execution backends, benchmark data, external
codec execution, or historical results.

## What The Test Proves

`tests/test_full_pluggability_proof.py` runs one end-to-end proof over the
available image, audio and video fixtures. For each domain it checks:

- the dataset manifest is valid;
- the new codec measurements are valid against the selected `DomainSpec`;
- ingestion succeeds and writes an R-D-E CSV;
- the generated R-D-E CSV validates against the `DomainSpec`;
- the router produces a valid dry decision;
- selected codec and config are present;
- the new codec is present in the ingested R-D-E CSV;
- the new codec is present in the router candidate pool.

The test writes an example aggregate report to the pytest temporary directory:

```text
full_pluggability_proof_report.json
```

That report records the domain, `DomainSpec`, new codec id, validation flags,
selected codec/config and the generated artifact paths for each proof case.
From v0.44.6.2 onward it also records `contract_id: rde_manifest_v1`, the
stable identifier for the manifest + measurements CSV + DomainSpec -> validated
R-D-E CSV -> router decision contract.

A static reference copy is committed at
`docs/examples/full_pluggability_proof_report.example.json`. It is only a
documentation artifact for reading and review: the runtime never reads it, tests
do not treat it as a source of truth, and fresh pytest runs still generate their
own equivalent report in `tmp_path`.

v0.44.6.1 also adds `test_full_pluggability_proof_with_real_jpeg_codec`, which
uses the existing image manifest and a small JPEG/JXL/AVIF measurements fixture.
That smoke proves the image path works for a real codec name already used by
the benchmark, not only for synthetic codec ids.

## Proof Fixtures

The proof reuses the pluggability fixtures introduced in the v0.44.x line:

- image: `configs/datasets/example_image_dataset.json` +
  `tests/fixtures/measurements_new_codec_image.csv` +
  `image_ssimulacra2`;
- audio: `configs/datasets/example_audio_dataset.json` +
  `tests/fixtures/measurements_new_codec_audio.csv` +
  `audio_visqol`;
- video: `configs/datasets/example_video_dataset.json` +
  `tests/fixtures/measurements_new_codec_video.csv` +
  `video_vmaf`.

The codec names live only in CSV rows:

- `example_neural_image_codec`;
- `example_audio_codec`;
- `example_video_codec`.

The router does not import codec-specific Python code for these codecs. It
only consumes validated R-D-E rows.

## Reproduce The Proof

Run the focused test:

```bash
python -m pytest -q tests/test_full_pluggability_proof.py --basetemp ".pytest_tmp_refactor"
```

Run the full suite:

```bash
python -m pytest -q --basetemp ".pytest_tmp_refactor"
```

The temporary proof report is generated under pytest's temp directory during
the test run. It is not committed because it is a generated example artifact.

## Boundary Of The Proof

This proof does not validate the scientific correctness of the measurements
themselves. A misreported energy value in the CSV would still pass ingestion if
it satisfies the `DomainSpec` schema. Scientific validation of measurements
remains the responsibility of the benchmarking pipeline and the measurement
protocol.

Ingestion validates structure, required columns, numeric convertibility and
domain compatibility. It does not prove that a codec implementation is correct,
measure energy, execute arbitrary models automatically, or certify that the
reported R-D-E values are physically accurate.

v0.44.6, v0.44.6.1 and v0.44.6.2 do not benchmark real audio or video codecs,
run external codec executables, add automatic dataset discovery, or add a
router plugin API. Those remain future work. The release proves the current
contract:

```text
manifest + measurements CSV + DomainSpec -> validated R-D-E CSV -> router decision
```

As long as a researcher can provide a valid manifest and measured R-D-E rows, a
new dataset and a new codec/model can enter the router workflow without editing
router Python code.
