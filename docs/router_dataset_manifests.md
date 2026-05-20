# Router Dataset Manifests

v0.44.1 introduces `DatasetManifest`, a JSON format for describing datasets
that future image, audio, and video ingestion tools can consume without
requiring Python code changes.

This is a schema and validation release only. It does not benchmark files,
compress files, generate R-D-E rows, change `J_RDE`, alter router ranking, or
integrate datasets automatically into the runtime router.

## DomainSpec Vs DatasetManifest

`DomainSpec` describes an R-D-E table: which columns contain rate, quality,
energy, codec, config, item identity, and which units and quality direction
apply.

`DatasetManifest` describes source media items: where the files live, what
their stable item ids are, which split each item belongs to, and which media
metadata are available. It prepares a dataset for future benchmarking or
ingestion, but it is not itself an R-D-E result table.

## Required Fields

Each manifest is a JSON object with:

- `schema_version`: currently `dataset_manifest_v1`.
- `dataset_id`: lowercase slug such as `kodak`, `example_audio`, or
  `ugc_video_set`.
- `domain`: one of `image`, `audio`, or `video`.
- `root`: dataset root directory.
- `items`: list of media items.
- `splits`: mapping from split name to item ids.

Each item requires:

- `item_id`: stable id unique within the dataset.
- `path`: path relative to `root`.

Paths must be relative child paths. Absolute paths and paths containing `..`
are rejected by the validator.

## Recommended Fields

The validator emits warnings, not errors, for recommended metadata:

- image: `width`, `height`; `pixels` is computed when both are present.
- audio: `duration_s`, `sample_rate`, `channels`.
- video: `width`, `height`, `fps`, `num_frames`.

Dataset-level `metadata`, `license`, and `source_url` are also recommended.
Missing values produce warnings so early manifests can be drafted before all
curation details are final.

## Example Image Manifest

```json
{
  "schema_version": "dataset_manifest_v1",
  "dataset_id": "example_images",
  "display_name": "Example Image Dataset",
  "domain": "image",
  "root": "datasets/example_images",
  "items": [
    {
      "item_id": "img001",
      "path": "img001.png",
      "width": 768,
      "height": 512,
      "metadata": {"source": "example"}
    }
  ],
  "splits": {
    "all": ["img001"],
    "test": ["img001"]
  }
}
```

The repository includes small placeholder examples:

- `configs/datasets/example_image_dataset.json`
- `configs/datasets/example_audio_dataset.json`
- `configs/datasets/example_video_dataset.json`

Their paths are intentionally synthetic. They validate with file checks off,
and fail file checks unless matching files are created under the declared
roots.

## Validation

Validate structure only:

```bash
python -m src.router.core.dataset_manifest \
  --manifest configs/datasets/example_image_dataset.json \
  --validate
```

Validate structure and file presence:

```bash
python -m src.router.core.dataset_manifest \
  --manifest configs/datasets/example_image_dataset.json \
  --validate \
  --check-files
```

Validate manifest domain against a built-in `DomainSpec`:

```bash
python -m src.router.core.dataset_manifest \
  --manifest configs/datasets/example_image_dataset.json \
  --domain-spec image_ssimulacra2 \
  --validate-domain
```

The JSON report contains `valid`, `errors`, `warnings`, item and split counts,
domain, root, missing file count, metadata summary, and the normalized
manifest.

## Exporting An Item Table

For future ingestion and quick inspection, export a flat item table:

```bash
python -m src.router.core.dataset_manifest \
  --manifest configs/datasets/example_image_dataset.json \
  --to-csv items.csv
```

The CSV includes `dataset_id`, `domain`, `root`, `item_id`, `path`, media
metadata columns, `metadata_json`, and flattened `metadata_*` columns.

## Adding A Dataset

To add a dataset:

1. Create a manifest JSON under an appropriate config or experiment folder.
2. Choose a stable `dataset_id` slug.
3. Use paths relative to `root`.
4. Add stable item ids and split membership.
5. Add recommended media metadata for the domain.
6. Run validation without `--check-files` while drafting.
7. Run validation with `--check-files` when the files are available locally.

This release stops at description and validation. Benchmarking, compression,
codec probing, energy measurement, and R-D-E CSV generation remain separate
steps.
