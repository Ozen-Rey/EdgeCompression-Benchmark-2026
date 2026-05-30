# Kodak Image Mini-Benchmark Validation

## Scope

This script runs a local image-only R-D-E mini-benchmark on Kodak. It is a
validation/sanity benchmark, not a replacement for the main thesis benchmark.

It uses Kodak, a small codec subset, and a small set of operating points to
measure:

- rate: `bpp`;
- quality: `SSIMULACRA2`;
- energy: `J/image`, when a local backend is available;
- time: `ms/image`.

It intentionally does not run video, audio, or the full metric zoo used by the
Chapter 4 benchmark pipeline.

## Why Kodak

Kodak is small, historical, and controllable. Its 24 images make it useful for a
cross-machine sanity check without turning validation into a full benchmark
campaign.

## Codecs

The validation target is:

- JPEG;
- JPEG XL;
- HEVC intra / x265;
- DCAE, when the existing local DCAE code and checkpoints are available.

DCAE rows are not simulated. If the local DCAE root or checkpoint is missing,
the report marks it unavailable and the script continues unless `--strict` is
set.

## Outputs

The script writes local artifacts under the selected `--out-dir`:

- `kodak_image_rde_mini_router_ready.csv`: router-schema CSV with `config`,
  `bpp`, `ssimulacra2`, `energy_per_image_j`, `time_ms`, status, and provenance
  aliases;
- `kodak_image_rde_mini_report.json`: command line, versions, platform/device
  information, energy and metric backend notes, skipped codecs, failures, and
  boundaries;
- `plots/*.png`: diagnostic rate/quality, energy/quality, rate/energy,
  time/quality, and codec energy plots unless `--skip-plots` is used;
- `router_*_report.json` and `router_*_summary.csv` when `--run-router` is used.

Generated outputs are local artifacts and should not be committed.

## How to run

Windows:

```bat
python scripts/validation/image_kodak_mini_benchmark.py ^
  --kodak-dir C:\path\to\kodak ^
  --out-dir validation_runs\image_kodak_mini_windows ^
  --codecs jpeg,jxl,hevc,dcae ^
  --run-router
```

Linux/Arch:

```bash
python scripts/validation/image_kodak_mini_benchmark.py \
  --kodak-dir /path/to/kodak \
  --out-dir validation_runs/image_kodak_mini_arch \
  --codecs jpeg,jxl,hevc,dcae \
  --run-router
```

For a pilot run:

```bash
python scripts/validation/image_kodak_mini_benchmark.py \
  --kodak-dir /path/to/kodak \
  --out-dir validation_runs/image_kodak_mini_pilot \
  --max-images 1 \
  --codecs jpeg \
  --energy-backend none
```

## Interpretation

Compare trends and pipeline behavior, not identical numeric values across
machines. Energy is platform-local and depends on the available backend,
hardware counters, drivers, and thermal/order effects.

Router decisions depend only on the measured R-D-E rows available in the CSV.
If energy or SSIMULACRA2 is unavailable, those rows are incomplete for router
replay and the report says so explicitly.

## Boundaries

- This is a local mini-benchmark validation, not the main thesis benchmark.
- It does not claim hardware-invariant energy measurements.
- It does not redistribute Kodak images, checkpoints, or codec binaries.
- It is intended to check whether the R-D-E measurement pipeline can be reapplied
  on a controlled small image set.
- Generated outputs are local artifacts and should not be committed.
- It does not establish a universal codec ranking.
