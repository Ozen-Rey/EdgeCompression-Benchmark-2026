# Kodak Image Mini-Benchmark Validation

## Scope

This script runs a local image-only R-D-E mini-benchmark on Kodak. It is a
validation/sanity benchmark, not a replacement for the main thesis benchmark.

It uses Kodak, a small codec subset, and a small set of operating points to
measure:

- rate: `bpp`;
- quality: `PSNR` (default, numpy-only) or `SSIMULACRA2` (`--quality-metric`);
- energy: `J/image`, when a local backend is available;
- time: `ms/image`.

It intentionally does not run video, audio, or the full metric zoo used by the
Chapter 4 benchmark pipeline.

## Quickstart (one command)

To set up the benchmark stack and produce a router-ready CSV on the spot, use
the dedicated benchmark setup helper. It is **separate from the router-only
setup** (`setup_router.py`): the router setup stays minimal, while this prepares
everything the image mini-benchmark needs.

Windows PowerShell:

```powershell
.\setup_benchmark.ps1 --yes
```

Linux / macOS:

```bash
./setup_benchmark.sh --yes
```

With confirmation (default `No`; `--yes` auto-accepts) it creates/uses a venv,
installs `pip install -e ".[benchmark]"` (numpy, imagecodecs, matplotlib),
downloads the 24 Kodak PNGs locally, then runs the mini-benchmark with the
numpy-only PSNR metric and replays the R-D-E router so the result is testable
immediately. The router-ready CSV lands in
`validation_runs/benchmark_quickstart/`.

Useful flags: `--dry-run`, `--no-venv`, `--skip-benchmark` (set up only),
`--codecs jpeg,jxl,hevc`, `--quality-metric {psnr,ssimulacra2}`,
`--max-images N`, `--no-router`, `--energy-proxy {off,time}`.

## Quality metric

The default metric is **PSNR**, computed with numpy only, so the benchmark runs
out of the box. Pass `--quality-metric ssimulacra2` to use SSIMULACRA2 instead;
the `ssimulacra2` package (a small pure-Python wheel) is installed by the
`benchmark` extra, so it works out of the box after the benchmark setup. PSNR
remains the default because it has no extra dependency and still works if that
import ever fails. Each metric maps to the matching built-in router domain spec
(`image_psnr` / `image_ssimulacra2`). Because the default router quality
thresholds are SSIMULACRA2-scaled (50/80/90), the PSNR router replay uses
`configs/quality_thresholds_psnr.json` (dB-scaled) so decisions are feasible.

## Energy provenance and the time proxy

Router-valid rows need an energy value. On hosts without RAPL/NVML telemetry no
energy is measured, so the router cannot rank. `--energy-proxy time` fills
missing energy with a time-proportional proxy at a fixed reference power and
labels every such row `energy_provenance=time_proxy_non_measured`. This is
**not a measurement**; it only lets the router be exercised end-to-end where no
telemetry exists. It is `off` by default in the mini-benchmark (so real
measurement runs are never polluted) and `time` by default in
`setup_benchmark.py` so the on-the-spot router replay works on any machine.

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
