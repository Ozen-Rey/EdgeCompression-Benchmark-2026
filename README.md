# EdgeCompression-Benchmark-2026

Rate–Distortion–Energy (R-D-E) benchmark of classical and neural compression
codecs across image, audio, and video, with an adaptive R-D-E router and a
content-aware routing extension.

## Overview

This repository contains the code, configuration, documentation, and routing infrastructure
underlying the bachelor's thesis _"Compressione dei Dati: Un'Analisi
Comparativa tra Metodi Tradizionali e Approcci basati su Intelligenza
Artificiale"_ (Università degli Studi di Padova, AA 2025-2026).

The work extends the traditional Rate–Distortion evaluation with an explicit
Energy axis, profiles classical and neural codecs in a common operational
space, and uses the resulting R-D-E benchmark as the decision base for an
adaptive codec router.

## Codec coverage

Image (9 codecs / families):

- Classical: JPEG, JPEG XL, HEVC Intra
- Emerging standard: JPEG AI
- Neural: Ballé '18 (hyperprior), Cheng '20, ELIC '22, TCM '23, DCAE '25

Audio (5 codecs):

- Classical: Opus
- Neural: EnCodec, DAC, SNAC, WavTokenizer

Video (8 configurations, UVG LDP profile):

- Classical: x264, x265, SVT-AV1, VVenC
- Neural: DCVC-DC, DCVC-FM, DCVC-RT, DCVC-RT-CUDA

## Quality metrics

- Image: PSNR, SSIM, MS-SSIM, LPIPS, DISTS, FSIM, GMSD, VIF, HaarPSI, DSS,
  MDSI, SSIMULACRA 2
- Audio: ViSQOL, FAD, PESQ, STOI, SI-SDR, Mel distance
- Video: PSNR-Y, VMAF, MS-SSIM

## Hardware and energy

- GPU: NVIDIA GeForce RTX 5090 (Blackwell, 32 GB VRAM)
- CPU: AMD Ryzen 7 9800X3D
- 64 GB DDR5, NVMe PCIe
- OS for measurements: Ubuntu 24.04 LTS
- Energy: Zeus / NVML (GPU), RAPL (CPU); idle-subtracted, batch protocol;
  perceptual metrics are computed in a separate pass to keep the codec
  operational cost isolated.

All energy values reported in the thesis are direct hardware measurements.
The router additionally supports benchmark-derived and time-scaled energy
provenance for deployment scenarios without direct telemetry; see
`docs/router_limitations.md` for the provenance tiers.

## R-D-E benchmark

Per-domain benchmark scripts live under `src/benchmark/`:

- `src/benchmark/image_v2/` — image codec runners and metrics
- `src/benchmark/video/` — per-codec encode/decode and quality scripts
- `src/benchmark/benchmark_audio_*.py` — audio benchmark and ViSQOL / FAD

Benchmark outputs are generated locally under `results/`, `plots/`, and
`figures/`. Those directories are ignored in the public repository layout and
are not redistributed as part of the source license.

## R-D-E router

The router lives under `src/router/` with stable subpackages:

- `core/` — R-D-E primitives, points, selection, normalization profiles,
  quality threshold resolution
- `codecs/` — codec capability metadata, external codec spec / probe /
  dry-run / benchmark / export / registry
- `calibration/` — local calibration, calibration application, calibration
  bundles, calibration impact audit
- `adaptation/` — content, system, context and energy provenance policies
- `observability/` — decision receipts, replay, run manifest, feedback
  logging, shadow comparison, normalization consistency, router audits
- `analysis/` — offline paper-facing modules: policy comparison, content
  oracle / classifier sweeps, neural-inclusive oracle, operational regime
  simulation, content predictor interpretability

The router is fully integrated for the image domain. Routing for audio and
video is a planned extension that will reuse the same admissibility and
ranking infrastructure.

Console entrypoints after `pip install -e .`:

- `rde-router`
- `rde-external-codec-probe`, `rde-external-codec-dry-run`,
  `rde-external-codec-benchmark`, `rde-external-codec-export`
- `rde-decision-explain`
- `rde-legacy-import-audit`

PowerShell scenario dispatcher: `scripts/run_router.ps1`.

## Setup

There are two independent setup paths: a **benchmark setup** that produces a
router-ready CSV and exercises the router on the spot, and a minimal
**router-only setup**. Both are driven by cross-platform Python helpers, so the
exact same commands work on Windows, Linux, and macOS.

### Prerequisites

- **Git** and **Python ≥ 3.10** (with `pip` and the `venv` module). Nothing
  else is required to start.
- Use `python` or `python3`, whichever your system exposes (on many Linux/macOS
  systems it is `python3`). Check with `python --version`.
- The helpers never install system packages. If a prerequisite is missing they
  print OS-specific install hints and stop without changing anything.

### Quickstart: run the benchmark and test the router (any OS)

1. Get the repository:

   ```bash
   git clone <repository-url>
   cd EdgeCompression-Benchmark-2026
   ```

2. Run the benchmark setup. This single command is identical on every OS:

   ```bash
   python scripts/setup/setup_benchmark.py --yes
   ```

   With `--yes` it auto-accepts every step (omit it to confirm each one; the
   default answer is `No`). It creates a `.venv`, installs the benchmark extra
   (`pip install -e ".[benchmark]"`: numpy, imagecodecs, matplotlib,
   ssimulacra2), downloads the 24 Kodak PNG images locally, runs the image
   mini-benchmark with the numpy-only PSNR metric, and replays the R-D-E router
   on the produced CSV — all inside the created virtual environment.

3. Inspect the outputs under `validation_runs/benchmark_quickstart/`:

   - `kodak_image_rde_mini_router_ready.csv` — the router-ready R-D-E rows;
   - `kodak_image_rde_mini_report.json` — run provenance, backends, warnings;
   - `router_<profile>_summary.csv` / `router_<profile>_report.json` — the
     router decision per profile (balanced, energy-limited, bandwidth-limited,
     quality-first).

4. Verify it worked: the report JSON should show `router_valid_rows > 0` and a
   `router_replay` block with `returncode: 0` for each profile, and the
   `router_*_summary.csv` files should name a `selected_codec`.

Notes:

- It works **regardless of telemetry**: where no hardware energy backend
  (RAPL/NVML) is available, missing energy is filled with a labeled time proxy
  (`energy_provenance=time_proxy_non_measured`) so the router replay still runs.
  This is explicitly **not** a measurement.
- SSIMULACRA2 is also installed; use `python scripts/setup/setup_benchmark.py
  --quality-metric ssimulacra2 ...` to route on it instead of PSNR.
- Useful flags: `--dry-run` (preview commands, change nothing), `--no-venv`
  (use the current environment), `--skip-benchmark` (set up only),
  `--with-hevc` (also install the pip HEVC backend), `--codecs jpeg,jxl,hevc`,
  `--max-images N`.
- Optional convenience wrappers exist but the Python command above is the
  portable one. On Linux/macOS run wrappers as `bash setup_benchmark.sh ...`
  (they are not marked executable); on Windows use `.\setup_benchmark.ps1 ...`.
- JPEG and JPEG XL run through the `imagecodecs` Python wheel installed by the
  extra (it bundles its own libjxl), so they work out of the box with **no**
  system binaries. The optional `cjxl` / `djxl` binaries are only a fallback if
  `imagecodecs` ever lacks JPEG XL.
- HEVC needs ffmpeg with libx265. You can either use a system `ffmpeg`, or
  install it via pip with no admin rights by adding the optional `benchmark-hevc`
  extra (`pip install -e ".[benchmark,benchmark-hevc]"`, or
  `python scripts/setup/setup_benchmark.py --with-hevc`), which pulls
  `imageio-ffmpeg` — a wheel that bundles an ffmpeg binary built with libx265.
  That bundled ffmpeg is GPL (via x265); it is fetched by pip onto your machine
  and run as a separate process, so it does not relicense this project. DCAE
  needs torch plus checkpoints. When neither HEVC nor DCAE backends are present
  they are simply skipped, and the rest of the benchmark still runs. See
  `docs/image_kodak_mini_benchmark.md`.

### Router-only setup

If you only want a router development environment (no benchmark stack), use the
separate router-only helper:

```bash
python scripts/setup/setup_router.py
python scripts/setup/doctor.py --report-out environment_doctor_report.json
```

It may create a virtual environment and install `python -m pip install -e
".[test]"`, but only after an explicit prompt (`No` is the default). It does not
install external codecs, system packages, datasets, checkpoints, or benchmark
outputs. The read-only `doctor.py` reports the local environment without
changing it. External codec dependencies (`cjxl`, `ffmpeg`, `vvenc`,
`SvtAv1EncApp`, `opusenc`, etc.) remain separate; see `docs/external_codecs.md`
and `docs/router_setup.md`.

## Repository layout

```text
src/
  benchmark/       per-domain R-D-E benchmark scripts
  router/          R-D-E adaptive router (core / codecs / calibration /
                   adaptation / observability / analysis)
  selector/        baseline selector and content classifier
  utils/           energy monitor, thesis number extraction, plot helpers
  plots/           thesis plot generators
configs/           router configs, codec catalogs, quality thresholds
docs/              architecture, content-aware, feedback loop, limitations
results/           local generated benchmark CSV / JSON outputs (ignored)
figures/           local generated figures (ignored)
tests/             pytest suite (router, codecs, observability, analysis)
scripts/           PowerShell entrypoints for router scenarios
```

## Documentation

The `docs/` directory contains:

- `router_architecture.md` — package layout after the v0.42 refactor
- `router_content_aware.md` — content-aware routing pipeline
- `router_system_aware.md` — system-aware policies and penalties
- `router_decision_explainability.md` — offline decision explanation render
- `router_feedback_loop.md` — observational feedback in shadow mode
- `router_limitations.md` — energy provenance tiers and known limits
- `router_config_reference.md`, `router_usage.md`, `router_scripts.md`
- `external_codecs.md`, `developer_setup.md`

## Changelog

See `CHANGELOG.md`. The current stable artifact is router `v0.43.7`.
The architectural refactor line was frozen at `v0.42.40`; the active
`v0.43.x` line adds paper / methodology analysis modules on top of that
freeze.

## Citation

If you use this benchmark, please cite the thesis (BibTeX forthcoming).

## License and third-party components

The original source code, tests, configuration templates, setup scripts and
project documentation authored in this repository are licensed under the
Apache License, Version 2.0, unless otherwise stated.

This license does not apply to third-party datasets, pretrained model weights,
codec binaries, external tools, or locally generated benchmark outputs. Those
components are not redistributed by this repository and remain subject to their
respective upstream licenses and terms.

The router is designed to interoperate with external codecs, datasets and
models through declarative specifications and measured R-D-E rows.
Interoperability does not imply endorsement, ownership, redistribution rights,
or relicensing of those components.

See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` for details.
