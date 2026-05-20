# EdgeCompression-Benchmark-2026

Rate–Distortion–Energy (R-D-E) benchmark of classical and neural compression
codecs across image, audio, and video, with an adaptive R-D-E router and a
content-aware routing extension.

## Overview

This repository contains the code, results, and routing infrastructure
underlying the master's thesis _"Compressione dei Dati: Un'Analisi
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

Ground-truth numbers used in the manuscript are stored in
`results/images/image_thesis_numbers.json` and
`results/video/video_thesis_numbers.json`. Thesis figures are regenerated
from the corresponding CSVs by the scripts under `src/utils/` and
`src/plots/`.

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

```bash
conda create -n tesi python=3.11
conda activate tesi
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -e .
```

External codec dependencies (`cjxl`, `ffmpeg`, `vvenc`, `SvtAv1EncApp`,
`opusenc`, etc.) are resolved through the external codec registry; see
`docs/external_codecs.md`.

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
results/           benchmark CSV / JSON outputs (thesis ground truth)
figures/           thesis figures
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

## License

See `LICENSE`.
