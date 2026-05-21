# Real Audio/Video Router Validation

v0.45.0 validates that the router can consume real measured audio and video
R-D-E rows, not synthetic placeholder smoke data. The goal is to exercise the
same router contract already used for images: measured rate, measured quality,
measured energy, a `DomainSpec`, router decisions, and offline oracle/regret
analysis.

This is not a new benchmark campaign. It reuses existing benchmark rows under
locally generated `results/audio` and `results/video` directories. Those rows
are not redistributed as part of the repository source license.

## Data Used

Audio uses:

- `results/audio/visqol_audio_mode_benchmark.csv` for per-item ViSQOL.
- `results/audio/audio_summary_full.csv` for measured codec/config bitrate.
- `results/audio/full_pipeline_energy_benchmark.csv` for measured per-item
  net energy and timing.

The router-ready energy column is:

```text
energy_j_per_second = energy_total_net_j / duration_s
```

Video uses:

- `results/video/video_LDP_reference_preset_paper_ready.csv` for per-sequence
  bitrate, VMAF, timing, and energy.

The router-ready rate column is:

```text
bitrate_kbps = actual_mbps * 1000
```

The router-ready energy column is the measured `energy_total_kj` renamed to
`energy_kj_per_sequence`.

## Metrics

Audio uses `audio_visqol`:

- rate: `bitrate_kbps`
- quality: `visqol`, higher is better
- energy: `energy_j_per_second`
- item id: `item_id`

FAD remains useful for benchmark-level scientific discussion, but it is not
used as the router quality metric here because the available FAD rows are
aggregate codec/config values, not reliable per-item router quality rows.

Video uses `video_vmaf`:

- rate: `bitrate_kbps`
- quality: `vmaf`, higher is better
- energy: `energy_kj_per_sequence`
- item id: `sequence`

VMAF is available, so no PSNR-Y fallback is used.

## Build Artifacts

Generate all artifacts:

```powershell
python -m src.router.analysis.audio_video_policy_validation `
  --root . `
  --out-dir results/routing_context/audio_video_real_validation `
  --mode all
```

Primary outputs are CSV/JSON files under:

```text
results/routing_context/audio_video_real_validation/
```

PNG figures, when present, are derived best-effort renderings. The CSV/JSON
files are the primary artifacts.

## Validation Commands

DomainSpec validation:

```powershell
python -m src.router.core.domain_spec `
  --csv results/routing_context/audio_video_real_validation/audio_rde_router_ready.csv `
  --builtin audio_visqol `
  --validate-csv

python -m src.router.core.domain_spec `
  --csv results/routing_context/audio_video_real_validation/video_rde_router_ready.csv `
  --builtin video_vmaf `
  --validate-csv
```

DatasetManifest validation:

```powershell
python -m src.router.core.dataset_manifest `
  --manifest results/routing_context/audio_video_real_validation/audio_dataset_manifest.json `
  --validate

python -m src.router.core.dataset_manifest `
  --manifest results/routing_context/audio_video_real_validation/video_dataset_manifest.json `
  --validate
```

Router runs use the existing profile names:

```powershell
python -m src.router.rde_router `
  --csv results/routing_context/audio_video_real_validation/audio_rde_router_ready.csv `
  --domain-spec audio_visqol `
  --profile balanced `
  --out results/routing_context/audio_video_real_validation/audio_router_balanced_report.json `
  --summary-out results/routing_context/audio_video_real_validation/audio_router_balanced_summary.csv
```

The same pattern is used for video with `video_vmaf`. `energy-limited` is the
existing router profile corresponding to the energy-saving regime.

## What The Results Show

The generated reports show that measured audio/video rows can be transformed
into router-ready CSVs, validated through `DomainSpec`, consumed by the router,
and compared offline against per-item oracle and robust global baseline
policies. The diagnostic tables summarize selected codecs, regret, rate,
quality, energy, feasible rows, and profile-level behavior.

## Boundaries

This validation does not:

- run a new benchmark campaign;
- grant rights to redistribute the underlying datasets, codecs, models,
  metrics, or generated benchmark rows;
- change Chapter 4 official benchmark results;
- add content-aware audio/video prediction;
- validate automatic execution backends for every codec;
- invent missing quality or energy measurements;
- replace the domain-specific benchmark discussion.
