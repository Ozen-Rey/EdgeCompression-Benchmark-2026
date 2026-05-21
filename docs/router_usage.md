\# R-D-E Router Prototype

## Router-only setup

Use the router-only setup helper before running local router development
checks:

```powershell
python scripts/setup/setup_router.py
python scripts/setup/doctor.py --report-out environment_doctor_report.json
```

Lo script di setup prepara l'ambiente di sviluppo del router. Non installa ne'
riproduce automaticamente l'intero stack di benchmark.

External codecs, datasets, checkpoints and energy measurement tools remain
benchmark/execution prerequisites and are not installed by the setup helper.



This prototype implements an adaptive codec selection procedure based on

Rate--Distortion--Energy measurements.



The router performs:



1\. loading of benchmark R-D-E points from CSV;

2\. aggregation by codec/configuration;

3\. global normalization of rate, distortion and energy;

4\. system-aware filtering based on CUDA availability;

5\. hard quality-guard filtering;

6\. optional degraded fallback;

7\. R-D-E cost minimization;

8\. JSON, top-k CSV and summary CSV reporting.



\## Reproduce router case study



From the project root:



```powershell

.\\scripts\\run\_router\_cases.ps1

```

## v0.2 execution mode

The router can also generate and execute an operational command for supported
codecs. In the current prototype, execution is implemented for JPEG through a
minimal Pillow backend.

Example:

```powershell
python -m src.router.rde_router `
  --csv "results\images\image_4dataset_RDE_paper_ready.csv" `
  --domain image `
  --auto-weights `
  --system-aware `
  --capability-aware `
  --strict-executables `
  --power-mode ac `
  --thermal-state nominal `
  --network-profile very-limited `
  --quality-target preview `
  --safe-mode `
  --quality-constraint-stat min `
  --quality-floor 70 `
  --allow-degraded-fallback `
  --near-quality-floor 55 `
  --codec-col codec `
  --config-col param `
  --rate-col bpp `
  --quality-col ssimulacra2 `
  --energy-col energy_per_image_j `
  --time-col time_ms `
  --aggregate-by-config `
  --generate-command `
  --execute `
  --input "test_images\input.png" `
  --output "test_images\router_output.jpg" `
  --out "results\routing_context\v02_execute_jpeg.json"
```

This produces:

```text
test_images\router_output.jpg
```

## v0.2 deployable backend mode

Version `v0.2` extends the R-D-E router from a pure decision engine to a
deployable-oriented prototype. The router can now check codec backend
availability, generate an execution plan and optionally execute the selected
configuration.

Implemented execution backends:

| Codec | Backend | Status |
|---|---|---|
| JPEG | Python/Pillow | command generation + execute |
| JPEG XL | `cjxl` | command generation + execute |
| HEVC Intra | `ffmpeg`/`libx265` | command generation + execute |

The router searches executables both in the system `PATH` and in local project
folders:

```text
tools/jxl/cjxl.exe
tools/jxl/djxl.exe
tools/ffmpeg/ffmpeg.exe
```

This makes the router usable without requiring all backend binaries to be
globally installed.

## v0.3 local calibration mode

Version `v0.3` adds local calibration. The router can run a small calibration
procedure on the target machine, measuring local execution time, output size and
effective bitrate for registered backends.

Calibration levels:

| Level | Purpose |
|---|---|
| `quick` | smoke test, one image and one representative config per codec |
| `standard` | practical calibration with more images/configurations |
| `full` | exhaustive calibration over the configured set |

The quick calibration mode has been validated end-to-end. The standard and full
levels use the same execution pipeline but increase the number of images,
configurations and repetitions.

### Run quick calibration

```powershell
python -m src.router.calibration.calibration `
  --level quick `
  --input-dir calibration_images `
  --codecs JPEG,JXL,HEVC `
  --out results\routing_calibration\local_quick.json
```

