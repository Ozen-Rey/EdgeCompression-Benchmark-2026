# DCAE Benchmark Provenance Audit

## Scope

Read-only audit to locate the DCAE benchmark backend, checkpoints, and related
provenance for the image benchmark path. No benchmark was executed, no
dependencies were installed, and no checkpoints/datasets were downloaded.

## Searches Performed

- `git status --short --branch`
- Repository search over `*.py`, `*.json`, `*.yaml`, `*.yml`, `*.md`, and
  `*.ipynb` for DCAE, checkpoint, pretrained, Hugging Face, and `torch.load`
  terms.
- Direct inspection of:
  - `src/benchmark/benchmark_images.py`
  - `src/benchmark/benchmark_sota_12metrics.py`
  - `src/benchmark/benchmark_full_pipeline_energy.py`
  - `src/benchmark/image_v2/codecs_neural_actual.py`
  - `src/benchmark/image_v2/run_dataset_neural_actual.py`
  - `src/benchmark/image_v2/run_dataset_energy_v2.py`
  - `src/utils/`
  - `scripts/`
  - `notebooks/`
- Repository search for environment/config variables:
  `DCAE_ROOT`, `DCAE_CHECKPOINT`, `DCAE_MODEL`, `DCAE_WEIGHTS`,
  `CHECKPOINT_ROOT`, `MODEL_ROOT`, `HF_HOME`, `TORCH_HOME`,
  `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`.
- Read-only scan of `C:\Users\valai` for model/checkpoint extensions:
  `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.tar`, `*.onnx`.
- Read-only scan of `C:\Users\valai` for suspicious model/cache/backend
  directories.
- Read-only check of Hugging Face and Torch cache locations:
  `%USERPROFILE%\.cache\huggingface`, `%USERPROFILE%\.cache\torch`, and
  filtered `%USERPROFILE%\.cache`.
- Search in `results/` for DCAE rows in CSV/JSON/TXT/MD files.

## Initial Git State

Branch:

- `router-v0453-licensing-attribution`

Tracked modified files at audit start:

- none

Untracked files/directories at audit start:

- `notebooks/`
- `router-v0.46.2-source.zip`
- `router-v0.46.3-source.zip`
- `src/benchmark/benchmark_images.py`
- `src/utils/energy_monitor.py`

`src/benchmark/benchmark_images.py` was inspected directly for DCAE/checkpoint
terms and had no useful DCAE matches.

## Findings

### Router References

The router side treats DCAE as a codec/family label, not as a runnable backend.
For example, `src/router/codecs/codec_capabilities.py` contains the DCAE codec
capability metadata, and tests/docs refer to DCAE as a neural codec. This is
nominal routing metadata, not an execution path to weights.

### Benchmark References

The operational image benchmark path exists in benchmark code:

- `src/benchmark/image_v2/codecs_neural_actual.py:240-278`
  defines `load_dcae(lam)`.
- `src/benchmark/image_v2/codecs_neural_actual.py:243-244`
  searches for a DCAE source root at `~/tesi/external_codecs/DCAE` or
  `/tmp/DCAE`.
- `src/benchmark/image_v2/codecs_neural_actual.py:251-259`
  maps DCAE rates to checkpoint names:
  - `lam=0.013` -> `checkpoints/dcae_mse_0013.pth.tar`
  - `lam=0.0035` -> `checkpoints/dcae_mse_0035.pth.tar`
- `src/benchmark/image_v2/codecs_neural_actual.py:270-275`
  imports `from models import DCAE`, instantiates `DCAE()`, uses
  `torch.load(...)`, strips `module.` prefixes, and calls `load_state_dict`.
- `src/benchmark/image_v2/run_dataset_neural_actual.py:152-163`
  runs DCAE over `lam=0.0035` and `lam=0.013`, with `pad_multiple=128` and
  `pad_mode="center"`.
- `src/benchmark/image_v2/run_dataset_energy_v2.py:1196-1208`
  uses the same DCAE lambdas for `actual_bitstream_energy`, with repeats from
  `--dcae_repeats`.

Older benchmark scripts confirm the same provenance:

- `src/benchmark/benchmark_sota_12metrics.py:8-11`
  says `/tmp/DCAE` must be cloned with checkpoints.
- `src/benchmark/benchmark_sota_12metrics.py:42-45`
  hardcodes `/tmp/DCAE/checkpoints/dcae_mse_0013.pth.tar` and
  `/tmp/DCAE/checkpoints/dcae_mse_0035.pth.tar`.
- `src/benchmark/benchmark_sota_12metrics.py:275-287`
  imports `DCAE` from `/tmp/DCAE/models.py`, loads the checkpoint with
  `torch.load`, strips `module.`, and calls `load_state_dict`.
- `src/benchmark/benchmark_full_pipeline_energy.py:44`
  disables cuDNN as necessary for DCAE compress/decompress.
- `src/benchmark/benchmark_full_pipeline_energy.py:720-741`
  inserts `~/tesi/external_codecs/DCAE` on `sys.path`, but uses `/tmp/DCAE`
  checkpoint paths for the two DCAE lambdas.

No `hf_hub_download`, `snapshot_download`, or `from_pretrained` DCAE path was
found. The code expects a local source checkout plus local checkpoint files.

### Checkpoint / Backend Files

Checked expected local paths:

- `C:\Users\valai\tesi\external_codecs\DCAE`
- `C:\Users\valai\tesi\external_codecs\DCAE\checkpoints\dcae_mse_0013.pth.tar`
- `C:\Users\valai\tesi\external_codecs\DCAE\checkpoints\dcae_mse_0035.pth.tar`
- `C:\tmp\DCAE`
- `\tmp\DCAE`

Result: none of these paths existed on this Windows machine.

The read-only recursive scan of `C:\Users\valai` for model/checkpoint files did
not find DCAE checkpoint files or an obvious DCAE backend/source tree. The
directory scan found unrelated model/cache directories such as LM Studio models
and package directories, but no DCAE directory.

The Hugging Face cache directory `%USERPROFILE%\.cache\huggingface` was missing.
The Torch cache directory `%USERPROFILE%\.cache\torch` was missing. A filtered
scan of `%USERPROFILE%\.cache` for DCAE/model/checkpoint terms returned no DCAE
model artifact.

No relevant environment variable reference was found for `DCAE_ROOT`,
`DCAE_CHECKPOINT`, `DCAE_MODEL`, or `DCAE_WEIGHTS`. The only DCAE-like variable
found was the in-code `DCAE_CHECKPOINTS` dictionary in
`src/benchmark/benchmark_sota_12metrics.py`.

### Result CSV / JSON Findings

Existing results contain DCAE measurements, but not enough backend provenance to
restore execution artifacts:

- `results/images/sota_12metric_benchmark.csv`
  - Columns: `codec,param,image,bpp,pipeline,time_ms,energy_total_j,energy_net_j,params_M,psnr,ssim,ms_ssim,lpips,dists,fsim,gmsd,vif,haarpsi,dss,mdsi,ssimulacra2`
  - DCAE rows: 48
  - Pipeline includes `forward_pass_only`.
- `results/images/sota_neural_benchmark.csv`
  - Columns: `codec,lambda,bpp,psnr,fwd_ms,params_M,energy_j`
  - DCAE rows: 2
- `results/images/image_4dataset_RDE_paper_ready.csv`
  - Contains per-image DCAE rows with metrics plus joined energy columns.
  - DCAE rows: 192
  - Includes `pipeline=actual_bitstream` and `energy_pipeline=actual_bitstream_energy`.
- `results/images/image_4dataset_energy_v2_with_jpeg_ai.csv`
  - Columns include `pipeline`, `n_images`, `n_repeats`, `avg_bpp`,
    `avg_psnr`, `energy_cpu_per_image_j`, `energy_gpu_per_image_j`,
    `energy_per_image_j`, `params_M`, `is_neural`.
  - DCAE rows: 8
- `results/images/full_pipeline_energy_benchmark.csv`
  - Columns include `avg_bpp`, `avg_psnr`, CPU/GPU/total energy, idle watts,
    `params_M`, and `is_neural`.
  - DCAE rows: 2
- `results/images/sota_12metric_benchmark.json` and
  `results/images/image_thesis_numbers.json` contain DCAE aggregate summaries.
- Router/routing-context outputs contain DCAE selections and analyses, but these
  are downstream uses of measured rows rather than backend provenance.

## DCAE Execution Path

The reproducible DCAE execution path, if artifacts are restored, is:

1. Restore the DCAE source checkout so that `models.py` exposes `DCAE`.
2. Place checkpoints at either:
   - `~/tesi/external_codecs/DCAE/checkpoints/dcae_mse_0013.pth.tar`
   - `~/tesi/external_codecs/DCAE/checkpoints/dcae_mse_0035.pth.tar`
   or:
   - `/tmp/DCAE/checkpoints/dcae_mse_0013.pth.tar`
   - `/tmp/DCAE/checkpoints/dcae_mse_0035.pth.tar`
3. The benchmark imports `from models import DCAE`.
4. It instantiates `DCAE()`.
5. It loads `state_dict` via `torch.load`, strips `module.` prefixes, and calls
   `load_state_dict`.
6. It calls `net.update()`.
7. Actual-bitstream paths call `compress(...)` and `decompress(...)` with
   `pad_multiple=128` and `pad_mode="center"`.

No automatic checkpoint download path was found.

## Evidence Table

| Evidence | File/path | Meaning | Confidence |
| --- | --- | --- | --- |
| `load_dcae(lam)` imports `from models import DCAE`, calls `torch.load`, and `load_state_dict` | `src/benchmark/image_v2/codecs_neural_actual.py:240-278` | Operational DCAE benchmark backend contract exists in code | High |
| DCAE root search is `~/tesi/external_codecs/DCAE` or `/tmp/DCAE` | `src/benchmark/image_v2/codecs_neural_actual.py:241-245` | Backend is expected as a local source checkout, not bundled in repo | High |
| DCAE checkpoint names are `dcae_mse_0013.pth.tar` and `dcae_mse_0035.pth.tar` | `src/benchmark/image_v2/codecs_neural_actual.py:251-259` | Required checkpoint filenames are known | High |
| Older SOTA script explicitly requires `/tmp/DCAE` cloned with checkpoints | `src/benchmark/benchmark_sota_12metrics.py:8-11` | Historical benchmark provenance points to external `/tmp/DCAE` checkout | High |
| Full-pipeline energy script disables cuDNN and uses DCAE checkpoints | `src/benchmark/benchmark_full_pipeline_energy.py:44`, `:720-741` | DCAE was intended to run as compress/decompress local neural codec | High |
| Expected Windows paths and `/tmp/DCAE` equivalents do not exist | filesystem checks | Current machine does not have the required DCAE backend/checkpoints in expected locations | High |
| Hugging Face and Torch caches missing/no DCAE artifacts | `%USERPROFILE%\.cache\huggingface`, `%USERPROFILE%\.cache\torch`, `%USERPROFILE%\.cache` | No evidence of DCAE restored via standard HF/Torch cache | Medium-High |
| Existing CSV/JSON contain DCAE metrics and energy rows | `results/images/*.csv`, `results/images/*.json` | Prior benchmark results exist, but do not include enough artifact provenance to re-run DCAE | High |
| `src/benchmark/benchmark_images.py` has no useful DCAE/checkpoint matches | untracked local file | This untracked legacy image script is not the DCAE backend | Medium |

## Required Artifacts for USB Transfer

No usable DCAE artifact source was located on this machine, so there is no
verified source path to copy now.

To make DCAE reproducible on another machine, the required artifacts are:

- DCAE source checkout containing `models.py` with class `DCAE`.
- `checkpoints/dcae_mse_0013.pth.tar`
- `checkpoints/dcae_mse_0035.pth.tar`
- Any upstream DCAE support files imported by `models.py`.

If those artifacts are restored under the expected path
`C:\Users\valai\tesi\external_codecs\DCAE`, copy them as a single directory.

## Copy Commands

No command below was executed during this audit.

If the DCAE source/checkpoints are restored at the expected Windows path:

```powershell
robocopy "C:\Users\valai\tesi\external_codecs\DCAE" "D:\EdgeCompression-transfer\dcae_backend" /E
```

If the source/checkpoints are restored under a Windows-accessible `/tmp/DCAE`
equivalent:

```powershell
robocopy "C:\tmp\DCAE" "D:\EdgeCompression-transfer\dcae_backend" /E
```

If only checkpoints are being transferred after the backend source is already
available elsewhere:

```powershell
robocopy "C:\Users\valai\tesi\external_codecs\DCAE\checkpoints" "D:\EdgeCompression-transfer\checkpoints\dcae" /E
```

## Conclusion

B. DCAE backend/checkpoints not located.

DCAE is present in benchmark results and router analyses as a measured codec,
and benchmark code contains an operational local backend contract. However, the
actual DCAE source checkout and required checkpoints were not found in the repo,
the expected external paths, or the searched Hugging Face/Torch/user caches.

For the Kodak mini-benchmark, DCAE must remain optional/unavailable unless the
upstream DCAE backend and checkpoints are restored. Do not invent DCAE rows. To
re-enable DCAE execution, recover the upstream source/config support files and
the two checkpoint files named above from the original benchmark environment.
