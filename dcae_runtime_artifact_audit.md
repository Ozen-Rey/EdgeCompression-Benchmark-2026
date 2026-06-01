# DCAE Runtime Artifact Audit

## Scope

Read-only audit to locate DCAE runtime code and checkpoints needed to run DCAE
in the image benchmark on Arch. No benchmark was executed, no dependencies were
installed, no downloads were performed, and no source code was modified.

## Initial git state

Command:

```powershell
git status --short --branch
```

Observed state:

```text
## router-v0453-licensing-attribution
?? docs/dcae_benchmark_provenance_audit.md
?? notebooks/
?? router-v0.46.2-source.zip
?? router-v0.46.3-source.zip
?? src/benchmark/benchmark_images.py
?? src/utils/energy_monitor.py
```

## Required by benchmark code

Inspected file:

```text
src/benchmark/image_v2/codecs_neural_actual.py
```

Relevant loader contract:

- `find_first_existing(paths)` is defined at line 185.
- `load_dcae(lam)` is defined at line 240.
- DCAE root candidates are:
  - line 243: `~/tesi/external_codecs/DCAE`
  - line 244: `/tmp/DCAE`
- Checkpoint candidates are:
  - line 253: `<root>/checkpoints/dcae_mse_0013.pth.tar`
  - line 254: `/tmp/DCAE/checkpoints/dcae_mse_0013.pth.tar`
  - line 257: `<root>/checkpoints/dcae_mse_0035.pth.tar`
  - line 258: `/tmp/DCAE/checkpoints/dcae_mse_0035.pth.tar`
- The expected class import is:
  - line 270: `from models import DCAE`
- The model is instantiated as `DCAE()`.
- The checkpoint is loaded with:
  - line 273: `sd_raw = torch.load(ckpt, map_location=device)`
- The checkpoint format is expected to contain `sd_raw["state_dict"]`; keys
  have a leading `module.` prefix stripped before loading.
- The model weights are loaded with:
  - line 275: `net.load_state_dict(sd)`

Conclusion from code: checkpoints alone are not enough. The benchmark requires
a DCAE source root on `sys.path` containing `models.py` or a `models` package
that exports class `DCAE`, plus any local modules imported by that source.

## Files found

| Artifact | Path | Size | Purpose | Confidence |
| --- | --- | ---: | --- | --- |
| DCAE benchmark loader | `src/benchmark/image_v2/codecs_neural_actual.py` | Not measured | Defines runtime contract for DCAE | High |
| Existing provenance note | `docs/dcae_benchmark_provenance_audit.md` | Not measured | Prior audit note; confirms no runtime artifacts found then | High |
| DCAE result references | `results/routing_context/force_low_bitrate_dcae.json`, `results/routing_context/force_low_bitrate_dcae_topk.csv` | Not measured | Results/routing context only, not runtime code | Medium |
| DCAE source folder | Not found | N/A | Needed runtime code | High |
| `models.py` with class `DCAE` | Not found | N/A | Needed import target | High |
| `dcae_mse_0013.pth.tar` | Not found | N/A | Needed checkpoint for `lam=0.013` | High |
| `dcae_mse_0035.pth.tar` | Not found | N/A | Needed checkpoint for `lam=0.0035` | High |

## DCAE code folder

No valid DCAE code folder was found under `C:\Users\valai`.

Searches performed:

```powershell
Get-ChildItem -Path C:\Users\valai -Recurse -Directory `
  -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "DCAE|dcae" } |
  Select-Object FullName

Get-ChildItem -Path C:\Users\valai -Recurse -File -Filter models.py `
  -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "DCAE|dcae|external_codecs|codec|compression|benchmark|neural" } |
  Select-Object FullName, Length, LastWriteTime

Get-ChildItem -Path C:\Users\valai -Recurse -File -Include *.py `
  -ErrorAction SilentlyContinue |
  Select-String -Pattern "class DCAE","from models import DCAE","def load_dcae","dcae_mse_0013","dcae_mse_0035" |
  Select-Object Path, LineNumber, Line
```

Compatibility note: this Windows PowerShell did not support `Get-ChildItem
-File`, so the searches were re-run with equivalent older PowerShell filters
and cross-checked with `rg` and `cmd /c dir /s /b`.

Independent filename/content checks found only repository references and cache
filenames with incidental `dcae` substrings. No source folder named `DCAE`, no
candidate DCAE `models.py`, and no `class DCAE` implementation were found.

Because no DCAE code folder was found, local imports and internal dependencies
could not be inspected. Probable files to preserve once the upstream folder is
recovered include `models.py` plus any local modules imported by it, such as
`layers.py`, `entropy_models.py`, `modules.py`, `ops.py`, `utils.py`, package
directories, and any CompressAI integration files.

## Checkpoints

The two required checkpoints were not found under `C:\Users\valai`:

- `dcae_mse_0013.pth.tar`: not found
- `dcae_mse_0035.pth.tar`: not found

The expected Windows mirror path was checked directly:

```text
C:\Users\valai\tesi\external_codecs\DCAE
C:\Users\valai\tesi\external_codecs\DCAE\checkpoints\dcae_mse_0013.pth.tar
C:\Users\valai\tesi\external_codecs\DCAE\checkpoints\dcae_mse_0035.pth.tar
```

All three checks returned missing. The transfer root
`D:\EdgeCompression-transfer` was also checked and was not present on this
machine at audit time.

No file size or last-write timestamp can be reported for the checkpoints
because the files were not located.

## USB bundle plan

No copy was executed because no valid source DCAE folder or checkpoint files
were found.

Desired USB structure:

```text
D:\EdgeCompression-transfer\DCAE_runtime_bundle\
  DCAE\
    models.py
    ...
    checkpoints\
      dcae_mse_0013.pth.tar
      dcae_mse_0035.pth.tar
  README_DCAE_RUNTIME_BUNDLE.txt
  SHA256SUMS_DCAE.txt
```

If the complete DCAE source folder is recovered at
`C:\Users\valai\tesi\external_codecs\DCAE`, use:

```powershell
robocopy "C:\Users\valai\tesi\external_codecs\DCAE" "D:\EdgeCompression-transfer\DCAE_runtime_bundle\DCAE" /E
```

If the checkpoints are recovered separately, copy them into the bundle after
the source folder exists:

```powershell
New-Item -ItemType Directory -Force "D:\EdgeCompression-transfer\DCAE_runtime_bundle\DCAE\checkpoints" | Out-Null

Copy-Item "C:\PATH\VERO\dcae_mse_0013.pth.tar" `
  "D:\EdgeCompression-transfer\DCAE_runtime_bundle\DCAE\checkpoints\" -Force

Copy-Item "C:\PATH\VERO\dcae_mse_0035.pth.tar" `
  "D:\EdgeCompression-transfer\DCAE_runtime_bundle\DCAE\checkpoints\" -Force
```

Generate checksums after the bundle is complete:

```powershell
cd D:\EdgeCompression-transfer\DCAE_runtime_bundle

Get-ChildItem -Recurse -File |
  Get-FileHash -Algorithm SHA256 |
  ForEach-Object { "$($_.Hash)  $($_.Path.Replace((Get-Location).Path + '\',''))" } |
  Set-Content SHA256SUMS_DCAE.txt
```

## Arch install plan

Install the bundle into the path expected by the benchmark:

```bash
mkdir -p ~/tesi/external_codecs

cp -r /run/media/Democrito/76E8-CACF/EdgeCompression-transfer/DCAE_runtime_bundle/DCAE \
  ~/tesi/external_codecs/
```

Verify files:

```bash
find ~/tesi/external_codecs/DCAE -maxdepth 2 -type f | head -50
ls -lh ~/tesi/external_codecs/DCAE/checkpoints
```

Test import:

```bash
python - <<'PY'
import sys
from pathlib import Path

root = Path("~/tesi/external_codecs/DCAE").expanduser()
sys.path.insert(0, str(root))

print("root exists:", root.exists())
print("models.py exists:", (root / "models.py").exists())
print("checkpoints:", list((root / "checkpoints").glob("*.pth.tar")))

from models import DCAE
print("DCAE import OK:", DCAE)
PY
```

## Conclusion

B. DCAE incomplete: checkpoints found in prior Arch context may exist elsewhere,
but on this Windows machine the audit found neither the DCAE source folder nor
the two required checkpoint files under `C:\Users\valai`.

The current machine does not contain enough artifacts to prepare a complete
`DCAE_runtime_bundle`. Recover the upstream DCAE source checkout and both
checkpoint files before attempting the USB bundle copy.
