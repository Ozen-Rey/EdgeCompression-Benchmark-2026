# Developer Setup

This repository supports editable installs for local development:

```powershell
python -m pip install -e .
python -m pip install -e ".[test]"
python -m pytest -q
```

The `[test]` extra pulls in `pytest`; use it (or the `[dev]` alias) when working
from a fresh environment. The editable install keeps the existing `src.router`
module path stable while making imports independent of the current working
directory. Router CLIs should continue to be invoked with module execution:

```powershell
python -m src.router.rde_router --help
python -m src.router.codecs.external_codec_spec --help
python -m src.router.codecs.external_codec_probe --help
python -m src.router.codecs.external_codec_dry_run --help
python -m src.router.codecs.external_codec_benchmark --help
python -m src.router.codecs.external_codec_rde_exporter --help
```

After an editable install, the following console entry points are also
available on `PATH` and call the same `main()` functions as the corresponding
`python -m` invocations:

```powershell
rde-router --help
rde-external-codec-probe --help
rde-external-codec-dry-run --help
rde-external-codec-benchmark --help
rde-external-codec-export --help
rde-legacy-import-audit --help
```

The `python -m src.router.*` invocations remain the canonical way to run the
router tools; the console scripts are a convenience and are not used by the
PowerShell smoke scripts under `scripts/`.

The legacy top-level wrapper modules were removed in v0.42.15. New code and
developer scripts should import and execute router tools through their
subpackage paths, as shown above. See `docs/router_architecture.md` for the
package map.

## Unified smoke dispatcher

`scripts/run_router.ps1` is the **preferred entrypoint** for running the
PowerShell smoke scenarios. It is a thin dispatcher that resolves a short
scenario name to one of the existing `scripts/run_router_*.ps1` smoke scripts
and invokes it via PowerShell with `-ExecutionPolicy Bypass`. It does not
duplicate any of the internal logic of those scripts.

```powershell
.\scripts\run_router.ps1 -Scenario list
.\scripts\run_router.ps1 -Scenario v02-backends
.\scripts\run_router.ps1 -Scenario v09-content-aware
```

`-Scenario list` prints all known scenarios with their target scripts.
Unknown scenarios raise a clear error listing the supported names. The
dispatcher propagates the non-zero exit code of the underlying script when the
smoke fails, so CI wrappers and `try`/`catch` blocks see the failure exactly
the way they do when invoking a script directly.

### Legacy scripts

The direct `scripts/run_router_v*.ps1` scripts are still supported for
backwards compatibility and **have not been removed** in this release; any
existing tooling that invokes them keeps working unchanged. New smoke
invocations should prefer the dispatcher so that the scenario surface stays
in a single, discoverable place.

## Backend Smoke Executables

Backend smoke scripts use the router with `--strict-executables`, so codecs that
depend on missing local tools are still filtered by the router at runtime. The
scripts do not install codecs, edit the Windows registry, or modify the global
Windows PATH.

Check the local executable visibility before running backend smokes:

```powershell
where.exe cjxl
where.exe ffmpeg
where.exe vvencapp
```

`scripts/run_router_v02_backends.ps1` performs a session-local PATH bootstrap for
common WinGet installs. If `cjxl` or `ffmpeg` are not already visible through
`where.exe`, the script searches under:

```powershell
$env:LOCALAPPDATA\Microsoft\WinGet\Packages
```

When it finds an executable there, it prepends that executable directory to
`$env:PATH` for the current PowerShell process only. Missing required backends
fail early with a clear prerequisite error, instead of surfacing later as an
ambiguous infeasible router decision.

This bootstrap is only a developer smoke-script convenience. The router itself
does not auto-install tools, does not perform implicit WinGet discovery, and
continues to honor strict capability filtering exactly as configured.

## Pytest temporary directory in smoke scripts

The PowerShell smoke scripts under `scripts/run_router_*.ps1` that invoke
pytest pass an explicit `--basetemp .pytest_tmp_<script_name>` directory local
to the repository and remove it before and after the run. This avoids
permission errors on the default global Windows pytest temp directory
(`$env:LOCALAPPDATA\Temp\pytest-of-*`), and lets the scripts fail loudly via
`throw` when pytest returns a non-zero exit code. The local basetemp paths are
ignored by `.gitignore` (`.pytest_tmp_*/`).
