# Developer Setup

This repository supports editable installs for local development:

```powershell
python -m pip install -e .
python -m pytest -q
```

The editable install keeps the existing `src.router` module path stable while
making imports independent of the current working directory. Router CLIs should
continue to be invoked with module execution:

```powershell
python -m src.router.rde_router --help
python -m src.router.codecs.external_codec_spec --help
python -m src.router.codecs.external_codec_probe --help
python -m src.router.codecs.external_codec_dry_run --help
python -m src.router.codecs.external_codec_benchmark --help
python -m src.router.codecs.external_codec_rde_exporter --help
```

The legacy top-level wrapper modules were removed in v0.42.15. New code and
developer scripts should import and execute router tools through their
subpackage paths, as shown above. See `docs/router_architecture.md` for the
package map.

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
