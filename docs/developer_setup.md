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

During the structural refactor, keep legacy direct-script import fallbacks in
place unless a module is fully covered by module-execution tests. This preserves
existing workflows while new code can rely on package imports.

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
