# Developer Setup

## Router-only setup helper

For a fresh router development environment, use the cross-platform setup helper:

```powershell
python scripts/setup/setup_router.py
python scripts/setup/setup_router.py --dry-run
python scripts/setup/setup_router.py --with-tests
```

The setup script prepares the router development environment. It does not
install or reproduce the full benchmark stack. It may create a virtual
environment and run `python -m pip install -e ".[test]"`, but only after an
explicit `y/N` prompt. The default answer is `No`.

The read-only doctor reports the local environment without modifying it:

```powershell
python scripts/setup/doctor.py --report-out environment_doctor_report.json
```

External codecs, datasets, checkpoints, hardware energy tools and full
benchmark reproduction are intentionally outside this setup flow. See
`docs/router_setup.md` for the full boundary.
They remain subject to their own upstream licenses and terms; this repository
does not redistribute them.

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

`scripts/run_router.ps1` is the **only PowerShell entrypoint** for running the
router smoke scenarios. As of v0.42.39 the legacy `scripts/run_router_v*.ps1`
files have been removed from the repository; their operative logic is inlined
into the dispatcher as `Invoke-Scenario<Name>` functions, one per scenario.
Public scenario names (`v02-backends`, `v09-content-aware`, …) and their
observable behavior are unchanged.

For a deeper walkthrough of the dispatcher (`$ScenarioMap`,
`Invoke-Scenario<Name>`, `Assert-LastExitCode`, `Invoke-LocalPytest`),
how to add a new scenario, the static checks in
`tests/test_smoke_dispatcher.py`, and the role of
`scripts/run_router_cases.ps1`, see `docs/router_scripts.md`.

```powershell
.\scripts\run_router.ps1 -Scenario list
.\scripts\run_router.ps1 -Scenario v02-backends
.\scripts\run_router.ps1 -Scenario v09-content-aware
```

`-Scenario list` prints both the single scenarios (with their
`Invoke-Scenario<Name>` handler function) and the aggregate scenarios
described below. Unknown scenarios raise a clear error listing the supported
names. The dispatcher propagates the non-zero exit code of the underlying
handler when the smoke fails, so CI wrappers and `try`/`catch` blocks see
the failure exactly the way they do when invoking a scenario directly.

### Aggregate scenarios

The dispatcher also exposes a few aggregate scenarios that bundle the common
verification flows into a single command:

```powershell
.\scripts\run_router.ps1 -Scenario test
.\scripts\run_router.ps1 -Scenario smoke
.\scripts\run_router.ps1 -Scenario all
```

- `test` runs the local Python verifications: `legacy_import_audit`, `pytest`
  with a private `--basetemp .pytest_tmp_dispatcher`, the `--help` smoke for
  both `python -m src.router.rde_router` and the direct `src\router\rde_router.py`
  invocation, and `py_compile` on the router core modules. After the steps
  succeed it cleans up the local `.pytest_tmp_*` scratch directories. Fast and
  safe to run frequently.
- `smoke` runs the two recommended PowerShell smoke scenarios in order:
  `v02-backends` followed by `v09-content-aware`. Stops at the first failure.
- `all` runs every single scenario declared in the dispatcher, in deterministic
  insertion order, excluding the aggregates themselves to prevent recursion.
  Longest variant, intended for full pre-release verification.

### Legacy scripts (removed in v0.42.39)

The standalone `scripts/run_router_v*.ps1` files were removed from the
repository in v0.42.39 once their bodies were inlined into the unified
dispatcher. Any reference to `scripts/run_router_v02_backends.ps1`,
`scripts/run_router_v09_content_aware.ps1`, etc. should be replaced with
`scripts/run_router.ps1 -Scenario <name>` (same scenario names, same outputs,
same exit-code semantics). The dispatcher remains the single source of truth
for the supported scenario set.

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

The `v02-backends` scenario in `scripts/run_router.ps1` performs a
session-local PATH bootstrap for common WinGet installs. If `cjxl` or
`ffmpeg` are not already visible through `where.exe`, the scenario searches
under:

```powershell
$env:LOCALAPPDATA\Microsoft\WinGet\Packages
```

When it finds an executable there, it prepends that executable directory to
`$env:PATH` for the current PowerShell process only. Missing required backends
fail early with a clear prerequisite error, instead of surfacing later as an
ambiguous infeasible router decision.

This bootstrap is only a developer smoke-script convenience for tools already
installed or provided by the user. The router itself does not auto-install
tools, does not perform implicit WinGet discovery, and continues to honor
strict capability filtering exactly as configured.

## Pytest temporary directory in smoke scenarios

Each scenario in `scripts/run_router.ps1` that invokes pytest passes an
explicit `--basetemp .pytest_tmp_run_router_<scenario>` directory local to the
repository and removes it before and after the run. This avoids permission
errors on the default global Windows pytest temp directory
(`$env:LOCALAPPDATA\Temp\pytest-of-*`), and lets the scenarios fail loudly via
`throw` when pytest returns a non-zero exit code. The local basetemp paths are
ignored by `.gitignore` (`.pytest_tmp_*/`).
