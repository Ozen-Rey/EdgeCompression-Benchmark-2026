# Router PowerShell scripts

This document describes the router's PowerShell entrypoints and how to
extend them. The unified dispatcher
(`scripts/run_router.ps1`) is the only supported PowerShell entrypoint
for router smoke scenarios; it is the place to add new scenarios.

## scripts/run_router.ps1 — the only official PowerShell entrypoint

`scripts/run_router.ps1` is the **single PowerShell entrypoint** for
the router smoke scenarios. It is self-contained: every public scenario
is implemented inside this file as a function.

Common invocations:

```powershell
.\scripts\run_router.ps1 -Scenario list
.\scripts\run_router.ps1 -Scenario v02-backends
.\scripts\run_router.ps1 -Scenario v09-content-aware
.\scripts\run_router.ps1 -Scenario test    # aggregate: Python verifications
.\scripts\run_router.ps1 -Scenario smoke   # aggregate: v02-backends + v09-content-aware
.\scripts\run_router.ps1 -Scenario all     # aggregate: every single scenario
```

The dispatcher runs under `Set-StrictMode -Version Latest` and
`$ErrorActionPreference = "Stop"`; it propagates the non-zero exit
code of any underlying step via `throw` so wrappers and `try`/`catch`
blocks see the failure exactly the way they do when invoking a script
directly.

The legacy `scripts/run_router_v*.ps1` files were removed in v0.42.39
and **must not be reintroduced**: their behavior is preserved by the
inlined `Invoke-Scenario<Name>` functions inside the dispatcher.

## Anatomy of the dispatcher

### `$ScenarioMap`

`$ScenarioMap` is an ordered dictionary mapping each public scenario
name (e.g. `"v02-backends"`) to the name of its handler function
(e.g. `"Invoke-ScenarioV02Backends"`). It is the single source of
truth for what scenarios exist, in what order they appear in
`-Scenario list`, and which function the dispatcher calls when a user
selects a scenario.

`$AggregateScenarios` lists the aggregate names (`test`, `smoke`,
`all`) and is intentionally separate from `$ScenarioMap`: the
`all` aggregate iterates over `$ScenarioMap.Keys` only, so no
aggregate name may appear inside `$ScenarioMap` (this would risk
recursion).

### `Invoke-Scenario<Name>` functions

Each public scenario is implemented as a single
`Invoke-Scenario<Name>` function inside the dispatcher. The naming is
mechanical: `v02-backends` → `Invoke-ScenarioV02Backends`,
`v09-content-aware` → `Invoke-ScenarioV09ContentAware`, and so on.
The static dispatcher tests assert that the handler named in
`$ScenarioMap` actually exists as a top-level function.

A scenario function is responsible for:

- printing a clear header (`Write-Host "=== ... ==="`);
- invoking the underlying `python -m src.router.*` commands (or
  helpers from this file) with the scenario-specific arguments;
- checking exit codes via `Assert-LastExitCode` after every native
  invocation;
- producing or refreshing the scenario's expected outputs;
- (when applicable) calling `Invoke-LocalPytest` at the end to run
  the test suite under a private basetemp.

Per-scenario state must stay function-local. In particular, do **not**
write to `$script:`-scoped variables from inside a nested function: it
fails under `Set-StrictMode -Version Latest` (see
`tests/test_smoke_dispatcher.py::test_dispatcher_does_not_use_script_scoped_rows_state`).

### `Assert-LastExitCode`

```powershell
function Assert-LastExitCode {
    param([string]$Description)
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed (exit $LASTEXITCODE): $Description"
    }
}
```

Call this immediately after every native (non-PowerShell) invocation —
typically a `python -m src.router.*` command. The dispatcher's
strict-mode/`Stop` configuration does not catch native exit codes on
its own, so failures must be lifted into PowerShell exceptions
explicitly. Pass a short human-readable description of the step so the
thrown message is actionable.

### `Invoke-LocalPytest`

```powershell
Invoke-LocalPytest -ScenarioName "v09-content-aware"
```

Helper that runs `python -m pytest tests -q --basetemp
.pytest_tmp_run_router_<scenario>`, cleans the basetemp before and
after, and `throw`s on a non-zero pytest exit code. Use this whenever
a scenario should re-run the test suite at the end. The basetemp
directory is local to the repository so Windows temp-permission issues
on `$env:LOCALAPPDATA\Temp\pytest-of-*` do not break the smoke; the
`.pytest_tmp_*/` glob is git-ignored.

## How to add a new scenario

1. **Pick a stable scenario name.** Use the same `vXX-kebab-case`
   convention as the existing scenarios. The name is part of the
   public CLI contract (people will type `-Scenario <name>`).
2. **Add a `$ScenarioMap` entry** for the new name, pointing at a
   handler function name with the mechanical `Invoke-Scenario<Name>`
   PascalCase rendering (drop the dashes). Insertion order is the
   order in which `-Scenario list` and the `all` aggregate visit
   scenarios, so add the entry where it belongs logically.
3. **Implement `function Invoke-Scenario<Name>`** in the dispatcher.
   Inline the operative logic — do **not** call out to a separate
   `scripts/run_router_<name>.ps1` file. Use `Assert-LastExitCode`
   after every native call and `Invoke-LocalPytest` at the end if the
   scenario should re-run the test suite.
4. **Update the static dispatcher tests** in
   `tests/test_smoke_dispatcher.py`:
   - add the new `(scenario, handler)` pair to `EXPECTED_SCENARIOS`;
   - run `python -m pytest tests/test_smoke_dispatcher.py -q` and
     verify it stays green. The tests assert that every expected
     scenario is declared, every declared scenario is expected, every
     handler function exists, no aggregate name appears in
     `$ScenarioMap`, and that no removed `run_router_v*.ps1` filename
     reappears in the dispatcher.
5. **Document the scenario** if its semantics are non-obvious. The
   in-source `Write-Host` headers should be enough for routine cases;
   anything else goes here or in `docs/developer_setup.md`.

What **not** to do:

- Do not create `scripts/run_router_<name>.ps1` to back the new
  scenario. The legacy `run_router_v*.ps1` files were removed in
  v0.42.39 and the static tests assert they stay removed.
- Do not add the new scenario name to `$AggregateScenarios`. The only
  aggregate names are `test`, `smoke`, `all`.
- Do not write per-scenario state into `$script:`-scoped variables;
  keep it function-local.

## scripts/run_router_cases.ps1 — system-aware case study sweep

`scripts/run_router_cases.ps1` is a separate, standalone case-study
script. It is **not** part of the smoke-dispatcher contract and is not
referenced by `scripts/run_router.ps1`, by the static dispatcher tests,
or by the `test` / `smoke` / `all` aggregates.

What it does: runs a fixed sweep of `--system-aware` decisions against
`results/images/image_4dataset_RDE_paper_ready.csv` covering safe-mode
on/off, simulated CUDA availability, an aggressive max-time-ms
constraint, and an ultra-low-bitrate target that is expected to fail
without CUDA. It then collapses the per-case JSON reports into a
single `results/routing_context/router_case_summary.csv` for
inspection.

When to run it: as an ad-hoc sanity check that the system-aware policy
behaves sensibly across the canonical cases. It depends on a
pre-existing paper-ready CSV under `results/images/` and is therefore
expected to be run by the user locally, not by CI. It is intentionally
kept out of the dispatcher because its scope (a fixed sweep against a
fixed input CSV) is narrower than what a public scenario should
guarantee.

## Test layer

`tests/test_smoke_dispatcher.py` enforces the static invariants of the
dispatcher described above (scenario set, handler functions, no legacy
script references, strict-mode hygiene, aggregate behavior). The tests
do **not** execute PowerShell. They read the dispatcher source and the
developer setup doc and assert structural properties only. Update this
file whenever you add, rename, or remove a scenario.

## Note on running smoke scenarios

The PowerShell scenarios are intended to be executed locally by the
user. The Python verifications under `Scenario test`
(`legacy_import_audit`, `pytest`, `--help` smoke for both invocation
styles, and `py_compile` on the router core) are safe to run from any
shell.
