# Changelog

All notable changes to the R-D-E router are recorded here. The router
follows a single linear version line (`ROUTER_VERSION` in
`src/router/version.py`, kept in sync with `pyproject.toml`).

## v0.42.40 — Refactor freeze (2026-05-15)

Closes the v0.42.x architectural refactor as a citable milestone. No
behavioral changes: same CLI flags, same report schema, same
ranking / scoring / normalization / calibration / external-codec
behavior. After this release the refactor is considered frozen; further
work resumes against this stable baseline.

The v0.42.x refactor in summary:

- **Package layout.** Router code lives under `src/router/` with stable
  subpackages (`core/`, `calibration/`, `adaptation/`, `analysis/`,
  `codecs/`, `observability/`, `feedback/`, ...). Public entrypoints
  are the `python -m src.router.*` invocations.
- **Legacy wrapper removal.** Top-level wrapper modules were removed in
  v0.42.15. New code imports from the subpackage paths directly.
- **Dual-import removal.** Each module is reachable via exactly one
  canonical import path; the legacy import audit
  (`python -m src.router.observability.legacy_import_audit`) enforces
  this read-only invariant.
- **RouterContext-only reports.** Report assembly flows through a
  single `RouterContext` object rather than being scattered across
  pipeline-local dictionaries.
- **CSV row diagnostics.** Codec-filtering, calibration, and
  normalization reports surface row-level diagnostics so infeasible
  decisions can be explained without re-running the pipeline.
- **English CLI/error surface.** All user-facing argparse `help=` text
  and `raise ValueError(...)` messages in the router package are in
  English. Italian text is intentionally retained only in paper-figure
  labels (Italian thesis artifacts) and historical commit messages.
- **Feature flag groups.** `FEATURE_GROUPS` in
  `src/router/version.py` partitions feature flags by category and
  flattens into the historical `FEATURE_LEVEL` dict; duplicate flag
  names raise at import time.
- **Test scratch cleanup.** Smoke scenarios pass an explicit per-scenario
  `--basetemp .pytest_tmp_run_router_<scenario>` and clean it up before
  and after each run; `.pytest_tmp_*/` is git-ignored.
- **`pyproject.toml` / console scripts.** The package is `pip install
  -e .`-able, with console entrypoints mirroring the `python -m`
  invocations (`rde-router`, `rde-external-codec-probe`,
  `rde-external-codec-dry-run`, `rde-external-codec-benchmark`,
  `rde-external-codec-export`, `rde-legacy-import-audit`).
- **Unified PowerShell dispatcher.** `scripts/run_router.ps1` is the
  single PowerShell entrypoint for every router smoke scenario. The
  legacy `scripts/run_router_v*.ps1` files were removed in v0.42.39
  and their bodies were inlined as `Invoke-Scenario<Name>` functions.
  Static invariants are enforced by `tests/test_smoke_dispatcher.py`.
  See `docs/router_scripts.md` for how to add a new scenario.
- **Final freeze point.** v0.42.40 is the citable end of the
  architectural refactor. No new features, no further structural
  refactors, and no script-layout changes are expected on the v0.42.x
  line after this release.
