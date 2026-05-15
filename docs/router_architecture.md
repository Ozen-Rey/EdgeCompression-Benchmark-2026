# Router Package Architecture

This document describes the router package layout after the v0.42 refactor,
the v0.42.15 removal of legacy top-level wrapper modules, and the v0.42.34
completion of the profile-runner extraction.

## Package Layout

```text
src/router/
  core/
  codecs/
  calibration/
  adaptation/
  observability/
  analysis/
  cli.py
  context.py
  execution.py
  outputs.py
  pipeline.py
  presentation.py
  profile_runner.py
  rde_router.py
  report.py
  version.py
```

`src/router/core/` contains the stable R-D-E primitives: points, selection,
normalization profiles, router config helpers and quality threshold resolution.

`src/router/codecs/` contains codec capability metadata and execution-plan
support, external codec specification validation, probe, dry-run, benchmark,
R-D-E export and registry tooling, plus the simple image encoder backend.

`src/router/calibration/` contains local calibration, calibration application,
calibration bundle validation and calibration impact audit tooling.

`src/router/adaptation/` contains content, system, context and energy
provenance policies. These modules build report-only or explicitly applied
adaptation inputs without changing the core scoring definitions.

`src/router/observability/` contains decision receipts, replay, run manifest
construction, feedback logging and analysis, shadow comparison/validation,
normalization consistency and router audit tools.

`src/router/analysis/` contains offline and paper-facing analysis tools for
content-aware experiments, oracle baselines, classifier sweeps, overhead tables
and paper artifacts.

## Top-level Modules

The top-level modules under `src/router/` split responsibilities as follows:

`src/router/rde_router.py` is the CLI entrypoint only. It wires `expand_argv_with_config`,
the argparse parser from `cli.py`, a fresh `RouterContext`, and a single call
into `pipeline.run_router(...)`. The `if __name__ == "__main__"` block adds the
infeasible-request exception wrapper that prints actionable hints and exits
with code 2. After v0.42.34 it contains no orchestration logic of its own.

`src/router/cli.py` owns argparse construction (`build_router_arg_parser`) and
the CLI/config schema. It does not execute the router.

`src/router/pipeline.py` owns global orchestration (`run_router`): system
probes, calibration bundle handling, CSV loading and row diagnostics,
codec-availability filtering, normalization resolution, the per-profile loop,
and the single-profile branch. It hosts the side-effect-free preprocessing
helpers (`normalize_token`, `parse_codec_list`, `is_neural_codec`,
`filter_points_by_codec_availability`, `apply_system_aware_policy`,
`annotate_points_with_calibration_provenance`, `summary_row_from_report`,
`topk_rows_from_report`) and re-exports `build_weights_for_profile` and
`run_profile` from `profile_runner.py`. Pipeline must not import from
`rde_router.py`.

`src/router/profile_runner.py` owns per-profile orchestration (`run_profile`):
weight resolution, system/content policy application, content-classifier
report, preferred-candidate resolution and J_RDE selection, plus the helpers
that feed it (`build_weights_for_profile`, `build_time_guard_report`,
`build_content_classifier_router_report`, `apply_preferred_candidate_override`).
Profile_runner must not import from `rde_router.py`. It is the canonical
location for any helper used only inside the per-profile loop.

`src/router/context.py` defines `RouterContext`, the metadata carrier that
flows through `run_router` and `run_profile` and accumulates the system,
content-policy, content-classifier, normalization, calibration, system-penalty
and time-guard reports consumed by `report.py`.

`src/router/report.py` assembles the structured router report. It consumes
`RouterContext` and decision metadata; it does not decide ranking.

`src/router/outputs.py` writes the report and summary/top-k CSVs to disk. It
does not decide ranking.

`src/router/presentation.py` prints the human-readable single-decision and
all-profiles selections to the console. It does not decide ranking.

`src/router/execution.py` owns side effects and backend execution once the
router has selected an execution plan; `execution_validation.py` validates
post-execution outputs.

`src/router/version.py` is the single source of truth for `ROUTER_VERSION`,
`FEATURE_GROUPS`/`FEATURE_LEVEL` and `DOMAIN_SUPPORT`. The same version is
mirrored in `pyproject.toml`.

## Dependency Direction

The router top-level modules form a strict one-way graph:

```text
rde_router  ->  pipeline  ->  profile_runner
                  |               |
                  +---->  context, report, outputs, presentation
                  +---->  adaptation/*, calibration/*, codecs/*, core/*,
                          observability/*, execution
```

Invariants enforced by the codebase (and by `tests/test_pipeline_module.py`):

- `rde_router.py` only imports from `pipeline.py`, `cli.py`, `context.py` and
  `core.router_config`. It does not import from `profile_runner.py`.
- `pipeline.py` must not import from `rde_router.py`. The historical lazy
  `pipeline -> rde_router` cycle was removed in v0.42.34; a textual import
  guard in `test_pipeline_module.py` keeps it from coming back.
- `profile_runner.py` must not import from `rde_router.py`.
- `report.py` consumes `RouterContext` and decision dicts. It does not decide
  ranking.
- `outputs.py` and `presentation.py` are pure I/O sinks. They do not decide
  ranking and they do not mutate the report schema.

## Decision Pipeline

The router decision path is intentionally explicit:

1. Load benchmark CSV rows and any explicitly named external codec rows.
2. Apply capability and executable filtering when requested.
3. Aggregate by codec/config when requested.
4. Validate and consume explicit calibration files or calibration bundles.
5. Normalize rate, distortion and energy terms.
6. Apply the quality guard and degraded fallback policy.
7. Score admissible candidates with the configured R-D-E weights.
8. Optionally prefer a safe content-policy candidate.
9. Assemble reports, receipts and audit metadata.
10. Optionally execute the selected backend plan and append feedback.

The pipeline does not perform automatic discovery of calibration, feedback,
external codec or validation artifacts. Those inputs are used only when named
explicitly by the caller.

## Imports And CLIs

Legacy top-level wrapper modules such as `src.router.rde_database` and
`src.router.external_codec_spec` were removed in v0.42.15. Official imports now
use subpackage paths, for example:

```python
from src.router.core.rde_database import RDEPoint, select_best_rde
from src.router.codecs.external_codec_spec import validate_external_codec_spec
from src.router.calibration.calibration_bundle import validate_calibration_bundle_manifest
from src.router.observability.decision_receipt import build_decision_receipt
```

Old top-level module paths are no longer supported. Tool CLIs should be invoked
with the subpackage module path:

```powershell
python -m src.router.codecs.external_codec_spec --help
python -m src.router.calibration.calibration_apply --help
python -m src.router.observability.decision_replay --help
python -m src.router.analysis.content_oracle_analysis --help
```

The main router CLI remains:

```powershell
python -m src.router.rde_router --help
```

After an editable install (`pip install -e .`), the `rde-router` console
entry point introduced in v0.42.30 calls the same `main()` function and is
equivalent to the `python -m` form.
