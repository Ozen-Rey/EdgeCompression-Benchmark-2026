# Router Package Architecture

This document describes the router package layout after the v0.42 refactor and
the v0.42.15 removal of legacy top-level wrapper modules.

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
  report.py
  rde_router.py
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

`src/router/execution.py` owns side effects and backend execution once the
router has selected an execution plan.

`src/router/report.py` assembles router reports and related report structures.

`src/router/cli.py` owns argument parser construction and config expansion.

`src/router/context.py` carries run metadata flow used by router reports.

`src/router/rde_router.py` is the main entrypoint and orchestrator. It wires
argument parsing, data loading, filtering, scoring, reporting and optional
execution together.

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
