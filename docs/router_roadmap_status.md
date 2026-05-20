# Router Roadmap Status After v0.42 Refactor Freeze

This document situates the R-D-E router relative to its original roadmap. It
records which milestones are already shipped, what the v0.42.x refactor line
changed, and which roadmap items remain open. It is meant as a single
landing page for "where are we?" questions after the v0.42 refactor freeze.

## Current stable artifact

The current stable artifact is **router v0.42.40**, tagged as
`router-v0.42.40`. v0.42.40 is the citable end of the v0.42.x architectural
refactor: it changes no observable router behavior (same CLI flags, same
report schema, same ranking / scoring / normalization / calibration /
external-codec behavior) and is intended as the baseline against which any
post-refactor work is measured. See `CHANGELOG.md` for the full v0.42.40
freeze note.

This document itself ships in v0.42.41, which is a documentation-only
release on top of v0.42.40.

**Active line.** v0.43.0 opened the paper / methodology track on top
of the v0.42.40 freeze. It adds offline analysis only — specifically
the `src.router.analysis.policy_comparison` module with paired
bootstrap confidence intervals on the content-aware policies. v0.43.1
added `src.router.observability.decision_explanation`, the
per-decision Markdown/JSON render. v0.43.2 added
`src.router.analysis.content_predictor_interpretability`, a paper /
methodology hardening pass that audits class balance, fits surrogate
decision trees against the kNN, runs a logistic regression with
pairwise interactions on the binary JPEG-vs-JXL subproblem, and
produces leave-one-feature-out and permutation attribution. v0.43.3
added `src.router.analysis.neural_inclusive_oracle`, which lifts the
analysis pool from the classical triple JPEG/JXL/HEVC to the full
benchmark (classical + JPEG_AI + Ballé + Cheng + ELIC + TCM + DCAE)
and quantifies when and where neural codecs become oracle-optimal
under each operational profile and quality floor. v0.43.4 adds
`src.router.analysis.neural_inclusive_predictive_router`, which
turns the oracle audit into an operational evaluation: under
leave-one-image-out and leave-one-dataset-out, five policies (robust
global baseline, source-aware majority, full-pool kNN, classic-only
kNN, and the full-pool oracle as upper bound) are scored against
the per-image full-pool oracle, with paired bootstrap CIs on
mean_regret and relative_reduction_vs_global plus neural-family
precision/recall against the oracle's classical/neural label. The
test image's measured R-D-E candidates are never used to choose
the codec; they are used only afterwards to look up the realised
J_RDE of the predicted pair and to compute regret against the
oracle. v0.43.5 integrates these predictive and neural-inclusive
results into thesis-facing text for Chapters 5 and 6. v0.43.6 adds
`src.router.analysis.operational_regime_simulation`, an offline
operational-regime simulation and predictive factor ablation that
compares global, metadata-only, system-only, metadata-plus-system,
classic-only and oracle policies across normal, bandwidth-limited,
energy-saving, battery/thermal, no-CUDA and memory/VRAM-pressure
regimes. It also emits paper/demo plot data and optional PNGs for
energy saving vs regret reduction, neural selection rate, winner
distributions, family confusion and rate-pressure sweeps. v0.43.2
remains the canonical classic-only interpretability/class-imbalance
audit; v0.43.3 is the neural-inclusive oracle complement; v0.43.4 is
the neural-inclusive predictive-router evaluation; v0.43.6 is the
operational-regime simulation layer. v0.43.6.1 adds
`src.router.analysis.operational_regime_diagnostics`, a read-only
diagnostic pass before any objective/quality fix. It inspects PSNR vs
SSIMULACRA2 metric contracts, no-leakage realized quality violations,
negative regret rows, objective-consistency hypotheses,
rate-pressure transition points and winner-distribution duplicate
appearance. The runtime router, the ranking score, and the
operational report schema are unchanged across these releases.
v0.43.6.2 adds objective-consistency accounting to the offline
simulation so regret is measured against the feasible oracle in the
same objective space, with objective gain against the global baseline
reported separately. It also adds neural/classical switch analysis:
per-image and aggregate comparisons of the best classical and neural
candidates, switch reasons, rate-pressure transition points and
plot-ready frontier data.
v0.43.6.3 makes the oracle quality contract explicit in the
operational-regime simulation: feasible oracles must satisfy the
active quality floor, infeasible cases are reported separately, and
predictive realized quality violations are distinguished from oracle
violations. It also adds a report-only expected-quality gate shadow
comparison based only on training-fold quality statistics.
v0.43.6.4 splits operational-regime artifact generation from plot
rendering: the simulation writes CSV/JSON only, while
`src.router.analysis.operational_regime_plots` renders PNGs
best-effort from those CSVs in a separate, rerunnable step.
v0.44.0 adds the `DomainSpec` foundation for generalized image/audio/video
R-D-E schemas: built-in metric/domain specs, offline spec and CSV validation,
and a minimal CLI for inspecting and validating schema contracts. The core
`J_RDE` formula, ranking behavior, benchmark data, execution backend,
external codec system and historical reports remain unchanged.
v0.44.1 adds `DatasetManifest`, a JSON schema and validator for pluggable
image/audio/video datasets. It records source item ids, relative paths,
splits and media metadata, exports a flat item table, and validates domain
compatibility against `DomainSpec` without changing router runtime behavior.

## Original roadmap recap

The R-D-E router was originally scoped along three parallel axes:

1. **Selection correctness.** A reproducible R-D-E (rate-distortion-energy)
   selector with explicit admissibility filtering, an explicit ranking
   score, and an auditable "why this codec/configuration?" trail.
2. **Energy and system awareness.** Energy measurement with explicit
   provenance, bounded-cost probing on Windows, optional system-penalty
   policies, and an audit trail for energy-tier compatibility decisions.
3. **Adaptation and feedback.** A feedback / calibration loop that is
   strictly observational by default, with shadow proposals, offline
   validation, an explicit promotion gate, and a calibration bundle
   lifecycle that ties calibration artifacts to the codec configurations
   they were measured against.

Around these axes the roadmap added content-aware extensions (image-content
features, oracle/classifier analyses), an external-codec pipeline
(spec → probe → dry-run → benchmark → R-D-E export → registry integration),
and a refactor track that turns the historical flat module layout into a
maintainable subpackage layout.

## Completed milestones

The following milestones are considered shipped and reachable from the
v0.42.40 baseline. Each is implemented under `src/router/` and exercised by
the smoke dispatcher (`scripts/run_router.ps1`) and the pytest suite.

- **Energy backends.** Pluggable energy backends with explicit provenance.
- **Strict energy mode.** Failures are surfaced instead of silently
  falling back to a less reliable backend.
- **Windows energy provenance.** GPU-partial provenance on Windows is
  reported as such; the router does not over-claim full-system measurement.
- **Feedback logging and analysis.** Append-only observational feedback
  log plus a read-only prediction-audit analysis pass.
- **Shadow calibration proposals.** Calibration proposals derived from the
  feedback log, kept strictly shadow (no implicit promotion).
- **Proposal validation.** Offline validation pass for shadow proposals,
  independent of the runtime decision path.
- **Promotion gate.** Explicit promotion gate that converts a validated
  proposal into a candidate profile, never bypassing manual opt-in.
- **Calibration bundle lifecycle.** Manifest, consumption, codec
  fingerprints, staleness gating, and impact audit for calibration bundles.
- **Shadow decision comparison and validation.** Read-only offline
  comparison and methodology gate for shadow decisions.
- **Decision receipt replay.** Audit-replay receipt plus an offline
  reproducibility check.
- **Overhead and effectiveness audits.** Read-only performance and
  baseline-policy audits, including cost-explainability completion.
- **Normalization consistency audit.** Previous-receipt report-only audit
  for normalization consistency.
- **Energy provenance tier and policy.** Tier reporting, compatibility
  audit, and a shadow tier-policy audit, all report-only.
- **External codec pipeline.** Specification schema, probe / fingerprint,
  single-input dry-run, raw-measurement benchmark, R-D-E exporter, and
  explicit registry integration as router input.
- **Content-aware routing artifacts.** Source-aware and source-agnostic
  classifiers, an oracle reference, and offline benchmark / overhead /
  paper-artifact scenarios in the dispatcher.
- **Package refactor and freeze.** Full move to `src/router/` subpackages,
  removal of legacy wrappers, single canonical import per module, English
  CLI/error surface, RouterContext-only reports, CSV row diagnostics, and
  a citable freeze point at v0.42.40.

## What v0.42.x changed

The v0.42.x line is **structural, not behavioral**. The selection logic,
the report schema, and the CLI flags are unchanged. What changed:

- **Package layout.** Router code lives under `src/router/` with stable
  subpackages (`core/`, `calibration/`, `adaptation/`, `analysis/`,
  `codecs/`, `observability/`, `feedback/`, ...). See
  `docs/router_architecture.md` for the package map.
- **RouterContext-only reports.** Report assembly flows through a single
  `RouterContext` object; pipeline-local report dictionaries were removed.
- **Legacy wrappers removed.** The top-level wrapper modules were removed
  in v0.42.15; new code imports from subpackage paths directly.
- **Dual imports removed.** Each module is reachable via exactly one
  canonical import path, enforced read-only by
  `python -m src.router.observability.legacy_import_audit`.
- **CSV row diagnostics.** Codec-filtering, calibration, and normalization
  reports surface row-level diagnostics, so infeasible decisions can be
  explained without re-running the pipeline.
- **English CLI and errors.** All user-facing argparse `help=` text and
  `raise ValueError(...)` messages in the router package are in English.
  Italian text is intentionally retained only in paper-figure labels and
  historical commit messages.
- **`pyproject.toml` and console scripts.** The package is
  `pip install -e .`-able, with console entrypoints mirroring the
  `python -m` invocations (`rde-router`, `rde-external-codec-*`,
  `rde-legacy-import-audit`).
- **Unified PowerShell dispatcher.** `scripts/run_router.ps1` is the
  single PowerShell entrypoint for every smoke scenario; the legacy
  `scripts/run_router_v*.ps1` files were removed in v0.42.39 and their
  bodies inlined as `Invoke-Scenario<Name>` handler functions, with
  static invariants enforced by `tests/test_smoke_dispatcher.py`.
- **Test scratch cleanup.** Smoke scenarios use explicit per-scenario
  `--basetemp .pytest_tmp_run_router_<scenario>` directories and clean
  them up before and after each run; `.pytest_tmp_*/` is git-ignored.

## Remaining roadmap toward v1.0 / v2.0

The following items were in the original roadmap and are **not** part of
the v0.42.x line. They are listed in roughly increasing scope rather than
in a committed order, and they are not all expected to land — the next
release will pick a small subset.

- **Drift detection.** Online detection of distribution drift in the
  feedback log relative to the calibration bundle baseline.
- **Active re-benchmarking.** Triggered re-runs of the benchmark when
  drift detection or proposal validation flags a configuration as stale.
- **Multi-domain core.** First-class image + video + audio routing in the
  same R-D-E core, beyond the current image-primary surface.
- **Generalized quality guard.** A quality guard that generalizes beyond
  per-codec hard thresholds (e.g. a learned guard or multi-metric guard).
- **Pareto-front routing.** Routing decisions made explicitly on the
  Pareto front instead of via a scalarized J score.
- **Uncertainty-aware routing.** Decisions that carry an uncertainty
  estimate and can defer or widen the admissible pool under high
  uncertainty.
- **Application profiles.** Pre-canned routing profiles per application
  class (archival, preview, real-time, edge upload, ...).
- **RAM/VRAM-aware routing.** Memory-budget-aware admissibility filtering
  in addition to the current latency/energy filters.
- **Context inheritance.** A report-only contextual prior that lets a
  decision inherit context (codec, content class, energy tier) from a
  preceding decision in the same batch. See "Candidate next feature"
  below.
- **Predictive models.** Cheap predictors for per-image quality, rate,
  and latency to short-circuit the full benchmark in fast paths.
- **Explainability.** A user-facing explainability surface on top of
  the existing audit trail.
- **API / service mode.** A long-lived router service with a stable
  request/response API, on top of the current one-shot CLI.
- **Dashboard.** A read-only dashboard over the audit and feedback logs.
- **Edge / cloud routing.** Routing decisions that include where to run
  the codec, not only which codec to run.
- **Privacy constraints.** First-class privacy constraints in the
  admissibility filter (e.g. forbid offloading certain content classes).
- **Real-time adaptive streaming.** Online adaptation for streaming
  workloads, with bounded per-frame decision latency.
- **Hybrid neural/classical routing.** Routing across hybrid pipelines
  that mix neural and classical codecs within a single asset.
- **Learned Pareto policy.** A learned policy over the Pareto front,
  validated offline before any runtime adoption.
- **Autonomous orchestrator.** A higher-level orchestrator that combines
  routing, calibration, feedback, drift detection, and re-benchmarking
  under a single control loop.

## Recommended next steps

1. **Freeze the refactor at v0.42.40 / v0.42.41.** v0.42.40 is the
   structural freeze; v0.42.41 is a documentation-only follow-up that
   adds this roadmap status page. No further structural refactor is
   planned on the v0.42.x line.
2. **v0.43.0 = paper / methodology package.** First step on top of the
   freeze: ship offline analysis that makes the existing prototype
   defensible as a paper artefact. v0.43.0 specifically adds
   `src.router.analysis.policy_comparison` — a single comparison table
   over the existing content-aware policies with paired bootstrap
   confidence intervals on `mean_regret` and
   `relative_reduction_vs_global`. No runtime changes.
3. **Open after v0.43.0:** confidence-gated content-aware routing
   (predictions accepted only when the classifier confidence exceeds a
   threshold), drift detection on the feedback log, and a context
   inheritance prior (report-only). See "Candidate next feature" below.
4. **No further refactor unless required by a feature.** Post-freeze
   refactors should be motivated by a specific feature on the roadmap
   above, not by general code hygiene, to keep the v0.42.40 baseline
   citable.

## Candidate next feature: Context Inheritance

Context inheritance is proposed as a **report-only contextual prior**,
not an online learning component.

The idea is that, within a routing session, a decision can inherit
context from a preceding decision in the same session — for example,
the previously selected codec/configuration, the previously observed
content class, or the previously reported energy provenance tier. The
inherited context becomes an additional signal in the decision report,
alongside the existing R-D-E score, system penalty, and content-aware
prediction.

Critically, this is **not** an online learner. There is no online
weight update, no implicit calibration change, and no implicit
modification of the admissible pool. The inherited context shows up
only in the decision report and in the audit trail, where it can be
inspected and validated offline before any future release decides to
let it actually move the decision. This matches the existing pattern
for content-aware predictions, which are advisory until the
admissibility and ranking-score gates accept them.
