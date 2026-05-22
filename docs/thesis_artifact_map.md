# Thesis Artifact Map

## Scope

This document maps the external thesis text to the repository artifacts that
support it. It is descriptive: it does not rewrite the thesis, introduce a new
methodology, or replace the thesis as the complete scientific document.

The thesis contains the methodological formulation, measured results,
interpretation, limitations, and future work. This repository contains code,
contracts, setup scripts, tests, documentation, and tooling that support those
experiments and decision audits. Generated benchmark outputs are local
artifacts and are not versioned as source-code ground truth.

This repository does not claim to invent the general
Rate--Distortion--Energy framework. It provides a reproducible implementation
and experimental infrastructure for applying an R-D-E perspective to codec
evaluation and routing.

## Read-only thesis inspection

The mapping was produced by reading the following external files without
modifying them:

- `C:\Projects\TesiUnipd\main.tex`
- `C:\Projects\TesiUnipd\Capitoli\capitolo4.tex`
- `C:\Projects\TesiUnipd\Capitoli\capitolo5.tex`
- `C:\Projects\TesiUnipd\Capitoli\capitolo6.tex`
- `C:\Projects\TesiUnipd\Capitoli\capitolo7.tex`
- `C:\Projects\TesiUnipd\Capitoli\sommario.tex`, inspected because it is
  included by `main.tex` and summarizes the scientific framing.

## Thesis topic map

| Theme | File | Section / label | Approx. lines | Summary |
|---|---|---|---:|---|
| R-D-E methodology | `capitolo4.tex` | `Spazio Rate--Distortion--Energy`, `sec:image_rde`; `sec:video_rde`; `sec:cross_domain` | 387-451, 796-875, 908-983 | Chapter 4 presents R-D-E views for image/video and a cross-domain synthesis. |
| Limits of RD / BD-rate | `capitolo4.tex` | `Analisi matriciale BD-rate ed epsilon indicator`, `sec:video_heatmap` | 821-875 | BD-rate is described as the traditional R-D view and is explicitly limited once operational energy becomes a third objective. |
| Classical vs neural comparison | `capitolo4.tex`, `capitolo7.tex` | image/audio/video result sections; `sec:conclusions_results` | ch.4: 219-983; ch.7: 44-83 | The thesis compares classical and neural codecs as occupying different measured R-D-E regions, not as a single fixed family ranking. |
| Oracle and regret | `capitolo5.tex`, `capitolo7.tex` | `sec:oracle_regret`; `sec:image_routing_validation`; `sec:content_aware_routing`; `sec:conclusions_results` | ch.5: 233-296, 428-589, 661-1075; ch.7: 85-112 | The thesis defines an R-D-E oracle, regret, robust baselines, and content-aware policy evaluation. |
| Predictive router | `capitolo5.tex`, `capitolo6.tex` | `sec:content_aware_routing`; `sec:router_arch_content_predictor` | ch.5: 661-1075; ch.6: 904-1147 | The predictor is consultive; router feasibility, quality guards, system filters, and fallback remain decisive. |
| Regime simulation / switch analysis | `capitolo6.tex` | `sec:router_arch_regime_simulation`; `subsec:router_arch_regime_switch_analysis`; `subsec:router_arch_regime_results` | 1340-1597 | Operational regimes, feasible oracle regret, quality-contract behavior, and classical/neural switch diagnostics are documented. |
| Quality contract | `capitolo6.tex` | `subsec:router_arch_regime_quality_contract`; invariants section | 1426-1438, 1637-1669 | Oracle and ingestion analyses separate feasible, quality-satisfying cases from infeasible cases. |
| Multi-domain validation | `capitolo4.tex`, `capitolo5.tex`, `capitolo6.tex` | `sec:cross_domain`; `sec:domain_independence`; `subsec:router_arch_ingestion_multidomain` | ch.4: 908-983; ch.5: 38-96; ch.6: 441-478 | The thesis separates domain semantics while keeping a common R-D-E decision contract. |
| Reproducibility by architecture | `capitolo6.tex` | `sec:router_arch_ingestion`; `sec:router_arch_observability`; `sec:router_arch_invariants`; `sec:router_arch_limits` | 252-508, 1148-1338, 1637-1862 | DomainSpec, DatasetManifest, receipts, replay, provenance tiers, tests, and explicit limits are treated as architectural concerns. |
| Setup/doctor/licensing/third-party boundary | Repository docs, not thesis text | `docs/router_setup.md`, `NOTICE`, `THIRD_PARTY_NOTICES.md` | n/a | These are repository-publication artifacts that support reproducibility and redistribution boundaries; they are not part of the external thesis text inspected here. |

## Thesis sections

### Chapter 4

Chapter 4 contains the measured evidence base. It covers experimental setup,
energy methodology, image/audio/video measurements, R-D-E spaces, BD-rate and
epsilon-indicator discussion, and cross-domain synthesis.

Connected repository artifacts:

- benchmark scripts under `src/benchmark/`;
- local generated outputs under `results/` when present on the author's
  machine;
- utility scripts under `src/utils/`;
- `src/router/analysis/audio_video_policy_validation.py` for router-facing
  validation of measured audio/video rows.

Generated results are local artifacts and are not versioned in the public
repository layout.

### Chapter 5

Chapter 5 turns the measured R-D-E benchmark into an adaptive decision problem.
It defines the admissible pool, weighted cost, R-D-E oracle, regret, image
routing validation, content-aware policies, LOIO/LODO evaluation, bootstrap
uncertainty, and the scope of the formulation.

Connected repository artifacts:

- `src/router/rde_router.py`;
- `src/router/analysis/policy_comparison.py`;
- `src/router/analysis/content_oracle_analysis.py`;
- `src/router/analysis/content_predictor_interpretability.py`;
- `src/router/analysis/neural_inclusive_oracle.py`;
- `src/router/analysis/neural_inclusive_predictive_router.py`;
- `docs/research_framing.md`;
- `docs/contribution_map.md`;
- `docs/claim_audit.md`.

### Chapter 6

Chapter 6 describes the router architecture and implementation. It maps the
runtime/offline split, ingestion contracts, codec capabilities, system policy,
energy provenance, content predictor integration, observability, operational
regime simulation, invariants, and implementation limits.

Connected repository artifacts:

- `src/router/core/domain_spec.py`;
- `src/router/core/dataset_manifest.py`;
- `src/router/core/dataset_ingestion.py`;
- `src/router/core/codec_onboarding.py`;
- `src/router/observability/decision_receipt.py`;
- `src/router/observability/decision_replay.py`;
- `src/router/analysis/operational_regime_simulation.py`;
- `src/router/analysis/operational_regime_diagnostics.py`;
- `scripts/setup/setup_router.py`;
- `scripts/setup/doctor.py`.

### Chapter 7

Chapter 7 synthesizes the work, states main results, documents limits, and
lists future directions. It already states that energy measurements are
platform-dependent, that content-aware audio/video routing remains future work,
and that a production-style deployment/orchestrator is outside the current
scope.

Connected repository artifacts:

- `docs/research_framing.md`;
- `docs/contribution_map.md`;
- `docs/claim_audit.md`;
- `docs/router_roadmap_status.md`;
- `THIRD_PARTY_NOTICES.md`.

## Repository artifacts

| Repository artifact | Purpose | Thesis chapter/section | Notes |
|---|---|---|---|
| `src/router/analysis/neural_inclusive_oracle.py` | Offline oracle analysis over the full image codec pool | Chapter 5, neural-inclusive/oracle discussion; Chapter 7 synthesis | Supports regime-dependent classical/neural interpretation. |
| `src/router/analysis/neural_inclusive_predictive_router.py` | Offline predictive routing evaluation against full-pool oracle labels | Chapter 5, predictive routing; Chapter 7 limits/future work | Image-domain validation only; not an audio/video predictor. |
| `src/router/analysis/operational_regime_simulation.py` | Simulates operational regimes and reports regret, feasibility, and policy summaries | Chapter 6, `sec:router_arch_regime_simulation` | Produces CSV/JSON primary artifacts; plot rendering is separate. |
| `src/router/analysis/operational_regime_diagnostics.py` | Read-only diagnostics for operational-regime artifacts | Chapter 6, regime diagnostics | Helps audit objective consistency and quality-contract behavior. |
| `src/router/analysis/audio_video_policy_validation.py` | Builds and validates measured audio/video router-facing artifacts | Chapter 4 cross-domain evidence; Chapter 6 multi-domain ingestion boundary | Validation over measured rows, not a new benchmark campaign. |
| `src/router/core/domain_spec.py` | Declares domain-specific rate/quality/energy contracts | Chapter 6, `DomainSpec` ingestion component | Enables CSV validation and router column mapping. |
| `src/router/core/dataset_manifest.py` | Declares dataset identity, items, metadata, and splits | Chapter 6, `DatasetManifest` ingestion component | Does not grant rights to external datasets. |
| `src/router/core/dataset_ingestion.py` | Joins manifests and measurement CSVs into router-ready R-D-E rows | Chapter 6, ingestion procedure | Operates on user-provided measured rows. |
| `src/router/core/codec_onboarding.py` | Validates measured codec rows and optional codec specs | Chapter 6, codec onboarding/pluggability | Does not validate scientific correctness of the measurements. |
| `src/router/observability/decision_receipt.py` | Defines and validates stable decision receipts | Chapter 6, decision receipt and invariants | Records decision context independently of execution. |
| `src/router/observability/decision_replay.py` | Replays decisions from receipts for deterministic intra-version checks | Chapter 6, decision replay | Supports auditability, not cross-version equivalence by itself. |
| `scripts/setup/setup_router.py` | Router-only development setup | Repository-publication support, not thesis methodology | Does not install benchmark stack, datasets, codecs, or checkpoints. |
| `scripts/setup/doctor.py` | Read-only environment doctor | Repository-publication support, not thesis methodology | Reports optional external tools without installing them. |
| `scripts/setup/validate_router_environment.py` | Automated router environment validation manifest | Repository-publication support, not thesis methodology | Captures setup/replay evidence for fixed R-D-E fixtures; does not reproduce benchmarks. |
| `THIRD_PARTY_NOTICES.md` | Documents third-party boundaries | Repository publication / reproducibility boundary | Clarifies that external datasets, codecs, models, tools, and generated outputs are not relicensed. |

## R-D-E claim boundary

No inspected thesis passage states that the work invented the general
Rate--Distortion--Energy idea. Some phrases are broad and should be read as
domain-specific operationalization rather than conceptual ownership:

| Location | Phrase / pattern | Why it could be ambiguous | Safer interpretation |
|---|---|---|---|
| `capitolo7.tex`, approx. lines 15-28 | "è stato costruito un benchmark Rate--Distortion--Energy" | Could be overread as introducing R-D-E itself if detached from context. | The work builds a measured R-D-E benchmark for the selected codec domains. |
| `capitolo7.tex`, approx. lines 30-35 | "formulato come routing R-D-E dominio-specifico" | Correct as routing formulation, but should not be read as claiming the general R-D-E concept. | The work instantiates an R-D-E routing formulation for measured codec candidates. |
| `capitolo7.tex`, approx. lines 237-244 | "costruisce progressivamente un framework R-D-E" | Could sound broad if quoted without the preceding limitations. | The framework is the repository/thesis implementation for this codec-evaluation setting. |
| `capitolo5.tex`, approx. lines 1079-1083 | "La formulazione proposta è generale rispetto al dispositivo" | Could be overread as universal experimental validity. | The decision variables are device-parameterized; measured values still require hardware-specific provenance. |

Recommended wording when summarizing the thesis:

> This work does not claim to introduce the general
> Rate--Distortion--Energy perspective. Rather, it operationalizes it for codec
> evaluation and routing under explicit energy and system constraints.

Italian equivalent:

> Il lavoro non introduce il concetto generale di
> Rate--Distortion--Energy, ma lo adotta come prospettiva operativa per valutare
> e selezionare codec in presenza di vincoli energetici espliciti.

## Boundaries

- R-D-E is not claimed as a general original invention.
- The contribution is the domain-specific instantiation, measurement pipeline,
  adaptive routing framework, declarative contracts, provenance, and
  reproducibility infrastructure.
- Third-party datasets, codec binaries, model weights, checkpoints, and metric
  tools are not redistributed by this repository.
- Full benchmark outputs are generated locally and are not versioned as public
  source artifacts.
- Energy and timing measurements depend on hardware, drivers, codec builds,
  thermal state, operating system, and measurement backend.
- The router is a research prototype and decision-audit framework, not a
  production-ready service.
