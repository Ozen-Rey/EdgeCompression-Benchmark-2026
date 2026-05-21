# R-D-E Router Research Framing

## 1. Motivation

Classical Rate-Distortion evaluation is not sufficient by itself when codec
selection is constrained by energy, hardware, and operational context. Bitrate
and energy can be weakly coupled: a codec can reduce rate while increasing
compute cost, memory pressure, latency, or hardware-specific energy. This is
especially visible when classical and neural codecs coexist, because the two
families often occupy different computational regimes.

Traditional BD-rate and BD-PSNR summaries are useful when curves are measured
over comparable quality ranges and overlapping operating domains. They become
fragile when the relevant comparison includes a third axis that is not a smooth
function of rate or quality. Energy-disjoint comparisons can require
extrapolation across non-overlapping curves or surfaces, which is
methodologically risky. A single scalar gain on the rate axis can hide whether
the operating point is feasible under a power budget, thermal state, hardware
availability, or application latency constraint.

The motivation for the R-D-E router is therefore not to replace
Rate-Distortion analysis, but to place it inside a broader decision problem:
codec evaluation should account for the rate, quality, and energy tradeoff that
will actually govern a constrained deployment or offline selection scenario.

## 2. Core thesis

Codec evaluation and selection should be treated as a Rate-Distortion-Energy
decision problem rather than a purely Rate-Distortion comparison, especially
when classical and neural codecs coexist under heterogeneous hardware and
operational constraints.

In this framing, the central object is not only a compression curve, but a set
of measured R-D-E candidates subject to domain-specific quality contracts and
operational constraints. The router is a research framework for auditing such
decisions: it records which candidates are feasible, which candidate is
selected under a profile, and how that selection compares with an offline
oracle or a robust baseline.

R-D-E is treated here as an adopted evaluation perspective, not as a newly
invented concept. The contribution is the operationalization of this
perspective for codec evaluation, routing, and reproducibility across measured
image, audio, and video settings.

## 3. Contribution A - R-D-E metrology

The methodological contribution is the shift from R-D comparison to R-D-E
metrology. The work treats energy as a first-class measurement axis alongside
rate and quality, then evaluates codec behavior through Pareto/frontier views,
oracle policies, regret, quality contracts, and normalized multi-objective
costs.

This contribution is deliberately not framed as a simple family hierarchy. The
measured setting supports a more bounded statement: classical and neural codecs
can occupy different regions of the R-D-E space, and the preferred family can
depend on the active objective, quality floor, rate pressure, and energy
constraint. The comparison is therefore operational and profile-dependent, not
a fixed ranking of codec families.

Already supported by current artifacts:

- measured image/audio/video R-D-E rows;
- explicit energy columns and energy provenance;
- quality contracts through `DomainSpec`;
- offline oracle and regret analyses;
- operational-regime simulation and classical/neural switch diagnostics;
- documentation of BD-rate limitations for cross-paradigm comparisons.

Still worth strengthening in future work:

- a tighter formal definition of R-D-E indicators;
- broader hardware and dataset coverage;
- uncertainty estimates for measured energy and quality;
- more explicit treatment of non-overlapping R-D-E surfaces;
- comparison against additional multi-objective baselines.

## 4. Contribution B - Adaptive R-D-E routing

The systems contribution is an adaptive R-D-E routing prototype. The router is
a decision system over measured candidates: it filters by feasibility, applies
profile-specific priorities, records diagnostics, and emits auditable decision
artifacts. It supports operational profiles, quality floors, system-aware
policies, content-aware predictors, oracle/regret evaluation, regime
simulation, and classical/neural switch analysis.

The current system should be described as a research prototype and offline
validation framework. It supports decision audits over measured benchmark rows;
it is not an operational service claim. The strongest supported claims are
about reproducible decision analysis within the measured setting: the router can
consume image, audio, and video R-D-E rows, apply domain-specific contracts, and
compare selected policies against oracles and baselines.

Key artifacts:

- `src.router.rde_router` for profile-based R-D-E decisions;
- `DomainSpec` for metric/domain-specific CSV contracts;
- image content-aware predictor analyses and neural-inclusive oracle studies;
- operational-regime simulation and diagnostics;
- real audio/video router validation from measured rows;
- router reports, summaries, and decision observability artifacts.

## 5. Contribution C - Reproducibility by architecture

Reproducibility is not treated as a README afterthought, but as an
architectural property of the experimental system. The repository separates
source code from generated outputs, keeps benchmark results local/generated,
and makes the boundaries around datasets, checkpoints, codecs, and external
tools explicit.

This architecture includes:

- `DomainSpec` for explicit rate, quality, energy, direction, and item-id
  contracts;
- `DatasetManifest` for explicit dataset item metadata and splits;
- codec onboarding and external codec specs for measured-codec integration;
- decision receipts and provenance reports;
- setup scripts scoped to router development only;
- a read-only environment doctor;
- third-party notices and generated-output boundaries;
- artifact reports that distinguish measured data from derived diagnostics.

The practical result is a repository that can expose its decision logic without
pretending to redistribute datasets, checkpoints, codec binaries, or generated
benchmark outputs.

## 6. What is already demonstrated

The current repository supports the following bounded claims:

- image R-D-E benchmark measurements exist and are used by router analyses;
- audio/video R-D-E benchmark rows are available for measured validation;
- the router consumes measured image/audio/video R-D-E rows through
  domain-specific contracts;
- neural-inclusive oracle and predictive routing analyses have been performed
  for the image setting;
- operational-regime simulation and switch analysis support profile-dependent
  interpretation;
- quality contracts and objective-consistency checks are explicit in offline
  analyses;
- dataset and codec pluggability are exercised through manifest/measurement
  onboarding;
- setup/doctor/repository hygiene make the router development environment more
  inspectable;
- licensing and third-party boundaries are documented.

## 7. What is not yet demonstrated

The current work does not yet support the following broader claims:

- a full cross-machine benchmark campaign;
- a full audio/video content-aware predictor;
- online deployment as a managed operational service;
- automatic installation of the benchmark stack;
- compatibility with every model/checkpoint/reference implementation;
- dominance of neural codecs across all settings;
- transferability of energy measurements across hardware, drivers, thermal
  state, and codec builds;
- automatic reproduction of every thesis number from a fresh checkout without
  external datasets, tools, and checkpoints supplied by the user.

## 8. Future extensions

Natural extensions include:

- cross-machine validation with repeated energy measurements;
- hardware-aware calibration and transfer diagnostics;
- online adaptive routing with bounded update rules;
- real-time constraints and latency-first profiles;
- broader codec/model coverage;
- formal R-D-E indicators and frontier metrics;
- uncertainty-aware routing;
- multi-objective regret bounds;
- application-weighted R-D-E capacity measures;
- a reproducibility protocol for codec evaluation that records dataset, codec,
  toolchain, driver, hardware, and energy-backend provenance together.
