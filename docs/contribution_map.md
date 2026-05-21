# Contribution Map

## Contribution 1 - R-D-E metrology

Core claim:

Rate-Distortion-only evaluation is insufficient or fragile when energy is
included and codec families occupy disjoint computational regimes.

Evidence:

- measured image/audio/video benchmark rows;
- energy scale differences across codec families and hardware paths;
- documented limitations of BD-rate-style comparisons when operating domains do
  not overlap cleanly;
- R-D-E plots and diagnostic summaries;
- Pareto/frontier, oracle, baseline, and regret formulations.

Current support:

- strong for the measured settings in the repository artifacts;
- requires careful wording outside those measured settings;
- strongest when phrased as a bounded methodological warning rather than a
  general rejection of R-D analysis.

Future work:

- tighter formalism for R-D-E indicators;
- broader literature positioning;
- broader datasets and hardware;
- uncertainty modeling for energy and quality;
- more explicit treatment of non-overlapping R-D-E surfaces.

## Contribution 2 - Adaptive R-D-E codec routing

Core claim:

A router can reduce operational regret by adapting codec selection to rate,
quality, and energy constraints.

Evidence:

- router architecture and profile-based selection;
- operational profiles/regimes;
- oracle and regret analyses;
- content-aware and system-aware policies;
- classical/neural switch analysis;
- validation on measured image/audio/video rows.

Current support:

- strong as an offline research prototype;
- strong for decision audit over measured benchmark rows;
- not an online deployment or managed service claim;
- not a claim that one codec family is preferable across all settings.

Future work:

- cross-machine validation;
- stronger and broader baselines;
- latency-focused analysis;
- broader datasets and codec families;
- online adaptation with explicit guardrails;
- uncertainty-aware policy acceptance.

## Contribution 3 - Reproducible codec evaluation through declarative contracts

Core claim:

Benchmark reproducibility is stronger when domains, datasets, codec
capabilities, ingestion contracts, and decision receipts are explicit
first-class objects.

Evidence:

- `DomainSpec`;
- `DatasetManifest`;
- codec specs and codec onboarding;
- decision receipts and decision replay;
- setup doctor;
- generated-output boundary;
- third-party attribution boundary;
- full pluggability proof from manifest plus measurements to router decision.

Current support:

- strong at repository/prototype level;
- strong for schema and contract validation;
- future external artifact evaluation would strengthen confidence across fresh
  machines and independent users.

Future work:

- external reproduction by a second environment;
- richer manifest provenance for toolchain and hardware details;
- standardized artifact bundles;
- stricter generated-output validation;
- public examples that use only redistributable fixture data.
