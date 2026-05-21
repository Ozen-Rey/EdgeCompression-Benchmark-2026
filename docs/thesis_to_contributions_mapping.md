# Thesis-to-Contributions Mapping

## Chapter 4

Chapter 4 provides the measurement base for the R-D-E metrology contribution.
Its role is evidence, not routing logic.

Mapped contributions:

- R-D-E benchmark evidence across image, audio, and video;
- cross-domain measurements of rate, quality, time, and energy;
- empirical rate/quality/energy tradeoffs;
- evidence that energy cannot be assumed to follow bitrate monotonically;
- domain-specific metrics and measurement protocols.

Relevant repository artifacts:

- benchmark scripts under `src/benchmark/`;
- generated local outputs under `results/` when available locally;
- image/audio/video summary utilities under `src/utils/`;
- real audio/video validation documentation in
  `docs/router_real_audio_video_validation.md`.

Bounded wording:

- "The measured setting supports an R-D-E interpretation of codec behavior."
- "The results show profile-dependent tradeoffs among rate, quality, and
  energy."
- Avoid wording that treats the measured hardware as representative of every
  machine.

## Chapter 5

Chapter 5 maps most directly to the R-D-E methodology and adaptive routing
contributions.

Mapped contributions:

- R-D-E decision formulation;
- oracle and regret framing;
- adaptive routing logic over measured candidates;
- content-aware experiments;
- neural-inclusive oracle and predictive routing;
- operational regime interpretation.

Relevant repository artifacts:

- `src.router.rde_router`;
- `src.router.analysis.policy_comparison`;
- `src.router.analysis.content_oracle_analysis`;
- `src.router.analysis.content_predictor_interpretability`;
- `src.router.analysis.neural_inclusive_oracle`;
- `src.router.analysis.neural_inclusive_predictive_router`;
- `src.router.analysis.operational_regime_simulation`;
- `docs/content_aware_paper_notes.md`;
- `docs/router_content_aware.md`.

Bounded wording:

- "The router reduces operational regret in the evaluated offline protocols."
- "Neural selection is regime-dependent in the measured image setting."
- "The predictor is evaluated without using the target image's measured R-D-E
  candidates to choose the codec."
- Avoid wording that presents the predictor as complete for audio/video or as
  an online service.

## Chapter 6

Chapter 6 maps to the architecture and reproducibility contribution.

Mapped contributions:

- system architecture of the router;
- domain and dataset contracts;
- pluggability through manifest and measurement ingestion;
- codec onboarding;
- setup/doctor workflows;
- decision observability and replay;
- provenance reports;
- licensing and third-party boundaries where relevant.

Relevant repository artifacts:

- `DomainSpec`;
- `DatasetManifest`;
- codec onboarding and external codec specs;
- full pluggability proof;
- decision receipts, decision replay, and observability modules;
- `scripts/setup/setup_router.py`;
- `scripts/setup/doctor.py`;
- `NOTICE`;
- `THIRD_PARTY_NOTICES.md`.

Bounded wording:

- "The repository makes reproducibility an architectural concern through
  explicit contracts and provenance."
- "The setup flow prepares the router development environment, not the full
  benchmark stack."
- "Third-party datasets, codecs, models, and generated outputs remain outside
  the repository license."

## Chapter 7

Chapter 7 should synthesize the three contribution families and state limits
plainly.

Mapped contributions:

- synthesis of R-D-E metrology;
- synthesis of adaptive routing;
- synthesis of reproducibility by architecture;
- limitations and future work.

Recommended emphasis:

- R-D-E is a decision framing, not only an added column;
- routing is supported as an offline research prototype over measured rows;
- reproducibility is supported through explicit contracts and boundaries;
- energy measurements are hardware- and toolchain-dependent;
- future work should address cross-machine validation, online adaptation,
  uncertainty, and broader audio/video policies.
