# v0.9 Content-Aware R-D-E Routing: Paper Notes

## Main Claim

Content-aware routing substantially reduces R-D-E regret relative to a robust global baseline while preserving safety through admissible-pool fallback.

## Key Results

| Method | Mean regret | Relative reduction |
|---|---:|---:|
| Robust global baseline | 0.09047 | 0.0% |
| Source-aware dataset majority | 0.01177 | 87.0% |
| Source-agnostic kNN, LOIO | 0.01167 | 87.1% |
| Source-agnostic kNN, LODO | 0.02197 | 75.7% |
| Per-image oracle | 0.00000 | 100.0% |

The stricter leave-one-dataset-out result is the most important generalization result: even when the target dataset is excluded from training, the source-agnostic classifier reduces mean regret by about 75.7%.

## Deployment Interpretation

Two deployment regimes emerge.

### Known-source batch routing

When the user knows that an image batch comes from a homogeneous source, the source-aware majority policy is preferred. It uses the source label to select a safe codec/configuration suggested by prior oracle analysis.

### Source-agnostic routing

When the source is unknown, the router uses a metadata-only kNN classifier with:

```text
feature_set = metadata_no_source
k = 7
```

This setting is selected from the stricter leave-one-dataset-out protocol.

## Overhead Interpretation

Pixel feature extraction has a mean overhead of about 30.52 ms per image.

This is:

```text
6.76x JPEG q=85 Tecnick encode time
4.84x JPEG q=85 global encode time
0.23x JXL d=1.0 global encode time
0.09x HEVC crf=15 global encode time
```

Therefore, full pixel features are not always suitable for the online fast path, especially when the selected codec is extremely fast. Metadata-only routing is the deployable default because it avoids full pixel scanning.

Pixel features remain useful for offline analysis, cached-feature pipelines, and future richer predictors.

## Safety Mechanism

The content-aware layer is advisory, not authoritative.

A content-aware prediction can only be selected if the predicted
codec/configuration is inside the admissible pool after all active constraints
have been applied and remains competitive under the active ranking score. If
the system penalty is disabled, the ranking score is `J_RDE`; if the system
penalty is applied, the ranking score is `J_total = J_RDE + lambda_sys P_sys`.
If the prediction is not admissible or not competitive, the router falls back
to the standard R-D-E/system-aware decision.

The validation script checks both cases:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-content-classifier-router
```

Observed cases:

```text
Global pool:
  classifier predicts JPEG q=85
  JPEG q=85 is not globally admissible
  fallback to HEVC crf=15

Tecnick source-filtered pool:
  classifier predicts JPEG q=85
  JPEG q=85 is admissible
  selected JPEG q=85
```

## Suggested Paper Figures

- Mean regret per method.
- Relative regret reduction per method.
- Sensitivity to k under LOIO and LODO.
- Oracle codec/configuration distribution.
- Feature overhead relative to encode time.

## Suggested Paper Tables

- Main content-aware benchmark table.
- Overhead table.
- Best-k sensitivity table.

## Policy comparison and bootstrap uncertainty

The router ships an offline analysis module
(`python -m src.router.analysis.policy_comparison`) that consolidates the
policies on a single comparison table and reports paired bootstrap
confidence intervals for `mean_regret` and
`relative_reduction_vs_global`. The module is read-only against
artefacts already produced by the content-aware pipeline; it does not
change the router runtime, the ranking score, or the operational report
schema.

Bootstrap is applied because the headline regret-reduction numbers are
estimated on the current N=96 multi-source corpus. Paired resampling on
image indices (1000 iterations by default, seedable) lets the report
state the relative reduction as a stable point estimate with an explicit
2.5%–97.5% interval, rather than as a single scalar that hides its own
sampling uncertainty.

The corresponding claim is therefore deliberately scoped: the regret
reduction is reported in a *multi-source heterogeneous setting*, where
the routing opportunity comes from the variance across sources
(Kodak / Tecnick / CLIC / …) rather than from a large per-source count.
The leave-one-dataset-out (LODO) protocol is the cross-source
generalization test; bootstrap CIs quantify the within-corpus sampling
uncertainty. No claim is made about a universal estimate over all
natural images.

Inputs consumed:

- `v09_content_oracle_by_image.csv` — per-image regret for the robust
  global baseline.
- `v09_metadata_policy_decisions.csv` — per-image regret and paired
  baseline for the source-aware (dataset-majority) policy.
- (optional) `v09_oracle_classifier_decisions.csv` or
  `v09_oracle_classifier_sweep_decisions.csv` — per-image kNN
  predictions under LOIO and LODO; the sweep variant supports
  `--classifier-feature-set` and `--classifier-k` filters.

Outputs: `policy_comparison.csv` (flat table, one row per policy) and
`policy_comparison.json` (same rows plus a top-level metadata block with
the bootstrap iterations, seed, CI quantiles, and an `inputs` /
provenance section that preserves "unavailable" markers when a policy
decisions file omits an optional column like `fallback_used`).

## Decision explanation and safety framing

The router ships a separate offline render
(`python -m src.router.observability.decision_explanation`) that turns
an existing router report JSON into a human-readable explanation of a
single decision. See `docs/router_decision_explainability.md` for the
full module documentation. The relevant framing for the paper is:

- The content-aware layer (content policy or content classifier) is
  **consultative**: it can suggest a codec/configuration based on
  metadata or pixel features.
- The router decides only **within the evaluated admissible pool**:
  the pool that passes the quality guard and the other active hard
  constraints (max rate / energy / time, codec availability,
  capability filtering, system policy / penalty exclusions).
- If the predictor's suggestion is not admissible, the router falls
  back to its own ranked choice on the admissible pool.
- If the predictor's suggestion is admissible but not competitive on
  the active ranking score (`J_RDE`, or `J_total` when the system
  penalty is applied), the router also falls back. The ranking score
  the suggestion must beat is the one actually in use, not a softer
  proxy.
- Only when the suggestion is both admissible and competitive does the
  router accept it. The decision is then recorded with selection
  reason `content_policy_preferred_candidate` or
  `content_classifier_preferred_candidate`.

This separation between *suggestion* and *admissibility/competitiveness*
is a deployable best practice: it caps the worst-case behavior of the
predictor by construction (catastrophic predictions are filtered out
before they can be selected), while still letting the predictor
recover the within-pool routing opportunity. The decision explanation
render makes this separation explicit in every decision it explains.

## Predictor interpretability and class imbalance

The router ships an offline interpretability audit
(`python -m src.router.analysis.content_predictor_interpretability`)
that opens the kNN metadata-only predictor without changing it. The
audit is read-only against the existing pipeline artefacts (oracle
by-image CSV, metadata/features CSV, optional classifier decisions
CSV) and emits a single JSON report plus a handful of CSV/TXT
artefacts.

The audit answers questions that the headline regret-reduction number
alone does not address.

**Class imbalance.** The current benchmark has 96 images across
4 datasets and 3 oracle configurations. The class distribution is
strongly skewed (HEVC oracle-optimal cases are very few). The audit
reports the global oracle distribution, the per-dataset distribution,
and the LOIO/LODO prediction distributions. When at least one LODO
training fold drops a class (e.g. the only HEVC cases are concentrated
in a single dataset), the audit emits
`class_missing_in_lodo_training_fold`. When a class has at most five
global examples, it emits
`minority_class_too_small_for_structural_claim`. Both warnings are
intended as scope markers in the paper, not as failure indicators.

**HEVC as a qualitative case study.** Because HEVC oracle-optimal
cases are few, the structural analysis (surrogate decision tree,
logistic regression, attribution) is performed on the binary
JPEG-vs-JXL subproblem. HEVC rows are kept in the report as a
qualitative case study with per-image listing (image_id, dataset,
features, oracle codec, classifier predictions under LOIO and LODO),
but no decision-boundary claim is made on them.

**Surrogate decision tree as interpretation, not replacement.** A
depth-sweep of decision trees (depth 1–4) is fit to imitate the kNN
predictions on the binary subproblem. For each depth the audit
reports `fidelity_to_knn`, `fidelity_to_oracle`, the number of
leaves, the confusion matrices, and an `export_text` rule listing.
The aim is to test whether a small number of interpretable thresholds
captures the kNN behavior. Shallow surrogates with high fidelity are
evidence that the predictor is largely explained by a small set of
splits; if depth must grow before fidelity rises, the boundary is not
that simple. The surrogate is never proposed as a replacement for the
kNN in the deployable router.

**Logistic regression with pairwise interactions.** A logistic
regression on the binary subproblem with explicit interaction terms
(`megapixels × aspect_ratio`,
`megapixels × orientation_class`,
`aspect_ratio × orientation_class`,
`aspect_ratio × resolution_class`) reports the coefficients sorted by
absolute magnitude. Coefficients are reported as descriptive
quantities; no claim of statistical significance is made unless an
appropriate test is performed, and none is performed here by design.
When the logistic regression cannot be fit (collinearity, single
class, separation), the audit records `logistic_regression_fit_failed`
or `logistic_regression_skipped_single_class_target` in its warnings
rather than reporting a fabricated number.

**Model-agnostic feature attribution.** Two attribution methods are
run against the surrogate's fidelity to the kNN: leave-one-feature-out
(zero out one column at a time) and permutation (shuffle one column
with a fixed seed). Both report `baseline_score`, `perturbed_score`,
`delta` and `rank`. SHAP is intentionally **not** required; if it is
installed it can be added as an optional path in a future release,
but the default behavior is to use only the two lightweight methods
above.

**Auto-generated interpretation.** The JSON report carries an
`interpretation` field with prudent wording (`suggests`,
`is consistent with`, `within this benchmark`) and explicitly
disclaims universal generalization. The audit is meant to make the
behavior of the predictor transparent for the paper's
predictive-methodology narrative, not to advertise it.

## Neural-inclusive R-D-E routing audit

The earlier content-aware analysis (`router_content_aware`,
`policy_comparison`, `content_predictor_interpretability`) restricted
the candidate pool to the deployable classical triple
JPEG / JXL / HEVC. That subset is a **classic-only ablation**: it
answers "given the deployable classical pool, can a lightweight
predictor reduce regret?", but it does not answer the broader R-D-E
question "when, and under which operational profile, do neural
codecs become oracle-optimal once they are included in the candidate
pool?".

v0.43.3 ships a separate offline audit
(`python -m src.router.analysis.neural_inclusive_oracle`) that uses
the full image benchmark (classical + JPEG_AI + Ballé + Cheng + ELIC
+ TCM + DCAE) and computes per-pool / per-profile / per-floor
oracles. The router runtime is not changed; the audit consumes only
existing benchmark artefacts (e.g.
`results/images/image_4dataset_RDE_paper_ready.csv`, or a metrics
file joined on `(codec, param)` with an energy side file).

Key design choices:

- **Normalization is computed once on the full pool**, not separately
  per pool. Re-normalizing per pool would break J_RDE comparability
  between classical-only and full-pool oracles. The report records
  `normalization.scope = full_pool_global`.
- **Profiles come from the official router profiles** in
  `src/router/core/profiles.py` (`balanced`, `energy-limited`,
  `bandwidth-limited`, `quality-first`), not from ad-hoc weights.
  The exact `(w_R, w_E, w_D)` used are written into
  `report.profile_weights` so the analysis is self-contained.
- **The audit framing avoids universal claims about neural codecs.**
  The interpretation strings use scoped wording (`within this
  benchmark`, `suggests`, `is consistent with`) and explicitly state
  that observed wins under one profile **do not imply universal
  dominance** of neural over classical codecs.

The audit's main scientific question is asymmetric: *the classical
pool is the deployable default*; the question is whether the neural
pool covers a region of the R-D-E space that classical codecs do not
reach under any tested profile, and how that region is described
operationally (low bitrate × high energy, low energy × moderate
quality, etc.). The output `neural_inclusive_pool_comparison.csv`
quantifies this region: `mean_regret_classic_vs_full` per
(profile, floor), `neural_selection_rate_in_full`,
`mean_rate_gain_when_neural_selected`,
`mean_energy_penalty_when_neural_selected`.

The audit also emits `neural_inclusive_per_group_labels.csv` with
`classic_pool_oracle_label`, `full_pool_oracle_label`,
`full_pool_oracle_family`, and `regret_classic_vs_full` per group.
These labels are the natural training target for a future
neural-inclusive predictor; v0.43.3 does not train such a predictor.

## Neural-inclusive predictive router evaluation

v0.43.3 computed the per-image full-pool oracle and answered the
theoretical question "where, and under which profile, are neural
codecs oracle-optimal?". v0.43.4 answers the practical, operational
question that follows: "can a lightweight routing policy anticipate
those cases *before* the compression, using only training-time
information?".

The module that ships in this release is
`src.router.analysis.neural_inclusive_predictive_router`. It is
offline / read-only against the existing R-D-E CSV, does not change
the router runtime, the ranking score, the operational report
schema, or any CLI flag of `rde_router`. Five evaluation policies
are run under both leave-one-image-out (LOIO) and
leave-one-dataset-out (LODO):

1. `robust_global_full_pool_baseline` — pick a single
   (codec, config) from training (lowest mean J_RDE under full
   training coverage), apply uniformly on test images.
2. `source_aware_full_pool_majority` — per-source majority vote of
   the per-training-image full-pool oracle; fallback to the global
   baseline for unknown sources.
3. `knn_metadata_full_pool` — kNN over metadata features
   (megapixels, aspect_ratio, resolution_class, orientation_class)
   with the per-training-image full-pool oracle label as the kNN
   target.
4. `classic_only_predictive_baseline` — same kNN as (3) but training
   labels are the *classical-pool* oracle on training images.
   Quantifies how much regret a classic-only predictor pays against
   the full-pool oracle when neural codecs are theoretically
   available.
5. `full_pool_oracle` — upper-bound reference; not a router.

The methodological rule of the module is strict. **The test image's
measured R-D-E candidates are never used to choose the codec.** They
are used only afterwards to look up the realised J_RDE of the
predicted candidate, to compute the test image's full-pool oracle,
and to compute regret against that oracle. A quality-floor violation
on the test image is reported as `quality_violation=true` but never
triggers a retroactive re-selection: that would leak target
information into the policy and break the leave-out methodology.

Outputs:

- `neural_inclusive_predictive_router_decisions.csv` — one row per
  (image, protocol, profile, floor, policy) with the predicted
  codec/config/family, the oracle codec/config/family, the realised
  J on the test image, the oracle J, the regret, family/exact match
  flags, confidence (kNN vote share), fallback flag + reason,
  quality_violation flag, and the provenance string that records
  whether the predicted pair was present on the test image.
- `neural_inclusive_predictive_router_summary.csv` — one row per
  (policy, protocol, profile, floor) with mean/median/p90/max
  regret, `relative_reduction_vs_global`, predicted
  `neural_selection_rate`, `oracle_neural_rate`,
  `neural_family_precision` / `neural_family_recall` against the
  oracle, exact-match and family-match rates, fallback and
  quality-violation rates, plus paired-bootstrap 95% CIs for
  `mean_regret` and `relative_reduction_vs_global` (1000
  resamples by default, deterministic under a fixed seed).
- `neural_inclusive_predictive_router_report.json` — the full
  structured report with inputs, codec_inventory, normalization
  scope, profile_weights, summaries, interpretation, and the
  provenance block that asserts
  `policy_does_not_see_test_image_rde = true` and
  `test_image_rde_used_only_for_realisation_and_oracle = true`.

Together, v0.43.3 and v0.43.4 answer two complementary questions:
the oracle audit (v0.43.3) describes the theoretical opportunity in
the R-D-E space; the predictive router evaluation (v0.43.4)
measures how much of that opportunity a lightweight metadata-only
policy can recover under each protocol. The gap between
`oracle_neural_rate` and predicted `neural_selection_rate` per
(profile, floor) records where the predictor remains conservative;
the gap between `knn_metadata_full_pool` and
`classic_only_predictive_baseline` regret records the operational
value of including neural codecs in the candidate pool.

## Operational regime simulation

v0.43.6 adds the offline module
`src.router.analysis.operational_regime_simulation`. The purpose is
paper/demo-facing: the analysis does not only ask whether a predictor
can recover oracle labels, but how routing decisions move under
operational regimes such as `normal`, `bandwidth_limited`,
`energy_saving`, `battery_pressure`, `thermal_pressure`, `no_cuda`,
and `low_memory_or_vram_pressure`.

The module is read-only against an existing image R-D-E CSV. It does
not execute codecs, regenerate benchmarks, change `J_RDE`, alter the
runtime router report schema, or add CLI flags to `rde_router`. Each
regime is written explicitly to the JSON report with `(w_R, w_E,
w_D)`, neural admissibility, any neural/system penalty, optional
energy pressure, and a textual description. Regimes that are offline
stress simulations rather than one-to-one runtime policies are marked
with `simulation_only=true`. The memory/VRAM regime is explicitly
proxy-based when no RAM/VRAM measurements are available.

The policies form a predictive factor ablation:

1. `robust_global_full_pool_baseline` -- a global full-pool baseline.
2. `metadata_only_full_pool` -- content metadata without system
   context.
3. `system_only_full_pool` -- system/regime context without metadata
   prediction.
4. `metadata_plus_system_full_pool` -- combined content and system
   context.
5. `metadata_only_classic_pool` -- metadata prediction with neural
   candidates excluded.
6. `full_pool_oracle` -- an upper-bound reference, not a deployable
   router.

The no-leakage rule remains central: the predictive policy does not
use the target image's measured R-D-E candidates to choose a codec.
Those rows are used only afterwards to realise the selected
codec/configuration and compute energy, rate, quality, quality-floor
violations and regret against the active-regime oracle. The report
records `policy_does_not_see_test_image_rde=true`.

The main summary metrics are `energy_saving_vs_global_baseline`,
`regret_reduction_vs_global_baseline`, `quality_violation_rate`, and
`neural_selection_rate`, alongside mean energy, rate, quality and
regret. These quantities support statements of the form: in a given
regime, the router reduces energy or regret while preserving the
quality floor. They should still be interpreted within the benchmark
considered, not as universal claims about all natural images or all
codec implementations.

The module emits plot-ready CSVs for the main paper/demo views:

- `operational_regime_plot_data.csv` for the energy-saving vs regret
  reduction scatter.
- `operational_regime_winner_distribution.csv` for stacked bars of
  selected codec/configuration winners by regime.
- `operational_regime_oracle_vs_prediction.csv` for oracle-family vs
  predicted-family rates.
- `operational_regime_family_confusion.csv` for classical/neural
  confusion heatmaps.
- `operational_regime_rate_pressure_sweep.csv` for the rate-weight
  sweep that shows how family and codec selections change as bitrate
  pressure increases.

When `matplotlib` is available, the audit also writes PNGs such as
`energy_saving_vs_regret_reduction.png`,
`neural_selection_rate_by_regime.png`,
`winner_family_by_regime.png`, `winner_codec_by_regime.png`,
`oracle_vs_predicted_neural_rate.png`,
`family_confusion_heatmap.png`,
`rate_pressure_family_shift.png`,
`rate_pressure_codec_shift.png`,
`rate_reduction_vs_energy_penalty_sweep.png`, and
`quality_violation_by_regime.png`. Plot generation is optional: if
`matplotlib` is unavailable, the CLI still writes all CSV/JSON
artifacts and records the skip reason in `plot_artifacts`.

## Caveats

The current benchmark has 96 images across 4 datasets. The classifier is intentionally simple and should be presented as a lightweight baseline, not as the final possible predictor.

The strongest deployable result is not that pixel features solve the task, but that a cheap metadata-only classifier already captures much of the routing opportunity.

The model ablation should be framed as a robustness check: it does not claim a
definitive superiority of kNN over more complex supervised models, but shows
that the content-aware gain is not an artifact of a single arbitrary
classifier.
