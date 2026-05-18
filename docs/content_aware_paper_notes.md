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

## Caveats

The current benchmark has 96 images across 4 datasets. The classifier is intentionally simple and should be presented as a lightweight baseline, not as the final possible predictor.

The strongest deployable result is not that pixel features solve the task, but that a cheap metadata-only classifier already captures much of the routing opportunity.

The model ablation should be framed as a robustness check: it does not claim a
definitive superiority of kNN over more complex supervised models, but shows
that the content-aware gain is not an artifact of a single arbitrary
classifier.
