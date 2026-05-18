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

## Caveats

The current benchmark has 96 images across 4 datasets. The classifier is intentionally simple and should be presented as a lightweight baseline, not as the final possible predictor.

The strongest deployable result is not that pixel features solve the task, but that a cheap metadata-only classifier already captures much of the routing opportunity.

The model ablation should be framed as a robustness check: it does not claim a
definitive superiority of kNN over more complex supervised models, but shows
that the content-aware gain is not an artifact of a single arbitrary
classifier.
