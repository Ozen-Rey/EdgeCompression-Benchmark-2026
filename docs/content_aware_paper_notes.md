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

A content-aware prediction can only be selected if the predicted codec/configuration is inside the admissible pool after all active constraints have been applied. If it is not admissible, the router falls back to the standard R-D-E/system-aware decision.

The validation script checks both cases:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_classifier_router.ps1
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

- Mean regret by method.
- Relative regret reduction by method.
- k-sensitivity under LOIO and LODO.
- Oracle codec/configuration distribution.
- Feature extraction overhead vs encoding time.

## Suggested Paper Tables

- Main content-aware benchmark table.
- Overhead table.
- Best-k sensitivity table.

## Caveats

The current benchmark has 96 images across 4 datasets. The classifier is intentionally simple and should be presented as a lightweight baseline, not as the final possible predictor.

The strongest deployable result is not that pixel features solve the task, but that a cheap metadata-only classifier already captures much of the routing opportunity.
