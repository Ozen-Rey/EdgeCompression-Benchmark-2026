# R-D-E Router v0.9: Content-Aware Routing

## Overview

The v0.9 router extension introduces content-aware decision support on top of the existing Rate--Distortion--Energy router.

The goal is to reduce the regret incurred by a single robust global configuration. A robust global configuration is safe across the whole benchmark, but it can be unnecessarily expensive for many individual images. Content-aware routing uses lightweight information about the image or its source to select a lower-cost codec/configuration while preserving the same quality constraints.

The v0.9 pipeline has four main components:

1. Per-image oracle analysis.
2. Metadata and pixel-level feature extraction.
3. Source-aware and source-agnostic policy evaluation.
4. Safe integration into the router with fallback to the standard R-D-E decision.

---

## Per-image R-D-E Oracle

For each image, the oracle selects the feasible codec/configuration with the minimum normalized R-D-E cost:

```text
J_RDE = w_R R_norm + w_E E_norm + w_D D_norm
```

In the image experiment, the weights are:

```text
w_R = 0.2
w_E = 0.2
w_D = 0.6
```

The quality constraint is:

```text
SSIMULACRA2 >= 80
```

The robust global baseline is the single codec/configuration that satisfies the quality constraint for all images and has minimum mean cost under that full-coverage requirement.

For the current image benchmark:

```text
Robust global baseline: HEVC crf=15
Coverage: 96 / 96 images
Oracle configurations: JPEG q=85, JXL d=1.0, HEVC crf=15
Oracle switch count: 93 / 96
Oracle switch rate: 96.875%
Mean regret of robust global baseline: 0.09047
```

This means that the globally robust baseline is safe, but it is rarely the per-image optimum.

## Metadata Features

The metadata feature extractor builds one row per image using information already available in the benchmark CSV or image manifest:

```text
dataset
image
width
height
pixels
megapixels
aspect_ratio
orientation_class
resolution_class
```

These features are cheap and do not require reading image pixels.

The source-aware interpretation of dataset is not meant to assume that every production image has a dataset label. Instead, it models the common batch setting where the user knows that a set of images comes from a homogeneous source.

Examples:

```text
kodak
tecnick
clic2020
div2k_valid
product_photos
screenshots
scanned_documents
satellite_tiles
```

When no source label is available, the router can use source-agnostic metadata and pixel features instead.

## Pixel-Level Features

The pixel feature extractor reads each image, resizes it to a bounded long side, and computes lightweight image statistics.

Current features include:

```text
luminance_mean
luminance_std
luminance_entropy_bits
luminance_entropy_norm
gradient_mean
gradient_std
edge_density
flat_area_ratio
colorfulness
resolution_class
orientation_class
entropy_class
edge_class
texture_class
color_class
feature_overhead_ms
```

The current extractor uses a resized image with long side 256.

Observed overhead:

```text
Mean feature extraction overhead: approximately 30.5 ms / image
```

This overhead is acceptable for offline analysis, batch processing, cached feature pipelines, and predictor training. For online routing of very fast codecs, the overhead can exceed the actual encoding time. In that case, source-aware routing, metadata-only routing, or cached features should be preferred.

## Source-Aware Metadata Policy

The simplest deployable content-aware policy is a source-aware majority rule:

```text
source/dataset -> most frequent oracle codec/configuration
```

Example rules:

```text
tecnick -> JPEG q=85
clic2020 -> JXL d=1.0
kodak -> JPEG q=85
div2k_valid -> JPEG q=85
```

This policy is useful when the user compresses a homogeneous batch and can provide the source label.

The router supports this through:

```text
--content-policy
--content-policy-mode report-only/apply
--content-policy-rules-file results/routing_context/v09_metadata_policy_dataset_rules.csv
--content-policy-key dataset
--content-source tecnick
--content-source-filter
```

The --content-source-filter option restricts the benchmark candidate pool to the declared source before aggregation and quality filtering. This is important because a rule learned for tecnick should be evaluated on the tecnick subset, not on the global benchmark.

Example:

```text
content_source = tecnick
content_filter = dataset=tecnick
rows: 3168 -> 792
safe points: 4
suggestion: JPEG q=85
selected: JPEG q=85
selected_reason: content_policy_preferred_candidate
```

The policy is safe: if the suggested codec/configuration is not admissible, the router falls back to the standard R-D-E selection.

## Source-Agnostic Oracle Classifier

For production-like settings where no source label is available, v0.9 includes a lightweight k-nearest-neighbor oracle classifier.

The classifier predicts the oracle codec/configuration from metadata and/or pixel features. The predicted candidate is only used if it satisfies the quality constraint. Otherwise, the router falls back to the robust R-D-E selection.

Feature sets evaluated:

```text
metadata_no_source
pixel_no_source
all_no_source
all_with_source
```

Evaluation protocols:

```text
leave-one-image-out
leave-one-dataset-out
```

The distinction is important:

```text
leave-one-image-out:
  tests generalization to new images from already observed sources.

leave-one-dataset-out:
  tests generalization to an entirely unseen source/dataset.
```

## Benchmark Table

The main benchmark table is generated by:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-content-aware-benchmark-table
```

Output:

```text
results/routing_context/v09_content_aware_benchmark_table.csv
results/routing_context/v09_content_aware_benchmark_all_methods.csv
```

Current paper table:

| Metodo | Protocollo | Deployment | Regret medio | Riduzione relativa | Match oracle | Fallback |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline globale robusta | global coverage | source-agnostic | 0.09047 | 0.0000 | 0.03125 | 0.0000 |
| Policy source-aware | leave-one-out | batch con sorgente | 0.01177 | 0.86995 | 0.77083 | 0.18750 |
| kNN source-agnostic | leave-one-image-out | source-agnostic | 0.01167 | 0.87105 | 0.76042 | 0.20833 |
| kNN source-agnostic | leave-one-dataset-out | source-agnostic | 0.02197 | 0.75716 | 0.57292 | 0.28125 |
| Oracle per immagine | oracle | non deployable | 0.00000 | 1.00000 | 1.00000 | 0.00000 |

Interpretation:

```text
The robust global baseline is safe but expensive.
The source-aware policy reduces mean regret by approximately 87%.
The source-agnostic classifier also reduces mean regret by approximately 87% under leave-one-image-out.
Under the stricter leave-one-dataset-out protocol, the source-agnostic classifier still reduces mean regret by approximately 75.7%.
```

## Router Safety Mechanism

Content-aware routing is never allowed to blindly override the quality guard,
that is, the robust quality constraint used by the R-D-E router.

The decision logic is:

1. Compute the normal admissible pool using the R-D-E router.
2. Obtain a content-aware suggestion.
3. Rank the admissible pool with the same score used by the router.
4. Select the suggestion only if it is admissible and competitive according to
   that ranking score.
5. Otherwise, fall back to the normal R-D-E/system-aware router selection.

In compact form, the ranking score is:

```text
J_rank(a) = J_RDE(a)                         if the system penalty is not applied
J_rank(a) = J_RDE(a) + lambda_sys P_sys(a)   if the system penalty is applied
```

The selected action is therefore:

```text
a_sel = a_cont_hat(x)                 if a_cont_hat(x) is admissible and
                                      competitive under J_rank
a_sel = argmin_{a in A_adm} J_rank(a) otherwise
```

This ensures:

```text
infeasible_rate = 0
```

for the evaluated policies.

The classifier therefore cannot bypass the quality guard, capability checks,
hardware availability, source filtering, time constraints, or the system
penalty. It only proposes a candidate inside the same decision process.

Because the offline validation uses 96 images, the reported reductions should
be read as prototypical evidence rather than as a definitive estimate of
generalization on much larger industrial distributions.

## Reproducibility Scripts

The v0.9 content-aware pipeline is validated by:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-content-aware
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-image-manifest
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-image-features
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-oracle-classifier
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-oracle-classifier-sweep
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-content-aware-benchmark-table
```

The full test suite currently passes with:

```text
89 passed
```

## Limitations

The current evaluation uses:

```text
96 images
4 datasets
3 oracle configurations
image domain only
kNN classifier baseline
```

The content-aware router is implemented as a deployable prototype and is
validated offline against the measured R-D-E operating points. The remaining
gap is not the absence of a predictor, but the lack of an end-to-end online
deployment where new content, new platforms and updated energy measurements
are observed in the loop.

The pixel feature extractor has non-negligible overhead compared with very fast codecs such as JPEG. Therefore, pixel features are most appropriate for batch routing, cached-feature pipelines, or cases where encoding cost is larger than feature extraction cost.

The source-aware policy assumes a homogeneous batch/source label. This is realistic for offline datasets, archives, and controlled pipelines, but it is not always available in streaming production settings.

The kNN classifier is intentionally simple. It establishes a baseline, not a final predictive model. Future work may evaluate decision trees, calibrated probabilistic classifiers, or direct R-D-E regression models.

The model ablation is not meant to prove that kNN is definitively superior to
more expressive supervised models. Its purpose is to verify that the
content-aware benefit does not depend on an arbitrary classifier choice.

## Overhead and Deployment Choice

The current pixel-level extractor has a mean overhead of approximately 30.52 ms per image.

Measured references:

| Component / case | Mean time |
|---|---:|
| Pixel feature extraction, long side 256 | 30.52 ms |
| JPEG q=85 on Tecnick | 4.52 ms |
| JPEG q=85 global | 6.31 ms |
| JXL d=1.0 global | 134.01 ms |
| HEVC crf=15 global | 342.22 ms |

The full pixel feature extractor is therefore not always suitable for the online fast path. For very fast codecs such as JPEG, feature extraction can dominate encode time. For slower codecs such as JXL and HEVC, the overhead is less problematic.

For this reason, the deployable source-agnostic router currently uses the metadata-only classifier:

```text
feature_set = metadata_no_source
k = 7
```

This model uses only width, height, megapixels, aspect ratio, resolution class and orientation class. It avoids full pixel scanning and remains suitable for low-overhead online routing.

Pixel features remain useful for offline analysis, cached-feature pipelines, and future richer predictors.

## k-Sensitivity

The kNN classifier was swept over:

```text
k in {1, 3, 5, 7, 9, 11}
```

For the stricter leave-one-dataset-out protocol, the best source-agnostic setting was:

```text
metadata_no_source, k = 7
mean regret = 0.02197
relative regret reduction = 75.72%
```

The same mean regret was also obtained for k=9 and k=11. This indicates that the selected metadata-only classifier is not highly sensitive to the exact k value in the higher-neighborhood range.

For leave-one-image-out, the best setting was:

```text
metadata_no_source, k = 11
mean regret = 0.01167
relative regret reduction = 87.10%
```

The router uses the LODO-selected setting because it is the more conservative generalization protocol.

## Interaction with System-Aware Routing

The content-aware layer is advisory, not authoritative.

The router follows this order:

```text
1. Build the candidate pool.
2. Apply capability and executable constraints.
3. Apply source filtering, if requested.
4. Apply system-aware policy and system penalty.
5. Apply the quality guard and define the admissible pool.
6. Let the source-aware policy or source-agnostic classifier suggest a candidate.
7. Accept the suggestion only if it is inside the admissible pool and
   competitive under the active ranking score.
8. Otherwise, fall back to the standard R-D-E/system-aware decision.
```

This means that the classifier cannot bypass quality constraints, system constraints, capability constraints, or source filtering.

When the system penalty is active in apply mode, the active ranking key is
`J_total = J_RDE + lambda_sys P_sys`; otherwise it is `J_RDE`. The
content-aware suggestion is audited against the same ranking key used for the
final router decision.

The validation script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_router.ps1 -Scenario v09-content-classifier-router
```

checks both safety cases:

```text
global pool:
  classifier predicts JPEG q=85
  JPEG q=85 is not globally admissible
  router falls back to HEVC crf=15

Tecnick source-filtered pool:
  classifier predicts JPEG q=85
  JPEG q=85 is admissible on Tecnick
  router selects JPEG q=85
```
