# Router Online Feedback Loop

Router v0.12.0 introduces append-only feedback logging for executions requested
with `--execute`.

The data sources stay separate:

```text
benchmark CSV = prior offline
calibration JSON = local controlled adaptation
online feedback CSV = observed deployment trace
```

The full methodological pipeline is:

```text
offline benchmark -> router decision
local calibration -> controlled local adaptation
online feedback -> append-only deployment trace
feedback analysis -> read-only prediction audit
feedback calibration proposal -> shadow correction proposal
feedback proposal validation -> offline scale validation
feedback calibration promotion -> audit-only candidate profile
promoted calibration apply -> explicit opt-in calibration input
calibration bundle manifest -> audit provenance for calibrated CSVs
calibration bundle consumption -> explicit manifest-validated router input
calibration impact audit -> read-only baseline vs bundle decision comparison
shadow decision comparison -> offline what-if comparison only
shadow decision validation -> offline methodological gate
normalization consistency audit -> explicit previous receipt comparability check
```

The feedback CSV records predicted router values, observed execution values and
energy provenance. It does not update benchmark CSVs and is not used as router
input in v0.12.0.

Router v0.13.0 adds a read-only feedback analysis layer:

```powershell
python -m src.router.feedback_analysis `
  --feedback results/routing_context/online_feedback.csv `
  --out-dir results/routing_context/feedback_analysis
```

The analyzer produces summary JSON/CSV files, per-codec aggregates and an error
row extract. It audits prediction error for rate, time and energy, but it does
not feed results back into the router. There is no online learning in v0.13.0.

Router v0.14.0 adds shadow feedback-derived calibration proposals:

```powershell
python -m src.router.feedback_calibration_proposal `
  --feedback results/routing_context/online_feedback.csv `
  --out results/routing_context/feedback_calibration_proposal.json `
  --summary-out results/routing_context/feedback_calibration_proposal.csv `
  --min-samples 3
```

The distinction is explicit:

```text
feedback_analysis = audit read-only
feedback_calibration_proposal = shadow correction proposal
calibration_apply = explicit opt-in only
router decision = unchanged unless a future release wires proposals explicitly
```

The proposal file estimates per-codec/config scale factors from observed
feedback, but it is not loaded by the router and does not modify benchmark CSVs,
normalization, ranking or `J_RDE`. It is a review artifact for future controlled
calibration, not online learning.

Router v0.15.0 adds offline validation for those shadow proposals:

```powershell
python -m src.router.feedback_proposal_validation `
  --feedback results/routing_context/online_feedback.csv `
  --proposal results/routing_context/feedback_calibration_proposal.json `
  --out results/routing_context/feedback_proposal_validation.json `
  --summary-out results/routing_context/feedback_proposal_validation.csv
```

The release ladder is:

```text
v0.12 feedback_logger: append-only observations
v0.13 feedback_analysis: audit/read-only analysis
v0.14 feedback_calibration_proposal: shadow correction candidates
v0.15 feedback_proposal_validation: offline validation of candidate scales
v0.16 feedback_calibration_promotion: audit-only candidate calibration profile
v0.17 calibration_apply --promotion-profile: explicit opt-in application
v0.18 calibration_apply --manifest-out: auditable calibration bundle manifest
v0.19 rde_router --calibration-bundle-manifest: explicit verified bundle input
v0.20 calibration_impact_audit: read-only decision impact audit
v0.21 shadow_decision_comparison: offline baseline vs calibrated what-if
v0.22 shadow_decision_validation: read-only shadow acceptance gate
v0.23 rde_router --calibration-bundle-validation: explicit validated consumption
v0.24 cross-artifact integrity: validation bound to exact bundle hash
v0.25 decision receipt/replay: auditable decision reproducibility check
v0.26 router_overhead_audit: read-only runtime overhead measurement
v0.27 router_effectiveness_audit: read-only comparison against simple baselines
v0.28 effectiveness cost explainability: explicit audit cost provenance
```

The validator compares prediction error before and after a proposed scale using
absolute log error. It is still read-only: it writes validation reports, but it
does not change router decisions, calibration files, benchmark CSVs or
normalization.

Router v0.16.0 adds a promotion gate:

```powershell
python -m src.router.feedback_calibration_promotion `
  --proposal results/routing_context/feedback_calibration_proposal.json `
  --validation results/routing_context/feedback_proposal_validation.json `
  --out results/routing_context/feedback_calibration_profile_candidate.json `
  --summary-out results/routing_context/feedback_calibration_profile_candidate.csv `
  --min-samples 3 `
  --min-improvement 0.10 `
  --max-after-error 0.25
```

`feedback_calibration_promotion` turns validated shadow proposals into an
audit-only candidate calibration profile. The router does not consume this
profile automatically, and no router flag is added in v0.16.0.

Router v0.17.0 adds explicit opt-in application of a promoted calibration
profile in `calibration_apply`:

```powershell
python -m src.router.calibration_apply `
  --benchmark data/rde_points.csv `
  --calibration results/routing_calibration/quick.json `
  --promotion-profile results/routing_context/feedback_calibration_profile_candidate.json `
  --out results/routing_context/rde_points_calibrated_with_feedback.csv
```

Without `--promotion-profile`, `calibration_apply` behaves as before. With the
flag, only promoted/accepted/usable scales are applied; rejected scales are
ignored and reported. This remains an explicit offline transformation. The
router does not auto-load promoted profiles in v0.17.0.

Router v0.18.0 adds an optional provenance manifest for calibrated CSV bundles:

```powershell
python -m src.router.calibration_apply `
  --benchmark data/rde_points.csv `
  --calibration results/routing_calibration/quick.json `
  --promotion-profile results/routing_context/feedback_calibration_profile_candidate.json `
  --out results/routing_context/rde_points_calibrated_with_feedback.csv `
  --manifest-out results/routing_calibration/calibrated_manifest.json
```

The manifest records source paths, output path, router version, UTC creation
time, applied promoted scales, rejected/non-applied scale count, energy policy
and SHA256 hashes for the source and output artifacts. It is provenance only:
the router does not read calibration manifests automatically in v0.18.0.

Router v0.19.0 adds explicit bundle consumption:

```powershell
python -m src.router.rde_router `
  --csv data/rde_points.csv `
  --calibration-bundle-manifest results/routing_calibration/calibrated_manifest.json `
  --out results/routing/router_decision_report.json
```

When this flag is present, the router validates the manifest, checks that the
calibrated CSV exists, verifies its SHA256 hash, and then uses the manifest's
CSV as the R-D-E input. Without the flag, behavior is unchanged. The router
still does not read `online_feedback.csv`, proposal, validation, promotion or
manifest files automatically.

Router v0.20.0 adds a read-only impact audit for explicit calibration bundles:

```powershell
python -m src.router.calibration_impact_audit `
  --config configs/router_image_v08.json `
  --csv results/routing_context/image_rde_points.csv `
  --calibration-bundle-manifest results/routing_context/calibration_bundle_manifest.json `
  --out results/routing_context/calibration_impact_audit.json `
  --summary-out results/routing_context/calibration_impact_audit.csv
```

The audit runs the same router decision once on the baseline CSV and once with
the validated bundle manifest, then reports whether the selected codec/config
and R-D-E terms changed. It does not modify benchmark files, feedback files,
calibration bundles or router defaults, and it does not execute codec backends.

Router v0.21.0 adds a stricter shadow decision comparison report:

```powershell
python -m src.router.shadow_decision_comparison `
  --baseline-csv results/routing_context/image_rde_points.csv `
  --bundle-manifest results/routing_context/calibration_bundle_manifest.json `
  --config configs/router_image_v08.json `
  --out results/routing_context/shadow_decision_comparison.json `
  --summary-out results/routing_context/shadow_decision_comparison.csv
```

This module answers only a what-if question: if this verified calibration
bundle were used as the R-D-E source, would the router choice change? It does
not prove the calibrated decision is better; utility still requires separate
offline regret, oracle or ablation validation.

Router v0.22.0 adds a validation gate for shadow comparison outputs:

```powershell
python -m src.router.shadow_decision_validation `
  --comparison results/routing_context/shadow_decision_comparison.json `
  --out results/routing_context/shadow_decision_validation.json `
  --summary-out results/routing_context/shadow_decision_validation.csv
```

The gate checks sample count, decision churn, mean cost regression, quality
regressions, coarse rate/time/energy regressions and unsafe energy provenance.
It produces an audit decision (`accepted=true/false`) for the shadow comparison
artifact only. A rejected candidate has no effect on the router, and an
accepted candidate is still not consumed automatically.

Router v0.23.0 adds an optional validation requirement when consuming a bundle:

```powershell
python -m src.router.rde_router `
  --csv results/routing_context/original.csv `
  --calibration-bundle-manifest results/routing_context/calibrated_bundle_manifest.json `
  --calibration-bundle-validation results/routing_context/shadow_decision_validation.json
```

Without `--calibration-bundle-validation`, bundle consumption behaves as in
v0.19. With the flag, the named validation report must have
`mode=shadow_decision_validation_only` and `accepted=true`; rejected, missing or
malformed validation reports block bundle consumption. The router still performs
no automatic discovery and does not read feedback, proposal, promotion,
comparison or validation artifacts unless the user names the relevant file.

Router v0.24.0 binds the validated bundle chain by artifact hash. Shadow
comparison reports include the baseline CSV hash, candidate calibration bundle
manifest hash and calibrated CSV hash. Shadow validation reports propagate the
comparison hash and candidate bundle hashes. When `rde_router` is given both
`--calibration-bundle-manifest` and `--calibration-bundle-validation`, it now
requires the validation report to refer to exactly that bundle manifest hash.
Legacy validation reports without v0.24 hash fields are rejected when the
validation flag is used.

Router v0.25.0 adds decision receipts and offline replay:

```powershell
python -m src.router.decision_replay `
  --receipt results/routing/router_decision_report.json `
  --out results/routing/decision_replay_validation.json
```

Every single-profile router report includes `decision_receipt`, which records
the selected decision, replay-safe router argv and SHA256 hashes of relevant
input artifacts. The replay validator checks those hashes, reruns the router
without execution/output side effects, and verifies that the same decision is
reproduced. This is audit only: it does not alter ranking, policy,
normalization, calibration, feedback, or execution behavior.

Router v0.26.0 adds a read-only overhead audit:

```powershell
python -m src.router.router_overhead_audit `
  --csv results/routing_context/original.csv `
  --config configs/router_image_v08.json `
  --bundle-manifest results/routing_context/calibrated_bundle_manifest.json `
  --bundle-validation results/routing_context/shadow_decision_validation.json `
  --out-dir results/routing_context/overhead_audit
```

The audit runs controlled offline router invocations for baseline, receipt,
bundle and validated-bundle modes when the required explicit files are provided.
It records wall time, process CPU time, optional Python peak-memory tracing,
candidate counts and decision match against baseline. Replay can be included
separately and is marked as offline, not as a normal runtime path.

Router v0.27.0 adds a read-only effectiveness audit:

```powershell
python -m src.router.router_effectiveness_audit `
  --csv results/routing_context/original.csv `
  --config configs/router_image_v08.json `
  --out-dir results/routing_context/effectiveness_audit
```

The audit compares the router-selected point against simple baseline policies
such as lowest rate, highest quality, lowest energy, fastest time,
fixed codec/config and the best costed admissible candidate. Baselines are
evaluated offline and must respect the same quality/constraint guards; a
baseline that would violate the guard is marked non-comparable. Explicit bundle
and validated-bundle scenarios can be audited only when their paths are passed
directly. The module does not alter ranking, `J_RDE`, calibration, bundle
validation, feedback, replay, content-aware logic or system-aware logic.

Router v0.28.0 improves the effectiveness audit's cost explainability without
changing the router. The by-policy CSV/JSON now includes an explicit
`policy=router` row plus `candidate_source`, `cost_status` and
`cost_reason_detail` fields. A baseline is `comparable` only when it satisfies
the guards and has a router-exported cost. A candidate can be feasible but
unscored when it appears in the raw candidate pool but not in the router's
exported scored pool; in that case regret is left empty and the reason is
reported as `feasible_but_unscored` rather than a generic missing-cost label.
Filtered candidates remain separate from feasible-but-unscored candidates.

Router v0.31.0 adds an optional normalization consistency audit to normal
router reports:

```powershell
python -m src.router.rde_router `
  --csv results/routing_context/original.csv `
  --previous-decision-receipt results/routing_context/old_router_report.json `
  --out results/routing_context/router_decision_report.json
```

The previous artifact must be named explicitly with
`--previous-decision-receipt`; the router never searches for the latest or
nearest receipt automatically. The artifact may be a decision receipt, a router
report with `decision_receipt.normalization_audit`, or a router report with a
top-level `normalization_audit`.

The check is report-only. It compares the current `normalization_audit` with
the previous one and writes `normalization_consistency` with comparability,
warnings and field-level difference flags. It does not change `J_RDE`, ranking,
normalization, calibration bundle handling, energy backends, content-aware
logic, system-aware logic or the selected codec/config. Its purpose is only to
make clear whether two runs are methodologically comparable.

Router v0.32.0 adds codec executable fingerprints to calibration bundle
manifests written by `calibration_apply --manifest-out`. The manifest records
the backend, version, binary path, binary SHA256 and availability for codecs
represented by accepted calibration scales. For Python/Pillow JPEG, the binary
fields are `null` and the version comes from the Python package.

When a calibration bundle is consumed explicitly with
`--calibration-bundle-manifest`, the router validates `codec_fingerprints` if
the manifest contains them. A version mismatch, binary hash mismatch or missing
declared binary fails with a controlled calibration staleness error. Legacy
manifests without `codec_fingerprints` remain compatible; the bundle report
marks `codec_fingerprint_validation.enabled=false` with
`reason=manifest_without_codec_fingerprints`.

Router v0.32.1 tightens the empty-manifest edge case. A manifest that contains
`codec_fingerprints={}` is not considered successfully validated; it is
reported with `codec_fingerprint_validation.enabled=false`,
`validated=false`, `reason=empty_codec_fingerprints`, empty
`validated_codecs`, and empty `mismatches`. Newly written manifests also
fingerprint codecs with applied local calibration even when no promotion
profile is provided.

This is a staleness/provenance gate for explicit bundles only. It does not
change `J_RDE`, ranking, normalization, calibration application without a
manifest, feedback, content-aware logic, system-aware logic or backend
execution. The router still performs no automatic bundle discovery.

Router v0.33.0 adds report-only energy provenance tiers. Candidate entries now
include `energy_provenance_tier` in the selected candidate, scored candidate
pool, unscored candidate pool and legacy `top_k` view. Reports also include
`energy_provenance_summary` with counts by tier. The tiers are:

- `measured_hw_total`: hardware-measured energy usable as total pipeline energy.
- `measured_hw_partial`: hardware energy is present but not usable as total
  pipeline energy, such as GPU-only telemetry.
- `derived_time_scaled`: energy derived from benchmark/local time scaling.
- `benchmark_reference`: benchmark CSV energy without local measurement
  metadata.
- `unknown`: missing or unclassifiable provenance.

This is observability only. Tiers do not add penalties, change `J_RDE`, change
ranking, change normalization, filter candidates, change bundle consumption,
alter feedback, alter content-aware or system-aware logic, or affect backend
execution.

Router v0.34.0 adds a read-only energy provenance compatibility audit. The
router report now includes `energy_provenance_compatibility`, which records the
selected tier, tier counts for scored and unscored pools, whether the scored
pool mixes tiers, warnings and a severity of `ok`, `warning` or `critical`.

The audit marks a scored pool with one tier as compatible. Mixed scored-pool
tiers are reported as `warning`; a selected `measured_hw_partial` candidate is
`critical` because partial telemetry is not total pipeline energy; a selected
`unknown` tier is a warning. This audit is report-only and does not change
costs, ordering, normalization, candidate inclusion, bundle handling, feedback,
content-aware logic, system-aware logic or backend execution.

Router v0.35.0 adds `energy_tier_policy`, a shadow report for a future
strict-compatible energy tier policy. It simulates what would happen if the
router preferred a compatible pool using this reliability order:
`measured_hw_total`, `derived_time_scaled`, `benchmark_reference`,
`measured_hw_partial`, `unknown`. The shadow policy reports the current
selection, a shadow selection, candidates that would be rejected by the policy,
and whether the decision would change.

This remains report-only. There is no apply flag, no candidate is actually
filtered, no cost is changed, and the selected codec/config in `decision` is
unchanged.

Router v0.36.0 adds an offline [external codec specification schema](external_codecs.md)
and validator. External codec specs are declaration artifacts only: validation
does not execute `version_probe`, does not check binary existence, does not
register the codec with the router, and does not change ranking, `J_RDE`,
normalization, energy policy, content-aware logic, system-aware logic or
backend execution.

By default, executed router runs append to:

```text
results/routing_context/online_feedback.csv
```

Use `--feedback-out <path>` to choose another destination. The logger creates
the file and header on first use, then appends rows without overwriting prior
observations.

Energy fields must be interpreted through provenance:

- `local_cpu_energy_j` records measured CPU/package energy when available.
- `local_gpu_energy_j` records measured GPU energy when available.
- `local_energy_j` is populated only when the measurement is usable as total
  pipeline energy.
- `energy_scope`, `energy_is_measured`, `energy_usable_for_total`,
  `energy_backend`, `energy_method` and `energy_quality` explain what the
  energy value means.

On Windows, NVML GPU-only readings are partial deployment telemetry. They are
not total pipeline energy and must not be used to overwrite benchmark energy.
The feedback analyzer follows the same rule: energy prediction errors are
computed only when `energy_usable_for_total=true` and `local_energy_j` is
numeric.

The feedback calibration proposal follows the same energy rule. It computes
`energy_scale` only from successful rows with numeric predicted energy, numeric
`local_energy_j` and `energy_usable_for_total=true`. GPU-only readings are
ignored for total-energy scale proposals.

The proposal validator uses the same rule again. Energy validation is computed
only for rows with usable total energy and a proposal marked
`usable_for_energy=true`; GPU-only rows remain excluded from total-pipeline
energy error.
