# R-D-E Router: methodological limitations

## Local energy calibration

The current local calibration layer estimates local energy by scaling benchmark energy with the local/benchmark runtime ratio:

```txt
energy_calibrated = energy_benchmark * (local_time_ms / benchmark_time_ms)
```

This is an approximation, not a direct energy measurement.

The approximation implicitly assumes that the average power draw is comparable between the benchmark machine and the local target machine. This assumption may not hold across heterogeneous systems.

Examples:

- a mobile laptop CPU may be slower but draw less power;
- a desktop workstation may be faster but draw more power;
- a server or edge device may have a very different power/performance curve.

Therefore, locally calibrated energy values produced by time scaling must be interpreted as estimates.

The router reports this explicitly through:

```json
{
  "estimated": ["energy_by_time_scaling"],
  "not_calibrated": ["quality"]
}
```

## Energy measurement provenance

In router v0.9.1, energy values used by the R-D-E decision layer are loaded
from benchmark CSV files or obtained through local calibration by time-scaling
benchmark energy. Therefore, locally calibrated energy should be interpreted as
an estimate, not as a direct hardware telemetry measurement.

Router v0.10.0 introduces hardware energy backends with explicit provenance.
When a backend is available, the calibration report stores measured local
energy and tags it with backend, method and quality fields. When no backend is
available, the router keeps `energy_is_measured=false` and falls back to the
time-scaling estimator instead of presenting the estimate as measured energy.

A hardware reading is not automatically equivalent to total pipeline energy.
For the currently executable image backends (JPEG/Pillow, JPEG XL/cjxl and
HEVC/ffmpeg), local energy is usable as total energy only when the measurement
scope includes CPU energy. A GPU-only NVML reading is still recorded, but it is
tagged as `energy_scope=gpu` and `energy_usable_for_total=false`; the router
then keeps using the benchmark-energy time-scaling fallback for the R-D-E
energy term.

Router v0.10.1 adds an explicit local energy policy, stored as
`energy_mode` in calibration and router reports:

- `auto`: use `local_energy_j` only when it is measured and
  `energy_usable_for_total=true`; otherwise use benchmark energy scaled by the
  local/benchmark time ratio.
- `require-measured-total`: require usable total hardware energy. Calibration
  fails fast when a local run cannot provide total energy, and applying a
  calibration skips points whose energy is not usable as total energy.
- `benchmark-only`: ignore local hardware energy even if present, and use only
  benchmark energy scaled by the local/benchmark time ratio.

Router v0.11.0 adds conservative Windows backend detection and provenance.
On Windows, NVIDIA NVML total-energy counters and NVML sampled-power integrals
are recorded as hardware GPU energy when available, with
`energy_scope=gpu` and `energy_usable_for_total=false`. This is intentionally
not treated as total pipeline energy for the current CPU-side codecs.

Windows Intel PCM detection is kept as a disabled skeleton until per-command
package-energy parsing is implemented and validated. Windows `powercfg` is
diagnostic-only and is not used as Joule telemetry for calibration or routing.

## Online feedback loop

Router v0.12.0 adds append-only online feedback logging for runs executed with
`--execute`. This feedback is observational only. It records what the router
predicted, what was observed during execution, and the energy provenance of the
execution measurement.

The feedback CSV does not update benchmark CSV files, does not update
calibration JSON files, and is not used as an input to `J_RDE` in this release.
Offline benchmark values, controlled local calibration, and online deployment
traces remain separate data sources.

Local execution energy in feedback can be total, partial, or unavailable. A
Windows NVML GPU-only reading remains partial (`energy_scope=gpu`,
`energy_usable_for_total=false`) and is not comparable with total pipeline
energy for CPU-side codecs.

Router v0.13.0 adds read-only feedback analysis. It audits prediction error in
the append-only feedback CSV but does not change router decisions, `J_RDE`,
normalization, calibration, or benchmark CSV files. Energy error metrics are
computed only for rows with usable total energy; partial GPU-only telemetry is
reported as provenance but excluded from total-energy error calculations.

Router v0.14.0 adds feedback-derived calibration proposals, but only in shadow
mode. The proposal JSON/CSV files can summarize observed rate, time and usable
total-energy scale factors from online feedback, yet the router does not load
or apply them automatically. This is not online learning: ranking,
normalization, `J_RDE`, `calibration_apply`, benchmark CSVs and content-aware
logic remain unchanged unless a future release explicitly wires proposal files
as an opt-in input.

Energy proposals follow the same provenance rule as the router. A GPU-only
Windows NVML observation is useful telemetry, but it is partial and is not used
to derive a total pipeline `energy_scale`.

Router v0.15.0 adds offline validation of feedback-derived proposal files. It
checks whether proposed rate, time and usable total-energy scales would reduce
observed prediction error on the feedback CSV, using absolute log error before
and after scaling. The validation report is read-only and still does not perform
online learning: it does not modify router decisions, `J_RDE`, normalization,
`calibration_apply`, benchmark CSVs, content-aware logic, or backend execution.

Energy validation is restricted to rows with `energy_usable_for_total=true`.
GPU-only telemetry remains excluded from total pipeline energy validation, even
when a GPU energy number is present in feedback.

Router v0.16.0 adds a promotion gate for validated feedback-derived scales.
Promotion only creates an offline candidate calibration profile after explicit
threshold checks for sample count, improvement and residual log error. Even
after v0.16.0, feedback-derived calibration is not online learning: the router
does not consume the candidate profile automatically, and deployment requires a
future explicit opt-in mechanism.

Energy scale promotion remains restricted to validation rows backed by usable
total-energy evidence. GPU-only telemetry is not promotable as total pipeline
energy.

Router v0.17.0 allows promoted calibration profiles to be applied only through
an explicit `calibration_apply --promotion-profile` opt-in. Default router and
calibration behavior remain benchmark/local-calibration driven when the flag is
absent. This is still not automatic online learning: promoted profiles are not
discovered or loaded by the router, and no feedback-derived scale is used unless
the user names the profile file explicitly.

Energy promotion application preserves the total-energy rule. An energy scale
is applied only when the promoted profile marks it as accepted/promoted/usable
and the profile carries usable total-energy evidence; GPU-only or otherwise
partial energy remains excluded.

Router v0.18.0 adds an optional calibration bundle manifest produced by
`calibration_apply --manifest-out`. The manifest is an audit artifact: it
records source files, output CSV, applied promoted scales, rejected/non-applied
scale count, energy policy and SHA256 hashes. It is not a decision input. The
router still does not read `online_feedback.csv`, proposal files, validation
files, promotion profiles or calibration manifests automatically.

This release prepares a future explicit `--calibrated-database` /
`--calibration-manifest` workflow, but does not introduce that router mode yet.

Router v0.19.0 introduces that workflow in conservative form through
`rde_router --calibration-bundle-manifest`. The router consumes a calibrated CSV
only when the user names a manifest explicitly. It validates manifest structure,
requires the calibrated CSV to exist, and verifies the CSV SHA256 hash before
loading it. Hash mismatch, missing CSV, or incomplete manifest produce a
controlled error.

No automatic discovery is performed in v0.19.0. The router still does not read
`online_feedback.csv`, proposal files, validation files, or promotion profiles
directly, and it does not use the latest available manifest implicitly.

Router v0.20.0 adds `calibration_impact_audit`, a read-only tool for comparing
the baseline router decision with the decision obtained from an explicitly
validated calibration bundle manifest. This measures the impact of a bundle on
selected codec/config and R-D-E terms, but it does not introduce automatic
bundle discovery, online learning, backend execution, or changes to router
defaults.

Router v0.21.0 adds `shadow_decision_comparison`, another offline/read-only
what-if layer. It validates a named bundle manifest, compares the baseline CSV
decision against the calibrated CSV decision under the same router config, and
reports changed/unchanged selections plus aggregate deltas. This still does not
show that the calibrated choice is better; it only measures potential decision
impact. Utility validation remains a future offline regret/oracle/ablation
step.

Router v0.22.0 adds `shadow_decision_validation`, a read-only gate over shadow
comparison reports. The gate can reject candidates for insufficient decisions,
excessive decision churn, mean cost regression, quality/rate/energy/time guard
violations, or unsafe energy provenance. It does not enable automatic learning,
automatic bundle consumption, or operational decision changes. A rejected
candidate has no effect on the router.

Router v0.23.0 optionally lets `rde_router` require an accepted shadow decision
validation report before consuming an explicitly named calibration bundle. This
does not introduce automatic learning. It only allows an explicitly provided
calibration bundle to be consumed together with an explicitly provided accepted
shadow validation report. Rejected, missing or malformed validation blocks
consumption when the validation flag is used; without that flag, v0.19 bundle
behavior is preserved.

Router v0.24.0 adds cross-artifact integrity binding. An accepted validation
report is no longer enough by itself: when the validation flag is used, the
validation report must carry v0.24 hash provenance and its candidate calibration
bundle manifest SHA256 must match the actual manifest passed to the router.
This prevents accidentally pairing a validation report for Bundle A with Bundle
B. No automatic discovery, online learning, ranking change or decision-policy
change is introduced.

Router v0.25.0 adds a decision receipt and offline replay validator. The
receipt is an audit artifact embedded in router reports; replay verifies that
the same inputs still hash to the recorded values and that rerunning the router
reproduces the same selected decision. Replay is not a policy gate and does not
prove the selected codec is better; it only checks reproducibility of the
recorded decision under the same inputs.

Router v0.26.0 adds `router_overhead_audit`, a read-only performance audit for
router modes. It measures runtime overhead empirically but does not change
decision logic, scoring, normalization, calibration, bundle validation,
feedback, replay semantics, content-aware behavior, system-aware behavior, or
backend execution. Bundle and validation modes are only measured when their
paths are provided explicitly; no latest-file discovery is performed.

Router v0.27.0 adds `router_effectiveness_audit`, a read-only baseline-policy
effectiveness audit. It compares the router choice against simple policies such
as lowest rate, highest quality, lowest energy and fastest time under the same
explicit CSV/config inputs. These baselines do not bypass quality or constraint
guards; violating candidates are reported as non-comparable rather than treated
as valid alternatives. Bundle and validated-bundle effectiveness are audited
only when their paths are provided explicitly. The module writes audit reports
but does not change `J_RDE`, ranking, normalization, calibration, bundle
validation, feedback, replay semantics, content-aware behavior, system-aware
behavior, or backend execution.

Router v0.28.0 refines the audit output rather than improving router behavior.
The effectiveness audit now separates three cases: comparable baselines with a
router-exported cost, filtered candidates that violate quality or constraints,
and feasible-but-unscored candidates that are visible in the raw candidate pool
but absent from the exported scored pool. Regret is computed only for the first
case. If cost is unavailable, the audit reports `cost_status` and
`cost_reason_detail` instead of inventing `J_RDE` outside the router.

Router v0.31.0 adds a normalization consistency audit for router reports, but
only when a previous receipt/report is passed explicitly with
`--previous-decision-receipt`. This is a provenance check, not a decision gate:
it reports whether the previous and current `normalization_audit` fields appear
comparable and emits warnings for changed modes, quality metrics, quality
directions, scale sources, runtime-computation status or numeric ranges.

No automatic discovery is performed. The router does not look for prior
receipts, latest reports or nearby artifacts. Missing, malformed or
normalization-audit-free previous files are reported as controlled
`normalization_consistency` warnings/errors in the current report.

The audit does not change `J_RDE`, ranking, normalization logic, calibration
bundle logic, energy backends, content-aware behavior, system-aware behavior or
the selected codec/config. It only helps decide whether two reported runs can
be compared methodologically.

Router v0.32.0 records codec executable fingerprints in newly written
calibration bundle manifests and validates them when those bundles are consumed
explicitly. This reduces the risk of applying a calibrated CSV produced with a
different codec backend than the one currently declared by the bundle.

The check is intentionally narrow. It verifies declared backend/version/hash
provenance for codec executables and fails closed on version mismatch, binary
hash mismatch or missing declared binary. It does not prove that the calibrated
bundle improves decisions, and it does not alter scores, ranking,
normalization, feedback, content-aware behavior, system-aware behavior or
backend execution. Legacy manifests without fingerprints are still accepted and
reported with `codec_fingerprint_validation.enabled=false`.

Router v0.32.1 clarifies that an empty `codec_fingerprints` object is also not
a successful validation. It is reported as disabled with
`reason=empty_codec_fingerprints`, while manifests that apply local calibration
without a promotion profile still attempt to fingerprint the affected codecs.

Router v0.33.0 adds energy provenance tier reporting. The selected candidate,
scored pool, unscored pool and `top_k` entries expose
`energy_provenance_tier`, and the report includes an
`energy_provenance_summary`. The tiers distinguish usable total hardware
energy, partial hardware telemetry, time-scaled estimates, benchmark reference
energy and unknown provenance.

These tiers are not policy inputs. GPU-only or otherwise partial hardware
telemetry is never promoted to total pipeline energy by the tier reporter, and
the tier does not change `J_RDE`, ranking, normalization, filtering, bundle
consumption, calibration application, content-aware behavior, system-aware
behavior or backend execution.

Router v0.34.0 adds an energy provenance compatibility audit over those tiers.
It reports whether scored candidates share one provenance tier or mix benchmark
reference, time-scaled, measured-total, measured-partial or unknown energy
sources. Mixed scored pools are warnings; a selected partial hardware tier is
critical; a selected unknown tier is a warning.

This is still not a router policy. The compatibility audit does not filter,
rerank, penalize or normalize differently. It only tells readers when an R-D-E
decision compared candidates whose energy values came from different provenance
classes.

Router v0.35.0 adds an `energy_tier_policy` shadow audit. It simulates a
future strict-compatible tier policy using a conservative reliability order,
but it does not apply that policy. In particular, `measured_hw_partial` is never
preferred over measured-total, time-scaled or benchmark-reference energy, and
`unknown` remains the weakest tier.

The shadow audit can say that a future policy would choose a different
candidate, but v0.35.0 still keeps the real router decision, ranking, `J_RDE`,
normalization, bundle consumption, calibration application, feedback,
content-aware behavior, system-aware behavior and backend execution unchanged.

Router v0.36.0 adds an offline external codec specification validator. A valid
spec is only a well-formed declaration; it is not proof that a codec is safe,
deterministic, accurate, comparable or locally installed. The validator does
not execute `version_probe`, does not run encode/decode commands, does not
check binary existence, and does not add the codec to the router registry.
Command templates must be argv lists and `security.allow_shell=true` is
rejected, but black-box codec behavior still requires separate review before
any future integration.

Router v0.37.0 adds a controlled external codec probe, but its scope remains
narrow. It can confirm that a declared executable path exists, compute a binary
SHA256, and run a declared version command with `shell=False` under a timeout.
This still does not validate encode/decode behavior, output correctness,
rate-distortion-energy characteristics, determinism, sandbox safety or dataset
performance. The probe does not register the codec, does not benchmark it, and
does not affect router decisions or existing backend execution.

Router v0.38.0 adds a one-input external codec dry-run contract check. It can
execute encode and optional decode commands declared in an external spec, but
only against one explicitly named input and only inside a controlled output
directory. Passing this dry-run means the basic I/O contract worked once; it
does not establish quality, rate, energy, dataset robustness, deterministic
behavior, security of black-box code, or suitability for router ranking. The
dry-run still does not add the codec to the registry, candidate pool, `J_RDE`,
normalization, content-aware/system-aware paths or existing backend execution.

The intended backend hierarchy is:

1. Linux: RAPL for CPU package energy and Zeus/NVML for NVIDIA GPU energy.
2. Windows NVIDIA GPU: NVML/Zeus when available, with nvidia-smi sampling as fallback.
3. Windows AMD CPU: AMD uProf CLI when available.
4. Windows Intel CPU: Intel PCM when available.
5. Fallback: calibrated time-scaling estimator.

All local energy values should be tagged with explicit provenance fields such
as `energy_backend`, `energy_method`, `energy_is_measured`, and
`energy_quality`.
