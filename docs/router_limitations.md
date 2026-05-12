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

The intended backend hierarchy is:

1. Linux: RAPL for CPU package energy and Zeus/NVML for NVIDIA GPU energy.
2. Windows NVIDIA GPU: NVML/Zeus when available, with nvidia-smi sampling as fallback.
3. Windows AMD CPU: AMD uProf CLI when available.
4. Windows Intel CPU: Intel PCM when available.
5. Fallback: calibrated time-scaling estimator.

All local energy values should be tagged with explicit provenance fields such
as `energy_backend`, `energy_method`, `energy_is_measured`, and
`energy_quality`.
