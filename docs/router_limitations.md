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

The intended backend hierarchy is:

1. Linux: RAPL for CPU package energy and Zeus/NVML for NVIDIA GPU energy.
2. Windows NVIDIA GPU: NVML/Zeus when available, with nvidia-smi sampling as fallback.
3. Windows AMD CPU: AMD uProf CLI when available.
4. Windows Intel CPU: Intel PCM when available.
5. Fallback: calibrated time-scaling estimator.

All local energy values should be tagged with explicit provenance fields such
as `energy_backend`, `energy_method`, `energy_is_measured`, and
`energy_quality`.
