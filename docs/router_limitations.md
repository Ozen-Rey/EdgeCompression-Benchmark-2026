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
