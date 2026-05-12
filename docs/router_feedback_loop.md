# Router Online Feedback Loop

Router v0.12.0 introduces append-only feedback logging for executions requested
with `--execute`.

The data sources stay separate:

```text
benchmark CSV = prior offline
calibration JSON = local controlled adaptation
online feedback CSV = observed deployment trace
```

The feedback CSV records predicted router values, observed execution values and
energy provenance. It does not update benchmark CSVs and is not used as router
input in v0.12.0.

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
