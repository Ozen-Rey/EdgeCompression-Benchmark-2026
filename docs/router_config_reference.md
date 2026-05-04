# R-D-E Router Configuration Reference

This document describes the JSON configuration format used by the R-D-E router.

Main example:

```txt
configs/router_image_v08.json
```

## Top-Level Fields

### experiment_name

Human-readable name of the experiment.

```json
"experiment_name": "image_v08_system_aware"
```

Used in reports for traceability.

### domain

Multimedia domain.

```json
"domain": "image"
```

Supported values:

- `image`
- `audio`
- `video`
- `data`

## data

Input/output files for the router.

```json
"data": {
  "csv": "results/images/image_4dataset_RDE_paper_ready.csv",
  "input": "test_images/input.png",
  "output": "test_images/v05_router_output.mp4",
  "out_report": "results/routing_context/v08_router_report.json",
  "out_dir": "results/routing_context"
}
```

Fields:

- `csv`: benchmark CSV.
- `input`: optional input file for command generation / execution.
- `output`: desired output path.
- `out_report`: JSON report path for single run.
- `out_dir`: output directory for multi-run use.

## columns

Column mapping for the benchmark CSV.

```json
"columns": {
  "codec": "codec",
  "config": "param",
  "rate": "bpp",
  "quality": "ssimulacra2",
  "energy": "energy_per_image_j",
  "time": "time_ms"
}
```

For image experiments, typical mapping is:

- `rate`: `bpp`
- `quality`: `ssimulacra2`
- `energy`: `energy_per_image_j`
- `time`: `time_ms`

## selection

Core selection options.

```json
"selection": {
  "available_codecs": ["JPEG", "JXL", "HEVC"],
  "exclude_codecs": null,
  "exclude_neural": false,
  "aggregate_by_config": true,
  "auto_weights": true,
  "all_profiles": false,
  "safe_mode": true,
  "quality_constraint_stat": "min",
  "quality_target": "high",
  "quality_floor": 70,
  "near_quality_floor": 60,
  "allow_degraded_fallback": true,
  "max_rate": null,
  "max_energy": null,
  "max_time_ms": null,
  "strict_time": false
}
```

Important fields:

- `available_codecs`: restricts candidate codecs.
- `aggregate_by_config`: aggregates benchmark rows by codec/config.
- `auto_weights`: derives weights from context policy.
- `safe_mode`: stricter quality guard.
- `quality_constraint_stat`: `mean` | `p25` | `p10` | `min`.
- `quality_target`: `preview` | `normal` | `high` | `very-high`.
- `quality_floor`: user floor; cannot weaken target floor.
- `near_quality_floor`: degraded fallback threshold.
- `allow_degraded_fallback`: enables degraded fallback pool.
- `max_rate`: hard rate constraint.
- `max_energy`: hard energy constraint.
- `max_time_ms`: hard time constraint.
- `strict_time`: error if timing data is missing.

## system

Hardware/software capability filtering.

```json
"system": {
  "system_aware": true,
  "simulate_no_cuda": false,
  "capability_aware": true,
  "strict_executables": true
}
```

Fields:

- `system_aware`: uses system state for CUDA/no-CUDA policy.
- `simulate_no_cuda`: debug option.
- `capability_aware`: filters candidates through codec capabilities.
- `strict_executables`: excludes codecs with missing required executables.

## system_features

Cheap system feature extraction.

```json
"system_features": {
  "enabled": true,
  "probe_level": "basic",
  "cache_ttl_s": 5.0,
  "cpu_interval_s": 0.0
}
```

Fields:

- `enabled`: include system features in report.
- `probe_level`: `basic` | `gpu` | `full`.
- `cache_ttl_s`: cache lifetime for repeated probes.
- `cpu_interval_s`: psutil CPU sampling interval.

Probe levels:

- `basic`: CPU, RAM, swap, battery, disk, executables.
- `gpu`: `basic` plus `nvidia-smi`.
- `full`: `gpu` plus torch CUDA probe.

## system_policy

System-aware weight adaptation.

```json
"system_policy": {
  "enabled": true,
  "mode": "report-only",
  "simulate": null
}
```

Fields:

- `enabled`: build system policy.
- `mode`: `report-only` | `apply`.
- `simulate`: optional simulated classes.

Example simulation:

```json
"simulate": "battery=critical,cpu=busy,memory=constrained"
```

In `report-only`, suggested weights are recorded but not used.

In `apply`, suggested weights become effective router weights.

## system_penalty

Operational risk penalty.

```json
"system_penalty": {
  "enabled": true,
  "mode": "report-only",
  "lambda_sys": 0.25,
  "weights_file": "configs/system_penalty_weights_v08.json"
}
```

Fields:

- `enabled`: compute system penalty.
- `mode`: `report-only` | `apply`.
- `lambda_sys`: multiplier for `P_sys`.
- `weights_file`: configurable coefficient file.

Formula:

```txt
J_total = J_RDE + lambda_sys * P_sys
```

In `report-only`, `J_total` is reported but ranking still uses `J_RDE`.

In `apply`, ranking uses `J_total`.

## context

Contextual policy input for auto weights.

```json
"context": {
  "power_mode": "ac",
  "battery_percent": null,
  "thermal_state": "nominal",
  "network_profile": "normal",
  "system_load": "normal"
}
```

Used when:

```json
"auto_weights": true
```

## normalization

Normalization profile.

```json
"normalization": {
  "mode": "global",
  "file": "results/routing_context/normalization_image.json"
}
```

Supported modes:

- `auto`
- `runtime`
- `global`
- `dataset`
- `local`

Use `global`, `dataset`, or `local` with a precomputed normalization profile.

Runtime normalization is useful for debugging but not recommended for cross-run comparisons.

## calibration

Local calibration file.

```json
"calibration": {
  "file": null
}
```

If provided, the router applies local calibration to rate/time and estimated energy.

Current limitation:

```txt
energy is estimated by benchmark_energy_scaled_by_time_ratio
```

See:

```txt
docs/router_limitations.md
```

## quality_thresholds

Domain-specific quality thresholds.

```json
"quality_thresholds": {
  "file": "configs/quality_thresholds.json"
}
```

Example thresholds:

- `image / ssimulacra2 / high`: 80.
- `video / vmaf / high`: 90.
- `audio / visqol / high`: 4.0.

The effective floor is:

```txt
max(target_floor, user_quality_floor)
```

The user can make the constraint stricter, not weaker.

## codec_registry

External codec/backend registry.

```json
"codec_registry": {
  "enabled": true,
  "file": "configs/codecs_image_v05.json"
}
```

Allows codec metadata and external command templates to be declared outside the Python code.

## execution

Command generation and execution.

```json
"execution": {
  "generate_command": true,
  "execute": false
}
```

Fields:

- `generate_command`: produce execution plan.
- `execute`: execute selected command.

When `execute=true`, the router also validates output existence, non-empty output size, and extension compatibility.

## topk

Top-k candidate export.

```json
"topk": {
  "export": false,
  "k": 5
}
```

Fields:

- `export`: write top-k CSV.
- `k`: number of candidates.

## Minimal Examples

### Static R-D-E Routing

```json
{
  "domain": "image",
  "data": {
    "csv": "results/images/image_4dataset_RDE_paper_ready.csv",
    "out_report": "results/routing_context/router_report.json"
  },
  "columns": {
    "codec": "codec",
    "config": "param",
    "rate": "bpp",
    "quality": "ssimulacra2",
    "energy": "energy_per_image_j",
    "time": "time_ms"
  },
  "selection": {
    "available_codecs": ["JPEG", "JXL", "HEVC"],
    "aggregate_by_config": true,
    "quality_target": "high",
    "quality_constraint_stat": "min",
    "allow_degraded_fallback": true
  }
}
```

### System-Aware Routing

```json
{
  "domain": "image",
  "system_features": {
    "enabled": true,
    "probe_level": "basic"
  },
  "system_policy": {
    "enabled": true,
    "mode": "apply"
  },
  "system_penalty": {
    "enabled": true,
    "mode": "apply",
    "lambda_sys": 0.25,
    "weights_file": "configs/system_penalty_weights_v08.json"
  }
}
```

## Recommended Workflow

For experiments:

```txt
system_policy.mode  = report-only
system_penalty.mode = report-only
```

For deployment/adaptive routing:

```txt
system_policy.mode  = apply
system_penalty.mode = apply
```

For reproducible stress tests:

```txt
system_policy.simulate = "battery=critical,cpu=busy,memory=constrained"
```
