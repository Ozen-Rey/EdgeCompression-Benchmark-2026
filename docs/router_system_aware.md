# R-D-E Router: system-aware layer

## Overview

The v0.8 router extends the static Rate-Distortion-Energy decision engine with a system-aware layer.

The baseline router minimizes:

```txt
J_RDE = w_R * R_norm + w_E * E_norm + w_D * D_norm
```

where:

- `R_norm` is normalized rate.
- `E_norm` is normalized energy.
- `D_norm` is normalized distortion, derived from quality.
- `w_R`, `w_E`, and `w_D` are normalized weights.

The system-aware layer adds two mechanisms:

1. System policy.
2. System penalty.

The system policy modifies the R-D-E weights according to the current or simulated system state.

The system penalty introduces an operational penalty:

```txt
J_total = J_RDE + lambda_sys * P_sys
```

where:

- `P_sys` is a normalized operational penalty.
- `lambda_sys` controls the strength of the system penalty.

When system penalty mode is `apply`, the router ranks candidates by `J_total` instead of `J_RDE`.

## System Feature Extraction

Implemented in:

```txt
src/router/system_features.py
```

The router can extract low-overhead system features:

- CPU load.
- Memory availability.
- Swap usage.
- Battery / AC status.
- Disk space.
- GPU status through `nvidia-smi`, when requested.
- Available executables.

The supported probe levels are:

- `basic`: CPU, RAM, swap, battery, disk, executables.
- `gpu`: `basic` plus `nvidia-smi`.
- `full`: `gpu` plus torch CUDA probe.

Example:

```powershell
python -m src.router.adaptation.system_features `
  --probe-level basic `
  --out results/routing_context/v08_system_features_basic.json
```

The report includes probe overhead:

```json
"probe_overhead": {
  "static_probe_ms": 47.1,
  "dynamic_probe_ms": 0.2,
  "gpu_probe_ms": null,
  "total_probe_ms": 47.3
}
```

The router also estimates whether the probe overhead is acceptable relative to the selected candidate timing.

## Derived System Constraints

Raw system features are converted into discrete classes:

```json
"classes": {
  "cpu": "normal",
  "memory": "normal",
  "battery": "unknown",
  "gpu": "unknown",
  "thermal": "unknown",
  "disk": "normal"
}
```

Possible values include:

- `cpu`: `normal` | `busy`
- `memory`: `normal` | `constrained` | `critical`
- `battery`: `ac` | `battery` | `low` | `critical` | `unknown`
- `gpu`: `available` | `busy` | `memory_constrained` | `unavailable` | `unknown`
- `thermal`: `nominal` | `hot` | `critical` | `unknown`
- `disk`: `normal` | `low_space`

Important methodological rule:

If `probe_level=basic`, GPU is not checked. Therefore GPU status is `unknown`, not `unavailable`.

This prevents false CUDA exclusions when the GPU was simply not probed.

## System Policy

Implemented in:

```txt
src/router/system_policy.py
```

The system policy maps system constraints to weight modifications.

Example rules:

- `battery=critical`: increase energy weight.
- `cpu=busy`: increase energy weight.
- `memory=constrained`: prefer lighter configurations.
- `thermal=hot`: increase energy weight.
- `gpu=unknown`: warning, no CUDA hard exclusion.

The policy supports two modes:

- `report-only`: compute suggested weights, but do not alter the decision.
- `apply`: apply the modified weights.

Example:

```powershell
python -m src.router.rde_router `
  --config configs/router_image_v08.json `
  --system-policy-mode apply `
  --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
  --out results/routing_context/v08_system_policy_apply.json
```

A simulated system state can be used for reproducible validation:

```txt
--system-policy-simulate "battery=critical,cpu=busy,memory=constrained"
```

## Codec Resource Profiles

Each codec can define a resource profile.

Example:

```json
"resource_profile": {
  "cpu_load": "high",
  "memory": "medium",
  "gpu": "none",
  "latency": "high",
  "energy": "high",
  "batch_friendly": true,
  "interactive_ok": false
}
```

The resource profile describes the expected operational cost of a codec/backend.

Typical values:

- `cpu_load`: `low` | `medium` | `high` | `very-high`
- `memory`: `low` | `medium` | `high` | `very-high`
- `gpu`: `none` | `low` | `medium` | `high` | `very-high`
- `latency`: `low` | `medium` | `high` | `very-high`
- `energy`: `low` | `medium` | `high` | `very-high`

These values are heuristic metadata. They are not measured during routing.

## System Penalty

Implemented in:

```txt
src/router/system_penalty.py
```

The system penalty computes an operational risk penalty from:

```txt
system constraints + codec resource profile
```

The total ranking score is:

```txt
J_total = J_RDE + lambda_sys * P_sys
```

where `P_sys` is normalized in `[0, 1]`.

Example system penalty rules:

- `battery=critical` plus `codec.energy=high`: penalty.
- `cpu=busy` plus `codec.cpu_load=high`: penalty.
- `memory=constrained` plus `codec.memory=medium`: penalty.
- `gpu=unavailable` plus `requires_cuda=true`: hard exclusion.
- `memory=critical` plus `codec.memory=high`: hard exclusion.

The system penalty has two modes:

- `report-only`: compute `J_total`, but rank by `J_RDE`.
- `apply`: rank by `J_total` and apply hard exclusions.

Example:

```powershell
python -m src.router.rde_router `
  --config configs/router_image_v08.json `
  --system-policy-mode apply `
  --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
  --system-penalty-mode apply `
  --system-penalty-lambda 0.5 `
  --out results/routing_context/v08_system_penalty_apply.json
```

## Configurable System Penalty Coefficients

System penalty coefficients are stored in:

```txt
configs/system_penalty_weights_v08.json
```

Example:

```json
{
  "battery": {
    "critical_energy": 0.10,
    "low_energy": 0.06,
    "battery_energy": 0.03
  },
  "cpu": {
    "busy_cpu": 0.08
  }
}
```

These coefficients are heuristic and should be interpreted as experimental parameters. They are configurable to support sensitivity analysis and ablation studies.

## Validation

The v0.8 validation runs through the unified smoke dispatcher:

```txt
scripts/run_router.ps1 -Scenario v08-system-aware
```

It validates:

- Basic system feature extraction.
- GPU system feature extraction.
- System policy in report-only mode.
- System policy in apply mode.
- System penalty ranking by `J_total`.
- Pytest regression suite.

Run:

```powershell
.\scripts\run_router.ps1 -Scenario v08-system-aware
```

Expected behavior on the synthetic decision-change case:

```txt
report-only -> QualityHeavy
apply       -> EnergyLight
penalty     -> ranking by J_total
```

## Methodological Interpretation

The system-aware layer separates three quantities:

1. R-D-E benchmark cost.
2. Current system state.
3. Operational risk penalty.

This allows the router to move from static benchmark-based selection to adaptive deployment-aware codec selection.

The router can therefore be interpreted as minimizing:

```txt
J_total = J_RDE + lambda_sys * P_sys
```

where `J_RDE` measures benchmark-derived R-D-E efficiency and `P_sys` measures system-dependent operational risk.
