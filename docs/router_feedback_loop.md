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
```

The validator compares prediction error before and after a proposed scale using
absolute log error. It is still read-only: it writes validation reports, but it
does not change router decisions, calibration files, benchmark CSVs or
normalization.

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
