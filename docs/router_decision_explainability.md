# Router Decision Explainability

## Scope

`src.router.observability.decision_explanation` is an offline,
read-only renderer that turns an existing router report JSON into a
human-readable decision explanation. It is intended for the paper /
thesis narrative and for debugging individual decisions.

What this module **does not** do:

- It does not re-run the router.
- It does not read benchmark CSVs, calibration bundles, or feedback
  logs.
- It does not change the operational report schema, the ranking score
  (J_RDE / J_total), or any CLI flag of `rde_router`.
- It does not propose, modify, or apply any calibration change.

The router runtime is unaffected by this module. The same input report
JSON always produces the same explanation: the render is deterministic.

## Usage

```powershell
python -m src.router.observability.decision_explanation `
  --report router_report.json `
  --out-md decision_explanation.md `
  --out-json decision_explanation.json
```

After `pip install -e .`, the `rde-decision-explain` console script is
also available on `PATH` and is equivalent to the module invocation
above.

The CLI accepts:

- `--report PATH` (required) — path to an existing router report JSON.
- `--out-md PATH` (optional) — write the Markdown explanation.
- `--out-json PATH` (optional) — write the structured explanation JSON.
- `--print` (optional) — also print the Markdown explanation to stdout;
  this is the default when neither `--out-md` nor `--out-json` is
  provided.

## What the explanation contains

The Markdown render and the structured JSON share the same sections:

- **Selected candidate** — codec, config, rate, quality, energy,
  time, the operational `selected_reason` string, the active pool
  (`safe_pool` or `near_pool`), the active ranking key
  (`minimize_J_RDE` or `minimize_J_total`), and the active ranking
  value.
- **Why this candidate was selected** — deterministic bullets derived
  from the report: passing the quality guard, membership in the
  evaluated admissible pool, having the lowest active ranking score
  among admissible candidates, and (when applicable) acceptance or
  rejection of a preferred candidate from the content-policy or
  content-classifier layer.
- **Active constraints** — quality guard, max rate / energy / time,
  normalization mode and scope, codec filtering (strict executables
  and capability filtering), system policy, system penalty,
  calibration / calibration bundle, energy provenance, and time
  guard. Disabled blocks are omitted entirely; missing values render
  as `unavailable`.
- **Predictor role** — frames the content-aware predictor as
  consultative: the predictor can suggest a codec/configuration, but
  the router accepts it only if it remains admissible (passes the
  quality guard and other hard constraints) and competitive under the
  active ranking score. Includes the predictor mode (`report-only` or
  `apply`), the applied flag, the suggestion or prediction itself,
  and the `preferred_candidate` audit block when present.
- **Fallback / safety** — exactly one of:
  - `safe_pool_selection` — selected from the safe admissible pool;
  - `degraded_fallback_selection` — safe pool empty under the active
    constraints; selected from the near-floor pool (operator opt-in);
  - `preferred_candidate_accepted` — the predictor's suggestion was
    admissible and competitive on the active ranking score;
  - `preferred_candidate_rejected` — the predictor's suggestion was
    not admissible or not competitive; the router fell back to its
    own ranked choice;
  - `infeasible_request` — the report does not carry a selected
    candidate (e.g. emitted when the request is infeasible under the
    active constraints and the report is produced defensively).
- **Cost decomposition** — `w_R`, `w_E`, `w_D`, the per-axis
  normalized values, the per-axis weighted terms, the unweighted
  J_RDE, and (when the system penalty is applied) the system penalty
  itself, `lambda_sys`, and J_total. The "active ranking" field
  always matches the field actually used by the router for the
  selection.

## Framing rules

The render uses scoped wording on purpose:

- It says "selected within the evaluated admissible pool" rather than
  "best possible". The router only ranks candidates that pass the hard
  constraints; candidates outside that pool are not compared.
- It treats predictor outputs as consultative. The framing block is
  fixed text and is not rewritten on a per-decision basis.
- It does not invent values. If a report block (cost decomposition,
  calibration bundle, energy provenance details, time guard, ...) is
  absent, the corresponding fields render as `unavailable` and the
  structured JSON carries `null` for those keys.

This is deliberate: the module is meant to make the router's actual
behavior explicit for an external reader (thesis reviewer, paper
reader, audit auditor), not to embellish it.

## Relation to other observability artefacts

- `decision_receipt` / `decision_replay` are the audit-replay receipts
  for full reproducibility of a decision; they are exact-replay
  artefacts, not explanations.
- `policy_comparison` (v0.43.0) compares the *aggregate* regret of
  content-aware policies across an evaluation corpus with paired
  bootstrap CIs; this module explains a *single* decision.
- `router_overhead_audit` and `router_effectiveness_audit` are
  performance / baseline-policy audits; they do not render the
  per-decision rationale.

The four modules are complementary: `decision_explanation` is the
narrative layer, the receipts/replay are the reproducibility layer,
and the audits are the cost/effectiveness layer.
