# Thesis Tables Provenance Audit

Date: 2026-05-20

Scope: provenance audit for the substantive table inconsistencies in
`<thesis-root>`, with read-only inspection of the router repository.
No router code, benchmark code, or heavy result generation was run.

## 1. Table inventory and provenance

| table_tex | source_csv_or_json | generator_script_or_module | generation_command_if_known | last_modified | included_in_chapter | caption | status |
|---|---|---|---|---|---|---|---|
| `Capitoli/capitolo5.tex` / `tab:routing_profiles` | inline/manual thesis table | manual | n/a | 2026-05-20 17:43 | Chapter 5 | Profili operativi utilizzati per la validazione offline del routing R-D-E nel dominio immagine. | manual/current |
| `Capitoli/capitolo5.tex` / `tab:image_routing_oracle` | inline/manual thesis table | manual | n/a | 2026-05-20 17:43 | Chapter 5 | Configurazioni selezionate dall'oracolo R-D-E nel dominio immagine per ciascun profilo operativo. | manual/current |
| `Capitoli/capitolo5.tex` / `tab:image_routing_regret` | inline/manual thesis table | manual | n/a | 2026-05-20 17:43 | Chapter 5 | Regret medio normalizzato delle policy di routing nel dominio immagine. | manual/current |
| `tables/routing_v09/v09_content_aware_final_table.tex` | `results/routing_context/paper_artifacts_v09/v09_content_aware_final_table.csv`; upstream `v09_content_aware_benchmark_table.csv` | `src.router.analysis.content_aware_paper_artifacts`; upstream `src.router.analysis.content_aware_benchmark_table` | `python -m src.router.analysis.content_aware_paper_artifacts --benchmark-table results/routing_context/v09_content_aware_benchmark_table.csv --overhead-table results/routing_context/v09_content_aware_overhead_table.csv --sensitivity-table results/routing_context/v09_knn_sensitivity_table.csv --oracle-summary results/routing_context/v09_content_oracle_summary.csv --out-dir results/routing_context/paper_artifacts_v09` | 2026-05-20 17:45 | Chapter 5 | Benchmark del routing R-D-E content-aware. | numeric current; caption clarified manually |
| `tables/routing_v09/v09_content_aware_best_k_table.tex` | `results/routing_context/paper_artifacts_v09/v09_content_aware_best_k_table.csv`; upstream `v09_knn_sensitivity_table.csv` | `src.router.analysis.content_aware_paper_artifacts`; upstream `src.router.analysis.content_aware_overhead_analysis` | same paper-artifacts command above | 2026-05-20 17:44 | Chapter 5 | Migliori configurazioni kNN nello sweep custom v0.9 per ciascun protocollo. | numeric current; caption clarified manually |
| `tables/routing_v09/v0910_sklearn_ablation_compact_table.tex` | `results/routing_context/paper_artifacts_v09/v0910_sklearn_ablation_compact_table.csv`; upstream `v0910_sklearn_ablation_summary.csv` | `scripts/make_v0910_sklearn_ablation_table.py`; upstream `src.router.analysis.content_oracle_classifier_sklearn_ablation` | `python scripts/make_v0910_sklearn_ablation_table.py` for compact table; upstream invocation not recorded in `run_router.ps1` | 2026-05-20 17:44 | Chapter 5 | Ablazione scikit-learn del modello predittivo. | numeric current; caption clarified manually |
| `Capitoli/capitolo5.tex` / `tab:policy_comparison_bootstrap` | `results/routing_context/policy_comparison.csv` and `.json` | `src.router.analysis.policy_comparison` | exact invocation not stored; JSON records inputs and `classifier_k=7` | 2026-05-20 17:43 | Chapter 5 | Confronto delle policy di routing content-aware sul benchmark multi-source. | current; inline table manually mirrors CSV |
| `Capitoli/capitolo6.tex` / `tab:codec_capabilities_examples` | inline/manual thesis table | manual, derived from router capability schema | n/a | 2026-05-20 17:46 | Chapter 6 | Esempi di entry della tabella CODEC_CAPABILITIES. | manual/current |
| `Capitoli/capitolo6.tex` / `tab:system_penalty_coefficients` | `configs/system_penalty_weights_v08.json` | manual thesis table from config | n/a | 2026-05-20 17:46 | Chapter 6 | Coefficienti default della system penalty. | manual/current |
| `tables/routing_v09/v09_content_aware_overhead_table_paper.tex` | `results/routing_context/paper_artifacts_v09/v09_content_aware_overhead_table_paper.csv`; upstream `v09_content_aware_overhead_table.csv` | `src.router.analysis.content_aware_paper_artifacts`; upstream `src.router.analysis.content_aware_overhead_analysis` | same paper-artifacts command above | 2026-05-20 17:45 | Chapter 6 | Overhead e tempi di codifica di riferimento. | numeric current; note added manually |
| `tables/routing_regime/table_regime_summary_core.tex` | `results/routing_context/operational_regime_simulation_ssimulacra2_quality_contract/operational_regime_summary.csv` | `src.router.analysis.operational_regime_simulation` | exact invocation not stored; report JSON records `quality_col=ssimulacra2`, `k=7`, protocols `loio/lodo`, floors 60/70/80 | 2026-05-20 17:45 | Chapter 6 | Sintesi dei regimi operativi per metadata-plus-system. | numeric current; units clarified manually |
| `tables/routing_regime/table_ssimulacra2_switch_summary.tex` | `results/routing_context/operational_regime_simulation_ssimulacra2_quality_contract/operational_regime_switch_summary.csv` | `src.router.analysis.operational_regime_simulation` | same operational-regime run | 2026-05-20 17:46 | Chapter 6 | Analisi dello switch classico-neurale. | numeric current; missing category added |
| `tables/routing_regime/table_expected_quality_gate_summary.tex` | `results/routing_context/operational_regime_simulation_ssimulacra2_quality_contract/quality_gate_comparison_summary.csv` | `src.router.analysis.operational_regime_simulation` | same operational-regime run | 2026-05-20 15:48 | not included in current thesis | Confronto shadow dell'expected-quality gate. | current vs source, but stale/unincluded in thesis |

## 2. kNN diagnosis

Source of truth by use case:

- Final policy comparison and bootstrap: `results/routing_context/policy_comparison.csv`
  and `policy_comparison.json`.
- v0.9 custom best-k sweep: `results/routing_context/v09_knn_sensitivity_table.csv`
  and generated `paper_artifacts_v09/v09_content_aware_best_k_table.csv`.
- sklearn model ablation: `results/routing_context/v0910_sklearn_ablation_summary.csv`
  and generated compact table.

Confirmed facts:

| source_file | protocol | model | feature_set | k | mean_regret | relative_reduction | selected_as | notes |
|---|---|---|---|---:|---:|---:|---|---|
| `v09_knn_sensitivity_table.csv` | LOIO | custom kNN | metadata_no_source | 11 | 0.011667 | 0.871050 | best custom LOIO | Best point in the custom v0.9 sweep. |
| `v09_knn_sensitivity_table.csv` | LOIO | custom kNN | metadata_no_source | 7 | 0.012376 | 0.863208 | deployable default when fixed to LODO-selected k | Used by final policy comparison. |
| `v09_knn_sensitivity_table.csv` | LODO | custom kNN | metadata_no_source | 7 | 0.021971 | 0.757157 | best custom LODO | k=7, 9, 11 are tied at reported precision. |
| `v0910_sklearn_ablation_summary.csv` | LOIO | sklearn kNN | metadata_no_source | 7 | 0.011667 | 0.871050 | best sklearn LOIO | Separate sklearn implementation/ablation pipeline. |
| `v0910_sklearn_ablation_summary.csv` | LODO | sklearn kNN | metadata_no_source | 7 | 0.021971 | 0.757157 | best/tied sklearn LODO | k=7, 9, 11 tied at reported precision. |
| `policy_comparison.csv` | LOIO | custom kNN fixed default | metadata_no_source | 7 | 0.012376 | 0.863208 | final deployable policy comparison | JSON records `classifier_k=7`. |
| `policy_comparison.csv` | LODO | custom kNN fixed default | metadata_no_source | 7 | 0.021971 | 0.757157 | final deployable policy comparison | JSON records `classifier_k=7`. |

The files `v09_classifier_metadata_no_source_k7_loio_decisions.csv` and
`v09_classifier_metadata_no_source_k7_lodo_decisions.csv` are exact subsets of
`v09_oracle_classifier_sweep_decisions.csv` filtered to
`feature_set=metadata_no_source`, `k=7` and the corresponding protocol.

Confirmed interpretation:

- Hypothesis A is true: k=11 is the best custom LOIO point, while k=7 is the
  LODO-selected deployable default.
- Hypothesis B is true for LODO only: k=7, k=9 and k=11 tie at the reported
  regret. It is false for the custom LOIO sweep, where k=11 is lower than k=7.
- Hypothesis C is false for the problematic rows: the feature set is
  `metadata_no_source` throughout.
- Hypothesis D is false for the policy-comparison discrepancy: it uses
  classifier decisions, not oracle labels.
- Hypothesis E is not supported for the numbers: the tables are not stale, but
  they mixed different questions without enough caption context.
- Hypothesis F is partly true: LOIO/LODO explain k selection in the custom
  sweep, but not the LOIO 0.01238 vs 0.01167 difference, which comes from using
  fixed deployable k=7 vs best LOIO k=11.

## 3. Switch-analysis diagnosis

Source of truth:

`results/routing_context/operational_regime_simulation_ssimulacra2_quality_contract/operational_regime_switch_summary.csv`

Rows audited:

| quality_floor | neural_win_rate | classic_win_rate | neural_necessary | neural_rde_efficient | classical_sufficient | neural_too_energy_expensive | no_neural_feasible | no_classic_feasible | winner_sum | reason_sum |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 0.135417 | 0.864583 | 0.000000 | 0.135417 | 0.052083 | 0.812500 | 0.000000 | 0.000000 | 1.0 | 1.0 |
| 70 | 0.093750 | 0.906250 | 0.000000 | 0.093750 | 0.062500 | 0.833333 | 0.010417 | 0.000000 | 1.0 | 1.0 |
| 80 | 0.020833 | 0.979167 | 0.000000 | 0.020833 | 0.010417 | 0.583333 | 0.385417 | 0.000000 | 1.0 | 1.0 |

The diagnostic switch-reason columns are mutually exclusive and sum to one only
when `no_neural_feasible_rate` is included. The previous thesis table omitted
that column, so the Q=80 row appeared semantically incomplete.

Final thesis semantics:

- "Vittoria neurale/classico" are aggregate winner-family columns.
- The reason columns are a diagnostic partition.
- `no_classic_feasible_rate` is zero in the displayed rows and remains omitted
  with an explicit caption note.

## 4. Unit and caption fixes

- Table 4.5: source is `results/audio/audio_energy_rigorous_batch.csv`,
  generated by `src/benchmark/benchmark_audio_energy.py`. `j_per_s` is
  normalized by total audio duration; `cpu_j` and `gpu_j` are net batch totals.
  Caption and headers now state this.
- Table 6.4: source is `operational_regime_summary.csv`, rows for
  `policy=metadata_plus_system_full_pool`, `quality_floor=70`. Energy is
  `mean_energy` in J/img; rate is `mean_rate` in bpp. Headers now state units.
- Table 4.6: exact image/video extrema are manually propagated in the thesis;
  the caption now states that dynamics are ratios computed before rounding the
  displayed values.
- Table 6.3: source `v09_content_aware_overhead_table.csv` explicitly notes
  that metadata-only overhead is not measured in the table because it derives
  from header/known metadata and avoids pixel scanning. A table note now states
  the meaning of `--`.

## 5. LaTeX changes applied

No numeric results were changed.

Changed:

- `<thesis-root>\Capitoli\capitolo4.tex`
- `<thesis-root>\Capitoli\capitolo5.tex`
- `<thesis-root>\Capitoli\capitolo6.tex`
- `<thesis-root>\tables\routing_v09\v09_content_aware_final_table.tex`
- `<thesis-root>\tables\routing_v09\v09_content_aware_best_k_table.tex`
- `<thesis-root>\tables\routing_v09\v0910_sklearn_ablation_compact_table.tex`
- `<thesis-root>\tables\routing_v09\v09_content_aware_overhead_table_paper.tex`
- `<thesis-root>\tables\routing_regime\table_regime_summary_core.tex`
- `<thesis-root>\tables\routing_regime\table_ssimulacra2_switch_summary.tex`

Manual thesis-only fixes:

- Caption clarifications for generated tables.
- Text explaining k=7 vs k=11 provenance.
- Added `No neurale fattibile` column to Table 6.5 from the source CSV.
- Added units to Table 6.4 and hardware-energy semantics to Table 4.5.

## 6. Stale or not included

`<thesis-root>\tables\routing_regime\table_expected_quality_gate_summary.tex`
is congruent with `quality_gate_comparison_summary.csv`, but it is not included
in the current `capitolo6.tex`. It should remain excluded if the expected-quality
gate is only discussed narratively/diagnostically, or be reintroduced near the
expected-quality-gate paragraph if the thesis should expose the numbers.

## 7. Final validation

Command run:

```powershell
cd <thesis-root>
latexmk -pdf main.tex
Select-String -Path <thesis-root>\main.log -Pattern "Undefined references|Reference.*undefined|File.*not found|Overfull|Underfull|LaTeX Error|Package .* Error"
```

Result: compilation succeeded and the warning/error query returned no matches.
