import csv
from pathlib import Path


src = Path("results/routing_context/v0910_sklearn_ablation_summary.csv")
out_dir = Path("results/routing_context/paper_artifacts_v09")
out_dir.mkdir(parents=True, exist_ok=True)

out_tex = out_dir / "v0910_sklearn_ablation_compact_table.tex"
out_csv = out_dir / "v0910_sklearn_ablation_compact_table.csv"

protocol_name = {
    "leave_one_image_out": "LOIO",
    "leave_one_dataset_out": "LODO",
}

model_name = {
    "knn": "kNN",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "mlp": "MLP",
}

protocol_order = {
    "leave_one_image_out": 0,
    "leave_one_dataset_out": 1,
}


def f(x):
    return float(x) if x not in (None, "") else float("inf")


rows = list(csv.DictReader(src.open("r", encoding="utf-8")))

# Fair deployable comparison: same feature family used by the deployable router.
rows = [r for r in rows if r["feature_set"] == "metadata_no_source"]

best = {}
for r in rows:
    key = (r["evaluation_mode"], r["model_id"])
    if key not in best or f(r["mean_regret"]) < f(best[key]["mean_regret"]):
        best[key] = r

selected = list(best.values())
selected.sort(
    key=lambda r: (
        protocol_order.get(r["evaluation_mode"], 99),
        f(r["mean_regret"]),
        r["model_id"],
    )
)

with out_csv.open("w", encoding="utf-8", newline="") as g:
    fieldnames = [
        "protocol",
        "model",
        "k",
        "mean_regret",
        "relative_regret_reduction_percent",
        "oracle_match_rate_percent",
        "fallback_rate_percent",
    ]
    writer = csv.DictWriter(g, fieldnames=fieldnames)
    writer.writeheader()

    for r in selected:
        writer.writerow(
            {
                "protocol": protocol_name.get(
                    r["evaluation_mode"],
                    r["evaluation_mode"],
                ),
                "model": model_name.get(r["model_id"], r["model_id"]),
                "k": r["k"] if r["k"] else "--",
                "mean_regret": f"{float(r['mean_regret']):.5f}",
                "relative_regret_reduction_percent": (
                    f"{100.0 * float(r['relative_regret_reduction']):.1f}"
                ),
                "oracle_match_rate_percent": (
                    f"{100.0 * float(r['oracle_match_rate']):.1f}"
                ),
                "fallback_rate_percent": (
                    f"{100.0 * float(r['fallback_rate']):.1f}"
                ),
            }
        )

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"    \centering")
lines.append(r"    \small")
lines.append(
    r"    \caption{Ablazione del modello predittivo per il routing "
    r"content-aware nel dominio immagine. Tutti i modelli utilizzano feature "
    r"\texttt{metadata\_no\_source}; per kNN viene riportato il miglior "
    r"valore di \(k\) nel protocollo considerato.}"
)
lines.append(r"    \label{tab:sklearn-ablation-metadata}")
lines.append(r"    \begin{tabular}{llcccc}")
lines.append(r"        \hline")
lines.append(
    r"        Protocollo & Modello & \(k\) & Regret medio & Riduz. regret & "
    r"Match oracle \\"
)
lines.append(r"        \hline")

last_protocol = None
for r in selected:
    protocol = protocol_name.get(r["evaluation_mode"], r["evaluation_mode"])
    model = model_name.get(r["model_id"], r["model_id"])
    k = r["k"] if r["k"] else "--"
    mean_regret = f"{float(r['mean_regret']):.5f}"
    reduction = f"{100.0 * float(r['relative_regret_reduction']):.1f}\\%"
    acc = f"{100.0 * float(r['oracle_match_rate']):.1f}\\%"

    if last_protocol is not None and protocol != last_protocol:
        lines.append(r"        \hline")

    lines.append(
        f"        {protocol} & {model} & {k} & {mean_regret} & "
        f"{reduction} & {acc} \\\\"
    )
    last_protocol = protocol

lines.append(r"        \hline")
lines.append(r"    \end{tabular}")
lines.append(r"\end{table}")

out_tex.write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote: {out_csv}")
print(f"Wrote: {out_tex}")
