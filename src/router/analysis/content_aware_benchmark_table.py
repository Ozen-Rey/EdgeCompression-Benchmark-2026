import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).replace(",", ".")))
    except Exception:
        return default


def _read_key_value_summary(path: str) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"section", "key", "value"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Summary CSV {path} missing columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            lookup[(str(row["section"]), str(row["key"]))] = str(row["value"])

    if not lookup:
        raise ValueError(f"No summary rows loaded from {path}")

    return lookup


def _read_flat_csv(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    return rows


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or abs(den) <= 1e-12:
        return None
    return num / den


def _method_row(
    *,
    method_id: str,
    method_family: str,
    evaluation_protocol: str,
    deployment_setting: str,
    feature_set: str = "",
    k: str = "",
    requires_source_label: bool,
    uses_pixel_features: bool,
    uses_oracle: bool,
    selected_policy: str,
    accuracy: Optional[float],
    mean_regret: Optional[float],
    median_regret: Optional[float],
    p90_regret: Optional[float],
    max_regret: Optional[float],
    global_mean_regret: Optional[float],
    mean_regret_reduction: Optional[float],
    relative_regret_reduction: Optional[float],
    fallback_rate: Optional[float],
    infeasible_rate: Optional[float],
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "method_id": method_id,
        "method_family": method_family,
        "evaluation_protocol": evaluation_protocol,
        "deployment_setting": deployment_setting,
        "feature_set": feature_set,
        "k": k,
        "requires_source_label": requires_source_label,
        "uses_pixel_features": uses_pixel_features,
        "uses_oracle": uses_oracle,
        "selected_policy": selected_policy,
        "accuracy": accuracy,
        "mean_regret": mean_regret,
        "median_regret": median_regret,
        "p90_regret": p90_regret,
        "max_regret": max_regret,
        "global_mean_regret": global_mean_regret,
        "mean_regret_reduction": mean_regret_reduction,
        "relative_regret_reduction": relative_regret_reduction,
        "fallback_rate": fallback_rate,
        "infeasible_rate": infeasible_rate,
        "notes": notes,
    }


def _best_sweep_row(
    rows: List[Dict[str, str]],
    *,
    evaluation_mode: str,
    require_no_source: Optional[bool] = None,
) -> Dict[str, str]:
    candidates = [row for row in rows if row.get("evaluation_mode") == evaluation_mode]

    if require_no_source is True:
        candidates = [
            row
            for row in candidates
            if str(row.get("feature_set", "")).endswith("_no_source")
        ]
    elif require_no_source is False:
        candidates = [
            row
            for row in candidates
            if not str(row.get("feature_set", "")).endswith("_no_source")
        ]

    if not candidates:
        raise ValueError(
            f"No sweep rows found for evaluation_mode={evaluation_mode}, "
            f"require_no_source={require_no_source}"
        )

    def sort_key(row: Dict[str, str]):
        mean_regret = _to_float(row.get("mean_regret"), math.inf)
        fallback_rate = _to_float(row.get("fallback_rate"), math.inf)
        accuracy = _to_float(row.get("accuracy"), -math.inf)

        return (
            mean_regret if mean_regret is not None else math.inf,
            fallback_rate if fallback_rate is not None else math.inf,
            -(accuracy if accuracy is not None else -math.inf),
            str(row.get("feature_set", "")),
            _to_int(row.get("k"), 10**9) or 10**9,
        )

    return sorted(candidates, key=sort_key)[0]


def _sweep_to_method_row(
    row: Dict[str, str], *, method_id: Optional[str] = None
) -> Dict[str, Any]:
    feature_set = str(row.get("feature_set", ""))
    k = str(row.get("k", ""))

    return _method_row(
        method_id=method_id
        or f"knn_{feature_set}_k{k}_{row.get('evaluation_mode')}",
        method_family="knn_oracle_classifier",
        evaluation_protocol=str(row.get("evaluation_mode", "")),
        deployment_setting=(
            "source_agnostic"
            if feature_set.endswith("_no_source")
            else "source_available"
        ),
        feature_set=feature_set,
        k=k,
        requires_source_label=not feature_set.endswith("_no_source"),
        uses_pixel_features=(
            feature_set.startswith("pixel_") or feature_set.startswith("all_")
        ),
        uses_oracle=False,
        selected_policy=f"kNN oracle classifier, feature_set={feature_set}, k={k}",
        accuracy=_to_float(row.get("accuracy")),
        mean_regret=_to_float(row.get("mean_regret")),
        median_regret=_to_float(row.get("median_regret")),
        p90_regret=_to_float(row.get("p90_regret")),
        max_regret=_to_float(row.get("max_regret")),
        global_mean_regret=_to_float(row.get("global_mean_regret")),
        mean_regret_reduction=_to_float(row.get("mean_regret_reduction")),
        relative_regret_reduction=_to_float(row.get("relative_regret_reduction")),
        fallback_rate=_to_float(row.get("fallback_rate")),
        infeasible_rate=_to_float(row.get("infeasible_rate")),
        notes="Selected by minimum mean regret within its evaluation protocol.",
    )


def build_content_aware_benchmark_tables(
    *,
    oracle_summary_csv: str,
    dataset_policy_summary_csv: str,
    classifier_sweep_summary_csv: str,
) -> Dict[str, List[Dict[str, Any]]]:
    oracle = _read_key_value_summary(oracle_summary_csv)
    dataset_policy = _read_key_value_summary(dataset_policy_summary_csv)
    sweep_rows = _read_flat_csv(classifier_sweep_summary_csv)

    num_images = _to_float(oracle.get(("summary", "num_images_analyzed")))
    robust_mean_regret = _to_float(oracle.get(("regret", "mean")))
    robust_median_regret = _to_float(oracle.get(("regret", "median")))
    robust_p90_regret = _to_float(oracle.get(("regret", "p90")))
    robust_max_regret = _to_float(oracle.get(("regret", "max")))

    global_codec = oracle.get(("global_best", "codec"), "")
    global_config = oracle.get(("global_best", "config"), "")

    matches_global = _to_float(
        oracle.get(("oracle_diversity", "oracle_matches_global_count"))
    )
    robust_accuracy = _safe_div(matches_global, num_images)

    robust_row = _method_row(
        method_id="robust_global_baseline",
        method_family="global_rde_baseline",
        evaluation_protocol="global_coverage_oracle_analysis",
        deployment_setting="source_agnostic",
        feature_set="none",
        k="",
        requires_source_label=False,
        uses_pixel_features=False,
        uses_oracle=False,
        selected_policy=f"{global_codec} {global_config}",
        accuracy=robust_accuracy,
        mean_regret=robust_mean_regret,
        median_regret=robust_median_regret,
        p90_regret=robust_p90_regret,
        max_regret=robust_max_regret,
        global_mean_regret=robust_mean_regret,
        mean_regret_reduction=0.0,
        relative_regret_reduction=0.0,
        fallback_rate=0.0,
        infeasible_rate=0.0,
        notes="Single globally feasible baseline selected under full quality coverage.",
    )

    oracle_row = _method_row(
        method_id="per_image_oracle",
        method_family="oracle_lower_bound",
        evaluation_protocol="oracle",
        deployment_setting="not_deployable",
        feature_set="oracle",
        k="",
        requires_source_label=False,
        uses_pixel_features=False,
        uses_oracle=True,
        selected_policy="argmin J_RDE per image",
        accuracy=1.0,
        mean_regret=0.0,
        median_regret=0.0,
        p90_regret=0.0,
        max_regret=0.0,
        global_mean_regret=robust_mean_regret,
        mean_regret_reduction=robust_mean_regret,
        relative_regret_reduction=1.0,
        fallback_rate=0.0,
        infeasible_rate=0.0,
        notes="Non-deployable lower bound used only for regret computation.",
    )

    dataset_policy_row = _method_row(
        method_id="source_aware_dataset_majority_policy",
        method_family="source_aware_majority_policy",
        evaluation_protocol=dataset_policy.get(("summary", "evaluation_mode"), ""),
        deployment_setting="batch_known_source",
        feature_set="dataset",
        k="",
        requires_source_label=True,
        uses_pixel_features=False,
        uses_oracle=False,
        selected_policy="majority oracle per dataset/source with safe fallback",
        accuracy=_to_float(dataset_policy.get(("accuracy", "oracle_match_rate"))),
        mean_regret=_to_float(dataset_policy.get(("regret", "mean"))),
        median_regret=_to_float(dataset_policy.get(("regret", "median"))),
        p90_regret=_to_float(dataset_policy.get(("regret", "p90"))),
        max_regret=_to_float(dataset_policy.get(("regret", "max"))),
        global_mean_regret=_to_float(dataset_policy.get(("global_regret", "mean"))),
        mean_regret_reduction=_to_float(
            dataset_policy.get(("regret_reduction", "mean"))
        ),
        relative_regret_reduction=_to_float(
            dataset_policy.get(("regret_reduction", "relative_mean"))
        ),
        fallback_rate=_to_float(dataset_policy.get(("fallback", "fallback_rate"))),
        infeasible_rate=_to_float(
            dataset_policy.get(("feasibility", "infeasible_rate"))
        ),
        notes="Deployable when the user provides a homogeneous content source label.",
    )

    best_loio = _best_sweep_row(
        sweep_rows,
        evaluation_mode="leave_one_image_out",
        require_no_source=True,
    )

    best_lodo = _best_sweep_row(
        sweep_rows,
        evaluation_mode="leave_one_dataset_out",
        require_no_source=True,
    )

    best_loio_row = _sweep_to_method_row(
        best_loio,
        method_id="best_source_agnostic_knn_leave_one_image_out",
    )

    best_lodo_row = _sweep_to_method_row(
        best_lodo,
        method_id="best_source_agnostic_knn_leave_one_dataset_out",
    )

    paper_table = [
        robust_row,
        dataset_policy_row,
        best_loio_row,
        best_lodo_row,
        oracle_row,
    ]

    all_methods = [robust_row, dataset_policy_row, oracle_row]

    for row in sweep_rows:
        all_methods.append(_sweep_to_method_row(row))

    all_methods = sorted(
        all_methods,
        key=lambda row: (
            float(row["mean_regret"]) if row["mean_regret"] is not None else math.inf,
            str(row["method_id"]),
        ),
    )

    return {
        "paper_table": paper_table,
        "all_methods": all_methods,
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper-grade benchmark tables for content-aware R-D-E routing."
    )

    parser.add_argument(
        "--oracle-summary",
        default="results/routing_context/v09_content_oracle_summary.csv",
    )
    parser.add_argument(
        "--dataset-policy-summary",
        default="results/routing_context/v09_metadata_policy_dataset_summary.csv",
    )
    parser.add_argument(
        "--classifier-sweep-summary",
        default="results/routing_context/v09_oracle_classifier_sweep_summary.csv",
    )
    parser.add_argument(
        "--paper-table-out",
        default="results/routing_context/v09_content_aware_benchmark_table.csv",
    )
    parser.add_argument(
        "--all-methods-out",
        default="results/routing_context/v09_content_aware_benchmark_all_methods.csv",
    )

    args = parser.parse_args()

    tables = build_content_aware_benchmark_tables(
        oracle_summary_csv=args.oracle_summary,
        dataset_policy_summary_csv=args.dataset_policy_summary,
        classifier_sweep_summary_csv=args.classifier_sweep_summary,
    )

    write_csv(args.paper_table_out, tables["paper_table"])
    write_csv(args.all_methods_out, tables["all_methods"])

    print("\n=== R-D-E Content-Aware Benchmark Table ===")
    print(f"Paper table:     {args.paper_table_out}")
    print(f"All methods:     {args.all_methods_out}")
    print()
    print("Paper table rows:")

    for row in tables["paper_table"]:
        print(
            f"  {row['method_id']}: "
            f"mean_regret={row['mean_regret']}, "
            f"relative_reduction={row['relative_regret_reduction']}, "
            f"accuracy={row['accuracy']}, "
            f"fallback={row['fallback_rate']}"
        )


if __name__ == "__main__":
    main()
