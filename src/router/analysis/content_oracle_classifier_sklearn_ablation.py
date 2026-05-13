import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..adaptation.content_metadata_policy import (
        build_candidate_lookup,
        resolve_global_baseline,
    )
    from .content_oracle_classifier import (
        FEATURE_SETS,
        _candidate_is_feasible,
        _encode_row,
        _feature_spec,
        _fit_categorical_values,
        _fit_numeric_stats,
        _split_label,
        _to_float,
        load_classifier_rows,
        summarize_classifier_decisions,
        write_csv,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.router.adaptation.content_metadata_policy import (
        build_candidate_lookup,
        resolve_global_baseline,
    )
    from src.router.analysis.content_oracle_classifier import (
        FEATURE_SETS,
        _candidate_is_feasible,
        _encode_row,
        _feature_spec,
        _fit_categorical_values,
        _fit_numeric_stats,
        _split_label,
        _to_float,
        load_classifier_rows,
        summarize_classifier_decisions,
        write_csv,
    )

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    SKLEARN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when sklearn is absent.
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    ConvergenceWarning = Warning
    LogisticRegression = None
    KNeighborsClassifier = None
    MLPClassifier = None
    DecisionTreeClassifier = None
    SKLEARN_IMPORT_ERROR = exc


Pair = Tuple[str, str]

SUPPORTED_MODELS = {
    "knn",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "logistic_regression",
    "mlp",
}


def _require_sklearn() -> None:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn is required for sklearn ablations. "
            "Install it with: python -m pip install scikit-learn"
        ) from SKLEARN_IMPORT_ERROR


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    values = []

    for item in _parse_csv_list(value):
        parsed = int(item)

        if parsed <= 0:
            raise ValueError("k values must be positive integers.")

        values.append(parsed)

    if not values:
        raise ValueError("At least one k value is required.")

    return values


def _maybe_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None

    return _to_float(value)


def _candidate_cost(candidate: Optional[Dict[str, Any]]) -> Optional[float]:
    if candidate is None:
        return None

    return _maybe_float(candidate.get("J_RDE"))


def _encode_fold(
    *,
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    feature_set: str,
) -> Tuple[List[List[float]], List[List[float]]]:
    numeric_features, categorical_features = _feature_spec(feature_set)
    numeric_stats = _fit_numeric_stats(train_rows, numeric_features)
    categorical_values = _fit_categorical_values(train_rows, categorical_features)

    x_train = [
        _encode_row(
            row,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_stats=numeric_stats,
            categorical_values=categorical_values,
        )
        for row in train_rows
    ]

    x_test = [
        _encode_row(
            row,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_stats=numeric_stats,
            categorical_values=categorical_values,
        )
        for row in test_rows
    ]

    return x_train, x_test


def _build_model(model_id: str, *, k: int, train_size: int):
    _require_sklearn()

    if model_id not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported sklearn model: {model_id}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_MODELS))}"
        )

    if model_id == "knn":
        return KNeighborsClassifier(n_neighbors=max(1, min(int(k), train_size)))

    if model_id == "decision_tree":
        return DecisionTreeClassifier(max_depth=4, random_state=0)

    if model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=0,
        )

    if model_id == "gradient_boosting":
        return GradientBoostingClassifier(random_state=0)

    if model_id == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=0,
        )

    if model_id == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(16,),
            alpha=1e-3,
            max_iter=1000,
            random_state=0,
        )

    raise AssertionError(f"Unhandled model id: {model_id}")


def _predict_sklearn_labels(
    *,
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    feature_set: str,
    model_id: str,
    k: int,
) -> List[str]:
    if not train_rows:
        raise ValueError("Cannot train sklearn classifier on an empty fold.")

    labels = [str(row["oracle_label"]) for row in train_rows]
    unique_labels = sorted(set(labels))

    if len(unique_labels) == 1:
        return [unique_labels[0] for _ in test_rows]

    x_train, x_test = _encode_fold(
        train_rows=train_rows,
        test_rows=test_rows,
        feature_set=feature_set,
    )

    model = _build_model(model_id, k=k, train_size=len(train_rows))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, labels)

    return [str(label) for label in model.predict(x_test)]


def _make_decision_row(
    *,
    test_row: Dict[str, Any],
    predicted_pair: Pair,
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    quality_floor: Optional[float],
    global_baseline: Pair,
    model_id: str,
    feature_set: str,
    k: Optional[int],
    evaluation_mode: str,
    fold_id: str,
) -> Dict[str, Any]:
    image_id = str(test_row["image_id"])

    predicted_candidate = candidate_lookup.get(
        (image_id, predicted_pair[0], predicted_pair[1])
    )
    predicted_feasible = _candidate_is_feasible(predicted_candidate, quality_floor)

    fallback_used = False

    if predicted_feasible:
        selected_pair = predicted_pair
        selected_candidate = predicted_candidate
    else:
        fallback_used = True
        selected_pair = global_baseline
        selected_candidate = candidate_lookup.get(
            (image_id, global_baseline[0], global_baseline[1])
        )

    selected_feasible = _candidate_is_feasible(selected_candidate, quality_floor)

    oracle_pair = _split_label(str(test_row["oracle_label"]))
    oracle_cost = _maybe_float(test_row.get("oracle_cost"))
    selected_cost = _candidate_cost(selected_candidate)

    global_candidate = candidate_lookup.get(
        (image_id, global_baseline[0], global_baseline[1])
    )
    global_cost = _candidate_cost(global_candidate)

    regret = (
        selected_cost - oracle_cost
        if selected_cost is not None and oracle_cost is not None
        else None
    )
    global_regret = (
        global_cost - oracle_cost
        if global_cost is not None and oracle_cost is not None
        else None
    )

    return {
        "image_id": image_id,
        "dataset": test_row.get("dataset"),
        "image": test_row.get("image"),
        "model_id": model_id,
        "feature_set": feature_set,
        "k": k if k is not None else "",
        "evaluation_mode": evaluation_mode,
        "fold_id": fold_id,
        "predicted_codec": predicted_pair[0],
        "predicted_config": predicted_pair[1],
        "predicted_feasible": predicted_feasible,
        "selected_codec": selected_pair[0],
        "selected_config": selected_pair[1],
        "selected_feasible": selected_feasible,
        "selected_cost": selected_cost,
        "fallback_used": fallback_used,
        "fallback_codec": global_baseline[0],
        "fallback_config": global_baseline[1],
        "oracle_codec": oracle_pair[0],
        "oracle_config": oracle_pair[1],
        "oracle_cost": oracle_cost,
        "global_cost": global_cost,
        "correct_oracle_match": selected_pair == oracle_pair,
        "regret": regret,
        "global_regret": global_regret,
        "regret_reduction_vs_global": (
            global_regret - regret
            if global_regret is not None and regret is not None
            else None
        ),
    }


def _summary_to_flat_row(
    summary: List[Dict[str, Any]],
    *,
    model_id: str,
    feature_set: str,
    k: Optional[int],
    evaluation_mode: str,
) -> Dict[str, Any]:
    lookup = {(row["section"], row["key"]): row["value"] for row in summary}

    method_id = f"sklearn_{model_id}_{feature_set}_{evaluation_mode}"

    if model_id == "knn":
        method_id += f"_k{k}"

    return {
        "method_id": method_id,
        "model_id": model_id,
        "feature_set": feature_set,
        "k": k if model_id == "knn" else "",
        "evaluation_mode": evaluation_mode,
        "num_images": lookup.get(("summary", "num_images")),
        "quality_floor": lookup.get(("summary", "quality_floor")),
        "global_baseline_codec": lookup.get(("summary", "global_baseline_codec")),
        "global_baseline_config": lookup.get(("summary", "global_baseline_config")),
        "oracle_match_rate": lookup.get(("accuracy", "oracle_match_rate")),
        "fallback_rate": lookup.get(("fallback", "fallback_rate")),
        "infeasible_rate": lookup.get(("feasibility", "infeasible_rate")),
        "mean_regret": lookup.get(("regret", "mean")),
        "median_regret": lookup.get(("regret", "median")),
        "p90_regret": lookup.get(("regret", "p90")),
        "max_regret": lookup.get(("regret", "max")),
        "global_mean_regret": lookup.get(("global_regret", "mean")),
        "mean_regret_reduction": lookup.get(("regret_reduction", "mean")),
        "relative_regret_reduction": lookup.get(("regret_reduction", "relative_mean")),
    }


def evaluate_sklearn_classifier(
    *,
    rows: List[Dict[str, Any]],
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    feature_set: str,
    model_id: str,
    k: Optional[int],
    quality_floor: Optional[float],
    global_baseline: Pair,
    evaluation_mode: str,
) -> Dict[str, Any]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}. "
            f"Expected one of: {', '.join(sorted(FEATURE_SETS))}"
        )

    if evaluation_mode not in {"leave_one_image_out", "leave_one_dataset_out"}:
        raise ValueError(
            "evaluation_mode must be 'leave_one_image_out' or 'leave_one_dataset_out'."
        )

    effective_k = int(k) if k is not None else 1
    decisions: List[Dict[str, Any]] = []

    if evaluation_mode == "leave_one_image_out":
        for index, test_row in enumerate(rows):
            train_rows = [row for i, row in enumerate(rows) if i != index]

            predicted_label = _predict_sklearn_labels(
                train_rows=train_rows,
                test_rows=[test_row],
                feature_set=feature_set,
                model_id=model_id,
                k=effective_k,
            )[0]

            decisions.append(
                _make_decision_row(
                    test_row=test_row,
                    predicted_pair=_split_label(predicted_label),
                    candidate_lookup=candidate_lookup,
                    quality_floor=quality_floor,
                    global_baseline=global_baseline,
                    model_id=model_id,
                    feature_set=feature_set,
                    k=k,
                    evaluation_mode=evaluation_mode,
                    fold_id=str(test_row.get("image_id")),
                )
            )

    else:
        datasets = sorted({str(row.get("dataset")) for row in rows})

        for dataset in datasets:
            train_rows = [row for row in rows if str(row.get("dataset")) != dataset]
            test_rows = [row for row in rows if str(row.get("dataset")) == dataset]

            if not train_rows or not test_rows:
                continue

            predicted_labels = _predict_sklearn_labels(
                train_rows=train_rows,
                test_rows=test_rows,
                feature_set=feature_set,
                model_id=model_id,
                k=effective_k,
            )

            for test_row, predicted_label in zip(test_rows, predicted_labels):
                decisions.append(
                    _make_decision_row(
                        test_row=test_row,
                        predicted_pair=_split_label(predicted_label),
                        candidate_lookup=candidate_lookup,
                        quality_floor=quality_floor,
                        global_baseline=global_baseline,
                        model_id=model_id,
                        feature_set=feature_set,
                        k=k,
                        evaluation_mode=evaluation_mode,
                        fold_id=dataset,
                    )
                )

    summary = summarize_classifier_decisions(
        decisions,
        feature_set=feature_set,
        k=effective_k,
        quality_floor=quality_floor,
        global_baseline=global_baseline,
    )

    return {
        "decisions": decisions,
        "summary": summary,
        "summary_flat": _summary_to_flat_row(
            summary,
            model_id=model_id,
            feature_set=feature_set,
            k=k,
            evaluation_mode=evaluation_mode,
        ),
    }


def run_sklearn_ablation(
    *,
    rows: List[Dict[str, Any]],
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    feature_sets: List[str],
    models: List[str],
    k_values: List[int],
    evaluation_modes: List[str],
    quality_floor: Optional[float],
    global_baseline: Pair,
) -> Dict[str, List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    for evaluation_mode in evaluation_modes:
        for feature_set in feature_sets:
            for model_id in models:
                if model_id not in SUPPORTED_MODELS:
                    raise ValueError(
                        f"Unsupported model: {model_id}. "
                        f"Expected one of: {', '.join(sorted(SUPPORTED_MODELS))}"
                    )

                model_k_values = k_values if model_id == "knn" else [None]

                for k in model_k_values:
                    evaluation = evaluate_sklearn_classifier(
                        rows=rows,
                        candidate_lookup=candidate_lookup,
                        feature_set=feature_set,
                        model_id=model_id,
                        k=k,
                        quality_floor=quality_floor,
                        global_baseline=global_baseline,
                        evaluation_mode=evaluation_mode,
                    )

                    summary_rows.append(evaluation["summary_flat"])
                    decision_rows.extend(evaluation["decisions"])

    return {
        "summary": summary_rows,
        "decisions": decision_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run optional scikit-learn classifier ablations for content-aware "
            "R-D-E routing."
        )
    )

    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--metadata-oracle-csv", required=True)
    parser.add_argument("--pixel-features-csv", required=True)

    parser.add_argument(
        "--summary-out",
        default="results/routing_context/v0910_sklearn_ablation_summary.csv",
    )
    parser.add_argument(
        "--decisions-out",
        default="results/routing_context/v0910_sklearn_ablation_decisions.csv",
    )

    parser.add_argument(
        "--feature-sets",
        default="metadata_no_source",
        help="Comma-separated feature sets.",
    )
    parser.add_argument(
        "--models",
        default="knn,decision_tree,random_forest,gradient_boosting,logistic_regression,mlp",
        help="Comma-separated sklearn models.",
    )
    parser.add_argument(
        "--k-values",
        default="3,5,7,9,11",
        help="Comma-separated k values used only by kNN.",
    )
    parser.add_argument(
        "--evaluation-modes",
        default="leave_one_image_out,leave_one_dataset_out",
        help="Comma-separated evaluation modes.",
    )

    parser.add_argument("--quality-floor", type=float, default=80.0)
    parser.add_argument("--global-baseline-codec", default=None)
    parser.add_argument("--global-baseline-config", default=None)

    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--image-col", default="image")
    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="param")
    parser.add_argument("--rate-col", default="bpp")
    parser.add_argument("--quality-col", default="ssimulacra2")
    parser.add_argument("--energy-col", default="energy_per_image_j")
    parser.add_argument("--time-col", default="time_ms")

    parser.add_argument("--available-codecs", default="JPEG,JXL,HEVC")
    parser.add_argument("--exclude-codecs", default=None)

    parser.add_argument("--wR", type=float, default=0.2)
    parser.add_argument("--wE", type=float, default=0.2)
    parser.add_argument("--wD", type=float, default=0.6)

    args = parser.parse_args()

    _require_sklearn()

    feature_sets = _parse_csv_list(args.feature_sets)
    models = _parse_csv_list(args.models)
    k_values = _parse_int_list(args.k_values)
    evaluation_modes = _parse_csv_list(args.evaluation_modes)

    rows = load_classifier_rows(
        metadata_oracle_csv=args.metadata_oracle_csv,
        pixel_features_csv=args.pixel_features_csv,
    )

    candidate_lookup = build_candidate_lookup(
        args.benchmark_csv,
        dataset_col=args.dataset_col,
        image_col=args.image_col,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        time_col=args.time_col,
        available_codecs=args.available_codecs,
        exclude_codecs=args.exclude_codecs,
        w_r=args.wR,
        w_e=args.wE,
        w_d=args.wD,
    )

    global_baseline = resolve_global_baseline(
        rows,
        global_baseline_codec=args.global_baseline_codec,
        global_baseline_config=args.global_baseline_config,
    )

    ablation = run_sklearn_ablation(
        rows=rows,
        candidate_lookup=candidate_lookup,
        feature_sets=feature_sets,
        models=models,
        k_values=k_values,
        evaluation_modes=evaluation_modes,
        quality_floor=args.quality_floor,
        global_baseline=global_baseline,
    )

    write_csv(args.summary_out, ablation["summary"])
    write_csv(args.decisions_out, ablation["decisions"])

    ranked = sorted(
        ablation["summary"],
        key=lambda row: float(row["mean_regret"])
        if row.get("mean_regret") not in {None, ""}
        else float("inf"),
    )

    print("\n=== R-D-E sklearn classifier ablation ===")
    print(f"Rows:              {len(rows)}")
    print(f"Feature sets:      {', '.join(feature_sets)}")
    print(f"Models:            {', '.join(models)}")
    print(f"k values:          {', '.join(str(k) for k in k_values)}")
    print(f"Evaluation modes:  {', '.join(evaluation_modes)}")
    print(f"Global baseline:   {global_baseline[0]} {global_baseline[1]}")
    print(f"Summary CSV:       {args.summary_out}")
    print(f"Decisions CSV:     {args.decisions_out}")

    print("\nTop configurations by mean regret:")
    for row in ranked[:10]:
        print(
            "  "
            f"{row['evaluation_mode']}, "
            f"{row['model_id']}, "
            f"{row['feature_set']}, "
            f"k={row['k'] or '-'}: "
            f"mean_regret={row['mean_regret']}, "
            f"relative_reduction={row['relative_regret_reduction']}"
        )


if __name__ == "__main__":
    main()
