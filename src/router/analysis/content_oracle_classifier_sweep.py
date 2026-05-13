import argparse
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
        _label_from_sorted_distances,
        _predict_knn_label,
        _squared_distance,
        _split_label,
        _to_float,
        load_classifier_rows,
        summarize_classifier_decisions,
        write_csv,
    )
except ImportError:
    from content_metadata_policy import build_candidate_lookup, resolve_global_baseline
    from content_oracle_classifier import (
        FEATURE_SETS,
        _candidate_is_feasible,
        _encode_row,
        _feature_spec,
        _fit_categorical_values,
        _fit_numeric_stats,
        _label_from_sorted_distances,
        _predict_knn_label,
        _squared_distance,
        _split_label,
        _to_float,
        load_classifier_rows,
        summarize_classifier_decisions,
        write_csv,
    )


Pair = Tuple[str, str]
NeighborCache = Dict[str, List[Tuple[float, str]]]
FoldNeighborCache = Dict[Tuple[str, str], List[Tuple[float, str]]]


def _build_loio_neighbor_cache(
    *,
    rows: List[Dict[str, Any]],
    feature_set: str,
) -> NeighborCache:
    numeric_features, categorical_features = _feature_spec(feature_set)
    cache: NeighborCache = {}

    for index, test_row in enumerate(rows):
        train_rows = [row for i, row in enumerate(rows) if i != index]

        numeric_stats = _fit_numeric_stats(train_rows, numeric_features)
        categorical_values = _fit_categorical_values(train_rows, categorical_features)

        test_vec = _encode_row(
            test_row,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_stats=numeric_stats,
            categorical_values=categorical_values,
        )

        distances: List[Tuple[float, str]] = []

        for train_row in train_rows:
            train_vec = _encode_row(
                train_row,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_stats=numeric_stats,
                categorical_values=categorical_values,
            )

            distances.append(
                (
                    _squared_distance(test_vec, train_vec),
                    str(train_row["oracle_label"]),
                )
            )

        distances.sort(key=lambda item: item[0])
        cache[str(test_row["image_id"])] = distances

    return cache


def _build_lodo_neighbor_cache(
    *,
    rows: List[Dict[str, Any]],
    feature_set: str,
) -> FoldNeighborCache:
    numeric_features, categorical_features = _feature_spec(feature_set)
    datasets = sorted({str(row.get("dataset")) for row in rows})
    cache: FoldNeighborCache = {}

    for dataset in datasets:
        train_rows = [row for row in rows if str(row.get("dataset")) != dataset]
        test_rows = [row for row in rows if str(row.get("dataset")) == dataset]

        if not train_rows:
            continue

        numeric_stats = _fit_numeric_stats(train_rows, numeric_features)
        categorical_values = _fit_categorical_values(train_rows, categorical_features)

        encoded_train_rows = []

        for train_row in train_rows:
            encoded_train_rows.append(
                (
                    _encode_row(
                        train_row,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        numeric_stats=numeric_stats,
                        categorical_values=categorical_values,
                    ),
                    str(train_row["oracle_label"]),
                )
            )

        for test_row in test_rows:
            test_vec = _encode_row(
                test_row,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_stats=numeric_stats,
                categorical_values=categorical_values,
            )

            distances = [
                (_squared_distance(test_vec, train_vec), label)
                for train_vec, label in encoded_train_rows
            ]

            distances.sort(key=lambda item: item[0])
            cache[(dataset, str(test_row["image_id"]))] = distances

    return cache


def _make_decision_row(
    *,
    test_row: Dict[str, Any],
    predicted_pair: Pair,
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    quality_floor: float,
    global_baseline: Pair,
    feature_set: str,
    k: int,
    evaluation_mode: str,
    fold_id: str,
) -> Dict[str, Any]:
    image_id = str(test_row["image_id"])

    predicted_candidate = candidate_lookup.get(
        (image_id, predicted_pair[0], predicted_pair[1])
    )

    predicted_feasible = _candidate_is_feasible(
        predicted_candidate,
        quality_floor,
    )

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

    selected_feasible = _candidate_is_feasible(
        selected_candidate,
        quality_floor,
    )

    oracle_pair = _split_label(str(test_row["oracle_label"]))
    oracle_cost = _to_float(test_row.get("oracle_cost"))

    selected_cost = (
        _to_float(selected_candidate.get("J_RDE"))
        if selected_candidate is not None
        else None
    )

    global_regret = _to_float(test_row.get("regret"))

    regret = selected_cost - oracle_cost if selected_cost is not None else None

    return {
        "dataset": test_row.get("dataset"),
        "image": test_row.get("image"),
        "image_id": image_id,
        "feature_set": feature_set,
        "k": k,
        "evaluation_mode": evaluation_mode,
        "fold_id": fold_id,
        "oracle_codec": oracle_pair[0],
        "oracle_config": oracle_pair[1],
        "oracle_cost": oracle_cost,
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
        "correct_oracle_match": selected_pair == oracle_pair,
        "regret": regret,
        "global_regret": global_regret,
        "regret_reduction_vs_global": (
            global_regret - regret
            if global_regret is not None and regret is not None
            else None
        ),
    }


def evaluate_leave_one_image_out(
    *,
    rows: List[Dict[str, Any]],
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    feature_set: str,
    k: int,
    quality_floor: float,
    global_baseline: Pair,
    neighbor_cache: Optional[NeighborCache] = None,
) -> List[Dict[str, Any]]:
    decisions = []

    for index, test_row in enumerate(rows):
        if neighbor_cache is None:
            train_rows = [row for i, row in enumerate(rows) if i != index]

            predicted_label = _predict_knn_label(
                train_rows=train_rows,
                test_row=test_row,
                feature_set=feature_set,
                k=k,
            )
        else:
            predicted_label = _label_from_sorted_distances(
                neighbor_cache[str(test_row["image_id"])],
                k,
            )

        predicted_pair = _split_label(predicted_label)

        decisions.append(
            _make_decision_row(
                test_row=test_row,
                predicted_pair=predicted_pair,
                candidate_lookup=candidate_lookup,
                quality_floor=quality_floor,
                global_baseline=global_baseline,
                feature_set=feature_set,
                k=k,
                evaluation_mode="leave_one_image_out",
                fold_id=str(test_row.get("image_id")),
            )
        )

    return decisions


def evaluate_leave_one_dataset_out(
    *,
    rows: List[Dict[str, Any]],
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    feature_set: str,
    k: int,
    quality_floor: float,
    global_baseline: Pair,
    neighbor_cache: Optional[FoldNeighborCache] = None,
) -> List[Dict[str, Any]]:
    decisions = []

    datasets = sorted({str(row.get("dataset")) for row in rows})

    for dataset in datasets:
        train_rows = [row for row in rows if str(row.get("dataset")) != dataset]

        test_rows = [row for row in rows if str(row.get("dataset")) == dataset]

        if not train_rows:
            continue

        for test_row in test_rows:
            if neighbor_cache is None:
                predicted_label = _predict_knn_label(
                    train_rows=train_rows,
                    test_row=test_row,
                    feature_set=feature_set,
                    k=k,
                )
            else:
                predicted_label = _label_from_sorted_distances(
                    neighbor_cache[(dataset, str(test_row["image_id"]))],
                    k,
                )

            predicted_pair = _split_label(predicted_label)

            decisions.append(
                _make_decision_row(
                    test_row=test_row,
                    predicted_pair=predicted_pair,
                    candidate_lookup=candidate_lookup,
                    quality_floor=quality_floor,
                    global_baseline=global_baseline,
                    feature_set=feature_set,
                    k=k,
                    evaluation_mode="leave_one_dataset_out",
                    fold_id=dataset,
                )
            )

    return decisions


def _summary_to_flat_row(summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    lookup = {(row["section"], row["key"]): row["value"] for row in summary}

    return {
        "evaluation_mode": lookup.get(("summary", "evaluation_mode")),
        "feature_set": lookup.get(("summary", "feature_set")),
        "k": lookup.get(("summary", "k")),
        "num_images": lookup.get(("summary", "num_images")),
        "accuracy": lookup.get(("accuracy", "oracle_match_rate")),
        "fallback_rate": lookup.get(("fallback", "fallback_rate")),
        "infeasible_rate": lookup.get(("feasibility", "infeasible_rate")),
        "mean_regret": lookup.get(("regret", "mean")),
        "median_regret": lookup.get(("regret", "median")),
        "p90_regret": lookup.get(("regret", "p90")),
        "max_regret": lookup.get(("regret", "max")),
        "global_mean_regret": lookup.get(("global_regret", "mean")),
        "mean_regret_reduction": lookup.get(("regret_reduction", "mean")),
        "relative_regret_reduction": lookup.get(
            ("regret_reduction", "relative_mean")
        ),
    }


def _inject_eval_mode(
    summary: List[Dict[str, Any]], evaluation_mode: str
) -> List[Dict[str, Any]]:
    out = []
    inserted = False

    for row in summary:
        out.append(dict(row))

        if row["section"] == "summary" and row["key"] == "k" and not inserted:
            out.append(
                {
                    "section": "summary",
                    "key": "evaluation_mode",
                    "value": evaluation_mode,
                }
            )
            inserted = True

    if not inserted:
        out.append(
            {
                "section": "summary",
                "key": "evaluation_mode",
                "value": evaluation_mode,
            }
        )

    return out


def run_sweep(
    *,
    rows: List[Dict[str, Any]],
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    feature_sets: List[str],
    k_values: List[int],
    evaluation_modes: List[str],
    quality_floor: float,
    global_baseline: Pair,
) -> Dict[str, List[Dict[str, Any]]]:
    flat_summary_rows = []
    all_decisions = []
    neighbor_caches: Dict[Tuple[str, str], Any] = {}

    for feature_set in feature_sets:
        if "leave_one_image_out" in evaluation_modes:
            neighbor_caches[("leave_one_image_out", feature_set)] = (
                _build_loio_neighbor_cache(
                    rows=rows,
                    feature_set=feature_set,
                )
            )

        if "leave_one_dataset_out" in evaluation_modes:
            neighbor_caches[("leave_one_dataset_out", feature_set)] = (
                _build_lodo_neighbor_cache(
                    rows=rows,
                    feature_set=feature_set,
                )
            )

    for evaluation_mode in evaluation_modes:
        for feature_set in feature_sets:
            for k in k_values:
                if evaluation_mode == "leave_one_image_out":
                    decisions = evaluate_leave_one_image_out(
                        rows=rows,
                        candidate_lookup=candidate_lookup,
                        feature_set=feature_set,
                        k=k,
                        quality_floor=quality_floor,
                        global_baseline=global_baseline,
                        neighbor_cache=neighbor_caches.get(
                            ("leave_one_image_out", feature_set)
                        ),
                    )
                elif evaluation_mode == "leave_one_dataset_out":
                    decisions = evaluate_leave_one_dataset_out(
                        rows=rows,
                        candidate_lookup=candidate_lookup,
                        feature_set=feature_set,
                        k=k,
                        quality_floor=quality_floor,
                        global_baseline=global_baseline,
                        neighbor_cache=neighbor_caches.get(
                            ("leave_one_dataset_out", feature_set)
                        ),
                    )
                else:
                    raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")

                summary = summarize_classifier_decisions(
                    decisions,
                    feature_set=feature_set,
                    k=k,
                    quality_floor=quality_floor,
                    global_baseline=global_baseline,
                )

                summary = _inject_eval_mode(summary, evaluation_mode)
                flat_summary_rows.append(_summary_to_flat_row(summary))
                all_decisions.extend(decisions)

    return {
        "summary": flat_summary_rows,
        "decisions": all_decisions,
    }


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item) for item in _parse_csv_list(value)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep kNN oracle classifier feature sets and evaluation modes."
    )

    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--metadata-oracle-csv", required=True)
    parser.add_argument("--pixel-features-csv", required=True)

    parser.add_argument(
        "--feature-sets",
        default="metadata_no_source,pixel_no_source,all_no_source,all_with_source",
    )

    parser.add_argument(
        "--k-values",
        default="1,3,5,7,9,11",
    )

    parser.add_argument(
        "--evaluation-modes",
        default="leave_one_image_out,leave_one_dataset_out",
    )

    parser.add_argument("--quality-floor", type=float, default=80.0)

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

    parser.add_argument(
        "--global-baseline-codec",
        default=None,
        help="Optional explicit robust global baseline codec. Must be used with --global-baseline-config.",
    )
    parser.add_argument(
        "--global-baseline-config",
        default=None,
        help="Optional explicit robust global baseline config. Must be used with --global-baseline-codec.",
    )

    parser.add_argument(
        "--summary-out",
        default="results/routing_context/v09_oracle_classifier_sweep_summary.csv",
    )

    parser.add_argument(
        "--decisions-out",
        default="results/routing_context/v09_oracle_classifier_sweep_decisions.csv",
    )

    args = parser.parse_args()

    feature_sets = _parse_csv_list(args.feature_sets)
    k_values = _parse_int_list(args.k_values)
    evaluation_modes = _parse_csv_list(args.evaluation_modes)

    for feature_set in feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature set: {feature_set}. "
                f"Expected one of: {', '.join(sorted(FEATURE_SETS))}"
            )

    rows = load_classifier_rows(
        metadata_oracle_csv=args.metadata_oracle_csv,
        pixel_features_csv=args.pixel_features_csv,
    )

    candidate_lookup = build_candidate_lookup(
        benchmark_csv=args.benchmark_csv,
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

    sweep = run_sweep(
        rows=rows,
        candidate_lookup=candidate_lookup,
        feature_sets=feature_sets,
        k_values=k_values,
        evaluation_modes=evaluation_modes,
        quality_floor=args.quality_floor,
        global_baseline=global_baseline,
    )

    write_csv(args.summary_out, sweep["summary"])
    write_csv(args.decisions_out, sweep["decisions"])

    print("\n=== R-D-E Oracle Classifier Sweep ===")
    print(f"Rows:              {len(rows)}")
    print(f"Feature sets:      {', '.join(feature_sets)}")
    print(f"k values:          {', '.join(str(k) for k in k_values)}")
    print(f"Evaluation modes:  {', '.join(evaluation_modes)}")
    print(f"Global baseline:   {global_baseline[0]} {global_baseline[1]}")
    print(f"Summary CSV:       {args.summary_out}")
    print(f"Decisions CSV:     {args.decisions_out}")

    ranked = sorted(
        sweep["summary"],
        key=lambda row: float(row["mean_regret"]),
    )

    print("\nTop configurations by mean regret:")
    for row in ranked[:10]:
        print(
            f"  {row['evaluation_mode']}, {row['feature_set']}, "
            f"k={row['k']}: mean_regret={row['mean_regret']}, "
            f"rel_reduction={row['relative_regret_reduction']}, "
            f"accuracy={row['accuracy']}, fallback={row['fallback_rate']}"
        )


if __name__ == "__main__":
    main()
