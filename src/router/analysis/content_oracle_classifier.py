import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from src.router.adaptation.content_metadata_policy import (
    build_candidate_lookup,
    load_metadata_oracle_rows,
    resolve_global_baseline,
)


Pair = Tuple[str, str]


FEATURE_SETS = {
    "metadata_no_source": {
        "numeric": ["megapixels", "aspect_ratio"],
        "categorical": ["resolution_class", "orientation_class"],
    },
    "pixel_no_source": {
        "numeric": [
            "megapixels",
            "aspect_ratio",
            "luminance_std",
            "luminance_entropy_norm",
            "gradient_mean",
            "gradient_std",
            "edge_density",
            "flat_area_ratio",
            "colorfulness",
        ],
        "categorical": [
            "resolution_class",
            "orientation_class",
            "entropy_class",
            "edge_class",
            "texture_class",
            "color_class",
        ],
    },
    "all_no_source": {
        "numeric": [
            "megapixels",
            "aspect_ratio",
            "luminance_std",
            "luminance_entropy_norm",
            "gradient_mean",
            "gradient_std",
            "edge_density",
            "flat_area_ratio",
            "colorfulness",
        ],
        "categorical": [
            "resolution_class",
            "orientation_class",
            "entropy_class",
            "edge_class",
            "texture_class",
            "color_class",
        ],
    },
    "all_with_source": {
        "numeric": [
            "megapixels",
            "aspect_ratio",
            "luminance_std",
            "luminance_entropy_norm",
            "gradient_mean",
            "gradient_std",
            "edge_density",
            "flat_area_ratio",
            "colorfulness",
        ],
        "categorical": [
            "dataset",
            "resolution_class",
            "orientation_class",
            "entropy_class",
            "edge_class",
            "texture_class",
            "color_class",
        ],
    },
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def _pair(codec: Any, config: Any) -> Pair:
    return str(codec), str(config)


def _label(pair: Pair) -> str:
    return f"{pair[0]}|{pair[1]}"


def _split_label(label: str) -> Pair:
    if "|" not in label:
        raise ValueError(f"Invalid oracle label: {label}")
    codec, config = label.split("|", 1)
    return codec, config


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    q = max(0.0, min(1.0, q))
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def load_pixel_feature_rows(path: str) -> Dict[str, Dict[str, Any]]:
    rows = {}

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                rows[image_id] = dict(row)

    if not rows:
        raise ValueError("No pixel feature rows loaded.")

    return rows


def load_classifier_rows(
    *,
    metadata_oracle_csv: str,
    pixel_features_csv: str,
) -> List[Dict[str, Any]]:
    metadata_rows = load_metadata_oracle_rows(metadata_oracle_csv)
    pixel_by_image = load_pixel_feature_rows(pixel_features_csv)

    rows = []

    for meta in metadata_rows:
        image_id = str(meta.get("image_id", "")).strip()
        pixel = pixel_by_image.get(image_id)

        if pixel is None:
            continue

        row = dict(meta)

        for key, value in pixel.items():
            if key in {"dataset", "image", "image_id"}:
                continue
            row[key] = value

        oracle_pair = _pair(row.get("oracle_codec"), row.get("oracle_config"))
        row["oracle_label"] = _label(oracle_pair)

        rows.append(row)

    if not rows:
        raise ValueError("Metadata/pixel join produced no classifier rows.")

    return rows


def _feature_spec(feature_set: str) -> Tuple[List[str], List[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}. "
            f"Expected one of: {', '.join(sorted(FEATURE_SETS))}"
        )

    spec = FEATURE_SETS[feature_set]
    return list(spec["numeric"]), list(spec["categorical"])


def _fit_numeric_stats(
    rows: List[Dict[str, Any]],
    numeric_features: List[str],
) -> Dict[str, Tuple[float, float]]:
    stats = {}

    for feature in numeric_features:
        values = [_to_float(row.get(feature)) for row in rows]

        if not values:
            stats[feature] = (0.0, 1.0)
            continue

        mu = mean(values)
        var = mean((x - mu) ** 2 for x in values)
        sigma = math.sqrt(var)

        if sigma <= 1e-12:
            sigma = 1.0

        stats[feature] = (mu, sigma)

    return stats


def _fit_categorical_values(
    rows: List[Dict[str, Any]],
    categorical_features: List[str],
) -> Dict[str, List[str]]:
    values = {}

    for feature in categorical_features:
        values[feature] = sorted({str(row.get(feature, "unknown")) for row in rows})

    return values


def _encode_row(
    row: Dict[str, Any],
    *,
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_stats: Dict[str, Tuple[float, float]],
    categorical_values: Dict[str, List[str]],
) -> List[float]:
    vector: List[float] = []

    for feature in numeric_features:
        value = _to_float(row.get(feature))
        mu, sigma = numeric_stats[feature]
        vector.append((value - mu) / sigma)

    for feature in categorical_features:
        value = str(row.get(feature, "unknown"))
        allowed = categorical_values.get(feature, [])

        for candidate in allowed:
            vector.append(1.0 if value == candidate else 0.0)

    return vector


def _squared_distance(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _label_from_sorted_distances(
    sorted_distances: List[Tuple[float, str]],
    k: int,
) -> str:
    if k <= 0:
        raise ValueError("k must be positive.")

    if not sorted_distances:
        raise ValueError("Cannot predict kNN label with an empty training set.")

    nearest = sorted_distances[: min(k, len(sorted_distances))]
    votes = Counter(label for _, label in nearest)

    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _predict_knn_label(
    *,
    train_rows: List[Dict[str, Any]],
    test_row: Dict[str, Any],
    feature_set: str,
    k: int,
) -> str:
    numeric_features, categorical_features = _feature_spec(feature_set)

    numeric_stats = _fit_numeric_stats(train_rows, numeric_features)
    categorical_values = _fit_categorical_values(train_rows, categorical_features)

    test_vec = _encode_row(
        test_row,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_stats=numeric_stats,
        categorical_values=categorical_values,
    )

    distances = []

    for row in train_rows:
        vec = _encode_row(
            row,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_stats=numeric_stats,
            categorical_values=categorical_values,
        )

        distances.append(
            (
                _squared_distance(test_vec, vec),
                str(row["oracle_label"]),
            )
        )

    distances.sort(key=lambda item: item[0])

    return _label_from_sorted_distances(distances, k)


def _candidate_is_feasible(
    candidate: Optional[Dict[str, Any]],
    quality_floor: Optional[float],
) -> bool:
    if candidate is None:
        return False

    quality = _to_float(candidate.get("quality"))

    if quality_floor is None:
        return True

    return quality >= quality_floor


def evaluate_oracle_classifier(
    *,
    rows: List[Dict[str, Any]],
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    feature_set: str,
    k: int,
    quality_floor: Optional[float],
    global_baseline: Pair,
) -> Dict[str, Any]:
    decisions = []

    for index, test_row in enumerate(rows):
        train_rows = [row for i, row in enumerate(rows) if i != index]

        predicted_label = _predict_knn_label(
            train_rows=train_rows,
            test_row=test_row,
            feature_set=feature_set,
            k=k,
        )

        predicted_pair = _split_label(predicted_label)
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

        decisions.append(
            {
                "dataset": test_row.get("dataset"),
                "image": test_row.get("image"),
                "image_id": image_id,
                "feature_set": feature_set,
                "k": k,
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
        )

    summary = summarize_classifier_decisions(
        decisions,
        feature_set=feature_set,
        k=k,
        quality_floor=quality_floor,
        global_baseline=global_baseline,
    )

    return {
        "decisions": decisions,
        "summary": summary,
    }


def summarize_classifier_decisions(
    decisions: List[Dict[str, Any]],
    *,
    feature_set: str,
    k: int,
    quality_floor: Optional[float],
    global_baseline: Pair,
) -> List[Dict[str, Any]]:
    summary = []

    def add(section: str, key: str, value: Any) -> None:
        summary.append(
            {
                "section": section,
                "key": key,
                "value": value,
            }
        )

    regrets = [
        float(row["regret"]) for row in decisions if row.get("regret") is not None
    ]

    global_regrets = [
        float(row["global_regret"])
        for row in decisions
        if row.get("global_regret") is not None
    ]

    reductions = [
        float(row["regret_reduction_vs_global"])
        for row in decisions
        if row.get("regret_reduction_vs_global") is not None
    ]

    correct = [row for row in decisions if row.get("correct_oracle_match") is True]

    fallback = [row for row in decisions if row.get("fallback_used") is True]

    infeasible = [row for row in decisions if row.get("selected_feasible") is not True]

    add("summary", "feature_set", feature_set)
    add("summary", "k", k)
    add("summary", "num_images", len(decisions))
    add("summary", "quality_floor", quality_floor)
    add("summary", "global_baseline_codec", global_baseline[0])
    add("summary", "global_baseline_config", global_baseline[1])

    add("accuracy", "oracle_match_count", len(correct))
    add(
        "accuracy",
        "oracle_match_rate",
        len(correct) / len(decisions) if decisions else None,
    )

    add("fallback", "fallback_count", len(fallback))
    add(
        "fallback",
        "fallback_rate",
        len(fallback) / len(decisions) if decisions else None,
    )

    add("feasibility", "infeasible_count", len(infeasible))
    add(
        "feasibility",
        "infeasible_rate",
        len(infeasible) / len(decisions) if decisions else None,
    )

    if regrets:
        add("regret", "mean", mean(regrets))
        add("regret", "median", median(regrets))
        add("regret", "p90", _quantile(regrets, 0.90))
        add("regret", "max", max(regrets))

    if global_regrets:
        add("global_regret", "mean", mean(global_regrets))
        add("global_regret", "median", median(global_regrets))
        add("global_regret", "p90", _quantile(global_regrets, 0.90))
        add("global_regret", "max", max(global_regrets))

    if reductions:
        add("regret_reduction", "mean", mean(reductions))
        add("regret_reduction", "median", median(reductions))
        add("regret_reduction", "p90", _quantile(reductions, 0.90))
        add("regret_reduction", "max", max(reductions))

        baseline_mean = mean(global_regrets) if global_regrets else None
        reduction_mean = mean(reductions)

        if baseline_mean and baseline_mean > 0:
            add(
                "regret_reduction",
                "relative_mean",
                reduction_mean / baseline_mean,
            )

    selected_counter = Counter(
        f"{row.get('selected_codec')}|{row.get('selected_config')}"
        for row in decisions
    )

    for key, count in selected_counter.most_common():
        add("selected_count", key, count)

    predicted_counter = Counter(
        f"{row.get('predicted_codec')}|{row.get('predicted_config')}"
        for row in decisions
    )

    for key, count in predicted_counter.most_common():
        add("predicted_count", key, count)

    oracle_counter = Counter(
        f"{row.get('oracle_codec')}|{row.get('oracle_config')}"
        for row in decisions
    )

    for key, count in oracle_counter.most_common():
        add("oracle_count", key, count)

    return summary


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
        description=(
            "Evaluate lightweight k-NN oracle classifier for content-aware R-D-E "
            "routing."
        )
    )

    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--metadata-oracle-csv", required=True)
    parser.add_argument("--pixel-features-csv", required=True)

    parser.add_argument(
        "--feature-set",
        default="all_no_source",
        choices=sorted(FEATURE_SETS.keys()),
    )

    parser.add_argument("--k", type=int, default=3)
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
        "--decisions-out",
        default="results/routing_context/v09_oracle_classifier_decisions.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="results/routing_context/v09_oracle_classifier_summary.csv",
    )

    args = parser.parse_args()

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

    evaluation = evaluate_oracle_classifier(
        rows=rows,
        candidate_lookup=candidate_lookup,
        feature_set=args.feature_set,
        k=args.k,
        quality_floor=args.quality_floor,
        global_baseline=global_baseline,
    )

    write_csv(args.decisions_out, evaluation["decisions"])
    write_csv(args.summary_out, evaluation["summary"])

    summary = {(row["section"], row["key"]): row["value"] for row in evaluation["summary"]}

    print("\n=== R-D-E Oracle Classifier ===")
    print(f"Feature set:            {args.feature_set}")
    print(f"k:                      {args.k}")
    print(f"Images:                 {summary.get(('summary', 'num_images'))}")
    print("Global baseline:        " f"{global_baseline[0]} {global_baseline[1]}")
    print(f"Oracle match rate:      {summary.get(('accuracy', 'oracle_match_rate'))}")
    print(f"Mean regret:            {summary.get(('regret', 'mean'))}")
    print(f"Global mean regret:     {summary.get(('global_regret', 'mean'))}")
    print(f"Mean regret reduction:  {summary.get(('regret_reduction', 'mean'))}")
    print(f"Relative reduction:     {summary.get(('regret_reduction', 'relative_mean'))}")
    print(f"Fallback rate:          {summary.get(('fallback', 'fallback_rate'))}")
    print(f"Decisions CSV:          {args.decisions_out}")
    print(f"Summary CSV:            {args.summary_out}")


if __name__ == "__main__":
    main()
