import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional


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


def _read_csv(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    clean = [v for v in values if v is not None]

    if not clean:
        return {
            "num_samples": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "max_ms": None,
        }

    return {
        "num_samples": len(clean),
        "mean_ms": mean(clean),
        "median_ms": median(clean),
        "p90_ms": _quantile(clean, 0.90),
        "max_ms": max(clean),
    }


def _case_encode_times(
    rows: List[Dict[str, str]],
    *,
    dataset_col: str,
    codec_col: str,
    config_col: str,
    time_col: str,
    codec: str,
    config: str,
    dataset: Optional[str] = None,
) -> List[float]:
    values = []

    for row in rows:
        if str(row.get(codec_col, "")).strip() != codec:
            continue

        if str(row.get(config_col, "")).strip() != config:
            continue

        if dataset is not None and str(row.get(dataset_col, "")).strip() != dataset:
            continue

        value = _to_float(row.get(time_col))

        if value is not None:
            values.append(value)

    return values


def build_overhead_table(
    *,
    pixel_features_csv: str,
    benchmark_csv: str,
    dataset_col: str = "dataset",
    codec_col: str = "codec",
    config_col: str = "param",
    time_col: str = "time_ms",
) -> List[Dict[str, Any]]:
    pixel_rows = _read_csv(pixel_features_csv)
    benchmark_rows = _read_csv(benchmark_csv)

    pixel_overheads = [
        _to_float(row.get("feature_overhead_ms"))
        for row in pixel_rows
    ]
    pixel_overheads = [v for v in pixel_overheads if v is not None]

    pixel_stats = _stats(pixel_overheads)
    pixel_mean = pixel_stats["mean_ms"]

    table: List[Dict[str, Any]] = []

    def add_row(
        *,
        component: str,
        case_id: str,
        scope: str,
        codec: str = "",
        config: str = "",
        dataset: str = "",
        stats: Dict[str, Any],
        notes: str,
    ) -> None:
        mean_ms = stats.get("mean_ms")

        ratio = None
        if pixel_mean is not None and mean_ms is not None and mean_ms > 0:
            ratio = pixel_mean / mean_ms

        table.append(
            {
                "component": component,
                "case_id": case_id,
                "scope": scope,
                "dataset": dataset,
                "codec": codec,
                "config": config,
                "num_samples": stats.get("num_samples"),
                "mean_ms": mean_ms,
                "median_ms": stats.get("median_ms"),
                "p90_ms": stats.get("p90_ms"),
                "max_ms": stats.get("max_ms"),
                "pixel_feature_mean_over_this_mean": ratio,
                "notes": notes,
            }
        )

    add_row(
        component="content_feature_extraction",
        case_id="pixel_features_long_side_256",
        scope="all_images",
        stats=pixel_stats,
        notes=(
            "Full pixel feature extraction: resize, luminance statistics, "
            "entropy, gradients, edge density, colorfulness."
        ),
    )

    add_row(
        component="content_feature_extraction",
        case_id="metadata_no_source",
        scope="image_header_or_known_metadata",
        stats={
            "num_samples": len(pixel_rows),
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "max_ms": None,
        },
        notes=(
            "Metadata-only classifier uses width, height, megapixels, aspect ratio, "
            "resolution class and orientation class. It avoids full pixel scanning; "
            "exact overhead is not measured in this table."
        ),
    )

    cases = [
        {
            "case_id": "jpeg_q85_tecnick",
            "scope": "source_filtered_tecnick",
            "dataset": "tecnick",
            "codec": "JPEG",
            "config": "q=85",
            "notes": "Fast source-filtered JPEG case selected by content-aware routing on Tecnick.",
        },
        {
            "case_id": "hevc_crf15_global",
            "scope": "global_all_datasets",
            "dataset": "",
            "codec": "HEVC",
            "config": "crf=15",
            "notes": "Robust global full-coverage baseline.",
        },
        {
            "case_id": "jpeg_q85_global",
            "scope": "global_all_datasets",
            "dataset": "",
            "codec": "JPEG",
            "config": "q=85",
            "notes": "Global JPEG q=85 timing reference; may not satisfy the global quality guard.",
        },
        {
            "case_id": "jxl_d1_global",
            "scope": "global_all_datasets",
            "dataset": "",
            "codec": "JXL",
            "config": "d=1.0",
            "notes": "Global JXL d=1.0 timing reference.",
        },
    ]

    for case in cases:
        values = _case_encode_times(
            benchmark_rows,
            dataset_col=dataset_col,
            codec_col=codec_col,
            config_col=config_col,
            time_col=time_col,
            codec=case["codec"],
            config=case["config"],
            dataset=case["dataset"] or None,
        )

        add_row(
            component="encoding_time",
            case_id=case["case_id"],
            scope=case["scope"],
            dataset=case["dataset"],
            codec=case["codec"],
            config=case["config"],
            stats=_stats(values),
            notes=case["notes"],
        )

    return table


def build_knn_sensitivity_table(
    *,
    classifier_sweep_summary_csv: str,
    feature_sets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rows = _read_csv(classifier_sweep_summary_csv)

    if feature_sets is not None:
        allowed = set(feature_sets)
        rows = [
            row for row in rows
            if row.get("feature_set") in allowed
        ]

    if not rows:
        raise ValueError("No sweep rows left after filtering.")

    best_by_eval = {}
    best_by_eval_feature = {}

    for row in rows:
        evaluation_mode = str(row.get("evaluation_mode"))
        feature_set = str(row.get("feature_set"))
        mean_regret = _to_float(row.get("mean_regret"), math.inf)

        eval_key = evaluation_mode
        eval_feature_key = (evaluation_mode, feature_set)

        if (
            eval_key not in best_by_eval
            or mean_regret < _to_float(best_by_eval[eval_key].get("mean_regret"), math.inf)
        ):
            best_by_eval[eval_key] = row

        if (
            eval_feature_key not in best_by_eval_feature
            or mean_regret < _to_float(best_by_eval_feature[eval_feature_key].get("mean_regret"), math.inf)
        ):
            best_by_eval_feature[eval_feature_key] = row

    out = []

    for row in rows:
        evaluation_mode = str(row.get("evaluation_mode"))
        feature_set = str(row.get("feature_set"))
        k = str(row.get("k"))

        best_eval = best_by_eval[evaluation_mode]
        best_feature = best_by_eval_feature[(evaluation_mode, feature_set)]

        out.append(
            {
                "evaluation_mode": evaluation_mode,
                "feature_set": feature_set,
                "k": k,
                "accuracy": _to_float(row.get("accuracy")),
                "mean_regret": _to_float(row.get("mean_regret")),
                "median_regret": _to_float(row.get("median_regret")),
                "p90_regret": _to_float(row.get("p90_regret")),
                "max_regret": _to_float(row.get("max_regret")),
                "relative_regret_reduction": _to_float(row.get("relative_regret_reduction")),
                "fallback_rate": _to_float(row.get("fallback_rate")),
                "infeasible_rate": _to_float(row.get("infeasible_rate")),
                "best_for_evaluation_mode": (
                    feature_set == str(best_eval.get("feature_set"))
                    and k == str(best_eval.get("k"))
                ),
                "best_for_feature_set": (
                    k == str(best_feature.get("k"))
                ),
            }
        )

    out.sort(
        key=lambda row: (
            row["evaluation_mode"],
            row["feature_set"],
            _to_int(row["k"], 10**9) or 10**9,
        )
    )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build overhead and k-sensitivity tables for content-aware R-D-E routing."
    )

    parser.add_argument(
        "--pixel-features-csv",
        default="results/routing_context/v09_image_pixel_features.csv",
    )
    parser.add_argument(
        "--benchmark-csv",
        default="results/images/image_4dataset_RDE_paper_ready.csv",
    )
    parser.add_argument(
        "--classifier-sweep-summary",
        default="results/routing_context/v09_oracle_classifier_sweep_summary.csv",
    )

    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="param")
    parser.add_argument("--time-col", default="time_ms")

    parser.add_argument(
        "--overhead-out",
        default="results/routing_context/v09_content_aware_overhead_table.csv",
    )
    parser.add_argument(
        "--sensitivity-out",
        default="results/routing_context/v09_knn_sensitivity_table.csv",
    )

    args = parser.parse_args()

    overhead = build_overhead_table(
        pixel_features_csv=args.pixel_features_csv,
        benchmark_csv=args.benchmark_csv,
        dataset_col=args.dataset_col,
        codec_col=args.codec_col,
        config_col=args.config_col,
        time_col=args.time_col,
    )

    sensitivity = build_knn_sensitivity_table(
        classifier_sweep_summary_csv=args.classifier_sweep_summary,
    )

    _write_csv(args.overhead_out, overhead)
    _write_csv(args.sensitivity_out, sensitivity)

    print("\n=== R-D-E Content-Aware Overhead and Sensitivity ===")
    print(f"Overhead table:     {args.overhead_out}")
    print(f"Sensitivity table:  {args.sensitivity_out}")

    print("\nOverhead rows:")
    for row in overhead:
        print(
            f"  {row['case_id']}: mean_ms={row['mean_ms']}, "
            f"ratio_pixel_over_case={row['pixel_feature_mean_over_this_mean']}"
        )

    print("\nBest sensitivity rows:")
    for row in sensitivity:
        if row["best_for_evaluation_mode"]:
            print(
                f"  {row['evaluation_mode']}: {row['feature_set']} k={row['k']}, "
                f"mean_regret={row['mean_regret']}, "
                f"relative_reduction={row['relative_regret_reduction']}"
            )


if __name__ == "__main__":
    main()