import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def _normalize_token(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _parse_codec_list(value: Optional[str]) -> Optional[set[str]]:
    if value is None or str(value).strip() == "":
        return None
    return {
        _normalize_token(x)
        for x in str(value).split(",")
        if str(x).strip()
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _minmax(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp01((value - lo) / (hi - lo))


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


def _safe_log10(value: float) -> float:
    return math.log10(max(value, 1e-12))


def _passes_quality(value: float, floor: Optional[float]) -> bool:
    if floor is None:
        return True
    return value >= floor


def _make_image_id(dataset: str, image: str) -> str:
    return f"{dataset}::{image}"


def load_oracle_rows(
    csv_path: str,
    *,
    dataset_col: str,
    image_col: str,
    codec_col: str,
    config_col: str,
    rate_col: str,
    quality_col: str,
    energy_col: str,
    time_col: Optional[str],
    available_codecs: Optional[str] = None,
    exclude_codecs: Optional[str] = None,
) -> List[Dict[str, Any]]:
    allowed = _parse_codec_list(available_codecs)
    excluded = _parse_codec_list(exclude_codecs)

    rows: List[Dict[str, Any]] = []

    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            codec = str(raw.get(codec_col, "")).strip()
            config = str(raw.get(config_col, "")).strip()
            dataset = str(raw.get(dataset_col, "")).strip()
            image = str(raw.get(image_col, "")).strip()

            if not codec or not config or not dataset or not image:
                continue

            codec_norm = _normalize_token(codec)

            if allowed is not None and codec_norm not in allowed:
                continue

            if excluded is not None and codec_norm in excluded:
                continue

            rate = _parse_float(raw.get(rate_col))
            quality = _parse_float(raw.get(quality_col))
            energy = _parse_float(raw.get(energy_col))
            time_ms = _parse_float(raw.get(time_col)) if time_col else None

            if rate is None or quality is None or energy is None:
                continue

            rows.append(
                {
                    "dataset": dataset,
                    "image": image,
                    "image_id": _make_image_id(dataset, image),
                    "codec": codec,
                    "config": config,
                    "rate": rate,
                    "quality": quality,
                    "energy": energy,
                    "time_ms": time_ms,
                    "raw": raw,
                }
            )

    if not rows:
        raise ValueError("No valid oracle-analysis rows were loaded.")

    return rows


def add_global_normalized_costs(
    rows: List[Dict[str, Any]],
    *,
    w_r: float,
    w_e: float,
    w_d: float,
) -> None:
    total = w_r + w_e + w_d
    if total <= 0:
        raise ValueError("Weight sum must be positive.")

    w_r = w_r / total
    w_e = w_e / total
    w_d = w_d / total

    log_rates = [_safe_log10(r["rate"]) for r in rows]
    log_energies = [_safe_log10(r["energy"]) for r in rows]
    qualities = [r["quality"] for r in rows]

    rate_lo, rate_hi = min(log_rates), max(log_rates)
    energy_lo, energy_hi = min(log_energies), max(log_energies)
    q_lo, q_hi = min(qualities), max(qualities)

    for r in rows:
        r_n = _minmax(_safe_log10(r["rate"]), rate_lo, rate_hi)
        e_n = _minmax(_safe_log10(r["energy"]), energy_lo, energy_hi)

        q_n = _minmax(r["quality"], q_lo, q_hi)
        d_n = 1.0 - q_n

        cost = w_r * r_n + w_e * e_n + w_d * d_n

        r["norm_rate"] = r_n
        r["norm_energy"] = e_n
        r["norm_distortion"] = d_n
        r["J_RDE"] = cost


def _select_global_best(
    rows: List[Dict[str, Any]],
    *,
    quality_floor: Optional[float],
    global_coverage_floor: float = 1.0,
) -> Dict[str, Any]:
    all_image_ids = sorted({r["image_id"] for r in rows})
    num_images = len(all_image_ids)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        grouped[(r["codec"], r["config"])].append(r)

    candidates = []

    for (codec, config), group in grouped.items():
        safe_rows = [
            r for r in group
            if _passes_quality(r["quality"], quality_floor)
        ]

        safe_image_ids = {r["image_id"] for r in safe_rows}
        coverage_rate = len(safe_image_ids) / num_images if num_images else 0.0

        if coverage_rate + 1e-12 < global_coverage_floor:
            continue

        if not safe_rows:
            continue

        candidates.append(
            {
                "codec": codec,
                "config": config,
                "mean_cost": mean(r["J_RDE"] for r in safe_rows),
                "mean_rate": mean(r["rate"] for r in safe_rows),
                "mean_quality": mean(r["quality"] for r in safe_rows),
                "mean_energy": mean(r["energy"] for r in safe_rows),
                "num_rows": len(group),
                "num_safe_images": len(safe_image_ids),
                "num_total_images": num_images,
                "coverage_rate": coverage_rate,
                "coverage_floor": global_coverage_floor,
            }
        )

    if not candidates:
        raise ValueError(
            "No global baseline satisfies the requested coverage floor. "
            f"coverage_floor={global_coverage_floor}, quality_floor={quality_floor}"
        )

    candidates.sort(key=lambda x: x["mean_cost"])
    return candidates[0]


def analyze_content_oracle(
    rows: List[Dict[str, Any]],
    *,
    quality_floor: Optional[float],
    w_r: float,
    w_e: float,
    w_d: float,
    global_coverage_floor: float = 1.0,
) -> Dict[str, Any]:
    add_global_normalized_costs(
        rows,
        w_r=w_r,
        w_e=w_e,
        w_d=w_d,
    )

    global_best = _select_global_best(
        rows,
        quality_floor=quality_floor,
        global_coverage_floor=global_coverage_floor,
    )

    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        by_image[r["image_id"]].append(r)

    image_reports = []
    oracle_counter = Counter()
    dataset_regrets: Dict[str, List[float]] = defaultdict(list)

    num_without_safe = 0
    num_global_infeasible = 0

    global_key = (global_best["codec"], global_best["config"])

    for image_id, group in sorted(by_image.items()):
        safe = [
            r for r in group
            if _passes_quality(r["quality"], quality_floor)
        ]

        dataset = group[0]["dataset"]
        image = group[0]["image"]

        if not safe:
            num_without_safe += 1
            image_reports.append(
                {
                    "dataset": dataset,
                    "image": image,
                    "image_id": image_id,
                    "num_candidates": len(group),
                    "num_safe_candidates": 0,
                    "oracle_codec": None,
                    "oracle_config": None,
                    "oracle_cost": None,
                    "global_codec": global_best["codec"],
                    "global_config": global_best["config"],
                    "global_cost": None,
                    "global_feasible": False,
                    "regret": None,
                }
            )
            continue

        oracle = min(safe, key=lambda r: r["J_RDE"])
        oracle_counter[(oracle["codec"], oracle["config"])] += 1

        global_rows = [
            r for r in group
            if (r["codec"], r["config"]) == global_key
        ]

        global_row = global_rows[0] if global_rows else None

        global_feasible = (
            global_row is not None
            and _passes_quality(global_row["quality"], quality_floor)
        )

        if global_feasible:
            regret = global_row["J_RDE"] - oracle["J_RDE"]
            dataset_regrets[dataset].append(regret)
            global_cost = global_row["J_RDE"]
        else:
            regret = None
            global_cost = None
            num_global_infeasible += 1

        image_reports.append(
            {
                "dataset": dataset,
                "image": image,
                "image_id": image_id,
                "num_candidates": len(group),
                "num_safe_candidates": len(safe),
                "oracle_codec": oracle["codec"],
                "oracle_config": oracle["config"],
                "oracle_rate": oracle["rate"],
                "oracle_quality": oracle["quality"],
                "oracle_energy": oracle["energy"],
                "oracle_time_ms": oracle["time_ms"],
                "oracle_cost": oracle["J_RDE"],
                "global_codec": global_best["codec"],
                "global_config": global_best["config"],
                "global_cost": global_cost,
                "global_feasible": global_feasible,
                "regret": regret,
            }
        )

    regrets = [
        r["regret"] for r in image_reports
        if r["regret"] is not None
    ]
    oracle_matches_global = oracle_counter.get(global_key, 0)
    num_analyzed = len(by_image) - num_without_safe
    oracle_switch_count = num_analyzed - oracle_matches_global
    oracle_switch_rate = (
        oracle_switch_count / num_analyzed
        if num_analyzed > 0
        else None
    )

    summary_rows = []

    def add(section: str, key: str, value: Any) -> None:
        summary_rows.append(
            {
                "section": section,
                "key": key,
                "value": value,
            }
        )

    add("summary", "num_rows", len(rows))
    add("summary", "num_images_total", len(by_image))
    add("summary", "num_images_analyzed", len(by_image) - num_without_safe)
    add("summary", "num_images_without_safe_candidates", num_without_safe)
    add("summary", "num_global_infeasible_images", num_global_infeasible)
    add("summary", "quality_floor", quality_floor)
    add("summary", "global_coverage_floor", global_coverage_floor)
    add("summary", "w_R", w_r)
    add("summary", "w_E", w_e)
    add("summary", "w_D", w_d)

    add("global_best", "codec", global_best["codec"])
    add("global_best", "config", global_best["config"])
    add("global_best", "mean_cost", global_best["mean_cost"])
    add("global_best", "mean_rate", global_best["mean_rate"])
    add("global_best", "mean_quality", global_best["mean_quality"])
    add("global_best", "mean_energy", global_best["mean_energy"])
    add("global_best", "num_safe_images", global_best["num_safe_images"])
    add("global_best", "num_total_images", global_best["num_total_images"])
    add("global_best", "coverage_rate", global_best["coverage_rate"])
    add("global_best", "coverage_floor", global_best["coverage_floor"])

    add("oracle_diversity", "num_distinct_oracle_configs", len(oracle_counter))
    add("oracle_diversity", "oracle_matches_global_count", oracle_matches_global)
    add("oracle_diversity", "oracle_switch_count", oracle_switch_count)
    add("oracle_diversity", "oracle_switch_rate", oracle_switch_rate)

    if regrets:
        add("regret", "mean", mean(regrets))
        add("regret", "median", median(regrets))
        add("regret", "p90", _quantile(regrets, 0.90))
        add("regret", "max", max(regrets))
    else:
        add("regret", "mean", None)
        add("regret", "median", None)
        add("regret", "p90", None)
        add("regret", "max", None)

    for (codec, config), count in oracle_counter.most_common():
        add("oracle_count", f"{codec}|{config}", count)

    for dataset, values in sorted(dataset_regrets.items()):
        if values:
            add("dataset_mean_regret", dataset, mean(values))
            add("dataset_p90_regret", dataset, _quantile(values, 0.90))

    return {
        "global_best": global_best,
        "by_image": image_reports,
        "summary": summary_rows,
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content oracle/regret analysis for R-D-E routing."
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--image-col", default="image")
    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="param")
    parser.add_argument("--rate-col", default="bpp")
    parser.add_argument("--quality-col", default="ssimulacra2")
    parser.add_argument("--energy-col", default="energy_per_image_j")
    parser.add_argument("--time-col", default="time_ms")

    parser.add_argument("--available-codecs", default=None)
    parser.add_argument("--exclude-codecs", default=None)

    parser.add_argument("--quality-floor", type=float, default=80.0)
    parser.add_argument(
        "--global-coverage-floor",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of images where the global baseline must "
            "satisfy the quality floor."
        ),
    )

    parser.add_argument("--wR", type=float, default=0.2)
    parser.add_argument("--wE", type=float, default=0.2)
    parser.add_argument("--wD", type=float, default=0.6)

    parser.add_argument(
        "--by-image-out",
        default="results/routing_context/v09_content_oracle_by_image.csv",
    )

    parser.add_argument(
        "--summary-out",
        default="results/routing_context/v09_content_oracle_summary.csv",
    )

    args = parser.parse_args()

    rows = load_oracle_rows(
        csv_path=args.csv,
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
    )

    analysis = analyze_content_oracle(
        rows,
        quality_floor=args.quality_floor,
        w_r=args.wR,
        w_e=args.wE,
        w_d=args.wD,
        global_coverage_floor=args.global_coverage_floor,
    )

    write_csv(args.by_image_out, analysis["by_image"])
    write_csv(args.summary_out, analysis["summary"])

    global_best = analysis["global_best"]
    summary = {
        (r["section"], r["key"]): r["value"]
        for r in analysis["summary"]
    }

    print("\n=== R-D-E Content Oracle Analysis ===")
    print(f"Rows loaded:          {len(rows)}")
    print(f"Quality floor:        {args.quality_floor}")
    print(
        "Global best:          "
        f"{global_best['codec']} {global_best['config']} "
        f"(mean J={global_best['mean_cost']})"
    )
    print(f"Global coverage:      {global_best.get('coverage_rate')}")
    print(f"Coverage floor:       {global_best.get('coverage_floor')}")
    print(
        "Images analyzed:      "
        f"{summary.get(('summary', 'num_images_analyzed'))} / "
        f"{summary.get(('summary', 'num_images_total'))}"
    )
    print(
        "Oracle configs:       "
        f"{summary.get(('oracle_diversity', 'num_distinct_oracle_configs'))}"
    )
    print(f"Mean regret:          {summary.get(('regret', 'mean'))}")
    print(f"Median regret:        {summary.get(('regret', 'median'))}")
    print(f"P90 regret:           {summary.get(('regret', 'p90'))}")
    print(f"Max regret:           {summary.get(('regret', 'max'))}")
    print(f"By-image CSV:         {args.by_image_out}")
    print(f"Summary CSV:          {args.summary_out}")


if __name__ == "__main__":
    main()
