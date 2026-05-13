import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..analysis.content_oracle_analysis import (
        add_global_normalized_costs,
        load_oracle_rows,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.router.analysis.content_oracle_analysis import (
        add_global_normalized_costs,
        load_oracle_rows,
    )


Pair = Tuple[str, str]


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    q = max(0.0, min(1.0, q))
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)

    if lo == hi:
        return values[lo]

    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _pair(codec: Any, config: Any) -> Pair:
    return str(codec), str(config)


def _sort_counter(counter: Counter) -> List[Tuple[Pair, int]]:
    return sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )


def load_metadata_oracle_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if not row.get("image_id"):
                continue

            rows.append(dict(row))

    if not rows:
        raise ValueError("No metadata/oracle rows loaded.")

    return rows


def infer_global_baseline(
    rows: List[Dict[str, Any]],
    *,
    require_unique: bool = True,
) -> Pair:
    """Infer the robust global baseline from metadata/oracle rows.

    The expected input is the by-image output produced by
    content_oracle_analysis, where every row should carry the same
    global_codec/global_config pair. That pair is the single globally feasible
    baseline selected by the full-coverage oracle analysis.

    By default, this function requires that the pair is unique across all rows.
    This avoids silently returning the most common pair when preprocessing or
    data merging accidentally makes the global baseline non-homogeneous.

    Set require_unique=False only for backward-compatible exploratory analysis,
    where choosing the modal pair is intentional.
    """
    counter = Counter(
        _pair(row.get("global_codec"), row.get("global_config"))
        for row in rows
        if row.get("global_codec") and row.get("global_config")
    )

    if not counter:
        raise ValueError("Cannot infer global baseline from metadata/oracle rows.")

    if require_unique and len(counter) != 1:
        details = ", ".join(
            f"{codec} {config}: {count}"
            for (codec, config), count in _sort_counter(counter)
        )
        raise ValueError(
            "Ambiguous global baseline in metadata/oracle rows. "
            "Expected one unique global_codec/global_config pair, but found: "
            f"{details}. "
            "Pass an explicit global_baseline to the evaluator, or regenerate "
            "the oracle rows from a single global coverage analysis."
        )

    return _sort_counter(counter)[0][0]


def resolve_global_baseline(
    rows: List[Dict[str, Any]],
    *,
    global_baseline_codec: Optional[str] = None,
    global_baseline_config: Optional[str] = None,
) -> Pair:
    """Resolve the global baseline either explicitly or from homogeneous rows."""
    has_codec = (
        global_baseline_codec is not None
        and str(global_baseline_codec).strip() != ""
    )
    has_config = (
        global_baseline_config is not None
        and str(global_baseline_config).strip() != ""
    )

    if has_codec != has_config:
        raise ValueError(
            "global_baseline_codec and global_baseline_config must be provided together."
        )

    if has_codec and has_config:
        return _pair(global_baseline_codec, global_baseline_config)

    return infer_global_baseline(rows, require_unique=True)


def build_candidate_lookup(
    benchmark_csv: str,
    *,
    dataset_col: str = "dataset",
    image_col: str = "image",
    codec_col: str = "codec",
    config_col: str = "param",
    rate_col: str = "bpp",
    quality_col: str = "ssimulacra2",
    energy_col: str = "energy_per_image_j",
    time_col: str = "time_ms",
    available_codecs: Optional[str] = None,
    exclude_codecs: Optional[str] = None,
    w_r: float = 0.2,
    w_e: float = 0.2,
    w_d: float = 0.6,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    rows = load_oracle_rows(
        csv_path=benchmark_csv,
        dataset_col=dataset_col,
        image_col=image_col,
        codec_col=codec_col,
        config_col=config_col,
        rate_col=rate_col,
        quality_col=quality_col,
        energy_col=energy_col,
        time_col=time_col,
        available_codecs=available_codecs,
        exclude_codecs=exclude_codecs,
    )

    add_global_normalized_costs(
        rows,
        w_r=w_r,
        w_e=w_e,
        w_d=w_d,
    )

    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for row in rows:
        key = (
            str(row["image_id"]),
            str(row["codec"]),
            str(row["config"]),
        )
        lookup[key] = row

    return lookup


def build_majority_rules(
    rows: List[Dict[str, Any]],
    *,
    policy_key: str,
    exclude_image_id: Optional[str] = None,
) -> Dict[str, Pair]:
    grouped: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        if exclude_image_id is not None and row.get("image_id") == exclude_image_id:
            continue

        group_value = str(row.get(policy_key, "unknown"))
        oracle_pair = _pair(row.get("oracle_codec"), row.get("oracle_config"))

        if oracle_pair[0] and oracle_pair[1]:
            grouped[group_value][oracle_pair] += 1

    rules: Dict[str, Pair] = {}

    for group_value, counter in grouped.items():
        if counter:
            rules[group_value] = _sort_counter(counter)[0][0]

    return rules


def _majority_for_group_leave_one_out(
    rows: List[Dict[str, Any]],
    *,
    policy_key: str,
    group_value: str,
    exclude_image_id: str,
) -> Optional[Pair]:
    counter = Counter()

    for row in rows:
        if row.get("image_id") == exclude_image_id:
            continue

        if str(row.get(policy_key, "unknown")) != group_value:
            continue

        oracle_pair = _pair(row.get("oracle_codec"), row.get("oracle_config"))

        if oracle_pair[0] and oracle_pair[1]:
            counter[oracle_pair] += 1

    if not counter:
        return None

    return _sort_counter(counter)[0][0]


def _candidate_is_feasible(
    candidate: Optional[Dict[str, Any]],
    quality_floor: Optional[float],
) -> bool:
    if candidate is None:
        return False

    quality = _parse_float(candidate.get("quality"))

    if quality is None:
        return False

    if quality_floor is None:
        return True

    return quality >= quality_floor


def evaluate_metadata_policy(
    metadata_oracle_rows: List[Dict[str, Any]],
    candidate_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    *,
    policy_key: str,
    quality_floor: Optional[float],
    global_baseline: Pair,
    evaluation_mode: str = "leave-one-out",
) -> Dict[str, Any]:
    if evaluation_mode not in {"in-sample", "leave-one-out"}:
        raise ValueError("evaluation_mode must be 'in-sample' or 'leave-one-out'.")

    full_rules = build_majority_rules(
        metadata_oracle_rows,
        policy_key=policy_key,
    )

    decisions: List[Dict[str, Any]] = []

    for row in metadata_oracle_rows:
        image_id = str(row["image_id"])
        group_value = str(row.get(policy_key, "unknown"))

        if evaluation_mode == "leave-one-out":
            proposed = _majority_for_group_leave_one_out(
                metadata_oracle_rows,
                policy_key=policy_key,
                group_value=group_value,
                exclude_image_id=image_id,
            )
        else:
            proposed = full_rules.get(group_value)

        if proposed is None:
            proposed = global_baseline

        proposed_candidate = candidate_lookup.get(
            (image_id, proposed[0], proposed[1])
        )

        proposed_feasible = _candidate_is_feasible(
            proposed_candidate,
            quality_floor,
        )

        fallback_used = False

        if proposed_feasible:
            selected_pair = proposed
            selected_candidate = proposed_candidate
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

        oracle_pair = _pair(row.get("oracle_codec"), row.get("oracle_config"))
        oracle_cost = _parse_float(row.get("oracle_cost"))

        selected_cost = (
            _parse_float(selected_candidate.get("J_RDE"))
            if selected_candidate is not None
            else None
        )

        global_regret = _parse_float(row.get("regret"))

        if selected_cost is not None and oracle_cost is not None:
            regret = selected_cost - oracle_cost
        else:
            regret = None

        decisions.append(
            {
                "dataset": row.get("dataset"),
                "image": row.get("image"),
                "image_id": image_id,
                "policy_key": policy_key,
                "policy_value": group_value,
                "evaluation_mode": evaluation_mode,
                "oracle_codec": oracle_pair[0],
                "oracle_config": oracle_pair[1],
                "oracle_cost": oracle_cost,
                "proposed_codec": proposed[0],
                "proposed_config": proposed[1],
                "proposed_feasible": proposed_feasible,
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

    rules_rows = [
        {
            "policy_key": policy_key,
            "policy_value": group_value,
            "selected_codec": pair[0],
            "selected_config": pair[1],
        }
        for group_value, pair in sorted(full_rules.items())
    ]

    summary_rows = summarize_policy_evaluation(
        decisions,
        rules_rows=rules_rows,
        policy_key=policy_key,
        evaluation_mode=evaluation_mode,
        global_baseline=global_baseline,
        quality_floor=quality_floor,
    )

    return {
        "decisions": decisions,
        "rules": rules_rows,
        "summary": summary_rows,
    }


def summarize_policy_evaluation(
    decisions: List[Dict[str, Any]],
    *,
    rules_rows: List[Dict[str, Any]],
    policy_key: str,
    evaluation_mode: str,
    global_baseline: Pair,
    quality_floor: Optional[float],
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []

    def add(section: str, key: str, value: Any) -> None:
        summary.append(
            {
                "section": section,
                "key": key,
                "value": value,
            }
        )

    regrets = [
        float(row["regret"])
        for row in decisions
        if row.get("regret") is not None
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

    correct = [
        row for row in decisions
        if row.get("correct_oracle_match") is True
    ]

    fallback = [
        row for row in decisions
        if row.get("fallback_used") is True
    ]

    infeasible = [
        row for row in decisions
        if row.get("selected_feasible") is not True
    ]

    add("summary", "policy_key", policy_key)
    add("summary", "evaluation_mode", evaluation_mode)
    add("summary", "num_images", len(decisions))
    add("summary", "quality_floor", quality_floor)
    add("summary", "global_baseline_codec", global_baseline[0])
    add("summary", "global_baseline_config", global_baseline[1])
    add("summary", "num_rules", len(rules_rows))

    add("accuracy", "oracle_match_count", len(correct))
    add("accuracy", "oracle_match_rate", len(correct) / len(decisions) if decisions else None)

    add("fallback", "fallback_count", len(fallback))
    add("fallback", "fallback_rate", len(fallback) / len(decisions) if decisions else None)

    add("feasibility", "infeasible_count", len(infeasible))
    add("feasibility", "infeasible_rate", len(infeasible) / len(decisions) if decisions else None)

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

    fieldnames = list(rows[0].keys())

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate simple metadata-majority content policies against "
            "per-image R-D-E oracle."
        )
    )

    parser.add_argument("--benchmark-csv", required=True)
    parser.add_argument("--metadata-oracle-csv", required=True)

    parser.add_argument(
        "--policy-key",
        default="dataset",
        choices=["dataset", "resolution_class", "orientation_class"],
    )

    parser.add_argument(
        "--evaluation-mode",
        default="leave-one-out",
        choices=["in-sample", "leave-one-out"],
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
        "--decisions-out",
        default="results/routing_context/v09_metadata_policy_decisions.csv",
    )
    parser.add_argument(
        "--rules-out",
        default="results/routing_context/v09_metadata_policy_rules.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="results/routing_context/v09_metadata_policy_summary.csv",
    )

    args = parser.parse_args()

    metadata_rows = load_metadata_oracle_rows(args.metadata_oracle_csv)

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
        metadata_rows,
        global_baseline_codec=args.global_baseline_codec,
        global_baseline_config=args.global_baseline_config,
    )

    evaluation = evaluate_metadata_policy(
        metadata_oracle_rows=metadata_rows,
        candidate_lookup=candidate_lookup,
        policy_key=args.policy_key,
        quality_floor=args.quality_floor,
        global_baseline=global_baseline,
        evaluation_mode=args.evaluation_mode,
    )

    write_csv(args.decisions_out, evaluation["decisions"])
    write_csv(args.rules_out, evaluation["rules"])
    write_csv(args.summary_out, evaluation["summary"])

    summary = {
        (row["section"], row["key"]): row["value"]
        for row in evaluation["summary"]
    }

    print("\n=== R-D-E Metadata Content Policy ===")
    print(f"Policy key:             {args.policy_key}")
    print(f"Evaluation mode:        {args.evaluation_mode}")
    print(f"Images:                 {summary.get(('summary', 'num_images'))}")
    print(
        "Global baseline:        "
        f"{global_baseline[0]} {global_baseline[1]}"
    )
    print(f"Oracle match rate:      {summary.get(('accuracy', 'oracle_match_rate'))}")
    print(f"Mean regret:            {summary.get(('regret', 'mean'))}")
    print(f"Global mean regret:     {summary.get(('global_regret', 'mean'))}")
    print(f"Mean regret reduction:  {summary.get(('regret_reduction', 'mean'))}")
    print(f"Relative reduction:     {summary.get(('regret_reduction', 'relative_mean'))}")
    print(f"Fallback rate:          {summary.get(('fallback', 'fallback_rate'))}")
    print(f"Decisions CSV:          {args.decisions_out}")
    print(f"Rules CSV:              {args.rules_out}")
    print(f"Summary CSV:            {args.summary_out}")


if __name__ == "__main__":
    main()
