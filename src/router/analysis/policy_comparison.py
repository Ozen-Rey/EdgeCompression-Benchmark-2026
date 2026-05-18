"""Paper-facing comparison of content-aware routing policies with bootstrap CIs.

This module is offline-only analysis: it does not change the router
runtime, the ranking score, or the operational report schema. It
consumes the per-image artefacts already produced by the content-aware
pipeline (oracle by-image CSV, metadata-policy decisions CSV, and
optionally classifier LOIO/LODO decisions CSVs) and emits a single
comparison report with bootstrap confidence intervals for ``mean_regret``
and ``relative_reduction_vs_global``.

Pairing convention. For every non-baseline policy row, the relative
reduction is computed against the global baseline regret aligned on the
same image set. The per-image baseline regret is read from the
decisions CSV's own ``global_regret`` column (which the upstream policy
modules already populate from the oracle by-image analysis). The
baseline row's regret distribution is read directly from the oracle
by-image CSV's ``regret`` column on feasible images.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "PolicyResult",
    "load_oracle_by_image",
    "load_decisions_csv",
    "build_baseline_result",
    "build_oracle_result",
    "build_policy_result",
    "bootstrap_pair_ci",
    "policy_result_to_row",
    "write_outputs",
    "main",
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    q = max(0.0, min(1.0, q))
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------


def _read_csv(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_oracle_by_image(path: str) -> List[Dict[str, Any]]:
    """Load per-image oracle/baseline rows.

    Expected columns include ``image_id``, ``regret``, ``global_feasible``,
    ``oracle_cost``, ``global_cost``. Rows are returned unfiltered; the
    caller decides which subset to use.
    """
    raw = _read_csv(path)
    if not raw:
        raise ValueError(f"No rows loaded from {path}")
    out: List[Dict[str, Any]] = []
    for row in raw:
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        out.append(
            {
                "image_id": image_id,
                "dataset": row.get("dataset"),
                "image": row.get("image"),
                "regret": _to_float(row.get("regret")),
                "global_feasible": _to_bool(row.get("global_feasible")),
                "oracle_cost": _to_float(row.get("oracle_cost")),
                "global_cost": _to_float(row.get("global_cost")),
            }
        )
    if not out:
        raise ValueError(f"No usable image rows in {path}")
    return out


def load_decisions_csv(
    path: str,
    *,
    evaluation_mode_filter: Optional[str] = None,
    feature_set_filter: Optional[str] = None,
    k_filter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load per-image policy/classifier decisions rows.

    The optional filters apply only when the corresponding column exists
    in the input CSV. They let the same loader consume both single-run
    decision files and sweep decision files.
    """
    raw = _read_csv(path)
    if not raw:
        raise ValueError(f"No rows loaded from {path}")

    has_evaluation_mode = "evaluation_mode" in raw[0]
    has_feature_set = "feature_set" in raw[0]
    has_k = "k" in raw[0]

    out: List[Dict[str, Any]] = []
    for row in raw:
        if (
            evaluation_mode_filter is not None
            and has_evaluation_mode
            and str(row.get("evaluation_mode", "")).strip() != evaluation_mode_filter
        ):
            continue
        if (
            feature_set_filter is not None
            and has_feature_set
            and str(row.get("feature_set", "")).strip() != feature_set_filter
        ):
            continue
        if k_filter is not None and has_k:
            row_k = _to_float(row.get("k"))
            if row_k is None or int(row_k) != int(k_filter):
                continue

        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue

        out.append(
            {
                "image_id": image_id,
                "regret": _to_float(row.get("regret")),
                "global_regret": _to_float(row.get("global_regret")),
                "selected_feasible": _to_bool(row.get("selected_feasible")),
                "fallback_used": _to_bool(row.get("fallback_used")),
                "correct_oracle_match": _to_bool(row.get("correct_oracle_match")),
            }
        )

    if not out:
        raise ValueError(
            f"No decisions rows matched the requested filters in {path}"
        )
    return out


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_pair_ci(
    *,
    policy_regrets: List[float],
    baseline_regrets: List[float],
    iterations: int,
    seed: int,
    ci_low_q: float = 0.025,
    ci_high_q: float = 0.975,
) -> Dict[str, Optional[float]]:
    """Paired bootstrap on aligned (policy, baseline) per-image regrets.

    Resampling is done on image indices, so each resample uses the same
    indices for the policy and the baseline arrays — this is the correct
    way to bootstrap a *relative* reduction.

    Returns mean_regret CI bounds and relative_reduction CI bounds. If
    a resample produces a degenerate baseline mean (zero), that resample
    is dropped from the relative-reduction distribution; the CI is then
    computed on whatever fraction remains, and is None if none remain.
    """
    n = len(policy_regrets)
    if n == 0 or n != len(baseline_regrets):
        return {
            "mean_regret_ci_low": None,
            "mean_regret_ci_high": None,
            "relative_reduction_ci_low": None,
            "relative_reduction_ci_high": None,
        }

    rng = random.Random(seed)
    mean_samples: List[float] = []
    rel_samples: List[float] = []

    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_policy = [policy_regrets[i] for i in indices]
        sampled_baseline = [baseline_regrets[i] for i in indices]

        m_policy = mean(sampled_policy)
        m_baseline = mean(sampled_baseline)
        mean_samples.append(m_policy)

        if m_baseline > 0.0:
            rel_samples.append((m_baseline - m_policy) / m_baseline)

    return {
        "mean_regret_ci_low": _quantile(mean_samples, ci_low_q),
        "mean_regret_ci_high": _quantile(mean_samples, ci_high_q),
        "relative_reduction_ci_low": _quantile(rel_samples, ci_low_q),
        "relative_reduction_ci_high": _quantile(rel_samples, ci_high_q),
    }


# ---------------------------------------------------------------------------
# Policy result construction
# ---------------------------------------------------------------------------


class PolicyResult(Dict[str, Any]):
    """Dict-like row that carries the comparison columns for a single policy."""


_DEFAULT_PROVENANCE: Dict[str, str] = {}


def _empty_result(
    *,
    policy_name: str,
    protocol: str,
    notes: str,
) -> PolicyResult:
    return PolicyResult(
        policy_name=policy_name,
        protocol=protocol,
        num_images=0,
        mean_regret=None,
        median_regret=None,
        p90_regret=None,
        max_regret=None,
        relative_reduction_vs_global=None,
        coverage_rate=None,
        fallback_rate=None,
        quality_violations=None,
        mean_regret_ci_low=None,
        mean_regret_ci_high=None,
        relative_reduction_ci_low=None,
        relative_reduction_ci_high=None,
        notes=notes,
        provenance=dict(_DEFAULT_PROVENANCE),
    )


def _aggregate_regrets(regrets: List[float]) -> Dict[str, Optional[float]]:
    if not regrets:
        return {
            "mean_regret": None,
            "median_regret": None,
            "p90_regret": None,
            "max_regret": None,
        }
    return {
        "mean_regret": mean(regrets),
        "median_regret": median(regrets),
        "p90_regret": _quantile(regrets, 0.90),
        "max_regret": max(regrets),
    }


def build_baseline_result(
    oracle_rows: List[Dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> PolicyResult:
    """Build the robust-global-baseline row from the oracle by-image CSV."""
    feasible = [r for r in oracle_rows if r.get("regret") is not None]
    regrets = [float(r["regret"]) for r in feasible]
    n_total = len(oracle_rows)
    n_feasible = len(feasible)

    result = _empty_result(
        policy_name="robust_global_baseline",
        protocol="global_coverage_oracle_analysis",
        notes=(
            "Baseline regret distribution read from oracle by-image CSV "
            "(column 'regret'). Relative reduction vs. self is 0 by "
            "construction."
        ),
    )
    result["num_images"] = n_feasible
    result.update(_aggregate_regrets(regrets))
    result["relative_reduction_vs_global"] = 0.0 if regrets else None
    result["coverage_rate"] = (n_feasible / n_total) if n_total > 0 else None
    result["fallback_rate"] = None
    result["quality_violations"] = (
        (n_total - n_feasible) / n_total if n_total > 0 else None
    )

    if regrets:
        ci = bootstrap_pair_ci(
            policy_regrets=regrets,
            baseline_regrets=regrets,
            iterations=iterations,
            seed=seed,
        )
        result["mean_regret_ci_low"] = ci["mean_regret_ci_low"]
        result["mean_regret_ci_high"] = ci["mean_regret_ci_high"]
        result["relative_reduction_ci_low"] = 0.0
        result["relative_reduction_ci_high"] = 0.0

    result["provenance"] = {
        "regret_source": "oracle_by_image.regret",
        "fallback_rate": "not_applicable",
        "relative_reduction_vs_global": "self_reference_zero",
    }
    return result


def build_oracle_result(
    oracle_rows: List[Dict[str, Any]],
) -> PolicyResult:
    """Build the per-image oracle row. Regret is identically zero by construction."""
    feasible = [r for r in oracle_rows if r.get("regret") is not None]
    n_feasible = len(feasible)

    result = _empty_result(
        policy_name="per_image_oracle",
        protocol="oracle",
        notes=(
            "Per-image oracle: regret is identically zero by construction. "
            "Reported as the lower bound for the routing opportunity, not as "
            "an achievable policy."
        ),
    )
    result["num_images"] = n_feasible

    if n_feasible > 0:
        result["mean_regret"] = 0.0
        result["median_regret"] = 0.0
        result["p90_regret"] = 0.0
        result["max_regret"] = 0.0
        result["relative_reduction_vs_global"] = 1.0
        result["coverage_rate"] = 1.0
        result["quality_violations"] = 0.0
        result["mean_regret_ci_low"] = 0.0
        result["mean_regret_ci_high"] = 0.0
        result["relative_reduction_ci_low"] = 1.0
        result["relative_reduction_ci_high"] = 1.0

    result["provenance"] = {
        "regret_source": "constructive_zero",
        "fallback_rate": "not_applicable",
        "relative_reduction_vs_global": "constructive_one",
    }
    return result


def _pair_regrets(decisions: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    policy: List[float] = []
    baseline: List[float] = []
    for row in decisions:
        r = row.get("regret")
        g = row.get("global_regret")
        if r is None or g is None:
            continue
        policy.append(float(r))
        baseline.append(float(g))
    return policy, baseline


def build_policy_result(
    decisions: List[Dict[str, Any]],
    *,
    policy_name: str,
    protocol: str,
    iterations: int,
    seed: int,
    notes: str = "",
) -> PolicyResult:
    """Build a comparison row for a policy whose per-image decisions are known."""
    if not decisions:
        return _empty_result(
            policy_name=policy_name,
            protocol=protocol,
            notes=notes or "No decisions rows provided.",
        )

    policy_regrets, baseline_regrets = _pair_regrets(decisions)

    feasibility_known = [
        row for row in decisions if row.get("selected_feasible") is not None
    ]
    fallback_known = [
        row for row in decisions if row.get("fallback_used") is not None
    ]

    selected_feasible_true = sum(
        1 for row in feasibility_known if row.get("selected_feasible") is True
    )
    fallback_true = sum(
        1 for row in fallback_known if row.get("fallback_used") is True
    )

    n = len(decisions)
    coverage_rate = (
        selected_feasible_true / len(feasibility_known)
        if feasibility_known
        else None
    )
    fallback_rate = (
        fallback_true / len(fallback_known) if fallback_known else None
    )
    quality_violations = (
        (len(feasibility_known) - selected_feasible_true) / len(feasibility_known)
        if feasibility_known
        else None
    )

    result = _empty_result(
        policy_name=policy_name,
        protocol=protocol,
        notes=notes or (
            "Per-image regret read from decisions CSV; paired bootstrap "
            "against the per-image global baseline regret."
        ),
    )
    result["num_images"] = n
    result.update(_aggregate_regrets(policy_regrets))
    result["coverage_rate"] = coverage_rate
    result["fallback_rate"] = fallback_rate
    result["quality_violations"] = quality_violations

    if policy_regrets and baseline_regrets:
        m_policy = mean(policy_regrets)
        m_baseline = mean(baseline_regrets)
        if m_baseline > 0.0:
            result["relative_reduction_vs_global"] = (
                (m_baseline - m_policy) / m_baseline
            )

        ci = bootstrap_pair_ci(
            policy_regrets=policy_regrets,
            baseline_regrets=baseline_regrets,
            iterations=iterations,
            seed=seed,
        )
        result.update(ci)

    provenance: Dict[str, str] = {
        "regret_source": "decisions_csv.regret",
        "relative_reduction_vs_global": (
            "paired_against_decisions_csv.global_regret"
        ),
    }
    if not fallback_known:
        provenance["fallback_rate"] = "unavailable"
    if not feasibility_known:
        provenance["coverage_rate"] = "unavailable"
        provenance["quality_violations"] = "unavailable"
    result["provenance"] = provenance

    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


_CSV_COLUMNS = [
    "policy_name",
    "protocol",
    "num_images",
    "mean_regret",
    "median_regret",
    "p90_regret",
    "max_regret",
    "relative_reduction_vs_global",
    "coverage_rate",
    "fallback_rate",
    "quality_violations",
    "mean_regret_ci_low",
    "mean_regret_ci_high",
    "relative_reduction_ci_low",
    "relative_reduction_ci_high",
    "notes",
]


def policy_result_to_row(result: PolicyResult) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in _CSV_COLUMNS:
        value = result.get(col)
        if value is None:
            out[col] = ""
        elif isinstance(value, bool):
            out[col] = "true" if value else "false"
        elif isinstance(value, float):
            out[col] = f"{value:.10g}"
        else:
            out[col] = str(value)
    return out


def write_outputs(
    results: List[PolicyResult],
    *,
    out_csv: Optional[str],
    out_json: Optional[str],
    metadata: Dict[str, Any],
) -> None:
    if out_csv:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            for r in results:
                writer.writerow(policy_result_to_row(r))

    if out_json:
        json_path = Path(out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **metadata,
            "policies": [
                {col: r.get(col) for col in _CSV_COLUMNS}
                | {"provenance": r.get("provenance", {})}
                for r in results
            ],
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
            f.write("\n")


# ---------------------------------------------------------------------------
# Optional oracle-summary parsing for provenance
# ---------------------------------------------------------------------------


def _read_oracle_summary_provenance(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    rows = _read_csv(path)
    keys = {"quality_floor", "global_coverage_floor", "w_R", "w_E", "w_D"}
    out: Dict[str, Any] = {}
    for row in rows:
        section = str(row.get("section", "")).strip()
        key = str(row.get("key", "")).strip()
        if section == "summary" and key in keys:
            out[key] = _to_float(row.get("value"))
        if section == "global_best" and key in {"codec", "config"}:
            out[f"global_baseline_{key}"] = row.get("value")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.router.analysis.policy_comparison",
        description=(
            "Offline paper-facing comparison of content-aware routing "
            "policies with paired bootstrap confidence intervals. "
            "Reads per-image artefacts already produced by the "
            "content-aware pipeline; does not change the router runtime."
        ),
    )

    parser.add_argument(
        "--oracle-by-image",
        required=True,
        help=(
            "Path to v09_content_oracle_by_image.csv (or equivalent). "
            "Source of the robust-global-baseline regret distribution."
        ),
    )
    parser.add_argument(
        "--oracle-summary",
        default=None,
        help=(
            "Optional path to v09_content_oracle_summary.csv. Used only "
            "to populate the report's provenance block (quality floor, "
            "weights, global baseline pair); never used to invent missing "
            "per-image data."
        ),
    )
    parser.add_argument(
        "--metadata-policy-decisions",
        default=None,
        help=(
            "Optional path to a metadata-policy per-image decisions CSV "
            "(v09_metadata_policy_decisions.csv). If omitted, the "
            "source-aware policy row is not emitted."
        ),
    )
    parser.add_argument(
        "--classifier-loio-decisions",
        default=None,
        help=(
            "Optional path to a classifier per-image decisions CSV with "
            "LOIO rows. Accepts either a single-run file or a sweep file "
            "(in which case the 'evaluation_mode' / 'feature_set' / 'k' "
            "filters below select a single configuration)."
        ),
    )
    parser.add_argument(
        "--classifier-lodo-decisions",
        default=None,
        help="Same as --classifier-loio-decisions, for LODO rows.",
    )
    parser.add_argument(
        "--classifier-feature-set",
        default=None,
        help=(
            "When the classifier decisions file is a sweep, restrict to "
            "this feature_set. Ignored if the column is absent."
        ),
    )
    parser.add_argument(
        "--classifier-k",
        type=int,
        default=None,
        help=(
            "When the classifier decisions file is a sweep, restrict to "
            "this k. Ignored if the column is absent."
        ),
    )

    parser.add_argument(
        "--out-csv",
        default="results/routing_context/policy_comparison.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-json",
        default="results/routing_context/policy_comparison.json",
        help="Output JSON path.",
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of paired bootstrap resamples per policy (default 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for the bootstrap RNG (default 12345).",
    )

    return parser


def _run(args: argparse.Namespace) -> List[PolicyResult]:
    oracle_rows = load_oracle_by_image(args.oracle_by_image)

    results: List[PolicyResult] = [
        build_baseline_result(
            oracle_rows,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
    ]

    if args.metadata_policy_decisions:
        metadata_decisions = load_decisions_csv(
            args.metadata_policy_decisions,
        )
        results.append(
            build_policy_result(
                metadata_decisions,
                policy_name="source_aware_metadata_majority",
                protocol="leave-one-out",
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            )
        )

    if args.classifier_loio_decisions:
        loio_decisions = load_decisions_csv(
            args.classifier_loio_decisions,
            evaluation_mode_filter="leave_one_image_out",
            feature_set_filter=args.classifier_feature_set,
            k_filter=args.classifier_k,
        )
        results.append(
            build_policy_result(
                loio_decisions,
                policy_name="knn_metadata_loio",
                protocol="leave_one_image_out",
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            )
        )

    if args.classifier_lodo_decisions:
        lodo_decisions = load_decisions_csv(
            args.classifier_lodo_decisions,
            evaluation_mode_filter="leave_one_dataset_out",
            feature_set_filter=args.classifier_feature_set,
            k_filter=args.classifier_k,
        )
        results.append(
            build_policy_result(
                lodo_decisions,
                policy_name="knn_metadata_lodo",
                protocol="leave_one_dataset_out",
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            )
        )

    results.append(build_oracle_result(oracle_rows))
    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    results = _run(args)

    inputs_meta: Dict[str, Any] = {
        "oracle_by_image": args.oracle_by_image,
        "oracle_summary": args.oracle_summary,
        "metadata_policy_decisions": args.metadata_policy_decisions,
        "classifier_loio_decisions": args.classifier_loio_decisions,
        "classifier_lodo_decisions": args.classifier_lodo_decisions,
        "classifier_feature_set": args.classifier_feature_set,
        "classifier_k": args.classifier_k,
    }
    inputs_meta = {k: v for k, v in inputs_meta.items() if v is not None}

    metadata: Dict[str, Any] = {
        "generated_at_unix": int(time.time()),
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "ci_low_quantile": 0.025,
        "ci_high_quantile": 0.975,
        "inputs": inputs_meta,
        "oracle_summary_provenance": _read_oracle_summary_provenance(
            args.oracle_summary
        ),
    }

    write_outputs(
        results,
        out_csv=args.out_csv,
        out_json=args.out_json,
        metadata=metadata,
    )

    print("\n=== R-D-E Policy Comparison ===")
    print(f"Bootstrap iterations: {args.bootstrap_iterations}")
    print(f"Seed:                 {args.seed}")
    for r in results:
        rel = r.get("relative_reduction_vs_global")
        rel_low = r.get("relative_reduction_ci_low")
        rel_high = r.get("relative_reduction_ci_high")
        mean_r = r.get("mean_regret")
        mean_low = r.get("mean_regret_ci_low")
        mean_high = r.get("mean_regret_ci_high")
        rel_str = (
            f"{rel:.4f} [{rel_low:.4f}, {rel_high:.4f}]"
            if rel is not None and rel_low is not None and rel_high is not None
            else "--"
        )
        mean_str = (
            f"{mean_r:.5f} [{mean_low:.5f}, {mean_high:.5f}]"
            if mean_r is not None
            and mean_low is not None
            and mean_high is not None
            else "--"
        )
        print(
            f"  {r['policy_name']:<36} N={r['num_images']:<4} "
            f"mean_regret={mean_str}  rel_reduction={rel_str}"
        )
    print(f"CSV:  {args.out_csv}")
    print(f"JSON: {args.out_json}")


if __name__ == "__main__":
    main()
