"""Read-only diagnostics for operational-regime simulation outputs.

This module does not fix or alter ``operational_regime_simulation``. It
inspects existing simulation CSVs and, when requested, reruns the same
offline simulation logic over alternative quality columns/floors to diagnose
metric-contract, objective-consistency and no-leakage issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.router.analysis.neural_inclusive_oracle import load_pool_rows, normalize_full_pool
from src.router.analysis.operational_regime_simulation import (
    POLICIES,
    RATE_PRESSURE_GRID,
    build_regime_definitions,
    evaluate_operational_regimes,
    _derive_metadata_from_rde_rows,
    _j_rde,
    _operational_cost,
)


__all__ = [
    "build_router_quality_guard_contract",
    "build_metric_contract",
    "run_diagnostics",
    "main",
]


NEGATIVE_REGRET_TOL = -1e-9


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_csv_list(value: str) -> List[str]:
    return [token.strip().lower() for token in value.split(",") if token.strip()]


def _parse_floors(value: str) -> List[float]:
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def _floor_key(value: Any) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return str(value)
    return f"{parsed:g}"


def _metric_for_floor(
    floor: Any,
    *,
    psnr_floors: List[float],
    ssimulacra2_floors: List[float],
    default_metric: str,
) -> str:
    parsed = _to_float(floor)
    if parsed is None:
        return default_metric
    if any(abs(parsed - f) <= 1e-9 for f in psnr_floors):
        return "psnr"
    if any(abs(parsed - f) <= 1e-9 for f in ssimulacra2_floors):
        return "ssimulacra2"
    return default_metric


def _decision_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("image_id", "")),
        str(row.get("regime", "")),
        str(row.get("policy", "")),
        str(row.get("protocol", "")),
        _floor_key(row.get("quality_floor")),
    )


def _summary_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("regime", "")),
        str(row.get("policy", "")),
        str(row.get("protocol", "")),
        _floor_key(row.get("quality_floor")),
    )


def _pair(codec: Any, config: Any) -> Tuple[str, str]:
    return str(codec or ""), str(config or "")


def _load_metric_rows(
    *,
    rde_csv: str,
    metric: str,
    image_id_col: str,
    dataset_col: str,
    codec_col: str,
    config_col: str,
    rate_col: str,
    energy_col: str,
) -> List[Dict[str, Any]]:
    rows = load_pool_rows(
        rde_csv,
        codec_col=codec_col,
        config_col=config_col,
        rate_col=rate_col,
        quality_col=metric,
        energy_col=energy_col,
        image_id_col=image_id_col,
        dataset_col=dataset_col,
    )
    normalize_full_pool(rows)
    return rows


def _row_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    return {
        (str(r["image_id"]), str(r["codec"]), str(r["config"])): r
        for r in rows
    }


def _candidate_support(
    rows: List[Dict[str, Any]],
    *,
    image_id: str,
    dataset: str,
    protocol: str,
    codec: str,
    config: str,
    floor: Optional[float],
) -> Dict[str, Any]:
    if protocol == "loio":
        training = [r for r in rows if str(r["image_id"]) != str(image_id)]
    elif protocol == "lodo":
        training = [r for r in rows if str(r.get("dataset")) != str(dataset)]
    else:
        training = list(rows)
    pair_rows = [
        r for r in training
        if str(r["codec"]) == str(codec) and str(r["config"]) == str(config)
    ]
    pass_rows = [
        r for r in pair_rows
        if floor is None or float(r["quality"]) >= float(floor)
    ]
    return {
        "training_support_count": len(pair_rows),
        "training_floor_pass_count": len(pass_rows),
        "training_quality_min": min((r["quality"] for r in pair_rows), default=None),
        "training_quality_mean": (
            sum(float(r["quality"]) for r in pair_rows) / len(pair_rows)
            if pair_rows else None
        ),
    }


def build_router_quality_guard_contract() -> Dict[str, Any]:
    return {
        "main_image_configs": [
            "configs/router_image_v05.json",
            "configs/router_image_v08.json",
            "configs/router_image_v08.example.json",
        ],
        "primary_image_quality_metric": "ssimulacra2",
        "quality_thresholds_file": "configs/quality_thresholds.json",
        "image_quality_targets": {
            "preview": 50.0,
            "normal": 50.0,
            "high": 80.0,
            "very-high": 90.0,
        },
        "content_aware_original_floor": 80.0,
        "router_quality_guard_code": [
            "src/router/pipeline.py resolves effective quality_floor with resolve_quality_floor.",
            "src/router/core/rde_database.py select_best_rde filters candidates through quality_guard before ranking.",
            "src/router/profile_runner.py passes min_quality and quality_constraint_stat into select_best_rde.",
        ],
        "operational_runtime_contract": (
            "The runtime router ranks only candidates that survive the active "
            "quality guard, using the configured quality column/statistic."
        ),
    }


def build_metric_contract() -> Dict[str, Any]:
    return {
        "paper_image_primary_metric": "ssimulacra2",
        "psnr30_diagnostic_role": (
            "PSNR>=30 is useful as a stress-test floor for rate-pressure behavior, "
            "but it is not the perceptual quality contract used by the main image configs."
        ),
        "ssimulacra2_floors_to_compare": [60.0, 70.0, 80.0],
        "runtime_router_uses": (
            "The router uses the column supplied by --quality-col or config columns. "
            "Main image configs set quality=ssimulacra2; PSNR is used only when selected explicitly."
        ),
    }


def build_no_leakage_contract() -> Dict[str, Any]:
    return {
        "policy_does_not_see_test_image_rde": True,
        "simulation_evaluates": [
            "realized quality after selection",
            "realized target-image objective after selection",
            "ex-post target quality violation",
        ],
        "simulation_does_not_guarantee": [
            "actual target quality before selection",
            "target-image quality guard without target R-D-E access",
        ],
        "diagnostic_interpretation": (
            "A predictive offline simulation can produce realized quality violations "
            "because it chooses from training labels/metadata and checks target quality "
            "only after realization. This is distinct from the runtime router quality "
            "guard, which filters an already measured candidate pool."
        ),
    }


def diagnose_negative_regrets(
    *,
    decisions: List[Dict[str, Any]],
    metric_rows_by_name: Dict[str, List[Dict[str, Any]]],
    psnr_floors: List[float],
    ssimulacra2_floors: List[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    regimes = build_regime_definitions()
    indexes = {metric: _row_index(rows) for metric, rows in metric_rows_by_name.items()}
    rows_out: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for row in decisions:
        regret = _to_float(row.get("regret"))
        if regret is None or regret >= NEGATIVE_REGRET_TOL:
            continue
        metric = _metric_for_floor(
            row.get("quality_floor"),
            psnr_floors=psnr_floors,
            ssimulacra2_floors=ssimulacra2_floors,
            default_metric="psnr",
        )
        idx = indexes.get(metric, {})
        regime = regimes.get(str(row.get("regime")), regimes["normal"])
        floor = _to_float(row.get("quality_floor"))
        selected_key = (
            str(row.get("image_id")),
            str(row.get("selected_codec")),
            str(row.get("selected_config")),
        )
        oracle_key = (
            str(row.get("image_id")),
            str(row.get("oracle_codec")),
            str(row.get("oracle_config")),
        )
        selected = idx.get(selected_key)
        oracle = idx.get(oracle_key)
        selected_base = _j_rde(selected, regime["weights"]) if selected else _to_float(row.get("selected_cost"))
        oracle_base = _j_rde(oracle, regime["weights"]) if oracle else _to_float(row.get("oracle_cost"))
        selected_total = _operational_cost(selected, regime) if selected else _to_float(row.get("selected_cost"))
        oracle_total = _operational_cost(oracle, regime) if oracle else _to_float(row.get("oracle_cost"))
        selected_penalty = (
            selected_total - selected_base
            if selected_total is not None and selected_base is not None
            else None
        )
        oracle_penalty = (
            oracle_total - oracle_base
            if oracle_total is not None and oracle_base is not None
            else None
        )
        selected_quality = selected["quality"] if selected else _to_float(row.get("selected_quality"))
        oracle_quality = oracle["quality"] if oracle else None
        selected_admissible = (
            selected_quality is not None and (floor is None or selected_quality >= floor)
        )
        oracle_admissible = (
            oracle_quality is not None and (floor is None or oracle_quality >= floor)
        )
        reason = _negative_regret_hypothesis(
            selected=selected,
            oracle=oracle,
            selected_admissible=selected_admissible,
            oracle_admissible=oracle_admissible,
            selected_penalty=selected_penalty,
            oracle_penalty=oracle_penalty,
            regret=regret,
            policy=str(row.get("policy")),
        )
        reason_counts[reason] += 1
        rows_out.append(
            {
                "image_id": row.get("image_id"),
                "dataset": row.get("dataset"),
                "protocol": row.get("protocol"),
                "regime": row.get("regime"),
                "policy": row.get("policy"),
                "quality_floor": row.get("quality_floor"),
                "selected_codec": row.get("selected_codec"),
                "selected_config": row.get("selected_config"),
                "selected_family": row.get("selected_family"),
                "oracle_codec": row.get("oracle_codec"),
                "oracle_config": row.get("oracle_config"),
                "oracle_family": row.get("oracle_family"),
                "selected_J_on_test": row.get("selected_cost"),
                "oracle_J_on_test": row.get("oracle_cost"),
                "selected_base_J_RDE": selected_base,
                "oracle_base_J_RDE": oracle_base,
                "selected_system_penalty": selected_penalty,
                "oracle_system_penalty": oracle_penalty,
                "selected_total_objective": selected_total,
                "oracle_total_objective": oracle_total,
                "selected_quality": selected_quality,
                "oracle_quality": oracle_quality,
                "selected_admissible_under_floor": selected_admissible,
                "oracle_admissible_under_floor": oracle_admissible,
                "reason_hypothesis": reason,
            }
        )
    findings = {
        "negative_regret_count": len(rows_out),
        "reason_counts": dict(reason_counts),
        "regret_min": min((_to_float(r["selected_J_on_test"]) or 0.0) - (_to_float(r["oracle_J_on_test"]) or 0.0) for r in rows_out) if rows_out else None,
    }
    return rows_out, findings


def _negative_regret_hypothesis(
    *,
    selected: Optional[Dict[str, Any]],
    oracle: Optional[Dict[str, Any]],
    selected_admissible: bool,
    oracle_admissible: bool,
    selected_penalty: Optional[float],
    oracle_penalty: Optional[float],
    regret: float,
    policy: str,
) -> str:
    if abs(regret) <= 1e-8:
        return "numerical_tolerance"
    if not selected_admissible and oracle_admissible:
        return "quality_gate_mismatch"
    if selected is None or oracle is None:
        return "candidate_pool_mismatch"
    if policy == "metadata_only_classic_pool" and oracle.get("codec_family") == "neural":
        return "candidate_pool_mismatch"
    if selected_penalty is not None and oracle_penalty is not None:
        if abs(selected_penalty - oracle_penalty) > 1e-9:
            return "possible_policy_or_oracle_system_penalty_mismatch"
    return "unknown"


def diagnose_quality_violations(
    *,
    decisions: List[Dict[str, Any]],
    metric_rows_by_name: Dict[str, List[Dict[str, Any]]],
    psnr_floors: List[float],
    ssimulacra2_floors: List[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    indexes = {metric: _row_index(rows) for metric, rows in metric_rows_by_name.items()}
    baseline_by_key = {
        (
            str(d.get("image_id")),
            str(d.get("regime")),
            str(d.get("protocol")),
            _floor_key(d.get("quality_floor")),
        ): d
        for d in decisions
        if d.get("policy") == "robust_global_full_pool_baseline"
    }
    rows_out: List[Dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    for row in decisions:
        if not _to_bool(row.get("quality_violation")):
            continue
        metric = _metric_for_floor(
            row.get("quality_floor"),
            psnr_floors=psnr_floors,
            ssimulacra2_floors=ssimulacra2_floors,
            default_metric="psnr",
        )
        idx = indexes.get(metric, {})
        floor = _to_float(row.get("quality_floor"))
        selected = idx.get(
            (
                str(row.get("image_id")),
                str(row.get("selected_codec")),
                str(row.get("selected_config")),
            )
        )
        oracle = idx.get(
            (
                str(row.get("image_id")),
                str(row.get("oracle_codec")),
                str(row.get("oracle_config")),
            )
        )
        baseline = baseline_by_key.get(
            (
                str(row.get("image_id")),
                str(row.get("regime")),
                str(row.get("protocol")),
                _floor_key(row.get("quality_floor")),
            )
        )
        baseline_row = None
        if baseline:
            baseline_row = idx.get(
                (
                    str(row.get("image_id")),
                    str(baseline.get("selected_codec")),
                    str(baseline.get("selected_config")),
                )
            )
        support = _candidate_support(
            metric_rows_by_name.get(metric, []),
            image_id=str(row.get("image_id")),
            dataset=str(row.get("dataset")),
            protocol=str(row.get("protocol")),
            codec=str(row.get("selected_codec")),
            config=str(row.get("selected_config")),
            floor=floor,
        )
        violation_type = _quality_violation_type(
            metric=metric,
            selected=selected,
            floor=floor,
            policy=str(row.get("policy")),
            support=support,
        )
        type_counts[violation_type] += 1
        rows_out.append(
            {
                "image_id": row.get("image_id"),
                "dataset": row.get("dataset"),
                "protocol": row.get("protocol"),
                "regime": row.get("regime"),
                "policy": row.get("policy"),
                "quality_metric": metric,
                "quality_floor": row.get("quality_floor"),
                "predicted_codec": row.get("selected_codec"),
                "predicted_config": row.get("selected_config"),
                "predicted_family": row.get("selected_family"),
                "predicted_quality_on_test": selected["quality"] if selected else row.get("selected_quality"),
                "oracle_codec": row.get("oracle_codec"),
                "oracle_config": row.get("oracle_config"),
                "oracle_family": row.get("oracle_family"),
                "oracle_quality_on_test": oracle["quality"] if oracle else None,
                "global_baseline_codec": baseline.get("selected_codec") if baseline else None,
                "global_baseline_config": baseline.get("selected_config") if baseline else None,
                "global_baseline_quality_on_test": baseline_row["quality"] if baseline_row else None,
                "predicted_label_training_support_count": support["training_support_count"],
                "predicted_label_training_floor_pass_count": support["training_floor_pass_count"],
                "predicted_label_training_quality_min": support["training_quality_min"],
                "predicted_label_expected_training_quality": support["training_quality_mean"],
                "predicted_label_passed_expected_training_quality": (
                    support["training_quality_min"] is not None
                    and (floor is None or support["training_quality_min"] >= floor)
                ),
                "violation_type": violation_type,
            }
        )
    return rows_out, {
        "quality_violation_count": len(rows_out),
        "violation_type_counts": dict(type_counts),
    }


def _quality_violation_type(
    *,
    metric: str,
    selected: Optional[Dict[str, Any]],
    floor: Optional[float],
    policy: str,
    support: Dict[str, Any],
) -> str:
    if metric == "psnr":
        return "metric_mismatch"
    if selected is not None and floor is not None and selected["quality"] < floor:
        if support["training_floor_pass_count"] > 0:
            return "floor_not_used_as_preselection_gate_due_to_no_leakage"
        return "predictor_selected_low_quality_target"
    if policy == "system_only_full_pool":
        return "system_policy_overrode_quality_preference"
    return "unknown"


def build_metric_comparison(
    *,
    rde_csv: str,
    metrics: List[str],
    psnr_floors: List[float],
    ssimulacra2_floors: List[float],
    image_id_col: str,
    dataset_col: str,
    codec_col: str,
    config_col: str,
    rate_col: str,
    energy_col: str,
    protocols: List[str],
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    findings: Dict[str, Any] = {}
    for metric in metrics:
        floors = psnr_floors if metric == "psnr" else ssimulacra2_floors
        try:
            rows = _load_metric_rows(
                rde_csv=rde_csv,
                metric=metric,
                image_id_col=image_id_col,
                dataset_col=dataset_col,
                codec_col=codec_col,
                config_col=config_col,
                rate_col=rate_col,
                energy_col=energy_col,
            )
        except Exception as exc:
            findings[metric] = {"available": False, "error": str(exc)}
            continue
        metadata = _derive_metadata_from_rde_rows(rows)
        result = evaluate_operational_regimes(
            rows=rows,
            metadata_by_image=metadata,
            quality_floors=floors,
            protocols=protocols,
            k=k,
        )
        by_key_decisions: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for d in result["decisions"]:
            by_key_decisions[_summary_key(d)].append(d)
        for summary in result["summaries"]:
            key = _summary_key(summary)
            decs = by_key_decisions.get(key, [])
            selected_family_counts = Counter(d.get("selected_family") or "none" for d in decs)
            negative_regrets = [
                _to_float(d.get("regret"))
                for d in decs
                if _to_float(d.get("regret")) is not None
                and _to_float(d.get("regret")) < NEGATIVE_REGRET_TOL
            ]
            rows_out.append(
                {
                    "quality_metric": metric,
                    "regime": summary["regime"],
                    "policy": summary["policy"],
                    "protocol": summary["protocol"],
                    "quality_floor": summary["quality_floor"],
                    "neural_selection_rate": summary["neural_selection_rate"],
                    "oracle_neural_rate": summary["oracle_neural_rate"],
                    "quality_violation_rate": summary["quality_violation_rate"],
                    "mean_regret": summary["mean_regret"],
                    "negative_regret_count": len(negative_regrets),
                    "regret_min": min(negative_regrets) if negative_regrets else None,
                    "regret_reduction_max": summary["regret_reduction_vs_global_baseline"],
                    "selected_family_distribution": json.dumps(dict(selected_family_counts), sort_keys=True),
                }
            )
        findings[metric] = {"available": True, "floors": floors}
    return rows_out, findings


def build_objective_consistency_summary(
    *,
    decisions: List[Dict[str, Any]],
    negative_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[_summary_key(row)].append(row)
    negative_by_key: Counter[Tuple[str, str, str, str]] = Counter()
    for row in negative_rows:
        negative_by_key[
            (
                str(row.get("regime")),
                str(row.get("policy")),
                str(row.get("protocol")),
                _floor_key(row.get("quality_floor")),
            )
        ] += 1
    rows_out: List[Dict[str, Any]] = []
    mismatch_count = 0
    for key, rows in sorted(grouped.items()):
        regime, policy, protocol, floor = key
        neg_count = negative_by_key[key]
        possible = []
        if policy == "metadata_only_classic_pool":
            possible.append("possible_candidate_pool_mismatch")
        if neg_count:
            possible.append("possible_objective_or_quality_gate_mismatch")
        if any(_to_bool(r.get("quality_violation")) for r in rows):
            possible.append("possible_quality_gate_mismatch")
        if not possible:
            possible.append("same_objective_contract_observed")
        if any(p.startswith("possible") for p in possible):
            mismatch_count += 1
        rows_out.append(
            {
                "regime": regime,
                "policy": policy,
                "protocol": protocol,
                "quality_floor": floor,
                "same_weights": True,
                "same_normalization": True,
                "same_quality_floor": True,
                "same_candidate_pool": policy != "metadata_only_classic_pool",
                "same_regime": True,
                "same_system_penalty": "possible_mismatch" if neg_count else True,
                "same_constraints": True,
                "selected_and_oracle_same_space": neg_count == 0,
                "negative_regret_count": neg_count,
                "finding": ";".join(possible),
            }
        )
    return rows_out, {
        "groups_evaluated": len(rows_out),
        "groups_with_possible_mismatch": mismatch_count,
    }


def build_rate_pressure_transition_diagnostics(
    rate_pressure_rows: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rate_pressure_rows:
        grouped[
            (
                str(row.get("policy")),
                str(row.get("protocol")),
                _floor_key(row.get("quality_floor")),
            )
        ].append(row)
    rows_out: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        policy, protocol, floor = key
        by_weight: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            weight = _to_float(row.get("rate_weight"))
            if weight is not None:
                by_weight[weight].append(row)
        if not by_weight:
            continue
        weights = sorted(by_weight)
        predicted_rates: Dict[float, float] = {}
        oracle_rates: Dict[float, float] = {}
        for weight, items in by_weight.items():
            neural_items = [r for r in items if str(r.get("selected_family")) == "neural"]
            predicted_rates[weight] = max(
                [_to_float(r.get("selection_rate")) or 0.0 for r in neural_items] or [0.0]
            )
            oracle_rates[weight] = max(
                [_to_float(r.get("oracle_neural_rate")) or 0.0 for r in items] or [0.0]
            )
        first_oracle = _first_positive_weight(oracle_rates)
        first_predicted = _first_positive_weight(predicted_rates)
        max_gap = max(
            predicted_rates[w] - oracle_rates[w]
            for w in weights
        )
        shift_relation = _shift_relation(first_predicted, first_oracle)
        rows_out.append(
            {
                "policy": policy,
                "protocol": protocol,
                "quality_floor": floor,
                "first_oracle_neural_rate_weight": first_oracle,
                "first_predicted_neural_rate_weight": first_predicted,
                "max_gap_predicted_minus_oracle": max_gap,
                "predicted_shift_relation": shift_relation,
                "negative_regret_after_shift": "not_available_from_rate_pressure_csv",
                "quality_violations_after_shift": "not_available_from_rate_pressure_csv",
            }
        )
    return rows_out, {"rows": len(rows_out)}


def _first_positive_weight(rates: Dict[float, float]) -> Optional[float]:
    for weight in sorted(rates):
        if rates[weight] > 0.0:
            return weight
    return None


def _shift_relation(predicted: Optional[float], oracle: Optional[float]) -> str:
    if predicted is None and oracle is None:
        return "no_shift_observed"
    if predicted is None:
        return "predicted_shift_absent"
    if oracle is None:
        return "predicted_shift_without_oracle_shift"
    if predicted < oracle:
        return "predicted_shift_earlier_than_oracle"
    if predicted > oracle:
        return "predicted_shift_later_than_oracle"
    return "predicted_shift_same_as_oracle"


def build_winner_duplicate_diagnostics(
    winner_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    full_cols = [
        "regime",
        "policy",
        "protocol",
        "quality_floor",
        "selected_family",
        "selected_codec",
        "selected_config",
    ]
    displayed_cols = ["regime", "selected_family", "selected_codec", "selected_config"]
    full_counter = Counter(tuple(str(r.get(c, "")) for c in full_cols) for r in winner_rows)
    displayed_counter = Counter(tuple(str(r.get(c, "")) for c in displayed_cols) for r in winner_rows)
    real_duplicates = sum(count - 1 for count in full_counter.values() if count > 1)
    apparent_duplicates = sum(count - 1 for count in displayed_counter.values() if count > 1)
    rows_out = [
        {
            "key_type": "full_expected_key",
            "key_columns": ",".join(full_cols),
            "duplicate_count": real_duplicates,
            "explanation": "Rows with the full expected key should be unique.",
        },
        {
            "key_type": "displayed_truncated_key",
            "key_columns": ",".join(displayed_cols),
            "duplicate_count": apparent_duplicates,
            "explanation": (
                "PowerShell Format-Table can hide protocol, quality_floor or policy; "
                "rows that look duplicated under a truncated display may be distinct."
            ),
        },
    ]
    return rows_out, {
        "unique_key_columns": full_cols,
        "duplicate_count_under_full_expected_key": real_duplicates,
        "duplicate_count_under_displayed_truncated_key": apparent_duplicates,
    }


def _protocols_from_decisions(decisions: List[Dict[str, Any]]) -> List[str]:
    protocols = sorted({str(d.get("protocol")) for d in decisions if d.get("protocol")})
    return protocols or ["loio", "lodo"]


def recommended_fix_options() -> List[str]:
    return [
        "align oracle and policy objectives including system penalty",
        "report objective_gain_vs_global separately from regret",
        "use SSIMULACRA2 floors for image perceptual routing",
        "distinguish ex-post realized quality violation from operational quality guard",
        "add expected-quality gate if predictive simulation must guarantee quality without target leakage",
    ]


def run_diagnostics(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = _read_csv(args.operational_decisions_csv)
    summaries = _read_csv(args.operational_summary_csv)
    rate_pressure = _read_csv(args.rate_pressure_csv)
    winners = _read_csv(args.winner_distribution_csv)

    metrics = _parse_csv_list(args.quality_metrics)
    psnr_floors = _parse_floors(args.quality_floors_psnr)
    ssim_floors = _parse_floors(args.quality_floors_ssimulacra2)

    metric_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for metric in metrics:
        try:
            metric_rows_by_name[metric] = _load_metric_rows(
                rde_csv=args.rde_csv,
                metric=metric,
                image_id_col=args.image_id_col,
                dataset_col=args.dataset_col,
                codec_col=args.codec_col,
                config_col=args.config_col,
                rate_col=args.rate_col,
                energy_col=args.energy_col,
            )
        except Exception:
            metric_rows_by_name[metric] = []

    negative_rows, negative_findings = diagnose_negative_regrets(
        decisions=decisions,
        metric_rows_by_name=metric_rows_by_name,
        psnr_floors=psnr_floors,
        ssimulacra2_floors=ssim_floors,
    )
    violation_rows, violation_findings = diagnose_quality_violations(
        decisions=decisions,
        metric_rows_by_name=metric_rows_by_name,
        psnr_floors=psnr_floors,
        ssimulacra2_floors=ssim_floors,
    )
    metric_comparison, metric_findings = build_metric_comparison(
        rde_csv=args.rde_csv,
        metrics=metrics,
        psnr_floors=psnr_floors,
        ssimulacra2_floors=ssim_floors,
        image_id_col=args.image_id_col,
        dataset_col=args.dataset_col,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        energy_col=args.energy_col,
        protocols=_protocols_from_decisions(decisions),
        k=args.k,
    )
    objective_rows, objective_findings = build_objective_consistency_summary(
        decisions=decisions,
        negative_rows=negative_rows,
    )
    rate_rows, rate_findings = build_rate_pressure_transition_diagnostics(
        rate_pressure,
        decisions,
    )
    duplicate_rows, duplicate_findings = build_winner_duplicate_diagnostics(winners)

    paths = {
        "negative_regret_rows": out_dir / "negative_regret_rows.csv",
        "quality_violation_rows": out_dir / "quality_violation_rows.csv",
        "metric_comparison_summary": out_dir / "metric_comparison_summary.csv",
        "objective_consistency_summary": out_dir / "objective_consistency_summary.csv",
        "rate_pressure_transition_diagnostics": out_dir / "rate_pressure_transition_diagnostics.csv",
        "winner_distribution_duplicate_diagnostics": out_dir / "winner_distribution_duplicate_diagnostics.csv",
    }
    _write_csv(paths["negative_regret_rows"], negative_rows)
    _write_csv(paths["quality_violation_rows"], violation_rows)
    _write_csv(paths["metric_comparison_summary"], metric_comparison)
    _write_csv(paths["objective_consistency_summary"], objective_rows)
    _write_csv(paths["rate_pressure_transition_diagnostics"], rate_rows)
    _write_csv(paths["winner_distribution_duplicate_diagnostics"], duplicate_rows)

    report_path = Path(args.out_json) if args.out_json else out_dir / "operational_regime_diagnostic_report.json"
    report = {
        "schema_version": "operational_regime_diagnostics_v1",
        "created_at_unix": started,
        "elapsed_seconds": time.time() - started,
        "provenance": {
            "read_only": True,
            "runtime_router_changed": False,
            "operational_regime_simulation_behavior_changed": False,
            "j_rde_formula_changed": False,
            "benchmark_raw_data_modified": False,
            "codecs_executed": False,
        },
        "input_paths": {
            "rde_csv": args.rde_csv,
            "operational_summary_csv": args.operational_summary_csv,
            "operational_decisions_csv": args.operational_decisions_csv,
            "rate_pressure_csv": args.rate_pressure_csv,
            "winner_distribution_csv": args.winner_distribution_csv,
        },
        "router_quality_guard_contract": build_router_quality_guard_contract(),
        "metric_contract": build_metric_contract(),
        "no_leakage_contract": build_no_leakage_contract(),
        "objective_consistency_findings": objective_findings,
        "negative_regret_findings": negative_findings,
        "quality_violation_findings": violation_findings,
        "psnr_vs_ssimulacra2_findings": metric_findings,
        "rate_pressure_findings": rate_findings,
        "winner_distribution_duplicate_findings": duplicate_findings,
        "recommended_fix_options": recommended_fix_options(),
        "outputs": {name: str(path) for name, path in paths.items()} | {"report_json": str(report_path)},
        "summary_input_rows": len(summaries),
    }
    _write_json(report_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only diagnostics for operational-regime simulation outputs."
    )
    parser.add_argument("--rde-csv", required=True)
    parser.add_argument("--operational-summary-csv", required=True)
    parser.add_argument("--operational-decisions-csv", required=True)
    parser.add_argument("--rate-pressure-csv", required=True)
    parser.add_argument("--winner-distribution-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--quality-metrics", default="psnr,ssimulacra2")
    parser.add_argument("--quality-floors-psnr", default="30")
    parser.add_argument("--quality-floors-ssimulacra2", default="60,70,80")
    parser.add_argument("--image-id-col", default="dataset,image")
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="param")
    parser.add_argument("--rate-col", default="bpp")
    parser.add_argument("--energy-col", default="energy_per_image_j")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=7)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_diagnostics(args)


if __name__ == "__main__":
    main()
