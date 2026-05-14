"""Read-only audit of router effectiveness against simple baseline policies."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from src.router.core.rde_database import RDEPoint, load_rde_points
from src.router.rde_router import main as router_main
from src.router.core.router_config import load_router_config
from src.router.version import ROUTER_VERSION


SUMMARY_FIELDS = [
    "scenario",
    "scenario_type",
    "data_source",
    "router_codec",
    "router_config",
    "router_cost",
    "num_candidates_total",
    "num_candidates_admissible",
    "num_candidates_comparable",
    "num_quality_guard_violations",
    "num_base_constraint_violations",
    "num_policies",
    "num_comparable_policies",
    "num_changed_vs_router",
    "notes",
]

POLICY_FIELDS = [
    "scenario",
    "scenario_type",
    "policy",
    "is_router_decision",
    "candidate_source",
    "cost_status",
    "cost_reason_detail",
    "comparable",
    "reason",
    "selected_codec",
    "selected_config",
    "decision_changed_vs_router",
    "rate",
    "quality",
    "energy",
    "time_ms",
    "cost",
    "ranking_cost",
    "regret",
    "router_relative_improvement_percent",
    "constraint_violations",
    "quality_guard_violations",
    "notes",
]


@dataclass
class Candidate:
    codec: str
    config: str
    rate: float
    quality: float
    energy: float
    time_ms: Optional[float]
    quality_constraint_value: float
    cost: Optional[float] = None
    ranking_cost: Optional[float] = None
    raw: dict[str, Any] | None = None
    source: str = "csv"

    @property
    def key(self) -> tuple[str, str]:
        return self.codec, self.config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _contains_flag(argv: Iterable[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def _validate_read_only_request(
    *,
    config_path: str | None,
    router_args: list[str],
) -> None:
    forbidden_flags = [
        "--execute",
        "--all-profiles",
        "--feedback-out",
    ]
    for flag in forbidden_flags:
        if _contains_flag(router_args, flag):
            raise ValueError(
                f"router_effectiveness_audit is read-only and does not allow {flag}."
            )

    if not config_path:
        return

    config = load_router_config(config_path)

    if (
        isinstance(config.get("execution"), dict)
        and config["execution"].get("execute") is True
    ):
        raise ValueError(
            "router_effectiveness_audit is read-only and cannot use a config "
            "with execution.execute=true."
        )

    if (
        isinstance(config.get("selection"), dict)
        and config["selection"].get("all_profiles") is True
    ):
        raise ValueError(
            "router_effectiveness_audit requires single-decision runs; "
            "selection.all_profiles=true is not supported."
        )


def _base_router_args(
    *,
    csv_path: str,
    config_path: str | None,
    router_args: list[str],
    audit_top_k: int,
) -> list[str]:
    args: list[str] = []
    if config_path:
        args.extend(["--config", config_path])
    args.extend(["--csv", csv_path])
    args.extend(router_args)
    args.extend(["--top-k", str(max(int(audit_top_k), 1))])
    return args


def _mode_router_args(
    *,
    base_args: list[str],
    bundle_manifest: str | None = None,
    bundle_validation: str | None = None,
) -> list[str]:
    args = list(base_args)
    if bundle_manifest:
        args.extend(["--calibration-bundle-manifest", bundle_manifest])
    if bundle_validation:
        args.extend(["--calibration-bundle-validation", bundle_validation])
    return args


def _run_router_report(argv: list[str], report_path: Path) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        router_main(argv + ["--out", str(report_path)])

    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _selected(report: dict[str, Any]) -> dict[str, Any]:
    selected = (report.get("decision", {}) or {}).get("selected", {}) or {}
    return {
        "codec": selected.get("codec"),
        "config": selected.get("config"),
        "rate": selected.get("rate"),
        "quality": selected.get("quality"),
        "energy": selected.get("energy"),
        "time_ms": selected.get("time_ms"),
        "cost": selected.get("cost"),
        "ranking_cost": selected.get("ranking_cost", selected.get("cost")),
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point_quality_guard_value(point: RDEPoint, stat: str) -> float:
    key = {
        "mean": "quality_mean",
        "min": "quality_min",
        "p10": "quality_p10",
        "p25": "quality_p25",
    }.get(stat, "quality_mean")
    return float(point.raw.get(key, point.quality))


def _candidate_from_scored(item: dict[str, Any]) -> Candidate:
    return Candidate(
        codec=str(item.get("codec")),
        config=str(item.get("config")),
        rate=float(item.get("rate")),
        quality=float(item.get("quality")),
        energy=float(item.get("energy")),
        time_ms=_float_or_none(item.get("time_ms")),
        quality_constraint_value=float(
            item.get("quality_constraint_value", item.get("quality"))
        ),
        cost=_float_or_none(item.get("cost")),
        ranking_cost=_float_or_none(item.get("ranking_cost", item.get("cost"))),
        raw=dict(item.get("raw", {}) or {}),
        source="router_scored_pool",
    )


def _load_csv_candidates(report: dict[str, Any]) -> list[Candidate]:
    resolved = report.get("resolved_args", {}) or {}
    constraints = report.get("constraints", {}) or {}
    points = load_rde_points(
        csv_path=report["csv"],
        codec_col=resolved.get("codec_col"),
        config_col=resolved.get("config_col"),
        rate_col=resolved.get("rate_col"),
        quality_col=resolved.get("quality_col"),
        energy_col=resolved.get("energy_col"),
        time_col=resolved.get("time_col"),
    )
    stat = str(constraints.get("quality_constraint_stat") or "mean")
    return [
        Candidate(
            codec=point.codec,
            config=point.config,
            rate=point.rate,
            quality=point.quality,
            energy=point.energy,
            time_ms=point.time_ms,
            quality_constraint_value=_point_quality_guard_value(point, stat),
            raw=dict(point.raw),
            source="raw_candidate_pool",
        )
        for point in points
    ]


def _candidate_pool(report: dict[str, Any]) -> list[Candidate]:
    decision = report.get("decision", {}) or {}
    scored_items = decision.get("scored_candidate_pool") or decision.get("top_k", [])
    scored_candidates = [
        _candidate_from_scored(item)
        for item in scored_items
    ]
    scored_by_key = {candidate.key: candidate for candidate in scored_candidates}

    merged: dict[tuple[str, str], Candidate] = {}
    for candidate in _load_csv_candidates(report):
        scored = scored_by_key.get(candidate.key)
        if scored is not None:
            candidate.cost = scored.cost
            candidate.ranking_cost = scored.ranking_cost
            candidate.quality_constraint_value = scored.quality_constraint_value
            candidate.source = "router_scored_pool"
        merged[candidate.key] = candidate

    for key, scored in scored_by_key.items():
        merged.setdefault(key, scored)

    return list(merged.values())


def _constraint_violations(
    candidate: Candidate,
    constraints: dict[str, Any],
) -> tuple[list[str], list[str]]:
    quality_violations: list[str] = []
    base_violations: list[str] = []

    min_quality = _float_or_none(constraints.get("min_quality"))
    if (
        min_quality is not None
        and candidate.quality_constraint_value < min_quality
    ):
        quality_violations.append("quality_guard_violation")

    max_rate = _float_or_none(constraints.get("max_rate"))
    if max_rate is not None and candidate.rate > max_rate:
        base_violations.append("max_rate_violation")

    max_energy = _float_or_none(constraints.get("max_energy"))
    if max_energy is not None and candidate.energy > max_energy:
        base_violations.append("max_energy_violation")

    max_time_ms = _float_or_none(constraints.get("max_time_ms"))
    if max_time_ms is not None:
        if candidate.time_ms is None:
            base_violations.append("missing_time_for_max_time_ms")
        elif candidate.time_ms > max_time_ms:
            base_violations.append("max_time_ms_violation")

    return quality_violations, base_violations


def _is_comparable(candidate: Candidate, constraints: dict[str, Any]) -> bool:
    quality, base = _constraint_violations(candidate, constraints)
    return not quality and not base


def _candidate_to_dict(candidate: Candidate | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "codec": None,
            "config": None,
            "rate": None,
            "quality": None,
            "energy": None,
            "time_ms": None,
            "cost": None,
            "ranking_cost": None,
            "candidate_source": None,
        }

    return {
        "codec": candidate.codec,
        "config": candidate.config,
        "rate": candidate.rate,
        "quality": candidate.quality,
        "energy": candidate.energy,
        "time_ms": candidate.time_ms,
        "quality_constraint_value": candidate.quality_constraint_value,
        "cost": candidate.cost,
        "ranking_cost": candidate.ranking_cost,
        "source": candidate.source,
        "candidate_source": _candidate_source(candidate),
    }


def _candidate_source(
    candidate: Candidate | None,
    *,
    is_router_decision: bool = False,
) -> str | None:
    if is_router_decision:
        return "router_decision"
    if candidate is None:
        return None
    if candidate.source == "router_selected":
        return "router_decision"
    if candidate.source == "router_scored_pool":
        return "router_scored_pool"
    if candidate.source == "raw_candidate_pool":
        return "raw_candidate_pool"
    return "baseline_only"


def _cost_status_and_detail(
    *,
    candidate: Candidate | None,
    quality_violations: list[str],
    base_violations: list[str],
    reason: str | None,
    is_router_decision: bool,
) -> tuple[str, str]:
    if candidate is None:
        if reason == "fixed_codec_config_not_requested":
            return "unavailable_missing_metric", "fixed_codec_config_not_requested"
        if reason in {
            "no_candidate_with_time",
            "no_quality_guard_passing_candidates",
        }:
            return "unavailable_missing_metric", "missing_required_metric"
        return "unavailable_not_exported", reason or "candidate_not_found"

    if candidate.cost is not None:
        return "available", "router_decision" if is_router_decision else "ok"

    if quality_violations:
        return "unavailable_filtered", "filtered_by_quality_guard"

    if base_violations:
        return "unavailable_filtered", "filtered_by_constraints"

    if reason == "no_candidate_with_time":
        return "unavailable_missing_metric", "missing_required_metric"

    if _candidate_source(candidate) == "raw_candidate_pool":
        return (
            "unavailable_feasible_but_unscored",
            (
                "feasible_but_unscored;"
                "candidate_not_in_scored_pool;"
                "cost_unavailable_report_lacks_scored_candidate"
            ),
        )

    return (
        "unavailable_not_exported",
        "cost_not_exported_by_router_report",
    )


def _make_policy_result(
    *,
    scenario: str,
    scenario_type: str,
    policy: str,
    candidate: Candidate | None,
    router_candidate: Candidate | None,
    best_cost: Optional[float],
    constraints: dict[str, Any],
    reason: str | None = None,
    notes: list[str] | None = None,
    is_router_decision: bool = False,
) -> dict[str, Any]:
    notes = list(notes or [])
    if candidate is None:
        cost_status, cost_detail = _cost_status_and_detail(
            candidate=None,
            quality_violations=[],
            base_violations=[],
            reason=reason,
            is_router_decision=is_router_decision,
        )
        return {
            "scenario": scenario,
            "scenario_type": scenario_type,
            "policy": policy,
            "is_router_decision": is_router_decision,
            "candidate_source": None,
            "cost_status": cost_status,
            "cost_reason_detail": cost_detail,
            "comparable": False,
            "reason": reason or "no_candidate",
            "selected": _candidate_to_dict(None),
            "selected_codec": None,
            "selected_config": None,
            "decision_changed_vs_router": None,
            "rate": None,
            "quality": None,
            "energy": None,
            "time_ms": None,
            "cost": None,
            "ranking_cost": None,
            "regret": None,
            "router_relative_improvement_percent": None,
            "constraint_violations": 0,
            "quality_guard_violations": 0,
            "notes": notes,
        }

    quality_violations, base_violations = _constraint_violations(
        candidate,
        constraints,
    )
    comparable = not quality_violations and not base_violations
    if comparable and candidate.cost is None:
        comparable = False
        notes.append("cost_unavailable_for_regret")

    effective_reason = reason
    if quality_violations:
        effective_reason = "filtered_by_quality_guard"
    elif base_violations:
        effective_reason = "filtered_by_constraints"
    elif candidate.cost is None:
        effective_reason = "feasible_but_unscored"
    elif effective_reason is None:
        effective_reason = "ok"

    cost_status, cost_detail = _cost_status_and_detail(
        candidate=candidate,
        quality_violations=quality_violations,
        base_violations=base_violations,
        reason=reason,
        is_router_decision=is_router_decision,
    )

    regret = None
    if is_router_decision and candidate.cost is not None:
        regret = 0.0
    elif comparable and best_cost is not None and candidate.cost is not None:
        regret = candidate.cost - best_cost

    router_cost = router_candidate.cost if router_candidate is not None else None
    improvement = None
    if is_router_decision and candidate.cost is not None:
        improvement = 0.0
    elif (
        comparable
        and router_cost is not None
        and candidate.cost is not None
        and candidate.cost != 0
    ):
        improvement = ((candidate.cost - router_cost) / abs(candidate.cost)) * 100.0

    decision_changed = False if is_router_decision else (
        candidate.key != router_candidate.key
        if router_candidate is not None
        else None
    )

    return {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "policy": policy,
        "is_router_decision": is_router_decision,
        "candidate_source": _candidate_source(
            candidate,
            is_router_decision=is_router_decision,
        ),
        "cost_status": cost_status,
        "cost_reason_detail": cost_detail,
        "comparable": comparable,
        "reason": effective_reason,
        "selected": _candidate_to_dict(candidate),
        "selected_codec": candidate.codec,
        "selected_config": candidate.config,
        "decision_changed_vs_router": decision_changed,
        "rate": candidate.rate,
        "quality": candidate.quality,
        "energy": candidate.energy,
        "time_ms": candidate.time_ms,
        "cost": candidate.cost,
        "ranking_cost": candidate.ranking_cost,
        "regret": regret,
        "router_relative_improvement_percent": improvement,
        "constraint_violations": len(base_violations),
        "quality_guard_violations": len(quality_violations),
        "notes": notes,
    }


def _pick_min(
    candidates: list[Candidate],
    key_fn,
) -> Candidate | None:
    return min(candidates, key=key_fn) if candidates else None


def _pick_max(
    candidates: list[Candidate],
    key_fn,
) -> Candidate | None:
    return max(candidates, key=key_fn) if candidates else None


def _safe_pool(
    candidates: list[Candidate],
    constraints: dict[str, Any],
) -> list[Candidate]:
    return [
        candidate
        for candidate in candidates
        if _is_comparable(candidate, constraints)
    ]


def _policy_results(
    *,
    scenario: str,
    scenario_type: str,
    candidates: list[Candidate],
    router_candidate: Candidate | None,
    constraints: dict[str, Any],
    fixed_codec: str | None,
    fixed_config: str | None,
    include_random_safe: bool,
) -> list[dict[str, Any]]:
    safe = _safe_pool(candidates, constraints)
    safe_with_cost = [
        candidate
        for candidate in safe
        if candidate.cost is not None
    ]
    best = _pick_min(safe_with_cost, lambda c: c.cost)
    best_cost = best.cost if best is not None else None

    results: list[dict[str, Any]] = []

    if router_candidate is not None:
        results.append(
            _make_policy_result(
                scenario=scenario,
                scenario_type=scenario_type,
                policy="router",
                candidate=router_candidate,
                router_candidate=router_candidate,
                best_cost=best_cost,
                constraints=constraints,
                reason=None,
                notes=[],
                is_router_decision=True,
            )
        )

    policies: list[tuple[str, Candidate | None, str | None, list[str]]] = [
        (
            "lowest_rate",
            _pick_min(safe, lambda c: (c.rate, c.codec, c.config)),
            None if safe else "no_quality_guard_passing_candidates",
            [],
        ),
        (
            "highest_quality",
            _pick_max(
                safe,
                lambda c: (
                    c.quality_constraint_value,
                    c.quality,
                    -c.rate,
                    c.codec,
                    c.config,
                ),
            ),
            None if safe else "no_quality_guard_passing_candidates",
            [],
        ),
        (
            "lowest_energy",
            _pick_min(safe, lambda c: (c.energy, c.codec, c.config)),
            None if safe else "no_quality_guard_passing_candidates",
            [],
        ),
        (
            "fastest_time",
            _pick_min(
                [candidate for candidate in safe if candidate.time_ms is not None],
                lambda c: (c.time_ms, c.codec, c.config),
            ),
            (
                "no_candidate_with_time"
                if safe and not any(c.time_ms is not None for c in safe)
                else (None if safe else "no_quality_guard_passing_candidates")
            ),
            [],
        ),
        (
            "global_best_average",
            best,
            None if best is not None else "no_costed_comparable_candidates",
            ["derived_from_router_scored_pool"],
        ),
    ]

    if fixed_codec and fixed_config:
        fixed = next(
            (
                candidate
                for candidate in candidates
                if candidate.codec == fixed_codec and candidate.config == fixed_config
            ),
            None,
        )
        fixed_reason = None if fixed is not None else "fixed_candidate_not_found"
        policies.append(("fixed_codec_config", fixed, fixed_reason, []))
    else:
        policies.append(
            (
                "fixed_codec_config",
                None,
                "fixed_codec_config_not_requested",
                [],
            )
        )

    if include_random_safe:
        random_safe = None
        if safe:
            rng = random.Random(0)
            random_safe = rng.choice(sorted(safe, key=lambda c: (c.codec, c.config)))
        policies.append(
            (
                "random_safe",
                random_safe,
                None if random_safe is not None else "no_quality_guard_passing_candidates",
                ["diagnostic_deterministic_seed_0"],
            )
        )

    for policy, candidate, reason, notes in policies:
        results.append(
            _make_policy_result(
                scenario=scenario,
                scenario_type=scenario_type,
                policy=policy,
                candidate=candidate,
                router_candidate=router_candidate,
                best_cost=best_cost,
                constraints=constraints,
                reason=reason,
                notes=notes,
            )
        )

    return results


def _router_candidate_from_report(
    report: dict[str, Any],
    candidates: list[Candidate],
) -> Candidate | None:
    selected = _selected(report)
    key = (str(selected.get("codec")), str(selected.get("config")))
    for candidate in candidates:
        if candidate.key == key:
            candidate.cost = _float_or_none(selected.get("cost")) or candidate.cost
            candidate.ranking_cost = (
                _float_or_none(selected.get("ranking_cost"))
                or candidate.ranking_cost
                or candidate.cost
            )
            return candidate

    if selected.get("codec") is None or selected.get("config") is None:
        return None

    return Candidate(
        codec=str(selected.get("codec")),
        config=str(selected.get("config")),
        rate=float(selected.get("rate")),
        quality=float(selected.get("quality")),
        energy=float(selected.get("energy")),
        time_ms=_float_or_none(selected.get("time_ms")),
        quality_constraint_value=float(
            selected.get("quality_constraint_value", selected.get("quality"))
        ),
        cost=_float_or_none(selected.get("cost")),
        ranking_cost=_float_or_none(selected.get("ranking_cost", selected.get("cost"))),
        source="router_selected",
    )


def _summarize_scenario(
    *,
    scenario: str,
    scenario_type: str,
    data_source: str,
    report: dict[str, Any],
    candidates: list[Candidate],
    policy_results: list[dict[str, Any]],
) -> dict[str, Any]:
    constraints = report.get("constraints", {}) or {}
    quality_guard_violations = 0
    base_constraint_violations = 0
    for candidate in candidates:
        quality, base = _constraint_violations(candidate, constraints)
        quality_guard_violations += 1 if quality else 0
        base_constraint_violations += 1 if base else 0

    router_decision = _selected(report)
    comparable_policies = [
        row for row in policy_results if row.get("comparable") is True
    ]
    changed_policies = [
        row
        for row in comparable_policies
        if row.get("decision_changed_vs_router") is True
    ]

    decision = report.get("decision", {}) or {}
    notes: list[str] = []
    top_k_count = len(decision.get("top_k", []) or [])
    if decision.get("num_points_admissible", 0) > top_k_count:
        notes.append("router_scored_pool_truncated_by_audit_top_k")

    return {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "data_source": data_source,
        "router_decision": router_decision,
        "candidate_counts": {
            "num_candidates_total": decision.get("num_points_total"),
            "num_candidates_admissible": decision.get("num_points_admissible"),
            "num_candidates_safe": decision.get("num_points_safe"),
            "num_candidates_near": decision.get("num_points_near"),
            "num_candidates_from_csv": len(candidates),
            "num_candidates_comparable": len(_safe_pool(candidates, constraints)),
            "num_quality_guard_violations": quality_guard_violations,
            "num_base_constraint_violations": base_constraint_violations,
        },
        "calibration_bundle": report.get(
            "calibration_bundle",
            {
                "enabled": False,
            },
        ),
        "calibration_bundle_validation": report.get(
            "calibration_bundle_validation",
            {
                "enabled": False,
            },
        ),
        "constraints": constraints,
        "policies": policy_results,
        "notes": notes,
    }


def _scenario_from_report(
    *,
    scenario: str,
    scenario_type: str,
    data_source: str,
    report: dict[str, Any],
    fixed_codec: str | None,
    fixed_config: str | None,
    include_random_safe: bool,
) -> dict[str, Any]:
    candidates = _candidate_pool(report)
    router_candidate = _router_candidate_from_report(report, candidates)
    policy_results = _policy_results(
        scenario=scenario,
        scenario_type=scenario_type,
        candidates=candidates,
        router_candidate=router_candidate,
        constraints=report.get("constraints", {}) or {},
        fixed_codec=fixed_codec,
        fixed_config=fixed_config,
        include_random_safe=include_random_safe,
    )
    return _summarize_scenario(
        scenario=scenario,
        scenario_type=scenario_type,
        data_source=data_source,
        report=report,
        candidates=candidates,
        policy_results=policy_results,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _write_summary_csv(path: Path, scenarios: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for scenario in scenarios:
            counts = scenario["candidate_counts"]
            router = scenario["router_decision"]
            policies = scenario["policies"]
            comparable = [
                row for row in policies if row.get("comparable") is True
            ]
            changed = [
                row
                for row in comparable
                if row.get("decision_changed_vs_router") is True
            ]
            writer.writerow(
                {
                    "scenario": scenario["scenario"],
                    "scenario_type": scenario["scenario_type"],
                    "data_source": scenario["data_source"],
                    "router_codec": router.get("codec"),
                    "router_config": router.get("config"),
                    "router_cost": router.get("cost"),
                    "num_candidates_total": counts.get("num_candidates_total"),
                    "num_candidates_admissible": counts.get(
                        "num_candidates_admissible"
                    ),
                    "num_candidates_comparable": counts.get(
                        "num_candidates_comparable"
                    ),
                    "num_quality_guard_violations": counts.get(
                        "num_quality_guard_violations"
                    ),
                    "num_base_constraint_violations": counts.get(
                        "num_base_constraint_violations"
                    ),
                    "num_policies": len(policies),
                    "num_comparable_policies": len(comparable),
                    "num_changed_vs_router": len(changed),
                    "notes": _csv_cell(scenario.get("notes", [])),
                }
            )


def _flatten_policy_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for policy in scenario["policies"]:
            row = {field: policy.get(field) for field in POLICY_FIELDS}
            rows.append(row)
    return rows


def _write_policy_csv(path: Path, scenarios: list[dict[str, Any]]) -> None:
    rows = _flatten_policy_rows(scenarios)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POLICY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_cell(row.get(field))
                    for field in POLICY_FIELDS
                }
            )


def run_router_effectiveness_audit(
    *,
    csv_path: str,
    config_path: str | None = None,
    bundle_manifest: str | None = None,
    bundle_validation: str | None = None,
    out_dir: str = "results/routing_context/effectiveness_audit",
    out_path: str | None = None,
    summary_out: str | None = None,
    by_policy_out: str | None = None,
    fixed_codec: str | None = None,
    fixed_config: str | None = None,
    include_random_safe: bool = False,
    audit_top_k: int = 10000,
    router_args: list[str] | None = None,
) -> dict[str, Any]:
    router_args = list(router_args or [])
    if bundle_validation and not bundle_manifest:
        raise ValueError(
            "--bundle-validation requires --bundle-manifest. "
            "No automatic discovery is performed."
        )

    _validate_read_only_request(
        config_path=config_path,
        router_args=router_args,
    )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    json_out = (
        Path(out_path)
        if out_path is not None
        else out_root / "router_effectiveness_audit.json"
    )
    summary_csv = (
        Path(summary_out)
        if summary_out is not None
        else out_root / "router_effectiveness_audit.csv"
    )
    policy_csv = (
        Path(by_policy_out)
        if by_policy_out is not None
        else out_root / "router_effectiveness_by_policy.csv"
    )

    base_args = _base_router_args(
        csv_path=csv_path,
        config_path=config_path,
        router_args=router_args,
        audit_top_k=audit_top_k,
    )

    scenarios: list[dict[str, Any]] = []

    baseline_report = _run_router_report(
        base_args,
        out_root / "benchmark_router_report.json",
    )
    scenarios.append(
        _scenario_from_report(
            scenario="benchmark",
            scenario_type="benchmark-original",
            data_source=str(csv_path),
            report=baseline_report,
            fixed_codec=fixed_codec,
            fixed_config=fixed_config,
            include_random_safe=include_random_safe,
        )
    )

    if bundle_manifest:
        bundle_args = _mode_router_args(
            base_args=base_args,
            bundle_manifest=bundle_manifest,
        )
        bundle_report = _run_router_report(
            bundle_args,
            out_root / "bundle_router_report.json",
        )
        bundle_csv = (
            (bundle_report.get("calibration_bundle", {}) or {})
            .get("calibrated_csv_path")
            or bundle_report.get("csv")
        )
        scenarios.append(
            _scenario_from_report(
                scenario="calibration_bundle",
                scenario_type="calibrated-data",
                data_source=str(bundle_csv),
                report=bundle_report,
                fixed_codec=fixed_codec,
                fixed_config=fixed_config,
                include_random_safe=include_random_safe,
            )
        )

        if bundle_validation:
            validated_args = _mode_router_args(
                base_args=base_args,
                bundle_manifest=bundle_manifest,
                bundle_validation=bundle_validation,
            )
            validated_report = _run_router_report(
                validated_args,
                out_root / "validated_bundle_router_report.json",
            )
            validated_csv = (
                (validated_report.get("calibration_bundle", {}) or {})
                .get("calibrated_csv_path")
                or validated_report.get("csv")
            )
            scenarios.append(
                _scenario_from_report(
                    scenario="validated_calibration_bundle",
                    scenario_type="validated-calibrated-data",
                    data_source=str(validated_csv),
                    report=validated_report,
                    fixed_codec=fixed_codec,
                    fixed_config=fixed_config,
                    include_random_safe=include_random_safe,
                )
            )

    report = {
        "mode": "router_effectiveness_audit",
        "router_version": ROUTER_VERSION,
        "created_at_utc": _utc_now(),
        "read_only": True,
        "inputs": {
            "csv": str(csv_path),
            "config": config_path,
            "bundle_manifest": bundle_manifest,
            "bundle_validation": bundle_validation,
            "fixed_codec": fixed_codec,
            "fixed_config": fixed_config,
            "include_random_safe": include_random_safe,
            "audit_top_k": audit_top_k,
        },
        "methodology": {
            "router_logic_changed": False,
            "j_rde_changed": False,
            "baseline_policies_respect_quality_guard": True,
            "no_auto_discovery": True,
            "regret_basis": "router_scored_comparable_candidates_only",
        },
        "scenarios": scenarios,
    }

    _write_json(json_out, report)
    _write_summary_csv(summary_csv, scenarios)
    _write_policy_csv(policy_csv, scenarios)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of router effectiveness against baselines."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bundle-manifest", default=None)
    parser.add_argument("--bundle-validation", default=None)
    parser.add_argument(
        "--out-dir",
        default="results/routing_context/effectiveness_audit",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--by-policy-out", default=None)
    parser.add_argument("--fixed-codec", default=None)
    parser.add_argument("--fixed-config", default=None)
    parser.add_argument("--include-random-safe", action="store_true")
    parser.add_argument("--audit-top-k", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args, router_args = parser.parse_known_args(argv)

    run_router_effectiveness_audit(
        csv_path=args.csv,
        config_path=args.config,
        bundle_manifest=args.bundle_manifest,
        bundle_validation=args.bundle_validation,
        out_dir=args.out_dir,
        out_path=args.out,
        summary_out=args.summary_out,
        by_policy_out=args.by_policy_out,
        fixed_codec=args.fixed_codec,
        fixed_config=args.fixed_config,
        include_random_safe=args.include_random_safe,
        audit_top_k=args.audit_top_k,
        router_args=router_args,
    )


if __name__ == "__main__":
    main()
