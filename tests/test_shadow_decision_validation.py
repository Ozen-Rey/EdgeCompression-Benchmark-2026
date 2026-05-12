import csv
import json
from pathlib import Path
from uuid import uuid4

from src.router.shadow_decision_validation import run_shadow_decision_validation
from src.router.version import ROUTER_VERSION


def _tmp_dir(name: str) -> Path:
    root = (
        Path(__file__).with_name("_tmp")
        / "shadow_decision_validation"
        / f"{name}_{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _case(
    case_id: str,
    *,
    changed: bool = False,
    baseline_cost: float = 1.0,
    shadow_cost: float = 0.9,
    baseline_quality: float = 95.0,
    shadow_quality: float = 95.0,
    baseline_rate: float = 1.0,
    shadow_rate: float = 0.9,
    baseline_energy: float = 1.0,
    shadow_energy: float = 0.9,
    baseline_time: float = 10.0,
    shadow_time: float = 9.0,
    unsafe_energy: bool = False,
) -> dict[str, object]:
    row = {
        "case_id": case_id,
        "baseline_codec": "JPEG",
        "baseline_config": "q=85",
        "shadow_codec": "JXL" if changed else "JPEG",
        "shadow_config": "d=1.0" if changed else "q=85",
        "decision_changed": changed,
        "baseline_rate": baseline_rate,
        "shadow_rate": shadow_rate,
        "baseline_quality": baseline_quality,
        "shadow_quality": shadow_quality,
        "baseline_energy": baseline_energy,
        "shadow_energy": shadow_energy,
        "baseline_time": baseline_time,
        "shadow_time": shadow_time,
        "baseline_cost": baseline_cost,
        "shadow_cost": shadow_cost,
        "rate_delta": shadow_rate - baseline_rate,
        "quality_delta": shadow_quality - baseline_quality,
        "energy_delta": shadow_energy - baseline_energy,
        "time_delta": shadow_time - baseline_time,
        "cost_delta": shadow_cost - baseline_cost,
        "notes": ["calibration_bundle_validated"],
    }

    if unsafe_energy:
        row["energy_scope"] = "gpu"
        row["energy_usable_for_total"] = False
        row["notes"] = [
            "calibration_bundle_validated",
            "gpu_only_energy_not_total",
        ]

    return row


def _comparison(rows: list[dict[str, object]]) -> dict[str, object]:
    changed = sum(1 for row in rows if row["decision_changed"])
    total = len(rows)
    return {
        "mode": "shadow_decision_comparison_only",
        "router_version": "0.21.0",
        "baseline_csv": "baseline.csv",
        "baseline_csv_sha256": "a" * 64,
        "bundle_manifest": "manifest.json",
        "calibrated_csv": "calibrated.csv",
        "candidate_calibration_bundle_manifest_path": "manifest.json",
        "candidate_calibration_bundle_manifest_sha256": "b" * 64,
        "candidate_calibrated_csv_sha256": "d" * 64,
        "calibration_bundle": {
            "validated": True,
            "energy_policy": "usable_total_only",
        },
        "total_cases": total,
        "changed_decisions": changed,
        "unchanged_decisions": total - changed,
        "changed_decision_rate": changed / total if total else None,
        "aggregate_deltas": {},
        "per_case": rows,
    }


def _write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(_comparison(rows)),
        encoding="utf-8",
    )


def _run(root: Path, rows: list[dict[str, object]], **kwargs) -> dict:
    comparison = root / "comparison.json"
    out = root / "validation.json"
    summary = root / "validation.csv"
    _write_comparison(comparison, rows)

    return run_shadow_decision_validation(
        comparison_path=str(comparison),
        out_path=str(out),
        summary_out=str(summary),
        **kwargs,
    )


def test_accepted_when_candidate_improves_cost_with_few_flips_and_no_violations():
    root = _tmp_dir("accepted")
    rows = [
        _case("case_1", changed=True, shadow_cost=0.90),
        _case("case_2", shadow_cost=0.85),
        _case("case_3", shadow_cost=0.95),
    ]

    report = _run(root, rows)

    assert report["mode"] == "shadow_decision_validation_only"
    assert report["router_version"] == ROUTER_VERSION
    assert report["accepted"] is True
    assert report["validated_comparison_path"].endswith("comparison.json")
    assert len(report["validated_comparison_sha256"]) == 64
    assert report["candidate_calibration_bundle_manifest_sha256"] == "b" * 64
    assert report["candidate_calibrated_csv_sha256"] == "d" * 64
    assert report["decision_count"] == 3
    assert report["changed_decision_count"] == 1
    assert report["decision_churn_rate"] == 1 / 3
    assert report["mean_candidate_cost"] < report["mean_baseline_cost"]
    assert report["rejection_reasons"] == []


def test_rejected_if_insufficient_samples():
    root = _tmp_dir("insufficient")

    report = _run(root, [_case("case_1", shadow_cost=0.5)])

    assert report["accepted"] is False
    assert report["insufficient_sample_groups"] == ["global"]
    assert "insufficient_decisions" in report["rejection_reasons"]


def test_rejected_if_candidate_worsens_mean_cost():
    root = _tmp_dir("cost_regression")
    rows = [
        _case("case_1", shadow_cost=1.10),
        _case("case_2", shadow_cost=1.20),
        _case("case_3", shadow_cost=1.05),
    ]

    report = _run(root, rows)

    assert report["accepted"] is False
    assert report["mean_candidate_cost"] > report["mean_baseline_cost"]
    assert "candidate_cost_regression" in report["rejection_reasons"]


def test_rejected_if_excessive_decision_churn():
    root = _tmp_dir("churn")
    rows = [
        _case("case_1", changed=True, shadow_cost=0.7),
        _case("case_2", changed=True, shadow_cost=0.7),
        _case("case_3", changed=True, shadow_cost=0.7),
    ]

    report = _run(root, rows)

    assert report["accepted"] is False
    assert report["decision_churn_rate"] == 1.0
    assert "excessive_decision_churn" in report["rejection_reasons"]


def test_rejected_if_quality_guard_violation():
    root = _tmp_dir("quality_violation")
    rows = [
        _case("case_1", shadow_quality=94.0, shadow_cost=0.7),
        _case("case_2", shadow_cost=0.7),
        _case("case_3", shadow_cost=0.7),
    ]

    report = _run(root, rows)

    assert report["accepted"] is False
    assert report["quality_guard_violations"] == 1
    assert "quality_guard_violation" in report["rejection_reasons"]


def test_rejected_if_unsafe_energy_provenance():
    root = _tmp_dir("unsafe_energy")
    rows = [
        _case("case_1", shadow_cost=0.7, unsafe_energy=True),
        _case("case_2", shadow_cost=0.7),
        _case("case_3", shadow_cost=0.7),
    ]

    report = _run(root, rows)

    assert report["accepted"] is False
    assert report["unsafe_energy_rows"] == 1
    assert "unsafe_energy_provenance" in report["rejection_reasons"]


def test_csv_and_json_outputs_are_written():
    root = _tmp_dir("outputs")
    comparison = root / "comparison.json"
    out = root / "validation.json"
    summary = root / "validation.csv"
    rows = [
        _case("case_1", changed=True, shadow_cost=0.90),
        _case("case_2", shadow_cost=0.85),
        _case("case_3", shadow_cost=0.95),
    ]
    _write_comparison(comparison, rows)

    run_shadow_decision_validation(
        comparison_path=str(comparison),
        out_path=str(out),
        summary_out=str(summary),
    )

    json_report = json.loads(out.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(summary.open("r", encoding="utf-8")))

    assert json_report["accepted"] is True
    assert len(csv_rows) == 1
    assert csv_rows[0]["mode"] == "shadow_decision_validation_only"
    assert csv_rows[0]["accepted"] == "True"
    assert csv_rows[0]["candidate_calibration_bundle_manifest_sha256"] == "b" * 64


def test_module_is_read_only_and_not_imported_by_router():
    router_path = Path(__file__).resolve().parents[1] / "src" / "router" / "rde_router.py"
    router_source = router_path.read_text(encoding="utf-8")

    assert "shadow_decision_validation" not in router_source
