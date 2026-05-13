import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from src.router.version import ROUTER_VERSION
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.router.version import ROUTER_VERSION


def load_experiment_suite(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Experiment suite not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        suite = json.load(f)

    if not isinstance(suite, dict):
        raise ValueError("Experiment suite root must be a JSON object.")

    if "base_config" not in suite:
        raise ValueError("Experiment suite missing required field: base_config")

    if "experiments" not in suite:
        raise ValueError("Experiment suite missing required field: experiments")

    if not isinstance(suite["experiments"], list):
        raise ValueError("Experiment suite field 'experiments' must be a list.")

    for exp in suite["experiments"]:
        if not isinstance(exp, dict):
            raise ValueError("Each experiment must be a JSON object.")

        if "name" not in exp:
            raise ValueError("Experiment missing required field: name")

        if "args" in exp and not isinstance(exp["args"], list):
            raise ValueError(f"Experiment {exp['name']} field 'args' must be a list.")

    return suite


def _safe_experiment_name(name: str) -> str:
    out = []

    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")

    cleaned = "".join(out).strip("_")
    return cleaned or "experiment"


def build_experiment_command(
    *,
    base_config: str,
    report_path: str,
    experiment_args: Optional[List[Any]] = None,
) -> List[str]:
    args = [str(x) for x in (experiment_args or [])]

    return [
        sys.executable,
        "-m",
        "src.router.rde_router",
        "--config",
        base_config,
        "--out",
        report_path,
        *args,
    ]


def run_single_experiment(
    *,
    experiment: Dict[str, Any],
    base_config: str,
    report_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    name = str(experiment["name"])
    safe_name = _safe_experiment_name(name)
    report_path = report_dir / f"{safe_name}.json"

    command = build_experiment_command(
        base_config=base_config,
        report_path=str(report_path),
        experiment_args=experiment.get("args", []),
    )

    result: Dict[str, Any] = {
        "name": name,
        "description": experiment.get("description"),
        "report_path": str(report_path),
        "command": command,
        "dry_run": dry_run,
        "success": False,
        "returncode": None,
        "stdout": None,
        "stderr": None,
    }

    if dry_run:
        result["success"] = True
        result["returncode"] = 0
        return result

    report_dir.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    result["returncode"] = completed.returncode
    result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    result["success"] = completed.returncode == 0

    return result


def _get_nested(data: Dict[str, Any], path: List[str], default=None):
    cur: Any = data

    for key in path:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]

    return cur


def summarize_report(
    *,
    experiment_name: str,
    run_result: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "experiment": experiment_name,
        "success": run_result.get("success"),
        "returncode": run_result.get("returncode"),
        "dry_run": bool(run_result.get("dry_run", False)),
        "command": " ".join(str(x) for x in run_result.get("command", [])),
        "report_path": run_result.get("report_path"),
        "selected_codec": None,
        "selected_config": None,
        "decision_mode": None,
        "selected_reason": None,
        "rate": None,
        "quality": None,
        "quality_guard_value": None,
        "energy": None,
        "time_ms": None,
        "J_RDE": None,
        "term_R": None,
        "term_E": None,
        "term_D": None,
        "execution_requested": None,
        "execution_success": None,
        "output_exists": None,
        "output_nonempty": None,
        "extension_valid": None,
        "output_size_bytes": None,
        "git_commit_short": None,
        "dirty_worktree": None,
    }

    if not run_result.get("success"):
        return row

    if run_result.get("dry_run", False):
        return row

    report_path = Path(str(run_result.get("report_path")))

    if not report_path.exists():
        row["success"] = False
        row["returncode"] = "missing_report"
        return row

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    selected = _get_nested(report, ["decision", "selected"], {})
    trace = _get_nested(report, ["decision", "decision_trace"], {})
    decomp = selected.get("cost_decomposition", {}) if isinstance(selected, dict) else {}
    execution_result = report.get("execution_result", {})
    execution_validation = report.get("execution_validation", {})
    run_manifest = report.get("run_manifest", {})
    git_info = run_manifest.get("git", {}) if isinstance(run_manifest, dict) else {}

    row.update(
        {
            "selected_codec": selected.get("codec"),
            "selected_config": selected.get("config"),
            "decision_mode": _get_nested(report, ["decision", "decision_mode"]),
            "selected_reason": trace.get("selected_reason"),
            "rate": selected.get("rate"),
            "quality": selected.get("quality"),
            "quality_guard_value": selected.get("quality_constraint_value"),
            "energy": selected.get("energy"),
            "time_ms": selected.get("time_ms"),
            "J_RDE": selected.get("cost"),
            "term_R": decomp.get("term_R"),
            "term_E": decomp.get("term_E"),
            "term_D": decomp.get("term_D"),
            "execution_requested": execution_result.get("requested"),
            "execution_success": execution_result.get("success"),
            "output_exists": execution_validation.get("output_exists"),
            "output_nonempty": execution_validation.get("output_nonempty"),
            "extension_valid": execution_validation.get("extension_valid"),
            "output_size_bytes": execution_validation.get("output_size_bytes"),
            "git_commit_short": git_info.get("commit_short"),
            "dirty_worktree": git_info.get("dirty_worktree"),
        }
    )

    return row


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment_suite(
    *,
    suite_path: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    suite = load_experiment_suite(suite_path)

    base_config = str(suite["base_config"])
    report_dir = Path(str(suite.get("report_dir", "results/routing_context/v07_experiments")))
    summary_csv = str(suite.get("summary_csv", "results/routing_context/v07_experiment_summary.csv"))
    stop_on_failure = bool(suite.get("stop_on_failure", True))

    run_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for experiment in suite["experiments"]:
        result = run_single_experiment(
            experiment=experiment,
            base_config=base_config,
            report_dir=report_dir,
            dry_run=dry_run,
        )

        run_results.append(result)

        row = summarize_report(
            experiment_name=str(experiment["name"]),
            run_result=result,
        )

        summary_rows.append(row)

        if stop_on_failure and not result.get("success"):
            break

    if not dry_run:
        write_summary_csv(summary_rows, summary_csv)

    return {
        "version": ROUTER_VERSION,
        "router_version": ROUTER_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite_path": suite_path,
        "base_config": base_config,
        "report_dir": str(report_dir),
        "summary_csv": summary_csv,
        "dry_run": dry_run,
        "num_experiments": len(suite["experiments"]),
        "num_completed": len(run_results),
        "num_success": sum(1 for r in run_results if r.get("success")),
        "run_results": run_results,
        "summary_rows": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a suite of R-D-E router experiments."
    )

    parser.add_argument(
        "--suite",
        required=True,
        help="Path to experiment suite JSON.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build commands without executing router runs.",
    )

    args = parser.parse_args()

    result = run_experiment_suite(
        suite_path=args.suite,
        dry_run=args.dry_run,
    )

    print("\n=== R-D-E Router Experiment Manager ===")
    print(f"Suite:        {result['suite_path']}")
    print(f"Base config:  {result['base_config']}")
    print(f"Report dir:   {result['report_dir']}")
    print(f"Summary CSV:  {result['summary_csv']}")
    print(f"Dry run:      {result['dry_run']}")
    print(f"Completed:    {result['num_completed']} / {result['num_experiments']}")
    print(f"Successful:   {result['num_success']} / {result['num_completed']}")
    print()

    for row in result["summary_rows"]:
        if result["dry_run"]:
            print(
                f"{row['experiment']:28s} -> DRY RUN"
            )
            print(f"  command: {row['command']}")
            continue

        if row["success"]:
            print(
                f"{row['experiment']:28s} -> "
                f"{row['selected_codec']} {row['selected_config']} "
                f"| mode={row['decision_mode']} "
                f"| J={row['J_RDE']}"
            )
        else:
            print(
                f"{row['experiment']:28s} -> FAILED "
                f"| returncode={row['returncode']}"
            )

    if not result["dry_run"]:
        print()
        print(f"Summary written to: {result['summary_csv']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n=== R-D-E Experiment Manager: failed ===")
        print(str(exc))
        raise SystemExit(2)
