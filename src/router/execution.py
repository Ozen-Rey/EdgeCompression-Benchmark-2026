import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.energy_backends import CompositeEnergyMeter
from src.router.execution_validation import validate_execution_output
from src.router.observability.feedback_logger import append_feedback_row


def _image_pixel_count(path: Optional[str]) -> Optional[int]:
    if not path:
        return None

    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
            return int(width * height)
    except Exception:
        return None


def execute_plan(execution_plan: Dict[str, Any]) -> Dict[str, Any]:
    command = execution_plan.get("command")

    if not execution_plan.get("requested", False):
        return {
            "requested": False,
            "executed": False,
            "success": False,
            "reason": "execution_not_requested",
        }

    if not execution_plan.get("can_execute", False):
        return {
            "requested": True,
            "executed": False,
            "success": False,
            "reason": "execution_plan_not_executable",
            "plan_reasons": execution_plan.get("reasons", []),
        }

    if not command:
        return {
            "requested": True,
            "executed": False,
            "success": False,
            "reason": "missing_command",
        }

    output_path = execution_plan.get("output")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    meter = CompositeEnergyMeter()

    def run_once():
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    result, energy = meter.measure_callable(run_once)

    return {
        "requested": True,
        "executed": True,
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": output_path,
        "execution_time_ms": energy.time_s * 1000.0,
        "local_cpu_energy_j": energy.cpu_j,
        "local_gpu_energy_j": energy.gpu_j,
        "local_energy_j": (
            energy.total_j if energy.energy_usable_for_total else None
        ),
        "energy_scope": energy.energy_scope,
        "energy_is_measured": energy.energy_is_measured,
        "energy_usable_for_total": energy.energy_usable_for_total,
        "energy_backend": energy.energy_backend,
        "energy_method": energy.energy_method,
        "energy_quality": energy.energy_quality,
        "energy_warnings": list(energy.warnings),
    }


def apply_execution_result(report: Dict[str, Any], *, execute: bool) -> Dict[str, Any]:
    if execute:
        execution_result = execute_plan(report.get("execution_plan", {}))
        report["execution_result"] = execution_result
        report["execution_validation"] = validate_execution_output(
            execution_plan=report.get("execution_plan", {}),
            execution_result=execution_result,
        )
    else:
        report["execution_result"] = {
            "requested": False,
            "executed": False,
        }
        report["execution_validation"] = {
            "enabled": False,
            "reason": "execution_not_requested",
        }

    return report


def build_feedback_row(
    *,
    report: Dict[str, Any],
    execution_result: Dict[str, Any],
    execution_validation: Dict[str, Any],
    report_path: Path,
) -> Dict[str, Any]:
    selected = report.get("decision", {}).get("selected", {})
    cost_decomp = selected.get("cost_decomposition", {}) or {}
    run_manifest = report.get("run_manifest", {}) or {}
    git_info = run_manifest.get("git", {}) or {}
    input_path = (
        report.get("execution_plan", {}).get("input")
        or report.get("resolved_args", {}).get("input")
    )

    output_bytes = execution_validation.get("output_size_bytes")
    input_pixels = _image_pixel_count(input_path)
    actual_rate = None
    if output_bytes is not None and input_pixels:
        actual_rate = (float(output_bytes) * 8.0) / float(input_pixels)

    error = execution_result.get("reason")
    if not execution_result.get("success", False):
        stderr = str(execution_result.get("stderr") or "").strip()
        if stderr:
            error = stderr

    return {
        "router_version": report.get("router_version"),
        "feature_level": report.get("feature_level"),
        "domain": report.get("domain"),
        "input_path": input_path,
        "input_id": Path(input_path).stem if input_path else None,
        "selected_codec": selected.get("codec"),
        "selected_config": selected.get("config"),
        "decision_mode": report.get("decision", {}).get("decision_mode"),
        "profile": report.get("profile"),
        "predicted_rate": selected.get("rate"),
        "predicted_quality": selected.get("quality"),
        "predicted_energy": selected.get("energy"),
        "predicted_time_ms": selected.get("time_ms"),
        "predicted_cost": selected.get("cost"),
        "term_R": cost_decomp.get("term_R"),
        "term_E": cost_decomp.get("term_E"),
        "term_D": cost_decomp.get("term_D"),
        "actual_output_bytes": output_bytes,
        "actual_rate": actual_rate,
        "actual_time_ms": execution_result.get("execution_time_ms"),
        "local_cpu_energy_j": execution_result.get("local_cpu_energy_j"),
        "local_gpu_energy_j": execution_result.get("local_gpu_energy_j"),
        "local_energy_j": execution_result.get("local_energy_j"),
        "energy_scope": execution_result.get("energy_scope"),
        "energy_is_measured": execution_result.get("energy_is_measured"),
        "energy_usable_for_total": execution_result.get("energy_usable_for_total"),
        "energy_backend": execution_result.get("energy_backend"),
        "energy_method": execution_result.get("energy_method"),
        "energy_quality": execution_result.get("energy_quality"),
        "execution_requested": execution_result.get("requested"),
        "execution_success": execution_result.get("success"),
        "output_exists": execution_validation.get("output_exists"),
        "output_nonempty": execution_validation.get("output_nonempty"),
        "error": error,
        "report_path": str(report_path),
        "git_commit_short": git_info.get("commit_short"),
    }


def write_feedback_report(
    report: Dict[str, Any],
    *,
    feedback_out: Optional[str],
    report_path: Path,
) -> Dict[str, Any]:
    if not feedback_out:
        return report

    feedback_row = build_feedback_row(
        report=report,
        execution_result=report["execution_result"],
        execution_validation=report["execution_validation"],
        report_path=report_path,
    )
    feedback_report = {
        "enabled": True,
        "path": feedback_out,
        "written": False,
    }

    try:
        append_feedback_row(feedback_out, feedback_row)
        feedback_report["written"] = True
    except Exception as exc:
        feedback_report["error"] = str(exc)

    report["online_feedback"] = feedback_report
    return report
