from pathlib import Path
from typing import Any, Dict, List, Optional


def _allowed_extensions_for_canonical(canonical: Optional[str]) -> List[str]:
    canonical = str(canonical or "").upper()

    if canonical == "JPEG":
        return [".jpg", ".jpeg"]

    if canonical == "JXL":
        return [".jxl"]

    if canonical == "HEVC":
        return [".mp4", ".mkv", ".hevc", ".h265"]

    return []


def validate_execution_output(
    *,
    execution_plan: Dict[str, Any],
    execution_result: Dict[str, Any],
) -> Dict[str, Any]:
    requested = bool(execution_result.get("requested", False))
    executed = bool(execution_result.get("executed", False))
    success = bool(execution_result.get("success", False))

    output = (
        execution_result.get("output")
        or execution_plan.get("output")
    )

    canonical = execution_plan.get("canonical")
    allowed_extensions = _allowed_extensions_for_canonical(canonical)

    validation: Dict[str, Any] = {
        "enabled": requested,
        "requested": requested,
        "executed": executed,
        "execution_success": success,
        "output": output,
        "output_exists": False,
        "output_size_bytes": None,
        "output_nonempty": False,
        "extension_valid": None,
        "allowed_extensions": allowed_extensions,
        "execution_time_ms": execution_result.get("execution_time_ms"),
        "warnings": [],
    }

    if not requested:
        validation["enabled"] = False
        validation["reason"] = "execution_not_requested"
        return validation

    if not executed:
        validation["warnings"].append("execution_not_performed")
        return validation

    if not output:
        validation["warnings"].append("missing_output_path")
        return validation

    output_path = Path(output)

    if not output_path.exists():
        validation["warnings"].append("output_file_missing")
        return validation

    validation["output_exists"] = True

    try:
        size_bytes = output_path.stat().st_size
        validation["output_size_bytes"] = size_bytes
        validation["output_nonempty"] = size_bytes > 0

        if size_bytes <= 0:
            validation["warnings"].append("output_file_empty")
    except Exception:
        validation["warnings"].append("could_not_read_output_size")

    if allowed_extensions:
        extension_valid = output_path.suffix.lower() in allowed_extensions
        validation["extension_valid"] = extension_valid

        if not extension_valid:
            validation["warnings"].append(
                f"unexpected_output_extension:{output_path.suffix}"
            )
    else:
        validation["extension_valid"] = None
        validation["warnings"].append("no_extension_policy_for_codec")

    return validation
