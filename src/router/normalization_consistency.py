"""Report-only consistency checks for router normalization audits."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


COMPARABILITY_LOCKING_MODES = {"global", "profile", "dataset"}


class NormalizationAuditLoadError(ValueError):
    """Raised when a previous receipt/report cannot be loaded as JSON."""


def _empty_differences() -> dict[str, bool]:
    return {
        "mode_changed": False,
        "scales_source_changed": False,
        "rate_range_changed": False,
        "energy_range_changed": False,
        "quality_range_changed": False,
        "quality_metric_changed": False,
        "quality_direction_changed": False,
    }


def load_previous_normalization_audit(path: str | Path) -> dict[str, Any] | None:
    """Load normalization_audit from a receipt or router report.

    Accepted JSON layouts:
    - direct receipt/report with top-level ``normalization_audit``
    - router report with ``decision_receipt.normalization_audit``

    Missing normalization audit returns ``None`` so callers can report a
    controlled comparability warning without failing the current run.
    """

    json_path = Path(path)

    if not json_path.exists():
        raise NormalizationAuditLoadError(
            f"previous decision receipt does not exist: {json_path}"
        )

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise NormalizationAuditLoadError(
            f"previous decision receipt is not valid JSON: {json_path}"
        ) from exc
    except OSError as exc:
        raise NormalizationAuditLoadError(
            f"previous decision receipt could not be read: {json_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise NormalizationAuditLoadError(
            f"previous decision receipt must be a JSON object: {json_path}"
        )

    audit = payload.get("normalization_audit")
    if isinstance(audit, dict):
        return audit

    decision_receipt = payload.get("decision_receipt")
    if isinstance(decision_receipt, dict):
        audit = decision_receipt.get("normalization_audit")
        if isinstance(audit, dict):
            return audit

    return None


def compare_normalization_audits(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compare two normalization audits without affecting router selection."""

    differences = _empty_differences()
    warnings: list[str] = []
    comparable = True

    if previous is None:
        return {
            "comparable": False,
            "warnings": ["previous_normalization_audit_missing"],
            "differences": differences,
        }

    if not isinstance(current, dict):
        return {
            "comparable": False,
            "warnings": ["current_normalization_audit_missing_or_invalid"],
            "differences": differences,
        }

    current_mode = current.get("mode")
    previous_mode = previous.get("mode")
    if current_mode != previous_mode:
        differences["mode_changed"] = True
        comparable = False
        warnings.append(
            f"mode_changed: previous={previous_mode!r} current={current_mode!r}"
        )

    current_quality_metric = current.get("quality_metric")
    previous_quality_metric = previous.get("quality_metric")
    if current_quality_metric != previous_quality_metric:
        differences["quality_metric_changed"] = True
        comparable = False
        warnings.append(
            "quality_metric_changed: "
            f"previous={previous_quality_metric!r} current={current_quality_metric!r}"
        )

    current_quality_direction = current.get("quality_direction")
    previous_quality_direction = previous.get("quality_direction")
    if current_quality_direction != previous_quality_direction:
        differences["quality_direction_changed"] = True
        comparable = False
        warnings.append(
            "quality_direction_changed: "
            f"previous={previous_quality_direction!r} "
            f"current={current_quality_direction!r}"
        )

    current_source = current.get("scales_source")
    previous_source = previous.get("scales_source")
    if current_source != previous_source:
        differences["scales_source_changed"] = True
        warnings.append(
            "scales_source_changed: "
            f"previous={previous_source!r} current={current_source!r}"
        )
        if (
            current_mode in COMPARABILITY_LOCKING_MODES
            or previous_mode in COMPARABILITY_LOCKING_MODES
        ):
            comparable = False

    if current.get("computed_at_runtime") != previous.get("computed_at_runtime"):
        warnings.append(
            "computed_at_runtime_changed: "
            f"previous={previous.get('computed_at_runtime')!r} "
            f"current={current.get('computed_at_runtime')!r}"
        )

    for axis in ("rate", "energy", "quality"):
        if _range_changed(
            current=current,
            previous=previous,
            axis=axis,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            differences[f"{axis}_range_changed"] = True
            warnings.append(
                f"{axis}_range_changed: "
                f"previous=({previous.get(axis + '_min')!r}, "
                f"{previous.get(axis + '_max')!r}) "
                f"current=({current.get(axis + '_min')!r}, "
                f"{current.get(axis + '_max')!r})"
            )

    return {
        "comparable": comparable,
        "warnings": warnings,
        "differences": differences,
    }


def _range_changed(
    *,
    current: dict[str, Any],
    previous: dict[str, Any],
    axis: str,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    return (
        not _numbers_close(
            previous.get(f"{axis}_min"),
            current.get(f"{axis}_min"),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        or not _numbers_close(
            previous.get(f"{axis}_max"),
            current.get(f"{axis}_max"),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    )


def _numbers_close(
    left: Any,
    right: Any,
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if left is None and right is None:
        return True

    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return left == right

    return math.isclose(
        left_float,
        right_float,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
