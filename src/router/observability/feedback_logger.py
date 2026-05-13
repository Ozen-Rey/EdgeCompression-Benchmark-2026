"""Append-only online execution feedback logging for router runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


FEEDBACK_FIELDS = [
    "timestamp",
    "router_version",
    "feature_level",
    "domain",
    "input_path",
    "input_id",
    "selected_codec",
    "selected_config",
    "decision_mode",
    "profile",
    "predicted_rate",
    "predicted_quality",
    "predicted_energy",
    "predicted_time_ms",
    "predicted_cost",
    "term_R",
    "term_E",
    "term_D",
    "actual_output_bytes",
    "actual_rate",
    "actual_time_ms",
    "local_cpu_energy_j",
    "local_gpu_energy_j",
    "local_energy_j",
    "energy_scope",
    "energy_is_measured",
    "energy_usable_for_total",
    "energy_backend",
    "energy_method",
    "energy_quality",
    "execution_requested",
    "execution_success",
    "output_exists",
    "output_nonempty",
    "error",
    "report_path",
    "git_commit_short",
]


def _serialize_value(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    if isinstance(value, Path):
        return str(value)

    return value


def _read_header(path: Path) -> list[str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def append_feedback_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one feedback row without rewriting or truncating existing data."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_with_defaults = dict(row)
    row_with_defaults.setdefault(
        "timestamp",
        datetime.now().isoformat(timespec="seconds"),
    )

    existing_header = _read_header(out_path)
    if existing_header is None:
        extra_fields = sorted(
            key for key in row_with_defaults.keys() if key not in FEEDBACK_FIELDS
        )
        fieldnames = FEEDBACK_FIELDS + extra_fields
        write_header = True
    else:
        fieldnames = existing_header
        write_header = False

    serialized = {
        field: _serialize_value(row_with_defaults.get(field))
        for field in fieldnames
    }

    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if write_header:
            writer.writeheader()

        writer.writerow(serialized)
