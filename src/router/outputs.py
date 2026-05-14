"""Router output writers and path helpers.

Side-effect-bearing serializers extracted from rde_router.py. The router
entrypoint orchestrates I/O and decision logic; this module hosts the I/O
half so that rde_router.py keeps shrinking toward orchestration only.

No decision, ranking, scoring, normalization, or report-assembly logic
lives here. Functions here only write what the caller built.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def safe_profile_filename(profile_name: str) -> str:
    """Normalize a profile name for use as a filesystem stem."""
    return profile_name.strip().lower().replace("-", "_").replace(" ", "_")


def write_json_report(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_summary_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_topk_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
