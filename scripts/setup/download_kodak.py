"""Cross-platform Kodak image downloader for the local mini-benchmark.

This helper fetches the 24 Kodak true-colour PNG images into a repository-local
directory so the image mini-benchmark can run without a separate, OS-specific
download step. It uses only the Python standard library (urllib) so it works on
Windows, Linux and macOS without extra dependencies.

It does not redistribute Kodak images; it downloads them on demand from the
upstream source and verifies that 24 PNG files are present.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "datasets" / "images" / "kodak"
KODAK_BASE_URL = "https://www.r0k.us/graphics/kodak/kodak/"
KODAK_COUNT = 24
KODAK_TIMEOUT_S = 60.0


def kodak_filenames() -> List[str]:
    return [f"kodim{index:02d}.png" for index in range(1, KODAK_COUNT + 1)]


def _download_one(filename: str, dest: Path, timeout_s: float) -> Dict[str, Any]:
    url = KODAK_BASE_URL + filename
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = response.read()
    except Exception as exc:  # pragma: no cover - network failure path
        return {"file": filename, "ok": False, "error": f"{exc.__class__.__name__}: {exc}"}

    if not payload:
        return {"file": filename, "ok": False, "error": "empty_response"}

    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return {"file": filename, "ok": True, "bytes": len(payload)}


def download_kodak(
    out_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    timeout_s: float = KODAK_TIMEOUT_S,
) -> Dict[str, Any]:
    out_dir = out_dir.expanduser().resolve()
    report: Dict[str, Any] = {
        "out_dir": str(out_dir),
        "source": KODAK_BASE_URL,
        "expected": KODAK_COUNT,
        "downloaded": 0,
        "already_present": 0,
        "failures": [],
        "dry_run": dry_run,
    }

    if dry_run:
        report["planned_files"] = kodak_filenames()
        report["complete"] = False
        return report

    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in kodak_filenames():
        dest = out_dir / filename
        if dest.exists() and dest.stat().st_size > 0 and not force:
            report["already_present"] += 1
            continue
        result = _download_one(filename, dest, timeout_s)
        if result["ok"]:
            report["downloaded"] += 1
        else:
            report["failures"].append(result)

    present = sum(
        1 for filename in kodak_filenames() if (out_dir / filename).exists()
    )
    report["present"] = present
    report["complete"] = present == KODAK_COUNT and not report["failures"]
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the 24 Kodak PNG images for the local image mini-benchmark. "
            "Standard-library only; no redistribution."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Target directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    parser.add_argument("--dry-run", action="store_true", help="List planned files without downloading.")
    parser.add_argument("--report-out", type=Path, help="Write a JSON report to this path.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = download_kodak(args.out_dir, force=args.force, dry_run=args.dry_run)

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.report_out and not args.dry_run:
        args.report_out.write_text(payload + "\n", encoding="utf-8")

    if args.dry_run:
        return 0
    return 0 if report.get("complete") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
