"""Audit references to legacy top-level router wrapper module paths.

The router has been progressively split into thematic subpackages
(core, codecs, calibration, adaptation, observability, analysis).
Each migration left a thin re-export wrapper at the legacy top-level
path (e.g. ``src/router/rde_database.py`` forwarding to
``src.router.core.rde_database``).

This module discovers those wrappers dynamically and scans the
repository for remaining references to the legacy ``src.router.<name>``
paths, classifying each finding so the project can confidently retire
the wrappers in a future release.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.router.version import ROUTER_VERSION


SELF_RELATIVE_PATH = "src/router/observability/legacy_import_audit.py"

DEFAULT_SCAN_SUFFIXES = frozenset(
    {".py", ".md", ".ps1", ".sh", ".bat", ".toml", ".json", ".rst", ".txt"}
)

DEFAULT_SCAN_ROOTS = ("src", "tests", "scripts", "docs")

CATEGORIES = (
    "suspicious_runtime",
    "scripts",
    "docs",
    "allowed_wrapper",
    "allowed_test",
    "other",
)


@dataclass(frozen=True)
class LegacyImportFinding:
    path: str
    line: int
    text: str
    module: str
    category: str


def discover_legacy_wrapper_modules(router_dir: Path) -> dict[str, str]:
    """Return ``{legacy_name: target_subpackage}`` for each wrapper file.

    A wrapper file is a ``src/router/<name>.py`` whose first lines re-export
    from a ``src.router.<subpkg>`` subpackage with the canonical pattern
    ``from src.router.<subpkg> import <name> as _module``.

    Wrappers whose ``<name>`` collides with a sibling subpackage directory
    are excluded: Python's import system resolves ``src.router.<name>`` to
    the package in that case, so the wrapper file is effectively shadowed
    and cannot be referenced as a legacy path.
    """

    wrappers: dict[str, str] = {}

    if not router_dir.is_dir():
        return wrappers

    subpackage_names = {
        child.name
        for child in router_dir.iterdir()
        if child.is_dir() and (child / "__init__.py").is_file()
    }

    wrapper_pattern = re.compile(
        r"^\s*from\s+src\.router\.(?P<subpkg>\w+)\s+import\s+(?P<mod>\w+)\s+as\s+_module\b"
    )

    for child in sorted(router_dir.iterdir()):
        if not child.is_file() or child.suffix != ".py":
            continue

        if child.stem in subpackage_names:
            continue

        try:
            text = child.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for line in text.splitlines()[:25]:
            match = wrapper_pattern.match(line)
            if not match:
                continue
            legacy_name = child.stem
            if legacy_name == match.group("mod"):
                wrappers[legacy_name] = match.group("subpkg")
            break

    return wrappers


def _classify_finding(rel_path: str, module: str) -> str:
    normalized = rel_path.replace("\\", "/")

    if normalized == f"src/router/{module}.py":
        return "allowed_wrapper"

    if normalized.startswith("tests/"):
        return "allowed_test"

    if normalized.startswith("scripts/"):
        return "scripts"

    if normalized.startswith("docs/"):
        return "docs"

    if normalized.startswith("src/"):
        return "suspicious_runtime"

    return "other"


def scan_for_legacy_references(
    *,
    repo_root: Path,
    roots: Iterable[Path],
    wrappers: dict[str, str],
    suffixes: Iterable[str] = DEFAULT_SCAN_SUFFIXES,
) -> list[LegacyImportFinding]:
    legacy_names = sorted(wrappers.keys())

    if not legacy_names:
        return []

    alternation = "|".join(re.escape(name) for name in legacy_names)
    pattern = re.compile(rf"\bsrc\.router\.(?P<mod>{alternation})\b(?![\w])")

    allowed_suffixes = {suffix.lower() for suffix in suffixes}
    findings: list[LegacyImportFinding] = []

    for root in roots:
        if not root.is_dir():
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue

            try:
                rel_path = path.relative_to(repo_root).as_posix()
            except ValueError:
                continue

            if rel_path == SELF_RELATIVE_PATH:
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for lineno, line in enumerate(text.splitlines(), 1):
                for match in pattern.finditer(line):
                    module = match.group("mod")
                    findings.append(
                        LegacyImportFinding(
                            path=rel_path,
                            line=lineno,
                            text=line.strip(),
                            module=module,
                            category=_classify_finding(rel_path, module),
                        )
                    )

    return findings


def run_audit(
    *,
    repo_root: Path,
    roots: Iterable[Path] | None = None,
) -> tuple[list[LegacyImportFinding], dict[str, str], list[Path]]:
    """Run the legacy import audit and return findings and discovered wrappers."""

    scan_roots: list[Path]
    if roots is None:
        scan_roots = [repo_root / name for name in DEFAULT_SCAN_ROOTS]
    else:
        scan_roots = [Path(r) for r in roots]

    wrappers = discover_legacy_wrapper_modules(repo_root / "src" / "router")
    findings = scan_for_legacy_references(
        repo_root=repo_root, roots=scan_roots, wrappers=wrappers
    )
    return findings, wrappers, scan_roots


def render_report(
    findings: list[LegacyImportFinding],
    wrappers: dict[str, str],
    *,
    repo_root: Path,
    scanned_roots: Iterable[Path],
    max_per_category: int = 50,
) -> str:
    by_category: dict[str, list[LegacyImportFinding]] = {cat: [] for cat in CATEGORIES}

    for finding in findings:
        by_category.setdefault(finding.category, []).append(finding)

    lines: list[str] = []
    lines.append(f"Router version       : {ROUTER_VERSION}")
    lines.append(f"Repository root      : {repo_root}")
    lines.append(
        "Scanned roots        : "
        + ", ".join(str(r.relative_to(repo_root) if r.is_relative_to(repo_root) else r) for r in scanned_roots)
    )
    lines.append(f"Legacy wrappers      : {len(wrappers)}")
    lines.append(f"Total references     : {len(findings)}")
    lines.append("")

    for category in CATEGORIES:
        items = by_category.get(category, [])
        lines.append(f"[{category}] {len(items)} reference(s)")
        for finding in items[:max_per_category]:
            lines.append(
                f"  {finding.path}:{finding.line}  legacy={finding.module}"
            )
            lines.append(f"      {finding.text}")
        if len(items) > max_per_category:
            lines.append(f"  ... {len(items) - max_per_category} more")
        lines.append("")

    suspicious = by_category.get("suspicious_runtime", [])
    if suspicious:
        lines.append(
            f"AUDIT RESULT: SUSPICIOUS — {len(suspicious)} runtime "
            "reference(s) to legacy wrapper paths detected."
        )
    else:
        lines.append("AUDIT RESULT: OK — no runtime references to legacy wrapper paths.")

    return "\n".join(lines)


def findings_to_json(
    findings: list[LegacyImportFinding],
    wrappers: dict[str, str],
    *,
    repo_root: Path,
    scanned_roots: Iterable[Path],
) -> dict[str, Any]:
    return {
        "router_version": ROUTER_VERSION,
        "repo_root": str(repo_root),
        "scanned_roots": [str(r) for r in scanned_roots],
        "wrappers": dict(sorted(wrappers.items())),
        "summary": {
            "total": len(findings),
            "by_category": {
                category: sum(1 for f in findings if f.category == category)
                for category in CATEGORIES
            },
        },
        "findings": [
            {
                "path": f.path,
                "line": f.line,
                "module": f.module,
                "category": f.category,
                "text": f.text,
            }
            for f in findings
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the repository for references to legacy router wrapper "
            "module paths (src.router.<old_module>)."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (defaults to the package's installed location).",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "Directory to scan, relative to --repo-root or absolute. "
            "Repeat to add more. Defaults to src, tests, scripts, docs."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to also write the audit report as JSON.",
    )
    args = parser.parse_args(argv)

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[3]

    if args.root:
        scan_roots = [
            (Path(r) if Path(r).is_absolute() else repo_root / r).resolve()
            for r in args.root
        ]
    else:
        scan_roots = [(repo_root / name) for name in DEFAULT_SCAN_ROOTS]

    findings, wrappers, _ = run_audit(repo_root=repo_root, roots=scan_roots)

    print(render_report(
        findings,
        wrappers,
        repo_root=repo_root,
        scanned_roots=scan_roots,
    ))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                findings_to_json(
                    findings,
                    wrappers,
                    repo_root=repo_root,
                    scanned_roots=scan_roots,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    suspicious = [f for f in findings if f.category == "suspicious_runtime"]
    return 1 if suspicious else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
