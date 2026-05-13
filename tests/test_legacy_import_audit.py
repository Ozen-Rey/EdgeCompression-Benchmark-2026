"""Tests for the legacy router wrapper usage audit."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.router.observability.legacy_import_audit import (
    discover_legacy_wrapper_modules,
    main as audit_main,
    run_audit,
)


WRAPPER_TEMPLATE = (
    "import sys as _sys\n"
    "from pathlib import Path as _Path\n"
    "\n"
    "if __package__ in {{None, ''}}:\n"
    "    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))\n"
    "\n"
    "from src.router.{subpkg} import {name} as _module\n"
    "from src.router.{subpkg}.{name} import *  # noqa: F401,F403\n"
    "\n"
    "_sys.modules[__name__] = _module\n"
)


def _build_fake_repo(
    tmp_path: Path,
    *,
    wrappers: dict[str, str] | None = None,
    extra_files: dict[str, str] | None = None,
    create_collision_subpackages: bool = False,
) -> Path:
    """Materialize a tiny repo skeleton used to exercise the audit.

    ``wrappers`` maps ``legacy_name`` -> ``target_subpackage``; for each
    entry we materialize both the wrapper at
    ``src/router/<legacy_name>.py`` and the subpackage destination at
    ``src/router/<subpkg>/<legacy_name>.py``. ``extra_files`` overlays
    arbitrary files (relative paths -> contents).
    """

    wrappers = wrappers or {}
    extra_files = extra_files or {}

    router_dir = tmp_path / "src" / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    (router_dir / "__init__.py").write_text("", encoding="utf-8")

    seen_subpackages: set[str] = set()
    for legacy_name, subpkg in wrappers.items():
        subpkg_dir = router_dir / subpkg
        if subpkg not in seen_subpackages:
            subpkg_dir.mkdir(parents=True, exist_ok=True)
            (subpkg_dir / "__init__.py").write_text("", encoding="utf-8")
            seen_subpackages.add(subpkg)

        (subpkg_dir / f"{legacy_name}.py").write_text(
            f"VALUE = '{legacy_name}'\n", encoding="utf-8"
        )

        if create_collision_subpackages and (router_dir / legacy_name).is_dir():
            continue

        (router_dir / f"{legacy_name}.py").write_text(
            WRAPPER_TEMPLATE.format(subpkg=subpkg, name=legacy_name),
            encoding="utf-8",
        )

    for rel_path, content in extra_files.items():
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return tmp_path


def test_discover_legacy_wrapper_modules_skips_subpackage_shadow(tmp_path):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={
            "rde_database": "core",
            "calibration_bundle": "calibration",
        },
    )

    (repo / "src" / "router" / "calibration").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "router" / "calibration" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "router" / "calibration.py").write_text(
        WRAPPER_TEMPLATE.format(subpkg="calibration", name="calibration"),
        encoding="utf-8",
    )

    wrappers = discover_legacy_wrapper_modules(repo / "src" / "router")

    assert "rde_database" in wrappers
    assert wrappers["rde_database"] == "core"
    assert "calibration_bundle" in wrappers
    assert wrappers["calibration_bundle"] == "calibration"
    assert "calibration" not in wrappers, (
        "wrapper 'calibration.py' must be excluded because the subpackage "
        "'calibration/' shadows it"
    )


def test_audit_flags_suspicious_runtime_import(tmp_path):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
        extra_files={
            "src/router/some_runtime.py": (
                "from src.router.rde_database import RDEPoint\n"
            ),
        },
    )

    findings, wrappers, _ = run_audit(repo_root=repo)

    suspicious = [f for f in findings if f.category == "suspicious_runtime"]
    assert "rde_database" in wrappers
    assert len(suspicious) == 1
    assert suspicious[0].path == "src/router/some_runtime.py"
    assert suspicious[0].module == "rde_database"


def test_audit_treats_wrapper_self_reference_as_allowed(tmp_path):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
        extra_files={
            "src/router/rde_database.py": (
                "# explicit wrapper aliasing src.router.rde_database to the\n"
                "# src.router.core.rde_database subpackage module.\n"
                "from src.router.core import rde_database as _module\n"
                "from src.router.core.rde_database import *  # noqa: F401,F403\n"
                "# legacy alias for src.router.rde_database\n"
            ),
        },
    )

    findings, wrappers, _ = run_audit(repo_root=repo)

    suspicious = [f for f in findings if f.category == "suspicious_runtime"]
    wrapper_findings = [f for f in findings if f.category == "allowed_wrapper"]

    assert "rde_database" in wrappers
    assert suspicious == []
    assert any(f.module == "rde_database" for f in wrapper_findings)


def test_audit_treats_test_legacy_reference_as_allowed(tmp_path):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
        extra_files={
            "tests/test_legacy_compat.py": (
                "from src.router.rde_database import RDEPoint\n"
                "from src.router.core.rde_database import RDEPoint as NewRDEPoint\n"
                "\n"
                "def test_legacy_alias():\n"
                "    assert RDEPoint is NewRDEPoint\n"
            ),
        },
    )

    findings, _, _ = run_audit(repo_root=repo)

    suspicious = [f for f in findings if f.category == "suspicious_runtime"]
    allowed_tests = [f for f in findings if f.category == "allowed_test"]

    assert suspicious == []
    assert any(f.module == "rde_database" for f in allowed_tests)


def test_audit_clean_when_only_new_path_used(tmp_path):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
        extra_files={
            "src/router/another_runtime.py": (
                "from src.router.core.rde_database import RDEPoint\n"
            ),
        },
    )

    findings, _, _ = run_audit(repo_root=repo)
    suspicious = [f for f in findings if f.category == "suspicious_runtime"]

    assert suspicious == []


def test_audit_separates_scripts_and_docs(tmp_path):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
        extra_files={
            "scripts/use_legacy.ps1": (
                "python -m src.router.rde_database --help\n"
            ),
            "docs/legacy_usage.md": (
                "Historical reference: `src.router.rde_database`.\n"
            ),
        },
    )

    findings, _, _ = run_audit(repo_root=repo)

    by_category = {f.category for f in findings}
    assert "scripts" in by_category
    assert "docs" in by_category
    assert "suspicious_runtime" not in by_category


def test_audit_cli_returns_nonzero_on_suspicious(tmp_path, capsys):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
        extra_files={
            "src/router/some_runtime.py": (
                "from src.router.rde_database import RDEPoint\n"
            ),
        },
    )

    rc = audit_main(["--repo-root", str(repo)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "SUSPICIOUS" in captured.out


def test_audit_cli_returns_zero_when_clean(tmp_path, capsys):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
    )

    rc = audit_main(["--repo-root", str(repo)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "OK" in captured.out


def test_audit_cli_writes_json_out(tmp_path):
    repo = _build_fake_repo(
        tmp_path,
        wrappers={"rde_database": "core"},
        extra_files={
            "src/router/some_runtime.py": (
                "from src.router.rde_database import RDEPoint\n"
            ),
        },
    )

    out_path = tmp_path / "audit.json"
    rc = audit_main([
        "--repo-root", str(repo),
        "--json-out", str(out_path),
    ])

    assert rc == 1
    assert out_path.is_file()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] >= 1
    assert payload["summary"]["by_category"]["suspicious_runtime"] >= 1
    assert "rde_database" in payload["wrappers"]


def test_audit_module_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "src.router.observability.legacy_import_audit", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
