"""Smoke checks for ``pyproject.toml`` package metadata.

The router pins its declared version through ``src.router.version`` and
exposes a stable ``rde-router`` console entry point via
``[project.scripts]``. These checks make sure both stay aligned so an
editable install (``pip install -e .``) keeps producing the documented
``rde-router`` command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.router.version import ROUTER_VERSION

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_pyproject() -> dict:
    if sys.version_info < (3, 11):
        pytest.skip("tomllib requires Python 3.11+")
    import tomllib

    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_pyproject_version_matches_router_version() -> None:
    data = _load_pyproject()
    assert data["project"]["version"] == ROUTER_VERSION


def test_pyproject_declares_rde_router_console_script() -> None:
    data = _load_pyproject()
    scripts = data["project"].get("scripts", {})
    assert scripts.get("rde-router") == "src.router.rde_router:main", (
        "pyproject.toml must expose 'rde-router' as a [project.scripts] entry "
        "pointing to src.router.rde_router:main"
    )


def test_pyproject_declares_test_extra() -> None:
    data = _load_pyproject()
    extras = data["project"].get("optional-dependencies", {})
    assert "pytest" in extras.get("test", []), (
        "pyproject.toml [project.optional-dependencies] must declare a 'test' "
        "extra containing pytest"
    )
