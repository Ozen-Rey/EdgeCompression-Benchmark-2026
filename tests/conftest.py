"""Pytest configuration for the router test suite.

Provides ``scratch_root()``: a module-level accessor that returns
the session-scoped scratch directory the test helpers use as a base for
their per-file subtrees. The directory is allocated by pytest's
``tmp_path_factory`` (so its lifecycle is governed by ``--basetemp``),
which keeps test artifacts out of the repository tree.

This was introduced in v0.42.28 to retire the in-tree ``tests/_tmp/``
scratch directory. Test helpers used to call
``Path(__file__).with_name("_tmp") / "<namespace>"``; they now call
``scratch_root() / "<namespace>"`` instead. The namespace is
preserved so test outputs are still grouped per test file when
inspected for debugging.
"""

from pathlib import Path
from typing import Optional

import pytest

_TEST_SCRATCH_ROOT: Optional[Path] = None


@pytest.fixture(autouse=True, scope="session")
def _init_scratch_root(tmp_path_factory: pytest.TempPathFactory):
    global _TEST_SCRATCH_ROOT
    _TEST_SCRATCH_ROOT = tmp_path_factory.mktemp("router_scratch", numbered=False)
    try:
        yield _TEST_SCRATCH_ROOT
    finally:
        _TEST_SCRATCH_ROOT = None


def scratch_root() -> Path:
    """Return the session-scoped scratch root for in-test file helpers.

    Tests must call this from within an active pytest session (the
    autouse session fixture initializes the path before any test runs).
    """
    if _TEST_SCRATCH_ROOT is None:
        raise RuntimeError(
            "scratch_root() called outside of an active pytest session"
        )
    return _TEST_SCRATCH_ROOT
