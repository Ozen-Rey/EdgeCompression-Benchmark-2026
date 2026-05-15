"""Static checks for ``scripts/run_router.ps1``.

The dispatcher resolves short scenario names to existing
``scripts/run_router_*.ps1`` smoke scripts. These tests do not execute
PowerShell; they only read the dispatcher source and verify that the
required scenarios are declared and that their target script files
exist on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DISPATCHER_PATH = SCRIPTS_DIR / "run_router.ps1"

REQUIRED_SCENARIOS: dict[str, str] = {
    "v02-backends": "run_router_v02_backends.ps1",
    "v09-content-aware": "run_router_v09_content_aware.ps1",
}


def test_dispatcher_script_exists() -> None:
    assert DISPATCHER_PATH.is_file(), (
        f"Unified smoke dispatcher missing: {DISPATCHER_PATH}"
    )


@pytest.mark.parametrize(
    ("scenario", "target_script"),
    sorted(REQUIRED_SCENARIOS.items()),
)
def test_dispatcher_declares_required_scenario(
    scenario: str, target_script: str
) -> None:
    text = DISPATCHER_PATH.read_text(encoding="utf-8")

    assert f'"{scenario}"' in text, (
        f"Dispatcher must declare scenario '{scenario}' in ScenarioMap"
    )
    assert target_script in text, (
        f"Dispatcher must route scenario '{scenario}' to '{target_script}'"
    )
    assert (SCRIPTS_DIR / target_script).is_file(), (
        f"Target smoke script '{target_script}' does not exist on disk"
    )
