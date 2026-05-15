"""Static checks for ``scripts/run_router.ps1``.

The dispatcher resolves short scenario names to existing
``scripts/run_router_*.ps1`` smoke scripts. These tests do not execute
PowerShell; they only read the dispatcher source and the developer
setup doc and verify structural invariants:

- the dispatcher exists and declares every required scenario;
- each declared scenario points to an existing smoke script;
- the dispatcher carries the PowerShell hygiene flags (``StrictMode``,
  ``$ErrorActionPreference = "Stop"``, ``ExecutionPolicy Bypass``);
- the dispatcher checks ``$LASTEXITCODE`` and propagates failures via
  ``throw``;
- the dispatcher exposes ``-Scenario list``;
- the developer setup documentation shows the three canonical
  dispatcher invocations.

If a new scenario is added to the dispatcher without being added to the
expected set here, ``test_dispatcher_has_no_unexpected_scenarios``
fails so the test acts as a tripwire on the supported scenario surface.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DISPATCHER_PATH = SCRIPTS_DIR / "run_router.ps1"
DEVELOPER_SETUP_PATH = REPO_ROOT / "docs" / "developer_setup.md"

EXPECTED_SCENARIOS: dict[str, str] = {
    "v02-backends": "run_router_v02_backends.ps1",
    "v05-config": "run_router_v05_config.ps1",
    "v06-validation": "run_router_v06_validation.ps1",
    "v07-experiments": "run_router_v07_experiments.ps1",
    "v08-system-aware": "run_router_v08_system_aware.ps1",
    "v09-content-aware": "run_router_v09_content_aware.ps1",
    "v09-content-aware-benchmark-table": "run_router_v09_content_aware_benchmark_table.ps1",
    "v09-content-aware-overhead": "run_router_v09_content_aware_overhead.ps1",
    "v09-content-aware-paper-artifacts": "run_router_v09_content_aware_paper_artifacts.ps1",
    "v09-content-classifier-model": "run_router_v09_content_classifier_model.ps1",
    "v09-content-classifier-router": "run_router_v09_content_classifier_router.ps1",
    "v09-content-metadata": "run_router_v09_content_metadata.ps1",
    "v09-content-oracle": "run_router_v09_content_oracle.ps1",
    "v09-image-features": "run_router_v09_image_features.ps1",
    "v09-image-manifest": "run_router_v09_image_manifest.ps1",
    "v09-metadata-policy": "run_router_v09_metadata_policy.ps1",
    "v09-oracle-classifier": "run_router_v09_oracle_classifier.ps1",
    "v09-oracle-classifier-sweep": "run_router_v09_oracle_classifier_sweep.ps1",
}

SCENARIO_ENTRY_RE = re.compile(
    r'^\s*"(?P<scenario>v\d[\w-]*)"\s*=\s*"(?P<script>run_router_[\w_]+\.ps1)"',
    re.MULTILINE,
)

REQUIRED_DOC_EXAMPLES = (
    r".\scripts\run_router.ps1 -Scenario list",
    r".\scripts\run_router.ps1 -Scenario v02-backends",
    r".\scripts\run_router.ps1 -Scenario v09-content-aware",
    r".\scripts\run_router.ps1 -Scenario test",
    r".\scripts\run_router.ps1 -Scenario smoke",
    r".\scripts\run_router.ps1 -Scenario all",
)

EXPECTED_AGGREGATE_SCENARIOS = ("test", "smoke", "all")


@pytest.fixture(scope="module")
def dispatcher_text() -> str:
    return DISPATCHER_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declared_scenarios(dispatcher_text: str) -> dict[str, str]:
    return {
        m.group("scenario"): m.group("script")
        for m in SCENARIO_ENTRY_RE.finditer(dispatcher_text)
    }


def test_dispatcher_script_exists() -> None:
    assert DISPATCHER_PATH.is_file(), (
        f"Unified smoke dispatcher missing: {DISPATCHER_PATH}"
    )


@pytest.mark.parametrize("scenario", sorted(EXPECTED_SCENARIOS))
def test_dispatcher_declares_every_expected_scenario(
    scenario: str, declared_scenarios: dict[str, str]
) -> None:
    assert scenario in declared_scenarios, (
        f"Dispatcher must declare scenario '{scenario}' in ScenarioMap"
    )
    assert declared_scenarios[scenario] == EXPECTED_SCENARIOS[scenario], (
        f"Scenario '{scenario}' must route to "
        f"'{EXPECTED_SCENARIOS[scenario]}' (got '{declared_scenarios[scenario]}')"
    )


@pytest.mark.parametrize(
    ("scenario", "target_script"),
    sorted(EXPECTED_SCENARIOS.items()),
)
def test_every_expected_scenario_target_exists_on_disk(
    scenario: str, target_script: str
) -> None:
    target_path = SCRIPTS_DIR / target_script
    assert target_path.is_file(), (
        f"Target smoke script '{target_script}' for scenario '{scenario}' "
        f"does not exist on disk at {target_path}"
    )


def test_dispatcher_has_no_unexpected_scenarios(
    declared_scenarios: dict[str, str],
) -> None:
    extra = set(declared_scenarios) - set(EXPECTED_SCENARIOS)
    assert not extra, (
        f"Dispatcher declares scenarios that are not in the expected set: "
        f"{sorted(extra)}. Either remove them or add them to "
        f"EXPECTED_SCENARIOS in this test with justification."
    )


def test_dispatcher_supports_scenario_list(dispatcher_text: str) -> None:
    assert '$Scenario -eq "list"' in dispatcher_text, (
        "Dispatcher must handle -Scenario list as a discovery command"
    )


def test_dispatcher_uses_strict_mode(dispatcher_text: str) -> None:
    assert "Set-StrictMode -Version Latest" in dispatcher_text, (
        "Dispatcher must enable Set-StrictMode -Version Latest"
    )


def test_dispatcher_sets_error_action_preference_to_stop(
    dispatcher_text: str,
) -> None:
    assert '$ErrorActionPreference = "Stop"' in dispatcher_text, (
        "Dispatcher must set $ErrorActionPreference = \"Stop\""
    )


def test_dispatcher_invokes_with_execution_policy_bypass(
    dispatcher_text: str,
) -> None:
    assert "-ExecutionPolicy Bypass" in dispatcher_text, (
        "Dispatcher must invoke target scripts with -ExecutionPolicy Bypass"
    )


def test_dispatcher_checks_last_exit_code(dispatcher_text: str) -> None:
    assert "$LASTEXITCODE" in dispatcher_text, (
        "Dispatcher must read $LASTEXITCODE to detect target script failures"
    )


def test_dispatcher_propagates_failure_via_throw(dispatcher_text: str) -> None:
    assert "throw" in dispatcher_text, (
        "Dispatcher must propagate failures via 'throw'"
    )


def test_dispatcher_uses_test_path_on_target_script(
    dispatcher_text: str,
) -> None:
    assert "Test-Path" in dispatcher_text, (
        "Dispatcher must verify the target script with Test-Path before "
        "invoking it"
    )


def test_developer_setup_documents_dispatcher_examples() -> None:
    text = DEVELOPER_SETUP_PATH.read_text(encoding="utf-8")
    for example in REQUIRED_DOC_EXAMPLES:
        assert example in text, (
            f"docs/developer_setup.md must document the dispatcher example "
            f"'{example}'"
        )


@pytest.mark.parametrize("aggregate", EXPECTED_AGGREGATE_SCENARIOS)
def test_dispatcher_declares_aggregate_scenario(
    aggregate: str, dispatcher_text: str
) -> None:
    assert f'"{aggregate}"' in dispatcher_text, (
        f"Dispatcher must declare aggregate scenario '{aggregate}' in "
        "AggregateScenarios"
    )


def test_aggregate_test_runs_required_python_steps(
    dispatcher_text: str,
) -> None:
    assert "Invoke-AggregateTest" in dispatcher_text, (
        "Dispatcher must define an Invoke-AggregateTest function"
    )
    required_calls = (
        "src.router.observability.legacy_import_audit",
        '--basetemp ".pytest_tmp_dispatcher"',
        "src.router.rde_router --help",
        "src\\router\\rde_router.py",
        "py_compile",
    )
    for token in required_calls:
        assert token in dispatcher_text, (
            f"Aggregate 'test' must invoke '{token}'"
        )


def test_aggregate_test_cleans_pytest_tmp_directories(
    dispatcher_text: str,
) -> None:
    assert "Invoke-PytestTempCleanup" in dispatcher_text, (
        "Aggregate 'test' must call a cleanup helper for .pytest_tmp_*"
    )
    assert ".pytest_tmp*" in dispatcher_text, (
        "Cleanup helper must target the .pytest_tmp* directory pattern"
    )


def test_aggregate_smoke_runs_recommended_pair(dispatcher_text: str) -> None:
    assert "SmokeAggregateOrder" in dispatcher_text, (
        "Dispatcher must declare an explicit SmokeAggregateOrder list"
    )
    assert '"v02-backends"' in dispatcher_text
    assert '"v09-content-aware"' in dispatcher_text
    smoke_block_start = dispatcher_text.find("$SmokeAggregateOrder")
    assert smoke_block_start != -1
    smoke_block_end = dispatcher_text.find(")", smoke_block_start)
    assert smoke_block_end != -1
    smoke_block = dispatcher_text[smoke_block_start:smoke_block_end]
    assert '"v02-backends"' in smoke_block, (
        "SmokeAggregateOrder must include 'v02-backends'"
    )
    assert '"v09-content-aware"' in smoke_block, (
        "SmokeAggregateOrder must include 'v09-content-aware'"
    )


def test_aggregate_all_iterates_over_scenario_map_only(
    dispatcher_text: str,
) -> None:
    assert "Invoke-AggregateAll" in dispatcher_text, (
        "Dispatcher must define an Invoke-AggregateAll function"
    )

    all_body_match = re.search(
        r"function Invoke-AggregateAll \{(?P<body>.*?)\n\}",
        dispatcher_text,
        re.DOTALL,
    )
    assert all_body_match is not None, (
        "Could not locate the body of Invoke-AggregateAll"
    )
    all_body = all_body_match.group("body")

    assert "$ScenarioMap.Keys" in all_body, (
        "Aggregate 'all' must iterate over $ScenarioMap.Keys"
    )
    for aggregate in EXPECTED_AGGREGATE_SCENARIOS:
        assert f'"{aggregate}"' not in all_body, (
            f"Aggregate 'all' must not reference aggregate '{aggregate}' "
            "in its body (would risk recursion)"
        )


def test_aggregate_dispatch_branch_routes_to_each_aggregate_handler(
    dispatcher_text: str,
) -> None:
    assert "$AggregateScenarios.Contains($Scenario)" in dispatcher_text, (
        "Dispatcher must check $AggregateScenarios.Contains before routing"
    )
    for aggregate, handler in (
        ("test", "Invoke-AggregateTest"),
        ("smoke", "Invoke-AggregateSmoke"),
        ("all", "Invoke-AggregateAll"),
    ):
        assert handler in dispatcher_text, (
            f"Aggregate '{aggregate}' must route to handler {handler}"
        )
