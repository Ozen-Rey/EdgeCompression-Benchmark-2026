"""Static checks for ``scripts/run_router.ps1``.

Since v0.42.39 the dispatcher is self-contained: each public scenario
name maps to an ``Invoke-Scenario<Name>`` function defined inside
``scripts/run_router.ps1`` itself, and the standalone
``scripts/run_router_v*.ps1`` files have been removed from the
repository.

These tests do not execute PowerShell; they only read the dispatcher
source and the developer setup doc and verify structural invariants:

- every expected scenario is mapped to a handler function name and the
  matching function definition is present;
- the dispatcher does not retain delegation to the deleted legacy
  scripts (no ``run_router_v*.ps1`` filenames as targets, no external
  PowerShell launch of those files);
- the dispatcher carries the PowerShell hygiene flags (StrictMode,
  ``$ErrorActionPreference = "Stop"``, ``ExecutionPolicy Bypass`` when
  applicable);
- aggregate scenarios (``test``, ``smoke``, ``all``) behave as
  documented;
- the developer setup documentation lists the canonical invocations.
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
    "v02-backends":                       "Invoke-ScenarioV02Backends",
    "v05-config":                         "Invoke-ScenarioV05Config",
    "v06-validation":                     "Invoke-ScenarioV06Validation",
    "v07-experiments":                    "Invoke-ScenarioV07Experiments",
    "v08-system-aware":                   "Invoke-ScenarioV08SystemAware",
    "v09-content-aware":                  "Invoke-ScenarioV09ContentAware",
    "v09-content-aware-benchmark-table":  "Invoke-ScenarioV09ContentAwareBenchmarkTable",
    "v09-content-aware-overhead":         "Invoke-ScenarioV09ContentAwareOverhead",
    "v09-content-aware-paper-artifacts":  "Invoke-ScenarioV09ContentAwarePaperArtifacts",
    "v09-content-classifier-model":       "Invoke-ScenarioV09ContentClassifierModel",
    "v09-content-classifier-router":      "Invoke-ScenarioV09ContentClassifierRouter",
    "v09-content-metadata":               "Invoke-ScenarioV09ContentMetadata",
    "v09-content-oracle":                 "Invoke-ScenarioV09ContentOracle",
    "v09-image-features":                 "Invoke-ScenarioV09ImageFeatures",
    "v09-image-manifest":                 "Invoke-ScenarioV09ImageManifest",
    "v09-metadata-policy":                "Invoke-ScenarioV09MetadataPolicy",
    "v09-oracle-classifier":              "Invoke-ScenarioV09OracleClassifier",
    "v09-oracle-classifier-sweep":        "Invoke-ScenarioV09OracleClassifierSweep",
}

EXPECTED_AGGREGATE_SCENARIOS = ("test", "smoke", "all")

REMOVED_LEGACY_SCRIPTS = tuple(
    f"run_router_{name.replace('-', '_')}.ps1"
    for name in EXPECTED_SCENARIOS
) + ("run_router_v03_quick_calibration.ps1",)

SCENARIO_ENTRY_RE = re.compile(
    r'^\s*"(?P<scenario>v\d[\w-]*)"\s*=\s*"(?P<handler>Invoke-Scenario\w+)"',
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


@pytest.fixture(scope="module")
def dispatcher_text() -> str:
    return DISPATCHER_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declared_scenarios(dispatcher_text: str) -> dict[str, str]:
    return {
        m.group("scenario"): m.group("handler")
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
        f"Scenario '{scenario}' must route to handler "
        f"'{EXPECTED_SCENARIOS[scenario]}' (got '{declared_scenarios[scenario]}')"
    )


@pytest.mark.parametrize(
    ("scenario", "handler"),
    sorted(EXPECTED_SCENARIOS.items()),
)
def test_dispatcher_defines_handler_function(
    scenario: str, handler: str, dispatcher_text: str
) -> None:
    pattern = re.compile(rf"^function {re.escape(handler)} \{{", re.MULTILINE)
    assert pattern.search(dispatcher_text), (
        f"Dispatcher must define function '{handler}' for scenario '{scenario}'"
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


def test_scenario_map_contains_no_aggregate_names(
    declared_scenarios: dict[str, str],
) -> None:
    for aggregate in EXPECTED_AGGREGATE_SCENARIOS:
        assert aggregate not in declared_scenarios, (
            f"Aggregate scenario '{aggregate}' must not appear in ScenarioMap "
            "(it is handled separately and including it would risk recursion "
            "from the 'all' aggregate)."
        )


@pytest.mark.parametrize("legacy_script", REMOVED_LEGACY_SCRIPTS)
def test_legacy_smoke_scripts_are_removed_from_repo(legacy_script: str) -> None:
    path = SCRIPTS_DIR / legacy_script
    assert not path.exists(), (
        f"Legacy smoke script '{legacy_script}' must be removed from the "
        f"repository in v0.42.39 (still present at {path})."
    )


def test_dispatcher_does_not_reference_removed_legacy_scripts(
    dispatcher_text: str,
) -> None:
    for legacy_script in REMOVED_LEGACY_SCRIPTS:
        assert legacy_script not in dispatcher_text, (
            f"Dispatcher must not reference the removed legacy script "
            f"'{legacy_script}'. ScenarioMap targets are now handler "
            "function names, not file paths."
        )


def test_dispatcher_does_not_launch_external_legacy_powershell_scripts(
    dispatcher_text: str,
) -> None:
    """The pre-v0.42.39 dispatcher used ``& powershell -File <ScriptPath>``
    to delegate to legacy scripts. After inlining, there should be no
    PowerShell launch of any ``run_router_*.ps1`` file.
    """
    forbidden_patterns = (
        re.compile(r"-File\s+\$ScriptPath"),
        re.compile(r"powershell\s+-ExecutionPolicy\s+Bypass\s+-File\s+\.[\\/]+scripts"),
    )
    for pattern in forbidden_patterns:
        match = pattern.search(dispatcher_text)
        assert match is None, (
            f"Dispatcher must not launch external smoke scripts; found "
            f"pattern: '{match.group(0) if match else ''}'."
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


def test_dispatcher_checks_last_exit_code(dispatcher_text: str) -> None:
    assert "$LASTEXITCODE" in dispatcher_text, (
        "Dispatcher must read $LASTEXITCODE after native python invocations"
    )


def test_dispatcher_propagates_failure_via_throw(dispatcher_text: str) -> None:
    assert "throw" in dispatcher_text, (
        "Dispatcher must propagate failures via 'throw'"
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
    assert "Invoke-AggregateTest" in dispatcher_text
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
    assert "Invoke-PytestTempCleanup" in dispatcher_text
    assert ".pytest_tmp*" in dispatcher_text


def test_pytest_temp_cleanup_is_strict_mode_safe(dispatcher_text: str) -> None:
    cleanup_body_match = re.search(
        r"function Invoke-PytestTempCleanup \{(?P<body>.*?)\n\}",
        dispatcher_text,
        re.DOTALL,
    )
    assert cleanup_body_match is not None
    body = cleanup_body_match.group("body")

    assert "-LiteralPath" in body
    assert "[System.IO.FileSystemInfo]" in body
    assert "Join-Path" in body
    assert "Test-Path -LiteralPath" in body
    assert "Remove-Item -LiteralPath" in body
    assert "try" in body and "catch" in body


def test_aggregate_smoke_runs_recommended_pair(dispatcher_text: str) -> None:
    assert "SmokeAggregateOrder" in dispatcher_text
    smoke_block_start = dispatcher_text.find("$SmokeAggregateOrder")
    assert smoke_block_start != -1
    smoke_block_end = dispatcher_text.find(")", smoke_block_start)
    assert smoke_block_end != -1
    smoke_block = dispatcher_text[smoke_block_start:smoke_block_end]
    assert '"v02-backends"' in smoke_block
    assert '"v09-content-aware"' in smoke_block


def test_aggregate_all_iterates_over_scenario_map_only(
    dispatcher_text: str,
) -> None:
    all_body_match = re.search(
        r"function Invoke-AggregateAll \{(?P<body>.*?)\n\}",
        dispatcher_text,
        re.DOTALL,
    )
    assert all_body_match is not None
    all_body = all_body_match.group("body")

    assert "$ScenarioMap.Keys" in all_body
    for aggregate in EXPECTED_AGGREGATE_SCENARIOS:
        assert f'"{aggregate}"' not in all_body, (
            f"Aggregate 'all' must not reference aggregate '{aggregate}' "
            "in its body (would risk recursion)"
        )


def test_aggregate_dispatch_branch_routes_to_each_aggregate_handler(
    dispatcher_text: str,
) -> None:
    assert "$AggregateScenarios.Contains($Scenario)" in dispatcher_text
    for handler in ("Invoke-AggregateTest", "Invoke-AggregateSmoke", "Invoke-AggregateAll"):
        assert handler in dispatcher_text


def test_dispatcher_does_not_use_script_scoped_rows_state(
    dispatcher_text: str,
) -> None:
    """The pre-v0.42.39.1 v02-backends inlining wrote summary rows to
    a script-scoped ``$script:Rows`` from inside a nested function.
    Under ``Set-StrictMode -Version Latest`` that read raised
    "cannot be retrieved because it has not been set." Guard against
    the script-scope round-trip coming back in any scenario.
    """
    assert "$script:Rows" not in dispatcher_text, (
        "Dispatcher must not use $script:Rows for per-scenario state; "
        "keep summary collections function-local."
    )


def test_v02_backends_uses_function_local_summary_rows(
    dispatcher_text: str,
) -> None:
    v02_body_match = re.search(
        r"function Invoke-ScenarioV02Backends \{(?P<body>.*?)\n\}",
        dispatcher_text,
        re.DOTALL,
    )
    assert v02_body_match is not None, (
        "Could not locate the body of Invoke-ScenarioV02Backends"
    )
    body = v02_body_match.group("body")

    assert "$summaryRows = @()" in body, (
        "Invoke-ScenarioV02Backends must initialize a function-local "
        "$summaryRows collection before populating it"
    )
    assert "$summaryRows +=" in body, (
        "Invoke-ScenarioV02Backends must append to the function-local "
        "$summaryRows collection"
    )
    assert "$summaryRows | Export-Csv" in body, (
        "Invoke-ScenarioV02Backends must export the function-local "
        "$summaryRows collection to CSV"
    )


def test_developer_setup_documents_dispatcher_examples() -> None:
    text = DEVELOPER_SETUP_PATH.read_text(encoding="utf-8")
    for example in REQUIRED_DOC_EXAMPLES:
        assert example in text, (
            f"docs/developer_setup.md must document the dispatcher example "
            f"'{example}'"
        )
