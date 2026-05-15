<#
.SYNOPSIS
    Unified router smoke dispatcher.

.DESCRIPTION
    Resolves a scenario name to either a single smoke script under
    scripts/run_router_*.ps1 or one of the aggregate scenarios
    (``test``, ``smoke``, ``all``) and runs it.

    Single scenarios call the corresponding scripts/run_router_*.ps1
    via PowerShell with -ExecutionPolicy Bypass; the dispatcher does
    not duplicate any of their internal logic.

    Aggregate scenarios:
      - ``test``  runs the local Python verifications (legacy import
        audit, pytest with a private --basetemp, ``--help`` smoke for
        the router CLI in both ``python -m`` and direct-script form,
        and ``py_compile`` on the router core modules), then cleans up
        the local ``.pytest_tmp_*`` scratch directories.
      - ``smoke`` runs the two recommended scenarios (v02-backends,
        v09-content-aware) in order.
      - ``all``   runs every single scenario in ScenarioMap in
        deterministic insertion order (excluding the aggregates to
        prevent recursion).

    All steps propagate failures via ``$LASTEXITCODE`` and ``throw``;
    aggregates stop at the first failure.

.PARAMETER Scenario
    Scenario name. Use -Scenario list to print all known single and
    aggregate scenarios.

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario list

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario v02-backends

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario v09-content-aware

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario test

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario smoke

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario all
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Scenario
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScenarioMap = [ordered]@{
    "v02-backends"                       = "run_router_v02_backends.ps1"
    "v05-config"                         = "run_router_v05_config.ps1"
    "v06-validation"                     = "run_router_v06_validation.ps1"
    "v07-experiments"                    = "run_router_v07_experiments.ps1"
    "v08-system-aware"                   = "run_router_v08_system_aware.ps1"
    "v09-content-aware"                  = "run_router_v09_content_aware.ps1"
    "v09-content-aware-benchmark-table"  = "run_router_v09_content_aware_benchmark_table.ps1"
    "v09-content-aware-overhead"         = "run_router_v09_content_aware_overhead.ps1"
    "v09-content-aware-paper-artifacts"  = "run_router_v09_content_aware_paper_artifacts.ps1"
    "v09-content-classifier-model"       = "run_router_v09_content_classifier_model.ps1"
    "v09-content-classifier-router"      = "run_router_v09_content_classifier_router.ps1"
    "v09-content-metadata"               = "run_router_v09_content_metadata.ps1"
    "v09-content-oracle"                 = "run_router_v09_content_oracle.ps1"
    "v09-image-features"                 = "run_router_v09_image_features.ps1"
    "v09-image-manifest"                 = "run_router_v09_image_manifest.ps1"
    "v09-metadata-policy"                = "run_router_v09_metadata_policy.ps1"
    "v09-oracle-classifier"              = "run_router_v09_oracle_classifier.ps1"
    "v09-oracle-classifier-sweep"        = "run_router_v09_oracle_classifier_sweep.ps1"
}

$AggregateScenarios = [ordered]@{
    "test"  = "Python verifications (legacy_import_audit, pytest, --help smoke, py_compile)"
    "smoke" = "v02-backends, v09-content-aware (recommended smoke pair)"
    "all"   = "every single scenario in the ScenarioMap, in order"
}

$SmokeAggregateOrder = @("v02-backends", "v09-content-aware")

$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptsDir


function Assert-LastExitCode {
    param([string]$Description)
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed (exit $LASTEXITCODE): $Description"
    }
}


function Invoke-PytestTempCleanup {
    # Robust under Set-StrictMode -Version Latest: each pipeline item may
    # not be a DirectoryInfo (provider-dependent or odd error paths), so
    # never assume the .FullName property is present. Resolve the path
    # either from a FileSystemInfo instance or by joining the raw value
    # back onto the repo root, and re-check Test-Path before removing.
    $candidates = Get-ChildItem -LiteralPath $RepoRoot -Directory `
        -Filter ".pytest_tmp*" -Force -ErrorAction SilentlyContinue

    if (-not $candidates) {
        return
    }

    foreach ($item in $candidates) {
        if ($item -is [System.IO.FileSystemInfo]) {
            $path  = $item.FullName
            $label = $item.Name
        }
        else {
            $label = [string]$item
            $path  = Join-Path $RepoRoot $label
        }

        if (-not (Test-Path -LiteralPath $path)) {
            continue
        }

        try {
            Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
            Write-Host "[run_router] cleaned : $label"
        }
        catch {
            Write-Host ("[run_router] cleanup skip: {0} ({1})" -f $path, $_.Exception.Message)
        }
    }
}


function Invoke-SingleScenario {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not $ScenarioMap.Contains($Name)) {
        throw "Internal: scenario '$Name' is not in the single-scenario map."
    }

    $scriptName = $ScenarioMap[$Name]
    $scriptPath = Join-Path $ScriptsDir $scriptName

    if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "Smoke script for scenario '$Name' not found at: $scriptPath"
    }

    Write-Host "[run_router] scenario : $Name"
    Write-Host "[run_router] script   : $scriptPath"

    & powershell -ExecutionPolicy Bypass -File $scriptPath
    Assert-LastExitCode "scenario '$Name' ($scriptName)"
}


function Invoke-AggregateTest {
    Write-Host "[run_router] aggregate: test"

    Push-Location $RepoRoot
    try {
        Write-Host "[run_router] step     : legacy_import_audit"
        & python -m src.router.observability.legacy_import_audit
        Assert-LastExitCode "legacy_import_audit"

        Write-Host "[run_router] step     : pytest --basetemp .pytest_tmp_dispatcher"
        & python -m pytest -q --basetemp ".pytest_tmp_dispatcher"
        Assert-LastExitCode "pytest"

        Write-Host "[run_router] step     : python -m src.router.rde_router --help"
        & python -m src.router.rde_router --help | Out-Null
        Assert-LastExitCode "python -m src.router.rde_router --help"

        Write-Host "[run_router] step     : python src\router\rde_router.py --help"
        & python "src\router\rde_router.py" --help | Out-Null
        Assert-LastExitCode "python src\router\rde_router.py --help"

        Write-Host "[run_router] step     : py_compile router core modules"
        & python -m py_compile `
            "src\router\rde_router.py" `
            "src\router\pipeline.py" `
            "src\router\profile_runner.py"
        Assert-LastExitCode "py_compile"

        Write-Host "[run_router] step     : cleanup .pytest_tmp_*"
        Invoke-PytestTempCleanup
    }
    finally {
        Pop-Location
    }
}


function Invoke-AggregateSmoke {
    Write-Host "[run_router] aggregate: smoke ($($SmokeAggregateOrder -join ', '))"
    foreach ($name in $SmokeAggregateOrder) {
        Invoke-SingleScenario -Name $name
    }
}


function Invoke-AggregateAll {
    Write-Host "[run_router] aggregate: all ($($ScenarioMap.Count) scenarios)"
    foreach ($name in $ScenarioMap.Keys) {
        Invoke-SingleScenario -Name $name
    }
}


if ($Scenario -eq "list") {
    Write-Host "Single smoke scenarios:"
    foreach ($name in $ScenarioMap.Keys) {
        Write-Host ("  - {0,-36}-> {1}" -f $name, $ScenarioMap[$name])
    }
    Write-Host ""
    Write-Host "Aggregate scenarios:"
    foreach ($name in $AggregateScenarios.Keys) {
        Write-Host ("  - {0,-36}-> {1}" -f $name, $AggregateScenarios[$name])
    }
    return
}

if ($AggregateScenarios.Contains($Scenario)) {
    switch ($Scenario) {
        "test"  { Invoke-AggregateTest }
        "smoke" { Invoke-AggregateSmoke }
        "all"   { Invoke-AggregateAll }
    }
    exit 0
}

if ($ScenarioMap.Contains($Scenario)) {
    Invoke-SingleScenario -Name $Scenario
    exit 0
}

$knownSingle    = ($ScenarioMap.Keys | ForEach-Object { "  - $_" }) -join [Environment]::NewLine
$knownAggregate = ($AggregateScenarios.Keys | ForEach-Object { "  - $_" }) -join [Environment]::NewLine
throw @"
Unknown smoke scenario: '$Scenario'.
Known single scenarios:
$knownSingle
Known aggregate scenarios:
$knownAggregate
Run with -Scenario list to print this list without raising.
"@
