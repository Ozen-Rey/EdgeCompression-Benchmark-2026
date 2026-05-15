<#
.SYNOPSIS
    Unified router smoke dispatcher.

.DESCRIPTION
    Resolves a short scenario name to one of the existing
    scripts/run_router_*.ps1 smoke scripts and invokes it via
    PowerShell with -ExecutionPolicy Bypass. The dispatcher does not
    duplicate any of the internal logic of those scripts; it only
    routes -Scenario to a fixed mapping. The legacy scripts remain
    callable directly and are unchanged by this wrapper.

.PARAMETER Scenario
    Short name of the smoke scenario to run. Use -Scenario list to
    print all known scenarios.

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario v02-backends

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario v09-content-aware

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario list
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

$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Scenario -eq "list") {
    Write-Host "Known smoke scenarios:"
    foreach ($name in $ScenarioMap.Keys) {
        Write-Host ("  - {0,-36}-> {1}" -f $name, $ScenarioMap[$name])
    }
    return
}

if (-not $ScenarioMap.Contains($Scenario)) {
    $known = ($ScenarioMap.Keys | ForEach-Object { "  - $_" }) -join [Environment]::NewLine
    throw @"
Unknown smoke scenario: '$Scenario'.
Known scenarios:
$known
Run with -Scenario list to print this list without raising.
"@
}

$ScriptName = $ScenarioMap[$Scenario]
$ScriptPath = Join-Path $ScriptsDir $ScriptName

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Smoke script for scenario '$Scenario' not found at: $ScriptPath"
}

Write-Host "[run_router] scenario : $Scenario"
Write-Host "[run_router] script   : $ScriptPath"

& powershell -ExecutionPolicy Bypass -File $ScriptPath
$invokedExitCode = $LASTEXITCODE

if ($invokedExitCode -ne 0) {
    throw "Smoke scenario '$Scenario' failed: '$ScriptName' exited with code $invokedExitCode."
}

exit 0
