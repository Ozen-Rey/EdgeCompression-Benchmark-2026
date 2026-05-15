<#
.SYNOPSIS
    Unified router smoke dispatcher (self-contained).

.DESCRIPTION
    Single PowerShell entrypoint for every router smoke scenario.
    Before v0.42.39 each scenario lived in its own
    scripts/run_router_v*.ps1 file and this dispatcher delegated to
    them; starting with v0.42.39 the legacy scripts have been removed
    from the repository and the operative content is inlined here as
    Invoke-Scenario<Name> functions.

    ScenarioMap maps each public scenario name to its handler
    function. -Scenario list prints both the single scenarios and the
    aggregate scenarios (``test``, ``smoke``, ``all``). Unknown
    scenarios raise a clear error.

.PARAMETER Scenario
    Scenario name. Use -Scenario list to print all known scenarios.

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario list

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario test

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario smoke

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario all

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario v02-backends

.EXAMPLE
    .\scripts\run_router.ps1 -Scenario v09-content-aware
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Scenario
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScenarioMap = [ordered]@{
    "v02-backends"                       = "Invoke-ScenarioV02Backends"
    "v05-config"                         = "Invoke-ScenarioV05Config"
    "v06-validation"                     = "Invoke-ScenarioV06Validation"
    "v07-experiments"                    = "Invoke-ScenarioV07Experiments"
    "v08-system-aware"                   = "Invoke-ScenarioV08SystemAware"
    "v09-content-aware"                  = "Invoke-ScenarioV09ContentAware"
    "v09-content-aware-benchmark-table"  = "Invoke-ScenarioV09ContentAwareBenchmarkTable"
    "v09-content-aware-overhead"         = "Invoke-ScenarioV09ContentAwareOverhead"
    "v09-content-aware-paper-artifacts"  = "Invoke-ScenarioV09ContentAwarePaperArtifacts"
    "v09-content-classifier-model"       = "Invoke-ScenarioV09ContentClassifierModel"
    "v09-content-classifier-router"      = "Invoke-ScenarioV09ContentClassifierRouter"
    "v09-content-metadata"               = "Invoke-ScenarioV09ContentMetadata"
    "v09-content-oracle"                 = "Invoke-ScenarioV09ContentOracle"
    "v09-image-features"                 = "Invoke-ScenarioV09ImageFeatures"
    "v09-image-manifest"                 = "Invoke-ScenarioV09ImageManifest"
    "v09-metadata-policy"                = "Invoke-ScenarioV09MetadataPolicy"
    "v09-oracle-classifier"              = "Invoke-ScenarioV09OracleClassifier"
    "v09-oracle-classifier-sweep"        = "Invoke-ScenarioV09OracleClassifierSweep"
}

$AggregateScenarios = [ordered]@{
    "test"  = "Python verifications (legacy_import_audit, pytest, --help smoke, py_compile)"
    "smoke" = "v02-backends, v09-content-aware (recommended smoke pair)"
    "all"   = "every single scenario in the ScenarioMap, in order"
}

$SmokeAggregateOrder = @("v02-backends", "v09-content-aware")

$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptsDir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


function Invoke-LocalPytest {
    # Each legacy script ran ``pytest tests -q --basetemp .pytest_tmp_<basename>``
    # with cleanup before and after. Preserve the basetemp naming so any
    # debugging notes referencing those paths still apply.
    param([Parameter(Mandatory = $true)][string]$ScenarioName)

    $basetemp = ".pytest_tmp_run_router_" + ($ScenarioName -replace "-", "_")
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $basetemp
    & python -m pytest tests -q --basetemp $basetemp
    $code = $LASTEXITCODE
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $basetemp
    if ($code -ne 0) {
        throw "pytest failed with exit code $code"
    }
}


# Executable discovery helpers used by the backend smoke (v02-backends).
$script:WinGetPackageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"

function Get-ExecutableFromPath {
    param([Parameter(Mandatory = $true)][string]$Name)

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"

    try {
        $found = & where.exe $Name 2>$null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -eq 0 -and $null -ne $found) {
        return [string]($found | Select-Object -First 1)
    }

    return $null
}


function Find-ExecutableInWinGetPackages {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Test-Path $script:WinGetPackageRoot)) {
        return $null
    }

    $exeName = if ($Name.EndsWith(".exe")) { $Name } else { "$Name.exe" }
    $found = Get-ChildItem `
        -Path $script:WinGetPackageRoot `
        -Recurse `
        -Filter $exeName `
        -File `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($null -eq $found) {
        return $null
    }

    return $found.FullName
}


function Add-SessionPathDirectory {
    param([Parameter(Mandatory = $true)][string]$Directory)

    $pathEntries = $env:PATH -split [IO.Path]::PathSeparator
    if ($pathEntries -notcontains $Directory) {
        $env:PATH = $Directory + [IO.Path]::PathSeparator + $env:PATH
    }
}


function Resolve-SmokeExecutable {
    param([Parameter(Mandatory = $true)][string]$Name)

    $pathHit = Get-ExecutableFromPath -Name $Name
    if ($null -ne $pathHit -and $pathHit.Trim() -ne "") {
        return $pathHit
    }

    $winGetHit = Find-ExecutableInWinGetPackages -Name $Name
    if ($null -eq $winGetHit -or $winGetHit.Trim() -eq "") {
        return $null
    }

    Add-SessionPathDirectory -Directory (Split-Path -Parent $winGetHit)

    $pathHit = Get-ExecutableFromPath -Name $Name
    if ($null -ne $pathHit -and $pathHit.Trim() -ne "") {
        return $pathHit
    }

    return $winGetHit
}


# ---------------------------------------------------------------------------
# Scenario: v02-backends
# ---------------------------------------------------------------------------

function Invoke-ScenarioV02Backends {
    $Csv        = "results\images\image_4dataset_RDE_paper_ready.csv"
    $OutDir     = "results\routing_context"
    $InputImage = "test_images\input.png"

    New-Item -ItemType Directory -Force $OutDir | Out-Null

    if (-not (Test-Path $InputImage)) {
        throw "Input image not found: $InputImage"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Backend smoke executable diagnostics"
    Write-Host "============================================================"

    $ExecutableDiagnostics = @{}
    foreach ($exe in @("cjxl", "ffmpeg", "vvencapp")) {
        $ExecutableDiagnostics[$exe] = Resolve-SmokeExecutable -Name $exe
        $status = if ($null -ne $ExecutableDiagnostics[$exe]) {
            $ExecutableDiagnostics[$exe]
        }
        else {
            "missing"
        }
        Write-Host "${exe} found: $status"
    }

    $RequiredBackendExecutables = @(
        [pscustomobject]@{ Backend = "jxl_execute";  Executable = "cjxl" },
        [pscustomobject]@{ Backend = "hevc_execute"; Executable = "ffmpeg" }
    )
    $MissingBackendExecutables = @()
    foreach ($requirement in $RequiredBackendExecutables) {
        $exePath = $ExecutableDiagnostics[$requirement.Executable]
        if ($null -eq $exePath -or [string]::IsNullOrWhiteSpace([string]$exePath)) {
            $MissingBackendExecutables += (
                "$($requirement.Backend) requires $($requirement.Executable)"
            )
        }
    }

    if ($MissingBackendExecutables.Count -gt 0) {
        throw (
            "Backend smoke prerequisites missing: " +
            ($MissingBackendExecutables -join "; ") +
            ". Checked PATH with where.exe and session-local WinGet package paths under " +
            "$script:WinGetPackageRoot. Install or expose the missing executables before " +
            "running this smoke scenario. The router runtime remains strict and does " +
            "not auto-install or implicitly discover codecs."
        )
    }

    $CommonArgs = @(
        "--csv", $Csv,
        "--domain", "image",
        "--auto-weights",
        "--system-aware",
        "--capability-aware",
        "--strict-executables",
        "--power-mode", "ac",
        "--thermal-state", "nominal",
        "--network-profile", "normal",
        "--codec-col", "codec",
        "--config-col", "param",
        "--rate-col", "bpp",
        "--quality-col", "ssimulacra2",
        "--energy-col", "energy_per_image_j",
        "--time-col", "time_ms",
        "--aggregate-by-config",
        "--generate-command",
        "--execute",
        "--input", $InputImage
    )

    function Invoke-RouterBackendCase {
        param([string]$Name, [string[]]$ExtraArgs)
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Running v0.2 backend case: $Name"
        Write-Host "============================================================"
        & python -m src.router.rde_router @CommonArgs @ExtraArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Backend case '$Name' failed with exit code $LASTEXITCODE."
        }
    }

    Invoke-RouterBackendCase `
        -Name "jpeg_execute" `
        -ExtraArgs @(
            "--available-codecs", "JPEG",
            "--quality-target", "preview",
            "--quality-constraint-stat", "mean",
            "--quality-floor", "60",
            "--output", "test_images\v02_backend_jpeg.jpg",
            "--out", "$OutDir\v02_backend_jpeg.json"
        )

    Invoke-RouterBackendCase `
        -Name "jxl_execute" `
        -ExtraArgs @(
            "--available-codecs", "JXL",
            "--quality-target", "high",
            "--safe-mode",
            "--quality-constraint-stat", "min",
            "--quality-floor", "70",
            "--allow-degraded-fallback",
            "--near-quality-floor", "60",
            "--output", "test_images\v02_backend_jxl.jxl",
            "--out", "$OutDir\v02_backend_jxl.json"
        )

    Invoke-RouterBackendCase `
        -Name "hevc_execute" `
        -ExtraArgs @(
            "--available-codecs", "HEVC",
            "--quality-target", "high",
            "--safe-mode",
            "--quality-constraint-stat", "min",
            "--quality-floor", "85",
            "--allow-degraded-fallback",
            "--near-quality-floor", "75",
            "--output", "test_images\v02_backend_hevc.mp4",
            "--out", "$OutDir\v02_backend_hevc.json"
        )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Building v0.2 backend summary"
    Write-Host "============================================================"

    # Keep summary state function-local. The pre-v0.42.39.1 inlining
    # used a nested ``Add-BackendReportRow`` helper that wrote to a
    # script-scoped Rows collection; under Set-StrictMode -Version
    # Latest, reading that script-scoped variable from inside the
    # function raised "cannot be retrieved because it has not been
    # set." Avoid the script-scope round-trip entirely: iterate over
    # the case table and append each row in place.
    $summaryRows = @()
    $BackendCases = @(
        [pscustomobject]@{ Name = "jpeg_execute"; Path = "$OutDir\v02_backend_jpeg.json" }
        [pscustomobject]@{ Name = "jxl_execute";  Path = "$OutDir\v02_backend_jxl.json" }
        [pscustomobject]@{ Name = "hevc_execute"; Path = "$OutDir\v02_backend_hevc.json" }
    )

    foreach ($case in $BackendCases) {
        $r = Get-Content $case.Path -Raw | ConvertFrom-Json
        $s = $r.decision.selected
        $p = $r.execution_plan
        $e = $r.execution_result
        $summaryRows += [pscustomobject]@{
            case_name           = $case.Name
            selected_codec      = $s.codec
            selected_config     = $s.config
            decision_mode       = $r.decision.decision_mode
            backend             = $p.execution_backend
            can_execute         = $p.can_execute
            executed            = $e.executed
            success             = $e.success
            output              = $e.output
            rate                = $s.rate
            quality_mean        = $s.quality
            quality_guard_value = $s.quality_constraint_value
            energy              = $s.energy
            time_ms             = $s.time_ms
            J_RDE               = $s.cost
        }
    }

    $SummaryPath = "$OutDir\v02_backend_summary.csv"
    $summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryPath

    Write-Host "Summary written to: $SummaryPath"
    Write-Host ""
    Write-Host "Generated files:"
    Get-ChildItem test_images\v02_backend_* | Select-Object Name, Length, LastWriteTime

    Write-Host ""
    Write-Host "Done."
}


# ---------------------------------------------------------------------------
# Scenario: v05-config
# ---------------------------------------------------------------------------

function Invoke-ScenarioV05Config {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.5 config test ==="

    $PureReport     = "results/routing_context/v05_router_config_report.json"
    $OverrideReport = "results/routing_context/v05_router_config_override_jxl.json"

    Write-Host ""
    Write-Host "[1/3] Running pure config..."
    & python -m src.router.rde_router `
        --config configs/router_image_v05.json `
        --out $PureReport
    if ($LASTEXITCODE -ne 0) { throw "Pure config router run failed." }
    if (-not (Test-Path $PureReport)) {
        throw "Pure config report not created: $PureReport"
    }

    $pure = Get-Content $PureReport | ConvertFrom-Json
    if ($pure.router_config.enabled -ne $true) {
        throw "router_config.enabled is not true in pure config report."
    }
    Write-Host "Pure config selected:" $pure.decision.selected.codec $pure.decision.selected.config
    Write-Host "Experiment:" $pure.router_config.experiment_name

    Write-Host ""
    Write-Host "[2/3] Running CLI override: available-codecs=JXL..."
    & python -m src.router.rde_router `
        --config configs/router_image_v05.json `
        --available-codecs "JXL" `
        --out $OverrideReport
    if ($LASTEXITCODE -ne 0) { throw "Override router run failed." }
    if (-not (Test-Path $OverrideReport)) {
        throw "Override report not created: $OverrideReport"
    }

    $override = Get-Content $OverrideReport | ConvertFrom-Json
    if ($override.decision.selected.codec -ne "JXL") {
        throw "Expected selected codec JXL, got: $($override.decision.selected.codec)"
    }
    Write-Host "Override selected:" $override.decision.selected.codec $override.decision.selected.config

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v05-config"

    Write-Host ""
    Write-Host "v0.5 config test completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v06-validation
# ---------------------------------------------------------------------------

function Invoke-ScenarioV06Validation {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.6 validation ==="

    $PureReport       = "results/routing_context/v06_config_report.json"
    $JxlPlanReport    = "results/routing_context/v06_jxl_external_command_plan.json"
    $JxlExecuteReport = "results/routing_context/v06_jxl_execution_validation.json"

    Write-Host ""
    Write-Host "[1/4] Pure config run..."
    & python -m src.router.rde_router `
        --config configs/router_image_v05.json `
        --out $PureReport
    if (-not (Test-Path $PureReport)) { throw "Pure config report not created: $PureReport" }

    $pure = Get-Content $PureReport | ConvertFrom-Json
    if ($pure.run_manifest.enabled -ne $true) { throw "run_manifest.enabled is not true." }
    if ($pure.decision.decision_trace.enabled -ne $true) { throw "decision_trace.enabled is not true." }
    if ($null -eq $pure.decision.selected.cost_decomposition) {
        throw "cost_decomposition missing in selected decision."
    }
    Write-Host "Pure config selected:" $pure.decision.selected.codec $pure.decision.selected.config
    Write-Host "Run manifest git:" $pure.run_manifest.git.commit_short
    Write-Host "Selected reason:" $pure.decision.decision_trace.selected_reason

    Write-Host ""
    Write-Host "[2/4] JXL external_command plan..."
    & python -m src.router.rde_router `
        --config configs/router_image_v05.json `
        --available-codecs "JXL" `
        --out $JxlPlanReport
    if (-not (Test-Path $JxlPlanReport)) { throw "JXL plan report not created: $JxlPlanReport" }

    $jxlPlan = Get-Content $JxlPlanReport | ConvertFrom-Json
    if ($jxlPlan.decision.selected.codec -ne "JXL") {
        throw "Expected JXL, got: $($jxlPlan.decision.selected.codec)"
    }
    if ($jxlPlan.execution_plan.execution_backend -ne "external_command") {
        throw "Expected external_command backend, got: $($jxlPlan.execution_plan.execution_backend)"
    }
    Write-Host "JXL plan backend:" $jxlPlan.execution_plan.execution_backend
    Write-Host "JXL selected:" $jxlPlan.decision.selected.codec $jxlPlan.decision.selected.config

    Write-Host ""
    Write-Host "[3/4] JXL execute + output validation..."
    & python -m src.router.rde_router `
        --config configs/router_image_v05.json `
        --available-codecs "JXL" `
        --execute `
        --out $JxlExecuteReport
    if (-not (Test-Path $JxlExecuteReport)) {
        throw "JXL execute report not created: $JxlExecuteReport"
    }

    $jxlExec = Get-Content $JxlExecuteReport | ConvertFrom-Json
    if ($jxlExec.execution_result.success -ne $true) { throw "JXL execution failed." }
    if ($jxlExec.execution_validation.output_exists -ne $true) {
        throw "Execution validation failed: output does not exist."
    }
    if ($jxlExec.execution_validation.output_nonempty -ne $true) {
        throw "Execution validation failed: output is empty."
    }
    if ($jxlExec.execution_validation.extension_valid -ne $true) {
        throw "Execution validation failed: extension is invalid."
    }
    Write-Host "Execution output:" $jxlExec.execution_validation.output
    Write-Host "Output size:" $jxlExec.execution_validation.output_size_bytes
    Write-Host "Execution time ms:" $jxlExec.execution_validation.execution_time_ms

    Write-Host ""
    Write-Host "[4/4] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v06-validation"

    Write-Host ""
    Write-Host "v0.6 validation completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v07-experiments
# ---------------------------------------------------------------------------

function Invoke-ScenarioV07Experiments {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.7 experiment validation ==="

    $Suite     = "configs/experiments_image_v07.json"
    $Summary   = "results/routing_context/v07_experiment_summary.csv"
    $ReportDir = "results/routing_context/v07_experiments"

    Write-Host ""
    Write-Host "[1/4] Dry-run experiment suite..."
    & python -m src.router.experiment_manager --suite $Suite --dry-run

    Write-Host ""
    Write-Host "[2/4] Running experiment suite..."
    & python -m src.router.experiment_manager --suite $Suite

    if (-not (Test-Path $Summary))   { throw "Summary CSV not created: $Summary" }
    if (-not (Test-Path $ReportDir)) { throw "Report directory not created: $ReportDir" }

    $reports = Get-ChildItem $ReportDir -Filter "*.json"
    if ($reports.Count -lt 6) {
        throw "Expected at least 6 experiment reports, found $($reports.Count)"
    }

    $summaryRows = Import-Csv $Summary
    if ($summaryRows.Count -lt 6) {
        throw "Expected at least 6 summary rows, found $($summaryRows.Count)"
    }

    $failed = $summaryRows | Where-Object { $_.success -ne "True" }
    if ($failed.Count -gt 0) {
        throw "Some experiments failed: $($failed.experiment -join ', ')"
    }

    $jxlExec = $summaryRows | Where-Object { $_.experiment -eq "jxl_execute_validation" }
    if ($null -eq $jxlExec) { throw "Missing jxl_execute_validation row." }
    if ($jxlExec.execution_success -ne "True") { throw "jxl_execute_validation did not execute successfully." }
    if ($jxlExec.output_exists -ne "True")     { throw "jxl_execute_validation output does not exist." }
    if ($jxlExec.output_nonempty -ne "True")   { throw "jxl_execute_validation output is empty." }
    if ($jxlExec.extension_valid -ne "True")   { throw "jxl_execute_validation output extension is invalid." }

    Write-Host "Reports:" $reports.Count
    Write-Host "Summary rows:" $summaryRows.Count
    Write-Host "JXL execute output size:" $jxlExec.output_size_bytes

    Write-Host ""
    Write-Host "[3/4] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v07-experiments"

    Write-Host ""
    Write-Host "[4/4] Validation completed."
    Write-Host "v0.7 experiment validation completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v08-system-aware
# ---------------------------------------------------------------------------

function Invoke-ScenarioV08SystemAware {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.8 system-aware validation ==="

    $TmpDir       = ".tmp_manual_probe"
    $SyntheticCsv = "$TmpDir/v08_policy_change.csv"
    $BasicFeatures = "results/routing_context/v08_system_features_basic.json"
    $GpuFeatures   = "results/routing_context/v08_system_features_gpu.json"
    $ReportOnly    = "results/routing_context/v08_policy_change_report_only.json"
    $Apply         = "results/routing_context/v08_policy_change_apply.json"

    New-Item -ItemType Directory -Force $TmpDir | Out-Null

    @"
codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms
QualityHeavy,mode=quality,1.0,95.0,100.0,200.0
EnergyLight,mode=energy,1.2,85.0,1.0,20.0
"@ | Set-Content -Encoding UTF8 $SyntheticCsv

    Write-Host ""
    Write-Host "[1/5] Extracting basic system features..."
    & python -m src.router.adaptation.system_features --probe-level basic --out $BasicFeatures
    if (-not (Test-Path $BasicFeatures)) { throw "Basic system feature report not created." }

    Write-Host ""
    Write-Host "[2/5] Extracting GPU system features..."
    & python -m src.router.adaptation.system_features --probe-level gpu --out $GpuFeatures
    if (-not (Test-Path $GpuFeatures)) { throw "GPU system feature report not created." }

    Write-Host ""
    Write-Host "[3/5] Running synthetic policy case in report-only mode..."
    & python -m src.router.rde_router `
        --csv $SyntheticCsv `
        --codec-col codec --config-col param --rate-col bpp `
        --quality-col ssimulacra2 --energy-col energy_per_image_j --time-col time_ms `
        --domain image --auto-weights --quality-target high --quality-floor 80 `
        --system-policy --system-policy-mode report-only `
        --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
        --out $ReportOnly

    Write-Host ""
    Write-Host "[4/5] Running synthetic policy case in apply mode..."
    & python -m src.router.rde_router `
        --csv $SyntheticCsv `
        --codec-col codec --config-col param --rate-col bpp `
        --quality-col ssimulacra2 --energy-col energy_per_image_j --time-col time_ms `
        --domain image --auto-weights --quality-target high --quality-floor 80 `
        --system-policy --system-policy-mode apply `
        --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
        --out $Apply

    $reportOnlyJson = Get-Content $ReportOnly | ConvertFrom-Json
    $applyJson      = Get-Content $Apply | ConvertFrom-Json
    $reportOnlySelected = $reportOnlyJson.decision.selected.codec
    $applySelected      = $applyJson.decision.selected.codec

    Write-Host ""
    Write-Host "Report-only selected:" $reportOnlySelected
    Write-Host "Apply selected:      " $applySelected
    Write-Host "Apply weights:       E=$($applyJson.system_policy.effective_weights.w_E), R=$($applyJson.system_policy.effective_weights.w_R), D=$($applyJson.system_policy.effective_weights.w_D)"

    if ($reportOnlySelected -ne "QualityHeavy") {
        throw "Expected report-only mode to select QualityHeavy, got $reportOnlySelected"
    }
    if ($applySelected -ne "EnergyLight") {
        throw "Expected apply mode to select EnergyLight, got $applySelected"
    }

    Write-Host ""
    Write-Host "[extra] Running synthetic policy + system penalty apply case..."
    $PenaltyApply = "results/routing_context/v08_policy_penalty_apply.json"

    & python -m src.router.rde_router `
        --csv $SyntheticCsv `
        --codec-col codec --config-col param --rate-col bpp `
        --quality-col ssimulacra2 --energy-col energy_per_image_j --time-col time_ms `
        --domain image --auto-weights --quality-target high --quality-floor 80 `
        --system-policy --system-policy-mode apply `
        --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
        --system-penalty --system-penalty-mode apply --system-penalty-lambda 1.0 `
        --out $PenaltyApply

    $penaltyJson = Get-Content $PenaltyApply | ConvertFrom-Json
    if ($penaltyJson.decision.decision_trace.ranking_key -ne "minimize_J_total") {
        throw "Expected ranking by J_total."
    }
    if ($null -eq $penaltyJson.decision.selected.system_penalty) {
        throw "Missing system_penalty in selected candidate."
    }
    Write-Host "Penalty selected:" $penaltyJson.decision.selected.codec
    Write-Host "J_RDE:" $penaltyJson.decision.selected.cost
    Write-Host "J_total:" $penaltyJson.decision.selected.J_total

    Write-Host ""
    Write-Host "[5/5] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v08-system-aware"

    Write-Host ""
    Write-Host "v0.8 system-aware validation completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-aware
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentAware {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 content-aware validation ==="

    $BenchmarkCsv = "results/images/image_4dataset_RDE_paper_ready.csv"

    $OracleByImage = "results/routing_context/v09_content_oracle_by_image.csv"
    $OracleSummary = "results/routing_context/v09_content_oracle_summary.csv"

    $MetadataFeatures = "results/routing_context/v09_content_metadata_features.csv"
    $MetadataOracle   = "results/routing_context/v09_content_metadata_oracle.csv"
    $MetadataSummary  = "results/routing_context/v09_content_metadata_summary.csv"

    $DatasetPolicyDecisions = "results/routing_context/v09_metadata_policy_dataset_decisions.csv"
    $DatasetPolicyRules     = "results/routing_context/v09_metadata_policy_dataset_rules.csv"
    $DatasetPolicySummary   = "results/routing_context/v09_metadata_policy_dataset_summary.csv"

    $RouterReportOnly = "results/routing_context/v09_router_content_policy_report_only_tecnick.json"
    $RouterApply      = "results/routing_context/v09_router_content_policy_apply_tecnick.json"

    if (-not (Test-Path $BenchmarkCsv)) {
        throw "Benchmark CSV not found: $BenchmarkCsv"
    }

    Write-Host ""
    Write-Host "[1/6] Running content oracle/regret analysis..."
    & python -m src.router.analysis.content_oracle_analysis `
        --csv $BenchmarkCsv `
        --dataset-col dataset --image-col image --codec-col codec --config-col param `
        --rate-col bpp --quality-col ssimulacra2 --energy-col energy_per_image_j --time-col time_ms `
        --available-codecs "JPEG,JXL,HEVC" --quality-floor 80 --global-coverage-floor 1.0 `
        --wR 0.2 --wE 0.2 --wD 0.6 `
        --by-image-out $OracleByImage --summary-out $OracleSummary
    if (-not (Test-Path $OracleByImage)) { throw "Oracle by-image CSV not created." }
    if (-not (Test-Path $OracleSummary)) { throw "Oracle summary CSV not created." }

    Write-Host ""
    Write-Host "[2/6] Extracting metadata features and joining oracle labels..."
    & python -m src.router.adaptation.content_metadata_features `
        --csv $BenchmarkCsv --oracle-by-image $OracleByImage `
        --dataset-col dataset --image-col image `
        --width-col width --height-col height --pixels-col pixels `
        --features-out $MetadataFeatures --joined-out $MetadataOracle --summary-out $MetadataSummary
    if (-not (Test-Path $MetadataOracle))  { throw "Metadata/oracle CSV not created." }
    if (-not (Test-Path $MetadataSummary)) { throw "Metadata summary CSV not created." }

    Write-Host ""
    Write-Host "[3/6] Evaluating dataset-majority metadata policy..."
    & python -m src.router.adaptation.content_metadata_policy `
        --benchmark-csv $BenchmarkCsv --metadata-oracle-csv $MetadataOracle `
        --policy-key dataset --evaluation-mode leave-one-out `
        --quality-floor 80 --available-codecs "JPEG,JXL,HEVC" `
        --wR 0.2 --wE 0.2 --wD 0.6 `
        --decisions-out $DatasetPolicyDecisions `
        --rules-out $DatasetPolicyRules `
        --summary-out $DatasetPolicySummary
    if (-not (Test-Path $DatasetPolicyRules))   { throw "Dataset policy rules CSV not created." }
    if (-not (Test-Path $DatasetPolicySummary)) { throw "Dataset policy summary CSV not created." }

    Write-Host ""
    Write-Host "[4/6] Running router with source-aware content policy in report-only mode..."
    & python -m src.router.rde_router `
        --config configs/router_image_v08.json `
        --content-policy --content-policy-mode report-only `
        --content-policy-rules-file $DatasetPolicyRules `
        --content-policy-key dataset `
        --content-source tecnick --content-source-filter `
        --out $RouterReportOnly

    Write-Host ""
    Write-Host "[5/6] Running router with source-aware content policy in apply mode..."
    & python -m src.router.rde_router `
        --config configs/router_image_v08.json `
        --content-policy --content-policy-mode apply `
        --content-policy-rules-file $DatasetPolicyRules `
        --content-policy-key dataset `
        --content-source tecnick --content-source-filter `
        --out $RouterApply
    if (-not (Test-Path $RouterApply)) { throw "Router apply report not created." }

    Write-Host ""
    Write-Host "[validation] Checking content-aware router report..."
    $ApplyJson = Get-Content $RouterApply | ConvertFrom-Json
    if ($ApplyJson.content_policy.applied -ne $true) {
        throw "Expected content_policy.applied = true."
    }
    if ($ApplyJson.decision.selected.codec -ne "JPEG") {
        throw "Expected selected codec JPEG, got $($ApplyJson.decision.selected.codec)."
    }
    if ($ApplyJson.decision.selected.config -ne "q=85") {
        throw "Expected selected config q=85, got $($ApplyJson.decision.selected.config)."
    }
    if ($ApplyJson.decision.decision_trace.selected_reason -ne "content_policy_preferred_candidate") {
        throw "Expected selected_reason content_policy_preferred_candidate."
    }
    if ($ApplyJson.content_filter.applied -ne $true) {
        throw "Expected content_filter.applied = true."
    }

    Write-Host ""
    Write-Host "[summary] Oracle summary:"
    Import-Csv $OracleSummary | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[summary] Dataset metadata policy summary:"
    Import-Csv $DatasetPolicySummary | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[summary] Router apply result:"
    Write-Host "Selected:" $ApplyJson.decision.selected.codec $ApplyJson.decision.selected.config
    Write-Host "Selected reason:" $ApplyJson.decision.decision_trace.selected_reason
    Write-Host "Content policy applied:" $ApplyJson.content_policy.applied
    Write-Host "Content filter:" $ApplyJson.content_filter.column "=" $ApplyJson.content_filter.value "(" $ApplyJson.content_filter.before_count "->" $ApplyJson.content_filter.after_count ")"

    Write-Host ""
    Write-Host "[6/6] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-aware"

    Write-Host ""
    Write-Host "v0.9 content-aware validation completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-oracle
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentOracle {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 content oracle analysis ==="

    $Csv         = "results/images/image_4dataset_RDE_paper_ready.csv"
    $ByImageOut  = "results/routing_context/v09_content_oracle_by_image.csv"
    $SummaryOut  = "results/routing_context/v09_content_oracle_summary.csv"

    if (-not (Test-Path $Csv)) { throw "Benchmark CSV not found: $Csv" }

    Write-Host ""
    Write-Host "[1/3] Running content oracle analysis..."
    & python -m src.router.analysis.content_oracle_analysis `
        --csv $Csv `
        --dataset-col dataset --image-col image --codec-col codec --config-col param `
        --rate-col bpp --quality-col ssimulacra2 --energy-col energy_per_image_j --time-col time_ms `
        --available-codecs "JPEG,JXL,HEVC" --quality-floor 80 --global-coverage-floor 1.0 `
        --wR 0.2 --wE 0.2 --wD 0.6 `
        --by-image-out $ByImageOut --summary-out $SummaryOut
    if (-not (Test-Path $ByImageOut)) { throw "By-image oracle CSV not created." }
    if (-not (Test-Path $SummaryOut)) { throw "Summary oracle CSV not created." }

    Write-Host ""
    Write-Host "[2/3] Preview summary..."
    Import-Csv $SummaryOut | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-oracle"

    Write-Host ""
    Write-Host "v0.9 content oracle analysis completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-metadata
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentMetadata {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 metadata content features ==="

    $Csv           = "results/images/image_4dataset_RDE_paper_ready.csv"
    $OracleByImage = "results/routing_context/v09_content_oracle_by_image.csv"
    $FeaturesOut   = "results/routing_context/v09_content_metadata_features.csv"
    $JoinedOut     = "results/routing_context/v09_content_metadata_oracle.csv"
    $SummaryOut    = "results/routing_context/v09_content_metadata_summary.csv"

    if (-not (Test-Path $Csv)) { throw "Benchmark CSV not found: $Csv" }

    if (-not (Test-Path $OracleByImage)) {
        Write-Host "Oracle by-image CSV not found. Running oracle analysis first..."
        & python -m src.router.analysis.content_oracle_analysis `
            --csv $Csv `
            --dataset-col dataset --image-col image --codec-col codec --config-col param `
            --rate-col bpp --quality-col ssimulacra2 --energy-col energy_per_image_j --time-col time_ms `
            --available-codecs "JPEG,JXL,HEVC" --quality-floor 80 --global-coverage-floor 1.0 `
            --wR 0.2 --wE 0.2 --wD 0.6 `
            --by-image-out $OracleByImage `
            --summary-out "results/routing_context/v09_content_oracle_summary.csv"
    }

    Write-Host ""
    Write-Host "[1/3] Extracting metadata features and joining oracle labels..."
    & python -m src.router.adaptation.content_metadata_features `
        --csv $Csv --oracle-by-image $OracleByImage `
        --dataset-col dataset --image-col image `
        --width-col width --height-col height --pixels-col pixels `
        --features-out $FeaturesOut --joined-out $JoinedOut --summary-out $SummaryOut
    if (-not (Test-Path $FeaturesOut)) { throw "Features CSV not created." }
    if (-not (Test-Path $JoinedOut))   { throw "Joined metadata/oracle CSV not created." }
    if (-not (Test-Path $SummaryOut))  { throw "Summary CSV not created." }

    Write-Host ""
    Write-Host "[2/3] Preview metadata summary..."
    Import-Csv $SummaryOut | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-metadata"

    Write-Host ""
    Write-Host "v0.9 metadata content features completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-image-manifest
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ImageManifest {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 image manifest ==="

    $Csv = "results/images/image_4dataset_RDE_paper_ready.csv"
    $Out = "results/routing_context/v09_image_manifest.csv"
    $Roots = "kodak=datasets/images/kodak;div2k_valid=datasets/images/div2k_valid;clic2020=datasets/images/clic2020;tecnick=datasets/images/tecnick"

    & python -m src.router.adaptation.content_image_manifest `
        --csv $Csv --roots $Roots `
        --dataset-col dataset --image-col image `
        --out $Out
    if (-not (Test-Path $Out)) { throw "Manifest not created." }

    $Rows = Import-Csv $Out
    if ($Rows.Count -ne 96) {
        throw "Expected 96 manifest rows, got $($Rows.Count)."
    }

    Write-Host ""
    Write-Host "Manifest counts:"
    $Rows | Group-Object dataset | Select-Object Count, Name | Format-Table -AutoSize

    Write-Host ""
    Write-Host "Mode/dimension checks:"
    $Rows | Group-Object dataset, mode | Select-Object Count, Name | Format-Table -AutoSize

    $TecnickBad = $Rows | Where-Object {
        $_.dataset -eq "tecnick" -and (
            $_.width -ne "1200" -or
            $_.height -ne "1200" -or
            $_.mode -ne "RGB"
        )
    }
    if ($TecnickBad.Count -gt 0) {
        throw "Tecnick validation failed: expected all images 1200x1200 RGB."
    }

    Write-Host ""
    Write-Host "Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-image-manifest"

    Write-Host ""
    Write-Host "v0.9 image manifest completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-image-features
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ImageFeatures {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 image pixel features ==="

    $Manifest    = "results/routing_context/v09_image_manifest.csv"
    $FeaturesOut = "results/routing_context/v09_image_pixel_features.csv"
    $SummaryOut  = "results/routing_context/v09_image_pixel_features_summary.csv"

    if (-not (Test-Path $Manifest)) {
        Write-Host "Image manifest not found. Running manifest builder first..."
        Invoke-ScenarioV09ImageManifest
    }

    Write-Host ""
    Write-Host "[1/3] Extracting pixel-level image features..."
    & python -m src.router.adaptation.content_image_features `
        --manifest $Manifest --resize-long-side 256 `
        --features-out $FeaturesOut --summary-out $SummaryOut
    if (-not (Test-Path $FeaturesOut)) { throw "Pixel features CSV not created." }
    if (-not (Test-Path $SummaryOut))  { throw "Pixel feature summary CSV not created." }

    $Rows = Import-Csv $FeaturesOut
    if ($Rows.Count -ne 96) {
        throw "Expected 96 feature rows, got $($Rows.Count)."
    }

    Write-Host ""
    Write-Host "[2/3] Feature summary..."
    Import-Csv $SummaryOut | Format-Table -AutoSize

    Write-Host ""
    Write-Host "Dataset counts:"
    $Rows | Group-Object dataset | Select-Object Count, Name | Format-Table -AutoSize

    Write-Host ""
    Write-Host "Texture classes:"
    $Rows | Group-Object texture_class | Select-Object Count, Name | Format-Table -AutoSize

    Write-Host ""
    Write-Host "Edge classes:"
    $Rows | Group-Object edge_class | Select-Object Count, Name | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-image-features"

    Write-Host ""
    Write-Host "v0.9 image pixel features completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-metadata-policy
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09MetadataPolicy {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 metadata content policy ==="

    $BenchmarkCsv      = "results/images/image_4dataset_RDE_paper_ready.csv"
    $MetadataOracleCsv = "results/routing_context/v09_content_metadata_oracle.csv"

    if (-not (Test-Path $BenchmarkCsv)) {
        throw "Benchmark CSV not found: $BenchmarkCsv"
    }
    if (-not (Test-Path $MetadataOracleCsv)) {
        Write-Host "Metadata/oracle CSV not found. Running metadata extraction first..."
        Invoke-ScenarioV09ContentMetadata
    }

    $PolicyKeys = @("dataset", "resolution_class", "orientation_class")
    foreach ($PolicyKey in $PolicyKeys) {
        Write-Host ""
        Write-Host "[policy=$PolicyKey] Evaluating metadata policy..."

        $DecisionsOut = "results/routing_context/v09_metadata_policy_${PolicyKey}_decisions.csv"
        $RulesOut     = "results/routing_context/v09_metadata_policy_${PolicyKey}_rules.csv"
        $SummaryOut   = "results/routing_context/v09_metadata_policy_${PolicyKey}_summary.csv"

        & python -m src.router.adaptation.content_metadata_policy `
            --benchmark-csv $BenchmarkCsv --metadata-oracle-csv $MetadataOracleCsv `
            --policy-key $PolicyKey --evaluation-mode leave-one-out `
            --quality-floor 80 --available-codecs "JPEG,JXL,HEVC" `
            --wR 0.2 --wE 0.2 --wD 0.6 `
            --decisions-out $DecisionsOut --rules-out $RulesOut --summary-out $SummaryOut

        if (-not (Test-Path $SummaryOut)) {
            throw "Summary CSV not created for policy=$PolicyKey"
        }

        Write-Host ""
        Write-Host "Summary for policy=$PolicyKey"
        Import-Csv $SummaryOut | Format-Table -AutoSize
    }

    Write-Host ""
    Write-Host "[final] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-metadata-policy"

    Write-Host ""
    Write-Host "v0.9 metadata content policy completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-oracle-classifier
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09OracleClassifier {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 oracle classifier baseline ==="

    $BenchmarkCsv      = "results/images/image_4dataset_RDE_paper_ready.csv"
    $MetadataOracleCsv = "results/routing_context/v09_content_metadata_oracle.csv"
    $PixelFeaturesCsv  = "results/routing_context/v09_image_pixel_features.csv"

    if (-not (Test-Path $MetadataOracleCsv)) {
        Write-Host "Metadata/oracle CSV not found. Running metadata pipeline..."
        Invoke-ScenarioV09ContentMetadata
    }
    if (-not (Test-Path $PixelFeaturesCsv)) {
        Write-Host "Pixel features CSV not found. Running pixel feature extraction..."
        Invoke-ScenarioV09ImageFeatures
    }

    $FeatureSets = @("metadata_no_source", "pixel_no_source", "all_no_source", "all_with_source")
    foreach ($FeatureSet in $FeatureSets) {
        Write-Host ""
        Write-Host "[classifier=$FeatureSet] Evaluating leave-one-out kNN oracle classifier..."

        $DecisionsOut = "results/routing_context/v09_oracle_classifier_${FeatureSet}_decisions.csv"
        $SummaryOut   = "results/routing_context/v09_oracle_classifier_${FeatureSet}_summary.csv"

        & python -m src.router.analysis.content_oracle_classifier `
            --benchmark-csv $BenchmarkCsv `
            --metadata-oracle-csv $MetadataOracleCsv `
            --pixel-features-csv $PixelFeaturesCsv `
            --feature-set $FeatureSet --k 3 `
            --quality-floor 80 --available-codecs "JPEG,JXL,HEVC" `
            --wR 0.2 --wE 0.2 --wD 0.6 `
            --decisions-out $DecisionsOut --summary-out $SummaryOut

        if (-not (Test-Path $SummaryOut)) {
            throw "Classifier summary not created for feature_set=$FeatureSet"
        }

        Write-Host ""
        Write-Host "Summary for feature_set=$FeatureSet"
        Import-Csv $SummaryOut | Format-Table -AutoSize
    }

    Write-Host ""
    Write-Host "[final] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-oracle-classifier"

    Write-Host ""
    Write-Host "v0.9 oracle classifier baseline completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-oracle-classifier-sweep
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09OracleClassifierSweep {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 oracle classifier sweep ==="

    $BenchmarkCsv      = "results/images/image_4dataset_RDE_paper_ready.csv"
    $MetadataOracleCsv = "results/routing_context/v09_content_metadata_oracle.csv"
    $PixelFeaturesCsv  = "results/routing_context/v09_image_pixel_features.csv"
    $SummaryOut        = "results/routing_context/v09_oracle_classifier_sweep_summary.csv"
    $DecisionsOut      = "results/routing_context/v09_oracle_classifier_sweep_decisions.csv"

    if (-not (Test-Path $MetadataOracleCsv)) {
        Write-Host "Metadata/oracle CSV not found. Running metadata pipeline..."
        Invoke-ScenarioV09ContentMetadata
    }
    if (-not (Test-Path $PixelFeaturesCsv)) {
        Write-Host "Pixel features CSV not found. Running pixel feature extraction..."
        Invoke-ScenarioV09ImageFeatures
    }

    Write-Host ""
    Write-Host "[1/3] Running classifier sweep..."
    & python -m src.router.analysis.content_oracle_classifier_sweep `
        --benchmark-csv $BenchmarkCsv `
        --metadata-oracle-csv $MetadataOracleCsv `
        --pixel-features-csv $PixelFeaturesCsv `
        --feature-sets "metadata_no_source,pixel_no_source,all_no_source,all_with_source" `
        --k-values "1,3,5,7,9,11" `
        --evaluation-modes "leave_one_image_out,leave_one_dataset_out" `
        --quality-floor 80 --available-codecs "JPEG,JXL,HEVC" `
        --wR 0.2 --wE 0.2 --wD 0.6 `
        --summary-out $SummaryOut --decisions-out $DecisionsOut

    if (-not (Test-Path $SummaryOut))   { throw "Sweep summary not created." }
    if (-not (Test-Path $DecisionsOut)) { throw "Sweep decisions not created." }

    Write-Host ""
    Write-Host "[2/3] Top sweep rows by mean regret..."
    Import-Csv $SummaryOut |
        Sort-Object { [double]$_.mean_regret } |
        Select-Object evaluation_mode, feature_set, k, accuracy, mean_regret, relative_regret_reduction, fallback_rate |
        Format-Table -AutoSize

    Write-Host ""
    Write-Host "[2b/3] Best row per evaluation mode..."
    Import-Csv $SummaryOut |
        Group-Object evaluation_mode |
        ForEach-Object {
            $_.Group |
                Sort-Object { [double]$_.mean_regret } |
                Select-Object -First 1 evaluation_mode, feature_set, k, accuracy, mean_regret, relative_regret_reduction, fallback_rate
        } |
        Format-Table -AutoSize

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-oracle-classifier-sweep"

    Write-Host ""
    Write-Host "v0.9 oracle classifier sweep completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-aware-benchmark-table
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentAwareBenchmarkTable {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 content-aware benchmark table ==="

    $OracleSummary         = "results/routing_context/v09_content_oracle_summary.csv"
    $DatasetPolicySummary  = "results/routing_context/v09_metadata_policy_dataset_summary.csv"
    $ClassifierSweepSummary = "results/routing_context/v09_oracle_classifier_sweep_summary.csv"
    $PaperTable            = "results/routing_context/v09_content_aware_benchmark_table.csv"
    $AllMethods            = "results/routing_context/v09_content_aware_benchmark_all_methods.csv"

    if (-not (Test-Path $OracleSummary)) {
        Write-Host "Oracle summary not found. Running content-aware validation..."
        Invoke-ScenarioV09ContentAware
    }
    if (-not (Test-Path $DatasetPolicySummary)) {
        Write-Host "Dataset policy summary not found. Running metadata policy..."
        Invoke-ScenarioV09MetadataPolicy
    }
    if (-not (Test-Path $ClassifierSweepSummary)) {
        Write-Host "Classifier sweep summary not found. Running classifier sweep..."
        Invoke-ScenarioV09OracleClassifierSweep
    }

    Write-Host ""
    Write-Host "[1/3] Building benchmark tables..."
    & python -m src.router.analysis.content_aware_benchmark_table `
        --oracle-summary $OracleSummary `
        --dataset-policy-summary $DatasetPolicySummary `
        --classifier-sweep-summary $ClassifierSweepSummary `
        --paper-table-out $PaperTable --all-methods-out $AllMethods

    if (-not (Test-Path $PaperTable)) { throw "Paper benchmark table not created." }
    if (-not (Test-Path $AllMethods)) { throw "All-methods benchmark table not created." }

    Write-Host ""
    Write-Host "[2/3] Paper table:"
    Import-Csv $PaperTable |
        Select-Object method_id, evaluation_protocol, deployment_setting, feature_set, k, accuracy, mean_regret, relative_regret_reduction, fallback_rate |
        Format-Table -AutoSize

    Write-Host ""
    Write-Host "[2b/3] Best deployable rows by mean regret:"
    Import-Csv $AllMethods |
        Where-Object { $_.uses_oracle -eq "False" } |
        Sort-Object { [double]$_.mean_regret } |
        Select-Object -First 12 method_id, evaluation_protocol, feature_set, k, accuracy, mean_regret, relative_regret_reduction, fallback_rate |
        Format-Table -AutoSize

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-aware-benchmark-table"

    Write-Host ""
    Write-Host "v0.9 content-aware benchmark table completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-aware-overhead
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentAwareOverhead {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 content-aware overhead and sensitivity ==="

    $PixelFeatures  = "results/routing_context/v09_image_pixel_features.csv"
    $BenchmarkCsv   = "results/images/image_4dataset_RDE_paper_ready.csv"
    $SweepSummary   = "results/routing_context/v09_oracle_classifier_sweep_summary.csv"
    $OverheadOut    = "results/routing_context/v09_content_aware_overhead_table.csv"
    $SensitivityOut = "results/routing_context/v09_knn_sensitivity_table.csv"

    if (-not (Test-Path $PixelFeatures)) {
        Write-Host "Pixel features not found. Running pixel feature extraction..."
        Invoke-ScenarioV09ImageFeatures
    }
    if (-not (Test-Path $SweepSummary)) {
        Write-Host "Classifier sweep not found. Running classifier sweep..."
        Invoke-ScenarioV09OracleClassifierSweep
    }

    Write-Host ""
    Write-Host "[1/3] Building overhead and sensitivity tables..."
    & python -m src.router.analysis.content_aware_overhead_analysis `
        --pixel-features-csv $PixelFeatures --benchmark-csv $BenchmarkCsv `
        --classifier-sweep-summary $SweepSummary `
        --dataset-col dataset --codec-col codec --config-col param --time-col time_ms `
        --overhead-out $OverheadOut --sensitivity-out $SensitivityOut

    if (-not (Test-Path $OverheadOut))    { throw "Overhead table not created." }
    if (-not (Test-Path $SensitivityOut)) { throw "Sensitivity table not created." }

    Write-Host ""
    Write-Host "[2/3] Overhead table:"
    Import-Csv $OverheadOut |
        Select-Object component, case_id, scope, dataset, codec, config, num_samples, mean_ms, median_ms, p90_ms, pixel_feature_mean_over_this_mean |
        Format-Table -AutoSize

    Write-Host ""
    Write-Host "[2b/3] k-sensitivity for metadata_no_source and pixel_no_source:"
    Import-Csv $SensitivityOut |
        Where-Object { $_.feature_set -in @("metadata_no_source", "pixel_no_source") } |
        Select-Object evaluation_mode, feature_set, k, accuracy, mean_regret, relative_regret_reduction, fallback_rate, best_for_evaluation_mode, best_for_feature_set |
        Format-Table -AutoSize

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-aware-overhead"

    Write-Host ""
    Write-Host "v0.9 content-aware overhead and sensitivity completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-aware-paper-artifacts
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentAwarePaperArtifacts {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 paper artifacts ==="

    $BenchmarkTable   = "results/routing_context/v09_content_aware_benchmark_table.csv"
    $OverheadTable    = "results/routing_context/v09_content_aware_overhead_table.csv"
    $SensitivityTable = "results/routing_context/v09_knn_sensitivity_table.csv"
    $OracleSummary    = "results/routing_context/v09_content_oracle_summary.csv"
    $OutDir           = "results/routing_context/paper_artifacts_v09"

    if (-not (Test-Path $BenchmarkTable)) {
        Write-Host "Benchmark table not found. Running benchmark table scenario..."
        Invoke-ScenarioV09ContentAwareBenchmarkTable
    }
    if (-not (Test-Path $OverheadTable) -or -not (Test-Path $SensitivityTable)) {
        Write-Host "Overhead/sensitivity tables not found. Running overhead scenario..."
        Invoke-ScenarioV09ContentAwareOverhead
    }
    if (-not (Test-Path $OracleSummary)) {
        Write-Host "Oracle summary not found. Running oracle scenario..."
        Invoke-ScenarioV09ContentOracle
    }

    Write-Host ""
    Write-Host "[1/3] Building paper-ready tables and figures..."
    & python -m src.router.analysis.content_aware_paper_artifacts `
        --benchmark-table $BenchmarkTable --overhead-table $OverheadTable `
        --sensitivity-table $SensitivityTable --oracle-summary $OracleSummary `
        --out-dir $OutDir

    if (-not (Test-Path "$OutDir\v09_content_aware_final_table.csv")) {
        throw "Main paper table CSV was not created."
    }
    if (-not (Test-Path "$OutDir\v09_content_aware_final_table.tex")) {
        throw "Main paper table LaTeX was not created."
    }
    if (-not (Test-Path "$OutDir\v09_content_aware_overhead_table_paper.csv")) {
        throw "Overhead paper table CSV was not created."
    }
    if (-not (Test-Path "$OutDir\v09_content_aware_best_k_table.csv")) {
        throw "Best-k paper table CSV was not created."
    }

    Write-Host ""
    Write-Host "[2/3] Main paper table:"
    Import-Csv "$OutDir\v09_content_aware_final_table.csv" | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[2b/3] Overhead paper table:"
    Import-Csv "$OutDir\v09_content_aware_overhead_table_paper.csv" | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[2c/3] Best-k paper table:"
    Import-Csv "$OutDir\v09_content_aware_best_k_table.csv" | Format-Table -AutoSize

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-aware-paper-artifacts"

    Write-Host ""
    Write-Host "v0.9 paper artifacts completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-classifier-model
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentClassifierModel {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 content classifier model ==="

    $Config         = "configs/content_classifier_image_v09.json"
    $MetadataOracle = "results/routing_context/v09_content_metadata_oracle.csv"
    $Out            = "results/routing_context/v09_content_classifier_prediction.json"

    if (-not (Test-Path $MetadataOracle)) {
        Write-Host "Metadata/oracle CSV not found. Running metadata extraction..."
        Invoke-ScenarioV09ContentMetadata
    }
    if (-not (Test-Path $Config)) {
        throw "Classifier config not found: $Config"
    }

    $Image = "test_images/input.png"
    if (-not (Test-Path $Image)) {
        $Manifest = "results/routing_context/v09_image_manifest.csv"
        if (-not (Test-Path $Manifest)) {
            Invoke-ScenarioV09ImageManifest
        }
        $First = Import-Csv $Manifest | Select-Object -First 1
        $Image = $First.path
    }

    Write-Host ""
    Write-Host "[1/3] Running classifier prediction..."
    Write-Host "Image: $Image"
    & python -m src.router.adaptation.content_classifier_model `
        --config $Config --image $Image --out $Out
    if (-not (Test-Path $Out)) { throw "Classifier prediction report not created." }

    $Report = Get-Content $Out | ConvertFrom-Json
    if ($null -eq $Report.prediction) { throw "Expected classifier prediction." }

    Write-Host ""
    Write-Host "[2/3] Prediction:"
    Write-Host "Codec:" $Report.prediction.codec
    Write-Host "Config:" $Report.prediction.config
    Write-Host "Feature set:" $Report.feature_set
    Write-Host "k:" $Report.k

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-classifier-model"

    Write-Host ""
    Write-Host "v0.9 content classifier model completed successfully."
}


# ---------------------------------------------------------------------------
# Scenario: v09-content-classifier-router
# ---------------------------------------------------------------------------

function Invoke-ScenarioV09ContentClassifierRouter {
    Write-Host ""
    Write-Host "=== R-D-E Router v0.9 content classifier router validation ==="

    $Config = "configs/content_classifier_image_v09.json"
    $Image  = "test_images/input.png"
    $GlobalReport  = "results/routing_context/v09_router_content_classifier_apply_global.json"
    $TecnickReport = "results/routing_context/v09_router_content_classifier_apply_tecnick.json"

    if (-not (Test-Path $Config)) { throw "Missing classifier config: $Config" }
    if (-not (Test-Path $Image))  { throw "Missing test image: $Image" }

    Write-Host ""
    Write-Host "[1/3] Global pool: classifier prediction should fall back if not admissible..."
    & python -m src.router.rde_router `
        --config configs/router_image_v08.json `
        --content-classifier --content-classifier-mode apply `
        --content-classifier-config $Config `
        --content-classifier-image $Image `
        --out $GlobalReport

    $Global = Get-Content $GlobalReport | ConvertFrom-Json
    if ($Global.content_classifier.prediction.codec -ne "JPEG") {
        throw "Expected global classifier prediction codec JPEG."
    }
    if ($Global.content_classifier.prediction.config -ne "q=85") {
        throw "Expected global classifier prediction config q=85."
    }
    if ($Global.content_classifier.applied -ne $false) {
        throw "Expected global classifier applied=false."
    }
    if ($Global.content_classifier.warnings -notcontains "content_classifier_prediction_not_admissible_fallback_to_router") {
        throw "Expected fallback warning in global classifier report."
    }
    if ($Global.decision.selected.codec -ne "HEVC") {
        throw "Expected global selected codec HEVC."
    }
    if ($Global.decision.selected.config -ne "crf=15") {
        throw "Expected global selected config crf=15."
    }
    if ($Global.decision.decision_trace.selected_reason -ne "lowest_J_RDE_in_safe_pool") {
        throw "Expected global selected_reason lowest_J_RDE_in_safe_pool."
    }

    Write-Host ""
    Write-Host "Global fallback validated:"
    Write-Host "  prediction:" $Global.content_classifier.prediction.codec $Global.content_classifier.prediction.config
    Write-Host "  applied:" $Global.content_classifier.applied
    Write-Host "  selected:" $Global.decision.selected.codec $Global.decision.selected.config
    Write-Host "  reason:" $Global.decision.decision_trace.selected_reason

    Write-Host ""
    Write-Host "[2/3] Tecnick source-filtered pool: classifier prediction should apply..."
    & python -m src.router.rde_router `
        --config configs/router_image_v08.json `
        --content-classifier --content-classifier-mode apply `
        --content-classifier-config $Config `
        --content-classifier-image $Image `
        --content-source tecnick --content-source-filter `
        --content-filter-column dataset `
        --out $TecnickReport

    $Tecnick = Get-Content $TecnickReport | ConvertFrom-Json
    if ($Tecnick.content_filter.applied -ne $true) {
        throw "Expected Tecnick content filter applied=true."
    }
    if ($Tecnick.content_filter.value -ne "tecnick") {
        throw "Expected Tecnick content filter value=tecnick."
    }
    if ($Tecnick.content_classifier.prediction.codec -ne "JPEG") {
        throw "Expected Tecnick classifier prediction codec JPEG."
    }
    if ($Tecnick.content_classifier.prediction.config -ne "q=85") {
        throw "Expected Tecnick classifier prediction config q=85."
    }
    if ($Tecnick.content_classifier.applied -ne $true) {
        throw "Expected Tecnick classifier applied=true."
    }
    if ($Tecnick.decision.selected.codec -ne "JPEG") {
        throw "Expected Tecnick selected codec JPEG."
    }
    if ($Tecnick.decision.selected.config -ne "q=85") {
        throw "Expected Tecnick selected config q=85."
    }
    if ($Tecnick.decision.decision_trace.selected_reason -ne "content_classifier_preferred_candidate") {
        throw "Expected Tecnick selected_reason content_classifier_preferred_candidate."
    }

    Write-Host ""
    Write-Host "Tecnick apply validated:"
    Write-Host "  filter:" $Tecnick.content_filter.column "=" $Tecnick.content_filter.value "(" $Tecnick.content_filter.before_count "->" $Tecnick.content_filter.after_count ")"
    Write-Host "  prediction:" $Tecnick.content_classifier.prediction.codec $Tecnick.content_classifier.prediction.config
    Write-Host "  applied:" $Tecnick.content_classifier.applied
    Write-Host "  selected:" $Tecnick.decision.selected.codec $Tecnick.decision.selected.config
    Write-Host "  reason:" $Tecnick.decision.decision_trace.selected_reason

    Write-Host ""
    Write-Host "[3/3] Running pytest..."
    Invoke-LocalPytest -ScenarioName "v09-content-classifier-router"

    Write-Host ""
    Write-Host "v0.9 content classifier router validation completed successfully."
}


# ---------------------------------------------------------------------------
# Aggregate scenarios
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-scenario invocation
# ---------------------------------------------------------------------------

function Invoke-SingleScenario {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not $ScenarioMap.Contains($Name)) {
        throw "Internal: scenario '$Name' is not in the single-scenario map."
    }

    $handler = $ScenarioMap[$Name]
    $command = Get-Command -Name $handler -CommandType Function -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Internal: handler function '$handler' for scenario '$Name' is not defined."
    }

    Write-Host "[run_router] scenario : $Name"
    Write-Host "[run_router] handler  : $handler"

    & $handler
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

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
