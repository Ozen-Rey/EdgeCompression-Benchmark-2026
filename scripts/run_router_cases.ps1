Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Csv = "results\images\image_4dataset_RDE_paper_ready.csv"
$OutDir = "results\routing_context"

New-Item -ItemType Directory -Force $OutDir | Out-Null

$CommonArgs = @(
    "--csv", $Csv,
    "--domain", "image",
    "--auto-weights",
    "--system-aware",
    "--power-mode", "ac",
    "--thermal-state", "nominal",
    "--network-profile", "very-limited",
    "--quality-target", "preview",
    "--codec-col", "codec",
    "--config-col", "param",
    "--rate-col", "bpp",
    "--quality-col", "ssimulacra2",
    "--energy-col", "energy_per_image_j",
    "--time-col", "time_ms",
    "--aggregate-by-config",
    "--export-topk",
    "--top-k", "10"
)

function Invoke-RouterCase {
    param(
        [string]$Name,
        [string[]]$ExtraArgs,
        [bool]$ExpectFailure = $false
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running case: $Name"
    Write-Host "============================================================"

    & python -m src.router.rde_router @CommonArgs @ExtraArgs
    $exitCode = $LASTEXITCODE

    if ($ExpectFailure) {
        if ($exitCode -eq 0) {
            throw "Case '$Name' was expected to fail, but it succeeded."
        }
        Write-Host "Expected infeasible case confirmed: $Name"
    }
    else {
        if ($exitCode -ne 0) {
            throw "Case '$Name' failed with exit code $exitCode."
        }
    }
}

Invoke-RouterCase `
    -Name "safe_cuda" `
    -ExtraArgs @(
        "--safe-mode",
        "--quality-constraint-stat", "min",
        "--quality-floor", "70",
        "--allow-degraded-fallback",
        "--near-quality-floor", "55",
        "--out", "$OutDir\system_aware_safe.json"
    )

Invoke-RouterCase `
    -Name "safe_no_cuda" `
    -ExtraArgs @(
        "--simulate-no-cuda",
        "--safe-mode",
        "--quality-constraint-stat", "min",
        "--quality-floor", "70",
        "--allow-degraded-fallback",
        "--near-quality-floor", "55",
        "--out", "$OutDir\system_aware_safe_no_cuda.json"
    )

Invoke-RouterCase `
    -Name "low_latency_maxtime100" `
    -ExtraArgs @(
        "--safe-mode",
        "--quality-constraint-stat", "min",
        "--quality-floor", "70",
        "--allow-degraded-fallback",
        "--near-quality-floor", "55",
        "--max-time-ms", "100",
        "--out", "$OutDir\system_aware_safe_maxtime100.json"
    )

Invoke-RouterCase `
    -Name "ultra_low_bitrate_cuda" `
    -ExtraArgs @(
        "--quality-constraint-stat", "mean",
        "--quality-floor", "60",
        "--allow-degraded-fallback",
        "--near-quality-floor", "30",
        "--max-rate", "0.16",
        "--out", "$OutDir\force_low_bitrate_dcae.json"
    )

Invoke-RouterCase `
    -Name "ultra_low_bitrate_no_cuda" `
    -ExpectFailure $true `
    -ExtraArgs @(
        "--simulate-no-cuda",
        "--quality-constraint-stat", "mean",
        "--quality-floor", "60",
        "--allow-degraded-fallback",
        "--near-quality-floor", "30",
        "--max-rate", "0.16",
        "--out", "$OutDir\force_low_bitrate_no_cuda.json"
    )

Write-Host ""
Write-Host "============================================================"
Write-Host "Building summary CSV"
Write-Host "============================================================"

$Rows = @()

function Add-ReportRow {
    param(
        [string]$CaseName,
        [string]$Path
    )

    $r = Get-Content $Path -Raw | ConvertFrom-Json
    $s = $r.decision.selected
    $f = $r.codec_filtering.system_aware
    $c = $r.constraints

    $script:Rows += [pscustomobject]@{
        case_name = $CaseName
        status = "selected"
        cuda_available = $f.cuda_available
        effective_exclude_neural = $f.effective_exclude_neural
        decision_mode = $r.decision.decision_mode
        selected_codec = $s.codec
        selected_config = $s.config
        rate = $s.rate
        quality_mean = $s.quality
        quality_guard_stat = $s.quality_constraint_stat
        quality_guard_value = $s.quality_constraint_value
        quality_min = $s.quality_stats.min
        quality_p10 = $s.quality_stats.p10
        energy = $s.energy
        time_ms = $s.time_ms
        J_RDE = $s.cost
        max_rate = $c.max_rate
        max_time_ms = $c.max_time_ms
        quality_floor = $c.quality_floor
        near_quality_floor = $c.near_quality_floor
    }
}

Add-ReportRow "safe_cuda" "$OutDir\system_aware_safe.json"
Add-ReportRow "safe_no_cuda" "$OutDir\system_aware_safe_no_cuda.json"
Add-ReportRow "low_latency_maxtime100" "$OutDir\system_aware_safe_maxtime100.json"
Add-ReportRow "ultra_low_bitrate_cuda" "$OutDir\force_low_bitrate_dcae.json"

$Rows += [pscustomobject]@{
    case_name = "ultra_low_bitrate_no_cuda"
    status = "infeasible"
    cuda_available = $false
    effective_exclude_neural = $true
    decision_mode = "infeasible"
    selected_codec = ""
    selected_config = ""
    rate = ""
    quality_mean = ""
    quality_guard_stat = "mean"
    quality_guard_value = ""
    quality_min = ""
    quality_p10 = ""
    energy = ""
    time_ms = ""
    J_RDE = ""
    max_rate = "0.16"
    max_time_ms = ""
    quality_floor = "60"
    near_quality_floor = "30"
}

$SummaryPath = "$OutDir\router_case_summary.csv"
$Rows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryPath

Write-Host "Summary written to: $SummaryPath"
Write-Host ""
Write-Host "Done."