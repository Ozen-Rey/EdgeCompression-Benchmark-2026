Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Csv = "results\images\image_4dataset_RDE_paper_ready.csv"
$OutDir = "results\routing_context"
$Input = "test_images\input.png"

New-Item -ItemType Directory -Force $OutDir | Out-Null

if (!(Test-Path $Input)) {
    throw "Input image not found: $Input"
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
    "--input", $Input
)

function Invoke-RouterBackendCase {
    param(
        [string]$Name,
        [string[]]$ExtraArgs
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running v0.2 backend case: $Name"
    Write-Host "============================================================"

    & python -m src.router.rde_router @CommonArgs @ExtraArgs
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "Backend case '$Name' failed with exit code $exitCode."
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

$Rows = @()

function Add-BackendReportRow {
    param(
        [string]$CaseName,
        [string]$Path
    )

    $r = Get-Content $Path -Raw | ConvertFrom-Json
    $s = $r.decision.selected
    $p = $r.execution_plan
    $e = $r.execution_result

    $script:Rows += [pscustomobject]@{
        case_name = $CaseName
        selected_codec = $s.codec
        selected_config = $s.config
        decision_mode = $r.decision.decision_mode
        backend = $p.execution_backend
        can_execute = $p.can_execute
        executed = $e.executed
        success = $e.success
        output = $e.output
        rate = $s.rate
        quality_mean = $s.quality
        quality_guard_value = $s.quality_constraint_value
        energy = $s.energy
        time_ms = $s.time_ms
        J_RDE = $s.cost
    }
}

Add-BackendReportRow "jpeg_execute" "$OutDir\v02_backend_jpeg.json"
Add-BackendReportRow "jxl_execute" "$OutDir\v02_backend_jxl.json"
Add-BackendReportRow "hevc_execute" "$OutDir\v02_backend_hevc.json"

$SummaryPath = "$OutDir\v02_backend_summary.csv"
$Rows | Export-Csv -NoTypeInformation -Encoding UTF8 $SummaryPath

Write-Host "Summary written to: $SummaryPath"
Write-Host ""
Write-Host "Generated files:"
Get-ChildItem test_images\v02_backend_* | Select-Object Name, Length, LastWriteTime

Write-Host ""
Write-Host "Done."
