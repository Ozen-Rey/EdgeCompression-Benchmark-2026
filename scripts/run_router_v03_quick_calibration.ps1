Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CalibrationInputDir = "calibration_images"
$CalibrationOutDir = "results\routing_calibration"
$RoutingOutDir = "results\routing_context"
$Csv = "results\images\image_4dataset_RDE_paper_ready.csv"

New-Item -ItemType Directory -Force $CalibrationInputDir | Out-Null
New-Item -ItemType Directory -Force $CalibrationOutDir | Out-Null
New-Item -ItemType Directory -Force $RoutingOutDir | Out-Null

if (!(Test-Path "test_images\input.png")) {
    throw "Missing source image: test_images\input.png"
}

Copy-Item "test_images\input.png" "$CalibrationInputDir\input.png" -Force

Write-Host ""
Write-Host "============================================================"
Write-Host "Running v0.3 quick local calibration"
Write-Host "============================================================"

python -m src.router.calibration `
  --level quick `
  --input-dir $CalibrationInputDir `
  --codecs JPEG,JXL,HEVC `
  --out "$CalibrationOutDir\local_quick.json" `
  --summary-csv "$CalibrationOutDir\local_quick_summary.csv"

if ($LASTEXITCODE -ne 0) {
    throw "Calibration failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Running calibrated router decision + execution"
Write-Host "============================================================"

python -m src.router.rde_router `
  --csv $Csv `
  --domain image `
  --auto-weights `
  --system-aware `
  --capability-aware `
  --strict-executables `
  --available-codecs "JXL" `
  --calibration-file "$CalibrationOutDir\local_quick.json" `
  --power-mode ac `
  --thermal-state nominal `
  --network-profile normal `
  --quality-target high `
  --safe-mode `
  --quality-constraint-stat min `
  --quality-floor 70 `
  --allow-degraded-fallback `
  --near-quality-floor 60 `
  --codec-col codec `
  --config-col param `
  --rate-col bpp `
  --quality-col ssimulacra2 `
  --energy-col energy_per_image_j `
  --time-col time_ms `
  --aggregate-by-config `
  --generate-command `
  --execute `
  --input "$CalibrationInputDir\input.png" `
  --output "$CalibrationInputDir\v03_calibrated_jxl.jxl" `
  --out "$RoutingOutDir\v03_jxl_with_calibration_execute.json"

if ($LASTEXITCODE -ne 0) {
    throw "Calibrated router execution failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "v0.3 quick calibration summary"
Write-Host "============================================================"

$r = Get-Content "$RoutingOutDir\v03_jxl_with_calibration_execute.json" -Raw | ConvertFrom-Json
$s = $r.decision.selected
$c = $r.calibration
$e = $r.execution_result

[pscustomobject]@{
    calibration_enabled = $c.enabled
    calibration_level = $c.level
    calibration_applied_points = $c.num_applied
    selected_codec = $s.codec
    selected_config = $s.config
    calibrated_rate = $s.rate
    calibrated_energy = $s.energy
    calibrated_time_ms = $s.time_ms
    execution_success = $e.success
    output = $e.output
} | Format-List

Write-Host ""
Write-Host "Generated calibration file:"
Write-Host "  $CalibrationOutDir\local_quick.json"

Write-Host ""
Write-Host "Generated calibration summary:"
Write-Host "  $CalibrationOutDir\local_quick_summary.csv"

Write-Host ""
Write-Host "Generated router report:"
Write-Host "  $RoutingOutDir\v03_jxl_with_calibration_execute.json"

Write-Host ""
Write-Host "Done."
