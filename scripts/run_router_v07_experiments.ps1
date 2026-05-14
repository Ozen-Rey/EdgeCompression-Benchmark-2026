$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.7 experiment validation ==="

$Suite = "configs/experiments_image_v07.json"
$Summary = "results/routing_context/v07_experiment_summary.csv"
$ReportDir = "results/routing_context/v07_experiments"

Write-Host ""
Write-Host "[1/4] Dry-run experiment suite..."
python -m src.router.experiment_manager `
  --suite $Suite `
  --dry-run

Write-Host ""
Write-Host "[2/4] Running experiment suite..."
python -m src.router.experiment_manager `
  --suite $Suite

if (!(Test-Path $Summary)) {
  throw "Summary CSV not created: $Summary"
}

if (!(Test-Path $ReportDir)) {
  throw "Report directory not created: $ReportDir"
}

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

if ($null -eq $jxlExec) {
  throw "Missing jxl_execute_validation row."
}

if ($jxlExec.execution_success -ne "True") {
  throw "jxl_execute_validation did not execute successfully."
}

if ($jxlExec.output_exists -ne "True") {
  throw "jxl_execute_validation output does not exist."
}

if ($jxlExec.output_nonempty -ne "True") {
  throw "jxl_execute_validation output is empty."
}

if ($jxlExec.extension_valid -ne "True") {
  throw "jxl_execute_validation output extension is invalid."
}

Write-Host "Reports:" $reports.Count
Write-Host "Summary rows:" $summaryRows.Count
Write-Host "JXL execute output size:" $jxlExec.output_size_bytes

Write-Host ""
Write-Host "[3/4] Running pytest..."
$PytestBasetemp = ".pytest_tmp_" + [IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $PytestBasetemp
python -m pytest tests -q --basetemp $PytestBasetemp
$PytestExit = $LASTEXITCODE
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $PytestBasetemp
if ($PytestExit -ne 0) {
    throw "pytest failed with exit code $PytestExit"
}

Write-Host ""
Write-Host "[4/4] Validation completed."
Write-Host "v0.7 experiment validation completed successfully."