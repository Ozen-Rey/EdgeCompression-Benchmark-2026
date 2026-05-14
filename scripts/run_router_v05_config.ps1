$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.5 config test ==="

$PureReport = "results/routing_context/v05_router_config_report.json"
$OverrideReport = "results/routing_context/v05_router_config_override_jxl.json"

Write-Host ""
Write-Host "[1/3] Running pure config..."
python -m src.router.rde_router `
  --config configs/router_image_v05.json `
  --out $PureReport

if ($LASTEXITCODE -ne 0) {
  throw "Pure config router run failed."
}

if (!(Test-Path $PureReport)) {
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
python -m src.router.rde_router `
  --config configs/router_image_v05.json `
  --available-codecs "JXL" `
  --out $OverrideReport

if ($LASTEXITCODE -ne 0) {
  throw "Override router run failed."
}

if (!(Test-Path $OverrideReport)) {
  throw "Override report not created: $OverrideReport"
}

$override = Get-Content $OverrideReport | ConvertFrom-Json

if ($override.decision.selected.codec -ne "JXL") {
  throw "Expected selected codec JXL, got: $($override.decision.selected.codec)"
}

Write-Host "Override selected:" $override.decision.selected.codec $override.decision.selected.config

Write-Host ""
Write-Host "[3/3] Running pytest..."
$PytestBasetemp = ".pytest_tmp_" + [IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $PytestBasetemp
python -m pytest tests -q --basetemp $PytestBasetemp
$PytestExit = $LASTEXITCODE
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $PytestBasetemp
if ($PytestExit -ne 0) {
    throw "pytest failed with exit code $PytestExit"
}

Write-Host ""
Write-Host "v0.5 config test completed successfully."
