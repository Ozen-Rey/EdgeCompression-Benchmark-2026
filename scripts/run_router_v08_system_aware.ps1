$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.8 system-aware validation ==="

$TmpDir = ".tmp_manual_probe"
$SyntheticCsv = "$TmpDir/v08_policy_change.csv"

$BasicFeatures = "results/routing_context/v08_system_features_basic.json"
$GpuFeatures = "results/routing_context/v08_system_features_gpu.json"

$ReportOnly = "results/routing_context/v08_policy_change_report_only.json"
$Apply = "results/routing_context/v08_policy_change_apply.json"

New-Item -ItemType Directory -Force $TmpDir | Out-Null

@"
codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms
QualityHeavy,mode=quality,1.0,95.0,100.0,200.0
EnergyLight,mode=energy,1.2,85.0,1.0,20.0
"@ | Set-Content -Encoding UTF8 $SyntheticCsv

Write-Host ""
Write-Host "[1/5] Extracting basic system features..."
python -m src.router.adaptation.system_features `
  --probe-level basic `
  --out $BasicFeatures

if (!(Test-Path $BasicFeatures)) {
  throw "Basic system feature report not created."
}

Write-Host ""
Write-Host "[2/5] Extracting GPU system features..."
python -m src.router.adaptation.system_features `
  --probe-level gpu `
  --out $GpuFeatures

if (!(Test-Path $GpuFeatures)) {
  throw "GPU system feature report not created."
}

Write-Host ""
Write-Host "[3/5] Running synthetic policy case in report-only mode..."
python -m src.router.rde_router `
  --csv $SyntheticCsv `
  --codec-col codec `
  --config-col param `
  --rate-col bpp `
  --quality-col ssimulacra2 `
  --energy-col energy_per_image_j `
  --time-col time_ms `
  --domain image `
  --auto-weights `
  --quality-target high `
  --quality-floor 80 `
  --system-policy `
  --system-policy-mode report-only `
  --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
  --out $ReportOnly

Write-Host ""
Write-Host "[4/5] Running synthetic policy case in apply mode..."
python -m src.router.rde_router `
  --csv $SyntheticCsv `
  --codec-col codec `
  --config-col param `
  --rate-col bpp `
  --quality-col ssimulacra2 `
  --energy-col energy_per_image_j `
  --time-col time_ms `
  --domain image `
  --auto-weights `
  --quality-target high `
  --quality-floor 80 `
  --system-policy `
  --system-policy-mode apply `
  --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
  --out $Apply

$reportOnlyJson = Get-Content $ReportOnly | ConvertFrom-Json
$applyJson = Get-Content $Apply | ConvertFrom-Json

$reportOnlySelected = $reportOnlyJson.decision.selected.codec
$applySelected = $applyJson.decision.selected.codec

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

python -m src.router.rde_router `
  --csv $SyntheticCsv `
  --codec-col codec `
  --config-col param `
  --rate-col bpp `
  --quality-col ssimulacra2 `
  --energy-col energy_per_image_j `
  --time-col time_ms `
  --domain image `
  --auto-weights `
  --quality-target high `
  --quality-floor 80 `
  --system-policy `
  --system-policy-mode apply `
  --system-policy-simulate "battery=critical,cpu=busy,memory=constrained" `
  --system-penalty `
  --system-penalty-mode apply `
  --system-penalty-lambda 1.0 `
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
$PytestBasetemp = ".pytest_tmp_" + [IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $PytestBasetemp
python -m pytest tests -q --basetemp $PytestBasetemp
$PytestExit = $LASTEXITCODE
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $PytestBasetemp
if ($PytestExit -ne 0) {
    throw "pytest failed with exit code $PytestExit"
}

Write-Host ""
Write-Host "v0.8 system-aware validation completed successfully."
