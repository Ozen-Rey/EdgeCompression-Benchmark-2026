$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 content-aware benchmark table ==="

$OracleSummary = "results/routing_context/v09_content_oracle_summary.csv"
$DatasetPolicySummary = "results/routing_context/v09_metadata_policy_dataset_summary.csv"
$ClassifierSweepSummary = "results/routing_context/v09_oracle_classifier_sweep_summary.csv"

$PaperTable = "results/routing_context/v09_content_aware_benchmark_table.csv"
$AllMethods = "results/routing_context/v09_content_aware_benchmark_all_methods.csv"

if (!(Test-Path $OracleSummary)) {
    Write-Host "Oracle summary not found. Running content-aware validation..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_aware.ps1
}

if (!(Test-Path $DatasetPolicySummary)) {
    Write-Host "Dataset policy summary not found. Running metadata policy..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_metadata_policy.ps1
}

if (!(Test-Path $ClassifierSweepSummary)) {
    Write-Host "Classifier sweep summary not found. Running classifier sweep..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_oracle_classifier_sweep.ps1
}

Write-Host ""
Write-Host "[1/3] Building benchmark tables..."

python -m src.router.content_aware_benchmark_table `
  --oracle-summary $OracleSummary `
  --dataset-policy-summary $DatasetPolicySummary `
  --classifier-sweep-summary $ClassifierSweepSummary `
  --paper-table-out $PaperTable `
  --all-methods-out $AllMethods

if (!(Test-Path $PaperTable)) {
    throw "Paper benchmark table not created."
}

if (!(Test-Path $AllMethods)) {
    throw "All-methods benchmark table not created."
}

Write-Host ""
Write-Host "[2/3] Paper table:"
Import-Csv $PaperTable |
  Select-Object method_id,evaluation_protocol,deployment_setting,feature_set,k,accuracy,mean_regret,relative_regret_reduction,fallback_rate |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[2b/3] Best deployable rows by mean regret:"
Import-Csv $AllMethods |
  Where-Object { $_.uses_oracle -eq "False" } |
  Sort-Object {[double]$_.mean_regret} |
  Select-Object -First 12 method_id,evaluation_protocol,feature_set,k,accuracy,mean_regret,relative_regret_reduction,fallback_rate |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[3/3] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 content-aware benchmark table completed successfully."
