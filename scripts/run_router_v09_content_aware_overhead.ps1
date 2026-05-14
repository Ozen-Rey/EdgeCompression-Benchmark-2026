$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 content-aware overhead and sensitivity ==="

$PixelFeatures = "results/routing_context/v09_image_pixel_features.csv"
$BenchmarkCsv = "results/images/image_4dataset_RDE_paper_ready.csv"
$SweepSummary = "results/routing_context/v09_oracle_classifier_sweep_summary.csv"

$OverheadOut = "results/routing_context/v09_content_aware_overhead_table.csv"
$SensitivityOut = "results/routing_context/v09_knn_sensitivity_table.csv"

if (!(Test-Path $PixelFeatures)) {
    Write-Host "Pixel features not found. Running pixel feature extraction..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_image_features.ps1
}

if (!(Test-Path $SweepSummary)) {
    Write-Host "Classifier sweep not found. Running classifier sweep..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_oracle_classifier_sweep.ps1
}

Write-Host ""
Write-Host "[1/3] Building overhead and sensitivity tables..."

python -m src.router.analysis.content_aware_overhead_analysis `
  --pixel-features-csv $PixelFeatures `
  --benchmark-csv $BenchmarkCsv `
  --classifier-sweep-summary $SweepSummary `
  --dataset-col dataset `
  --codec-col codec `
  --config-col param `
  --time-col time_ms `
  --overhead-out $OverheadOut `
  --sensitivity-out $SensitivityOut

if (!(Test-Path $OverheadOut)) {
    throw "Overhead table not created."
}

if (!(Test-Path $SensitivityOut)) {
    throw "Sensitivity table not created."
}

Write-Host ""
Write-Host "[2/3] Overhead table:"
Import-Csv $OverheadOut |
  Select-Object component,case_id,scope,dataset,codec,config,num_samples,mean_ms,median_ms,p90_ms,pixel_feature_mean_over_this_mean |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[2b/3] k-sensitivity for metadata_no_source and pixel_no_source:"
Import-Csv $SensitivityOut |
  Where-Object { $_.feature_set -in @("metadata_no_source", "pixel_no_source") } |
  Select-Object evaluation_mode,feature_set,k,accuracy,mean_regret,relative_regret_reduction,fallback_rate,best_for_evaluation_mode,best_for_feature_set |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[3/3] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 content-aware overhead and sensitivity completed successfully."