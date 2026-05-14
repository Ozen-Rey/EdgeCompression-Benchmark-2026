$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 oracle classifier sweep ==="

$BenchmarkCsv = "results/images/image_4dataset_RDE_paper_ready.csv"
$MetadataOracleCsv = "results/routing_context/v09_content_metadata_oracle.csv"
$PixelFeaturesCsv = "results/routing_context/v09_image_pixel_features.csv"

$SummaryOut = "results/routing_context/v09_oracle_classifier_sweep_summary.csv"
$DecisionsOut = "results/routing_context/v09_oracle_classifier_sweep_decisions.csv"

if (!(Test-Path $MetadataOracleCsv)) {
    Write-Host "Metadata/oracle CSV not found. Running metadata pipeline..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_metadata.ps1
}

if (!(Test-Path $PixelFeaturesCsv)) {
    Write-Host "Pixel features CSV not found. Running pixel feature extraction..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_image_features.ps1
}

Write-Host ""
Write-Host "[1/3] Running classifier sweep..."

python -m src.router.analysis.content_oracle_classifier_sweep `
  --benchmark-csv $BenchmarkCsv `
  --metadata-oracle-csv $MetadataOracleCsv `
  --pixel-features-csv $PixelFeaturesCsv `
  --feature-sets "metadata_no_source,pixel_no_source,all_no_source,all_with_source" `
  --k-values "1,3,5,7,9,11" `
  --evaluation-modes "leave_one_image_out,leave_one_dataset_out" `
  --quality-floor 80 `
  --available-codecs "JPEG,JXL,HEVC" `
  --wR 0.2 `
  --wE 0.2 `
  --wD 0.6 `
  --summary-out $SummaryOut `
  --decisions-out $DecisionsOut

if (!(Test-Path $SummaryOut)) {
    throw "Sweep summary not created."
}

if (!(Test-Path $DecisionsOut)) {
    throw "Sweep decisions not created."
}

Write-Host ""
Write-Host "[2/3] Top sweep rows by mean regret..."
Import-Csv $SummaryOut |
  Sort-Object {[double]$_.mean_regret} |
  Select-Object evaluation_mode,feature_set,k,accuracy,mean_regret,relative_regret_reduction,fallback_rate |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[2b/3] Best row per evaluation mode..."
Import-Csv $SummaryOut |
  Group-Object evaluation_mode |
  ForEach-Object {
    $_.Group |
      Sort-Object {[double]$_.mean_regret} |
      Select-Object -First 1 evaluation_mode,feature_set,k,accuracy,mean_regret,relative_regret_reduction,fallback_rate
  } |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[3/3] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 oracle classifier sweep completed successfully."
