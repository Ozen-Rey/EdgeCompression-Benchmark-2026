$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 oracle classifier baseline ==="

$BenchmarkCsv = "results/images/image_4dataset_RDE_paper_ready.csv"
$MetadataOracleCsv = "results/routing_context/v09_content_metadata_oracle.csv"
$PixelFeaturesCsv = "results/routing_context/v09_image_pixel_features.csv"

if (!(Test-Path $MetadataOracleCsv)) {
    Write-Host "Metadata/oracle CSV not found. Running metadata pipeline..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_metadata.ps1
}

if (!(Test-Path $PixelFeaturesCsv)) {
    Write-Host "Pixel features CSV not found. Running pixel feature extraction..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_image_features.ps1
}

$FeatureSets = @(
    "metadata_no_source",
    "pixel_no_source",
    "all_no_source",
    "all_with_source"
)

foreach ($FeatureSet in $FeatureSets) {
    Write-Host ""
    Write-Host "[classifier=$FeatureSet] Evaluating leave-one-out kNN oracle classifier..."

    $DecisionsOut = "results/routing_context/v09_oracle_classifier_${FeatureSet}_decisions.csv"
    $SummaryOut = "results/routing_context/v09_oracle_classifier_${FeatureSet}_summary.csv"

    python -m src.router.analysis.content_oracle_classifier `
      --benchmark-csv $BenchmarkCsv `
      --metadata-oracle-csv $MetadataOracleCsv `
      --pixel-features-csv $PixelFeaturesCsv `
      --feature-set $FeatureSet `
      --k 3 `
      --quality-floor 80 `
      --available-codecs "JPEG,JXL,HEVC" `
      --wR 0.2 `
      --wE 0.2 `
      --wD 0.6 `
      --decisions-out $DecisionsOut `
      --summary-out $SummaryOut

    if (!(Test-Path $SummaryOut)) {
        throw "Classifier summary not created for feature_set=$FeatureSet"
    }

    Write-Host ""
    Write-Host "Summary for feature_set=$FeatureSet"
    Import-Csv $SummaryOut | Format-Table -AutoSize
}

Write-Host ""
Write-Host "[final] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 oracle classifier baseline completed successfully."
