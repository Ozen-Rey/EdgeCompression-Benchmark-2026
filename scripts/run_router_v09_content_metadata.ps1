$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 metadata content features ==="

$Csv = "results/images/image_4dataset_RDE_paper_ready.csv"
$OracleByImage = "results/routing_context/v09_content_oracle_by_image.csv"

$FeaturesOut = "results/routing_context/v09_content_metadata_features.csv"
$JoinedOut = "results/routing_context/v09_content_metadata_oracle.csv"
$SummaryOut = "results/routing_context/v09_content_metadata_summary.csv"

if (!(Test-Path $Csv)) {
    throw "Benchmark CSV not found: $Csv"
}

if (!(Test-Path $OracleByImage)) {
    Write-Host "Oracle by-image CSV not found. Running oracle analysis first..."

    python -m src.router.content_oracle_analysis `
      --csv $Csv `
      --dataset-col dataset `
      --image-col image `
      --codec-col codec `
      --config-col param `
      --rate-col bpp `
      --quality-col ssimulacra2 `
      --energy-col energy_per_image_j `
      --time-col time_ms `
      --available-codecs "JPEG,JXL,HEVC" `
      --quality-floor 80 `
      --global-coverage-floor 1.0 `
      --wR 0.2 `
      --wE 0.2 `
      --wD 0.6 `
      --by-image-out $OracleByImage `
      --summary-out "results/routing_context/v09_content_oracle_summary.csv"
}

Write-Host ""
Write-Host "[1/3] Extracting metadata features and joining oracle labels..."

python -m src.router.content_metadata_features `
  --csv $Csv `
  --oracle-by-image $OracleByImage `
  --dataset-col dataset `
  --image-col image `
  --width-col width `
  --height-col height `
  --pixels-col pixels `
  --features-out $FeaturesOut `
  --joined-out $JoinedOut `
  --summary-out $SummaryOut

if (!(Test-Path $FeaturesOut)) {
    throw "Features CSV not created."
}

if (!(Test-Path $JoinedOut)) {
    throw "Joined metadata/oracle CSV not created."
}

if (!(Test-Path $SummaryOut)) {
    throw "Summary CSV not created."
}

Write-Host ""
Write-Host "[2/3] Preview metadata summary..."
Import-Csv $SummaryOut | Format-Table -AutoSize

Write-Host ""
Write-Host "[3/3] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 metadata content features completed successfully."
