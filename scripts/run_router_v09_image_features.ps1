$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 image pixel features ==="

$Manifest = "results/routing_context/v09_image_manifest.csv"
$FeaturesOut = "results/routing_context/v09_image_pixel_features.csv"
$SummaryOut = "results/routing_context/v09_image_pixel_features_summary.csv"

if (!(Test-Path $Manifest)) {
    Write-Host "Image manifest not found. Running manifest builder first..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_image_manifest.ps1
}

Write-Host ""
Write-Host "[1/3] Extracting pixel-level image features..."

python -m src.router.adaptation.content_image_features `
  --manifest $Manifest `
  --resize-long-side 256 `
  --features-out $FeaturesOut `
  --summary-out $SummaryOut

if (!(Test-Path $FeaturesOut)) {
    throw "Pixel features CSV not created."
}

if (!(Test-Path $SummaryOut)) {
    throw "Pixel feature summary CSV not created."
}

$Rows = Import-Csv $FeaturesOut

if ($Rows.Count -ne 96) {
    throw "Expected 96 feature rows, got $($Rows.Count)."
}

Write-Host ""
Write-Host "[2/3] Feature summary..."
Import-Csv $SummaryOut | Format-Table -AutoSize

Write-Host ""
Write-Host "Dataset counts:"
$Rows | Group-Object dataset | Select-Object Count,Name | Format-Table -AutoSize

Write-Host ""
Write-Host "Texture classes:"
$Rows | Group-Object texture_class | Select-Object Count,Name | Format-Table -AutoSize

Write-Host ""
Write-Host "Edge classes:"
$Rows | Group-Object edge_class | Select-Object Count,Name | Format-Table -AutoSize

Write-Host ""
Write-Host "[3/3] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 image pixel features completed successfully."
