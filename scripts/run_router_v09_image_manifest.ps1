$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 image manifest ==="

$Csv = "results/images/image_4dataset_RDE_paper_ready.csv"
$Out = "results/routing_context/v09_image_manifest.csv"

$Roots = "kodak=datasets/images/kodak;div2k_valid=datasets/images/div2k_valid;clic2020=datasets/images/clic2020;tecnick=datasets/images/tecnick"

python -m src.router.content_image_manifest `
  --csv $Csv `
  --roots $Roots `
  --dataset-col dataset `
  --image-col image `
  --out $Out

if (!(Test-Path $Out)) {
    throw "Manifest not created."
}

$Rows = Import-Csv $Out

if ($Rows.Count -ne 96) {
    throw "Expected 96 manifest rows, got $($Rows.Count)."
}

Write-Host ""
Write-Host "Manifest counts:"
$Rows | Group-Object dataset | Select-Object Count,Name | Format-Table -AutoSize

Write-Host ""
Write-Host "Mode/dimension checks:"
$Rows | Group-Object dataset,mode | Select-Object Count,Name | Format-Table -AutoSize

$TecnickBad = $Rows | Where-Object {
    $_.dataset -eq "tecnick" -and (
        $_.width -ne "1200" -or
        $_.height -ne "1200" -or
        $_.mode -ne "RGB"
    )
}

if ($TecnickBad.Count -gt 0) {
    throw "Tecnick validation failed: expected all images 1200x1200 RGB."
}

Write-Host ""
Write-Host "Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 image manifest completed successfully."
