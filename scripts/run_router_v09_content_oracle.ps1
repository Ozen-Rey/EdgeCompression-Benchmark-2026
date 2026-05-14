$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 content oracle analysis ==="

$Csv = "results/images/image_4dataset_RDE_paper_ready.csv"
$ByImageOut = "results/routing_context/v09_content_oracle_by_image.csv"
$SummaryOut = "results/routing_context/v09_content_oracle_summary.csv"

if (!(Test-Path $Csv)) {
    throw "Benchmark CSV not found: $Csv"
}

Write-Host ""
Write-Host "[1/3] Running content oracle analysis..."

python -m src.router.analysis.content_oracle_analysis `
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
  --by-image-out $ByImageOut `
  --summary-out $SummaryOut

if (!(Test-Path $ByImageOut)) {
    throw "By-image oracle CSV not created."
}

if (!(Test-Path $SummaryOut)) {
    throw "Summary oracle CSV not created."
}

Write-Host ""
Write-Host "[2/3] Preview summary..."
Import-Csv $SummaryOut | Format-Table -AutoSize

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
Write-Host "v0.9 content oracle analysis completed successfully."
