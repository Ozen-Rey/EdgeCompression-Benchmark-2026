$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 content classifier model ==="

$Config = "configs/content_classifier_image_v09.json"
$MetadataOracle = "results/routing_context/v09_content_metadata_oracle.csv"
$Out = "results/routing_context/v09_content_classifier_prediction.json"

if (!(Test-Path $MetadataOracle)) {
    Write-Host "Metadata/oracle CSV not found. Running metadata extraction..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_metadata.ps1
}

if (!(Test-Path $Config)) {
    throw "Classifier config not found: $Config"
}

$Image = "test_images/input.png"

if (!(Test-Path $Image)) {
    $Manifest = "results/routing_context/v09_image_manifest.csv"

    if (!(Test-Path $Manifest)) {
        powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_image_manifest.ps1
    }

    $First = Import-Csv $Manifest | Select-Object -First 1
    $Image = $First.path
}

Write-Host ""
Write-Host "[1/3] Running classifier prediction..."
Write-Host "Image: $Image"

python -m src.router.adaptation.content_classifier_model `
  --config $Config `
  --image $Image `
  --out $Out

if (!(Test-Path $Out)) {
    throw "Classifier prediction report not created."
}

$Report = Get-Content $Out | ConvertFrom-Json

if ($null -eq $Report.prediction) {
    throw "Expected classifier prediction."
}

Write-Host ""
Write-Host "[2/3] Prediction:"
Write-Host "Codec:" $Report.prediction.codec
Write-Host "Config:" $Report.prediction.config
Write-Host "Feature set:" $Report.feature_set
Write-Host "k:" $Report.k

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
Write-Host "v0.9 content classifier model completed successfully."
