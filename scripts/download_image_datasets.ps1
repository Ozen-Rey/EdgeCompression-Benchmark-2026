param(
  [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [switch]$SetupOnly,
  [switch]$KodakOnly,
  [switch]$Div2kOnly,
  [switch]$KeepDownloads
)

$ErrorActionPreference = "Stop"

if ($KodakOnly -and $Div2kOnly) {
  throw "Use only one subset switch: -KodakOnly or -Div2kOnly."
}

$ImagesRoot = Join-Path $Root "datasets/images"
$DownloadsRoot = Join-Path $Root "datasets/downloads"
$KodakDir = Join-Path $ImagesRoot "kodak"
$Div2kDir = Join-Path $ImagesRoot "div2k_valid"
$ClicDir = Join-Path $ImagesRoot "clic2020"
$TecnickDir = Join-Path $ImagesRoot "tecnick"

$Div2kZip = Join-Path $DownloadsRoot "DIV2K_valid_HR.zip"
$Div2kExtracted = Join-Path $DownloadsRoot "DIV2K_valid_HR_extracted"
$Div2kSourceDir = Join-Path $Div2kExtracted "DIV2K_valid_HR"

function New-DatasetDirectory {
  param([string]$Path)

  if (!(Test-Path $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Assert-ImageCount {
  param(
    [string]$Path,
    [string]$Name,
    [int]$Expected
  )

  $count = 0
  if (Test-Path $Path) {
    $count = (Get-ChildItem -Path $Path -Filter "*.png" -File | Measure-Object).Count
  }

  Write-Host ("{0}: {1}/{2} PNG" -f $Name, $count, $Expected)

  if ($count -ne $Expected) {
    throw ("Expected {0} PNG files in {1}, found {2}." -f $Expected, $Path, $count)
  }
}

function Invoke-DownloadFile {
  param(
    [string]$Uri,
    [string]$OutFile
  )

  if (Test-Path $OutFile) {
    Write-Host "Already present: $OutFile"
    return
  }

  Write-Host "Downloading: $Uri"
  Invoke-WebRequest -Uri $Uri -OutFile $OutFile
}

function Download-Kodak {
  New-DatasetDirectory $KodakDir

  Write-Host ""
  Write-Host "[1/2] Kodak lossless PNG images"

  1..24 | ForEach-Object {
    $n = "{0:D2}" -f $_
    $url = "https://www.r0k.us/graphics/kodak/kodak/kodim$n.png"
    $out = Join-Path $KodakDir "kodim$n.png"
    Invoke-DownloadFile -Uri $url -OutFile $out
  }

  Assert-ImageCount -Path $KodakDir -Name "Kodak" -Expected 24
}

function Download-Div2kValid {
  New-DatasetDirectory $Div2kDir
  New-DatasetDirectory $DownloadsRoot

  Write-Host ""
  Write-Host "[2/2] DIV2K valid HR first 24 images"

  Invoke-DownloadFile `
    -Uri "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" `
    -OutFile $Div2kZip

  if (!(Test-Path $Div2kSourceDir)) {
    Write-Host "Extracting: $Div2kZip"
    Expand-Archive `
      -Path $Div2kZip `
      -DestinationPath $Div2kExtracted `
      -Force
  }
  else {
    Write-Host "Already extracted: $Div2kSourceDir"
  }

  1..24 | ForEach-Object {
    $id = 800 + $_
    $name = "{0:D4}.png" -f $id
    $src = Join-Path $Div2kSourceDir $name
    $dst = Join-Path $Div2kDir $name

    if (!(Test-Path $src)) {
      throw "Missing DIV2K source image: $src"
    }

    Copy-Item -Path $src -Destination $dst -Force
  }

  Assert-ImageCount -Path $Div2kDir -Name "DIV2K valid HR subset" -Expected 24

  if (!$KeepDownloads) {
    Write-Host "Removing temporary DIV2K extraction directory."
    Remove-Item -Path $Div2kExtracted -Recurse -Force
  }
}

Write-Host ""
Write-Host "=== Local image dataset reconstruction ==="
Write-Host "Root: $Root"

New-DatasetDirectory $KodakDir
New-DatasetDirectory $Div2kDir
New-DatasetDirectory $ClicDir
New-DatasetDirectory $TecnickDir

if ($SetupOnly) {
  Write-Host ""
  Write-Host "Dataset directories created."
  Write-Host "Kodak:     $KodakDir"
  Write-Host "DIV2K:     $Div2kDir"
  Write-Host "CLIC 2020: $ClicDir"
  Write-Host "Tecnick:   $TecnickDir"
  exit 0
}

if (!$Div2kOnly) {
  Download-Kodak
}

if (!$KodakOnly) {
  Download-Div2kValid
}

Write-Host ""
Write-Host "Manual/import-helper placeholders:"
Write-Host "CLIC 2020 validation subset: $ClicDir"
Write-Host "Tecnick TESTIMAGES subset:    $TecnickDir"

Write-Host ""
Write-Host "Dataset reconstruction completed."
