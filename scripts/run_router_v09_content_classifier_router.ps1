$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 content classifier router validation ==="

$Config = "configs/content_classifier_image_v09.json"
$Image = "test_images/input.png"

$GlobalReport = "results/routing_context/v09_router_content_classifier_apply_global.json"
$TecnickReport = "results/routing_context/v09_router_content_classifier_apply_tecnick.json"

if (!(Test-Path $Config)) {
    throw "Missing classifier config: $Config"
}

if (!(Test-Path $Image)) {
    throw "Missing test image: $Image"
}

Write-Host ""
Write-Host "[1/3] Global pool: classifier prediction should fall back if not admissible..."

python -m src.router.rde_router `
  --config configs/router_image_v08.json `
  --content-classifier `
  --content-classifier-mode apply `
  --content-classifier-config $Config `
  --content-classifier-image $Image `
  --out $GlobalReport

$Global = Get-Content $GlobalReport | ConvertFrom-Json

if ($Global.content_classifier.prediction.codec -ne "JPEG") {
    throw "Expected global classifier prediction codec JPEG."
}

if ($Global.content_classifier.prediction.config -ne "q=85") {
    throw "Expected global classifier prediction config q=85."
}

if ($Global.content_classifier.applied -ne $false) {
    throw "Expected global classifier applied=false."
}

if ($Global.content_classifier.warnings -notcontains "content_classifier_prediction_not_admissible_fallback_to_router") {
    throw "Expected fallback warning in global classifier report."
}

if ($Global.decision.selected.codec -ne "HEVC") {
    throw "Expected global selected codec HEVC."
}

if ($Global.decision.selected.config -ne "crf=15") {
    throw "Expected global selected config crf=15."
}

if ($Global.decision.decision_trace.selected_reason -ne "lowest_J_RDE_in_safe_pool") {
    throw "Expected global selected_reason lowest_J_RDE_in_safe_pool."
}

Write-Host ""
Write-Host "Global fallback validated:"
Write-Host "  prediction:" $Global.content_classifier.prediction.codec $Global.content_classifier.prediction.config
Write-Host "  applied:" $Global.content_classifier.applied
Write-Host "  selected:" $Global.decision.selected.codec $Global.decision.selected.config
Write-Host "  reason:" $Global.decision.decision_trace.selected_reason

Write-Host ""
Write-Host "[2/3] Tecnick source-filtered pool: classifier prediction should apply..."

python -m src.router.rde_router `
  --config configs/router_image_v08.json `
  --content-classifier `
  --content-classifier-mode apply `
  --content-classifier-config $Config `
  --content-classifier-image $Image `
  --content-source tecnick `
  --content-source-filter `
  --content-filter-column dataset `
  --out $TecnickReport

$Tecnick = Get-Content $TecnickReport | ConvertFrom-Json

if ($Tecnick.content_filter.applied -ne $true) {
    throw "Expected Tecnick content filter applied=true."
}

if ($Tecnick.content_filter.value -ne "tecnick") {
    throw "Expected Tecnick content filter value=tecnick."
}

if ($Tecnick.content_classifier.prediction.codec -ne "JPEG") {
    throw "Expected Tecnick classifier prediction codec JPEG."
}

if ($Tecnick.content_classifier.prediction.config -ne "q=85") {
    throw "Expected Tecnick classifier prediction config q=85."
}

if ($Tecnick.content_classifier.applied -ne $true) {
    throw "Expected Tecnick classifier applied=true."
}

if ($Tecnick.decision.selected.codec -ne "JPEG") {
    throw "Expected Tecnick selected codec JPEG."
}

if ($Tecnick.decision.selected.config -ne "q=85") {
    throw "Expected Tecnick selected config q=85."
}

if ($Tecnick.decision.decision_trace.selected_reason -ne "content_classifier_preferred_candidate") {
    throw "Expected Tecnick selected_reason content_classifier_preferred_candidate."
}

Write-Host ""
Write-Host "Tecnick apply validated:"
Write-Host "  filter:" $Tecnick.content_filter.column "=" $Tecnick.content_filter.value "(" $Tecnick.content_filter.before_count "->" $Tecnick.content_filter.after_count ")"
Write-Host "  prediction:" $Tecnick.content_classifier.prediction.codec $Tecnick.content_classifier.prediction.config
Write-Host "  applied:" $Tecnick.content_classifier.applied
Write-Host "  selected:" $Tecnick.decision.selected.codec $Tecnick.decision.selected.config
Write-Host "  reason:" $Tecnick.decision.decision_trace.selected_reason

Write-Host ""
Write-Host "[3/3] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 content classifier router validation completed successfully."