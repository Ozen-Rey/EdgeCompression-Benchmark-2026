$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.6 validation ==="

$PureReport = "results/routing_context/v06_config_report.json"
$JxlPlanReport = "results/routing_context/v06_jxl_external_command_plan.json"
$JxlExecuteReport = "results/routing_context/v06_jxl_execution_validation.json"

Write-Host ""
Write-Host "[1/4] Pure config run..."
python -m src.router.rde_router `
  --config configs/router_image_v05.json `
  --out $PureReport

if (!(Test-Path $PureReport)) {
  throw "Pure config report not created: $PureReport"
}

$pure = Get-Content $PureReport | ConvertFrom-Json

if ($pure.run_manifest.enabled -ne $true) {
  throw "run_manifest.enabled is not true."
}

if ($pure.decision.decision_trace.enabled -ne $true) {
  throw "decision_trace.enabled is not true."
}

if ($null -eq $pure.decision.selected.cost_decomposition) {
  throw "cost_decomposition missing in selected decision."
}

Write-Host "Pure config selected:" $pure.decision.selected.codec $pure.decision.selected.config
Write-Host "Run manifest git:" $pure.run_manifest.git.commit_short
Write-Host "Selected reason:" $pure.decision.decision_trace.selected_reason

Write-Host ""
Write-Host "[2/4] JXL external_command plan..."
python -m src.router.rde_router `
  --config configs/router_image_v05.json `
  --available-codecs "JXL" `
  --out $JxlPlanReport

if (!(Test-Path $JxlPlanReport)) {
  throw "JXL plan report not created: $JxlPlanReport"
}

$jxlPlan = Get-Content $JxlPlanReport | ConvertFrom-Json

if ($jxlPlan.decision.selected.codec -ne "JXL") {
  throw "Expected JXL, got: $($jxlPlan.decision.selected.codec)"
}

if ($jxlPlan.execution_plan.execution_backend -ne "external_command") {
  throw "Expected external_command backend, got: $($jxlPlan.execution_plan.execution_backend)"
}

Write-Host "JXL plan backend:" $jxlPlan.execution_plan.execution_backend
Write-Host "JXL selected:" $jxlPlan.decision.selected.codec $jxlPlan.decision.selected.config

Write-Host ""
Write-Host "[3/4] JXL execute + output validation..."
python -m src.router.rde_router `
  --config configs/router_image_v05.json `
  --available-codecs "JXL" `
  --execute `
  --out $JxlExecuteReport

if (!(Test-Path $JxlExecuteReport)) {
  throw "JXL execute report not created: $JxlExecuteReport"
}

$jxlExec = Get-Content $JxlExecuteReport | ConvertFrom-Json

if ($jxlExec.execution_result.success -ne $true) {
  throw "JXL execution failed."
}

if ($jxlExec.execution_validation.output_exists -ne $true) {
  throw "Execution validation failed: output does not exist."
}

if ($jxlExec.execution_validation.output_nonempty -ne $true) {
  throw "Execution validation failed: output is empty."
}

if ($jxlExec.execution_validation.extension_valid -ne $true) {
  throw "Execution validation failed: extension is invalid."
}

Write-Host "Execution output:" $jxlExec.execution_validation.output
Write-Host "Output size:" $jxlExec.execution_validation.output_size_bytes
Write-Host "Execution time ms:" $jxlExec.execution_validation.execution_time_ms

Write-Host ""
Write-Host "[4/4] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.6 validation completed successfully."
