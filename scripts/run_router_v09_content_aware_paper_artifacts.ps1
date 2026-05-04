$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 paper artifacts ==="

$BenchmarkTable = "results/routing_context/v09_content_aware_benchmark_table.csv"
$OverheadTable = "results/routing_context/v09_content_aware_overhead_table.csv"
$SensitivityTable = "results/routing_context/v09_knn_sensitivity_table.csv"
$OracleSummary = "results/routing_context/v09_content_oracle_summary.csv"
$OutDir = "results/routing_context/paper_artifacts_v09"

if (!(Test-Path $BenchmarkTable)) {
    Write-Host "Benchmark table not found. Running benchmark table script..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_aware_benchmark_table.ps1
}

if (!(Test-Path $OverheadTable) -or !(Test-Path $SensitivityTable)) {
    Write-Host "Overhead/sensitivity tables not found. Running overhead script..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_aware_overhead.ps1
}

if (!(Test-Path $OracleSummary)) {
    Write-Host "Oracle summary not found. Running oracle script..."
    powershell -ExecutionPolicy Bypass -File .\scripts\run_router_v09_content_oracle.ps1
}

Write-Host ""
Write-Host "[1/3] Building paper-ready tables and figures..."

python -m src.router.content_aware_paper_artifacts `
  --benchmark-table $BenchmarkTable `
  --overhead-table $OverheadTable `
  --sensitivity-table $SensitivityTable `
  --oracle-summary $OracleSummary `
  --out-dir $OutDir

if (!(Test-Path "$OutDir\v09_content_aware_final_table.csv")) {
    throw "Main paper table CSV was not created."
}

if (!(Test-Path "$OutDir\v09_content_aware_final_table.tex")) {
    throw "Main paper table LaTeX was not created."
}

if (!(Test-Path "$OutDir\v09_content_aware_overhead_table_paper.csv")) {
    throw "Overhead paper table CSV was not created."
}

if (!(Test-Path "$OutDir\v09_content_aware_best_k_table.csv")) {
    throw "Best-k paper table CSV was not created."
}

Write-Host ""
Write-Host "[2/3] Main paper table:"
Import-Csv "$OutDir\v09_content_aware_final_table.csv" |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[2b/3] Overhead paper table:"
Import-Csv "$OutDir\v09_content_aware_overhead_table_paper.csv" |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[2c/3] Best-k paper table:"
Import-Csv "$OutDir\v09_content_aware_best_k_table.csv" |
  Format-Table -AutoSize

Write-Host ""
Write-Host "[3/3] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 paper artifacts completed successfully."
