$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 content-aware validation ==="

$BenchmarkCsv = "results/images/image_4dataset_RDE_paper_ready.csv"

$OracleByImage = "results/routing_context/v09_content_oracle_by_image.csv"
$OracleSummary = "results/routing_context/v09_content_oracle_summary.csv"

$MetadataFeatures = "results/routing_context/v09_content_metadata_features.csv"
$MetadataOracle = "results/routing_context/v09_content_metadata_oracle.csv"
$MetadataSummary = "results/routing_context/v09_content_metadata_summary.csv"

$DatasetPolicyDecisions = "results/routing_context/v09_metadata_policy_dataset_decisions.csv"
$DatasetPolicyRules = "results/routing_context/v09_metadata_policy_dataset_rules.csv"
$DatasetPolicySummary = "results/routing_context/v09_metadata_policy_dataset_summary.csv"

$RouterReportOnly = "results/routing_context/v09_router_content_policy_report_only_tecnick.json"
$RouterApply = "results/routing_context/v09_router_content_policy_apply_tecnick.json"

if (!(Test-Path $BenchmarkCsv)) {
    throw "Benchmark CSV not found: $BenchmarkCsv"
}

Write-Host ""
Write-Host "[1/6] Running content oracle/regret analysis..."

python -m src.router.analysis.content_oracle_analysis `
  --csv $BenchmarkCsv `
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
  --summary-out $OracleSummary

if (!(Test-Path $OracleByImage)) {
    throw "Oracle by-image CSV not created."
}

if (!(Test-Path $OracleSummary)) {
    throw "Oracle summary CSV not created."
}

Write-Host ""
Write-Host "[2/6] Extracting metadata features and joining oracle labels..."

python -m src.router.adaptation.content_metadata_features `
  --csv $BenchmarkCsv `
  --oracle-by-image $OracleByImage `
  --dataset-col dataset `
  --image-col image `
  --width-col width `
  --height-col height `
  --pixels-col pixels `
  --features-out $MetadataFeatures `
  --joined-out $MetadataOracle `
  --summary-out $MetadataSummary

if (!(Test-Path $MetadataOracle)) {
    throw "Metadata/oracle CSV not created."
}

if (!(Test-Path $MetadataSummary)) {
    throw "Metadata summary CSV not created."
}

Write-Host ""
Write-Host "[3/6] Evaluating dataset-majority metadata policy..."

python -m src.router.adaptation.content_metadata_policy `
  --benchmark-csv $BenchmarkCsv `
  --metadata-oracle-csv $MetadataOracle `
  --policy-key dataset `
  --evaluation-mode leave-one-out `
  --quality-floor 80 `
  --available-codecs "JPEG,JXL,HEVC" `
  --wR 0.2 `
  --wE 0.2 `
  --wD 0.6 `
  --decisions-out $DatasetPolicyDecisions `
  --rules-out $DatasetPolicyRules `
  --summary-out $DatasetPolicySummary

if (!(Test-Path $DatasetPolicyRules)) {
    throw "Dataset policy rules CSV not created."
}

if (!(Test-Path $DatasetPolicySummary)) {
    throw "Dataset policy summary CSV not created."
}

Write-Host ""
Write-Host "[4/6] Running router with source-aware content policy in report-only mode..."

python -m src.router.rde_router `
  --config configs/router_image_v08.json `
  --content-policy `
  --content-policy-mode report-only `
  --content-policy-rules-file $DatasetPolicyRules `
  --content-policy-key dataset `
  --content-source tecnick `
  --content-source-filter `
  --out $RouterReportOnly

Write-Host ""
Write-Host "[5/6] Running router with source-aware content policy in apply mode..."

python -m src.router.rde_router `
  --config configs/router_image_v08.json `
  --content-policy `
  --content-policy-mode apply `
  --content-policy-rules-file $DatasetPolicyRules `
  --content-policy-key dataset `
  --content-source tecnick `
  --content-source-filter `
  --out $RouterApply

if (!(Test-Path $RouterApply)) {
    throw "Router apply report not created."
}

Write-Host ""
Write-Host "[validation] Checking content-aware router report..."

$ApplyJson = Get-Content $RouterApply | ConvertFrom-Json

if ($ApplyJson.content_policy.applied -ne $true) {
    throw "Expected content_policy.applied = true."
}

if ($ApplyJson.decision.selected.codec -ne "JPEG") {
    throw "Expected selected codec JPEG, got $($ApplyJson.decision.selected.codec)."
}

if ($ApplyJson.decision.selected.config -ne "q=85") {
    throw "Expected selected config q=85, got $($ApplyJson.decision.selected.config)."
}

if ($ApplyJson.decision.decision_trace.selected_reason -ne "content_policy_preferred_candidate") {
    throw "Expected selected_reason content_policy_preferred_candidate."
}

if ($ApplyJson.content_filter.applied -ne $true) {
    throw "Expected content_filter.applied = true."
}

Write-Host ""
Write-Host "[summary] Oracle summary:"
Import-Csv $OracleSummary | Format-Table -AutoSize

Write-Host ""
Write-Host "[summary] Dataset metadata policy summary:"
Import-Csv $DatasetPolicySummary | Format-Table -AutoSize

Write-Host ""
Write-Host "[summary] Router apply result:"
Write-Host "Selected:" $ApplyJson.decision.selected.codec $ApplyJson.decision.selected.config
Write-Host "Selected reason:" $ApplyJson.decision.decision_trace.selected_reason
Write-Host "Content policy applied:" $ApplyJson.content_policy.applied
Write-Host "Content filter:" $ApplyJson.content_filter.column "=" $ApplyJson.content_filter.value "(" $ApplyJson.content_filter.before_count "->" $ApplyJson.content_filter.after_count ")"

Write-Host ""
Write-Host "[6/6] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 content-aware validation completed successfully."