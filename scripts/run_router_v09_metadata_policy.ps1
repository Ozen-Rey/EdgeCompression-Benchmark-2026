$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== R-D-E Router v0.9 metadata content policy ==="

$BenchmarkCsv = "results/images/image_4dataset_RDE_paper_ready.csv"
$MetadataOracleCsv = "results/routing_context/v09_content_metadata_oracle.csv"

if (!(Test-Path $BenchmarkCsv)) {
    throw "Benchmark CSV not found: $BenchmarkCsv"
}

if (!(Test-Path $MetadataOracleCsv)) {
    Write-Host "Metadata/oracle CSV not found. Running metadata extraction first..."

    .\scripts\run_router_v09_content_metadata.ps1
}

$PolicyKeys = @("dataset", "resolution_class", "orientation_class")

foreach ($PolicyKey in $PolicyKeys) {
    Write-Host ""
    Write-Host "[policy=$PolicyKey] Evaluating metadata policy..."

    $DecisionsOut = "results/routing_context/v09_metadata_policy_${PolicyKey}_decisions.csv"
    $RulesOut = "results/routing_context/v09_metadata_policy_${PolicyKey}_rules.csv"
    $SummaryOut = "results/routing_context/v09_metadata_policy_${PolicyKey}_summary.csv"

    python -m src.router.adaptation.content_metadata_policy `
      --benchmark-csv $BenchmarkCsv `
      --metadata-oracle-csv $MetadataOracleCsv `
      --policy-key $PolicyKey `
      --evaluation-mode leave-one-out `
      --quality-floor 80 `
      --available-codecs "JPEG,JXL,HEVC" `
      --wR 0.2 `
      --wE 0.2 `
      --wD 0.6 `
      --decisions-out $DecisionsOut `
      --rules-out $RulesOut `
      --summary-out $SummaryOut

    if (!(Test-Path $SummaryOut)) {
        throw "Summary CSV not created for policy=$PolicyKey"
    }

    Write-Host ""
    Write-Host "Summary for policy=$PolicyKey"
    Import-Csv $SummaryOut | Format-Table -AutoSize
}

Write-Host ""
Write-Host "[final] Running pytest..."
python -m pytest tests -q

Write-Host ""
Write-Host "v0.9 metadata content policy completed successfully."
