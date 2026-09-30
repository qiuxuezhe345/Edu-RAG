param(
    [int]$Limit = 0,
    [switch]$SkipBuild,
    [switch]$UseTemplateDataset
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".evalvenv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Push-Location $ProjectRoot
try {
    if (-not $SkipBuild) {
        $BuildArgs = @("-m", "evaluation.build_dataset")
        if ($UseTemplateDataset) {
            $BuildArgs += "--no-llm"
        }
        & $Python @BuildArgs
        if ($LASTEXITCODE -ne 0) { throw "Dataset build failed" }
    }

    $RetrievalArgs = @("-m", "evaluation.run_retrieval_eval")
    $GenerationArgs = @("-m", "evaluation.run_generation_eval")
    if ($Limit -gt 0) {
        $RetrievalArgs += @("--limit", $Limit)
        $GenerationArgs += @("--limit", $Limit)
    }

    & $Python @RetrievalArgs
    if ($LASTEXITCODE -ne 0) { throw "Retrieval evaluation failed" }

    & $Python @GenerationArgs
    if ($LASTEXITCODE -ne 0) { throw "Generation evaluation failed" }

    Write-Host "Benchmark complete. Results are under evaluation/results/."
}
finally {
    Pop-Location
}
