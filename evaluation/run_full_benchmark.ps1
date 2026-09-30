param(
    [int]$Limit = 0,
    [switch]$SkipBuild,
    [switch]$UseTemplateDataset
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".evalvenv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "未找到项目解释器：$VenvPython。请先创建并安装 .evalvenv 依赖。"
}
$Python = $VenvPython

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
