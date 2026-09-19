param(
    [string]$ModelPath = "artifacts/model.joblib",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ is required." }
    & $pythonPath -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
& $pythonPath -c "import fraud_detection, uvicorn, xgboost, lightgbm"
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    if ($ModelPath -ne "artifacts/model.joblib") { throw "Model not found: $ModelPath" }
    if (-not (Test-Path -LiteralPath "data/creditcard.csv")) {
        & $pythonPath -m fraud_detection.cli download --output data/creditcard.csv
        if ($LASTEXITCODE -ne 0) { throw "Download failed. Place creditcard.csv in data/." }
    }
    & $pythonPath -m fraud_detection.cli train --output artifacts --resume --jobs 2
    if ($LASTEXITCODE -ne 0) { throw "Training failed; inspect the error above." }
}
$env:FRAUD_MODEL_PATH = (Resolve-Path -LiteralPath $ModelPath).Path
Write-Host "Dashboard: http://127.0.0.1:$Port | API docs: http://127.0.0.1:$Port/docs"
& $pythonPath -m uvicorn fraud_detection.api:app --host 127.0.0.1 --port $Port
