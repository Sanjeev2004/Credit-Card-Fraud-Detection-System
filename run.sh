#!/usr/bin/env bash
# Linux/macOS launcher mirroring run.ps1 essentials.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH="${MODEL_PATH:-artifacts/model.joblib}"
PORT="${PORT:-8000}"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv .venv
  "$PYTHON" -m pip install -e .
fi

if ! "$PYTHON" -c "import fraud_detection, uvicorn, xgboost, lightgbm" >/dev/null 2>&1; then
  "$PYTHON" -m pip install -e .
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ "$MODEL_PATH" != "artifacts/model.joblib" ]]; then
    echo "Model not found: $MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! -f "data/creditcard.csv" ]]; then
    "$PYTHON" -m fraud_detection.cli download --output data/creditcard.csv
  fi
  "$PYTHON" -m fraud_detection.cli train --output artifacts --resume --jobs 2
fi

export FRAUD_MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
echo "Dashboard: http://127.0.0.1:${PORT} | API docs: http://127.0.0.1:${PORT}/docs"
exec "$PYTHON" -m uvicorn fraud_detection.api:app --host 127.0.0.1 --port "$PORT"
