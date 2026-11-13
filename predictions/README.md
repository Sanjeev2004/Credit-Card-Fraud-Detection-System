# Predictions

Batch scoring output is generated locally and is **not** committed to Git.

Example (after training a model and installing the package):

```bash
python -m fraud_detection.cli predict \
  --model artifacts/model.joblib \
  --data data/creditcard.csv \
  --output predictions/scores.csv \
  --chunk-size 10000
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli predict `
  --model artifacts/model.joblib `
  --data data/creditcard.csv `
  --output predictions/scores.csv `
  --chunk-size 10000
```

Output columns: `row_number`, `fraud_score`, `is_fraud`.
