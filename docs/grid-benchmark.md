# Credit Card Fraud Detection

A portfolio benchmark comparing Logistic Regression, Random Forest, XGBoost, and LightGBM, with chunked CSV scoring and a FastAPI inference service. This README describes the implementation, not measured performance or production readiness.

## Dataset and Input Contract

Use the original ULB Machine Learning Group **Credit Card Fraud Detection** dataset: [Kaggle `mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), file `creditcard.csv`. It contains European cardholder transactions from two days in September 2013, approximately **284,807 transactions and 492 frauds before deduplication** (about 0.172%). The dataset arose from a collaboration between Worldline and ULB's Machine Learning Group.

Review the dataset's current license, access conditions, and Kaggle terms before downloading, using, or redistributing it. The download command uses `kagglehub`; if authentication is required, use its supported local credential configuration or interactive `kagglehub.login()`. Keep credentials outside the repository and container image; never paste tokens into commands, logs, screenshots, or commits. A manually downloaded `creditcard.csv` can also be placed in `data/`.

- Exactly 30 features are required: `Time`, `V1` through `V28`, and `Amount`.
- All features must be finite numeric values, with no missing values or booleans. `Time` and `Amount` must be nonnegative.
- `Time` is elapsed seconds from the first dataset transaction; `Amount` is the transaction amount. `V1` through `V28` are anonymized PCA features.
- `Class` is the training-only target: `0` legitimate, `1` fraud. Training requires both classes and at least 20 unique examples per class after deduplication.
- This is **not compatible with raw card input** such as card numbers, CVVs, merchant records, or arbitrary payment fields. The original PCA transformation is not supplied here, so raw transactions cannot be converted into compatible features by this project.

## Windows Quick Start

This is the optional legacy grid benchmark. For the new fast notebook and default workflow, see the root README.

### Local CPU training

Run from the project root in Windows PowerShell. Python 3.11+ is required; the Docker image uses Python 3.13. Explicit interpreter paths avoid activation and execution-policy changes.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m fraud_detection.cli download --output data/creditcard.csv
.\.venv\Scripts\python.exe -m fraud_detection.cli train --mode grid --data data/creditcard.csv --output artifacts --jobs 2 --min-precision 0.80
```

The downloader refuses to overwrite an existing destination. Training requires a new or empty output directory. Default seed is `42`; `--jobs` controls model-level parallelism, not parallel grid searches.

For a shorter run, use a separate output directory:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli train --mode grid --data data/creditcard.csv --output artifacts-quick --quick --jobs 2
```

`--quick` changes tree counts from 200 to 40, **not the data size**. All four models still search four parameter candidates with three folds each: 48 CV fits plus four best-candidate refits. Logistic Regression is unchanged. Training loads the full CSV into memory, including validation and deduplication; it is not streaming or distributed training.

## Methodology

1. Validate the exact training schema, reject identical feature rows with conflicting labels, and remove exact-feature duplicates before splitting.
2. Create stratified random train/validation/test splits of approximately 70/15/15 using the configured seed.
3. Tune each model on training data only with shuffled three-fold stratified CV, scored by average precision (AP). Grid search runs serially and refits its best candidate on the training split.
4. Choose each model's threshold on validation: maximize recall subject to the requested minimum precision, breaking equal-recall ties toward the higher threshold. **If the target is unattainable, fall back to maximum validation F1**, marked by `precision_target_met=false` in validation results.
5. Select the deployed model by **validation AP**, not thresholded F1, test metrics, or whether the precision target was met. The selected pipeline is not retrained on train plus validation.
6. Freeze selection and thresholds, then evaluate all four models once on the held-out test set for comparison. Each also gets an untimed prediction warmup on up to 100 test rows before timed full-test scoring; this does not tune the models.

| Model | Search and preprocessing |
| --- | --- |
| Logistic Regression | Pipeline `StandardScaler` fitted within each CV training fold; `C` in `{0.1, 1.0}`, `class_weight` in `{None, balanced}`; `max_iter=2000` |
| Random Forest | No scaling; `max_depth` in `{8, None}`, `class_weight` in `{None, balanced}` |
| XGBoost | No scaling; histogram trees, learning rate 0.1; `max_depth` in `{3, 6}`, `scale_pos_weight` in `{1, training negative/positive ratio}` |
| LightGBM | No scaling; learning rate 0.1, deterministic column-wise mode; `num_leaves` in `{15, 31}`, same weight choices as XGBoost |

The boosting weight ratio is calculated once from the entire training split, not separately per CV fold. There is no resampling, SMOTE, calibration, or early stopping. Class weighting can change calibration: `fraud_score` is a model score in `[0, 1]`, **not a calibrated fraud risk**. Validation precision targets are not guarantees on test or future data.

Random splits do not measure future-transaction generalization, account-level separation, or distribution drift. Exact deduplication only prevents identical feature rows crossing splits. The original PCA fitting procedure and fitting population are unknown to this project; fold-local scaling cannot establish that upstream PCA was leakage-free. The small, historical two-day sample is not evidence of present-day payment performance.

## Training Outputs

Each run writes the following to its output directory:

- `model.joblib`: selected fitted pipeline, feature order, threshold, model name, and timestamp version; only the selected model is saved.
- `report.json`: dataset counts and SHA-256, split counts, seed/options, environment versions, candidate CV results, validation/test metrics, selection policy, and limitations.
- `validation_metrics.csv`: per-model metrics, thresholds, precision-target status, and search/refit duration.
- `test_metrics.csv`: AP, ROC AUC, precision, recall, F1, confusion counts, alert rate, threshold, and batch prediction timing.
- `comparison.csv`: test results joined with training durations.
- `precision_recall.png` and four `confusion_<model>.png` plots.

Use generated results for any portfolio performance claims; none are asserted here. Test batch transactions/second and amortized milliseconds/transaction are throughput measurements, not per-request API latency or a scalability guarantee.

## CSV Prediction

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli predict --model artifacts/model.joblib --data data/creditcard.csv --output predictions/scores.csv --chunk-size 10000
```

Prediction requires all 30 features; the CSV command tolerates and discards `Class` for convenience when scoring a labeled dataset. Other extra columns are rejected. Output preserves input order using zero-based `row_number`, followed by `fraud_score` and `is_fraud` (`score >= threshold`), **without raw features or labels**.

CSV input is processed in bounded-memory chunks. Results go to a temporary file in the destination directory, then atomically replace the destination only after all chunks succeed. Empty input or validation failure prevents publishing a partial result. Existing prediction output may be replaced, but input and model paths are protected from overwrite.

**Load only trusted artifacts.** Joblib uses pickle and can execute code during loading; schema checks happen afterward and are not a security boundary. Preserve the exact training dependency versions and use a compatible Python/platform runtime for serving. `pyproject.toml` contains version ranges, not a lockfile; a fresh install or Docker build may resolve different versions. `report.json` records Python/platform and core model-library versions, but not a complete environment lock. Cross-version or Windows-to-Linux artifact compatibility is not guaranteed; validate or retrain in the serving environment.

## Local API

Start the service after training, bound to loopback:

```powershell
$env:FRAUD_MODEL_PATH = (Resolve-Path .\artifacts\model.joblib).Path
.\.venv\Scripts\python.exe -m uvicorn fraud_detection.api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal, check health and send one synthetic schema-valid transaction. No browser is needed; zero-valued PCA features are only a connectivity example, not a meaningful fraud scenario.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$transaction = @{ Time = 0.0; Amount = 10.0 }
1..28 | ForEach-Object { $transaction["V$_"] = 0.0 }
$body = @{ transactions = @($transaction) } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```

`POST /predict` accepts exactly a `transactions` list of **1..1000** objects, each containing all 30 features and **no `Class` or extra keys**. JSON numbers are required, not numeric strings, booleans, nulls, or nonfinite values. Invalid requests return a generic 422 without echoing submitted features. Responses include `model_name`, `model_version`, `threshold`, and ordered `predictions` containing `fraud_score` and `is_fraud`. `GET /health` returns status and model version.

The trusted model is loaded once per process at startup; missing artifacts prevent startup. Requests are stateless, but every worker/replica holds its own model copy. Horizontal scaling still needs consistent artifacts, a load balancer, memory/CPU budgeting, and concurrency testing. Multiple workers plus model-internal threads can oversubscribe CPUs.

There is **no built-in authentication, TLS, rate limiting, or request-body byte limit**. The 1000-row validation limit is not a body-size defense because parsing happens first. Keep local use on loopback; before exposing the service, place it behind a reverse proxy providing TLS, authentication, explicit body-size limits, rate limits, and timeouts. Avoid logging financial payloads.

## Docker on Windows

Use Docker Desktop in Linux-container mode. The Dockerfile uses `python:3.13-slim`, installs Linux `libgomp1` for native OpenMP dependencies, and runs as a non-root user. The artifact must be readable by that user and compatible with the image's dependency versions.

```powershell
docker build -t fraud-detection .
$artifacts = (Resolve-Path .\artifacts).Path
docker run --rm -p 127.0.0.1:8000:8000 --mount "type=bind,source=$artifacts,target=/app/artifacts,readonly" fraud-detection
```

Stop any local service using port 8000 first. The read-only mount supplies `/app/artifacts/model.joblib`; container binding is `0.0.0.0`, but the published host port above is loopback-only. On other minimal Linux installations, native libraries may similarly require `libgomp1` (Debian/Ubuntu name).

## Load Test

With the API running, replay a small CSV sample:

```powershell
.\.venv\Scripts\python.exe scripts/load_test.py --url http://127.0.0.1:8000 --data data/creditcard.csv --requests 1000 --concurrency 10 --batch-size 32 --timeout 30
```

The script repeatedly sends the first `--batch-size` rows (1..1000), selecting only model features, with bounded concurrency and **no warmup**. It reports attempted/successful requests and transactions, failure categories, throughput, and p50/p95 request latency for all requests and successful requests separately. It exits nonzero if any request fails.

This synthetic workload/replay checks HTTP response structure and score ranges, **not predictive accuracy**; labels are not evaluated. Request latency covers a whole batch and client-side work, while transactions/second is batch throughput, not individual-transaction latency. Repeated identical rows and initial connection costs do not represent realistic production traffic.

## Development Checks

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

These are commands to run locally, not a claim that checks, training, Docker, or benchmarks have been executed for this README.
