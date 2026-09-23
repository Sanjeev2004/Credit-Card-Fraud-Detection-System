# FraudLens · Credit Card Fraud Detection

**From transaction data to explainable evaluation and interactive fraud scoring.**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Models](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM%20%7C%20Logistic%20Regression-557C55)
![Interface](https://img.shields.io/badge/Dashboard-FraudLens-91AD68)

FraudLens is an end-to-end machine learning project that trains and compares fraud
detection models, exports the best validation performer, and serves predictions
through a **FastAPI backend and an interactive dashboard**. It focuses on the
metrics that matter for rare fraud: precision, recall, and average precision.

[Quick start](#quick-start) · [Results](#measured-results) · [Training](#train-your-own-model) · [Colab notebook](#train-in-google-colab) · [API](#api-and-batch-predictions) · [Troubleshooting](#troubleshooting)

## Highlights

- **Three-model comparison:** Logistic Regression, XGBoost, and LightGBM.
- **Fast, staged training:** one fit per model, early stopping for boosters, and checkpoints after each completed fit.
- **Automatic device selection:** use CUDA when available or continue on CPU with `device="auto"`.
- **Leakage-aware evaluation:** deduplication before splitting, separate early-stopping data, and validation-only model and threshold selection.
- **Interactive dashboard:** compare models, inspect detected and missed fraud, score transactions, and download results.
- **Flexible inference:** single or batch API requests, plus chunked CSV scoring for larger datasets.
- **Reproducible outputs:** dataset fingerprint, dependency versions, metrics, plots, and reload-verified model artifacts.

## How it works

```text
Transaction CSV
      │
      ▼
Schema validation → Exact-feature deduplication
      │
      ▼
Stratified split: 70% train / 15% validation / 15% test
      │
      ▼
Logistic Regression · XGBoost · LightGBM
      │
      ▼
Validation-based model selection and decision thresholds
      │
      ▼
Held-out test evaluation → Saved model + reports
      │
      ▼
FastAPI → FraudLens dashboard / Batch CSV predictions
```

## Quick start

### Requirements

- Python **3.11 or newer**
- Git
- Windows PowerShell for the commands below
- Internet access for first-time dependency installation and dataset download

A GPU is optional. The local launcher uses CPU training by default.

### 1. Clone the repository

```powershell
git clone https://github.com/Sanjeev2004/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System
```

### 2. Start the project

```powershell
.\run.ps1
```

On a fresh setup, the launcher creates a virtual environment, installs dependencies,
downloads the dataset if needed, trains the models, and starts the application.
If `artifacts/model.joblib` already exists, it loads that model directly.

### 3. Open the application

| Page | Local URL |
| --- | --- |
| Dashboard | http://127.0.0.1:8000 |
| Interactive API documentation | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

Keep the terminal running while using the app. Press **Ctrl+C** to stop the server.

<details>
<summary><strong>Manual setup or PowerShell script restrictions</strong></summary>

Run these commands from the repository root. Create the virtual environment only
if `.venv` does not already exist, and download the dataset only if the CSV is missing.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m fraud_detection.cli download --output data/creditcard.csv
.\.venv\Scripts\python.exe -m fraud_detection.cli train --output artifacts/manual-run --jobs 2
$env:FRAUD_MODEL_PATH = "artifacts/manual-run/model.joblib"
.\.venv\Scripts\python.exe -m uvicorn fraud_detection.api:app --host 127.0.0.1 --port 8000
```

For another training run, choose a new output directory or use `--resume` with
identical settings. On macOS/Linux, use `.venv/bin/python` and your shell's
environment-variable syntax instead.

</details>

## Explore the dashboard

| Section | What you can do |
| --- | --- |
| **Overview** | Inspect precision, recall, AP, fraud detection counts, and validation/test comparisons. |
| **Transaction scanner** | Load a held-out example, paste transaction JSON, or upload a numeric CSV. |
| **Model evaluation** | Compare test metrics, fit times, the selected model, and the saved threshold. |
| **Exports** | Download the evaluation report as JSON and prediction results as CSV. |

**Try it:** open **Transaction scanner**, choose an example, click **Load example**,
then **Score transactions**. The dashboard displays a fraud score and a
**Pass / Flag for review** decision for each transaction.

Dashboard batches support up to **1,000 transactions**; CSV uploads are limited
to **3 MB**. Use the CLI for the full dataset.

## Measured results

The following results come from the recorded full-dataset local CPU run dated
**September 18, 2026**. XGBoost was selected by **validation average precision**;
test results were not used to choose the model or tune its threshold.

| Dataset detail | Value |
| --- | ---: |
| Original transactions | 284,807 |
| Exact duplicates removed | 1,081 |
| Unique transactions | 283,726 |
| Fraud cases after deduplication | 473 |
| Held-out test transactions | 42,559 |
| Fraud cases in the test set | 71 |

| XGBoost test metric | Result |
| --- | ---: |
| Average precision | **0.7735** |
| Fraud precision | **75.34%** |
| Fraud recall | **77.46%** |
| F1 score | **0.7639** |
| ROC-AUC | **0.9711** |
| Accuracy | **99.9201%** |
| Detected / missed fraud | **55 / 16** |
| False alerts / correct legitimate predictions | **18 / 42,470** |

> **Why accuracy is not enough:** predicting every transaction as legitimate would
> already achieve **99.8332% accuracy**, while detecting **zero fraud cases**.
> Precision, recall, and AP provide a more useful picture of model performance.

The run's evidence is stored in generated `artifacts/report.json`,
`artifacts/comparison.csv`, and `artifacts/resume_bullets.md`. Datasets and trained
artifacts are excluded from Git; train locally or import an exported model after cloning.

## Dataset and input schema

This project uses the [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
containing anonymized European credit-card transactions from September 2013.

| Columns | Description |
| --- | --- |
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1`–`V28` | Anonymized PCA-transformed features |
| `Amount` | Transaction amount |
| `Class` | Training label: `0` = legitimate, `1` = fraud |

Training requires all **30 features plus `Class`**. Prediction requests require
exactly the 30 features, with finite numeric values and nonnegative `Time` and
`Amount`. CSV inference accepts an optional `Class` column and removes it before scoring.

Raw card numbers and CVVs are not model inputs. This project does not reconstruct
the anonymized PCA features from raw payment details.

## Train your own model

### Default three-model run

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli train --data data/creditcard.csv --output artifacts/my-run --jobs 2
```

### Automatically choose GPU or CPU

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli train --device auto --output artifacts/auto-run
```

### Train only XGBoost

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli train --models xgboost --output artifacts/xgb-only
```

### Resume an interrupted run

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli train --data data/creditcard.csv --output artifacts/my-run --jobs 2 --resume
```

Completed models load from checkpoints. An unfinished model restarts its fit.
Resuming requires the same dataset, code, dependencies, and configuration.

| Option | Default | Purpose |
| --- | --- | --- |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto`; explicit `cuda` requires working GPU support |
| `--jobs` | `2` | CPU thread limit for model fitting |
| `--max-trees` | `300` | Maximum boosting trees |
| `--patience` | `30` | Early-stopping patience |
| `--min-precision` | `0.80` | Validation precision target for threshold selection |
| `--seed` | `42` | Random seed |
| `--quick` | Off | Cap boosting at 40 trees while retaining all data rows |

To serve your new model:

```powershell
.\run.ps1 -ModelPath artifacts/my-run/model.joblib
```

An optional CPU-only four-model grid search also includes Random Forest:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli train --mode grid --output artifacts/grid --jobs 2
```

This performs **52 fits**, compared with three in the default workflow.
See [grid-search methodology](docs/grid-benchmark.md) for details.

## Train in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sanjeev2004/Credit-Card-Fraud-Detection-System/blob/main/notebooks/train_fraud_detection.ipynb)

The [standalone notebook](notebooks/train_fraud_detection.ipynb) embeds the training
source and provides a separate cell for each model.

1. Install the dependencies and configure your run.
2. Keep `DEVICE = "auto"` to use CUDA when available and CPU otherwise.
3. Run **Load the project training code**, then prepare the dataset and splits.
4. Run each model cell, followed by evaluation and export.
5. Download the serving ZIP and extract it into a new local folder, such as `artifacts/colab-fast`.
6. Install the recorded model dependencies and start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r artifacts/colab-fast/requirements-model.txt
.\run.ps1 -ModelPath artifacts/colab-fast/model.joblib
```

Set `SAVE_TO_DRIVE = True` to preserve checkpoints across Colab runtime deletion.
To resume, keep the same `RUN_NAME`, settings, and `RESUME = True`. Choose a new run
name when changing code, dependencies, data, or the resolved training device.
The serving ZIP excludes training checkpoints and the raw dataset.

## Evaluation methodology

1. **Validate and deduplicate:** reject invalid features and conflicting labels;
   remove exact-feature duplicates before splitting.
2. **Separate the data:** use a seeded stratified 70/15/15 train/validation/test split.
3. **Fit without test exposure:** Logistic Regression uses training-only scaling.
   Boosters reserve 15% of the training partition for early stopping, separate
   from both validation and test.
4. **Choose thresholds on validation:** maximize recall subject to the requested
   precision target. If the target is unattainable, use maximum F1 and record it.
5. **Select and evaluate:** choose the model with the highest validation AP,
   freeze thresholds, and evaluate configured models on the held-out test set.
6. **Verify the export:** reload the CPU-ready model and check prediction and
   decision parity.

The validation precision target is not a guarantee of test precision. These are
single-split historical results, not evidence of future payment performance;
only 71 fraud cases are present in the recorded test set. Scores are not calibrated
fraud probabilities. Actual CUDA/T4 execution has not been verified in this workspace.

## API and batch predictions

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | FraudLens dashboard |
| `GET` | `/health` | Service status and loaded model version |
| `GET` | `/metrics` | Evaluation report matching the loaded model |
| `GET` | `/examples` | Curated held-out examples |
| `POST` | `/predict` | Score 1–1,000 transactions |
| `GET` | `/docs` | Interactive OpenAPI documentation |

### Score a sample through the API

With the server running, execute in another PowerShell terminal:

```powershell
$examples = Invoke-RestMethod "http://127.0.0.1:8000/examples"
$body = @{ transactions = @($examples.examples[0].transaction) } | ConvertTo-Json -Depth 6
Invoke-RestMethod "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

Requests use `{"transactions": [...]}`. Each transaction must contain the exact
feature schema above. Responses include model metadata, the decision threshold,
and each transaction's `fraud_score` and `is_fraud` decision.

### Score a full CSV

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli predict --model artifacts/model.joblib --data data/creditcard.csv --output predictions/scores.csv --chunk-size 10000
```

The CLI processes bounded-memory chunks, preserves row order, and publishes the
output only after all chunks succeed. Output columns are `row_number`,
`fraud_score`, and `is_fraud`.

## Project structure

```text
├── src/fraud_detection/
│   ├── api.py                    # FastAPI routes and request validation
│   ├── dashboard.html            # Interactive dashboard
│   ├── data.py                   # Schema validation and dataset splitting
│   ├── fast_training.py          # Staged training, checkpoints, and export
│   ├── training.py               # Metrics and optional grid-search benchmark
│   ├── prediction.py             # Artifact loading and batch scoring
│   └── cli.py                    # Download, train, and predict commands
├── notebooks/
│   └── train_fraud_detection.ipynb
├── scripts/                      # Notebook generation and load testing
├── tests/                        # Pipeline, notebook, and API tests
├── docs/                         # Benchmark methodology
├── data/                         # Local dataset (generated/downloaded)
├── artifacts/                    # Local models and reports (generated)
├── run.ps1                       # Windows launcher
├── pyproject.toml                # Dependencies and project configuration
└── Dockerfile                    # Container build definition
```

### Training outputs

| Output | Contents |
| --- | --- |
| `model.joblib` | Selected pipeline, feature order, threshold, and version |
| `report.json` | Dataset hash, splits, configuration, metrics, and environment |
| `comparison.csv`, `*_metrics.csv` | Model comparison and validation/test results |
| `precision_recall.png`, `confusion_*.png` | Evaluation plots |
| `run_config.json`, `checkpoints/` | Resume configuration and completed model fits |
| `requirements-model.txt` | Recorded core model-library versions |
| `demo_samples.json` | Curated labeled examples for the dashboard |
| `resume_bullets.md` | Measured project-summary statements |

## Testing and development

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

**Latest verified suite: 44 tests passed.** Coverage includes training and
resumption, split separation, device selection, standalone notebook execution,
export parity, API inference, and CSV prediction. Full-dataset CPU training has
also completed successfully.

After changing the embedded training modules, regenerate the notebook:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py
```

## Troubleshooting

| Problem | Solution |
| --- | --- |
| **“XGBoost fell back to CPU”** | In the updated notebook, set `DEVICE = "auto"`. In an older open notebook, use `DEVICE = "cpu"`, rerun configuration, then rerun preparation. |
| **The notebook still runs old code** | Reopen the updated notebook and rerun **Load the project training code**. Local Python edits do not update the source embedded in an existing notebook session. |
| **“Cannot resume … changed”** | Choose a new `RUN_NAME` or output directory; the existing checkpoint configuration differs. |
| **“Output directory must be empty”** | Use a new output directory, or `--resume` for an unchanged interrupted run. |
| **Dataset download requires authentication** | Run `kagglehub.login()` interactively or place `creditcard.csv` in `data/`. |
| **Model file is missing** | Train a model or pass an existing artifact to `run.ps1 -ModelPath ...`. |
| **Port 8000 is in use** | Start with `.\run.ps1 -Port 8001` and open `http://127.0.0.1:8001`. |
| **Prediction input is rejected** | Include exactly `Time`, `V1`–`V28`, and `Amount` as finite numbers; omit `Class` from API requests. |

---

Built with **Python · scikit-learn · XGBoost · LightGBM · FastAPI · pandas · NumPy · Matplotlib**.

**Author:** [Sanjeev](https://github.com/Sanjeev2004)
