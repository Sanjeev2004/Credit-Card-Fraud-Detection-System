# FraudLens — Credit Card Fraud Detection

Train, evaluate, and serve a credit-card fraud model with a local FastAPI dashboard.
The default workflow uses three model fits, separate notebook cells, early stopping,
and resumable checkpoints. No Docker is required.

## Run the project

From this folder in Windows PowerShell:

```powershell
.\run.ps1
```

Open **http://127.0.0.1:8000**. API documentation: **http://127.0.0.1:8000/docs**.
The launcher uses the existing `artifacts/model.joblib`. On a fresh setup it creates
a virtual environment, installs dependencies, downloads the dataset if missing,
trains, and starts the API. Python 3.11+ is required. Stop with Ctrl+C.

The dashboard shows the matching evaluation report, model comparison, confusion
matrix, and predictions. Choose a labeled example, click **Load example**, then
**Score transactions**. You can also paste a JSON array or load a numeric sample
CSV of up to 1,000 rows. Example labels are for comparison and are not model inputs.

If your PowerShell policy blocks scripts, run the explicit commands below instead
of changing a machine-wide execution policy:

```powershell
python -m venv .venv  # Only if .venv does not already exist
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn fraud_detection.api:app --host 127.0.0.1 --port 8000
```

Training is needed first if `artifacts/model.joblib` is absent; see below.

## Measured local result (2026-09-18)

These numbers were obtained from the full local `data/creditcard.csv`, not synthetic
test fixtures. The selected model is **XGBoost**, selected by **validation AP**.
The test metrics below were not used for selection or threshold tuning.

| Item | Measured value |
| --- | ---: |
| Original transactions | 284,807 |
| Exact duplicates removed | 1,081 |
| Unique transactions / frauds | 283,726 / 473 |
| Held-out test transactions / frauds | 42,559 / 71 |
| Test accuracy | 99.9201% |
| Fraud precision | 75.34% |
| Fraud recall | 77.46% |
| Fraud F1 | 0.7639 |
| Average precision (AP) | 0.7735 |
| ROC-AUC | 0.9711 |
| Detected fraud / missed fraud | 55 / 16 |
| False alerts / correct legitimate predictions | 18 / 42,470 |
| Combined model fit time, local CPU, 2 threads | 113.37 seconds |

Preparation took another 31.73 seconds; fit time excludes loading, validation,
plotting, export, installation and download. The three fit times were 76.42s
(Logistic Regression), 34.45s (XGBoost), and 2.50s (LightGBM). These are one-run
measurements, not a CPU/GPU benchmark or an old-vs-new wall-time speedup claim.

**Always predicting legitimate already achieves 99.8332% accuracy but 0% fraud
recall.** Include fraud precision, recall and AP when presenting this project.
Only 71 frauds are in the test set, so these metrics have meaningful uncertainty.

Evidence: [`artifacts/report.json`](artifacts/report.json),
[`artifacts/comparison.csv`](artifacts/comparison.csv), and
[`artifacts/resume_bullets.md`](artifacts/resume_bullets.md).
Artifacts are local/generated and ignored by Git; a clone must train or import a model.
Dataset SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

Resume bullet based on this run:

> Built a fraud detection pipeline comparing three models on 283,726 deduplicated
> transactions; achieved 0.773 test AP, 75.3% fraud precision and 77.5% recall with
> XGBoost on 42,559 held-out transactions, and integrated batch inference through
> FastAPI with a local evaluation dashboard.

Call these **historical dataset results**, not live banking or production outcomes.
Do not claim calibrated risk scores, production capacity, or a GPU runtime that
has not actually been measured.

## Why the previous notebook was slow

It ran 4 models × 4 parameter settings × 3 CV folds + 4 final refits = **52 fits**.
Selecting a T4 only accelerated XGBoost. Logistic Regression, Random Forest and
standard LightGBM still used CPU. The old `QUICK` switch reduced tree counts but
retained all 52 fits.

The new default:

- One fit each for Logistic Regression, XGBoost and LightGBM: **3 fits**.
- No Random Forest or grid search in the default path; the old benchmark remains optional.
- Histogram boosting with at most 300 trees and patience 30 for early stopping.
- Separate cells and immediate per-model checkpoints; resume completed models.
- Full dataset, unchanged class prevalence in validation/test, no SMOTE or undersampling.
- CPU-ready artifacts, so serving needs no GPU.

XGBoost uses `tree_method="hist"`, `device="cuda"` on Colab, and its sklearn wrapper
uses the best iteration after early stopping. LightGBM uses its early-stopping
callback on CPU. See the official [XGBoost API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html)
and [LightGBM callback documentation](https://lightgbm.readthedocs.io/en/stable/_modules/lightgbm/callback.html).

## Colab / T4 notebook

Open [`notebooks/train_fraud_detection.ipynb`](notebooks/train_fraud_detection.ipynb)
in Colab or your VS Code Colab kernel. It embeds the project training code and
does not need your local source files on the remote runtime.

1. Install packages and configure the run. Default: `DEVICE="auto"`, three models.
   XGBoost uses CUDA when available and otherwise continues on CPU. To require a
   GPU, select a GPU runtime and set `DEVICE="cuda"`. If the resolved device changes
   when resuming, use a new `RUN_NAME`.
2. Load embedded code, download/locate the dataset, verify CUDA and prepare splits.
3. Run the separate Logistic Regression, XGBoost and LightGBM cells.
4. Run evaluation/export once, then package and download the ZIP.
5. Extract it to `artifacts/colab-fast`, then serve it locally:

```powershell
.\.venv\Scripts\python.exe -m pip install -r artifacts/colab-fast/requirements-model.txt
.\run.ps1 -ModelPath artifacts/colab-fast/model.joblib
```

For **only one fast model**, set `MODELS = ("xgboost",)` before preparing the run.
The other model cells skip automatically. Do not change model choices in response
to test metrics; make development decisions using validation.

Set `SAVE_TO_DRIVE=True` to preserve checkpoints across runtime deletion. With the
same `RUN_NAME`, settings and `RESUME=True`, completed model cells load checkpoints.
An interrupted in-progress fit restarts. Without Drive the checkpoints disappear
when Colab deletes the runtime. A Windows path cannot be read by a remote Colab kernel.

Use a **new run name** when changing source, dependencies, dataset or settings.
The resume guard rejects mismatches. Do not use the serving ZIP as a training
checkpoint folder: it deliberately excludes the checkpoint directory.

The notebook's end-to-end CPU path has been tested on synthetic data. **The revised
CUDA/T4 path has not been executed in this workspace.** GPU speed and metrics may
differ; the GPU preflight prevents silently claiming CUDA while falling back to CPU.

## Local training

The dataset is the original [ULB/Kaggle creditcardfraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Use its `creditcard.csv`; the current workspace already has it. The downloader
refuses to overwrite an existing file. If Kaggle requires authentication, use
`kagglehub.login()` interactively and keep credentials out of project files.

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli download --output data/creditcard.csv
# Run download only when the CSV is missing.
.\.venv\Scripts\python.exe -m fraud_detection.cli train --data data/creditcard.csv --output artifacts/new-fast --jobs 2
# Resume an interrupted run with identical options:
.\.venv\Scripts\python.exe -m fraud_detection.cli train --data data/creditcard.csv --output artifacts/new-fast --jobs 2 --resume
# One model only:
.\.venv\Scripts\python.exe -m fraud_detection.cli train --models xgboost --output artifacts/xgb-only
# Serve the chosen run:
.\run.ps1 -ModelPath artifacts/new-fast/model.joblib
```

Outputs must go to an empty/new directory unless `--resume` is used. `--quick`
caps boosting at 40 trees for a smoke run; it still uses all rows. Other options:
`--device cuda`, `--max-trees 300`, `--patience 30`, `--min-precision 0.80`.

Optional expensive legacy comparison:

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli train --mode grid --output artifacts/grid --jobs 2
```

See [`docs/grid-benchmark.md`](docs/grid-benchmark.md) for its 52-fit methodology.
Grid mode remains CPU-only and does not support fast-mode checkpoint resumption.

## Evaluation design

1. Validate schema, reject identical features with conflicting labels and remove exact duplicates.
2. Seeded stratified random split: 70% train, 15% validation, 15% test.
3. Logistic Regression fits on all training rows with training-only scaling.
   Boosters fit on 85% of training; its remaining 15% selects the stopping iteration.
   This inner stopping split is disjoint from model-selection validation and test.
4. On validation, maximize recall subject to the requested minimum precision;
   if unattainable, use maximum F1 and record `precision_target_met=false`.
5. Select the deployed model by **validation AP**, freeze model and thresholds,
   then evaluate all configured models on test. No train-plus-validation refit.
6. Export and reload the selected CPU model; verify prediction and decision parity.

The 80% precision target applies to validation, not future/test data. This run's
selected model obtained 75.34% test precision; that is reported as measured.
Random splits of two days of historical transactions do not measure future drift
or account-level generalization. The upstream PCA fitting procedure is unknown.

## Generated files

| File | Purpose |
| --- | --- |
| `model.joblib` | Selected pipeline, feature order, threshold and run version |
| `report.json` | Dataset hash, splits, parameters, versions, metrics and limitations |
| `comparison.csv`, `test_metrics.csv`, `validation_metrics.csv` | Numeric evidence |
| `precision_recall.png`, `confusion_*.png` | Plots for project presentation |
| `resume_bullets.md` | Measured resume statements with context |
| `requirements-model.txt` | Exact core model-library versions for serving |
| `demo_samples.json` | Curated labeled test examples, not representative traffic |
| `run_config.json`, `checkpoints/` | Resume identity and completed model fits |

Load only your own trusted joblib artifacts: they use pickle. Core version pins
help reproduce inference, but are not a full cross-platform environment lock.

## Batch scoring and API

```powershell
.\.venv\Scripts\python.exe -m fraud_detection.cli predict --model artifacts/model.joblib --data data/creditcard.csv --output predictions/scores.csv --chunk-size 10000
```

The full CSV is scored in bounded-memory chunks, keeping input row order. Output
is atomically published only after all chunks succeed. Output contains `row_number`,
`fraud_score`, `is_fraud`. Optional `Class` is discarded; other extra columns fail validation.

`POST /predict` accepts `{"transactions": [{...}]}` with 1–1,000 objects. Each must
contain exactly **Time, V1–V28 and Amount**, all finite JSON numbers. Time/Amount
must be nonnegative. No Class, strings, booleans, missing values or extra keys.
`GET /health`, `/metrics`, `/examples`, `/` and `/docs` support the local demo.

These anonymized PCA features cannot be derived from raw card numbers/CVVs with
this project. Scores are not calibrated risk probabilities. Keep the service on
loopback; it has no authentication, TLS, rate limits or request-body byte limit.

## Verification and maintenance

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
# Regenerate the standalone notebook after changing training source:
.\.venv\Scripts\python.exe scripts/build_notebook.py
```

Verified in this workspace: 40 tests passed; full local CPU training completed;
standalone notebook CPU execution, resumable checkpoints, split separation, export
parity, API prediction parity and CLI batch scoring are covered. Browser visual QA
and revised T4 execution are not verified. This is a runnable local portfolio
project, not a production banking system.
