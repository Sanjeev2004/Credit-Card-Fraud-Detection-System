"""Compare four models without using test data for selection."""

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection.data import FEATURES, TARGET, load_dataset, split_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_candidates(ratio: float, seed: int, jobs: int, quick: bool):
    trees = 40 if quick else 200
    return {
        "logistic_regression": (
            Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
                max_iter=2000, random_state=seed))]),
            {"model__C": [0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            Pipeline([("model", RandomForestClassifier(
                n_estimators=trees, n_jobs=jobs, random_state=seed))]),
            {"model__max_depth": [8, None], "model__class_weight": [None, "balanced"]},
        ),
        "xgboost": (
            Pipeline([("model", XGBClassifier(
                n_estimators=trees, learning_rate=0.1, tree_method="hist",
                eval_metric="logloss", n_jobs=jobs, random_state=seed))]),
            {"model__max_depth": [3, 6], "model__scale_pos_weight": [1.0, ratio]},
        ),
        "lightgbm": (
            Pipeline([("model", LGBMClassifier(
                n_estimators=trees, learning_rate=0.1, n_jobs=jobs,
                random_state=seed, verbosity=-1, deterministic=True, force_col_wise=True))]),
            {"model__num_leaves": [15, 31], "model__scale_pos_weight": [1.0, ratio]},
        ),
    }


def choose_threshold(labels, scores, min_precision: float) -> tuple[float, bool]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(precision[:-1] >= min_precision)
    if len(eligible):
        # Thresholds are ascending: break equal-recall ties toward fewer alerts.
        best_recall = recall[eligible].max()
        index = eligible[recall[eligible] == best_recall][-1]
        return float(thresholds[index]), True
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    return float(thresholds[np.argmax(f1)]), False


def evaluate(labels, scores, threshold: float) -> dict:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(np.mean(np.asarray(labels) == predicted)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "alert_rate": float(np.mean(predicted)), "threshold": float(threshold),
    }


def train(data: str | Path, output: str | Path, seed: int = 42,
          jobs: int = 2, quick: bool = False, min_precision: float = 0.80) -> dict:
    if not 0 < min_precision <= 1:
        raise ValueError("Minimum precision must be in (0, 1].")
    if jobs < 1:
        raise ValueError("Jobs must be positive.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new run directory.")
    frame, data_summary = load_dataset(data)
    training, validation, test = split_dataset(frame, seed)
    output.mkdir(parents=True, exist_ok=True)
    ratio = float((training[TARGET] == 0).sum() / (training[TARGET] == 1).sum())
    models, validation_rows, search_rows = {}, [], []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for name, (pipeline, parameters) in model_candidates(ratio, seed, jobs, quick).items():
        print(f"Training {name} (4 candidates, 3 folds)...", flush=True)
        start = time.perf_counter()
        search = GridSearchCV(pipeline, parameters, scoring="average_precision", cv=cv,
                              n_jobs=1, error_score="raise", refit=True)
        search.fit(training[FEATURES], training[TARGET])
        elapsed = time.perf_counter() - start
        scores = search.best_estimator_.predict_proba(validation[FEATURES])[:, 1]
        threshold, met = choose_threshold(validation[TARGET], scores, min_precision)
        validation_rows.append({"model": name, **evaluate(validation[TARGET], scores, threshold),
                                "precision_target_met": met, "training_seconds": elapsed})
        models[name] = (search.best_estimator_, threshold)
        search_rows.append({"model": name, "best_parameters": search.best_params_,
                            "cv_average_precision": float(search.best_score_),
                            "candidates": [
                                {"parameters": p, "mean_cv_average_precision": float(s)}
                                for p, s in zip(search.cv_results_["params"],
                                                search.cv_results_["mean_test_score"])]})
    # Freeze selection before looking at any held-out test predictions.
    selected = max(validation_rows, key=lambda row: row["average_precision"])["model"]
    test_rows = []
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, (pipeline, threshold) in models.items():
        pipeline.predict_proba(test[FEATURES].iloc[:min(100, len(test))])
        start = time.perf_counter()
        scores = pipeline.predict_proba(test[FEATURES])[:, 1]
        seconds = time.perf_counter() - start
        test_rows.append({"model": name, **evaluate(test[TARGET], scores, threshold),
                          "prediction_seconds": seconds,
                          "batch_transactions_per_second": len(test) / seconds,
                          "amortized_ms_per_transaction": seconds * 1000 / len(test)})
        PrecisionRecallDisplay.from_predictions(test[TARGET], scores, name=name, ax=axis)
        cm_fig, cm_axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            test[TARGET], scores >= threshold, labels=[0, 1],
            display_labels=["Legitimate", "Fraud"], ax=cm_axis, colorbar=False)
        cm_axis.set_title(name)
        cm_fig.tight_layout()
        cm_fig.savefig(output / f"confusion_{name}.png", dpi=140)
        plt.close(cm_fig)
    axis.axhline(test[TARGET].mean(), linestyle="--", color="gray", label="Prevalence")
    axis.set_title("Held-out test precision-recall comparison")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=140)
    plt.close(fig)
    validation_table, test_table = pd.DataFrame(validation_rows), pd.DataFrame(test_rows)
    validation_table.to_csv(output / "validation_metrics.csv", index=False)
    test_table.to_csv(output / "test_metrics.csv", index=False)
    test_table.merge(validation_table[["model", "training_seconds"]], on="model").to_csv(
        output / "comparison.csv", index=False)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    pipeline, threshold = models[selected]
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "features": FEATURES,
                 "model_name": selected, "version": timestamp}, output / "model.joblib")
    with Path(data).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "version": timestamp, "selected_model": selected,
        "selection_metric": "validation average_precision",
        "threshold_policy": "Maximum validation recall at requested minimum precision; "
                            "fallback to maximum validation F1 if target is unattainable.",
        "minimum_validation_precision": min_precision,
        "dataset": {**data_summary, "sha256": digest},
        "splits": {name: {"rows": len(part), "frauds": int(part[TARGET].sum())}
                   for name, part in [("train", training), ("validation", validation), ("test", test)]},
        "seed": seed, "jobs": jobs, "quick": quick,
        "split_strategy": "stratified random 70/15/15 after exact-feature deduplication",
        "limitations": ["Random splits do not measure future-transaction generalization.",
                        "Class weighting can affect probability calibration; scores are not calibrated risks.",
                        "Precision targets on validation are not guarantees on test or production data.",
                        "Batch timing is not API latency or a production scalability guarantee."],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn",
                                                             "xgboost", "lightgbm", "joblib"]}},
        "search": search_rows, "validation": validation_rows, "test": test_rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(test_table.to_string(index=False))
    print(f"Selected on validation: {selected}. Artifacts: {output.resolve()}")
    return report
