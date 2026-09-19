"""Staged, resumable training: one fit per model, with a separate stopping split."""

import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from fraud_detection.data import FEATURES, TARGET, load_dataset, split_dataset
from fraud_detection.training import choose_threshold, evaluate

FAST_MODELS = ("logistic_regression", "xgboost", "lightgbm")
PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def dump_model(path, value):
    temporary = Path(str(path) + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def check_device(device, seed=42):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda.")
    if device == "cuda":
        probe = XGBClassifier(n_estimators=1, max_depth=1, tree_method="hist", device="cuda")
        probe.fit(np.random.default_rng(seed).normal(size=(32, 4)), np.tile([0, 1], 16))
        actual = json.loads(probe.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError("XGBoost fell back to CPU. Select a GPU runtime or device='cpu'.")
        print(f"XGBoost GPU verified: {actual}", flush=True)


def prepare_run(data, output, seed=42, jobs=2, min_precision=0.8,
                device="cpu", models=FAST_MODELS, max_trees=300, patience=30, resume=False):
    """Validate once, then reuse the returned state across notebook cells."""
    if not 0 < min_precision <= 1 or jobs < 1:
        raise ValueError("Minimum precision must be in (0, 1] and jobs must be positive.")
    models = tuple(models)
    if not models or len(set(models)) != len(models) or set(models) - set(FAST_MODELS):
        raise ValueError(f"Choose unique models from {FAST_MODELS}.")
    if max_trees < 1 or patience < 1:
        raise ValueError("max_trees and patience must be positive.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda.")
    start = time.perf_counter()
    data, output = Path(data), Path(output)
    with data.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    # Do not reuse checkpoints after code, data, dependencies, or settings change.
    source_hash = hashlib.sha256()
    for name in ("data.py", "training.py", "fast_training.py"):
        source_hash.update(Path(__file__).with_name(name).read_bytes())
    packages = {p: version(p) for p in PACKAGES}
    config = {"seed": seed, "jobs": jobs, "min_precision": min_precision, "device": device,
                  "models": list(models), "max_trees": max_trees, "patience": patience,
                  "dataset_sha256": digest, "source_sha256": source_hash.hexdigest(),
                  "packages": packages, "python": platform.python_version()}
    manifest_path = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError("Output directory must be empty; use --resume or a new run directory.")
        if not manifest_path.exists():
            raise ValueError("Cannot resume: run_config.json is missing. Use a new directory.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Cannot resume: data, code, settings or versions changed. Use a new directory.")
    else:
        manifest = {"version": datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"), "config": config}
    if "xgboost" in models:
        check_device(device, seed)
    print("Loading, validating and deduplicating dataset...", flush=True)
    frame, summary = load_dataset(data)
    training, validation, test = split_dataset(frame, seed)
    # Boosting stopping data is drawn ONLY from the training partition.
    fit, stopping = train_test_split(training, test_size=0.15, random_state=seed,
                                    stratify=training[TARGET])
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    write_json(manifest_path, manifest)
    state = {"config": config, "output": output, "manifest": manifest, "summary": summary,
                 "training": training, "validation": validation, "test": test, "fit": fit, "stopping": stopping,
                 "results": {}, "preparation_seconds": time.perf_counter() - start}
    print(f"Ready: {len(frame):,} unique rows; {len(test):,} held out. "
          f"{len(models)} model fits, no grid search.", flush=True)
    return state


def fit_model(state, name):
    """One model per call/cell; successful models are checkpointed immediately."""
    config, output = state["config"], state["output"]
    if name not in config["models"]:
        raise ValueError(f"Model {name} was not configured for this run.")
    checkpoint = output / "checkpoints" / f"{name}.joblib"
    if checkpoint.exists():
        result = joblib.load(checkpoint)  # Only resume your own trusted run.
        state["results"][name] = result
        print(f"Reused completed checkpoint: {name}", flush=True)
        return result["validation"]
    if (output / "report.json").exists():
        raise ValueError("Completed run cannot be retrained. Use a new directory.")
    seed, jobs = config["seed"], config["jobs"]
    fit, stopping, training = state["fit"], state["stopping"], state["training"]
    validation = state["validation"]
    common = {"n_estimators": config["max_trees"], "learning_rate": 0.05,
                  "n_jobs": jobs, "random_state": seed}
    if name == "logistic_regression":
        pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
            C=1.0, max_iter=500, tol=1e-3, random_state=seed))])
    elif name == "xgboost":
        pipeline = Pipeline([("model", XGBClassifier(
            **common, tree_method="hist", device=config["device"], max_depth=4,
            max_bin=128, eval_metric="aucpr", early_stopping_rounds=config["patience"]))])
    else:
        pipeline = Pipeline([("model", LGBMClassifier(
            **common, num_leaves=15, max_bin=127, metric="average_precision",
            verbosity=-1, deterministic=True, force_col_wise=True))])
    print(f"Training {name} on {config['device'] if name == 'xgboost' else 'cpu'}...", flush=True)
    start = time.perf_counter()
    with threadpool_limits(limits=jobs):
        if name == "logistic_regression":
            pipeline.fit(training[FEATURES], training[TARGET])
            best_iteration = None
        else:
            options = {"eval_set": [(stopping[FEATURES], stopping[TARGET])]}
            if name == "xgboost":
                options["verbose"] = False
            else:
                options["callbacks"] = [early_stopping(config["patience"],
                                                       first_metric_only=True, verbose=False)]
            pipeline.named_steps["model"].fit(fit[FEATURES], fit[TARGET], **options)
            estimator = pipeline.named_steps["model"]
            best_iteration = (int(estimator.best_iteration) + 1 if name == "xgboost"
                              else int(estimator.best_iteration_))
    elapsed = time.perf_counter() - start
    # The same CPU inference configuration is evaluated, exported and used by the API.
    if name == "xgboost":
        pipeline.set_params(model__device="cpu")
    scores = pipeline.predict_proba(validation[FEATURES])[:, 1]
    threshold, met = choose_threshold(validation[TARGET], scores, config["min_precision"])
    row = {"model": name, **evaluate(validation[TARGET], scores, threshold),
           "precision_target_met": met, "training_seconds": elapsed,
           "best_iteration": best_iteration}
    result = {"pipeline": pipeline, "threshold": threshold, "validation": row}
    dump_model(checkpoint, result)
    state["results"][name] = result
    print(f"Saved {name}: {elapsed:.2f}s, validation AP={row['average_precision']:.4f}, "
          f"trees={best_iteration}", flush=True)
    return row


def finish_run(state):
    """Freeze validation selection, evaluate test once, export a CPU-ready bundle."""
    output, config = state["output"], state["config"]
    if (output / "report.json").exists():
        print("Run already complete; returning saved metrics without re-evaluating test.")
        return json.loads((output / "report.json").read_text(encoding="utf-8"))
    missing = set(config["models"]) - set(state["results"])
    if missing:
        raise ValueError(f"Train the remaining models before evaluation: {sorted(missing)}")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

    rows = [state["results"][name]["validation"] for name in config["models"]]
    selected = max(rows, key=lambda row: row["average_precision"])["model"]
    test = state["test"]
    test_rows = []
    fig, axis = plt.subplots(figsize=(8, 6))
    for name in config["models"]:
        result = state["results"][name]
        pipeline, threshold = result["pipeline"], result["threshold"]
        pipeline.predict_proba(test[FEATURES].iloc[:100])
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
    validation_table, test_table = pd.DataFrame(rows), pd.DataFrame(test_rows)
    validation_table.to_csv(output / "validation_metrics.csv", index=False)
    test_table.to_csv(output / "test_metrics.csv", index=False)
    test_table.merge(validation_table[["model", "training_seconds", "best_iteration"]],
                     on="model").to_csv(output / "comparison.csv", index=False)
    result = state["results"][selected]
    artifact = {"pipeline": result["pipeline"], "threshold": result["threshold"], "features": FEATURES,
                    "model_name": selected, "version": state["manifest"]["version"]}
    dump_model(output / "model.joblib", artifact)
    sample = test[FEATURES].iloc[:32]
    reloaded = joblib.load(output / "model.joblib")
    before = artifact["pipeline"].predict_proba(sample)[:, 1]
    after = reloaded["pipeline"].predict_proba(sample)[:, 1]
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(before >= artifact["threshold"], after >= artifact["threshold"])
    splits = {name: {"rows": len(state[key]), "frauds": int(state[key][TARGET].sum())}
              for name, key in [("train", "training"), ("validation", "validation"),
                                ("test", "test"), ("boost_fit", "fit"),
                                ("boost_stopping", "stopping")]}
    baseline = evaluate(test[TARGET], np.zeros(len(test)), 0.5)
    report = {
        "version": artifact["version"], "selected_model": selected, "mode": "fast",
        "selection_metric": "validation average_precision",
        "threshold_policy": "Maximum validation recall at minimum precision; fallback maximum F1.",
        "minimum_validation_precision": config["min_precision"],
        "dataset": {**state["summary"], "sha256": config["dataset_sha256"]},
        "splits": splits, "seed": config["seed"], "jobs": config["jobs"],
        "split_strategy": "stratified random 70/15/15 after exact-feature deduplication",
        "stopping_policy": "15% of training only; validation and test excluded from early stopping",
        "training_devices": {n: config["device"] if n == "xgboost" else "cpu"
                             for n in config["models"]},
        "export_device": "cpu", "configuration": config,
        "preparation_seconds": state["preparation_seconds"],
        "total_model_fit_seconds": sum(row["training_seconds"] for row in rows),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": config["packages"]},
        "validation": rows, "test": test_rows, "all_legitimate_baseline": baseline,
        "limitations": [
            "Single seeded random split of a historical dataset, not future payment performance.",
            "Only a small number of frauds are present in the test set; metrics have uncertainty.",
            "Scores are not calibrated fraud probabilities.",
            "Validation precision target is not a guarantee on test or future data.",
            "Upstream PCA fitting procedure is unknown; raw card fields are not supported.",
            "Batch timing is not HTTP latency or a production capacity claim.",
        ],
    }
    selected_metrics = next(row for row in test_rows if row["model"] == selected)
    (output / "resume_bullets.md").write_text(resume_text(report, selected_metrics), encoding="utf-8")
    (output / "requirements-model.txt").write_text(
        "\n".join(f"{p}=={v}" for p, v in config["packages"].items()) + "\n", encoding="utf-8")
    # Deliberately balanced examples for UI demonstration, never a performance sample.
    examples = test.groupby(TARGET, sort=True).head(4)
    write_json(output / "demo_samples.json", {
        "version": artifact["version"], "note": "Curated held-out examples, not representative traffic.",
        "examples": [{"label": int(row[TARGET]), "transaction": row[FEATURES].to_dict()}
                     for _, row in examples.iterrows()],
    })
    # Completion marker is written last so interrupted exports can be regenerated.
    write_json(output / "report.json", report)
    print(test_table.to_string(index=False))
    print(f"Selected by validation AP: {selected}. Reload verified. Results: {output.resolve()}")
    return report


def resume_text(report, metrics):
    test = report["splits"]["test"]
    return (
        "# Measured project results\n\n"
        f"Run: {report['version']}; dataset SHA-256: {report['dataset']['sha256']}\n\n"
        "Use these claims only for the dataset actually used in this run. "
        "Synthetic test runs are software checks, not resume evidence.\n\n"
        f"- Built a credit-card fraud detection pipeline comparing {len(report['test'])} models "
        f"on {report['dataset']['unique_rows']:,} deduplicated transactions; selected "
        f"{report['selected_model']} by validation average precision, achieving "
        f"{metrics['average_precision']:.3f} test AP, {metrics['precision']:.1%} fraud precision "
        f"and {metrics['recall']:.1%} fraud recall on a {test['rows']:,}-transaction held-out set "
        f"({test['frauds']} frauds).\n"
        "- Integrated the trained model with FastAPI batch inference and a local dashboard; "
        "added schema validation, resumable training checkpoints and reproducible evaluation reports.\n\n"
        f"Supporting metrics: accuracy {metrics['accuracy']:.4%}, F1 {metrics['f1']:.4f}, "
        f"ROC-AUC {metrics['roc_auc']:.4f}; TP={metrics['true_positives']}, "
        f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}, "
        f"TN={metrics['true_negatives']}.\n\n"
        f"Predicting every transaction legitimate already gives "
        f"{report['all_legitimate_baseline']['accuracy']:.4%} accuracy and 0% fraud recall. "
        "Do not present accuracy alone. These are single-split historical results, "
        "not production results or a measured training speedup.\n"
    )


def train_fast(data, output, **options):
    state = prepare_run(data, output, **options)
    for name in state["config"]["models"]:
        fit_model(state, name)
    return finish_run(state)
