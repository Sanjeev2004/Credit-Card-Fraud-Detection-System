import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import app
from fraud_detection.data import FEATURES
from fraud_detection.fast_training import (
    FAST_MODELS,
    check_device,
    finish_run,
    fit_model,
    prepare_run,
)
from fraud_detection.prediction import load_artifact, score_frame


def test_fast_pipeline_resume_leakage_and_serving(dataset, tmp_path, monkeypatch):
    data, output = tmp_path / "data.csv", tmp_path / "results"
    dataset.to_csv(data, index=False)
    settings = {"jobs": 1, "max_trees": 20, "patience": 3}
    state = prepare_run(data, output, **settings)
    parts = [state[key] for key in ("fit", "stopping", "validation", "test")]
    assert sum(map(len, parts)) == len(dataset)
    for i, part in enumerate(parts):
        for other in parts[i+1:]:
            assert set(part.index).isdisjoint(other.index)
    with pytest.raises(ValueError, match="remaining models"):
        finish_run(state)
    original = fit_model(state, "logistic_regression")
    resumed = prepare_run(data, output, resume=True, **settings)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **kw: pytest.fail("Refit checkpoint"))
    assert fit_model(resumed, "logistic_regression") == original
    for name in FAST_MODELS[1:]:
        fit_model(resumed, name)
    report = finish_run(resumed)
    assert report["selected_model"] == max(
        report["validation"], key=lambda row: row["average_precision"])["model"]
    assert report["all_legitimate_baseline"]["recall"] == 0
    assert report["total_model_fit_seconds"] > 0
    assert finish_run(resumed) == report
    artifact = load_artifact(output / "model.joblib")
    direct = score_frame(artifact, dataset[FEATURES].iloc[:3])
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(output / "model.joblib"))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert "FraudLens" in client.get("/").text
        assert client.get("/metrics").json() == report
        assert len(client.get("/examples").json()["examples"]) == 8
        response = client.post("/predict", json={
            "transactions": dataset[FEATURES].iloc[:3].to_dict("records")})
        assert response.status_code == 200
        np.testing.assert_allclose([p["fraud_score"] for p in response.json()["predictions"]],
                                   direct.fraud_score)
    with pytest.raises(ValueError, match="changed"):
        prepare_run(data, output, resume=True, seed=123, **settings)
    report["version"] = "different-run"
    (output / "report.json").write_text(json.dumps(report))
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404


@pytest.mark.parametrize("options", [
    {"models": []}, {"models": ["xgboost", "xgboost"]}, {"device": "invalid"},
    {"max_trees": 0}, {"patience": 0}, {"jobs": 0}, {"min_precision": 0},
])
def test_invalid_options_rejected_before_training(tmp_path, options):
    with pytest.raises(ValueError):
        prepare_run(tmp_path / "missing.csv", tmp_path / "run", **options)


@pytest.mark.parametrize("actual, expected", [("cpu", "cpu"), ("cuda:0", "cuda")])
def test_auto_device_checks_actual_xgboost_device(monkeypatch, actual, expected):
    from xgboost import XGBClassifier
    from xgboost.core import Booster

    monkeypatch.setattr(XGBClassifier, "fit", lambda *a, **kw: None)
    monkeypatch.setattr(XGBClassifier, "get_booster", lambda self: Booster())
    monkeypatch.setattr(Booster, "save_config", lambda self: json.dumps(
        {"learner": {"generic_param": {"device": actual}}}))
    assert check_device("auto") == expected
    if actual == "cpu":
        with pytest.raises(RuntimeError, match="fell back to CPU"):
            check_device("cuda")


def test_auto_device_handles_unavailable_cuda_build(monkeypatch):
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError

    def unavailable(*args, **kwargs):
        raise XGBoostError("Not compiled with GPU support")

    monkeypatch.setattr(XGBClassifier, "fit", unavailable)
    assert check_device("auto") == "cpu"
    assert check_device("cpu") == "cpu"
    with pytest.raises(XGBoostError):
        check_device("cuda")


def test_auto_device_records_resolved_device(dataset, tmp_path, monkeypatch):
    data = tmp_path / "data.csv"
    dataset.to_csv(data, index=False)
    monkeypatch.setattr("fraud_detection.fast_training.check_device", lambda *a: "cpu")
    state = prepare_run(data, tmp_path / "run", device="auto")
    assert state["config"]["device"] == "cpu"
    resumed = prepare_run(data, tmp_path / "run", device="cpu", resume=True)
    assert resumed["config"] == state["config"]
