import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection import api
from fraud_detection.data import FEATURES


@pytest.fixture
def row():
    return {name: 0.0 for name in FEATURES}


@pytest.fixture
def service(monkeypatch):
    artifact = {"pipeline": object(), "threshold": 0.4, "model_name": "test-model",
                "version": "test-v1", "features": FEATURES}
    loads = []
    frames = []

    def load(path):
        loads.append(path)
        return artifact

    def score(model, frame):
        assert model is artifact
        frames.append(frame)
        return pd.DataFrame({"fraud_score": [0.75] * len(frame), "is_fraud": [True] * len(frame)})

    monkeypatch.setattr(api, "load_artifact", load)
    monkeypatch.setattr(api, "score_frame", score)
    monkeypatch.setenv("FRAUD_MODEL_PATH", "custom/model.joblib")
    with TestClient(api.app) as client:
        yield client, loads, frames


def test_health_and_predict_load_once(service, row):
    client, loads, frames = service
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-v1"}
    for _ in range(2):
        response = client.post("/predict", json={"transactions": [row, row]})
        assert response.status_code == 200
        assert response.json() == {
            "model_name": "test-model", "model_version": "test-v1", "threshold": 0.4,
            "predictions": [{"fraud_score": 0.75, "is_fraud": True}] * 2,
        }
    assert loads == ["custom/model.joblib"]
    assert list(frames[0].columns) == FEATURES
    assert frames[0].shape == (2, len(FEATURES))


@pytest.mark.parametrize("kind", ["missing", "extra", "negative_time", "negative_amount", "string", "bool", "null"])
def test_rejects_invalid_features(service, row, kind):
    client, _, frames = service
    if kind == "missing":
        del row["V1"]
    elif kind == "extra":
        row["private_extra_928374"] = 928374
    else:
        name, value = {
            "negative_time": ("Time", -928374), "negative_amount": ("Amount", -928374),
            "string": ("V1", "private_value_928374"), "bool": ("V1", True),
            "null": ("V1", None),
        }[kind]
        row[name] = value
    response = client.post("/predict", json={"transactions": [row]})
    assert response.status_code == 422
    assert "928374" not in response.text
    assert not frames


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10 ** 400])
def test_rejects_nonfinite(service, row, value):
    client, _, frames = service
    row["V1"] = value
    response = client.post("/predict", content=json.dumps({"transactions": [row]}),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert not frames


@pytest.mark.parametrize("count, status", [(0, 422), (1000, 200), (1001, 422)])
def test_batch_limits(service, row, count, status):
    client, _, _ = service
    assert client.post("/predict", json={"transactions": [row] * count}).status_code == status


@pytest.mark.parametrize("body", [{}, {"transactions": None}, {"transactions": [1]},
                                  {"transactions": [], "private_928374": "secret_928374"}])
def test_invalid_envelope(service, body):
    client, _, _ = service
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "928374" not in response.text


def test_malformed_json_is_redacted(service):
    client, _, _ = service
    response = client.post("/predict", content='{"secret_928374":',
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "928374" not in response.text


def test_missing_artifact_fails_startup(monkeypatch):
    monkeypatch.delenv("FRAUD_MODEL_PATH", raising=False)

    def missing(path):
        assert path == "artifacts/model.joblib"
        raise FileNotFoundError(path)

    monkeypatch.setattr(api, "load_artifact", missing)
    with pytest.raises(RuntimeError, match="FRAUD_MODEL_PATH"), TestClient(api.app):
        pass
