import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.data import FEATURES, TARGET, load_dataset, split_dataset, validate_features
from fraud_detection.prediction import load_artifact, predict_csv, score_frame
from fraud_detection.training import choose_threshold, evaluate, train


def test_deduplication_and_disjoint_reproducible_splits(dataset, tmp_path):
    path = tmp_path / "data.csv"
    pd.concat([dataset, dataset.iloc[:10]]).to_csv(path, index=False)
    cleaned, summary = load_dataset(path)
    assert summary["duplicate_rows_removed"] == 10
    parts = split_dataset(cleaned)
    assert [len(part) for part in parts] == [280, 60, 60]
    for i, part in enumerate(parts):
        assert part[TARGET].nunique() == 2
        pd.testing.assert_frame_equal(part, split_dataset(cleaned)[i])
        for other in parts[i + 1:]:
            assert set(part.index).isdisjoint(other.index)


def test_conflicting_labels_rejected(dataset, tmp_path):
    duplicate = dataset.iloc[:1].copy()
    duplicate[TARGET] = 1 - duplicate[TARGET]
    path = tmp_path / "conflicting.csv"
    pd.concat([dataset, duplicate]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        load_dataset(path)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "invalid", True])
def test_invalid_features(dataset, bad_value):
    frame = dataset[FEATURES].copy()
    frame["V1"] = bad_value
    with pytest.raises(ValueError):
        validate_features(frame)


def test_schema_order_and_negatives(dataset):
    frame = dataset[FEATURES]
    assert list(validate_features(frame[FEATURES[::-1]]).columns) == FEATURES
    with pytest.raises(ValueError):
        validate_features(frame.drop(columns="V1"))
    with pytest.raises(ValueError):
        validate_features(frame.assign(Amount=-1))
    with pytest.raises(ValueError):
        validate_features(frame.iloc[:0])


def test_threshold_selection_and_unreachable_target():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.6, 0.9])
    threshold, met = choose_threshold(labels, scores, 0.9)
    assert met and threshold == pytest.approx(0.8)
    assert evaluate(labels, scores, threshold)["recall"] == 1
    threshold, met = choose_threshold([1, 0], [0.1, 0.9], 1.0)
    assert not met and threshold == pytest.approx(0.1)


def test_four_model_training_and_inference(dataset, tmp_path, monkeypatch):
    source, output = tmp_path / "synthetic.csv", tmp_path / "models"
    dataset.to_csv(source, index=False)
    report = train(source, output, jobs=1, quick=True)
    assert {row["model"] for row in report["test"]} == {
        "logistic_regression", "random_forest", "xgboost", "lightgbm"
    }
    expected = max(report["validation"], key=lambda row: row["average_precision"])["model"]
    assert report["selected_model"] == expected
    assert all(len(search["candidates"]) == 4 for search in report["search"])
    assert len(list(output.glob("confusion_*.png"))) == 4
    assert json.loads((output / "report.json").read_text())["selected_model"] == expected
    model_path = output / "model.joblib"
    artifact = load_artifact(model_path)
    direct = score_frame(artifact, dataset[FEATURES])
    predicted = tmp_path / "predictions.csv"
    assert predict_csv(model_path, source, predicted, chunk_size=37) == len(dataset)
    batch = pd.read_csv(predicted)
    np.testing.assert_allclose(direct["fraud_score"], batch["fraud_score"])
    np.testing.assert_array_equal(direct["is_fraud"], batch["is_fraud"])
    assert list(batch["row_number"]) == list(range(len(dataset)))
    from fraud_detection.api import app
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(model_path))
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "transactions": dataset[FEATURES].iloc[:2].to_dict(orient="records")
        })
        assert response.status_code == 200
        np.testing.assert_allclose(
            [row["fraud_score"] for row in response.json()["predictions"]],
            direct["fraud_score"].iloc[:2],
        )
    with pytest.raises(ValueError, match="empty"):
        train(source, output, quick=True)
    with pytest.raises(ValueError, match="overwrite"):
        predict_csv(model_path, source, source)
    # A late invalid chunk must not replace an existing complete result.
    original = predicted.read_bytes()
    dataset.loc[len(dataset) - 1, "Amount"] = -1
    dataset.to_csv(source, index=False)
    with pytest.raises(ValueError, match="nonnegative"):
        predict_csv(model_path, source, predicted, chunk_size=37)
    assert predicted.read_bytes() == original
    assert not list(tmp_path.glob("*.csv.tmp"))
