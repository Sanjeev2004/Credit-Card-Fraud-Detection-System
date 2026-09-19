"""Trusted artifact loading and bounded-memory batch prediction."""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.data import FEATURES, TARGET, validate_features


def load_artifact(path: str | Path) -> dict:
    # Joblib uses pickle: only load artifacts produced by a trusted training process.
    artifact = joblib.load(path)
    required = {"pipeline", "threshold", "features", "model_name", "version"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Invalid model artifact.")
    if artifact["features"] != FEATURES or not 0 <= artifact["threshold"] <= 1:
        raise ValueError("Incompatible model schema or decision threshold.")
    return artifact


def score_frame(artifact: dict, frame: pd.DataFrame) -> pd.DataFrame:
    features = validate_features(frame)
    scores = artifact["pipeline"].predict_proba(features)[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model returned invalid probability scores.")
    return pd.DataFrame({
        "fraud_score": scores,
        "is_fraud": scores >= artifact["threshold"],
    }, index=frame.index)


def predict_csv(model: str | Path, source: str | Path, output: str | Path,
                chunk_size: int = 10_000) -> int:
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive.")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output in (source, Path(model).resolve()):
        raise ValueError("Prediction output must not overwrite input or model.")
    artifact = load_artifact(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, suffix=".csv.tmp")
    os.close(descriptor)
    count = 0
    try:
        for chunk in pd.read_csv(source, chunksize=chunk_size):
            if len(chunk) == 0:
                continue
            result = score_frame(artifact, chunk.drop(columns=TARGET, errors="ignore"))
            result.insert(0, "row_number", np.arange(count, count + len(chunk)))
            result.to_csv(temporary, mode="a", header=count == 0, index=False)
            count += len(chunk)
        if not count:
            raise ValueError("Input CSV has no transactions.")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return count
