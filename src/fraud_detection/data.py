"""Schema validation and duplicate-safe benchmark splits."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 0:
        raise ValueError("At least one transaction is required.")
    if set(frame.columns) != set(FEATURES) or len(frame.columns) != len(FEATURES):
        raise ValueError("Expected exactly Time, V1 through V28, and Amount.")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) or
           pd.api.types.is_bool_dtype(frame[c]) for c in FEATURES):
        raise ValueError("All features must be numeric, not boolean.")
    values = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features must contain only finite, non-missing values.")
    if (frame[["Time", "Amount"]] < 0).any().any():
        raise ValueError("Time and Amount must be nonnegative.")
    return frame.loc[:, FEATURES]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    if set(frame.columns) != {*FEATURES, TARGET}:
        raise ValueError("Training CSV must contain the 30 features and Class only.")
    validate_features(frame.drop(columns=TARGET))
    if not frame[TARGET].isin([0, 1]).all() or frame[TARGET].nunique() != 2:
        raise ValueError("Class must contain both 0 (legitimate) and 1 (fraud).")
    original_rows = len(frame)
    # Identical inputs must not cross splits, including conflicting labels.
    duplicates = frame[frame.duplicated(subset=FEATURES, keep=False)]
    if not duplicates.empty and duplicates.groupby(FEATURES, dropna=False)[TARGET].nunique().gt(1).any():
        raise ValueError("Identical feature rows have conflicting Class labels.")
    frame = frame.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    frame[TARGET] = frame[TARGET].astype(int)
    if frame[TARGET].value_counts().min() < 20:
        raise ValueError("Need at least 20 unique examples of each class for reliable splits.")
    return frame, {
        "input_rows": original_rows,
        "duplicate_rows_removed": original_rows - len(frame),
        "unique_rows": len(frame),
        "fraud_count": int(frame[TARGET].sum()),
        "fraud_prevalence": float(frame[TARGET].mean()),
    }


def split_dataset(frame: pd.DataFrame, seed: int = 42):
    train, remainder = train_test_split(
        frame, test_size=0.30, random_state=seed, stratify=frame[TARGET]
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=seed, stratify=remainder[TARGET]
    )
    return train, validation, test
