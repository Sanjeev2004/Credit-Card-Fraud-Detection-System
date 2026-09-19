import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from fraud_detection.data import FEATURES, TARGET


@pytest.fixture
def dataset():
    # Synthetic data tests software behavior, never benchmark quality.
    values, labels = make_classification(
        n_samples=400, n_features=30, n_informative=8, weights=[0.8, 0.2], random_state=12
    )
    frame = pd.DataFrame(values, columns=FEATURES)
    frame["Time"] = np.arange(len(frame), dtype=float)
    frame["Amount"] = frame["Amount"].abs() * 100
    frame[TARGET] = labels
    return frame
