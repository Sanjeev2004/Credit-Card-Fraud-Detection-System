"""Execute the standalone notebook in a fresh CPU process without local imports."""

import json
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from fraud_detection.data import FEATURES
from fraud_detection.prediction import load_artifact, score_frame


def test_colab_notebook_cpu_workflow(dataset, tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/train_fraud_detection.ipynb").read_text("utf-8"))
    nbformat.validate(nbformat.from_dict(notebook))
    source_path = tmp_path / "synthetic.csv"
    dataset.to_csv(source_path, index=False)
    output = tmp_path / "run"
    namespace = {
        "DATA_PATH": str(source_path), "DATA_SOURCE": "path", "OUTPUT_DIR": str(output),
        "DEVICE": "cpu", "MODELS": ("logistic_regression", "xgboost", "lightgbm"),
        "SEED": 42, "JOBS": 1, "MAX_TREES": 12, "PATIENCE": 3,
        "MIN_PRECISION": 0.8, "RESUME": True,
    }
    program = f"from pathlib import Path\nsettings = {namespace!r}\nglobals().update(settings)\n"
    program += "DATA_PATH = Path(DATA_PATH)\nOUTPUT_DIR = Path(OUTPUT_DIR)\n"
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if cell["metadata"]["tags"][0] in {"install", "config"}:
            continue
        compile(source, "<notebook>", "exec")
        program += source + "\n"
    program += f"\nroot = Path({str(root)!r})\n"
    program += "for name, source in PROJECT_SOURCES.items():\n"
    program += "    assert source == (root / 'src/fraud_detection' / name).read_text('utf-8')\n"
    result = subprocess.run([sys.executable, "-"], input=program, text=True,
                            capture_output=True, cwd=tmp_path, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert len(report["test"]) == 3
    assert report["training_devices"]["xgboost"] == "cpu"
    assert output.with_suffix(".zip").is_file()
    assert len(list(output.glob("confusion_*.png"))) == 3
    artifact = load_artifact(output / "model.joblib")
    prediction = score_frame(artifact, dataset[FEATURES])
    assert len(prediction) == len(dataset)
    if artifact["model_name"] == "xgboost":
        assert artifact["pipeline"].get_params()["model__device"] == "cpu"
    table = pd.read_csv(output / "test_metrics.csv")
    np.testing.assert_allclose(table.accuracy, [row["accuracy"] for row in report["test"]])
