"""HTTP inference service. Load the trusted model artifact once at startup."""

import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_detection.data import FEATURES
from fraud_detection.prediction import load_artifact, score_frame


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("FRAUD_MODEL_PATH", "artifacts/model.joblib")
    try:
        app.state.artifact = load_artifact(path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Fraud model artifact is missing. Set FRAUD_MODEL_PATH to an existing "
            "trusted artifact (default: artifacts/model.joblib)."
        ) from exc
    app.state.report = None
    app.state.examples = {"examples": []}
    for filename, attribute in [("report.json", "report"), ("demo_samples.json", "examples")]:
        candidate = Path(path).with_name(filename)
        if candidate.exists():
            content = json.loads(candidate.read_text(encoding="utf-8"))
            if content.get("version") == app.state.artifact["version"]:
                setattr(app.state, attribute, content)
    yield
    del app.state.artifact


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


@app.get("/metrics")
def metrics(request: Request):
    report = request.app.state.report
    if report is None:
        return JSONResponse(status_code=404, content={"detail": "No matching evaluation report."})
    return report


@app.get("/examples")
def examples(request: Request):
    return request.app.state.examples


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[dict[str, float]] = Field(min_length=1, max_length=1000)

    @field_validator("transactions", mode="before")
    @classmethod
    def validate_transactions(cls, rows):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
            raise ValueError("Expected between 1 and 1000 transactions")
        expected = set(FEATURES)
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError("Each transaction must contain exactly the model features")
            for name, value in row.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    # Pydantic converts ValueError, but not TypeError, into a 422 response.
                    raise ValueError("Features must be finite numbers")  # noqa: TRY004
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError("Features must be finite numbers")
                if name in ("Time", "Amount") and value < 0:
                    raise ValueError("Time and Amount must be nonnegative")
        return rows


class Prediction(BaseModel):
    fraud_score: float
    is_fraud: bool


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    predictions: list[Prediction]


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Pydantic's default errors include the input, which can contain financial data.
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request: provide 1-1000 transactions with exactly "
                 "the model features, finite numeric values, and nonnegative Time and Amount."},
    )


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.artifact["version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest, request: Request):
    artifact = request.app.state.artifact
    frame = pd.DataFrame(body.transactions, columns=FEATURES)
    scores = score_frame(artifact, frame)
    return {
        "model_name": artifact["model_name"],
        "model_version": artifact["version"],
        "threshold": artifact["threshold"],
        "predictions": scores[["fraud_score", "is_fraud"]].to_dict(orient="records"),
    }
