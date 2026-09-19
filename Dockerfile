FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUD_MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir /app/artifacts \
    && chown app:app /app/artifacts

COPY . .
RUN pip install .

USER app
EXPOSE 8000
CMD ["uvicorn", "fraud_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]
