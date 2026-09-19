"""Bounded-concurrency HTTP load test using a small sample from a real CSV."""

import argparse
import asyncio
import csv
import json
import math
import time
from pathlib import Path

import httpx

from fraud_detection.data import FEATURES


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_sample(path: Path, batch_size: int) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(FEATURES).issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing required model feature columns")
        for _ in range(batch_size):
            row = next(reader, None)
            if row is None:
                raise ValueError("CSV has fewer rows than --batch-size")
            try:
                transaction = {name: float(row[name]) for name in FEATURES}
            except (ValueError, TypeError) as exc:
                raise ValueError("CSV sample contains invalid numeric features") from exc
            if not all(math.isfinite(value) for value in transaction.values()):
                raise ValueError("CSV sample contains nonfinite features")
            if transaction["Time"] < 0 or transaction["Amount"] < 0:
                raise ValueError("CSV sample contains negative Time or Amount")
            rows.append(transaction)
    return rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def run(args: argparse.Namespace, rows: list[dict[str, float]]) -> dict:
    payload = {"transactions": rows}
    latencies = []
    successful_latencies = []
    failures: dict[str, int] = {}
    work = iter(range(args.requests))
    concurrency = min(args.concurrency, args.requests)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        async def worker():
            for _ in work:
                started = time.perf_counter()
                error = None
                try:
                    # An overall deadline also bounds responses that trickle bytes indefinitely.
                    async with asyncio.timeout(args.timeout):
                        response = await client.post(
                            args.url.rstrip("/") + "/predict", json=payload
                        )
                    if response.status_code != 200:
                        error = f"http_{response.status_code}"
                    else:
                        result = response.json()
                        predictions = result.get("predictions") if isinstance(result, dict) else None
                        if not isinstance(predictions, list) or len(predictions) != len(rows) or any(
                            not isinstance(item, dict)
                            or type(item.get("is_fraud")) is not bool
                            or type(item.get("fraud_score")) not in (int, float)
                            or not math.isfinite(item["fraud_score"])
                            or not 0 <= item["fraud_score"] <= 1
                            for item in predictions
                        ):
                            error = "invalid_response"
                except (httpx.RequestError, TimeoutError) as exc:
                    error = type(exc).__name__
                except (ValueError, OverflowError):
                    error = "invalid_response"
                elapsed = (time.perf_counter() - started) * 1000
                latencies.append(elapsed)
                if error is None:
                    successful_latencies.append(elapsed)
                else:
                    failures[error] = failures.get(error, 0) + 1

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - started

    succeeded = len(successful_latencies)
    failed = len(latencies) - succeeded
    return {
        "elapsed_seconds": elapsed,
        "batch_size": len(rows),
        "concurrency": concurrency,
        "requests": {"attempted": len(latencies), "succeeded": succeeded, "failed": failed},
        "transactions": {
            "attempted": len(latencies) * len(rows),
            "succeeded": succeeded * len(rows),
            "failed": failed * len(rows),
        },
        "successful_requests_per_second": succeeded / elapsed,
        "successful_transactions_per_second": succeeded * len(rows) / elapsed,
        "attempted_requests_per_second": len(latencies) / elapsed,
        "attempted_transactions_per_second": len(latencies) * len(rows) / elapsed,
        "latency_ms_all_requests": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95)},
        "latency_ms_successful_requests": {
            "p50": percentile(successful_latencies, .5),
            "p95": percentile(successful_latencies, .95),
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--requests", type=positive_int, default=1000)
    parser.add_argument("--concurrency", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request deadline in seconds")
    args = parser.parse_args()
    if args.batch_size > 1000:
        parser.error("--batch-size cannot exceed 1000")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if httpx.URL(args.url).scheme not in ("http", "https"):
        parser.error("--url must use http or https")
    try:
        rows = load_sample(args.data, args.batch_size)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Reusing the first {len(rows)} CSV rows for every request; no warmup.", flush=True)
    report = asyncio.run(run(args, rows))
    print(json.dumps(report, indent=2))
    return 1 if report["requests"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
