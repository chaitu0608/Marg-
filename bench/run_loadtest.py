from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from marg.fraud.data import FEATURES, LABEL
from marg.fraud.explain import Explainer

ROOT = Path(__file__).resolve().parents[1]


async def exercise(url: str, features: list[float], requests: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    inference_latencies: list[float] = []
    async with httpx.AsyncClient(timeout=30) as client:
        async def one() -> None:
            async with sem:
                begin = time.perf_counter()
                response = await client.post(url, json={"features": features})
                response.raise_for_status()
                payload = response.json()
                if not 0.0 <= float(payload["score"]) <= 1.0:
                    raise AssertionError("/score returned a value outside [0, 1]")
                if payload["decision"] not in {"allow", "review", "block"}:
                    raise AssertionError("/score returned an unsupported decision")
                latencies.append((time.perf_counter() - begin) * 1_000)
                inference_latencies.append(float(payload["latency_ms"]))

        for _ in range(100):
            await one()
        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(requests)))
        elapsed = time.perf_counter() - start
    values = np.asarray(latencies[100:])
    model_values = np.asarray(inference_latencies[100:])
    return {"requests": requests, "concurrency": concurrency, "elapsed_seconds": elapsed,
            "throughput_rps": requests / elapsed, "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
            "inference_p50_ms": float(np.percentile(model_values, 50)),
            "inference_p95_ms": float(np.percentile(model_values, 95)),
            "inference_p99_ms": float(np.percentile(model_values, 99)),
            "target_p99_ms": 100.0, "target_met": bool(np.percentile(values, 99) < 100)}


def main() -> None:
    data_path = ROOT / "data" / "creditcard.csv"
    frame = pd.read_csv(data_path)
    features = frame.iloc[0][FEATURES].astype(float).tolist()
    fraud_row = frame.loc[frame[LABEL] == 1].iloc[0][FEATURES].astype(float).to_numpy()
    command = [sys.executable, "-m", "uvicorn", "marg.fraud.serve:app", "--host", "127.0.0.1",
               "--port", "8877", "--log-level", "error"]
    server = subprocess.Popen(command, cwd=ROOT)
    try:
        async def wait_ready() -> None:
            async with httpx.AsyncClient() as client:
                for _ in range(100):
                    try:
                        response = await client.get("http://127.0.0.1:8877/health")
                        if response.status_code == 200:
                            return
                    except httpx.HTTPError:
                        await asyncio.sleep(0.2)
                raise RuntimeError("Scoring service did not become ready")

        asyncio.run(wait_ready())
        async def explain_example() -> dict:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post("http://127.0.0.1:8877/explain",
                                             json={"features": fraud_row.tolist()})
                response.raise_for_status()
                served = response.json()["top_features"]
            expected = Explainer().top_features(fraud_row, 5)
            if [x["feature"] for x in served] != [x["feature"] for x in expected]:
                raise AssertionError("/explain top factors did not match direct SHAP values")
            if any(x["feature"] not in FEATURES for x in served):
                raise AssertionError("/explain returned an unknown feature name")
            return {"example_label": 1, "top_features": served}

        explanation_check = asyncio.run(explain_example())
        metrics = asyncio.run(exercise("http://127.0.0.1:8877/score", features, 10_000, 32))
    finally:
        server.terminate()
        server.wait(timeout=10)
    metrics["explain_known_fraud_example"] = explanation_check
    if metrics["inference_p99_ms"] >= metrics["p99_ms"] * 0.5:
        metrics["bottleneck"] = "Model inference and thread contention under concurrency dominate the tail."
    else:
        metrics["bottleneck"] = "HTTP round-trip, event-loop scheduling, and JSON overhead dominate the tail."
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "loadtest.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
