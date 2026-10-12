from __future__ import annotations

import argparse
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
    latencies: list[float] = []
    inference_latencies: list[float] = []
    app_latencies: list[float] = []
    server_queue_latencies: list[float] = []
    request_body = json.dumps({"features": features}, separators=(",", ":")).encode()
    headers = {"content-type": "application/json"}
    limits = httpx.Limits(max_connections=concurrency,
                          max_keepalive_connections=concurrency, keepalive_expiry=60.0)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        async def one() -> None:
            begin = time.perf_counter()
            response = await client.post(url, content=request_body, headers=headers)
            response.raise_for_status()
            payload = response.json()
            if not 0.0 <= float(payload["score"]) <= 1.0:
                raise AssertionError("/score returned a value outside [0, 1]")
            if payload["decision"] not in {"allow", "review", "block"}:
                raise AssertionError("/score returned an unsupported decision")
            latencies.append((time.perf_counter() - begin) * 1_000)
            inference_latencies.append(float(payload["inference_ms"]))
            app_latencies.append(float(response.headers["x-marg-app-ms"]))
            server_queue_latencies.append(float(response.headers["x-marg-server-queue-ms"]))

        for _ in range(100):
            await one()
        start = time.perf_counter()
        remaining = iter(range(requests))

        async def worker() -> None:
            for _ in remaining:
                await one()

        # Keep exactly `concurrency` request workers; avoid timing thousands of queued tasks.
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
    values = np.asarray(latencies[100:])
    model_values = np.asarray(inference_latencies[100:])
    app_values = np.asarray(app_latencies[100:])
    queue_values = np.asarray(server_queue_latencies[100:])
    outside_instrumented_values = np.maximum(values - app_values - queue_values, 0.0)
    non_inference = np.maximum(values - model_values, 0.0)
    return {"requests": requests, "concurrency": concurrency, "elapsed_seconds": elapsed,
            "throughput_rps": requests / elapsed, "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
            "inference_p50_ms": float(np.percentile(model_values, 50)),
            "inference_p95_ms": float(np.percentile(model_values, 95)),
            "inference_p99_ms": float(np.percentile(model_values, 99)),
            "app_processing_p50_ms": float(np.percentile(app_values, 50)),
            "app_processing_p95_ms": float(np.percentile(app_values, 95)),
            "app_processing_p99_ms": float(np.percentile(app_values, 99)),
            "server_dispatch_queue_p50_ms": float(np.percentile(queue_values, 50)),
            "server_dispatch_queue_p95_ms": float(np.percentile(queue_values, 95)),
            "server_dispatch_queue_p99_ms": float(np.percentile(queue_values, 99)),
            "client_loopback_and_response_p99_ms": float(
                np.percentile(outside_instrumented_values, 99)),
            "non_inference_overhead_p99_ms": float(np.percentile(non_inference, 99)),
            "measurement": "fixed-worker HTTP client with pre-encoded input; app timing spans ASGI dispatch through response serialization; server queue is Uvicorn header-parse-to-ASGI scheduling delay; residual includes client scheduling, loopback/socket transport, and response decoding",
            "target_p99_ms": 100.0, "target_met": bool(np.percentile(values, 99) < 100)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-workers", type=int, default=4)
    parser.add_argument("--url", help="Use a separately hosted instrumented Marg server.")
    args = parser.parse_args()
    data_path = ROOT / "data" / "creditcard.csv"
    frame = pd.read_csv(data_path)
    features = frame.iloc[0][FEATURES].astype(float).tolist()
    fraud_row = frame.loc[frame[LABEL] == 1].iloc[0][FEATURES].astype(float).to_numpy()
    command = [sys.executable, "-m", "marg.fraud.server", "--host", "127.0.0.1",
               "--port", "8877", "--workers", str(args.server_workers)]
    server = None if args.url else subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = args.url.rstrip("/") if args.url else "http://127.0.0.1:8877"
    try:
        async def wait_ready() -> None:
            async with httpx.AsyncClient() as client:
                for _ in range(100):
                    try:
                        response = await client.get(f"{base_url}/health")
                        if response.status_code == 200:
                            return
                    except httpx.HTTPError:
                        await asyncio.sleep(0.2)
                raise RuntimeError("Scoring service did not become ready")

        asyncio.run(wait_ready())
        async def explain_example() -> dict:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(f"{base_url}/explain",
                                             json={"features": fraud_row.tolist()})
                response.raise_for_status()
                served = response.json()["top_features"]
            expected = Explainer().top_features(fraud_row, 5)
            if [x["feature"] for x in served] != [x["feature"] for x in expected]:
                raise AssertionError("/explain top factors did not match direct SHAP values")
            if any(x["feature"] not in FEATURES for x in served):
                raise AssertionError("/explain returned an unknown feature name")
            return {"example_label": 1, "top_features": served}

        explanation_check = asyncio.run(explain_example()) if server else None
        metrics = asyncio.run(exercise(f"{base_url}/score", features, 10_000, 32))
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=10)
    metrics["explain_known_fraud_example"] = explanation_check
    metrics["server_workers"] = args.server_workers if server is not None else None
    metrics["load_generator"] = (
        "benchmark process host (local loopback server)" if server is not None
        else "benchmark process host (external server URL)")
    if metrics["p99_ms"] >= metrics["target_p99_ms"] and (
            metrics["client_loopback_and_response_p99_ms"]
            > metrics["app_processing_p99_ms"] * 10):
        metrics["bottleneck"] = (
            "The client-observed tail is dominated by client/loopback/socket/response time; "
            f"ASGI processing is {metrics['app_processing_p99_ms']:.3f} ms p99 and measured "
            f"Uvicorn dispatch queue is {metrics['server_dispatch_queue_p99_ms']:.3f} ms p99."
        )
    elif metrics["inference_p99_ms"] >= metrics["p99_ms"] * 0.5:
        metrics["bottleneck"] = "Model inference and thread contention under concurrency dominate the tail."
    else:
        metrics["bottleneck"] = "HTTP round-trip, event-loop scheduling, and JSON overhead dominate the tail."
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "loadtest.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
