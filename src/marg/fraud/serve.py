from __future__ import annotations

import json
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat, field_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from marg.demo.runtime import Decision, DemoRuntime, IdempotencyConflict
from marg.fraud.explain import Explainer
from marg.sim.constants import BANKS, GATEWAYS, METHODS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_ROOT = PROJECT_ROOT / "web"
RESULT_FILES = {
    "routing": "routing.json",
    "fraud": "fraud_metrics.json",
    "latency": "loadtest.json",
    "outage_detector": "outage_detection.json",
    "outage_tradeoff": "outage_tradeoff.json",
    "fraud_sensitivity": "fraud_sensitivity.json",
    "pipeline": "pipeline.json",
}


class ScoreRequest(BaseModel):
    features: list[FiniteFloat] = Field(min_length=30, max_length=30)


class ScoreResponse(BaseModel):
    score: float
    decision: Literal["allow", "review", "block"]
    model_version: str
    inference_ms: float


class DemoTransactionRequest(ScoreRequest):
    bank: str
    method: str
    amount: FiniteFloat = Field(gt=0)
    outage_gateway: str | None = None

    @field_validator("bank")
    @classmethod
    def validate_bank(cls, value: str) -> str:
        if value not in BANKS:
            raise ValueError(f"bank must be one of {', '.join(BANKS)}")
        return value

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        if value not in METHODS:
            raise ValueError(f"method must be one of {', '.join(METHODS)}")
        return value

    @field_validator("outage_gateway")
    @classmethod
    def validate_outage_gateway(cls, value: str | None) -> str | None:
        if value is not None and value not in GATEWAYS:
            raise ValueError(f"outage_gateway must be one of {', '.join(GATEWAYS)}")
        return value


class AppTimingMiddleware:
    """Measure request parsing through response serialization, excluding socket delivery."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/score":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()

        async def timed_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                elapsed = (time.perf_counter() - started) * 1000
                queue_ms = scope.get("extensions", {}).get(
                    "marg.server_dispatch_queue_ms")
                headers = [
                    *message.get("headers", []),
                    (b"x-marg-app-ms", f"{elapsed:.6f}".encode("ascii")),
                ]
                if queue_ms is not None:
                    headers.append((b"x-marg-server-queue-ms", f"{queue_ms:.6f}".encode("ascii")))
                message = {**message, "headers": [
                    *headers,
                ]}
            await send(message)

        await self.app(scope, receive, timed_send)


def create_app(model_path: str = "models/fraud_model.joblib",
               thresholds_path: str = "models/thresholds.json",
               demo_db_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = joblib.load(model_path)
        # Single-row scoring avoids starting several native LightGBM threads per request.
        if hasattr(model, "set_params"):
            model.set_params(n_jobs=1)
        app.state.model = model
        app.state.thresholds = json.loads(Path(thresholds_path).read_text())
        app.state.explainer = None
        database_path = demo_db_path or os.environ.get(
            "MARG_DEMO_DB_PATH", str(PROJECT_ROOT / "artifacts" / "demo-transactions.sqlite3")
        )
        app.state.demo_runtime = DemoRuntime(database_path)
        yield

    app = FastAPI(title="Marg fraud scoring", lifespan=lifespan)
    app.add_middleware(AppTimingMiddleware)

    if WEB_ROOT.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def portfolio() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/api/results")
    def results() -> dict[str, dict]:
        """Return only the committed benchmark files used by the project site."""
        snapshots: dict[str, dict] = {}
        for key, filename in RESULT_FILES.items():
            path = PROJECT_ROOT / "results" / filename
            if not path.is_file():
                raise HTTPException(status_code=503, detail=f"Missing result: {filename}")
            snapshots[key] = json.loads(path.read_text())
        attestation_path = PROJECT_ROOT / "artifacts" / "marg-attestation.json"
        if attestation_path.is_file():
            snapshots["attestation"] = json.loads(attestation_path.read_text())
        return snapshots

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    async def score(request: ScoreRequest) -> ScoreResponse:
        row = np.asarray(request.features, dtype=np.float32).reshape(1, -1)
        started = time.perf_counter()
        value = float(app.state.model.predict_proba(row)[0, 1])
        inference_ms = (time.perf_counter() - started) * 1000
        thresholds = app.state.thresholds
        decision = "block" if value >= thresholds["block"] else (
            "review" if value >= thresholds["review"] else "allow")
        return ScoreResponse(score=value, decision=decision, model_version="lightgbm-v1",
                             inference_ms=inference_ms)

    @app.post("/explain")
    def explain(request: ScoreRequest) -> dict:
        # SHAP is deliberately lazy and stays off the scoring path to protect latency.
        if app.state.explainer is None:
            app.state.explainer = Explainer(model_path)
        row = np.asarray(request.features, dtype=np.float32)
        return {"top_features": app.state.explainer.top_features(row, 5)}

    @app.post("/demo/transactions")
    def demo_transaction(
        body: DemoTransactionRequest,
        request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
    ) -> dict:
        """Score and route one local synthetic transaction through a fake gateway."""
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(status_code=403, detail="synthetic demo writes are local-only")

        row = np.asarray(body.features, dtype=np.float32).reshape(1, -1)
        started = time.perf_counter()
        score_value = float(app.state.model.predict_proba(row)[0, 1])
        inference_ms = (time.perf_counter() - started) * 1000
        if not math.isfinite(score_value):
            raise HTTPException(status_code=503, detail="model returned a non-finite score")
        thresholds = app.state.thresholds
        decision: Decision = "block" if score_value >= thresholds["block"] else (
            "review" if score_value >= thresholds["review"] else "allow"
        )
        request_data = {
            "features": body.features,
            "bank": body.bank,
            "method": body.method,
            "amount": body.amount,
            "outage_gateway": body.outage_gateway,
        }
        try:
            response = app.state.demo_runtime.submit(
                idempotency_key=idempotency_key,
                request_data=request_data,
                score=score_value,
                decision=decision,
                inference_ms=inference_ms,
            )
        except IdempotencyConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return response

    @app.get("/demo/transactions")
    def demo_transactions(limit: int = 20) -> list[dict]:
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        return app.state.demo_runtime.recent_transactions(limit)

    @app.get("/demo/summary")
    def demo_summary() -> dict:
        return app.state.demo_runtime.summary()

    @app.get("/demo/examples")
    def demo_examples() -> dict:
        examples_path = PROJECT_ROOT / "artifacts" / "demo-samples.json"
        if not examples_path.is_file():
            raise HTTPException(
                status_code=503,
                detail="demo examples are not generated; run make demo-examples",
            )
        return json.loads(examples_path.read_text())

    return app


app = create_app()
