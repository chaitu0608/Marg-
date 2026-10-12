from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from marg.fraud.explain import Explainer


class ScoreRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


class ScoreResponse(BaseModel):
    score: float
    decision: Literal["allow", "review", "block"]
    model_version: str
    inference_ms: float


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
               thresholds_path: str = "models/thresholds.json") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = joblib.load(model_path)
        # Single-row scoring avoids starting several native LightGBM threads per request.
        if hasattr(model, "set_params"):
            model.set_params(n_jobs=1)
        app.state.model = model
        app.state.thresholds = json.loads(Path(thresholds_path).read_text())
        app.state.explainer = None
        yield

    app = FastAPI(title="Marg fraud scoring", lifespan=lifespan)
    app.add_middleware(AppTimingMiddleware)

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

    return app


app = create_app()
