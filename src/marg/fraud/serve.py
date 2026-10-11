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

from marg.fraud.explain import Explainer


class ScoreRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


class ScoreResponse(BaseModel):
    score: float
    decision: Literal["allow", "review", "block"]
    model_version: str
    latency_ms: float


def create_app(model_path: str = "models/fraud_model.joblib",
               thresholds_path: str = "models/thresholds.json") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = joblib.load(model_path)
        app.state.thresholds = json.loads(Path(thresholds_path).read_text())
        app.state.explainer = None
        yield

    app = FastAPI(title="Marg fraud scoring", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        started = time.perf_counter()
        row = np.asarray(request.features, dtype=np.float32).reshape(1, -1)
        value = float(app.state.model.predict_proba(row)[0, 1])
        thresholds = app.state.thresholds
        decision = "block" if value >= thresholds["block"] else (
            "review" if value >= thresholds["review"] else "allow")
        return ScoreResponse(score=value, decision=decision, model_version="lightgbm-v1",
                             latency_ms=(time.perf_counter() - started) * 1000)

    @app.post("/explain")
    def explain(request: ScoreRequest) -> dict:
        # SHAP is deliberately lazy and stays off the scoring path to protect latency.
        if app.state.explainer is None:
            app.state.explainer = Explainer(model_path)
        row = np.asarray(request.features, dtype=np.float32)
        return {"top_features": app.state.explainer.top_features(row, 5)}

    return app


app = create_app()
