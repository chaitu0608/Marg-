from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from marg.routing.thompson import SlidingWindowThompsonRouter
from marg.sim.constants import BANKS, GATEWAYS, METHODS

Decision = Literal["allow", "review", "block"]
DEMO_SEED = 73
ROUTER_WINDOW = 200


class IdempotencyConflict(ValueError):
    """The same idempotency key was reused for a different request."""


class FakeGatewayWorld:
    """A seeded gateway simulator whose success probabilities never leave this class."""

    def __init__(self, seed: int = DEMO_SEED) -> None:
        rng = np.random.default_rng(seed)
        self._rates = rng.uniform(0.80, 0.97, (len(BANKS), len(METHODS), len(GATEWAYS)))
        self._seed = seed

    def outcome(self, gateway: str, bank: str, method: str, tick: int,
                outage_gateway: str | None) -> bool:
        bank_index = BANKS.index(bank)
        method_index = METHODS.index(method)
        gateway_index = GATEWAYS.index(gateway)
        probability = float(self._rates[bank_index, method_index, gateway_index])
        # A small deterministic wobble keeps the demo time-varying without exposing rates.
        probability += 0.02 * math.sin(tick / 90 + gateway_index + method_index)
        if gateway == outage_gateway:
            probability *= 0.12
        probability = min(0.99, max(0.30, probability))

        # Hash-based draws make retries and seeded runs stable across worker processes.
        draw_material = f"{self._seed}:{tick}:{bank}:{method}:{gateway}".encode()
        draw = int.from_bytes(hashlib.blake2b(draw_material, digest_size=8).digest(), "big")
        uniform = draw / 2**64
        return uniform < probability


class DemoRuntime:
    """Persist synthetic outcomes and rebuild the router's recent window between requests."""

    def __init__(self, database_path: str | Path, seed: int = DEMO_SEED) -> None:
        self.database_path = Path(database_path)
        self.seed = seed
        self.world = FakeGatewayWorld(seed)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS demo_transactions (
                    id TEXT PRIMARY KEY,
                    tick INTEGER NOT NULL UNIQUE,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_fingerprint TEXT NOT NULL,
                    bank TEXT NOT NULL,
                    method TEXT NOT NULL,
                    amount REAL NOT NULL,
                    score REAL NOT NULL,
                    decision TEXT NOT NULL,
                    status TEXT NOT NULL,
                    gateway TEXT,
                    success INTEGER,
                    scenario TEXT NOT NULL,
                    inference_ms REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS demo_observations (
                    tick INTEGER NOT NULL,
                    bank TEXT NOT NULL,
                    method TEXT NOT NULL,
                    gateway TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    transaction_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (tick, gateway),
                    FOREIGN KEY (transaction_id) REFERENCES demo_transactions(id)
                );
                CREATE INDEX IF NOT EXISTS demo_observation_context_idx
                    ON demo_observations(bank, method, gateway, tick DESC);
                CREATE INDEX IF NOT EXISTS demo_transaction_tick_idx
                    ON demo_transactions(tick DESC);
                """
            )

    @staticmethod
    def fingerprint(request_data: dict) -> str:
        canonical = json.dumps(request_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _seed_router(self, connection: sqlite3.Connection, bank: str, method: str,
                     tick: int) -> SlidingWindowThompsonRouter:
        router = SlidingWindowThompsonRouter(window=ROUTER_WINDOW, seed=self.seed + tick)
        for gateway in GATEWAYS:
            recent = connection.execute(
                """SELECT tick, success FROM demo_observations
                   WHERE bank = ? AND method = ? AND gateway = ?
                   ORDER BY tick DESC LIMIT ?""",
                (bank, method, gateway, ROUTER_WINDOW),
            ).fetchall()
            # Replay oldest-first so each arm's deque matches the actual recent window.
            for observation in reversed(recent):
                router.update((bank, method), gateway, bool(observation["success"]),
                              int(observation["tick"]))
        return router

    def submit(self, *, idempotency_key: str, request_data: dict, score: float,
               decision: Decision, inference_ms: float) -> dict:
        fingerprint = self.fingerprint(request_data)
        bank = request_data["bank"]
        method = request_data["method"]
        amount = float(request_data["amount"])
        outage_gateway = request_data.get("outage_gateway")
        scenario = f"{outage_gateway} outage" if outage_gateway else "normal gateway conditions"

        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_fingerprint, response_json FROM demo_transactions "
                "WHERE idempotency_key = ?", (idempotency_key,),
            ).fetchone()
            if existing:
                if existing["request_fingerprint"] != fingerprint:
                    raise IdempotencyConflict("idempotency key was used with a different request")
                response = json.loads(existing["response_json"])
                response["idempotent_replay"] = True
                return response

            tick = int(connection.execute(
                "SELECT COALESCE(MAX(tick), 0) + 1 FROM demo_transactions"
            ).fetchone()[0])
            transaction_id = f"marg-demo-{tick:08d}"
            gateway: str | None = None
            success: bool | None = None
            observation_count = 0

            if decision == "block":
                status = "blocked"
            elif decision == "review":
                status = "awaiting_review"
            else:
                context = (bank, method)
                router = self._seed_router(connection, bank, method, tick)
                gateway = router.select(context, tick)
                success = self.world.outcome(gateway, bank, method, tick, outage_gateway)
                router.update(context, gateway, success, tick)
                observation_count = len(router.observations[(context, gateway)])
                status = "succeeded" if success else "failed"

            response = {
                "transaction_id": transaction_id,
                "tick": tick,
                "bank": bank,
                "method": method,
                "amount": amount,
                "decision": decision,
                "score": float(score),
                "model_version": "lightgbm-v1",
                "status": status,
                "gateway": gateway,
                "gateway_outcome": ("success" if success else "failure") if success is not None else None,
                "router_observations_for_gateway": observation_count,
                "scenario": scenario,
                "inference_ms": float(inference_ms),
                "synthetic": True,
                "idempotent_replay": False,
            }
            created_at = datetime.now(UTC).isoformat()
            connection.execute(
                """INSERT INTO demo_transactions
                   (id, tick, idempotency_key, request_fingerprint, bank, method, amount,
                    score, decision, status, gateway, success, scenario, inference_ms,
                    created_at, response_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (transaction_id, tick, idempotency_key, fingerprint, bank, method, amount,
                 float(score), decision, status, gateway,
                 int(success) if success is not None else None, scenario, inference_ms,
                 created_at, json.dumps(response, separators=(",", ":"))),
            )
            if gateway is not None and success is not None:
                connection.execute(
                    """INSERT INTO demo_observations
                       (tick, bank, method, gateway, success, transaction_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (tick, bank, method, gateway, int(success), transaction_id),
                )
            connection.commit()
            return response

    def recent_transactions(self, limit: int = 20) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT response_json FROM demo_transactions ORDER BY tick DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["response_json"]) for row in rows]

    def summary(self) -> dict:
        with self.connect() as connection:
            counts = connection.execute(
                "SELECT status, COUNT(*) AS count FROM demo_transactions GROUP BY status"
            ).fetchall()
            gateway_rows = connection.execute(
                """SELECT gateway, COUNT(*) AS attempts, SUM(success) AS successes
                   FROM demo_observations GROUP BY gateway ORDER BY gateway"""
            ).fetchall()
            recent_rows = connection.execute(
                """SELECT gateway, success FROM demo_observations
                   ORDER BY tick DESC LIMIT 50"""
            ).fetchall()
            total = int(connection.execute("SELECT COUNT(*) FROM demo_transactions").fetchone()[0])
            observed = int(connection.execute("SELECT COUNT(*) FROM demo_observations").fetchone()[0])
        # Recent counts expose adaptation without letting the UI inspect router probabilities.
        recent = {gateway: {"attempts": 0, "successes": 0} for gateway in GATEWAYS}
        for row in recent_rows:
            recent[row["gateway"]]["attempts"] += 1
            recent[row["gateway"]]["successes"] += int(row["success"])
        return {
            "synthetic": True,
            "transaction_count": total,
            "router_observation_count": observed,
            "status_counts": {row["status"]: int(row["count"]) for row in counts},
            "gateways": [
                {"gateway": row["gateway"], "attempts": int(row["attempts"]),
                 "successes": int(row["successes"] or 0)}
                for row in gateway_rows
            ],
            "recent_route_window": [
                {"gateway": gateway, **recent[gateway]} for gateway in GATEWAYS
            ],
        }
