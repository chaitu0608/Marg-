from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marg.sim.constants import BANKS, METHODS
from marg.sim.world import World


@dataclass(frozen=True)
class Transaction:
    id: int
    t: int
    bank: str
    method: str
    amount: float

    @property
    def context(self) -> tuple[str, str]:
        return self.bank, self.method


def generate_traffic(world: World, seed: int | None = None) -> list[Transaction]:
    rng = np.random.default_rng(world.seed if seed is None else seed)
    bank_p = np.array([0.28, 0.22, 0.18, 0.14, 0.10, 0.08])
    method_p = np.array([0.72, 0.21, 0.07])
    records = []
    tick = 0
    for txn_id in range(world.horizon):
        bank = str(rng.choice(BANKS, p=bank_p))
        method = str(rng.choice(METHODS, p=method_p))
        if txn_id % 10_000 in (500, 501, 502, 503):
            method = "upi"
        records.append(Transaction(txn_id, tick, bank, method, float(rng.lognormal(5.0, 1.1))))
        offset = txn_id % 10_000
        # Four transactions share a tick during a burst, followed by a gap that preserves
        # one transaction per tick on average across the full stream.
        tick += 0 if offset in (500, 501, 502) else 4 if offset == 503 else 1
    return records
