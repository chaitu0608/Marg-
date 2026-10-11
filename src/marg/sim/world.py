from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from marg.sim.constants import BANKS, CONTEXTS, GATEWAYS


@dataclass(frozen=True)
class Outage:
    start: int
    end: int
    gateway: str
    bank: str


@dataclass
class World:
    seed: int
    horizon: int
    rates: NDArray[np.float64]
    uniforms: NDArray[np.float64]
    outages: list[Outage]
    costs: NDArray[np.float64]
    latencies_ms: NDArray[np.float64]

    def p_true(self, gateway: str, context: tuple[str, str], t: int) -> float:
        gi = GATEWAYS.index(gateway)
        ci = CONTEXTS.index(context)
        return float(self.rates[min(max(t, 0), self.horizon - 1), gi, ci])

    def outcome(self, gateway: str, context: tuple[str, str], t: int, u: float) -> bool:
        return u < self.p_true(gateway, context, t)

    def oracle_choice(self, context: tuple[str, str], t: int) -> str:
        ci = CONTEXTS.index(context)
        tick = min(max(t, 0), self.horizon - 1)
        return GATEWAYS[int(np.argmax(self.rates[tick, :, ci]))]


def make_world(seed: int, horizon: int = 200_000) -> World:
    rng = np.random.default_rng(seed)
    g_count, c_count = len(GATEWAYS), len(CONTEXTS)
    # Independent gateway-context baselines prevent a single universally best gateway.
    baselines = rng.uniform(0.80, 0.97, (g_count, c_count))
    rates = np.empty((horizon, g_count, c_count), dtype=np.float32)
    current = baselines.copy()
    outage_list: list[Outage] = []
    t_next = int(rng.exponential(1_800))
    while t_next < horizon:
        duration = int(rng.integers(300, 1_501))
        outage_list.append(Outage(t_next, min(horizon, t_next + duration),
                                  GATEWAYS[int(rng.integers(g_count))],
                                  BANKS[int(rng.integers(len(BANKS)))]))
        t_next += max(1, int(rng.exponential(1_800)))
    starts: dict[int, list[Outage]] = {}
    ends: dict[int, list[Outage]] = {}
    for outage in outage_list:
        starts.setdefault(outage.start, []).append(outage)
        ends.setdefault(outage.end, []).append(outage)
    active: list[Outage] = []
    for tick in range(horizon):
        for event in ends.get(tick, []):
            active.remove(event)
        current += rng.normal(0.0, 0.002, current.shape)
        np.clip(current, 0.30, 0.99, out=current)
        active.extend(starts.get(tick, []))
        rates[tick] = current
        for event in active:
            gi = GATEWAYS.index(event.gateway)
            for ci, context in enumerate(CONTEXTS):
                if context[0] == event.bank:
                    rates[tick, gi, ci] *= 0.25
    uniforms = rng.random((horizon, g_count), dtype=np.float32)
    costs = rng.uniform(0.002, 0.012, g_count)
    medians = np.array([70.0, 95.0, 125.0, 160.0])
    latencies = rng.lognormal(np.log(medians) - 0.5 * 0.25**2, 0.25,
                              size=(horizon, g_count))
    return World(seed, horizon, rates, uniforms, outage_list, costs, latencies)
