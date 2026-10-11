from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.run_routing import _recovery_times
from marg.routing.epsilon_greedy import EpsilonGreedyRouter
from marg.routing.thompson import SlidingWindowThompsonRouter
from marg.sim.traffic import generate_traffic
from marg.sim.world import CONTEXTS, GATEWAYS, make_world

ROOT = Path(__file__).resolve().parents[1]


def measure(window: int, epsilon: float, seed: int = 101, horizon: int = 8_000) -> dict:
    world = make_world(seed, horizon)
    traffic = generate_traffic(world, seed + 100_000)
    ids = [CONTEXTS.index(tx.context) for tx in traffic]
    router = (SlidingWindowThompsonRouter(window, seed) if epsilon == 0 else
              EpsilonGreedyRouter(epsilon, seed))
    success = 0
    total = 0
    outage_success = outage_total = 0
    choices = []
    for t, tx in enumerate(traffic):
        gateway = router.select(tx.context, tx.t)
        choices.append(gateway)
        gi = GATEWAYS.index(gateway)
        ok = bool(world.uniforms[tx.id, gi] < world.rates[tx.t, gi, ids[t]])
        router.update(tx.context, gateway, ok, tx.t)
        success += ok
        total += 1
        if any(e.start <= tx.t < e.end and e.bank == tx.bank for e in world.outages):
            outage_success += ok
            outage_total += 1
    recovery = _recovery_times(world, choices, np.asarray(ids), traffic)
    completed = [r for r in recovery if r is not None]
    return {"success_rate": success / total,
            "outage_success_rate": outage_success / outage_total if outage_total else None,
            "median_recovery_ticks": float(np.median(completed)) if completed else None,
            "never_recovered": sum(r is None for r in recovery)}


def main() -> None:
    windows = [50, 100, 200, 500, 1_000, 5_000]
    epsilons = [0.01, 0.05, 0.1, 0.2]
    rows = []
    for window in windows:
        row = {"window": window, "metrics": measure(window, 0)}
        rows.append(row)
    epsilon_rows = [{"epsilon": eps, "metrics": measure(200, eps)} for eps in epsilons]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "window_sweep.json").write_text(json.dumps(
        {"seed": 101, "horizon": 8_000, "thompson": rows, "epsilon_greedy": epsilon_rows}, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(windows, [r["metrics"]["success_rate"] for r in rows], marker="o", label="success rate")
    ax.set_xscale("log")
    ax.set(title="Thompson window size trade-off", xlabel="Window W (observations per gateway)",
           ylabel="Success rate")
    recovery_ax = ax.twinx()
    recovery_ax.plot(windows, [r["metrics"]["median_recovery_ticks"] or 0 for r in rows],
                     color="tab:red", marker="s", label="median recovery")
    recovery_ax.set_ylabel("Median recovery ticks", color="tab:red")
    ax.text(0.01, 0.02, "Synthetic traffic and gateway behavior", transform=ax.transAxes,
            fontsize=8, color="dimgray")
    ax.legend(loc="upper left")
    recovery_ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "window_sweep.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"thompson": rows, "epsilon_greedy": epsilon_rows}, indent=2))


if __name__ == "__main__":
    main()
