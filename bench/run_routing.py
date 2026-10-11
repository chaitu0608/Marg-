from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marg.routing.epsilon_greedy import EpsilonGreedyRouter
from marg.routing.round_robin import RoundRobinRouter
from marg.routing.static import StaticRouter
from marg.routing.thompson import SlidingWindowThompsonRouter
from marg.sim.traffic import generate_traffic
from marg.sim.world import CONTEXTS, GATEWAYS, World, make_world

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 20
HORIZON = 20_000


def router_set(seed: int) -> dict[str, Any]:
    return {
        "static": StaticRouter(seed),
        "round_robin": RoundRobinRouter(),
        "epsilon_greedy": EpsilonGreedyRouter(0.1, seed),
        "thompson_w200": SlidingWindowThompsonRouter(200, seed),
    }


def _recovery_times(world: World, choices: list[str], context_ids: np.ndarray,
                    traffic: list) -> list[int | None]:
    results: list[int | None] = []
    for event in world.outages:
        before = [choices[i] == event.gateway for i, tx in enumerate(traffic)
                  if tx.t < event.start and CONTEXTS[int(context_ids[i])][0] == event.bank]
        # If routing was already below the limit, no post-outage adaptation was needed.
        baseline = before[-50:]
        if baseline and sum(baseline) / len(baseline) <= 0.10:
            results.append(0)
            continue
        # Count affected-bank transactions after outage start, not wall-clock ticks.
        recent: deque[bool] = deque(maxlen=50)
        recovered = None
        seen = 0
        for txn_id, tx in enumerate(traffic):
            if tx.t < event.start or CONTEXTS[int(context_ids[txn_id])][0] != event.bank:
                continue
            recent.append(choices[txn_id] == event.gateway)
            seen += 1
            if len(recent) == 50 and sum(recent) <= 5:
                recovered = seen
                break
        results.append(recovered)
    return results


def run_one(seed: int, horizon: int) -> tuple[dict[str, Any], dict[str, Any]]:
    world = make_world(seed, horizon)
    traffic = generate_traffic(world, seed + 100_000)
    ids = np.fromiter((CONTEXTS.index(tx.context) for tx in traffic), dtype=np.int16)
    outcomes: dict[str, dict[str, Any]] = {}
    curves: dict[str, Any] = {}
    for name, router in router_set(seed).items():
        selected: list[str] = []
        successes = np.zeros(horizon, dtype=np.uint8)
        regrets = np.zeros(horizon, dtype=np.float64)
        outage_mask = np.zeros(horizon, dtype=bool)
        for event in world.outages:
            for txn_id, tx in enumerate(traffic):
                if event.start <= tx.t < event.end and tx.bank == event.bank:
                    outage_mask[txn_id] = True
        outage_success = 0
        outage_count = 0
        for tick, tx in enumerate(traffic):
            gateway = router.select(tx.context, tx.t)
            selected.append(gateway)
            gi = GATEWAYS.index(gateway)
            success = bool(world.uniforms[tx.id, gi] < world.rates[tx.t, gi, ids[tx.id]])
            router.update(tx.context, gateway, success, tx.t)
            successes[tx.id] = success
            oracle_rate = float(world.rates[tx.t, :, ids[tx.id]].max())
            chosen_rate = float(world.rates[tx.t, gi, ids[tx.id]])
            regrets[tx.id] = oracle_rate - chosen_rate
            if outage_mask[tx.id]:
                outage_success += int(success)
                outage_count += 1
        recovery = _recovery_times(world, selected, ids, traffic)
        outcomes[name] = {
            "success_rate": float(successes.mean()),
            "cumulative_regret": float(regrets.sum()),
            "outage_success_rate": float(outage_success / outage_count) if outage_count else None,
            "recovery_ticks": recovery,
            "never_recovered": sum(x is None for x in recovery),
            "outage_count": len(world.outages),
        }
        curves[name] = {"regret": np.cumsum(regrets), "success": successes}
    oracle = np.empty(horizon, dtype=np.uint8)
    oracle_outage_count = 0
    oracle_outage_success = 0
    outage_mask = np.zeros(horizon, dtype=bool)
    for event in world.outages:
        for txn_id, tx in enumerate(traffic):
            if event.start <= tx.t < event.end and tx.bank == event.bank:
                outage_mask[txn_id] = True
    for txn_id, ci in enumerate(ids):
        tx = traffic[txn_id]
        oracle[txn_id] = world.uniforms[tx.id, int(np.argmax(world.rates[tx.t, :, ci]))] < world.rates[tx.t, :, ci].max()
        if outage_mask[txn_id]:
            oracle_outage_count += 1
            oracle_outage_success += int(oracle[txn_id])
    outcomes["oracle"] = {
        "success_rate": float(oracle.mean()), "cumulative_regret": 0.0,
        "outage_success_rate": oracle_outage_success / oracle_outage_count if oracle_outage_count else None,
        "recovery_ticks": [], "never_recovered": 0,
        "outage_count": len(world.outages),
    }
    curves["oracle"] = {"success": oracle}
    return outcomes, curves


def mean_ci(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    half = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "ci95_low": float(a.mean() - half),
            "ci95_high": float(a.mean() + half)}


def bootstrap_mean_ci(values: list[float], seed: int = 12_345) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    # Resample per-seed medians so the interval respects their nonnegative, skewed shape.
    samples = rng.choice(a, size=(20_000, len(a)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"mean": float(a.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    all_runs: dict[str, list[dict[str, Any]]] = {}
    all_curves: dict[str, list[Any]] = {}
    for seed in range(args.seed, args.seed + args.seeds):
        run, curves = run_one(seed, args.horizon)
        for key, value in run.items():
            all_runs.setdefault(key, []).append(value)
        for key, value in curves.items():
            all_curves.setdefault(key, []).append(value)
    summary: dict[str, Any] = {}
    for name, runs in all_runs.items():
        valid_recovery = [float(r) for run in runs for r in run["recovery_ticks"] if r is not None]
        per_seed_medians = [float(np.median([x for x in run["recovery_ticks"] if x is not None]))
                            for run in runs if any(x is not None for x in run["recovery_ticks"])]
        summary[name] = {
            "success_rate": mean_ci([r["success_rate"] for r in runs]),
            "cumulative_regret": mean_ci([r["cumulative_regret"] for r in runs]),
            "outage_success_rate": mean_ci([r["outage_success_rate"] for r in runs
                                              if r["outage_success_rate"] is not None]),
            "median_recovery_ticks": float(np.median(valid_recovery)) if valid_recovery else None,
            "mean_seed_median_recovery_ticks": bootstrap_mean_ci(per_seed_medians)
            if per_seed_medians else None,
            "recovery_median_seed_count": len(per_seed_medians),
            "never_recovered": sum(r["never_recovered"] for r in runs),
            "outage_count": sum(r["outage_count"] for r in runs),
        }
    output = {"seed_start": args.seed, "seed_count": args.seeds, "horizon": args.horizon,
              "routing": summary, "runs": all_runs}
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "routing.json").write_text(json.dumps(output, indent=2))
    make_plots(all_curves, results_dir, args.seed)
    print(json.dumps(summary, indent=2))


def make_plots(curves: dict[str, list[Any]], out: Path, seed: int) -> None:
    names = list(curves)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in names:
        if name == "oracle":
            continue
        mean_curve = np.mean([r["regret"] for r in curves[name]], axis=0)
        ax.plot(mean_curve, label=name)
    ax.set(title="Cumulative expected regret (mean over seeds)", xlabel="Transaction",
           ylabel="Expected regret")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "routing_regret.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ("static", "thompson_w200"):
        roll = np.convolve(curves[name][0]["success"], np.ones(500) / 500, mode="valid")
        ax.plot(np.arange(len(roll)), roll, label=name)
    sample_world = make_world(seed, len(curves["static"][0]["success"]))
    for event in sample_world.outages:
        ax.axvspan(event.start, event.end, color="crimson", alpha=0.12)
    ax.set(title="Rolling success rate through synthetic outages (first seed)", xlabel="Transaction",
           ylabel="Success rate")
    ax.text(0.01, 0.02, "Synthetic traffic and gateway behavior", transform=ax.transAxes,
            fontsize=8, color="dimgray")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "routing_success.png", dpi=140)
    plt.close(fig)

    vals = []
    labels = []
    payload = json.loads((out / "routing.json").read_text())
    for name, stats in payload["routing"].items():
        if name != "oracle":
            valid = [x for run in payload["runs"][name] for x in run["recovery_ticks"]
                     if x is not None]
            vals.append(valid)
            labels.append(f"{name}\nnever={stats['never_recovered']}")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(vals, tick_labels=labels, showfliers=False)
    ax.set(title="Outage recovery ticks (successful recoveries)", ylabel="Affected-bank transactions")
    fig.tight_layout()
    fig.savefig(out / "routing_recovery.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
