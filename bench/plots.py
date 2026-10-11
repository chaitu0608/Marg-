from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marg.sim.world import CONTEXTS, GATEWAYS, make_world

ROOT = Path(__file__).resolve().parents[1]


def plot_world(seed: int = 7, horizon: int = 20_000) -> Path:
    world = make_world(seed, horizon)
    ctx = CONTEXTS[0]
    ci = CONTEXTS.index(ctx)
    path = ROOT / "results" / "world.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for gi, gateway in enumerate(GATEWAYS):
        ax.plot(world.rates[:, gi, ci], label=gateway, linewidth=0.8)
    for event in world.outages:
        if event.bank == ctx[0]:
            ax.axvspan(event.start, event.end, color="crimson", alpha=0.12)
    ax.set(title=f"Marg synthetic gateway rates · {ctx[0]} / {ctx[1]} · seed {seed}",
           xlabel="Tick", ylabel="True success probability", ylim=(0, 1))
    ax.legend(ncol=4)
    ax.text(0.01, 0.02, "Synthetic traffic and gateway behavior", transform=ax.transAxes,
            fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=20_000)
    args = parser.parse_args()
    if args.world:
        print(plot_world(args.seed, args.horizon))


if __name__ == "__main__":
    main()
