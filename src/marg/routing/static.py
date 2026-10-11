from collections import defaultdict

import numpy as np

from marg.sim.constants import GATEWAYS


class StaticRouter:
    name = "static"

    def __init__(self, seed: int = 0, warmup: int = 1_000) -> None:
        self.rng = np.random.default_rng(seed)
        self.warmup = warmup
        self.total = 0
        self.stats: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
            lambda: {g: [0, 0] for g in GATEWAYS})
        self.choices: dict[tuple[str, str], str] = {}

    def select(self, context: tuple[str, str], t: int) -> str:
        if self.total < self.warmup:
            return GATEWAYS[int(self.rng.integers(len(GATEWAYS)))]
        if context not in self.choices:
            self.choices[context] = max(GATEWAYS, key=lambda g: self._mean(context, g))
        return self.choices[context]

    def update(self, context: tuple[str, str], gateway: str, success: bool, t: int) -> None:
        self.total += 1
        self.stats[context][gateway][0 if success else 1] += 1
        if self.total == self.warmup:
            self.choices = {ctx: max(GATEWAYS, key=lambda g: self._mean(ctx, g))
                            for ctx in self.stats}

    def _mean(self, context: tuple[str, str], gateway: str) -> float:
        wins, losses = self.stats[context][gateway]
        return wins / (wins + losses) if wins + losses else 0.5
