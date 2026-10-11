from collections import defaultdict

import numpy as np

from marg.sim.constants import GATEWAYS


class EpsilonGreedyRouter:
    name = "epsilon_greedy"

    def __init__(self, epsilon: float = 0.1, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.stats: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
            lambda: {g: [0, 0] for g in GATEWAYS})

    def select(self, context: tuple[str, str], t: int) -> str:
        if self.rng.random() < self.epsilon:
            return GATEWAYS[int(self.rng.integers(len(GATEWAYS)))]
        return max(GATEWAYS, key=lambda g: self._mean(context, g))

    def update(self, context: tuple[str, str], gateway: str, success: bool, t: int) -> None:
        self.stats[context][gateway][0 if success else 1] += 1

    def _mean(self, context: tuple[str, str], gateway: str) -> float:
        wins, losses = self.stats[context][gateway]
        return (1 + wins) / (2 + wins + losses)
