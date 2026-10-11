from collections import defaultdict, deque

import numpy as np

from marg.sim.constants import GATEWAYS


class SlidingWindowThompsonRouter:
    name = "sliding_window_thompson"

    def __init__(self, window: int = 200, seed: int = 0) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.observations: dict[tuple[tuple[str, str], str], deque[bool]] = defaultdict(
            lambda: deque(maxlen=window))

    def select(self, context: tuple[str, str], t: int) -> str:
        # Beta(1,1) starts uniform; posterior samples naturally trade exploration for exploitation.
        draws = []
        for gateway in GATEWAYS:
            outcomes = self.observations[(context, gateway)]
            wins = sum(outcomes)
            losses = len(outcomes) - wins
            draws.append(self.rng.beta(1 + wins, 1 + losses))
        return GATEWAYS[int(np.argmax(draws))]

    def update(self, context: tuple[str, str], gateway: str, success: bool, t: int) -> None:
        # The bounded deque forgets old outcomes once the environment has changed.
        self.observations[(context, gateway)].append(success)

    def posterior_mean(self, context: tuple[str, str], gateway: str) -> float:
        outcomes = self.observations[(context, gateway)]
        return (1 + sum(outcomes)) / (2 + len(outcomes))
