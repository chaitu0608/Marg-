from collections import defaultdict

from marg.sim.constants import GATEWAYS


class RoundRobinRouter:
    name = "round_robin"

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def select(self, context: tuple[str, str], t: int) -> str:
        i = self.counts[context]
        self.counts[context] += 1
        return GATEWAYS[i % len(GATEWAYS)]

    def update(self, context: tuple[str, str], gateway: str, success: bool, t: int) -> None:
        pass
