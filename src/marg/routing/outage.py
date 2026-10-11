from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class OutageSignal:
    kind: str
    bank: str
    gateway: str
    tick: int
    baseline_success_rate: float
    recent_success_rate: float
    evidence_start_tick: int
    evidence_end_tick: int


class ObservedOutageDetector:
    """Signal on a persistent success-rate drop using only outcomes actually observed."""

    def __init__(self, baseline_window: int = 100, detection_window: int = 25,
                 min_baseline_success: float = 0.75, max_recent_success: float = 0.50) -> None:
        if baseline_window < 1 or detection_window < 1:
            raise ValueError("observation windows must be positive")
        if not 0.0 <= max_recent_success < min_baseline_success <= 1.0:
            raise ValueError("success-rate thresholds must satisfy 0 <= recent < baseline <= 1")
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.min_baseline_success = min_baseline_success
        self.max_recent_success = max_recent_success
        self.history: dict[tuple[str, str], deque[tuple[int, bool]]] = defaultdict(
            lambda: deque(maxlen=baseline_window + detection_window))
        self.active: dict[tuple[str, str], tuple[float, float]] = {}

    def observe(self, bank: str, gateway: str, success: bool, tick: int) -> OutageSignal | None:
        key = (bank, gateway)
        outcomes = self.history[key]
        outcomes.append((tick, bool(success)))
        required = self.baseline_window + self.detection_window
        if len(outcomes) < required:
            return None

        # Compare an older baseline window with only the latest observed outcomes.
        baseline = list(outcomes)[:self.baseline_window]
        recent = list(outcomes)[-self.detection_window:]
        baseline_rate = sum(result for _, result in baseline) / len(baseline)
        recent_rate = sum(result for _, result in recent) / len(recent)
        if (baseline_rate >= self.min_baseline_success
                and recent_rate <= self.max_recent_success and key not in self.active):
            self.active[key] = (baseline_rate, recent_rate)
            return OutageSignal("detected", bank, gateway, tick, baseline_rate, recent_rate,
                                recent[0][0], recent[-1][0])
        # Keep long outages active after the sliding baseline itself has been contaminated.
        if recent_rate > self.max_recent_success and key in self.active:
            del self.active[key]
            return OutageSignal("recovered", bank, gateway, tick, baseline_rate, recent_rate,
                                recent[0][0], recent[-1][0])
        return None
