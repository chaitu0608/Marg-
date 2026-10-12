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


class ConsecutiveOutcomeDetector:
    """Small run-length baseline that alerts on repeated failures and clears on successes."""

    def __init__(self, failure_run: int = 5, recovery_run: int = 8) -> None:
        if failure_run < 1 or recovery_run < 1:
            raise ValueError("run lengths must be positive")
        self.failure_run = failure_run
        self.recovery_run = recovery_run
        self.failures: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.successes: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.active: set[tuple[str, str]] = set()

    def observe(self, bank: str, gateway: str, success: bool, tick: int) -> OutageSignal | None:
        key = (bank, gateway)
        if success:
            self.failures[key].clear()
            if key not in self.active:
                return None
            self.successes[key].append(tick)
            if len(self.successes[key]) >= self.recovery_run:
                self.active.remove(key)
                evidence = self.successes[key][-self.recovery_run:]
                self.successes[key].clear()
                return OutageSignal("recovered", bank, gateway, tick, 1.0, 1.0,
                                    evidence[0], evidence[-1])
            return None

        self.successes[key].clear()
        # A success breaks a failure run; active alerts need no more failure history.
        if key in self.active:
            return None
        self.failures[key].append(tick)
        if len(self.failures[key]) < self.failure_run:
            return None
        self.active.add(key)
        evidence = self.failures[key][-self.failure_run:]
        self.failures[key].clear()
        return OutageSignal("detected", bank, gateway, tick, 1.0, 0.0,
                            evidence[0], evidence[-1])
