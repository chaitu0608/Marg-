from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marg.routing.outage import ObservedOutageDetector
from marg.sim.constants import CONTEXTS, GATEWAYS
from marg.sim.traffic import generate_traffic
from marg.sim.world import Outage, make_world

ROOT = Path(__file__).resolve().parents[1]


def evaluate_seed(seed: int, horizon: int) -> dict:
    world = make_world(seed, horizon)
    traffic = generate_traffic(world, seed + 100_000)
    detector = ObservedOutageDetector()
    bank_counts: dict[str, int] = {}
    gateway_outages: dict[tuple[str, str], list[Outage]] = {}
    for event in world.outages:
        gateway_outages.setdefault((event.bank, event.gateway), []).append(event)

    detected: set[tuple[str, int]] = set()
    pending: dict[tuple[str, str], Outage | None] = {}
    detection_delays: list[int] = []
    recovery_delays: list[int] = []
    probe_counts = {(event.bank, event.start): 0 for event in world.outages}
    false_alarms = 0
    premature_clears = 0
    for tx in traffic:
        # Balanced probing makes every gateway observable without consulting its true rate.
        count = bank_counts.get(tx.bank, 0)
        gateway = GATEWAYS[count % len(GATEWAYS)]
        bank_counts[tx.bank] = count + 1
        gateway_index = GATEWAYS.index(gateway)
        context_index = CONTEXTS.index(tx.context)
        for event in gateway_outages.get((tx.bank, gateway), []):
            if event.start <= tx.t < event.end:
                probe_counts[(event.bank, event.start)] += 1
        observed_success = bool(
            world.uniforms[tx.id, gateway_index]
            < world.rates[tx.t, gateway_index, context_index]
        )
        signal = detector.observe(tx.bank, gateway, observed_success, tx.t)
        if signal is None:
            continue

        key = (tx.bank, gateway)
        if signal.kind == "detected":
            # Attribute a delayed alert when the observations that triggered it overlap an outage.
            event = next((candidate for candidate in gateway_outages.get(key, [])
                          if candidate.start <= signal.evidence_end_tick
                          and candidate.end > signal.evidence_start_tick), None)
            pending[key] = event
            if event is None:
                false_alarms += 1
            else:
                detected.add((event.bank, event.start))
                detection_delays.append(tx.t - event.start)
        elif signal.kind == "recovered":
            event = pending.pop(key, None)
            if event is not None:
                if tx.t < event.end:
                    premature_clears += 1
                else:
                    recovery_delays.append(tx.t - event.end)

    return {
        "seed": seed,
        "outage_count": len(world.outages),
        "detected_outage_count": len(detected),
        "missed_outage_count": len(world.outages) - len(detected),
        "false_alarm_count": false_alarms,
        "premature_alert_clear_count": premature_clears,
        "observed_outcomes": len(traffic),
        "outage_probe_counts": list(probe_counts.values()),
        "detection_delay_ticks": detection_delays,
        "recovery_delay_ticks": recovery_delays,
        "unrecovered_detected_outage_count": sum(event is not None for event in pending.values()),
        "unrecovered_false_alarm_count": sum(event is None for event in pending.values()),
    }


def run(seeds: int = 20, horizon: int = 20_000, seed_start: int = 7) -> dict:
    runs = [evaluate_seed(seed, horizon) for seed in range(seed_start, seed_start + seeds)]
    detections = [delay for run in runs for delay in run["detection_delay_ticks"]]
    recoveries = [delay for run in runs for delay in run["recovery_delay_ticks"]]
    probe_counts = [count for run in runs for count in run["outage_probe_counts"]]
    outage_count = sum(run["outage_count"] for run in runs)
    observed = sum(run["observed_outcomes"] for run in runs)
    false_alarms = sum(run["false_alarm_count"] for run in runs)
    return {
        "seed_start": seed_start,
        "seed_count": seeds,
        "horizon": horizon,
        "policy": "balanced per-bank gateway probes; detector sees only selected outcomes",
        "detector_parameters": {"baseline_window": 100, "detection_window": 25,
                                 "min_baseline_success": 0.75, "max_recent_success": 0.50},
        "outage_count": outage_count,
        "detected_outage_count": sum(run["detected_outage_count"] for run in runs),
        "missed_outage_count": sum(run["missed_outage_count"] for run in runs),
        "false_alarm_count": false_alarms,
        "premature_alert_clear_count": sum(run["premature_alert_clear_count"] for run in runs),
        "false_alarms_per_10000_observed_outcomes": false_alarms / observed * 10_000,
        "median_probe_outcomes_per_outage": float(np.median(probe_counts)) if probe_counts else None,
        "outages_with_fewer_than_detection_window_probes": sum(count < 25 for count in probe_counts),
        "median_detection_delay_ticks": float(np.median(detections)) if detections else None,
        "median_recovery_delay_ticks": float(np.median(recoveries)) if recoveries else None,
        "unrecovered_detected_outage_count": sum(
            run["unrecovered_detected_outage_count"] for run in runs),
        "unrecovered_false_alarm_count": sum(
            run["unrecovered_false_alarm_count"] for run in runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20_000)
    args = parser.parse_args()
    result = run(args.seeds, args.horizon, args.seed)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "outage_detection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
