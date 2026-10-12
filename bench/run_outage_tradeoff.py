from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from marg.routing.outage import ConsecutiveOutcomeDetector, ObservedOutageDetector, OutageSignal
from marg.sim.constants import CONTEXTS, GATEWAYS
from marg.sim.traffic import generate_traffic
from marg.sim.world import Outage, World, make_world

ROOT = Path(__file__).resolve().parents[1]

DETECTORS: dict[str, Callable[[], object]] = {
    "consecutive_5_failures": lambda: ConsecutiveOutcomeDetector(5, 8),
    "drop_50_vs_10": lambda: ObservedOutageDetector(50, 10, 0.75, 0.50),
    "drop_100_vs_25": lambda: ObservedOutageDetector(100, 25, 0.75, 0.50),
    "drop_200_vs_25": lambda: ObservedOutageDetector(200, 25, 0.80, 0.50),
}
MINIMUM_PROBES = {
    "consecutive_5_failures": 5,
    "drop_50_vs_10": 10,
    "drop_100_vs_25": 25,
    "drop_200_vs_25": 25,
}


def evaluate_seed(seed: int, horizon: int, detector_name: str,
                  probe_every: int, world: World | None = None,
                  traffic: list | None = None) -> dict:
    world = make_world(seed, horizon) if world is None else world
    traffic = generate_traffic(world, seed + 100_000) if traffic is None else traffic
    detector = DETECTORS[detector_name]()
    bank_counts: dict[str, int] = {}
    gateway_outages: dict[tuple[str, str], list[Outage]] = {}
    for event in world.outages:
        gateway_outages.setdefault((event.bank, event.gateway), []).append(event)

    detected: set[tuple[str, int]] = set()
    pending: dict[tuple[str, str], Outage | None] = {}
    delays: list[int] = []
    probe_counts = {(event.bank, event.start): 0 for event in world.outages}
    false_alarms = 0
    probes = 0
    for tx in traffic:
        if tx.t % probe_every:
            continue
        count = bank_counts.get(tx.bank, 0)
        gateway = GATEWAYS[count % len(GATEWAYS)]
        bank_counts[tx.bank] = count + 1
        probes += 1
        gateway_index = GATEWAYS.index(gateway)
        context_index = CONTEXTS.index(tx.context)
        key = (tx.bank, gateway)
        for event in gateway_outages.get(key, []):
            if event.start <= tx.t < event.end:
                probe_counts[(event.bank, event.start)] += 1
        success = bool(world.uniforms[tx.id, gateway_index]
                       < world.rates[tx.t, gateway_index, context_index])
        signal: OutageSignal | None = detector.observe(tx.bank, gateway, success, tx.t)
        if signal is None:
            continue
        if signal.kind == "detected":
            event = next((candidate for candidate in gateway_outages.get(key, [])
                          if candidate.start <= signal.evidence_end_tick
                          and candidate.end > signal.evidence_start_tick), None)
            pending[key] = event
            if event is None:
                false_alarms += 1
            else:
                detected.add((event.bank, event.start))
                delays.append(tx.t - event.start)
        elif signal.kind == "recovered":
            pending.pop(key, None)

    return {
        "outage_count": len(world.outages),
        "detected_outage_count": len(detected),
        "false_alarm_count": false_alarms,
        "probe_count": probes,
        "outage_probe_counts": list(probe_counts.values()),
        "detection_delay_ticks": delays,
    }


def run(seeds: int = 20, horizon: int = 20_000, seed_start: int = 7) -> dict:
    grouped_runs: dict[tuple[str, int], list[dict]] = {
        (name, cadence): [] for name in DETECTORS for cadence in (1, 2, 4)}
    # Reuse each seeded world for all detector settings so every comparison is paired.
    for seed in range(seed_start, seed_start + seeds):
        world = make_world(seed, horizon)
        traffic = generate_traffic(world, seed + 100_000)
        for detector_name in DETECTORS:
            for probe_every in (1, 2, 4):
                grouped_runs[(detector_name, probe_every)].append(
                    evaluate_seed(seed, horizon, detector_name, probe_every, world, traffic))
    scenarios = []
    for detector_name in DETECTORS:
        for probe_every in (1, 2, 4):
            runs = grouped_runs[(detector_name, probe_every)]
            outages = sum(item["outage_count"] for item in runs)
            detected = sum(item["detected_outage_count"] for item in runs)
            false_alarms = sum(item["false_alarm_count"] for item in runs)
            probes = sum(item["probe_count"] for item in runs)
            delays = [delay for item in runs for delay in item["detection_delay_ticks"]]
            probe_counts = [count for item in runs for count in item["outage_probe_counts"]]
            scenarios.append({
                "detector": detector_name,
                "probe_every_nth_transaction": probe_every,
                "outage_count": outages,
                "detected_outage_count": detected,
                "missed_outage_count": outages - detected,
                "detection_rate": detected / outages if outages else None,
                "false_alarm_count": false_alarms,
                "false_alarms_per_10000_probes": false_alarms / probes * 10_000 if probes else None,
                "median_detection_delay_ticks": float(np.median(delays)) if delays else None,
                "recent_window_probe_requirement": MINIMUM_PROBES[detector_name],
                "outages_below_recent_window_requirement": sum(
                    count < MINIMUM_PROBES[detector_name] for count in probe_counts),
                "median_probes_per_outage": float(np.median(probe_counts)) if probe_counts else None,
                "total_probes": probes,
            })
    return {"seed_start": seed_start, "seed_count": seeds, "horizon": horizon,
            "traffic_note": "Every n-th transaction per generated stream probes the next gateway for its bank; the detector only receives that probe's boolean outcome.",
            "scenarios": scenarios}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20_000)
    args = parser.parse_args()
    result = run(args.seeds, args.horizon, args.seed)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "outage_tradeoff.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
