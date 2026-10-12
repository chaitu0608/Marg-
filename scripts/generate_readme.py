from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text())


def interval(value: dict) -> str:
    return f"{value['mean']:.4f} [{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"


def render() -> str:
    routing = load("routing.json")
    fraud = load("fraud_metrics.json")
    latency = load("loadtest.json")
    pipeline = load("pipeline.json")
    outage = load("outage_detection.json")
    outage_tradeoff = load("outage_tradeoff.json")
    fraud_sensitivity = load("fraud_sensitivity.json")
    sweep = load("window_sweep.json")
    route_rows = []
    for name, values in routing["routing"].items():
        rec = values["median_recovery_ticks"]
        recovery_stats = values["mean_seed_median_recovery_ticks"]
        recovery = "n/a" if rec is None else f"{rec:.1f}; {interval(recovery_stats)}"
        route_rows.append(
            f"| {name} | {interval(values['success_rate'])} | "
            f"{interval(values['cumulative_regret'])} | {interval(values['outage_success_rate'])} | "
            f"{recovery} | {values['never_recovered']} / {values['outage_count']} |"
        )
    cm = fraud["confusion_matrix"]
    tradeoff_rows = "\n".join(
        f"| {row['detector']} | {row['probe_every_nth_transaction']} | "
        f"{row['detected_outage_count']} / {row['missed_outage_count']} of "
        f"{row['outage_count']} | {row['false_alarms_per_10000_probes']:.3f} | "
        f"{row['median_detection_delay_ticks']} | "
        f"{row['outages_below_recent_window_requirement']} / "
        f"{row['recent_window_probe_requirement']} |"
        for row in outage_tradeoff["scenarios"]
    )
    sensitivity_rows = []
    for cap in fraud_sensitivity["review_caps"]:
        values = []
        for cost in fraud_sensitivity["review_cost_amount_units_per_case"]:
            scenario = next(item for item in fraud_sensitivity["scenarios"]
                            if item["max_review_rate"] == cap
                            and item["review_cost_amount_units_per_case"] == cost)
            values.append(f"{scenario['newer_test_period']['total_cost_amount_units']:.1f}")
        sensitivity_rows.append(f"| {cap:.2%} | " + " | ".join(values) + " |")
    sensitivity_table = "\n".join(sensitivity_rows)
    return f"""# Marg

[![Marg CI](https://github.com/chaitu0608/Marg-/actions/workflows/ci.yml/badge.svg)](https://github.com/chaitu0608/Marg-/actions/workflows/ci.yml)

Marg is a small, reproducible simulation of two payment decisions: routing a transaction across drifting, outage-prone gateways, and scoring it for fraud with a gradient-boosted model. Gateway behavior and routing traffic are synthetic; fraud metrics below use the data source recorded in the saved result file.

The name is a sentimental nod to the Hindi word **मार्ग (mārg)**, meaning “path” or “way.” It fits a project about the path each payment takes—from scoring, through a routing choice, to an outcome that helps the system learn.

## Architecture

```text
Transaction -> Fraud score -> allow / review / block
                   |                  |
                   |                  +-> simulated router -> simulated gateway outcome
                   +-> separate SHAP explanation -> guarded investigator report
                    review -> investigator queue
```

The investigator only writes a case report after the scoring decision. It cannot modify the score, decision, or route.

## Results

Routing averages and 95% confidence intervals use {routing['seed_count']} seeded worlds with {routing['horizon']:,} ticks per seed. Recovery is the median successful recovery time in affected-bank transactions; the last column reports never-recovered events explicitly.

A median of zero means the selected policy already sent at most 10% of the affected bank's recent traffic to the degraded gateway before the outage began. Those events required no adaptation.

| Strategy | Overall success mean [95% CI] | Cumulative expected regret mean [95% CI] | Outage success mean [95% CI] | Median recovery ticks; mean per-seed median [95% CI] | Never recovered / outages |
|---|---:|---:|---:|---:|---:|
{chr(10).join(route_rows)}

Fraud dataset: **{fraud['dataset']}** at `{fraud['dataset_path']}` ({fraud['rows']:,} rows, fraud rate {fraud['fraud_rate']:.5%}). The split is time-ordered with sizes {fraud['split_sizes']['train']:,}/{fraud['split_sizes']['validation']:,}/{fraud['split_sizes']['test']:,} for train/validation/test.

| Fraud test metric | Measured value |
|---|---:|
| PR-AUC | {fraud['pr_auc']:.6f} |
| Recall at 1% FPR | {fraud['recall_at_1pct_fpr']:.6f} |
| Precision at validation block threshold | {fraud['precision_at_block_threshold']:.6f} |
| Confusion matrix (TN, FP / FN, TP) | {cm[0][0]:,}, {cm[0][1]:,} / {cm[1][0]:,}, {cm[1][1]:,} |

Thresholds are selected on validation data by minimizing missed-fraud Amount plus false-blocked Amount plus review count times the configured review cost, subject to a {fraud['cost_policy']['max_review_rate']:.1%} review cap. Amount values are dataset units; the source does not establish a currency or actual fraud loss. The model assumes manual review resolves a fraudulent transaction. On the newer test period, measured cost was {fraud['cost_policy']['newer_test_period']['total_cost_amount_units']:.3f} Amount units; review rate was {fraud['cost_policy']['newer_test_period']['review_rate']:.3%}.

Accuracy is not a headline metric because the legitimate class dominates; predicting every transaction as legitimate would appear accurate while missing fraud.

| Scoring service load test | Measured value |
|---|---:|
| Requests at concurrency | {latency['requests']:,} at {latency['concurrency']} |
| Uvicorn workers | {latency['server_workers']} |
| p50 / p95 / p99 latency | {latency['p50_ms']:.3f} / {latency['p95_ms']:.3f} / {latency['p99_ms']:.3f} ms |
| Throughput | {latency['throughput_rps']:.2f} requests/s |
| p99 target ({latency['target_p99_ms']:.0f} ms) | {'met' if latency['target_met'] else 'missed'} |
| Bottleneck | {latency['bottleneck']} |
| Measurement method | {latency['measurement']} |
| App processing p99 | {latency['app_processing_p99_ms']:.3f} ms |
| Uvicorn dispatch queue p99 | {latency['server_dispatch_queue_p99_ms']:.3f} ms |
| Client/loopback/response residual p99 (paired estimate) | {latency['client_loopback_and_response_p99_ms']:.3f} ms |
| Non-inference overhead p99 (paired request estimate) | {latency['non_inference_overhead_p99_ms']:.3f} ms |

| End-to-end test-stream metric | Measured value |
|---|---:|
| Blocked fraud rate | {pipeline['blocked_fraud_rate']:.6f} |
| False block rate | {pipeline['false_block_rate']:.6f} |
| Routing success rate | {pipeline['routing_success_rate']:.6f} |
| Review queue size | {pipeline['review_queue_count']:,} |

## Observed-outcome outage detector

The detector receives only outcomes from a balanced probe policy and compares the previous {outage['detector_parameters']['baseline_window']} observed outcomes with the latest {outage['detector_parameters']['detection_window']}. Across {outage['seed_count']} seeded synthetic worlds, it detected {outage['detected_outage_count']} of {outage['outage_count']} outages whose alert evidence overlapped the outage, missed {outage['missed_outage_count']}, and emitted {outage['false_alarm_count']} false alarms ({outage['false_alarms_per_10000_observed_outcomes']:.3f} per 10,000 observations). Median detection delay was {outage['median_detection_delay_ticks']} ticks; median delay to clear an alert after recovery was {outage['median_recovery_delay_ticks']} ticks. Median probes per outage were {outage['median_probe_outcomes_per_outage']:.1f}, and {outage['outages_with_fewer_than_detection_window_probes']} outages had fewer probes than the detector's recent window. Premature alert clears: {outage['premature_alert_clear_count']}. These results describe the synthetic probe setup, not production alert quality.

The window sweep measured these seeded Thompson success rates: {', '.join(f"W={row['window']}: {row['metrics']['success_rate']:.4f}" for row in sweep['thompson'])}. See the saved sweep JSON for the full values.

## Outage detector comparison

This comparison uses {outage_tradeoff['seed_count']} seeded synthetic worlds. Detectors see only observed outcomes. Probe cadence changes gateway coverage and detection opportunity. A run-length detector is included beside three drop-window configurations.

| Detector | Probe every nth transaction | Detected / missed outages | False alarms per 10,000 probes | Median detection delay (ticks) | Outages below recent-window probes / requirement |
|---|---:|---:|---:|---:|---:|
{tradeoff_rows}

All missed outages remain counted. Probe-rate scenarios are synthetic and are not production alert guarantees. Full values are in `results/outage_tradeoff.json`.

## Fraud cost sensitivity scenarios

Each validation-selected policy is evaluated on the later test period. The table varies review capacity and per-case review cost. Values are test-period cost proxy in dataset Amount units, not business-loss estimates; Amount's unit/currency and real fraud loss are unknown.

| Max review capacity | Review cost 0 | Review cost 0.5 | Review cost 1 | Review cost 5 |
|---:|---:|---:|---:|---:|
{sensitivity_table}

See [the sensitivity plot](results/fraud_sensitivity.png) and `results/fraud_sensitivity.json` for policy thresholds and cost breakdowns.

## Plots

- [Synthetic world rates](results/world.png)
- [Routing cumulative regret](results/routing_regret.png)
- [Routing success rate](results/routing_success.png)
- [Outage recovery distribution](results/routing_recovery.png)
- [Thompson window sweep](results/window_sweep.png)
- [Fraud precision-recall curve](results/fraud_pr_curve.png)
- [Fraud score distributions](results/fraud_scores.png)
- [Fraud policy sensitivity](results/fraud_sensitivity.png)

## Run it

Use Python 3.11; on macOS, install LightGBM's OpenMP runtime with `brew install libomp`. Install the project with `python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'`. Put the real `creditcard.csv` at `data/creditcard.csv`, then run `make test`, `make bench`, or `make verify`. `make serve` starts the API. To use an external, instrumented Marg server from another load-generator host, run `python -m bench.run_loadtest --url http://SERVER:8000`; the current recorded benchmark uses a local client process. `/score` uses model inference only; `/explain` computes SHAP separately. Run `pip install -e '.[mcp]'` and `make mcp` to expose the optional read-only Marg MCP server over stdio; see [docs/MCP.md](docs/MCP.md) for Codex configuration.

## Limitations

- Gateway behavior and generated routing traffic are synthetic, not production payment data.
- Gateway outages are simplified seeded intervals that affect one gateway and bank across methods.
- The fraud dataset uses anonymized PCA features and historical European card transactions; this does not establish performance on current or Marg-specific traffic.
- The fraud cost policy uses dataset Amount as a proxy and assumes manual review resolves fraud; its cost units are not verified currency or observed business losses.
- Local HTTP load-test latency includes serialization and loopback networking and does not represent a production deployment.
- In the saved load test, client-observed tail latency is measured under local 32-way load; Uvicorn dispatch queue measures header-parse-to-ASGI scheduling, while the client residual combines client scheduling, loopback/socket transport, and response decoding.
- SHAP explanations are separate post-score work and are not included in `/score` latency.
- Gateway dispatch is fail-closed and has no live adapter. Its idempotency memory is process-local; production requires durable idempotency, authorization, reconciliation, and operator controls.
- The optional MCP server exposes saved result files through a fixed allowlist. It cannot change decisions or dispatch payments.
- GitHub Actions runs unit tests, Ruff, generated-README consistency, and an MCP stdio smoke test; full `make verify` stays local because the real dataset is not stored in Git.

## Next improvements

- Validate fraud costs with real loss, false-decline, and review-capacity data, then evaluate on later time periods.
- Test outage detection against seasonal changes, sparse gateway traffic, and overlapping incidents; the measured probe traffic is balanced by construction.
- Use durable idempotency and audited operator approval before wiring a real gateway adapter.
- Profile production-like networking and serialization; local loopback p99 does not predict deployed tail latency.
- Before a real integration, move idempotency into durable storage with atomic reservation, provider-key propagation, reconciliation, audit logs, approval controls, circuit breakers, and tested rollback procedures.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    target = ROOT / "README.md"
    if args.check:
        if not target.exists() or target.read_text() != expected:
            raise SystemExit("README is stale; run `make docs` to regenerate it from results/*.json")
        print("README matches the saved result JSON files.")
    else:
        target.write_text(expected)
        print(f"Generated {target}")


if __name__ == "__main__":
    main()
