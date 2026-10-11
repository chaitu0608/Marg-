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
    return f"""# Marg

Marg is a small, reproducible simulation of two payment decisions: routing a transaction across drifting, outage-prone gateways, and scoring it for fraud with a gradient-boosted model. Gateway behavior and routing traffic are synthetic; fraud metrics below use the data source recorded in the saved result file.

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

Accuracy is not a headline metric because the legitimate class dominates; predicting every transaction as legitimate would appear accurate while missing fraud.

| Scoring service load test | Measured value |
|---|---:|
| Requests at concurrency | {latency['requests']:,} at {latency['concurrency']} |
| p50 / p95 / p99 latency | {latency['p50_ms']:.3f} / {latency['p95_ms']:.3f} / {latency['p99_ms']:.3f} ms |
| Throughput | {latency['throughput_rps']:.2f} requests/s |
| p99 target ({latency['target_p99_ms']:.0f} ms) | {'met' if latency['target_met'] else 'missed'} |
| Bottleneck | {latency['bottleneck']} |

| End-to-end test-stream metric | Measured value |
|---|---:|
| Blocked fraud rate | {pipeline['blocked_fraud_rate']:.6f} |
| False block rate | {pipeline['false_block_rate']:.6f} |
| Routing success rate | {pipeline['routing_success_rate']:.6f} |
| Review queue size | {pipeline['review_queue_count']:,} |

The window sweep measured these seeded Thompson success rates: {', '.join(f"W={row['window']}: {row['metrics']['success_rate']:.4f}" for row in sweep['thompson'])}. See the saved sweep JSON for the full values.

## Plots

- [Synthetic world rates](results/world.png)
- [Routing cumulative regret](results/routing_regret.png)
- [Routing success rate](results/routing_success.png)
- [Outage recovery distribution](results/routing_recovery.png)
- [Thompson window sweep](results/window_sweep.png)
- [Fraud precision-recall curve](results/fraud_pr_curve.png)
- [Fraud score distributions](results/fraud_scores.png)

## Run it

Use Python 3.11; on macOS, install LightGBM's OpenMP runtime with `brew install libomp`. Install the project with `python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'`. Put the real `creditcard.csv` at `data/creditcard.csv`, then run `make test`, `make bench`, or `make verify`. `make serve` starts the API. `/score` uses model inference only; `/explain` computes SHAP separately.

## Limitations

- Gateway behavior and generated routing traffic are synthetic, not production payment data.
- Gateway outages are simplified seeded intervals that affect one gateway and bank across methods.
- The fraud dataset uses anonymized PCA features and historical European card transactions; this does not establish performance on current or Marg-specific traffic.
- Local HTTP load-test latency includes serialization and loopback networking and does not represent a production deployment.
- SHAP explanations are separate post-score work and are not included in `/score` latency.
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
