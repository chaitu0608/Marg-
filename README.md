# Marg

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

Routing averages and 95% confidence intervals use 20 seeded worlds with 20,000 ticks per seed. Recovery is the median successful recovery time in affected-bank transactions; the last column reports never-recovered events explicitly.

A median of zero means the selected policy already sent at most 10% of the affected bank's recent traffic to the degraded gateway before the outage began. Those events required no adaptation.

| Strategy | Overall success mean [95% CI] | Cumulative expected regret mean [95% CI] | Outage success mean [95% CI] | Median recovery ticks; mean per-seed median [95% CI] | Never recovered / outages |
|---|---:|---:|---:|---:|---:|
| static | 0.8053 [0.7900, 0.8206] | 2480.5597 [2200.4429, 2760.6765] | 0.6555 [0.6101, 0.7008] | 0.0; 30.4750 [1.7250, 68.3500] | 57 / 221 |
| round_robin | 0.7982 [0.7903, 0.8061] | 2621.4170 [2488.7781, 2754.0559] | 0.6520 [0.6384, 0.6657] | n/a | 221 / 221 |
| epsilon_greedy | 0.8625 [0.8573, 0.8677] | 1327.4995 [1260.0480, 1394.9509] | 0.7665 [0.7457, 0.7874] | 0.0; 17.3250 [5.6000, 30.6750] | 23 / 221 |
| thompson_w200 | 0.8826 [0.8782, 0.8871] | 921.2982 [877.1637, 965.4327] | 0.8074 [0.7936, 0.8211] | 50.0; 53.2750 [40.8750, 65.9250] | 11 / 221 |
| oracle | 0.9284 [0.9247, 0.9321] | 0.0000 [0.0000, 0.0000] | 0.9124 [0.9050, 0.9198] | n/a | 0 / 221 |

Fraud dataset: **real Kaggle Credit Card Fraud Detection via Zenodo mirror** at `data/creditcard.csv` (284,807 rows, fraud rate 0.17275%). The split is time-ordered with sizes 199,368/42,718/42,721 for train/validation/test.

| Fraud test metric | Measured value |
|---|---:|
| PR-AUC | 0.760501 |
| Recall at 1% FPR | 0.826923 |
| Precision at validation block threshold | 0.165992 |
| Confusion matrix (TN, FP / FN, TP) | 42,463, 206 / 11, 41 |

Accuracy is not a headline metric because the legitimate class dominates; predicting every transaction as legitimate would appear accurate while missing fraud.

| Scoring service load test | Measured value |
|---|---:|
| Requests at concurrency | 10,000 at 32 |
| p50 / p95 / p99 latency | 32.373 / 123.681 / 195.922 ms |
| Throughput | 693.17 requests/s |
| p99 target (100 ms) | missed |
| Bottleneck | HTTP round-trip, event-loop scheduling, and JSON overhead dominate the tail. |

| End-to-end test-stream metric | Measured value |
|---|---:|
| Blocked fraud rate | 0.788462 |
| False block rate | 0.004828 |
| Routing success rate | 0.826270 |
| Review queue size | 347 |

The window sweep measured these seeded Thompson success rates: W=50: 0.9081, W=100: 0.9086, W=200: 0.9079, W=500: 0.9085, W=1000: 0.9087, W=5000: 0.9087. See the saved sweep JSON for the full values.

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
