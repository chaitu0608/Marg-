# Marg

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
| Precision at validation block threshold | 1.000000 |
| Confusion matrix (TN, FP / FN, TP) | 42,669, 0 / 32, 20 |

Thresholds are selected on validation data by minimizing missed-fraud Amount plus false-blocked Amount plus review count times the configured review cost, subject to a 1.0% review cap. Amount values are dataset units; the source does not establish a currency or actual fraud loss. The model assumes manual review resolves a fraudulent transaction. On the newer test period, measured cost was 2527.200 Amount units; review rate was 0.386%.

Accuracy is not a headline metric because the legitimate class dominates; predicting every transaction as legitimate would appear accurate while missing fraud.

| Scoring service load test | Measured value |
|---|---:|
| Requests at concurrency | 10,000 at 32 |
| Uvicorn workers | 4 |
| p50 / p95 / p99 latency | 20.368 / 144.393 / 235.924 ms |
| Throughput | 756.25 requests/s |
| p99 target (100 ms) | missed |
| Bottleneck | The client-observed tail is dominated by time outside the ASGI app (loopback transport, socket queueing, and client scheduling); app processing is 2.657 ms p99. |
| Measurement method | fixed-worker HTTP clients with pre-encoded input; client latency includes response JSON decode; app timing spans request parse through response serialization |
| App processing p99 | 2.657 ms |
| Outside-app time p99 (paired estimate) | 234.831 ms |
| Non-inference overhead p99 (paired request estimate) | 235.703 ms |

| End-to-end test-stream metric | Measured value |
|---|---:|
| Blocked fraud rate | 0.384615 |
| False block rate | 0.000000 |
| Routing success rate | 0.830262 |
| Review queue size | 165 |

## Observed-outcome outage detector

The detector receives only outcomes from a balanced probe policy and compares the previous 100 observed outcomes with the latest 25. Across 20 seeded synthetic worlds, it detected 141 of 221 outages whose alert evidence overlapped the outage, missed 80, and emitted 143 false alarms (3.575 per 10,000 observations). Median detection delay was 318.5 ticks; median delay to clear an alert after recovery was 265.5 ticks. Median probes per outage were 32.0, and 86 outages had fewer probes than the detector's recent window. Premature alert clears: 6. These results describe the synthetic probe setup, not production alert quality.

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
- The fraud cost policy uses dataset Amount as a proxy and assumes manual review resolves fraud; its cost units are not verified currency or observed business losses.
- Local HTTP load-test latency includes serialization and loopback networking and does not represent a production deployment.
- In the saved load test, client-observed tail latency is measured under local 32-way load; app timing excludes time waiting to enter the ASGI app and socket delivery.
- SHAP explanations are separate post-score work and are not included in `/score` latency.
- Gateway dispatch is fail-closed and has no live adapter. Its idempotency memory is process-local; production requires durable idempotency, authorization, reconciliation, and operator controls.

## Next improvements

- Validate fraud costs with real loss, false-decline, and review-capacity data, then evaluate on later time periods.
- Test outage detection against seasonal changes, sparse gateway traffic, and overlapping incidents; the measured probe traffic is balanced by construction.
- Use durable idempotency and audited operator approval before wiring a real gateway adapter.
- Profile production-like networking and serialization; local loopback p99 does not predict deployed tail latency.
