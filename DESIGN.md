# Marg design decisions

## Routing and exploration

**Why Thompson sampling over UCB and epsilon-greedy?** Each gateway's recent binary outcomes form a Beta posterior. Sampling one plausible success rate per gateway gives a compact exploration rule: uncertain gateways sometimes win the sample, while gateways with consistent successes usually win. Epsilon-greedy instead explores at a fixed rate even after confidence is high; UCB was not implemented, so this project makes no measured comparison against UCB.

**How does window size trade adaptation speed against noise?** A short window forgets stale outcomes quickly after drift or an outage, but its estimate moves more with random outcome noise. A long window is steadier and slower to react. `results/window_sweep.json` and its plot show the seeded sweep produced by `bench.run_window_sweep`; it is one reproducible run, not a universal setting.

**How is cold start handled for a new gateway?** The posterior begins at Beta(1, 1), so all gateways begin with the same uniform prior. Thompson samples naturally try gateways with little evidence. There is no hidden true-rate initialization.

**How would you distinguish an outage from noise?** The simulation supplies a known outage schedule only to evaluation. A production detector would need a separate sequential test or change-point detector over observed declines, with persistence, traffic-volume checks, and a false-alarm budget. Marg does not claim to solve that production detection problem.

Marg now includes `ObservedOutageDetector`, which compares a 100-outcome baseline with the latest 25 observed outcomes for a bank/gateway pair. It signals a drop only when baseline success is at least 75% and recent success is at most 50%; it clears after recent success rises above 50%. These initial cutoffs are simple simulation parameters, not production-calibrated probabilities. The benchmark uses a balanced probe policy and checks alerts against the schedule only after the detector emits them.

`bench/run_outage_tradeoff.py` compares this detector family at 50/10, 100/25, and 200/25 windows plus a five-consecutive-failures baseline. It also probes at every transaction or every second/fourth transaction. The evaluator reports detected and missed outages, false alarms per probe volume, detection delay, and low-probe coverage. Probe cadence trades off observed-outcome coverage and detector cost; it does not establish the best production rate. The 200/25 setup's minimum baseline success is 80%, while the other window configurations use 75%.

## Fraud and explanations

**Why is SHAP kept out of `/score`?** Model inference returns a score and threshold decision. SHAP is much slower and runs only when `/explain` is called, so its work cannot increase the `/score` path's measured latency.

**Why a time-based split? Why PR-AUC instead of accuracy?** Rows are sorted by `Time` and divided in chronological order to expose drift between training and evaluation. With a rare positive class, accuracy can look high when a classifier misses almost every fraud. PR-AUC focuses on precision and recall for the positive class; recall at a fixed false-positive rate gives a threshold-oriented companion metric.

**How are thresholds selected?** A deterministic validation grid chooses the review/block thresholds minimizing missed-fraud Amount plus false-blocked Amount plus a configured per-review cost, subject to a maximum review fraction. Values use the source dataset's Amount units, whose currency and relation to actual loss are unknown. The policy assumes review resolves fraud. The latest chronological test split evaluates the fixed thresholds without retuning.

**What does cost-aware mean here?** It is a transparent scenario using transaction Amount as a proxy for fraud loss and false-decline cost, with a configured review cost and capacity. It is not a measured business cost model; production use would require agreed loss, customer-friction, and review-capacity inputs.

## Scale and the investigator

**What would change at 350M transactions per day?** I would profile the actual service, then reduce feature construction and serialization work, cache immutable lookup features, avoid unnecessary network hops in the hot path, batch model inference where latency permits, and evaluate a smaller model. The local benchmark uses a fixed set of concurrent HTTP workers and cannot forecast production scale.

**Why is the LLM kept out of the decision path?** Fraud model thresholds decide allow/review/block, and the router chooses the gateway. The investigator runs only after the decision for a case report. Its output is validated, its factors must be in the supplied SHAP list, and a failed response falls back to a fixed report.

**What prevents accidental live gateway calls?** `GatewayDispatchGuard` is disabled by default. Even if explicitly enabled, it requires an adapter, nonempty gateway allowlist, finite positive amount cap, an `allow` decision, an amount within the cap, and unique transaction and idempotency identifiers. Its deduplication store is process-local, so a production adapter must replace that boundary with durable deduplication, approval, reconciliation, and audit controls.

The instrumented server adds a narrow queue measurement in `src/marg/fraud/server.py:TimedRequestResponseCycle`: elapsed time between Uvicorn parsing request headers and the scheduled ASGI coroutine beginning. `AppTimingMiddleware` measures ASGI processing through response serialization. Subtracting both from paired client latency gives a residual that still combines client scheduling, socket/loopback transport, and response JSON decoding. No separate remote load-generator host was available for the saved run; `bench.run_loadtest --url` can drive an instrumented endpoint remotely.

**How does fraud policy change with assumptions?** `bench/run_fraud_sensitivity.py:evaluate_scenarios` reuses the saved model, selects each scenario's thresholds on validation, and evaluates them on the later test split. The table varies review capacity and per-case review cost. Dataset Amount is only a proxy: its currency and relationship to actual loss are unknown, so these are scenarios rather than business estimates.

## DevOps and MCP

`.github/workflows/ci.yml` runs tests, Ruff, and a check that the README matches committed JSON results. It does not run the full data-dependent benchmark because the real credit-card dataset is ignored by Git; run `make verify` locally after providing it. This gives code changes a fast gate without copying local data into CI.

The optional `marg.mcp_server` exposes fixed-name result files, saved fraud sensitivity cases, and guardrail descriptions over MCP stdio. It is read-only and cannot execute shell commands, accept arbitrary file paths, change decisions, or call a gateway. This lets a compatible agent inspect Marg evidence while keeping the fraud model and router as decision owners. Install with `pip install -e '.[mcp]'`, then run `make mcp`.

The project now has a small static site served by the existing FastAPI app. It reads a fixed allowlist of saved JSON results and exposes a clearly labeled all-zero API demo. It adds no frontend framework or scoring logic. The site explains the experiment and its limitations; it is not an operator dashboard. The optional EVM registry only attests to hashes of source and result files and never influences payments or model decisions.

## Ambiguity

The routing evaluation uses 20,000 ticks per seed to keep the required 20-seed comparison practical; `make verify` records that horizon in `results/routing.json`.
