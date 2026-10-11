# Marg design decisions

## Routing and exploration

**Why Thompson sampling over UCB and epsilon-greedy?** Each gateway's recent binary outcomes form a Beta posterior. Sampling one plausible success rate per gateway gives a compact exploration rule: uncertain gateways sometimes win the sample, while gateways with consistent successes usually win. Epsilon-greedy instead explores at a fixed rate even after confidence is high; UCB was not implemented, so this project makes no measured comparison against UCB.

**How does window size trade adaptation speed against noise?** A short window forgets stale outcomes quickly after drift or an outage, but its estimate moves more with random outcome noise. A long window is steadier and slower to react. `results/window_sweep.json` and its plot show the seeded sweep produced by `bench.run_window_sweep`; it is one reproducible run, not a universal setting.

**How is cold start handled for a new gateway?** The posterior begins at Beta(1, 1), so all gateways begin with the same uniform prior. Thompson samples naturally try gateways with little evidence. There is no hidden true-rate initialization.

**How would you distinguish an outage from noise?** The simulation supplies a known outage schedule only to evaluation. A production detector would need a separate sequential test or change-point detector over observed declines, with persistence, traffic-volume checks, and a false-alarm budget. Marg does not claim to solve that production detection problem.

## Fraud and explanations

**Why is SHAP kept out of `/score`?** Model inference returns a score and threshold decision. SHAP is much slower and runs only when `/explain` is called, so its work cannot increase the `/score` path's measured latency.

**Why a time-based split? Why PR-AUC instead of accuracy?** Rows are sorted by `Time` and divided in chronological order to expose drift between training and evaluation. With a rare positive class, accuracy can look high when a classifier misses almost every fraud. PR-AUC focuses on precision and recall for the positive class; recall at a fixed false-positive rate gives a threshold-oriented companion metric.

**How are thresholds selected?** The block threshold is chosen on validation scores at a target false-positive rate. The review threshold is chosen at a looser target, making an intermediate zone. Test metrics are computed after thresholds are fixed.

## Scale and the investigator

**What would change at 350M transactions per day?** I would profile the actual service, then reduce feature construction and serialization work, cache immutable lookup features, avoid unnecessary network hops in the hot path, batch model inference where latency permits, and evaluate a smaller model. The current local benchmark cannot forecast that scale.

**Why is the LLM kept out of the decision path?** Fraud model thresholds decide allow/review/block, and the router chooses the gateway. The investigator runs only after the decision for a case report. Its output is validated, its factors must be in the supplied SHAP list, and a failed response falls back to a fixed report.

## Ambiguity

The routing evaluation uses 20,000 ticks per seed to keep the required 20-seed comparison practical; `make verify` records that horizon in `results/routing.json`.
