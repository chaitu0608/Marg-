# Marg interview walkthrough

This guide describes the code as implemented. The routing world and traffic are synthetic. Fraud metrics are only meaningful for the dataset path and saved model run recorded in `results/fraud_metrics.json`.

## 1. World simulator and common random numbers

`src/marg/sim/world.py:make_world` samples a baseline rate for every gateway/context pair, adds a bounded per-tick random walk, creates outage intervals, and precomputes one uniform draw per tick and gateway. `World.p_true` exposes true rates to the benchmark, while routing modules receive only the context, tick, and observed binary outcome through the `Router` protocol in `src/marg/routing/base.py`.

The benchmark computes an outcome as the precomputed uniform draw being less than the selected gateway's true probability. It uses the same world, transaction contexts, and uniform array for every strategy. That common-random-numbers design makes strategy differences come from their routing choices instead of independent luck. The uniforms are keyed by transaction and gateway, so a policy can choose a different gateway and still see that gateway's fixed draw for that transaction.

**Edge cases:** Outage intervals are end-exclusive and may overlap; overlapping events multiply their outage factors. The raw random walk is clipped, while an outage's separate multiplier can take the effective probability below the raw lower bound. A query outside the horizon clamps to its nearest endpoint.

## 2. Sliding-window Thompson update

`src/marg/routing/thompson.py:SlidingWindowThompsonRouter.select` samples from `Beta(1 + wins, 1 + losses)` for each gateway in the requested context and picks the highest sampled rate. `update` appends only the selected gateway's observed success or failure to its bounded deque. A new observation evicts the oldest one once that gateway/context window is full.

The Beta posterior represents uncertainty about a binary success probability. A poorly observed gateway has a broad posterior, so it sometimes produces a high sample and gets explored. As evidence grows, the posterior narrows around the observed rate and the better gateway is selected more often. A fixed-size deque makes old observations stop influencing the current posterior after enough new outcomes.

**Edge cases:** Every gateway starts with a uniform Beta(1,1) prior, so the policy needs no true rate at cold start. Gateways with no observations remain selectable. `window=1` keeps only the latest outcome; the implementation expects a positive window.

## 3. Cumulative regret and recovery detection

`bench/run_routing.py:run_one` computes per-tick expected regret as the best true rate for that context minus the selected gateway's true rate, then sums it. This uses expected rates rather than realized binary outcomes, which reduces outcome noise in the regret metric. The oracle is used only for evaluation.

`bench/run_routing.py:_recovery_times` scans affected-bank transactions after each outage starts. It records the first point where the last 50 such transactions include at most five choices of the degraded gateway. If that condition never appears before the world ends, the event is retained as `null` and counted under `never_recovered`; it is not silently dropped from the failure count.

The separate `src/marg/routing/outage.py:ObservedOutageDetector` compares a baseline deque with a recent deque for each bank/gateway pair. Its `observe` method accepts only one observed boolean result and a key/tick. `bench/run_outage_detector.py` chooses balanced gateway probes, simulates the selected outcomes outside the detector, and only then compares emitted alerts with the hidden schedule to score false alarms, detection delay, and alert-clear delay.

**Edge cases:** Recovery is measured in affected-bank transactions, not wall-clock ticks. A gateway can already be used rarely at outage onset; the earliest eligible recovery still requires a full 50 affected-bank observations. Very late outages can be censored by the end of the simulated horizon.

## 4. Fraud time split, thresholds, and PR-AUC

`src/marg/fraud/data.py:time_split` sorts rows by `Time`, then puts the earliest portion in training, the next portion in validation, and the latest portion in test. `src/marg/fraud/train.py:train` fits LightGBM with a positive-class weight computed from training labels. `src/marg/fraud/thresholds.py:select_cost_thresholds` searches a deterministic validation score grid, minimizing missed-fraud Amount plus false-blocked Amount plus configured review cost, subject to a review-rate cap. It evaluates this fixed policy on the strictly newer test period and saves the breakdown to `results/fraud_metrics.json`.

This is an explicit scenario, not known business economics: Kaggle's Amount unit/currency is not specified for this project, and the review model assumes manual review resolves fraud. The maximum review fraction and review cost are configurable CLI arguments; actual deployment thresholds need validated loss and operations costs.

Accuracy is misleading for a rare class: a classifier can predict almost everything as legitimate and still score high accuracy while missing fraud. PR-AUC reports the precision/recall tradeoff for the positive class. Recall at a fixed false-positive rate reports a concrete operating point. The test split remains chronologically later than the training split to expose temporal shift.

**Edge cases:** A real input must have the expected columns and approximately the expected row count. Threshold selection is discrete, so a threshold may land below the requested false-positive rate. Tied timestamps stay in stable row order; the split is by ordered row position rather than requiring every duplicate timestamp to stay together.

## 5. SHAP, `/score`, and latency

`src/marg/fraud/serve.py:create_app` loads the model and thresholds at startup. `/score` converts the 30 numeric values to a NumPy row and performs model inference and threshold selection. It does not instantiate the explainer. `/explain` lazily creates the SHAP tree explainer and returns the largest absolute contributions.

Keeping explanations off `/score` avoids making every allow/review/block request pay SHAP's compute cost. `/score` is async, the loaded single-row LightGBM model is set to one native thread, and `make serve` starts four Uvicorn worker processes. `bench/run_loadtest.py` uses a fixed number of request workers instead of creating thousands of queued client tasks. `src/marg/fraud/server.py:TimedRequestResponseCycle` measures Uvicorn header-parse-to-ASGI scheduling delay; the ASGI middleware measures app work; client time minus both is a paired residual containing client scheduling, loopback/socket transport, and response decoding. This is a local client run; `--url` supports a separately hosted instrumented server.

**Edge cases:** The API validates the feature count before prediction. If the saved model or threshold file is missing, service startup fails rather than inventing a fallback model. SHAP output has to be requested explicitly and may take longer than scoring.

## 6. Investigator guardrails

`src/marg/agent/investigator.py:Investigator.report` builds a prompt from the transaction, model score, and supplied SHAP factors. It validates the returned JSON as `CaseReport`, then checks that every risk factor is in the supplied list. It retries once after invalid output and then returns a templated review report. Inputs, model output, validation failures, and fallback are logged.

The investigator has no API to change a model score, a routing choice, or an allow/review/block result. `MockLLM` makes tests deterministic and keeps tests offline; `default_llm` selects it when no API key exists and an environment-keyed OpenAI adapter otherwise. The pipeline records review cases in a queue and does not call any LLM.

**Edge cases:** Malformed JSON, an unsupported action, an out-of-range confidence, or an invented factor is rejected. The fallback lists only factors already provided by SHAP and recommends review.

## 7. Gateway dispatch safeguards

`src/marg/gateway/safety.py:GatewayDispatchGuard` is a fail-closed adapter boundary, not a real payment integration. Live dispatch is disabled by default. Explicit opt-in requires a configured adapter, nonempty gateway allowlist, and finite positive amount cap; each request must also have an allow decision, a finite positive amount within the cap, and unique transaction and idempotency identifiers.

**Edge cases:** Idempotency keys are recorded before the adapter call, so an ambiguous timeout cannot be retried through the same guard instance. The in-memory set does not survive process restarts and is not enough for real payments; a production adapter needs durable idempotency, audited operator approval, reconciliation, and operational rollback controls.

`tests/test_gateway_safety.py` uses fake adapters to prove a timeout is surfaced, a second guard-level submission is rejected, rejected fraud decisions never reach an adapter, and an adapter may retry an ambiguous request with the same provider idempotency key. The retry test models provider-side deduplication; it does not add retries to the production guard.

## 8. MCP and CI

`src/marg/mcp_server.py:create_server` registers read-only MCP tools for a fixed allowlist of committed result files, saved fraud-policy scenarios, and Marg's decision boundaries. `read_result` rejects arbitrary paths. The tools cannot execute shell commands, modify result files, score transactions, route payments, or call the gateway adapter. An MCP-capable client can use them to inspect evidence and draft an explanation; the investigator remains post-decision and the model/router remain decision owners.

`.github/workflows/ci.yml` runs pytest, Ruff, and generated-README consistency checks on push and pull request. It does not download the credit-card dataset or rerun the full benchmark. The dataset is local/ignored, so full regeneration stays in `make verify` for an environment with that file.

**Edge cases:** The MCP extra is optional. Without it, the package's pure result-reader helpers still work, while starting the stdio server gives a clear install instruction. A missing result artifact returns an error instead of fabricating values.

## Likely interview questions

1. **What problem does Marg demonstrate?** Two sequential payment decisions: fraud scoring and gateway routing, evaluated with synthetic routing conditions and the dataset named in the fraud result file.
2. **Why simulate gateway behavior?** It makes drift and outages reproducible while keeping the project independent of real payment integrations.
3. **How is leakage prevented?** Router methods accept only context, tick, gateway, and observed success; only benchmark evaluation code reads true rates.
4. **What are common random numbers?** All policies share the same world and transaction/gateway uniform draws so their outcomes are paired fairly.
5. **What does expected regret measure?** The sum of the best gateway's true probability minus the chosen gateway's true probability at each tick.
6. **Why use Thompson sampling?** Posterior sampling explores uncertain choices while favoring arms with strong recent evidence.
7. **What does the window do?** It bounds how much old feedback can affect the current posterior, which lets the policy respond to changing rates.
8. **What does a longer window cost?** It reduces sampling noise but can make adaptation slower; use the saved sweep rather than claiming one window is universally best.
9. **How does cold start work?** Each gateway begins from the same Beta(1,1) prior.
10. **How does observed-outcome outage detection avoid leakage?** The detector receives binary outcomes only; the benchmark uses rates to generate the selected probe's result and the hidden schedule later to score alerts.
11. **What is the outage success metric?** The realized success fraction among contexts in a bank during that bank's outage intervals.
12. **How is recovery defined?** Within the affected bank's stream, the rolling 50-choice share to the affected gateway must be at most 10%; never-recovered events are counted.
13. **Why a time split?** It trains on earlier transactions and evaluates on later ones, making temporal drift visible.
14. **Why PR-AUC instead of accuracy?** Fraud is rare, so accuracy can hide poor detection of the positive class.
15. **How are review and block thresholds chosen?** On a validation-only grid minimizing Amount-weighted cost under a review cap, then evaluated on a newer period. Amount is a proxy; currency, actual fraud loss, and review capacity are unknown.
16. **Why is SHAP separate?** Explanation computation is slower and should not be part of every score request.
17. **What does the load test measure?** Fixed-worker client-observed local HTTP latency and throughput. Inference is reported separately; the difference includes HTTP, JSON, queueing, loopback, and response decode, not production network latency.
18. **Can the LLM block a transaction?** No. It produces a validated post-decision report; the scoring model and threshold own the decision.
19. **What happens when the investigator invents a risk factor?** Validation rejects the response, retries once, then returns a constrained review template.
20. **What would you change before real deployment?** Validate temporal drift and label delay, replace Amount proxies with agreed loss and review costs, test outage detection under production-like traffic, profile the deployment topology, and require durable idempotency plus operator controls before a real gateway adapter. MCP remains read-only evidence access, not an operational control plane.

## What I would do differently with real data

I would separate feature and label timestamps, account for delayed fraud labels, test on later live-like periods, calibrate thresholds against an agreed financial cost, monitor drift and gateway health independently, and evaluate privacy-safe explanations with investigators. I would also benchmark the service in its actual deployment topology before claiming a latency target.
