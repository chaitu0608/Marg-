# Marg and Juspay interview alignment

## What the public sources support

Juspay’s current public backend role calls out intelligent traffic routing, self-healing systems, automatic anomaly detection, intelligent payment assistance, and AI-powered payment operations. That makes Marg’s routing, observed-outcome detector, fraud-serving boundary, and guarded investigator relevant discussion material. The role is broad: the same listing emphasizes payment business logic, API integrations, functional programming, reliability, and infrastructure, so Marg does not replace fundamentals or demonstrate those stacks by itself.

Juspay’s January 2025 engineering blog explains rule-based ordering and dynamic gateway ordering. It describes recent gateway performance, sliding windows, exploration to avoid starving gateways, and a feedback loop for downtime. The October 2025 arXiv paper describes delayed feedback, per-dimension gateway scores, exploration, downtime detection, cascading retries, and configuration experiments. Marg is a smaller simulator: a per-context sliding-window Thompson bandit with synthetic gateway outcomes. It does not reproduce Juspay’s scoring formula, feedback timing, eligibility checks, retry semantics, or production architecture.

The quoted wording about a specific internship JD requiring control-theoretic feedback loops, multi-arm RL bandits, explainable fraud scoring under 100 ms was not found on the official role page I checked. Treat that wording as a role-specific or historical copy until you can match it to the actual posting you are applying for. The official backend listing supports the general routing/anomaly-detection overlap, not that exact wording.

## How to pitch this project

Lead with the experiment and the boundary: “I built a reproducible simulator and tested adaptive routing under drift and outages. Every strategy saw the same seeded world and random draws. In my saved comparison, Thompson W=200 outperformed static routing overall and during outages; see the generated results table for the means and confidence intervals. The fraud model uses a chronological split on the public anonymized dataset. PR-AUC and recall at a fixed false-positive rate are in the same saved table. My local HTTP client p99 misses the target, so I report that plainly. This is a portfolio experiment, not production evidence.”

Then open the code for common random numbers, the router interface that cannot access true rates, the sliding-window update, time split, and threshold selection. Use the website’s charts as navigation, not as a substitute for the source and saved result files.

## Questions worth preparing

- **Why Thompson sampling?** Explain the Beta posterior and probability matching, then compare it to epsilon-greedy from the same run. Be explicit that UCB was not benchmarked.
- **What does W do?** Explain the noise/adaptation tradeoff and point to `results/window_sweep.json`. Acknowledge that the ranking depends on the simulator.
- **How do you know routing is not leaking?** Show that the router receives only context, time, selected gateway, and observed boolean outcome. The benchmark alone consults the simulator’s true rates.
- **How is the comparison fair?** Explain the shared world and precomputed per-transaction/per-gateway uniform draws. Strategies differ by choice, not by a luckier random stream.
- **How do you detect an outage without the schedule?** The detector reads observed outcomes; the schedule is used after alerting to score true detections, false alarms, and delay. The results page shows misses and the false-alarm/delay tradeoff.
- **Why a time split and PR-AUC?** Explain temporal drift and severe class imbalance. Do not present the accuracy of a majority-class classifier as evidence.
- **Why separate SHAP from `/score`?** Explain the hot-path latency boundary and deferred case investigation.
- **What does the fraud feature set mean?** V1–V28 are anonymized PCA features. The model demonstrates training/serving and threshold evaluation, not interpretable fraud behavior.
- **Why report a p99 miss?** Report client, ASGI, dispatch queue, and model timing separately. The saved local run misses its target; its residual estimate includes client/loopback/socket/response work. A remote generator has not been used.
- **Why add an on-chain hash receipt?** For optional tamper evidence over the exact artifact bytes, not as a payment feature. It proves identity of bytes only, not validity of metrics.

## Interview process claims

Juspay’s official public job page does not publish a fixed interview sequence. Candidate reports mention online coding assessments, hackathons, Tree of Space, concurrency, and DSA, but formats vary by role, campus, and hiring cycle. Verify the current process with the recruiter or role-specific notice. For preparation, the safe emphasis is data structures and algorithms, concurrency/thread safety, backend APIs, distributed systems, and the ability to explain your own design choices.

## Sources

- [Juspay careers: Software Development Engineer Backend](https://juspay.io/careers/DEV-BE01)
- [Juspay engineering blog: Orchestrator and Merchant controlled Routing Engine](https://juspay.io/blog/juspay-orchestrator-and-merchant-controlled-routing-engine)
- [Agrawal and Patil, A Control-Theoretic Approach to Dynamic Payment Routing for Success Rate Optimization (arXiv:2510.16735)](https://arxiv.org/abs/2510.16735)
- [Juspay careers index](https://juspay.io/careers)
- Candidate reports are anecdotal, including [Glassdoor internship interview reports](https://www.glassdoor.co.in/Interview/JUSPAY-Interview-Questions-E1102352.htm?filter.jobTitleExact=Software+Engineer%28Internship%29); they are not a guaranteed process description.
