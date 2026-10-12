const $ = (selector) => document.querySelector(selector);
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const num = (value, digits = 1) => Number(value).toLocaleString("en-US", {
  minimumFractionDigits: digits,
  maximumFractionDigits: digits,
});

function renderRouting(routing) {
  $("#routing-kicker").textContent = `ROUTING / ${routing.seed_count} SEEDED WORLDS`;
  const strategies = [
    ["Static", "static"],
    ["Round robin", "round_robin"],
    ["ε-greedy", "epsilon_greedy"],
    ["Thompson · W=200", "thompson_w200"],
  ];
  const chart = $("#routing-chart");
  chart.innerHTML = strategies.map(([label, key]) => {
    const result = routing.routing[key];
    return `<div class="bar-row"><span class="bar-label">${label}</span><div class="bar-track"><div class="bar-fill" style="width:${result.success_rate.mean * 100}%"></div><div class="bar-fill outage" style="width:${result.outage_success_rate.mean * 100}%"></div></div><span class="bar-value">${pct(result.success_rate.mean)}</span></div>`;
  }).join("");
  const thompson = routing.routing.thompson_w200;
  const fixed = routing.routing.static;
  $("#hero-routing").textContent = pct(thompson.success_rate.mean, 2);
  $("#routing-note").textContent = `Across ${routing.seed_count} seeded worlds, outage success rose from ${pct(fixed.outage_success_rate.mean, 2)} with static routing to ${pct(thompson.outage_success_rate.mean, 2)} with Thompson W=200. Its mean per-seed median recovery was ${num(thompson.mean_seed_median_recovery_ticks.mean, 1)} ticks; ${thompson.never_recovered} outage windows never met the recovery rule.`;
}

function renderLatency(latency) {
  const rows = [
    ["Model inference", latency.inference_p99_ms],
    ["ASGI app", latency.app_processing_p99_ms],
    ["Dispatch queue", latency.server_dispatch_queue_p99_ms],
    ["Client residual", latency.client_loopback_and_response_p99_ms],
  ];
  const max = Math.max(...rows.map((row) => row[1]));
  $("#latency-number").textContent = num(latency.p99_ms, 3);
  $("#hero-latency").textContent = `${num(latency.p99_ms, 1)} ms`;
  $("#latency-chart").innerHTML = rows.map(([label, value]) => `<div class="latency-row"><span class="latency-label">${label}</span><div class="latency-track"><div class="latency-fill" style="width:${Math.max(1, value / max * 100)}%"></div></div><span class="latency-value">${num(value, 3)} ms</span></div>`).join("");
  $("#latency-note").textContent = `The 100 ms target was missed. ${num(latency.app_processing_p99_ms, 3)} ms p99 inside ASGI and ${num(latency.server_dispatch_queue_p99_ms, 3)} ms dispatch queue; the local client residual dominates. This is a same-machine test, not a remote load test.`;
}

function renderDetector(detector, tradeoff) {
  const fast = tradeoff.scenarios.find((item) => item.detector === "consecutive_5_failures" && item.probe_every_nth_transaction === 1);
  const balanced = tradeoff.scenarios.find((item) => item.detector === "drop_100_vs_25" && item.probe_every_nth_transaction === 1);
  const conservative = tradeoff.scenarios.find((item) => item.detector === "drop_200_vs_25" && item.probe_every_nth_transaction === 1);
  $("#detector-summary").innerHTML = `<div><strong>${detector.detected_outage_count}/${detector.outage_count}</strong><span>BASELINE DETECTED</span></div><div><strong>${detector.false_alarm_count}</strong><span>FALSE ALERTS</span></div><div><strong>${num(detector.median_detection_delay_ticks, 1)}</strong><span>MEDIAN TICKS</span></div>`;
  const maxRate = Math.max(fast.false_alarms_per_10000_probes, balanced.false_alarms_per_10000_probes, conservative.false_alarms_per_10000_probes);
  const rows = [["Fast · 5 consecutive", fast], ["Balanced · 100/25", balanced], ["Conservative · 200/25", conservative]];
  $("#detector-bars").innerHTML = rows.map(([label, item]) => `<div class="detector-row"><span class="detector-label"><strong>${label}</strong><small>${item.detected_outage_count}/${item.outage_count} detected · ${num(item.median_detection_delay_ticks, 0)} ticks</small></span><div class="detector-track"><div class="detector-fill" style="width:${item.false_alarms_per_10000_probes / maxRate * 100}%"></div></div><span class="detector-value">${num(item.false_alarms_per_10000_probes, 2)} / 10k</span></div>`).join("");
}

function renderFraud(fraud, sensitivity) {
  $("#hero-fraud").textContent = num(fraud.pr_auc, 4);
  $("#fraud-prauc").textContent = num(fraud.pr_auc, 4);
  $("#fraud-recall").textContent = pct(fraud.recall_at_1pct_fpr, 2);
  const capacity = $("#capacity-select");
  const cost = $("#cost-select");
  capacity.innerHTML = sensitivity.review_caps.map((value) => `<option value="${value}">${pct(value, 2)} max</option>`).join("");
  cost.innerHTML = sensitivity.review_cost_amount_units_per_case.map((value) => `<option value="${value}">${num(value, value % 1 ? 1 : 0)} units / case</option>`).join("");
  capacity.value = "0.01";
  cost.value = "1";
  const update = () => {
    const scenario = sensitivity.scenarios.find((item) => Number(item.max_review_rate) === Number(capacity.value) && Number(item.review_cost_amount_units_per_case) === Number(cost.value));
    if (!scenario) return;
    const test = scenario.newer_test_period;
    $("#scenario-result").innerHTML = `<div><strong>${num(test.total_cost_amount_units, 2)}</strong><span>TEST COST UNITS</span></div><div><strong>${test.review_count.toLocaleString()}</strong><span>REVIEWED</span></div><div><strong>${test.missed_fraud_count}</strong><span>MISSED FRAUD</span></div>`;
  };
  capacity.addEventListener("change", update);
  cost.addEventListener("change", update);
  update();
}

async function loadResults() {
  try {
    const response = await fetch("/api/results");
    if (!response.ok) throw new Error(`Results endpoint returned ${response.status}`);
    const results = await response.json();
    renderRouting(results.routing);
    renderLatency(results.latency);
    renderDetector(results.outage_detector, results.outage_tradeoff);
    renderFraud(results.fraud, results.fraud_sensitivity);
    if (results.attestation) {
      $("#web3-digest").textContent = `0x${results.attestation.result_digest}`;
      $("#web3-state").textContent = `LOCAL SHA-256 / ${results.attestation.source_files} SOURCE FILES / NO PUBLIC RECEIPT`;
    }
  } catch (error) {
    $("#routing-note").textContent = `Could not load saved results: ${error.message}`;
    $("#latency-note").textContent = "Saved benchmark results are unavailable from this backend.";
    $(".signal-status span").textContent = "RESULTS UNAVAILABLE";
    $(".signal-status i").style.background = "#df765e";
  }
}

const demoRequest = { features: Array(30).fill(0) };
let lastScore = null;

async function setApiStatus() {
  const status = $("#api-status");
  const badge = $(".api-status");
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("offline");
    status.textContent = "LOCAL API ONLINE";
    badge.classList.add("online");
    $("#api-dot").classList.add("online");
  } catch {
    status.textContent = "LOCAL API OFFLINE";
    badge.classList.add("offline");
  }
}

async function requestScore() {
  const button = $("#run-score");
  button.disabled = true;
  button.textContent = "Scoring zero vector…";
  const started = performance.now();
  try {
    const response = await fetch("/score", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(demoRequest) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Scoring failed (${response.status})`);
    lastScore = payload;
    $("#score-value").textContent = Number(payload.score).toFixed(6);
    $("#decision-value").textContent = payload.decision;
    $("#decision-value").style.color = payload.decision === "block" ? "#df8068" : payload.decision === "review" ? "#d4e451" : "#a9d2ad";
    $("#response-time").textContent = `${(performance.now() - started).toFixed(1)} ms CLIENT / ${Number(payload.inference_ms).toFixed(3)} ms MODEL`;
    $("#response-json").textContent = JSON.stringify(payload, null, 2);
    $("#run-explain").disabled = false;
  } catch (error) {
    $("#response-time").textContent = "REQUEST FAILED";
    $("#response-json").textContent = error.message;
  } finally {
    button.disabled = false;
    button.innerHTML = "Run the zero-vector demo <span>↗</span>";
  }
}

async function requestExplain() {
  const button = $("#run-explain");
  button.disabled = true;
  button.textContent = "Computing SHAP…";
  try {
    const response = await fetch("/explain", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(demoRequest) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Explanation failed (${response.status})`);
    $("#response-json").textContent = JSON.stringify({ score_response: lastScore, explanation: payload }, null, 2);
    $("#response-time").textContent = "SEPARATE /EXPLAIN REQUEST";
  } catch (error) {
    $("#response-json").textContent = error.message;
    $("#response-time").textContent = "EXPLANATION FAILED";
  } finally {
    button.disabled = false;
    button.innerHTML = "Request SHAP separately <span>↗</span>";
  }
}

async function requestSmokeCheck() {
  const button = $("#run-smoke");
  const status = $("#smoke-status");
  const checks = [];
  button.disabled = true;
  button.textContent = "Checking API routes…";
  status.textContent = "Running safe browser API checks";
  try {
    const health = await fetch("/health");
    if (!health.ok || (await health.json()).status !== "ok") throw new Error("GET /health failed");
    checks.push("GET /health: ok");

    const resultsResponse = await fetch("/api/results");
    if (!resultsResponse.ok) throw new Error(`GET /api/results returned ${resultsResponse.status}`);
    const results = await resultsResponse.json();
    if (!results.routing || !results.fraud || !results.latency) throw new Error("Saved results response is incomplete");
    checks.push("GET /api/results: saved benchmark sections present");

    const scoreResponse = await fetch("/score", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(demoRequest),
    });
    const score = await scoreResponse.json();
    if (!scoreResponse.ok || !["allow", "review", "block"].includes(score.decision)) {
      throw new Error(score.detail || "POST /score failed validation");
    }
    checks.push(`POST /score: ${score.decision}, ${Number(score.inference_ms).toFixed(3)} ms model inference`);

    const explainResponse = await fetch("/explain", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(demoRequest),
    });
    const explanation = await explainResponse.json();
    if (!explainResponse.ok || !Array.isArray(explanation.top_features)) {
      throw new Error(explanation.detail || "POST /explain returned no top_features list");
    }
    checks.push(`POST /explain: ${explanation.top_features.length} SHAP features returned`);
    status.textContent = "4/4 API CHECKS PASSED · not pytest";
    status.classList.add("smoke-ok");
    $("#response-json").textContent = JSON.stringify({ checks, score, explanation }, null, 2);
    $("#response-time").textContent = "4 ROUTES / BROWSER SMOKE CHECK";
  } catch (error) {
    checks.push(`FAILED: ${error.message}`);
    status.textContent = "API CHECK FAILED · see response";
    status.classList.remove("smoke-ok");
    $("#response-json").textContent = JSON.stringify({ checks }, null, 2);
    $("#response-time").textContent = "SMOKE CHECK FAILED";
  } finally {
    button.disabled = false;
    button.innerHTML = "Run backend API smoke check <span>✓</span>";
  }
}

$("#run-score").addEventListener("click", requestScore);
$("#run-explain").addEventListener("click", requestExplain);
$("#run-smoke").addEventListener("click", requestSmokeCheck);
loadResults();
setApiStatus();
