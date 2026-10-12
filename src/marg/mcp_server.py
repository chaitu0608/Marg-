"""Read-only MCP tools for an agent to inspect Marg's saved evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
RESULT_FILES = {
    "routing": "routing.json",
    "fraud": "fraud_metrics.json",
    "latency": "loadtest.json",
    "outage": "outage_detection.json",
    "outage_tradeoff": "outage_tradeoff.json",
    "fraud_sensitivity": "fraud_sensitivity.json",
    "pipeline": "pipeline.json",
}


def read_result(name: str) -> dict:
    """Read one named artifact; callers cannot provide filesystem paths."""
    if name not in RESULT_FILES:
        raise ValueError(f"unknown Marg result: {name}")
    path = ROOT / "results" / RESULT_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} is not generated yet; run `make bench`")
    return json.loads(path.read_text())


def closest_fraud_scenario(review_cap: float, review_cost: float) -> dict:
    """Return nearest saved cost-sensitivity case without recalculating policy."""
    result = read_result("fraud_sensitivity")
    scenarios = result["scenarios"]
    row = min(scenarios, key=lambda item: (
        abs(item["max_review_rate"] - review_cap)
        + abs(item["review_cost_amount_units_per_case"] - review_cost)))
    return {"requested_review_cap": review_cap,
            "requested_review_cost_amount_units_per_case": review_cost,
            "matched_saved_scenario": row,
            "scenario_note": result["scenario_note"]}


def create_server():
    """Build the optional MCP SDK server only when the user installs the MCP extra."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install Marg's optional MCP tools with `pip install -e '.[mcp]'`") from exc

    server = FastMCP("Marg read-only analyst")

    @server.tool()
    def get_marg_result(name: Literal[
            "routing", "fraud", "latency", "outage", "outage_tradeoff",
            "fraud_sensitivity", "pipeline"]) -> dict:
        """Read one saved Marg JSON result by its allowlisted name."""
        return read_result(name)

    @server.tool()
    def compare_fraud_policy(review_cap: float, review_cost_amount_units_per_case: float) -> dict:
        """Inspect the nearest saved validation-selected fraud policy scenario."""
        return closest_fraud_scenario(review_cap, review_cost_amount_units_per_case)

    @server.tool()
    def marg_guardrails() -> dict[str, str]:
        """Describe the project boundary between models, investigator, and gateway adapter."""
        return {
            "fraud_decision": "Model thresholds decide allow, review, or block.",
            "routing_decision": "The router selects a simulated gateway from observed outcomes.",
            "investigator": "The investigator writes a validated post-decision case report only.",
            "gateway": "Dispatch is disabled by default; current adapter guard uses in-memory deduplication.",
            "mcp": "These tools read saved result files only and cannot score, route, or dispatch payments.",
        }

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
