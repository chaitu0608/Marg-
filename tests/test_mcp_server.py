import pytest

import marg.mcp_server as server


def test_mcp_result_access_is_limited_to_named_artifacts(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "routing.json").write_text('{"seed_count":20}')
    monkeypatch.setattr(server, "ROOT", tmp_path)
    assert server.read_result("routing")["seed_count"] == 20
    with pytest.raises(ValueError, match="unknown Marg result"):
        server.read_result("../../README.md")


def test_mcp_cost_scenario_matches_a_saved_policy_scenario(monkeypatch):
    row = {"max_review_rate": 0.01, "review_cost_amount_units_per_case": 1.0}
    result_data = {"scenarios": [row], "scenario_note": "unit, currency unknown"}
    monkeypatch.setattr(server, "read_result", lambda _: result_data)
    result = server.closest_fraud_scenario(0.01, 1.0)
    saved = result["matched_saved_scenario"]
    assert saved["max_review_rate"] == 0.01
    assert saved["review_cost_amount_units_per_case"] == 1.0
    assert "unit, currency" in result["scenario_note"]
