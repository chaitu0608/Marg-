import json

from marg.agent.investigator import Investigator
from marg.agent.llm import MockLLM, default_llm


def test_investigator_accepts_only_given_shap_factors():
    response = json.dumps({"summary": "Review needed", "risk_factors": ["V1", "secret"],
                           "confidence": 0.8, "recommended_action": "block"})
    llm = MockLLM(response)
    report = Investigator(llm).report({"id": 4}, 0.91, ["V1", "Amount"])
    assert report.risk_factors == ["V1", "Amount"]
    assert report.recommended_action == "review"
    assert llm.calls == 2


def test_investigator_valid_report_cannot_change_decision_input():
    response = json.dumps({"summary": "Looks risky", "risk_factors": ["V1"],
                           "confidence": 0.7, "recommended_action": "allow"})
    report = Investigator(MockLLM(response)).report({}, 0.95, ["V1"])
    assert report.recommended_action == "allow"
    assert report.risk_factors == ["V1"]


def test_no_key_selects_deterministic_mock(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(default_llm(), MockLLM)
