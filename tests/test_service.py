import json

import numpy as np
from fastapi.testclient import TestClient

from marg.fraud.serve import create_app


class TinyModel:
    def predict_proba(self, row):
        score = min(max(float(row[0, 0]), 0), 1)
        return np.array([[1 - score, score]])


def test_score_returns_decision_without_explanation(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"block": 0.8, "review": 0.4}))
    monkeypatch.setattr("marg.fraud.serve.joblib.load", lambda _: TinyModel())
    app = create_app(str(model_path), str(thresholds))
    with TestClient(app) as client:
        body = {"features": [0.9] + [0.0] * 29}
        response = client.post("/score", json=body)
        assert response.status_code == 200
        assert response.json()["decision"] == "block"
        assert response.json()["inference_ms"] >= 0.0
        assert float(response.headers["x-marg-app-ms"]) >= response.json()["inference_ms"]
        assert app.state.explainer is None


def test_service_rejects_wrong_feature_count(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text('{"block": 0.8, "review": 0.4}')
    monkeypatch.setattr("marg.fraud.serve.joblib.load", lambda _: TinyModel())
    with TestClient(create_app(str(model_path), str(thresholds))) as client:
        assert client.post("/score", json={"features": [0.1]}).status_code == 422


def test_explain_is_separate_and_returns_feature_contributions(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text('{"block": 0.8, "review": 0.4}')
    monkeypatch.setattr("marg.fraud.serve.joblib.load", lambda _: TinyModel())

    class FakeExplainer:
        def __init__(self, path):
            pass

        def top_features(self, row, top_k):
            return [{"feature": "V1", "contribution": 0.7}]

    monkeypatch.setattr("marg.fraud.serve.Explainer", FakeExplainer)
    with TestClient(create_app(str(model_path), str(thresholds))) as client:
        result = client.post("/explain", json={"features": [0.1] * 30}).json()
        assert result["top_features"][0]["feature"] == "V1"


def test_portfolio_serves_page_and_saved_results(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text('{"block": 0.8, "review": 0.4}')
    monkeypatch.setattr("marg.fraud.serve.joblib.load", lambda _: TinyModel())

    # These fixture JSON files are for the API test only; they are not benchmark results.
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    fixture_results = {
        "routing": {"seed_count": 1},
        "fraud_metrics": {"dataset_path": "test-fixture-only"},
        "loadtest": {"target_met": False},
        "outage_detection": {},
        "outage_tradeoff": {},
        "fraud_sensitivity": {},
        "pipeline": {},
    }
    for filename, payload in fixture_results.items():
        (result_dir / f"{filename}.json").write_text(json.dumps(payload))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "marg-attestation.json").write_text('{"result_digest": "test-digest"}')
    monkeypatch.setattr("marg.fraud.serve.PROJECT_ROOT", tmp_path)

    with TestClient(create_app(str(model_path), str(thresholds))) as client:
        page = client.get("/")
        results = client.get("/api/results")
        css = client.get("/assets/site.css")
        javascript = client.get("/assets/site.js")

    assert page.status_code == 200
    assert "Make every" in page.text
    assert "Mārg (मार्ग · maarg)" in page.text
    assert 'id="backend-tour"' in page.text
    assert "same precomputed world" in page.text
    assert 'id="run-smoke"' in page.text
    assert 'id="transaction-demo"' in page.text
    assert 'id="transaction-form"' in page.text
    assert 'id="replay-transaction"' in page.text
    assert "/docs" in page.text
    assert results.status_code == 200
    payload = results.json()
    assert payload["routing"]["seed_count"] == 1
    assert payload["fraud"]["dataset_path"] == "test-fixture-only"
    assert payload["latency"]["target_met"] is False
    assert payload["attestation"]["result_digest"] == "test-digest"
    assert css.status_code == 200 and "color-scheme:dark" in css.text
    assert "--lime:#d8ff63" in css.text
    assert javascript.status_code == 200 and "/api/results" in javascript.text
    assert "requestSmokeCheck" in javascript.text
    assert '"/demo/transactions"' in javascript.text
    assert '"/demo/examples"' in javascript.text
    assert "not pytest" in javascript.text
