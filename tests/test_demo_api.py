import json

import numpy as np
from fastapi.testclient import TestClient

from marg.fraud.serve import create_app


class TinyModel:
    def predict_proba(self, row):
        score = min(max(float(row[0, 0]), 0), 1)
        return np.array([[1 - score, score]])


def make_client(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"test model placeholder")
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps({"block": 0.8, "review": 0.4}))
    monkeypatch.setattr("marg.fraud.serve.joblib.load", lambda _: TinyModel())
    app = create_app(str(model_path), str(thresholds_path), str(tmp_path / "demo.sqlite3"))
    return TestClient(app)


def payload(score_input=0.1):
    return {
        "features": [score_input] + [0.0] * 29,
        "bank": "bank_0",
        "method": "card",
        "amount": 15.0,
        "outage_gateway": "gw_a",
    }


def submit(client, data, key="request-0001"):
    return client.post("/demo/transactions", json=data,
                       headers={"Idempotency-Key": key})


def test_demo_allow_routes_and_duplicate_key_replays_without_learning_twice(
    tmp_path, monkeypatch,
):
    with make_client(tmp_path, monkeypatch) as client:
        first = submit(client, payload())
        replay = submit(client, payload())
        summary = client.get("/demo/summary").json()
        history = client.get("/demo/transactions").json()

    assert first.status_code == 200
    result = first.json()
    assert result["decision"] == "allow"
    assert result["gateway"] in {"gw_a", "gw_b", "gw_c", "gw_d"}
    assert result["gateway_outcome"] in {"success", "failure"}
    assert result["synthetic"] is True
    assert "success_probability" not in result
    assert replay.json()["transaction_id"] == result["transaction_id"]
    assert replay.json()["idempotent_replay"] is True
    assert summary["transaction_count"] == summary["router_observation_count"] == 1
    assert len(history) == 1


def test_demo_review_and_block_stop_before_fake_gateway(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        review = submit(client, payload(0.6), "review-key-0001").json()
        block = submit(client, payload(0.9), "block-key-00001").json()
        summary = client.get("/demo/summary").json()

    assert review["status"] == "awaiting_review"
    assert review["gateway"] is None
    assert block["status"] == "blocked"
    assert block["gateway"] is None
    assert summary["router_observation_count"] == 0


def test_demo_rejects_changed_request_under_same_idempotency_key(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        assert submit(client, payload()).status_code == 200
        conflict = submit(client, payload(0.2))

    assert conflict.status_code == 409


def test_demo_validates_context_amount_and_feature_values(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        invalid_bank = payload()
        invalid_bank["bank"] = "not-a-bank"
        invalid_amount = payload()
        invalid_amount["amount"] = None
        invalid_feature = payload()
        invalid_feature["features"][2] = None

        assert submit(client, invalid_bank).status_code == 422
        assert submit(client, invalid_amount, "amount-key-0001").status_code == 422
        assert submit(client, invalid_feature, "feature-key-001").status_code == 422


def test_demo_examples_serves_only_generated_feature_profiles(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    sample = {"split": "test-only", "examples": [{"decision": "allow", "features": [0.0] * 30}]}
    (artifact_dir / "demo-samples.json").write_text(json.dumps(sample))
    monkeypatch.setattr("marg.fraud.serve.PROJECT_ROOT", tmp_path)

    with make_client(tmp_path, monkeypatch) as client:
        response = client.get("/demo/examples")

    assert response.status_code == 200
    assert len(response.json()["examples"][0]["features"]) == 30
    assert "labels" not in response.json()["examples"][0]
