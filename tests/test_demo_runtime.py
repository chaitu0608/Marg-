import pytest

from marg.demo.runtime import DemoRuntime, IdempotencyConflict


def transaction_data(*, amount=12.5, outage_gateway=None):
    return {
        "features": [0.1] + [0.0] * 29,
        "bank": "bank_0",
        "method": "card",
        "amount": amount,
        "outage_gateway": outage_gateway,
    }


def test_allow_routes_updates_observed_history_and_replay_is_idempotent(tmp_path):
    runtime = DemoRuntime(tmp_path / "demo.sqlite3")
    request_data = transaction_data(outage_gateway="gw_a")
    first = runtime.submit(idempotency_key="request-0001", request_data=request_data,
                           score=0.1, decision="allow", inference_ms=0.2)
    replay = runtime.submit(idempotency_key="request-0001", request_data=request_data,
                            score=0.1, decision="allow", inference_ms=0.2)

    assert first["synthetic"] is True
    assert first["gateway"] in {"gw_a", "gw_b", "gw_c", "gw_d"}
    assert first["gateway_outcome"] in {"success", "failure"}
    assert first["router_observations_for_gateway"] == 1
    assert "success_probability" not in first
    assert replay["transaction_id"] == first["transaction_id"]
    assert replay["idempotent_replay"] is True
    assert runtime.summary()["transaction_count"] == 1
    assert runtime.summary()["router_observation_count"] == 1
    assert sum(row["attempts"] for row in runtime.summary()["recent_route_window"]) == 1


def test_idempotency_key_cannot_be_reused_for_different_request(tmp_path):
    runtime = DemoRuntime(tmp_path / "demo.sqlite3")
    runtime.submit(idempotency_key="request-0002", request_data=transaction_data(),
                   score=0.1, decision="allow", inference_ms=0.2)

    with pytest.raises(IdempotencyConflict):
        runtime.submit(idempotency_key="request-0002", request_data=transaction_data(amount=15),
                       score=0.1, decision="allow", inference_ms=0.2)


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("review", "awaiting_review"), ("block", "blocked")],
)
def test_review_and_block_do_not_reach_fake_gateway(tmp_path, decision, expected_status):
    runtime = DemoRuntime(tmp_path / f"{decision}.sqlite3")
    response = runtime.submit(idempotency_key=f"request-{decision}",
                              request_data=transaction_data(), score=0.9,
                              decision=decision, inference_ms=0.2)

    assert response["status"] == expected_status
    assert response["gateway"] is None
    assert response["gateway_outcome"] is None
    assert runtime.summary()["router_observation_count"] == 0


def test_router_uses_persisted_observations_after_gateway_outage(tmp_path):
    runtime = DemoRuntime(tmp_path / "demo.sqlite3")
    normal_routes = []
    for index in range(1, 51):
        data = transaction_data()
        response = runtime.submit(
            idempotency_key=f"normal-{index:04d}", request_data=data,
            score=0.1, decision="allow", inference_ms=0.2,
        )
        normal_routes.append(response["gateway"])

    # Degrade the gateway the router has learned to use most often in this context.
    from collections import Counter

    degraded_gateway = Counter(normal_routes).most_common(1)[0][0]
    outage_routes = []
    for index in range(51, 121):
        response = runtime.submit(
            idempotency_key=f"outage-{index:04d}",
            request_data=transaction_data(outage_gateway=degraded_gateway),
            score=0.1, decision="allow", inference_ms=0.2,
        )
        outage_routes.append(response["gateway"])

    early_outage_share = outage_routes[:20].count(degraded_gateway) / 20
    late_outage_share = outage_routes[-20:].count(degraded_gateway) / 20
    assert runtime.summary()["router_observation_count"] == 120
    assert sum(row["attempts"] for row in runtime.summary()["recent_route_window"]) == 50
    assert early_outage_share > late_outage_share
