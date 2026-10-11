import pytest

from marg.gateway.safety import (
    DispatchRejected,
    GatewayCommand,
    GatewayDispatchGuard,
)


class RecordingAdapter:
    def __init__(self):
        self.calls = []

    def submit(self, **payload):
        self.calls.append(payload)
        return "accepted"


def command(**overrides):
    fields = {"transaction_id": "txn-1", "gateway": "gw-a", "amount": 25.0,
              "decision": "allow", "idempotency_key": "txn-1"}
    fields.update(overrides)
    return GatewayCommand(**fields)


def test_gateway_dispatch_is_disabled_by_default():
    adapter = RecordingAdapter()
    guard = GatewayDispatchGuard(adapter)
    with pytest.raises(DispatchRejected, match="disabled"):
        guard.dispatch(command())
    assert adapter.calls == []


def test_live_dispatch_requires_allow_decision_allowlist_cap_and_unique_key():
    adapter = RecordingAdapter()
    guard = GatewayDispatchGuard(
        adapter, live_enabled=True, allowlisted_gateways=frozenset({"gw-a"}), max_amount=50.0)
    assert guard.dispatch(command()) == "accepted"
    assert len(adapter.calls) == 1
    with pytest.raises(DispatchRejected, match="already been used"):
        guard.dispatch(command())
    with pytest.raises(DispatchRejected, match="already been dispatched"):
        guard.dispatch(command(idempotency_key="new-key"))
    with pytest.raises(DispatchRejected, match="only allow"):
        guard.dispatch(command(idempotency_key="txn-2", decision="review"))
    with pytest.raises(DispatchRejected, match="not allowlisted"):
        guard.dispatch(command(idempotency_key="txn-3", gateway="other"))
    with pytest.raises(DispatchRejected, match="dispatch cap"):
        guard.dispatch(command(idempotency_key="txn-4", amount=51.0))
    assert len(adapter.calls) == 1


def test_live_dispatch_cannot_be_configured_without_required_limits():
    with pytest.raises(ValueError, match="allowlist"):
        GatewayDispatchGuard(RecordingAdapter(), live_enabled=True)
    with pytest.raises(ValueError, match="allowlist"):
        GatewayDispatchGuard(RecordingAdapter(), live_enabled=True,
                             allowlisted_gateways=frozenset({" "}), max_amount=10.0)
    with pytest.raises(ValueError, match="amount cap"):
        GatewayDispatchGuard(RecordingAdapter(), live_enabled=True,
                             allowlisted_gateways=frozenset({"gw-a"}), max_amount=float("nan"))


def test_non_finite_or_empty_identifiers_are_rejected():
    guard = GatewayDispatchGuard(
        RecordingAdapter(), live_enabled=True, allowlisted_gateways=frozenset({"gw-a"}),
        max_amount=50.0)
    with pytest.raises(DispatchRejected, match="finite and positive"):
        guard.dispatch(command(amount=float("nan")))
    with pytest.raises(DispatchRejected, match="identifiers are required"):
        guard.dispatch(command(idempotency_key=" "))
