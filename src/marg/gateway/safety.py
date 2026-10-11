from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol


class GatewayAdapter(Protocol):
    def submit(self, *, transaction_id: str, gateway: str, amount: float,
               idempotency_key: str) -> str: ...


class DispatchRejected(ValueError):
    """A payment dispatch failed a local safety check."""


@dataclass(frozen=True)
class GatewayCommand:
    transaction_id: str
    gateway: str
    amount: float
    decision: Literal["allow", "review", "block"]
    idempotency_key: str


class GatewayDispatchGuard:
    """Fail-closed adapter boundary; live dispatch requires explicit limits and opt-in."""

    def __init__(self, adapter: GatewayAdapter | None = None, *, live_enabled: bool = False,
                 allowlisted_gateways: frozenset[str] = frozenset(),
                 max_amount: float | None = None) -> None:
        self.adapter = adapter
        self.live_enabled = live_enabled
        self.allowlisted_gateways = allowlisted_gateways
        self.max_amount = max_amount
        self.used_idempotency_keys: set[str] = set()
        self.used_transaction_ids: set[str] = set()
        invalid_allowlist = not allowlisted_gateways or any(
            not gateway.strip() for gateway in allowlisted_gateways)
        invalid_cap = (max_amount is None or not math.isfinite(max_amount) or max_amount <= 0)
        if live_enabled and (adapter is None or invalid_allowlist or invalid_cap):
            raise ValueError("live dispatch needs an adapter, gateway allowlist, and positive amount cap")

    def dispatch(self, command: GatewayCommand) -> str:
        if not self.live_enabled:
            raise DispatchRejected("live gateway dispatch is disabled")
        if command.decision != "allow":
            raise DispatchRejected("only allow decisions may reach a gateway")
        if command.gateway not in self.allowlisted_gateways:
            raise DispatchRejected("gateway is not allowlisted")
        if not math.isfinite(command.amount) or command.amount <= 0:
            raise DispatchRejected("amount must be finite and positive")
        if self.max_amount is None or command.amount > self.max_amount:
            raise DispatchRejected("amount exceeds the configured dispatch cap")
        if not command.transaction_id.strip() or not command.idempotency_key.strip():
            raise DispatchRejected("transaction and idempotency identifiers are required")
        if command.idempotency_key in self.used_idempotency_keys:
            raise DispatchRejected("idempotency key has already been used")
        if command.transaction_id in self.used_transaction_ids:
            raise DispatchRejected("transaction has already been dispatched")

        # Record before network I/O so retries after ambiguous timeouts cannot double-submit here.
        self.used_idempotency_keys.add(command.idempotency_key)
        self.used_transaction_ids.add(command.transaction_id)
        assert self.adapter is not None
        return self.adapter.submit(
            transaction_id=command.transaction_id,
            gateway=command.gateway,
            amount=command.amount,
            idempotency_key=command.idempotency_key,
        )
