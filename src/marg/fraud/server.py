"""Uvicorn entry point that adds request-dispatch queue timing to the ASGI scope."""

from __future__ import annotations

import argparse
import time
from typing import Any

import uvicorn
from uvicorn.protocols.http import h11_impl


class TimedRequestResponseCycle(h11_impl.RequestResponseCycle):
    """Measure event-loop delay from Uvicorn parsing headers to ASGI dispatch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.marg_scheduled_ns = time.perf_counter_ns()

    async def run_asgi(self, app: Any) -> None:
        extensions = self.scope.setdefault("extensions", {})
        extensions["marg.server_dispatch_queue_ms"] = (
            time.perf_counter_ns() - self.marg_scheduled_ns
        ) / 1_000_000
        await super().run_asgi(app)


# Each Uvicorn worker imports this module to load the custom protocol class.
h11_impl.RequestResponseCycle = TimedRequestResponseCycle


class TimedH11Protocol(h11_impl.H11Protocol):
    """HTTP/1.1 protocol whose request cycles expose their dispatch delay."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # Workers import this module as the protocol so the cycle timing is active in each process.
    uvicorn.run("marg.fraud.serve:app", host=args.host, port=args.port, workers=args.workers,
                http="marg.fraud.server:TimedH11Protocol")


if __name__ == "__main__":
    main()
