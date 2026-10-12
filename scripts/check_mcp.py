from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def check_server() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "marg.mcp_server"], cwd=str(ROOT))
    async with (stdio_client(params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session):
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        expected = {"get_marg_result", "compare_fraud_policy", "marg_guardrails"}
        if tools != expected:
            raise AssertionError(f"MCP tool set mismatch: {tools}")
        result = await session.call_tool("marg_guardrails", {})
        if result.isError or not result.content:
            raise AssertionError("MCP guardrail tool returned an error or no content")
        artifact = await session.call_tool("get_marg_result", {"name": "fraud_sensitivity"})
        if artifact.isError or not artifact.content:
            raise AssertionError("MCP result tool could not read the saved fraud sensitivity result")


def main() -> None:
    asyncio.run(check_server())
    print("MCP stdio tools initialized and read the saved fraud sensitivity result.")


if __name__ == "__main__":
    main()
