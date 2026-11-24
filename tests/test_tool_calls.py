"""MCP 도구 호출 도우미 검증. 실제 MCP 서버를 프로세스 안에서 붙인다."""

import asyncio

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from common.tool_calls import ToolCallError, call_tool, call_tool_data
from tests.memory_tracer import MemoryTracer


def server() -> MCPServer:
    s = MCPServer("t")

    @s.tool()
    def as_list() -> list[dict]:
        return [{"a": 1}]

    @s.tool()
    def as_dict() -> dict:
        return {"b": 2}

    @s.tool()
    def as_text() -> str:
        return "이상 없음"

    @s.tool()
    def fails() -> str:
        raise ToolError("정의되지 않은 ID다: C99")

    return s


def run(fn):
    async def go():
        async with Client(server()) as client:
            return await fn(client)
    return asyncio.run(go())


def test_results_are_unwrapped_and_traced():
    tracer = MemoryTracer()

    async def calls(client):
        return [await call_tool_data(client, tracer, server="t", name=n, args={}) for n in ("as_list", "as_dict", "as_text")]

    assert run(calls) == [[{"a": 1}], {"b": 2}, "이상 없음"]
    spans = tracer.spans()
    assert [s["name"] for s in spans] == ["t.as_list", "t.as_dict", "t.as_text"]
    assert all(s["type"] == "tool" for s in spans)


def test_tool_error_is_returned_or_raised():
    tracer = MemoryTracer()
    out = run(lambda c: call_tool(c, tracer, server="t", name="fails", args={}))
    assert out.is_error and "C99" in out.text

    async def raised(client):
        # 클라이언트 연결 안에서 잡는다. 밖으로 나가면 작업 묶음 예외로 감싸진다
        with pytest.raises(ToolCallError, match="C99"):
            await call_tool_data(client, tracer, server="t", name="fails", args={})

    run(raised)
    assert tracer.spans()[0]["level"] == "ERROR"
