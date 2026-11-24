"""MCP 도구 호출과 실행 기록 (설계 5단계 6장, 1장 "모든 실행을 남긴다").

에이전트와 지식 공급자는 MCP 서버의 도구를 이 함수로 부른다. 호출마다 실행 기록에 도구 호출 1건을 남긴다.
"""

import json
from dataclasses import dataclass
from typing import Any

from mcp import Client

from common.tracing import Tracer


@dataclass(frozen=True)
class ToolOutput:
    text: str
    data: Any
    is_error: bool


class ToolCallError(RuntimeError):
    """도구가 오류로 답했다. 내용(정의되지 않은 ID 등)을 담는다."""


def _unwrap(structured: Any) -> Any:
    # 목록이나 문자열을 돌려주는 도구는 {"result": 값}으로 감싸져 온다
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    return structured


async def call_tool(client: Client, tracer: Tracer, *, server: str, name: str, args: dict[str, Any]) -> ToolOutput:
    """도구를 한 번 부르고 결과를 돌려준다. 오류 응답도 그대로 돌려준다 (is_error)."""
    with tracer.tool(f"{server}.{name}", input=args, metadata={"server": server}) as obs:
        result = await client.call_tool(name, args)
        text = "".join(getattr(block, "text", "") for block in result.content)
        data = _unwrap(result.structured_content)
        if data is None and not result.is_error:
            # dict를 돌려주는 도구는 구조화 결과 없이 본문에 JSON을 담아 온다
            try:
                data = json.loads(text)
            except ValueError:
                data = None
        output = ToolOutput(text=text, data=data, is_error=bool(result.is_error))
        obs.update(output=output.data if output.data is not None else text, level="ERROR" if output.is_error else "DEFAULT")
    return output


async def call_tool_data(client: Client, tracer: Tracer, *, server: str, name: str, args: dict[str, Any]) -> Any:
    """결과 데이터가 필요한 호출. 도구가 오류로 답하면 ToolCallError를 낸다."""
    output = await call_tool(client, tracer, server=server, name=name, args=args)
    if output.is_error:
        raise ToolCallError(f"{server}.{name} 오류: {output.text}")
    return output.data
