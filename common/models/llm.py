"""LLM 호출 (설계 5단계 3장 모델 선택 규칙, 1장 "모든 실행을 남긴다").

두 에이전트의 모든 LLM 호출은 이 모듈을 거친다. 작업 이름으로 모델을 고르고,
호출을 실행 기록에 남기고, 응답의 토큰 사용량으로 비용을 계산해 함께 남긴다.

API 키와 주소는 코드에서 정한다. 작업 환경에는 Claude Code가 쓰는 ANTHROPIC_BASE_URL이 들어와 있고
Anthropic 라이브러리는 이 이름을 자동으로 읽는다. 주소를 지정하지 않으면 에이전트 호출이
Claude Code의 통신 경로로 갈 수 있어서 공식 주소를 직접 넘긴다.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

from common.models.config import ModelConfig, usage_from_response
from common.tracing import Tracer

API_KEY_ENV = "ONTOLOGY_ANTHROPIC_API_KEY"
API_BASE_URL = "https://api.anthropic.com"
DEFAULT_MAX_TOKENS = 16000


class LLMConfigError(RuntimeError):
    """LLM을 부를 수 없는 설정이다."""


def _check_tools(tools: list[dict[str, Any]]) -> None:
    """직접 정의한 도구만 받는다.

    웹 검색 같은 서버 도구는 토큰 외 요금이 붙는데 비용 계산에 없어서, 비용 기록이 빠지지 않도록 호출 전에 거부한다.
    """
    for tool in tools:
        kind = tool.get("type")
        if kind not in (None, "custom"):
            raise LLMConfigError(f"서버 도구는 쓰지 않는다: {tool.get('name')} ({kind})")


def make_anthropic_client() -> Anthropic:
    """이 프로젝트 전용 키와 공식 주소로 클라이언트를 만든다.

    SDK 자체의 OpenTelemetry 기록은 끈다. 켜 두면 Langfuse가 그 기록도 LLM 호출로 받아서
    호출 1회가 두 번 기록되고 비용 합계가 두 배로 보인다 (2026-11-24 실측). 호출 기록은 LLM.call이 남긴다.
    추적 헤더를 API로 보내는 것도 쓸 데가 없어 끈다.
    """
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise LLMConfigError(f"환경 변수 {API_KEY_ENV}가 없다")
    return Anthropic(
        api_key=key,
        base_url=API_BASE_URL,
        open_telemetry={"traces": False, "propagation": False},
    )


@dataclass(frozen=True)
class LLMResult:
    message: Message
    model: str
    usage: dict[str, int]
    cost: dict[str, float]

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.message.content if b.type == "text")

    def json(self) -> Any:
        """JSON 형식 출력(output_schema)을 요청한 호출의 결과를 읽는다. 형식이 깨졌으면 ValueError."""
        return json.loads(self.text)


class LLM:
    def __init__(self, config: ModelConfig, tracer: Tracer, client: Anthropic):
        self.config = config
        self.tracer = tracer
        self.client = client

    def call(
        self,
        *,
        task: str,
        name: str,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        metadata: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        """작업에 맞는 모델로 Messages API를 한 번 부르고 기록을 남긴다.

        name은 실행 기록에 보일 이름이다 (예: 진단 단계 이름).
        refusal 같은 stop_reason은 그대로 돌려준다. 어떻게 다룰지는 부르는 쪽이 정한다.
        output_schema(JSON Schema)를 주면 응답을 그 형식의 JSON으로 받는다 (structured outputs).
        형식은 API가 보장하지만 값이 정의된 ID인지는 부르는 쪽이 검증한다.
        """
        model = self.config.model_for(task)
        if tools is not None:
            _check_tools(tools)
        params: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            params["system"] = system
        if tools is not None:
            params["tools"] = tools
        if output_schema is not None:
            params["output_config"] = {"format": {"type": "json_schema", "schema": output_schema}}
        trace_input = {"system": system, "messages": messages, "tools": [t.get("name") for t in tools or []]}
        with self.tracer.generation(
            name,
            model=model,
            input=trace_input,
            metadata={"task": task, "package": self.config.package, **(metadata or {})},
            model_parameters={"max_tokens": max_tokens},
        ) as gen:
            message = self.client.messages.create(**params)
            # 받은 응답은 비용 계산 전에 먼저 남긴다. 계산이 실패해도 호출과 원래 사용량이 기록에 남는다
            gen.update(
                output=[block.model_dump(mode="json") for block in message.content],
                metadata={
                    "stop_reason": message.stop_reason,
                    "request_id": getattr(message, "_request_id", None),
                    "raw_usage": message.usage.model_dump(mode="json"),
                },
                level="WARNING" if message.stop_reason in ("refusal", "max_tokens") else "DEFAULT",
            )
            usage = usage_from_response(message.usage)
            cost = self.config.cost(model, usage)
            gen.update(usage_details=usage, cost_details=cost)
        return LLMResult(message=message, model=model, usage=usage, cost=cost)
