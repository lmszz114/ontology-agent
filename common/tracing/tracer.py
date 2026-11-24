"""실행 기록 (설계 5단계 1장 "모든 실행을 남긴다", 3장).

LLM 호출, 도구 호출, 승인 게이트 처리를 Langfuse에 남긴다. LLM 호출에는 토큰과 비용을 함께 남긴다.
기록 없이 돌지 않는다: Langfuse 키가 없거나 인증이 안 되면 from_env가 멈춘다.

기록 단위
- run: 진단 1회처럼 한 번의 실행. 그 아래 기록이 한 trace로 묶인다
- generation: LLM 호출 1회
- tool: 도구 호출 1회
- gate: 승인 게이트 처리 1회 (요청, 승인, 반려, 완료)
"""

import json
import os
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator

from langfuse import Langfuse, propagate_attributes

REQUIRED_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
# SDK는 주소를 LANGFUSE_BASE_URL과 LANGFUSE_HOST 두 이름으로 읽는다
URL_ENV = ("LANGFUSE_BASE_URL", "LANGFUSE_HOST")


class TracingConfigError(RuntimeError):
    """실행 기록을 남길 수 없는 설정이다. 이 상태로는 실행하지 않는다."""


def _jsonable(value: Any) -> Any:
    """Langfuse에 넘길 수 있는 값으로 바꾼다. 시각과 그래프 값은 문자열로 남긴다."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "iso_format"):
        return value.iso_format()
    return str(value)


class Observation:
    """기록 하나. 끝날 때 결과를 붙인다."""

    def __init__(self, span):
        self._span = span

    def update(self, **fields: Any) -> None:
        if "output" in fields:
            fields["output"] = _jsonable(fields["output"])
        if "metadata" in fields:
            fields["metadata"] = _jsonable(fields["metadata"])
        self._span.update(**fields)


class Tracer:
    def __init__(self, client: Langfuse):
        self._client = client

    @classmethod
    def from_env(cls, *, check: bool = True) -> "Tracer":
        """환경 변수로 Langfuse에 연결한다. 키가 없거나 인증이 실패하면 멈춘다.

        check=True면 인증 확인 요청을 한 번 보낸다. 잘못된 키로 실행이 끝까지 돌고
        기록만 사라지는 것을 막기 위해서다.
        """
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if not any(os.environ.get(name) for name in URL_ENV):
            missing.append(" 또는 ".join(URL_ENV))
        if missing:
            raise TracingConfigError(f"실행 기록에 필요한 환경 변수가 없다: {missing}")
        client = Langfuse()
        if check and not client.auth_check():
            raise TracingConfigError("Langfuse 인증에 실패했다. 키와 주소를 확인한다")
        return cls(client)

    @contextmanager
    def _observe(self, *, name: str, as_type: str, input: Any = None, **fields: Any) -> Iterator[Observation]:
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, input=_jsonable(input), **fields
        ) as span:
            obs = Observation(span)
            try:
                yield obs
            except Exception as exc:
                span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
                raise

    @contextmanager
    def run(
        self,
        name: str,
        *,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        input: Any = None,
    ) -> Iterator[Observation]:
        """실행 1회. 그 안의 기록은 한 trace로 묶이고 session_id, tags, metadata를 함께 가진다.

        metadata 값은 문자열로 남긴다 (Langfuse trace 속성 제약).
        """
        meta = {k: v if isinstance(v, str) else json.dumps(_jsonable(v), ensure_ascii=False)
                for k, v in (metadata or {}).items()}
        with self._observe(name=name, as_type="agent", input=input) as obs:
            with propagate_attributes(session_id=session_id, tags=tags, metadata=meta, trace_name=name):
                yield obs

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any,
        metadata: dict[str, Any] | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        """LLM 호출 1회. 끝날 때 usage_details와 cost_details를 붙인다."""
        with self._observe(
            name=name,
            as_type="generation",
            input=input,
            model=model,
            metadata=_jsonable(metadata),
            model_parameters=model_parameters,
        ) as obs:
            yield obs

    @contextmanager
    def tool(self, name: str, *, input: Any, metadata: dict[str, Any] | None = None) -> Iterator[Observation]:
        """도구 호출 1회."""
        with self._observe(name=name, as_type="tool", input=input, metadata=_jsonable(metadata)) as obs:
            yield obs

    @contextmanager
    def gate(self, name: str, *, input: Any) -> Iterator[Observation]:
        """승인 게이트 처리 1회."""
        with self._observe(name=name, as_type="guardrail", input=input) as obs:
            yield obs

    def flush(self) -> None:
        """쌓인 기록을 보낸다. 프로세스가 끝나기 전에 부른다."""
        self._client.flush()
