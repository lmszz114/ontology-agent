"""실행 기록 검증. 네트워크를 쓰지 않는다."""

from datetime import datetime, timedelta, timezone

import pytest

from common.tracing import Tracer, TracingConfigError
from tests.memory_tracer import MemoryTracer

KST = timezone(timedelta(hours=9))


@pytest.mark.parametrize("missing", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "url"])
def test_missing_keys_stop_the_run(monkeypatch, missing):
    """키 없이는 실행하지 않는다 (사용자 결정, 2026-11-24)."""
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_BASE_URL": "https://x.invalid"}
    for name in (*env, "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        if missing == "url" and name == "LANGFUSE_BASE_URL":
            continue
        if name != missing:
            monkeypatch.setenv(name, value)
    with pytest.raises(TracingConfigError, match="환경 변수가 없다"):
        Tracer.from_env()


def test_failed_auth_stops_the_run(monkeypatch):
    class Refusing:
        def __init__(self, *a, **k):
            pass

        def auth_check(self):
            return False

    for name, value in {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk",
                        "LANGFUSE_BASE_URL": "https://x.invalid"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("common.tracing.tracer.Langfuse", Refusing)
    with pytest.raises(TracingConfigError, match="인증에 실패"):
        Tracer.from_env()


def test_run_groups_tool_and_gate_records_in_one_trace():
    tracer = MemoryTracer()
    with tracer.run("diagnosis", session_id="SC-S2-01", tags=["dev", "C"],
                    metadata={"scenario": "SC-S2-01", "repeat": 1}) as run:
        with tracer.tool("get_checks", input={"cause_id": "C03"}) as tool:
            tool.update(output={"checks": ["E03"]})
        with tracer.gate("gate.request_action", input={"action_id": "A01"}) as gate:
            gate.update(output={"status": "auto_done", "requested_at": datetime(2026, 12, 3, 20, 30, tzinfo=KST)})
        run.update(output={"top1": "C03"})

    tool_span, gate_span, run_span = tracer.spans()
    assert {s["trace_id"] for s in (tool_span, gate_span, run_span)} == {run_span["trace_id"]}
    assert (tool_span["type"], gate_span["type"], run_span["type"]) == ("tool", "guardrail", "agent")
    assert tool_span["output"] == {"checks": ["E03"]}
    assert gate_span["output"]["requested_at"] == "2026-12-03T20:30:00+09:00"
    assert tool_span["session_id"] == "SC-S2-01"
    # trace 메타데이터는 문자열로 남는다
    assert tool_span["trace_metadata"] == {"scenario": "SC-S2-01", "repeat": "1"}


def test_exception_inside_record_is_marked_error_and_reraised():
    tracer = MemoryTracer()
    with pytest.raises(ValueError):
        with tracer.tool("get_checks", input={}):
            raise ValueError("정의되지 않은 ID")
    (span,) = tracer.spans()
    assert span["level"] == "ERROR" and span["status_message"] == "ValueError: 정의되지 않은 ID"
