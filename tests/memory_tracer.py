"""테스트용 실행 기록. 운영 코드와 같은 Tracer를 쓰고, 보내는 곳만 메모리로 바꾼다.

네트워크를 쓰지 않고 Langfuse에 넘어갈 속성을 그대로 검사할 수 있다.
"""

import json
import uuid

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from common.tracing import Tracer


class MemoryTracer(Tracer):
    def __init__(self):
        self.exporter = InMemorySpanExporter()
        # Langfuse 클라이언트는 공개 키마다 하나씩 공유되므로 테스트마다 다른 키를 쓴다
        client = Langfuse(
            public_key=f"pk-test-{uuid.uuid4().hex}",
            secret_key="sk-test",
            base_url="http://localhost:1",
            span_exporter=self.exporter,
            tracer_provider=TracerProvider(),
        )
        super().__init__(client)

    def spans(self) -> list[dict]:
        """끝난 기록을 이름, 종류, 주요 속성으로 돌려준다."""
        self.flush()
        out = []
        for span in self.exporter.get_finished_spans():
            attrs = dict(span.attributes)

            def field(name):
                raw = attrs.get(f"langfuse.observation.{name}")
                if raw is None:
                    return None
                try:
                    return json.loads(raw)
                except (TypeError, ValueError):
                    return raw

            out.append({
                "name": span.name,
                "type": attrs.get("langfuse.observation.type"),
                "level": attrs.get("langfuse.observation.level", "DEFAULT"),
                "status_message": attrs.get("langfuse.observation.status_message"),
                "input": field("input"),
                "output": field("output"),
                "metadata": {k.removeprefix("langfuse.observation.metadata."): json.loads(v)
                             for k, v in attrs.items() if k.startswith("langfuse.observation.metadata.")},
                "model": attrs.get("langfuse.observation.model.name"),
                "usage": field("usage_details"),
                "cost": field("cost_details"),
                "session_id": attrs.get("session.id"),
                "trace_metadata": {k.removeprefix("langfuse.trace.metadata."): json.loads(v)
                                   for k, v in attrs.items() if k.startswith("langfuse.trace.metadata.")},
                "trace_id": span.context.trace_id,
            })
        return out
