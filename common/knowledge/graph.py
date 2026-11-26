"""그래프 지식 공급자 (구성 C, C-ext) (설계 5단계 4-2).

온톨로지 조회 MCP 서버의 도구로 후보 원인, 점수, 판정 근거, 담당 팀, 절차를 받는다.
원인 점수는 서버의 고정 쿼리가 그래프에서 계산한다. LLM은 이 값을 바꾸지 않는다.
C와 C-ext는 같은 공급자를 쓰고, 서버가 연결된 그래프만 다르다.
"""

from typing import Any

from mcp import Client

from common.tool_calls import call_tool_data
from common.tracing import Tracer

SERVER = "ontology_query"


class GraphKnowledge:
    def __init__(self, client: Client, tracer: Tracer, *, name: str = "C"):
        self.client = client
        self.tracer = tracer
        self.name = name

    async def _call(self, tool: str, **args: Any) -> Any:
        return await call_tool_data(self.client, self.tracer, server=SERVER, name=tool, args=args)

    async def begin_incident(self, incident_id: str, reports: list[str]) -> dict | None:
        """그래프는 사건과 상관없이 같은 지식을 준다."""
        return None

    async def rank_causes(self, symptom_id: str, started_at: str) -> list[dict[str, Any]]:
        return await self._call("rank_causes", symptom_id=symptom_id, started_at=started_at)

    async def get_checks(self, cause_id: str) -> list[dict[str, Any]]:
        return await self._call("get_checks", cause_id=cause_id)

    async def owner_team(self, cause_id: str) -> dict[str, Any]:
        return await self._call("get_owner_team", cause_id=cause_id)

    async def cause_procedures(self, cause_id: str, severity_id: str) -> dict[str, Any]:
        return await self._call("get_procedures", cause_id=cause_id, severity_id=severity_id)

    async def procedure(self, procedure_id: str) -> dict[str, Any]:
        return await self._call("get_procedure", procedure_id=procedure_id)
