"""온톨로지 조회 MCP 서버 (설계 5단계 6장).

실행: python -m mcp_servers.ontology_query.server

도구는 queries.py의 고정 읽기 쿼리만 부른다. 임의 Cypher를 받는 도구는 없다.
"""

from collections.abc import Callable
from functools import wraps

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from neo4j import Driver

from mcp_servers.ontology_query import queries
from ontology.db import get_driver


def _input_errors_as_tool_errors(fn: Callable) -> Callable:
    """잘못된 입력(정의되지 않은 ID, 형식이 틀린 시각)의 내용을 호출한 쪽에 전달한다.

    SDK는 ToolError가 아닌 예외의 내용을 숨긴다. 에이전트가 무엇을 틀렸는지 알아야
    다시 요청할 수 있으므로 입력 오류만 ToolError로 바꾼다. 그 밖의 오류는 숨긴 채로 둔다.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def build_server(driver: Driver) -> MCPServer:
    server = MCPServer(
        "ontology-query",
        instructions="가상 MMORPG 운영 조직의 온톨로지를 조회한다. 읽기 전용이다.",
    )

    @server.tool()
    @_input_errors_as_tool_errors
    def get_symptom_types() -> list[dict]:
        """증상 유형 전체를 돌려준다. 이름, 유사 표현, 핵심 기능 여부, 증상이 드러나는 시스템."""
        return queries.get_symptom_types(driver)

    @server.tool()
    @_input_errors_as_tool_errors
    def get_cause_candidates(symptom_id: str) -> list[dict]:
        """증상 유형(S1~S8)의 원인 후보 전체와 사전 가능성, 원인이 있는 시스템을 돌려준다."""
        return queries.get_cause_candidates(driver, symptom_id)

    @server.tool()
    @_input_errors_as_tool_errors
    def rank_causes(symptom_id: str, started_at: str) -> list[dict]:
        """원인 후보의 점수와 순위를 돌려준다.

        점수는 사전 가능성 점수에, 원인 위치 시스템에 증상 시작 전 24시간 안의 변경 이력이 있으면 가중을 더한 값이다.
        started_at은 증상 시작 시각이다. 시간대가 붙은 ISO 8601 형식 (예: 2026-12-03T20:14+09:00).
        """
        return queries.rank_causes(driver, symptom_id, started_at)

    @server.tool()
    @_input_errors_as_tool_errors
    def get_checks(cause_id: str) -> list[dict]:
        """원인(C01~C21)을 확정하거나 배제하는 판정 근거와 확정 조건, 조회 도구 이름과 넘길 인자(args)를 돌려준다."""
        return queries.get_checks(driver, cause_id)

    @server.tool()
    @_input_errors_as_tool_errors
    def get_owner_team(cause_id: str) -> dict:
        """원인의 1차 담당 팀, 같은 시스템의 다른 부분을 맡은 팀, 조건부 협업 팀을 돌려준다."""
        return queries.get_owner_team(driver, cause_id)

    @server.tool()
    @_input_errors_as_tool_errors
    def get_procedures(cause_id: str, severity_id: str | None = None) -> dict:
        """원인이 확정됐을 때 여는 절차와 조치를 돌려준다. severity_id(SEV1~SEV4)를 주면 그 등급이 여는 절차도 함께 준다."""
        return queries.get_procedures(driver, cause_id, severity_id)

    @server.tool()
    @_input_errors_as_tool_errors
    def get_action_policy(action_id: str) -> dict:
        """조치(A01~A17)의 위험도, 자동 실행 여부, 실행 팀, 승인권자를 돌려준다."""
        return queries.get_action_policy(driver, action_id)

    return server


if __name__ == "__main__":
    with get_driver() as drv:
        build_server(drv).run()
