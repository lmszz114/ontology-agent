"""지식 공급자 인터페이스 (설계 5단계 3장, 4-2).

진단 에이전트는 하나다. 비교 구성(A, B, C, C-ext)은 이 인터페이스의 구현체만 바꿔 끼운다.
에이전트가 회사 지식을 얻는 단계(원인 순위 6, 판정 근거 7, 담당 팀 8, 절차 9)는 모두 이 인터페이스를 거친다.
코드 단계(1차 처리 역할 2, 심각도 5, 승인 게이트 10)는 모든 구성이 같은 기준 지식을 쓰므로 여기를 거치지 않는다.
"""

from typing import Any, Protocol


class KnowledgeProvider(Protocol):
    name: str  # 구성 이름 (C, C-ext 등)

    async def rank_causes(self, symptom_id: str, started_at: str) -> list[dict[str, Any]]:
        """원인 후보와 순위 (R4-1~3). 각 항목에 cause_id, name, score, rank, recent_changes가 있다."""

    async def get_checks(self, cause_id: str) -> list[dict[str, Any]]:
        """원인의 판정 근거와 조회 도구, 인자, 확정 조건 여부 (R4-4)."""

    async def owner_team(self, cause_id: str) -> dict[str, Any]:
        """원인의 1차 담당 팀, 같은 시스템의 다른 부분을 맡은 팀, 조건부 협업 팀 (R5)."""

    async def cause_procedures(self, cause_id: str, severity_id: str) -> dict[str, Any]:
        """원인이 여는 절차와 등급이 여는 절차, 각 절차의 조치와 위험도 (R6, R7)."""

    async def procedure(self, procedure_id: str) -> dict[str, Any]:
        """원인과 상관없이 여는 절차 하나 (접수 P1, 상향 P2, 긴급 점검 P4, 원인 수정 P10)."""
