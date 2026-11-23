"""사건 계층 기록 함수 (설계 3단계 2-2, 3-2, 5단계 1장).

LLM은 그래프에 직접 쓰지 않는다. 사건 계층은 이 모듈의 함수가 정해진 형식으로만 쓴다.
정의되지 않은 값(없는 시스템, 허용되지 않은 종류, 시간대 없는 시각)은 쓰기 전에 거부한다.
함수는 그것을 쓰는 단계를 만들 때 함께 추가한다.
"""

from datetime import datetime

from neo4j import Driver

# 사건 계층의 노드 레이블 (설계 3단계 2-2)
EVENT_LABELS = ("Report", "Incident", "Finding", "ActionRun", "Change")

# Change.kind 허용 값 (설계 3단계 2-2)
CHANGE_KINDS = {"deploy", "table_change", "event_config", "product_config", "patch"}


class RecordError(ValueError):
    """기록할 수 없는 값."""


def _require_aware(value: datetime, name: str) -> None:
    # 시각은 시간대가 붙은 datetime만 받는다. 문자열이나 시간대 없는 시각이 섞이면
    # 변경 가중(R4)의 시간 비교가 실패하거나 9시간 어긋난다
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RecordError(f"{name}은 시간대가 붙은 datetime이어야 한다: {value!r}")


def record_change(
    driver: Driver,
    *,
    change_id: str,
    kind: str,
    system_id: str,
    changed_at: datetime,
    changed_by_team: str | None = None,
) -> None:
    """변경 이력 하나를 기록한다 (Change 노드와 CHANGED 관계).

    원인 순위의 변경 가중(R4)이 이 기록을 읽는다.
    """
    if kind not in CHANGE_KINDS:
        raise RecordError(f"Change.kind에 쓸 수 없는 값이다: {kind}")
    _require_aware(changed_at, "changed_at")

    def write(tx) -> None:
        if tx.run("MATCH (n:Change {id: $id}) RETURN n.id", id=change_id).single():
            raise RecordError(f"이미 있는 Change다: {change_id}")
        if changed_by_team is not None and not tx.run(
            "MATCH (t:Team {id: $id}) RETURN t.id", id=changed_by_team
        ).single():
            raise RecordError(f"정의되지 않은 팀이다: {changed_by_team}")
        created = tx.run(
            "MATCH (sys:System {id: $system}) "
            "CREATE (c:Change {id: $id, kind: $kind, changed_at: $at})-[:CHANGED]->(sys) "
            "SET c.changed_by_team = $team "
            "RETURN c.id",
            system=system_id, id=change_id, kind=kind, at=changed_at, team=changed_by_team,
        ).single()
        if created is None:
            raise RecordError(f"정의되지 않은 시스템이다: {system_id}")

    with driver.session() as session:
        session.execute_write(write)


def reset_event_layer(driver: Driver) -> int:
    """사건 계층 노드를 모두 지우고 지운 수를 돌려준다. 정적 지식 계층은 건드리지 않는다.

    평가 모드는 실행마다 사건 계층을 새로 만든다 (설계 5단계 7장).
    """
    deleted = 0
    with driver.session() as session:
        for label in EVENT_LABELS:
            summary = session.run(f"MATCH (n:{label}) DETACH DELETE n").consume()
            deleted += summary.counters.nodes_deleted
    return deleted
