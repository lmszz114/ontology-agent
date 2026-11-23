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


# 노드에 쓰기 잠금을 거는 문장. 잠금 전용 속성에 상수를 쓴다.
# "SET n.status = n.status"처럼 기존 값을 다시 쓰면 안 된다. 잠금을 기다리는 동안
# 다른 트랜잭션이 바꾼 값을, 기다리기 전에 읽어 둔 옛 값으로 덮어쓴다
LOCK = "MATCH (n:{label} {{id: $id}}) SET n._lock = true RETURN n.id"


class RecordError(ValueError):
    """기록할 수 없는 값."""


def _require_aware(value: datetime, name: str) -> None:
    # 시각은 시간대가 붙은 datetime만 받는다. 문자열이나 시간대 없는 시각이 섞이면
    # 변경 가중(R4)의 시간 비교가 실패하거나 9시간 어긋난다
    # tzinfo가 있어도 utcoffset()이 None이면 시간대가 없는 값이다
    if not isinstance(value, datetime) or value.utcoffset() is None:
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


# Incident.status, ASSIGNED_TEAM.role 허용 값 (설계 3단계 2-2, 3-2)
INCIDENT_STATUSES = {"open", "investigating", "mitigated", "closed"}
TEAM_ROLES = {"primary", "collaborator"}

# ActionRun.status, actor 허용 값 (설계 3단계 2-2)
RUN_STATUSES = {"auto_done", "pending_approval", "approved", "rejected", "done"}
RUN_ACTORS = {"agent", "human"}


def record_incident(driver: Driver, *, incident_id: str, severity_id: str, opened_at: datetime) -> None:
    """사건 하나를 연다 (Incident 노드). 상태는 open으로 시작한다."""
    _require_aware(opened_at, "opened_at")

    def write(tx) -> None:
        if tx.run("MATCH (n:Incident {id: $id}) RETURN n.id", id=incident_id).single():
            raise RecordError(f"이미 있는 Incident다: {incident_id}")
        if not tx.run("MATCH (s:Severity {id: $id}) RETURN s.id", id=severity_id).single():
            raise RecordError(f"정의되지 않은 심각도다: {severity_id}")
        tx.run(
            "CREATE (:Incident {id: $id, status: 'open', severity_id: $severity, opened_at: $at, report_count: 0})",
            id=incident_id, severity=severity_id, at=opened_at,
        ).consume()

    with driver.session() as session:
        session.execute_write(write)


def assign_team(driver: Driver, *, incident_id: str, team_id: str, role: str) -> None:
    """사건에 팀을 배정한다 (ASSIGNED_TEAM). 1차 담당은 한 팀만 둔다 (R5-3)."""
    if role not in TEAM_ROLES:
        raise RecordError(f"ASSIGNED_TEAM.role에 쓸 수 없는 값이다: {role}")

    def write(tx) -> None:
        # 사건 노드에 쓰기 잠금을 건 뒤에 1차 담당을 읽는다. 잠금 없이 읽으면 동시에 들어온
        # 두 배정이 모두 "1차 담당 없음"을 보고 통과해 1차 담당이 둘이 된다
        if not tx.run(LOCK.format(label="Incident"), id=incident_id).single():
            raise RecordError(f"없는 Incident다: {incident_id}")
        row = tx.run(
            "MATCH (i:Incident {id: $id}) "
            "RETURN [(i)-[r:ASSIGNED_TEAM {role: 'primary'}]->(t:Team) | t.id] AS primaries",
            id=incident_id,
        ).single()
        if not tx.run("MATCH (t:Team {id: $id}) RETURN t.id", id=team_id).single():
            raise RecordError(f"정의되지 않은 팀이다: {team_id}")
        if role == "primary" and row["primaries"] and row["primaries"] != [team_id]:
            raise RecordError(f"{incident_id}의 1차 담당은 이미 {row['primaries'][0]}이다")
        tx.run(
            "MATCH (i:Incident {id: $id}), (t:Team {id: $team}) "
            "MERGE (i)-[r:ASSIGNED_TEAM]->(t) SET r.role = $role",
            id=incident_id, team=team_id, role=role,
        ).consume()

    with driver.session() as session:
        session.execute_write(write)


def create_action_run(
    driver: Driver,
    *,
    run_id: str,
    incident_id: str,
    action_id: str,
    status: str,
    actor: str,
    requested_at: datetime,
    requested_mode: str,
    gate_blocked: bool,
    required_approver_teams: list[str],
    executed_by_team: str | None,
) -> None:
    """조치 실행 기록을 만든다 (ActionRun 노드, FOR와 OF 관계). 승인 게이트만 부른다."""
    if status not in RUN_STATUSES or actor not in RUN_ACTORS:
        raise RecordError(f"ActionRun에 쓸 수 없는 값이다: status={status}, actor={actor}")
    _require_aware(requested_at, "requested_at")

    def write(tx) -> None:
        created = tx.run(
            "MATCH (i:Incident {id: $incident}), (a:Action {id: $action}) "
            "CREATE (r:ActionRun {id: $id, action_id: $action, status: $status, actor: $actor, "
            "requested_at: $at, requested_mode: $mode, gate_blocked: $blocked, "
            "required_approver_teams: $required, approved_by_teams: []}) "
            "CREATE (r)-[:FOR]->(i) CREATE (r)-[:OF]->(a) "
            "SET r.executed_by_team = $team "
            "RETURN r.id",
            incident=incident_id, action=action_id, id=run_id, status=status, actor=actor,
            at=requested_at, mode=requested_mode, blocked=gate_blocked,
            required=required_approver_teams, team=executed_by_team,
        ).single()
        if created is None:
            raise RecordError(f"없는 Incident 또는 Action이다: {incident_id}, {action_id}")

    with driver.session() as session:
        session.execute_write(write)


RUN_FIELDS = (
    "RETURN r.id AS run_id, r.action_id AS action_id, i.id AS incident_id, r.status AS status, "
    "r.actor AS actor, r.requested_mode AS requested_mode, r.gate_blocked AS gate_blocked, "
    "r.required_approver_teams AS required_approver_teams, r.approved_by_teams AS approved_by_teams, "
    "r.rejected_by_team AS rejected_by_team, r.executed_by_team AS executed_by_team"
)


def read_action_run(driver: Driver, run_id: str) -> dict | None:
    """조치 실행 기록 하나를 읽는다. 없으면 None."""
    with driver.session() as session:
        row = session.execute_read(
            lambda tx: tx.run(
                "MATCH (r:ActionRun {id: $id})-[:FOR]->(i:Incident) " + RUN_FIELDS, id=run_id
            ).single()
        )
    return dict(row) if row else None


def transition_action_run(driver: Driver, run_id: str, decide) -> dict:
    """조치 실행 기록의 상태를 바꾼다. 읽기, 판단, 쓰기를 잠금을 건 한 트랜잭션에서 한다.

    decide는 현재 기록을 받아 바꿀 속성을 돌려주거나, 바꿀 수 없으면 예외를 낸다.
    따로 읽고 따로 쓰면 동시에 들어온 승인과 반려가 서로를 덮어쓴다.
    예를 들어 반려된 기록이 뒤늦게 도착한 승인으로 승인 상태가 될 수 있다.
    승인 게이트만 부른다.
    """
    allowed = {"status", "approved_by_teams", "approved_at", "rejected_by_team", "rejected_at", "done_at"}

    def write(tx) -> dict:
        # 실행 기록 노드에 쓰기 잠금을 건다. 같은 기록을 바꾸려는 다른 트랜잭션은 여기서 기다린다
        if not tx.run(LOCK.format(label="ActionRun"), id=run_id).single():
            raise RecordError(f"없는 ActionRun이다: {run_id}")
        # 잠금을 얻은 뒤에 현재 값을 읽는다
        read = "MATCH (r:ActionRun {id: $id})-[:FOR]->(i:Incident) " + RUN_FIELDS
        changes = decide(dict(tx.run(read, id=run_id).single()))
        if not changes or set(changes) - allowed:
            raise RecordError(f"ActionRun에서 바꿀 수 없는 속성이다: {sorted(set(changes) - allowed)}")
        if "status" in changes and changes["status"] not in RUN_STATUSES:
            raise RecordError(f"ActionRun.status에 쓸 수 없는 값이다: {changes['status']}")
        for name in ("approved_at", "rejected_at", "done_at"):
            if name in changes:
                _require_aware(changes[name], name)
        tx.run("MATCH (r:ActionRun {id: $id}) SET r += $changes", id=run_id, changes=changes).consume()
        return dict(tx.run(read, id=run_id).single())

    with driver.session() as session:
        return session.execute_write(write)


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
