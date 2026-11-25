"""사건 계층 기록 함수 (설계 3단계 2-2, 3-2, 5단계 1장).

LLM은 그래프에 직접 쓰지 않는다. 사건 계층은 이 모듈의 함수가 정해진 형식으로만 쓴다.
정의되지 않은 값(없는 시스템, 허용되지 않은 종류, 시간대 없는 시각)은 쓰기 전에 거부한다.
함수는 그것을 쓰는 단계를 만들 때 함께 추가한다.
"""

import json
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
    incident_id: str | None = None,
    action_id: str,
    status: str,
    actor: str,
    requested_at: datetime,
    requested_mode: str,
    gate_blocked: bool,
    required_approver_teams: list[str],
    executed_by_team: str | None,
    target_report_id: str | None = None,
    candidate_ids: list[str] | None = None,
) -> None:
    """조치 실행 기록을 만든다 (ActionRun 노드, FOR와 OF 관계). 승인 게이트만 부른다.

    target_report_id가 있으면 그 접수를 대상으로 한 조치다 (예: 묶인 제보의 처리 중 회신). 사건에 속한 접수여야 한다.
    지식 반영(A17)은 사건이 아니라 추출 후보에 붙는다(FOR Candidate, M4 PR 5 결정 1). incident_id와 candidate_ids 중
    하나만 준다.
    """
    if status not in RUN_STATUSES or actor not in RUN_ACTORS:
        raise RecordError(f"ActionRun에 쓸 수 없는 값이다: status={status}, actor={actor}")
    if (incident_id is None) == (not candidate_ids):
        raise RecordError("ActionRun은 사건(incident_id)이나 후보(candidate_ids) 중 하나에만 붙는다")
    _require_aware(requested_at, "requested_at")
    if candidate_ids:
        _create_candidate_run(driver, run_id=run_id, candidate_ids=candidate_ids, action_id=action_id, status=status,
                              actor=actor, requested_at=requested_at, requested_mode=requested_mode,
                              gate_blocked=gate_blocked, required=required_approver_teams, team=executed_by_team)
        return

    def write(tx) -> None:
        if target_report_id is not None and not tx.run(
            "MATCH (:Report {id: $report})-[:PART_OF]->(:Incident {id: $incident}) RETURN 1",
            report=target_report_id, incident=incident_id,
        ).single():
            raise RecordError(f"조치 대상 접수 {target_report_id}는 사건 {incident_id}에 속하지 않는다")
        created = tx.run(
            "MATCH (i:Incident {id: $incident}), (a:Action {id: $action}) "
            "CREATE (r:ActionRun {id: $id, action_id: $action, status: $status, actor: $actor, "
            "requested_at: $at, requested_mode: $mode, gate_blocked: $blocked, "
            "required_approver_teams: $required, approved_by_teams: []}) "
            "CREATE (r)-[:FOR]->(i) CREATE (r)-[:OF]->(a) "
            "SET r.executed_by_team = $team, r.target_report_id = $report "
            "RETURN r.id",
            incident=incident_id, action=action_id, id=run_id, status=status, actor=actor,
            at=requested_at, mode=requested_mode, blocked=gate_blocked,
            required=required_approver_teams, team=executed_by_team, report=target_report_id,
        ).single()
        if created is None:
            raise RecordError(f"없는 Incident 또는 Action이다: {incident_id}, {action_id}")

    with driver.session() as session:
        session.execute_write(write)


def _create_candidate_run(driver: Driver, *, run_id: str, candidate_ids: list[str], action_id: str, status: str,
                          actor: str, requested_at: datetime, requested_mode: str, gate_blocked: bool,
                          required: list[str], team: str | None) -> None:
    def write(tx) -> None:
        found = tx.run("MATCH (c:Candidate) WHERE c.id IN $ids RETURN collect(c.id) AS ids", ids=candidate_ids).single()
        missing = sorted(set(candidate_ids) - set(found["ids"]))
        if missing:
            raise RecordError(f"없는 Candidate다: {missing}")
        created = tx.run(
            "MATCH (a:Action {id: $action}) "
            "CREATE (r:ActionRun {id: $id, action_id: $action, status: $status, actor: $actor, "
            "requested_at: $at, requested_mode: $mode, gate_blocked: $blocked, "
            "required_approver_teams: $required, approved_by_teams: []})-[:OF]->(a) "
            "SET r.executed_by_team = $team "
            "WITH r MATCH (c:Candidate) WHERE c.id IN $ids CREATE (r)-[:FOR]->(c) RETURN count(c) AS n",
            action=action_id, id=run_id, status=status, actor=actor, at=requested_at, mode=requested_mode,
            blocked=gate_blocked, required=required, team=team, ids=candidate_ids,
        ).single()
        if created is None or created["n"] != len(set(candidate_ids)):
            raise RecordError(f"없는 Action이다: {action_id}")

    with driver.session() as session:
        session.execute_write(write)


# 사건에 붙은 실행 기록과 후보에 붙은 실행 기록(A17)을 같은 모양으로 읽는다
RUN_MATCH = ("MATCH (r:ActionRun {id: $id}) OPTIONAL MATCH (r)-[:FOR]->(i:Incident) "
             "OPTIONAL MATCH (r)-[:FOR]->(c:Candidate) WITH r, i, collect(c.id) AS candidate_ids ")

RUN_FIELDS = (
    "RETURN r.id AS run_id, r.action_id AS action_id, i.id AS incident_id, r.status AS status, "
    "r.actor AS actor, r.requested_mode AS requested_mode, r.gate_blocked AS gate_blocked, "
    "r.required_approver_teams AS required_approver_teams, r.approved_by_teams AS approved_by_teams, "
    "r.rejected_by_team AS rejected_by_team, r.executed_by_team AS executed_by_team, "
    "r.target_report_id AS target_report_id, candidate_ids, r.commit AS commit"
)


def read_action_run(driver: Driver, run_id: str) -> dict | None:
    """조치 실행 기록 하나를 읽는다. 없으면 None."""
    with driver.session() as session:
        row = session.execute_read(
            lambda tx: tx.run(
                RUN_MATCH + RUN_FIELDS, id=run_id
            ).single()
        )
    return dict(row) if row else None


def list_knowledge_runs(driver: Driver) -> list[dict]:
    """후보에 붙은 지식 반영(A17) 실행 기록 전체. 검수 화면이 읽는다."""
    with driver.session() as session:
        ids = session.execute_read(lambda tx: tx.run(
            "MATCH (r:ActionRun)-[:FOR]->(:Candidate) RETURN DISTINCT r.id AS id ORDER BY id").value())
    return [read_action_run(driver, run_id) for run_id in ids]


def list_changes(driver: Driver) -> list[dict]:
    """기록된 변경 이력 전체 (사건 데이터). 구성 A의 원인 순위가 변경 가중을 판단할 때 읽는다."""
    with driver.session() as session:
        rows = session.execute_read(lambda tx: tx.run(
            "MATCH (c:Change)-[:CHANGED]->(sys:System) "
            "RETURN c.id AS id, c.kind AS kind, sys.id AS system_id, toString(c.changed_at) AS changed_at "
            "ORDER BY id"
        ).data())
    return rows


def transition_action_run(driver: Driver, run_id: str, decide) -> dict:
    """조치 실행 기록의 상태를 바꾼다. 읽기, 판단, 쓰기를 잠금을 건 한 트랜잭션에서 한다.

    decide는 현재 기록을 받아 바꿀 속성을 돌려주거나, 바꿀 수 없으면 예외를 낸다.
    따로 읽고 따로 쓰면 동시에 들어온 승인과 반려가 서로를 덮어쓴다.
    예를 들어 반려된 기록이 뒤늦게 도착한 승인으로 승인 상태가 될 수 있다.
    승인 게이트만 부른다.
    """
    allowed = {"status", "approved_by_teams", "approved_at", "rejected_by_team", "rejected_at", "done_at", "commit"}

    def write(tx) -> dict:
        # 실행 기록 노드에 쓰기 잠금을 건다. 같은 기록을 바꾸려는 다른 트랜잭션은 여기서 기다린다
        if not tx.run(LOCK.format(label="ActionRun"), id=run_id).single():
            raise RecordError(f"없는 ActionRun이다: {run_id}")
        # 잠금을 얻은 뒤에 현재 값을 읽는다
        read = RUN_MATCH + RUN_FIELDS
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
            # 지식 반영(A17) 기록은 후보에 붙어 있어 사건 계층과 함께 지우지 않는다 (M4 PR 5 결정 1)
            keep = " WHERE NOT (n)-[:FOR]->(:Candidate)" if label == "ActionRun" else ""
            summary = session.run(f"MATCH (n:{label}){keep} DETACH DELETE n").consume()
            deleted += summary.counters.nodes_deleted
    return deleted


# INDICATES로 쓸 수 있는 영향 범위 (설계 3단계 4-3)와 가설, 근거 결과의 허용 값 (3단계 2-2, 3-2)
REPORT_SCOPES = {"all_service", "one_world", "some_users"}
HYPOTHESIS_STATUSES = {"candidate", "confirmed", "ruled_out"}
FINDING_RESULTS = {"confirm", "rule_out", "inconclusive"}
REPORT_LINKS = {"part_of": "PART_OF", "suspected_duplicate": "SUSPECTED_DUPLICATE_OF"}


def record_report(
    driver: Driver,
    *,
    report_id: str,
    channel_id: str,
    reporter_id: str,
    raw_text: str,
    received_at: datetime,
    symptom_id: str | None,
    scope: str | None,
) -> None:
    """접수 하나를 기록한다 (Report, RECEIVED_VIA, 분류했으면 INDICATES) (R1).

    symptom_id가 None이면 증상 8개에 해당하지 않는 제보다. INDICATES를 만들지 않는다.
    """
    _require_aware(received_at, "received_at")
    if scope is not None and scope not in REPORT_SCOPES:
        raise RecordError(f"영향 범위에 쓸 수 없는 값이다: {scope}")

    def write(tx) -> None:
        if tx.run("MATCH (n:Report {id: $id}) RETURN n.id", id=report_id).single():
            raise RecordError(f"이미 있는 Report다: {report_id}")
        if symptom_id is not None and not tx.run(
            "MATCH (s:SymptomType {id: $id}) RETURN s.id", id=symptom_id
        ).single():
            raise RecordError(f"정의되지 않은 증상이다: {symptom_id}")
        created = tx.run(
            "MATCH (ch:Channel {id: $channel}) "
            "CREATE (r:Report {id: $id, channel_id: $channel, reporter_id: $reporter, raw_text: $text, "
            "received_at: $at, extracted_symptom_id: $symptom, extracted_scope: $scope})-[:RECEIVED_VIA]->(ch) "
            "RETURN r.id",
            channel=channel_id, id=report_id, reporter=reporter_id, text=raw_text,
            at=received_at, symptom=symptom_id, scope=scope,
        ).single()
        if created is None:
            raise RecordError(f"정의되지 않은 채널이다: {channel_id}")
        if symptom_id is not None:
            tx.run(
                "MATCH (r:Report {id: $id}), (s:SymptomType {id: $symptom}) CREATE (r)-[:INDICATES]->(s)",
                id=report_id, symptom=symptom_id,
            ).consume()

    with driver.session() as session:
        session.execute_write(write)


def link_report(driver: Driver, *, report_id: str, incident_id: str, link: str, linked_by: str = "agent") -> None:
    """접수를 사건에 묶는다 (R2). part_of는 같은 사건, suspected_duplicate는 중복 의심(사람 확인)이다.

    같은 사건으로 묶으면 사건의 접수 수(report_count)를 하나 올린다. 등급 상향(R3)이 이 값을 읽는다.
    """
    if link not in REPORT_LINKS:
        raise RecordError(f"접수 연결에 쓸 수 없는 값이다: {link}")

    def write(tx) -> None:
        if not tx.run(LOCK.format(label="Incident"), id=incident_id).single():
            raise RecordError(f"없는 Incident다: {incident_id}")
        # 접수에도 잠금을 건다. 같은 접수를 두 사건에 동시에 묶는 요청이 둘 다 통과하지 않게 한다
        if not tx.run(LOCK.format(label="Report"), id=report_id).single():
            raise RecordError(f"없는 Report다: {report_id}")
        rel = REPORT_LINKS[link]
        # 접수 하나는 한 사건에만 속한다. 다른 사건에 이미 속하면 막는다. 접수 수가 두 사건에 함께 세어지면
        # 상향(R3)이 틀어진다. 중복 의심은 속한 사건이 아닌 다른 사건에만 건다
        owner = tx.run("MATCH (:Report {id: $id})-[:PART_OF]->(i:Incident) RETURN i.id AS id", id=report_id).single()
        if owner and owner["id"] != incident_id and link == "part_of":
            raise RecordError(f"접수 {report_id}는 이미 사건 {owner['id']}에 속한다")
        if owner and owner["id"] == incident_id and link == "suspected_duplicate":
            raise RecordError(f"접수 {report_id}는 사건 {incident_id}에 속해 있어 그 사건의 중복 의심으로 걸 수 없다")
        # 이미 같은 연결이 있으면 관계와 접수 수를 다시 늘리지 않는다. 잠금을 건 뒤에 확인한다
        if tx.run(
            f"MATCH (:Report {{id: $report}})-[:{rel}]->(:Incident {{id: $incident}}) RETURN 1",
            report=report_id, incident=incident_id,
        ).single():
            return
        tx.run(
            f"MATCH (r:Report {{id: $report}}), (i:Incident {{id: $incident}}) "
            f"CREATE (r)-[:{rel} {{linked_by: $by}}]->(i)",
            report=report_id, incident=incident_id, by=linked_by,
        ).consume()
        if link == "part_of":
            # 같은 사건으로 확정되면 그 사건에 걸려 있던 중복 의심은 지운다. 두 관계가 함께 남으면 안 된다
            tx.run(
                "MATCH (:Report {id: $report})-[s:SUSPECTED_DUPLICATE_OF]->(:Incident {id: $incident}) DELETE s",
                report=report_id, incident=incident_id,
            ).consume()
            tx.run(
                "MATCH (i:Incident {id: $id}) SET i.report_count = i.report_count + 1", id=incident_id
            ).consume()

    with driver.session() as session:
        session.execute_write(write)


def set_incident_severity(driver: Driver, *, incident_id: str, severity_id: str) -> None:
    """사건의 심각도를 바꾼다. 접수가 늘어 등급이 상향될 때 쓴다 (R3)."""
    def write(tx) -> None:
        if not tx.run("MATCH (s:Severity {id: $id}) RETURN s.id", id=severity_id).single():
            raise RecordError(f"정의되지 않은 심각도다: {severity_id}")
        if not tx.run(
            "MATCH (i:Incident {id: $id}) SET i.severity_id = $severity RETURN i.id",
            id=incident_id, severity=severity_id,
        ).single():
            raise RecordError(f"없는 Incident다: {incident_id}")

    with driver.session() as session:
        session.execute_write(write)


def record_affects(driver: Driver, *, incident_id: str, system_ids: list[str]) -> None:
    """사건이 영향을 준 시스템을 기록한다 (AFFECTS). 정의되지 않은 시스템이 하나라도 있으면 쓰지 않는다."""
    def write(tx) -> None:
        known = tx.run(
            "MATCH (s:System) WHERE s.id IN $ids RETURN collect(s.id) AS ids", ids=system_ids
        ).single()["ids"]
        unknown = sorted(set(system_ids) - set(known))
        if unknown:
            raise RecordError(f"정의되지 않은 시스템이다: {unknown}")
        if not tx.run("MATCH (i:Incident {id: $id}) RETURN i.id", id=incident_id).single():
            raise RecordError(f"없는 Incident다: {incident_id}")
        tx.run(
            "MATCH (i:Incident {id: $id}) UNWIND $ids AS sid MATCH (s:System {id: sid}) "
            "MERGE (i)-[:AFFECTS]->(s)",
            id=incident_id, ids=system_ids,
        ).consume()

    with driver.session() as session:
        session.execute_write(write)


def record_started_after(driver: Driver, *, incident_id: str, change_id: str, gap_minutes: int) -> None:
    """증상 시작 직전의 변경 이력을 사건에 잇는다 (STARTED_AFTER). 원인 순위의 변경 가중 근거다 (R4)."""
    def write(tx) -> None:
        if not tx.run(
            "MATCH (i:Incident {id: $incident}), (c:Change {id: $change}) "
            "MERGE (i)-[r:STARTED_AFTER]->(c) SET r.gap_minutes = $gap RETURN i.id",
            incident=incident_id, change=change_id, gap=gap_minutes,
        ).single():
            raise RecordError(f"없는 Incident 또는 Change다: {incident_id}, {change_id}")

    with driver.session() as session:
        session.execute_write(write)


def record_hypothesis(
    driver: Driver, *, incident_id: str, cause_id: str, score: int, rank: int, status: str
) -> None:
    """원인 가설을 기록하거나 상태를 바꾼다 (HAS_HYPOTHESIS) (R4)."""
    if status not in HYPOTHESIS_STATUSES:
        raise RecordError(f"가설 상태에 쓸 수 없는 값이다: {status}")

    def write(tx) -> None:
        if not tx.run(
            "MATCH (i:Incident {id: $incident}), (c:Cause {id: $cause}) "
            "MERGE (i)-[h:HAS_HYPOTHESIS]->(c) SET h.score = $score, h.rank = $rank, h.status = $status "
            "RETURN i.id",
            incident=incident_id, cause=cause_id, score=score, rank=rank, status=status,
        ).single():
            raise RecordError(f"없는 Incident 또는 Cause다: {incident_id}, {cause_id}")

    with driver.session() as session:
        session.execute_write(write)


def record_finding(
    driver: Driver,
    *,
    finding_id: str,
    incident_id: str,
    check_id: str,
    cause_id: str,
    result: str,
    detail: str,
    checked_at: datetime,
    tool: str,
    args: dict,
) -> None:
    """판정 근거 확인 결과를 기록한다 (Finding, FOR, CHECKS) (R4-4).

    result는 이 근거로 cause_id 가설을 확정, 배제, 불명확 중 무엇으로 판단했는지다.
    tool과 args는 이 결과를 낸 조회다. 같은 근거를 다른 인자로 조회한 결과를 구분한다.
    Neo4j 속성은 맵을 담지 못해서 args는 JSON 문자열로 둔다.
    """
    if result not in FINDING_RESULTS:
        raise RecordError(f"Finding.result에 쓸 수 없는 값이다: {result}")
    _require_aware(checked_at, "checked_at")

    def write(tx) -> None:
        if tx.run("MATCH (n:Finding {id: $id}) RETURN n.id", id=finding_id).single():
            raise RecordError(f"이미 있는 Finding이다: {finding_id}")
        if not tx.run(
            "MATCH (i:Incident {id: $incident}), (e:EvidenceCheck {id: $check}), (:Cause {id: $cause}) "
            "CREATE (f:Finding {id: $id, check_id: $check, cause_id: $cause, result: $result, "
            "detail: $detail, checked_at: $at, tool: $tool, args: $args}) "
            "CREATE (f)-[:FOR]->(i) CREATE (f)-[:CHECKS]->(e) RETURN f.id",
            incident=incident_id, check=check_id, cause=cause_id, id=finding_id,
            result=result, detail=detail, at=checked_at,
            tool=tool, args=json.dumps(args, ensure_ascii=False, sort_keys=True),
        ).single():
            raise RecordError(f"없는 Incident, EvidenceCheck 또는 Cause다: {incident_id}, {check_id}, {cause_id}")

    with driver.session() as session:
        session.execute_write(write)


def open_incidents(driver: Driver, *, since: datetime) -> list[dict]:
    """중복 판정의 비교 대상 (R2). closed가 아니고 since 이후에 열린 사건과 그 접수, 영향 시스템."""
    _require_aware(since, "since")
    with driver.session() as session:
        rows = session.execute_read(lambda tx: tx.run(
            "MATCH (i:Incident) WHERE i.status <> 'closed' "
            "AND datetime({datetime: i.opened_at, timezone: 'UTC'}) >= datetime({datetime: $since, timezone: 'UTC'}) "
            "RETURN i.id AS incident_id, toString(i.opened_at) AS opened_at, i.severity_id AS severity_id, "
            "i.report_count AS report_count, "
            "[(r:Report)-[:PART_OF]->(i) | {report_id: r.id, reporter_id: r.reporter_id, "
            "symptom_id: r.extracted_symptom_id, received_at: toString(r.received_at), text: r.raw_text}] AS reports, "
            "[(i)-[:AFFECTS]->(s:System) | s.id] AS systems "
            "ORDER BY i.opened_at",
            since=since,
        ).data())
    return rows
