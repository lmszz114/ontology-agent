"""온톨로지 조회 도구 7개의 읽기 쿼리 (설계 5단계 6장).

미리 정한 Cypher만 실행한다. 호출하는 쪽이 넘긴 값은 쿼리 매개변수로만 들어가고,
모든 쿼리는 읽기 전용 트랜잭션에서 돈다. 정의되지 않은 ID는 오류로 돌려준다.
"""

import json
from datetime import datetime

from neo4j import Driver

# R4 원인 점수 (설계 3단계 4-4). 모두 초기 가정값이다.
PRIOR_SCORE = {"높음": 3, "중간": 2, "낮음": 1}
CHANGE_BONUS = 2
CHANGE_WINDOW_HOURS = 24


class InputError(ValueError):
    """호출한 쪽이 넘긴 값이 틀렸다. 내용을 그대로 돌려줘도 되는 오류다."""


class UnknownId(InputError):
    """온톨로지에 정의되지 않은 ID."""


def _read(driver: Driver, query: str, **params) -> list[dict]:
    with driver.session() as session:
        return session.execute_read(lambda tx: tx.run(query, **params).data())


def _require(driver: Driver, label: str, node_id: str) -> None:
    # 레이블은 이 모듈 안의 고정 문자열로만 넘긴다
    if not _read(driver, f"MATCH (n:{label} {{id: $id}}) RETURN n.id AS id", id=node_id):
        raise UnknownId(f"{label}에 정의되지 않은 ID다: {node_id}")


def get_symptom_types(driver: Driver) -> list[dict]:
    """증상 유형 전체. 제보 문장을 이 유형 중 하나로 분류할 때 쓴다 (R1)."""
    return _read(
        driver,
        "MATCH (s:SymptomType) "
        "OPTIONAL MATCH (s)-[:OBSERVED_IN]->(sys:System) "
        "WITH s, sys ORDER BY sys.id "
        "RETURN s.id AS id, s.name AS name, s.aliases AS aliases, s.core_function AS core_function, "
        "s.min_severity AS min_severity, "
        "collect(CASE WHEN sys IS NULL THEN null ELSE {id: sys.id, name: sys.name} END) AS observed_in "
        "ORDER BY id",
    )


def get_cause_candidates(driver: Driver, symptom_id: str) -> list[dict]:
    """증상의 원인 후보 전체와 사전 가능성, 원인 위치 (R4-1)."""
    _require(driver, "SymptomType", symptom_id)
    return _read(
        driver,
        "MATCH (:SymptomType {id: $id})-[m:MAY_BE_CAUSED_BY]->(c:Cause)-[l:LOCATED_IN]->(sys:System) "
        "RETURN c.id AS cause_id, c.name AS name, m.prior AS prior, "
        "{system_id: sys.id, system_name: sys.name, scope: l.scope, area: l.area} AS located_in "
        "ORDER BY cause_id",
        id=symptom_id,
    )


def rank_causes(driver: Driver, symptom_id: str, started_at: str) -> list[dict]:
    """원인 후보에 점수를 매겨 순위를 낸다 (R4-2, R4-3).

    점수 = 사전 가능성 점수 + 변경 가중.
    변경 가중은 원인 위치 시스템에 증상 시작 전 24시간 안의 변경 이력(Change)이 있을 때 붙는다.
    started_at은 증상 시작 시각이다 (시간대가 붙은 ISO 8601).
    """
    try:
        start = datetime.fromisoformat(started_at)
    except ValueError:
        raise InputError(f"started_at이 ISO 8601 형식이 아니다: {started_at}") from None
    if start.tzinfo is None:
        raise InputError(f"started_at에 시간대가 없다: {started_at}")
    _require(driver, "SymptomType", symptom_id)
    # 양쪽 시각을 UTC로 맞춘 뒤 비교한다. datetime을 그대로 비교하면 같은 순간이라도
    # 시간대 표기가 다를 때 경계에서 어긋난다. 소수 초를 버리지 않으려고 초 단위 정수로 바꾸지 않는다
    rows = _read(
        driver,
        "WITH datetime({datetime: $start, timezone: 'UTC'}) AS start "
        "MATCH (:SymptomType {id: $id})-[m:MAY_BE_CAUSED_BY]->(c:Cause)-[:LOCATED_IN]->(sys:System) "
        "OPTIONAL MATCH (ch:Change)-[:CHANGED]->(sys) "
        "WHERE datetime({datetime: ch.changed_at, timezone: 'UTC'}) <= start "
        "AND datetime({datetime: ch.changed_at, timezone: 'UTC'}) >= start - duration({hours: $window}) "
        "WITH c, m, sys, ch, duration.inSeconds(ch.changed_at, start).minutes AS gap ORDER BY ch.changed_at DESC "
        "RETURN c.id AS cause_id, c.name AS name, m.prior AS prior, sys.id AS system_id, "
        # collect는 null을 건너뛴다. 변경 이력이 없는 후보는 빈 목록이 된다
        "collect(CASE WHEN ch IS NULL THEN null ELSE "
        "{id: ch.id, kind: ch.kind, changed_at: toString(ch.changed_at), gap_minutes: gap} END) AS recent_changes",
        id=symptom_id, start=start, window=CHANGE_WINDOW_HOURS,
    )
    for row in rows:
        row["prior_score"] = PRIOR_SCORE[row["prior"]]
        row["change_bonus"] = CHANGE_BONUS if row["recent_changes"] else 0
        row["score"] = row["prior_score"] + row["change_bonus"]
    # 점수가 같으면 원인 ID 순으로 둔다. 같은 입력에 같은 순위가 나와야 평가가 재현된다
    rows.sort(key=lambda r: (-r["score"], r["cause_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def get_checks(driver: Driver, cause_id: str) -> list[dict]:
    """원인의 판정 근거 (R4-4).

    required가 참이면 확정 조건이다. expect대로 나와야 확정이고 다르게 나오면 배제다.
    required가 거짓이면 참고용이다. 확정과 배제에 쓰지 않고 note에 적힌 용도로 본다.
    """
    _require(driver, "Cause", cause_id)
    rows = _read(
        driver,
        "MATCH (:Cause {id: $id})-[r:CHECKED_BY]->(e:EvidenceCheck) "
        "OPTIONAL MATCH (jt:Team {id: e.judged_by_team}) "
        "OPTIONAL MATCH (pt:Team {id: e.performed_by_team}) "
        "RETURN e.id AS check_id, e.name AS name, e.tool AS tool, e.performed_by AS performed_by, "
        "r.required AS required, r.expect AS expect, r.note AS note, r.args AS args, "
        "jt.name AS judged_by_team, pt.name AS performed_by_team "
        "ORDER BY check_id",
        id=cause_id,
    )
    # args는 이 원인을 확인할 때 조회 도구에 넘길 인자다 (예: 어느 시스템의 로그인지)
    for row in rows:
        row["args"] = json.loads(row["args"]) if row["args"] else {}
    return rows


def get_owner_team(driver: Driver, cause_id: str) -> dict:
    """원인의 1차 담당 팀과 협업으로 붙는 팀 (R5).

    1차 담당은 원인 위치의 scope와 area가 담당의 scope, area와 같은 팀이다.
    area가 없는 쪽은 없는 쪽끼리만 맞는다.
    """
    _require(driver, "Cause", cause_id)
    row = _read(
        driver,
        "MATCH (c:Cause {id: $id})-[l:LOCATED_IN]->(sys:System) "
        "RETURN c.id AS cause_id, {system_id: sys.id, system_name: sys.name, scope: l.scope, area: l.area} AS located_in, "
        "[(t:Team)-[o:OWNS]->(sys) WHERE o.scope = l.scope AND coalesce(o.area, '') = coalesce(l.area, '') "
        "| {team_id: t.id, name: t.name}] AS primary, "
        "[(t:Team)-[o:OWNS]->(sys) WHERE NOT (o.scope = l.scope AND coalesce(o.area, '') = coalesce(l.area, '')) "
        "| {team_id: t.id, name: t.name, scope: o.scope, area: o.area}] AS other_owners, "
        "[(t:Team)-[k:COLLABORATES_ON]->(sys) | {team_id: t.id, name: t.name, condition: k.condition}] AS collaborators",
        id=cause_id,
    )[0]
    primary_ids = {t["team_id"] for t in row["primary"]}
    # 같은 시스템의 다른 부분을 맡은 팀. 1차 담당 팀 자신은 뺀다
    others: dict[str, dict] = {}
    for o in sorted(row.pop("other_owners"), key=lambda o: (o["team_id"], o["scope"])):
        if o["team_id"] in primary_ids:
            continue
        entry = others.setdefault(o["team_id"], {"team_id": o["team_id"], "name": o["name"], "owns": []})
        entry["owns"].append({"scope": o["scope"], "area": o["area"]})
    row["other_owners"] = list(others.values())
    row["collaborators"].sort(key=lambda t: t["team_id"])
    return row


def get_procedures(driver: Driver, cause_id: str, severity_id: str | None = None) -> dict:
    """원인이 확정됐을 때 여는 절차와 그 절차의 조치 (R6, R7).

    severity_id를 주면 그 등급이 자동으로 여는 절차(TRIGGERS)도 함께 돌려준다.
    """
    _require(driver, "Cause", cause_id)
    fields = (
        "p.id AS procedure_id, p.name AS name, p.steps AS steps, "
        "head([(p)-[:LED_BY]->(t:Team) | t.name]) AS led_by, actions "
    )
    # 조치는 HAS_ACTION.order 순서로 돌려준다
    actions = (
        "CALL (p) { MATCH (p)-[h:HAS_ACTION]->(a:Action) WITH a, h ORDER BY h.order "
        "RETURN collect({action_id: a.id, name: a.name, risk: a.risk}) AS actions } "
    )
    def by_number(rows: list[dict]) -> list[dict]:
        # P10이 P3 앞에 오지 않게 번호 순으로 둔다
        return sorted(rows, key=lambda r: int(r["procedure_id"][1:]))

    result = {
        "cause_id": cause_id,
        "resolved_by": by_number(_read(
            driver,
            "MATCH (:Cause {id: $id})-[r:RESOLVED_BY]->(p:Procedure) " + actions
            + "RETURN r.condition AS condition, " + fields,
            id=cause_id,
        )),
    }
    if severity_id is not None:
        _require(driver, "Severity", severity_id)
        result["triggered_by_severity"] = by_number(_read(
            driver,
            "MATCH (:Severity {id: $id})-[:TRIGGERS]->(p:Procedure) " + actions
            + "RETURN " + fields,
            id=severity_id,
        ))
    return result


def get_action_policy(driver: Driver, action_id: str) -> dict:
    """조치의 위험도, 자동 실행 여부, 실행 팀, 승인권자 (R6)."""
    _require(driver, "Action", action_id)
    row = _read(
        driver,
        "MATCH (a:Action {id: $id}) "
        "OPTIONAL MATCH (sys:System {id: a.uses_system}) "
        "RETURN a.id AS action_id, a.name AS name, a.kind AS kind, a.risk AS risk, "
        "a.auto_executable AS auto_executable, a.simulated AS simulated, "
        "head([(a)-[:EXECUTED_BY]->(t:Team) | {team_id: t.id, name: t.name}]) AS executed_by, "
        "a.executed_by_rule AS executed_by_rule, "
        "[(a)-[r:REQUIRES_APPROVAL_FROM]->(t:Team) | {team_id: t.id, name: t.name, condition: r.condition}] AS approvers, "
        "a.approver_rule AS approver_rule, sys.name AS uses_system",
        id=action_id,
    )[0]
    row["approvers"].sort(key=lambda t: t["team_id"])
    return row
