"""승인 게이트 (설계 5단계 3-1, 3단계 R6).

LLM은 조치를 요청만 한다. 실행 여부는 이 모듈이 그래프의 Action.risk와 승인 상태를 보고 정한다.
- 저위험: 바로 실행하고 (시뮬레이션) 실행 기록을 auto_done으로 남긴다
- 고위험: 실행하지 않고 실행 기록을 pending_approval로 남긴다. 승인권자 전원이 승인해야 넘어간다
- LLM이 고위험 조치를 자동 실행으로 표시하면 막고 차단 기록을 남긴다 (평가 지표)

고위험 조치를 실행하는 길은 approve를 거친 mark_done뿐이다. 이 함수들은 LLM 도구로 내놓지 않는다.
MVP에서 실행은 모두 시뮬레이션이라 실제로 바뀌는 것은 실행 기록뿐이다.
"""

import uuid
from datetime import datetime

from neo4j import Driver

from common.writers.events import create_action_run, update_action_run
from mcp_servers.ontology_query import queries

REQUESTED_MODES = {"auto", "approval"}


class GateError(ValueError):
    """게이트가 받을 수 없는 요청."""


def _read_run(driver: Driver, run_id: str) -> dict:
    with driver.session() as session:
        row = session.execute_read(
            lambda tx: tx.run(
                "MATCH (r:ActionRun {id: $id})-[:FOR]->(i:Incident) "
                "RETURN r.id AS run_id, r.action_id AS action_id, i.id AS incident_id, r.status AS status, "
                "r.actor AS actor, r.requested_mode AS requested_mode, r.gate_blocked AS gate_blocked, "
                "r.required_approver_teams AS required_approver_teams, "
                "r.approved_by_teams AS approved_by_teams, r.rejected_by_team AS rejected_by_team, "
                "r.executed_by_team AS executed_by_team",
                id=run_id,
            ).single()
        )
    if row is None:
        raise GateError(f"없는 실행 기록이다: {run_id}")
    return dict(row)


def _primary_team(driver: Driver, incident_id: str) -> str | None:
    with driver.session() as session:
        row = session.execute_read(
            lambda tx: tx.run(
                "MATCH (i:Incident {id: $id}) "
                "RETURN head([(i)-[:ASSIGNED_TEAM {role: 'primary'}]->(t:Team) | t.id]) AS team",
                id=incident_id,
            ).single()
        )
    if row is None:
        raise GateError(f"없는 사건이다: {incident_id}")
    return row["team"]


def _required_approvers(policy: dict, condition: str | None) -> list[str]:
    """이 요청에 필요한 승인 팀. 조건이 붙은 승인권자는 조건이 같을 때만 필요하다."""
    if policy["approver_rule"]:
        # 승인권자가 규칙으로 정해지는 조치(A17)는 추출 에이전트를 만들 때 지원한다
        raise GateError(f"{policy['action_id']}: 승인권자 규칙 {policy['approver_rule']}은 아직 지원하지 않는다")
    conditions = {a["condition"] for a in policy["approvers"] if a["condition"]}
    if conditions and condition not in conditions:
        raise GateError(f"{policy['action_id']}: condition은 {sorted(conditions)} 중 하나여야 한다: {condition}")
    if not conditions and condition is not None:
        raise GateError(f"{policy['action_id']}: 조건이 없는 조치에 condition을 줄 수 없다: {condition}")
    teams = [a["team_id"] for a in policy["approvers"] if a["condition"] in (None, condition)]
    if not teams:
        raise GateError(f"{policy['action_id']}: 고위험 조치인데 승인권자가 없다")
    return teams


def request_action(
    driver: Driver,
    *,
    incident_id: str,
    action_id: str,
    requested_mode: str,
    now: datetime,
    condition: str | None = None,
) -> dict:
    """조치 요청 하나를 처리하고 실행 기록을 돌려준다.

    requested_mode는 요청한 쪽(LLM)이 표시한 값이다. auto는 자동 실행, approval은 승인 요청.
    실행 여부는 이 값이 아니라 그래프의 Action.risk로 정한다.
    condition은 승인권자가 조건에 따라 달라지는 조치(A15)에서 어느 조건인지 가리킨다.
    """
    if requested_mode not in REQUESTED_MODES:
        raise GateError(f"requested_mode는 auto 또는 approval이어야 한다: {requested_mode}")
    try:
        policy = queries.get_action_policy(driver, action_id)
    except queries.UnknownId as exc:
        raise GateError(str(exc)) from exc
    primary = _primary_team(driver, incident_id)

    executed_by = (policy["executed_by"] or {}).get("team_id")
    if policy["executed_by_rule"] == "incident_primary_team":
        if primary is None:
            raise GateError(f"{action_id}: 1차 담당 팀이 정해지지 않은 사건이다: {incident_id}")
        executed_by = primary

    high = policy["risk"] == "high"
    run = {
        "run_id": f"AR-{uuid.uuid4().hex[:12]}",
        "incident_id": incident_id,
        "action_id": action_id,
        "status": "pending_approval" if high else "auto_done",
        # 사람이 실행하는 조치는 승인 뒤 실행 팀이 처리한다 (R6)
        "actor": "human" if executed_by else "agent",
        "requested_at": now,
        "requested_mode": requested_mode,
        # 고위험 조치를 자동 실행으로 표시한 요청을 막았다는 기록
        "gate_blocked": high and requested_mode == "auto",
        "required_approver_teams": _required_approvers(policy, condition) if high else [],
        "executed_by_team": executed_by,
    }
    if not high and condition is not None:
        raise GateError(f"{action_id}: 저위험 조치에 condition을 줄 수 없다")
    create_action_run(driver, **run)
    return _read_run(driver, run["run_id"])


def approve(driver: Driver, *, run_id: str, team_id: str, now: datetime) -> dict:
    """승인권자 한 팀의 승인을 기록한다. 전원이 승인하면 approved가 된다."""
    run = _read_run(driver, run_id)
    if run["status"] != "pending_approval":
        raise GateError(f"{run_id}: 승인 대기 상태가 아니다: {run['status']}")
    if team_id not in run["required_approver_teams"]:
        raise GateError(f"{run_id}: {team_id}은 이 조치의 승인권자가 아니다")
    approved = sorted(set(run["approved_by_teams"]) | {team_id})
    changes = {"approved_by_teams": approved}
    if set(approved) == set(run["required_approver_teams"]):
        changes |= {"status": "approved", "approved_at": now}
    update_action_run(driver, run_id=run_id, **changes)
    return _read_run(driver, run_id)


def reject(driver: Driver, *, run_id: str, team_id: str, now: datetime) -> dict:
    """승인권자 한 팀이 반려하면 실행하지 않는다. 다른 절차를 다시 고르는 것은 에이전트 몫이다 (R6)."""
    run = _read_run(driver, run_id)
    if run["status"] != "pending_approval":
        raise GateError(f"{run_id}: 승인 대기 상태가 아니다: {run['status']}")
    if team_id not in run["required_approver_teams"]:
        raise GateError(f"{run_id}: {team_id}은 이 조치의 승인권자가 아니다")
    update_action_run(driver, run_id=run_id, status="rejected", rejected_by_team=team_id, rejected_at=now)
    return _read_run(driver, run_id)


def mark_done(driver: Driver, *, run_id: str, now: datetime) -> dict:
    """승인된 조치의 실행 완료를 기록한다 (시뮬레이션). 승인되지 않은 조치는 완료할 수 없다."""
    run = _read_run(driver, run_id)
    if run["status"] != "approved":
        raise GateError(f"{run_id}: 승인된 조치만 실행할 수 있다: {run['status']}")
    update_action_run(driver, run_id=run_id, status="done", done_at=now)
    return _read_run(driver, run_id)
