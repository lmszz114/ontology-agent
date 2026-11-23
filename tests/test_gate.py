"""승인 게이트 검증 (R6). 시드를 적재한 테스트용 Neo4j에서 돈다."""

from datetime import datetime, timedelta, timezone

import pytest

from common.gate import gate
from common.gate.gate import GateError
from common.writers.events import RecordError, assign_team, record_incident, reset_event_layer
from ontology.apply_schema import apply_schema
from ontology.load import load_static
from ontology.seed import load_seed

pytestmark = pytest.mark.integration

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 12, 3, 20, 30, tzinfo=KST)
LATER = NOW + timedelta(minutes=10)


@pytest.fixture(scope="module")
def seed():
    return load_seed("org-a")


@pytest.fixture(scope="module")
def db(driver, seed):
    apply_schema(driver)
    load_static(driver, seed)
    return driver


@pytest.fixture
def incident(db):
    """1차 담당이 서버개발팀인 사건 하나."""
    reset_event_layer(db)
    record_incident(db, incident_id="INC-1", severity_id="SEV2", opened_at=NOW)
    assign_team(db, incident_id="INC-1", team_id="TM03", role="primary")
    yield "INC-1"
    reset_event_layer(db)


def request(db, incident, action_id, mode="approval", **kwargs):
    return gate.request_action(db, incident_id=incident, action_id=action_id, requested_mode=mode, now=NOW, **kwargs)


def test_low_risk_action_runs_automatically(db, incident):
    run = request(db, incident, "A01", mode="auto")
    assert run["status"] == "auto_done" and run["actor"] == "agent"
    assert run["gate_blocked"] is False and run["required_approver_teams"] == []


def test_high_risk_action_waits_for_approval(db, incident):
    run = request(db, incident, "A10")
    assert run["status"] == "pending_approval" and run["gate_blocked"] is False
    assert run["required_approver_teams"] == ["TM06"]
    assert (run["actor"], run["executed_by_team"]) == ("human", "TM07")


def test_high_risk_marked_auto_is_blocked(db, incident):
    """LLM이 고위험 조치를 자동 실행으로 표시해도 실행하지 않고 차단 기록을 남긴다."""
    run = request(db, incident, "A10", mode="auto")
    assert run["status"] == "pending_approval" and run["gate_blocked"] is True


def test_every_action_follows_its_risk(db, incident, seed):
    """17개 조치 전부: 요청 표시와 상관없이 위험도대로 처리된다. A17은 승인권자 규칙이 아직 없다."""
    for action in seed["actions"]:
        if action["id"] == "A17":
            with pytest.raises(GateError, match="아직 지원하지 않는다"):
                request(db, incident, "A17", mode="auto")
            continue
        condition = "긴급 점검 반영" if action["id"] == "A15" else None
        for mode in ("auto", "approval"):
            run = request(db, incident, action["id"], mode=mode, condition=condition)
            if action["risk"] == "high":
                assert run["status"] == "pending_approval", action["id"]
                assert run["gate_blocked"] is (mode == "auto"), action["id"]
                assert run["required_approver_teams"], action["id"]
            else:
                assert run["status"] == "auto_done" and run["gate_blocked"] is False, action["id"]


def test_all_approvers_must_approve(db, incident):
    """긴급 점검(A08)은 게임사업PM팀과 인프라운영팀이 모두 승인해야 한다."""
    run = request(db, incident, "A08")
    assert run["required_approver_teams"] == ["TM06", "TM10"]
    run = gate.approve(db, run_id=run["run_id"], team_id="TM06", now=LATER)
    assert run["status"] == "pending_approval" and run["approved_by_teams"] == ["TM06"]
    with pytest.raises(GateError, match="승인된 조치만"):
        gate.mark_done(db, run_id=run["run_id"], now=LATER)
    run = gate.approve(db, run_id=run["run_id"], team_id="TM10", now=LATER)
    assert run["status"] == "approved" and run["approved_by_teams"] == ["TM06", "TM10"]
    assert gate.mark_done(db, run_id=run["run_id"], now=LATER)["status"] == "done"


def test_only_approvers_can_decide(db, incident):
    run = request(db, incident, "A10")
    for decide in (gate.approve, gate.reject):
        with pytest.raises(GateError, match="승인권자가 아니다"):
            decide(db, run_id=run["run_id"], team_id="TM03", now=LATER)
    assert gate._read_run(db, run["run_id"])["status"] == "pending_approval"


def test_rejected_action_cannot_run(db, incident):
    run = request(db, incident, "A08")
    run = gate.reject(db, run_id=run["run_id"], team_id="TM10", now=LATER)
    assert (run["status"], run["rejected_by_team"]) == ("rejected", "TM10")
    for call in (
        lambda: gate.approve(db, run_id=run["run_id"], team_id="TM06", now=LATER),
        lambda: gate.mark_done(db, run_id=run["run_id"], now=LATER),
    ):
        with pytest.raises(GateError):
            call()


def test_pending_and_auto_done_cannot_be_marked_done(db, incident):
    """승인 없이 고위험 조치를 완료 처리하는 길이 없다 (무단 고위험 실행 0건)."""
    for action_id in ("A10", "A01"):
        run = request(db, incident, action_id)
        with pytest.raises(GateError, match="승인된 조치만"):
            gate.mark_done(db, run_id=run["run_id"], now=LATER)


@pytest.mark.parametrize("condition,approvers", [
    ("정기 점검 반영", ["TM05"]),
    ("긴급 점검 반영", ["TM06", "TM10"]),
])
def test_cause_fix_approvers_depend_on_condition(db, incident, condition, approvers):
    """원인 수정(A15)은 반영 시점에 따라 승인권자가 다르고, 사건의 1차 담당 팀이 실행한다 (R5, R7)."""
    run = request(db, incident, "A15", condition=condition)
    assert run["required_approver_teams"] == approvers
    assert (run["actor"], run["executed_by_team"]) == ("human", "TM03")


@pytest.mark.parametrize("action_id,kwargs,message", [
    ("A15", {}, "condition"),                              # 조건이 필요한 조치
    ("A15", {"condition": "아무 때나"}, "condition"),
    ("A10", {"condition": "긴급 점검 반영"}, "condition"),   # 조건이 없는 조치
    ("A01", {"condition": "긴급 점검 반영"}, "condition"),
    ("A99", {}, "A99"),
    ("A10", {"mode": "force"}, "requested_mode"),
])
def test_invalid_requests_are_rejected(db, incident, action_id, kwargs, message):
    with pytest.raises(GateError, match=message):
        request(db, incident, action_id, **kwargs)
    with db.session() as session:
        assert session.run("MATCH (r:ActionRun) RETURN count(r) AS c").single()["c"] == 0


def test_unknown_incident_is_rejected(db, incident):
    with pytest.raises(GateError, match="없는 사건"):
        request(db, "INC-404", "A01")


def test_cause_fix_needs_primary_team(db, incident):
    record_incident(db, incident_id="INC-2", severity_id="SEV3", opened_at=NOW)
    with pytest.raises(GateError, match="1차 담당"):
        request(db, "INC-2", "A15", condition="정기 점검 반영")


def test_run_is_linked_to_incident_and_action(db, incident):
    run = request(db, incident, "A10")
    with db.session() as session:
        row = session.run(
            "MATCH (i:Incident)<-[:FOR]-(r:ActionRun {id: $id})-[:OF]->(a:Action) RETURN i.id AS i, a.id AS a",
            id=run["run_id"],
        ).single()
    assert (row["i"], row["a"]) == ("INC-1", "A10")


def test_incident_has_one_primary_team(db, incident):
    with pytest.raises(RecordError, match="1차 담당은 이미"):
        assign_team(db, incident_id=incident, team_id="TM10", role="primary")
    assign_team(db, incident_id=incident, team_id="TM10", role="collaborator")
    assign_team(db, incident_id=incident, team_id="TM03", role="primary")  # 같은 팀을 다시 배정해도 된다
    with db.session() as session:
        rows = session.run(
            "MATCH (:Incident {id: $id})-[r:ASSIGNED_TEAM]->(t:Team) RETURN t.id AS team, r.role AS role ORDER BY team",
            id=incident,
        ).data()
    assert rows == [{"team": "TM03", "role": "primary"}, {"team": "TM10", "role": "collaborator"}]


@pytest.mark.parametrize("call,message", [
    (lambda d: record_incident(d, incident_id="INC-1", severity_id="SEV2", opened_at=NOW), "이미 있는"),
    (lambda d: record_incident(d, incident_id="INC-9", severity_id="SEV9", opened_at=NOW), "심각도"),
    (lambda d: record_incident(d, incident_id="INC-9", severity_id="SEV2", opened_at=datetime(2026, 12, 3)), "시간대"),
    (lambda d: assign_team(d, incident_id="INC-1", team_id="TM99", role="primary"), "팀"),
    (lambda d: assign_team(d, incident_id="INC-1", team_id="TM03", role="owner"), "role"),
    (lambda d: assign_team(d, incident_id="INC-404", team_id="TM03", role="primary"), "없는 Incident"),
])
def test_incident_writers_reject_undefined_values(db, incident, call, message):
    with pytest.raises(RecordError, match=message):
        call(db)
