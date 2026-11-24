"""사건 계층 기록 함수 검증. 시드를 적재한 테스트용 Neo4j에서 돈다."""

from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from common.writers.events import RecordError, record_change, reset_event_layer
from mcp_servers.ontology_query import queries
from ontology.apply_schema import apply_schema
from ontology.load import STATIC_LABELS, load_static
from ontology.seed import load_seed

pytestmark = pytest.mark.integration

KST = timezone(timedelta(hours=9))
AT = datetime(2026, 12, 3, 19, 40, tzinfo=KST)


@pytest.fixture(scope="module")
def db(driver):
    apply_schema(driver)
    load_static(driver, load_seed("org-a"))
    return driver


@pytest.fixture(autouse=True)
def clean(db):
    reset_event_layer(db)
    yield
    reset_event_layer(db)


def count(db, label):
    with db.session() as session:
        return session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]


def test_record_change_is_read_by_rank_causes(db):
    """기록 함수가 쓴 변경 이력을 원인 순위가 그대로 읽는다 (설계 4단계 3-4 예시)."""
    record_change(db, change_id="CG-1", kind="deploy", system_id="SY13", changed_at=AT, changed_by_team="TM03")
    top = queries.rank_causes(db, "S1", "2026-12-03T20:14+09:00")[0]
    assert (top["cause_id"], top["change_bonus"]) == ("C02", 2)
    assert top["recent_changes"] == [
        {"id": "CG-1", "kind": "deploy", "changed_at": "2026-12-03T19:40:00+09:00", "gap_minutes": 34}
    ]
    with db.session() as session:
        row = session.run(
            "MATCH (c:Change {id: 'CG-1'})-[:CHANGED]->(s:System) "
            "RETURN c.changed_by_team AS team, s.id AS system"
        ).single()
    assert (row["team"], row["system"]) == ("TM03", "SY13")


def test_record_change_accepts_other_timezones(db):
    record_change(db, change_id="CG-1", kind="patch", system_id="SY13", changed_at=AT.astimezone(timezone.utc))
    top = queries.rank_causes(db, "S1", "2026-12-03T20:14+09:00")[0]
    assert top["recent_changes"][0]["gap_minutes"] == 34


class NoOffset(tzinfo):
    """tzinfo는 있지만 오프셋을 주지 않는 시간대."""

    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return None


@pytest.mark.parametrize("value", ["", None])
def test_test_db_uri_never_falls_back_to_dev_db(monkeypatch, value):
    """테스트용 주소가 비어 있거나 없어도 개발용 DB(NEO4J_URI)로 넘어가지 않는다."""
    from conftest import test_db_uri

    monkeypatch.setenv("NEO4J_URI", "bolt://dev-db.invalid:7687")
    if value is None:
        monkeypatch.delenv("NEO4J_TEST_URI", raising=False)
    else:
        monkeypatch.setenv("NEO4J_TEST_URI", value)
    assert test_db_uri() == "bolt://localhost:7688"


@pytest.mark.parametrize("override,message", [
    ({"changed_at": datetime(2026, 12, 3, 19, 40, tzinfo=NoOffset())}, "시간대"),
    ({"kind": "hotfix"}, "kind"),
    ({"system_id": "SY99"}, "시스템"),
    ({"changed_by_team": "TM99"}, "팀"),
    ({"changed_at": datetime(2026, 12, 3, 19, 40)}, "시간대"),
    ({"changed_at": "2026-12-03T19:40+09:00"}, "시간대"),
])
def test_record_change_rejects_undefined_values(db, override, message):
    args = {"change_id": "CG-1", "kind": "deploy", "system_id": "SY13", "changed_at": AT} | override
    with pytest.raises(RecordError, match=message):
        record_change(db, **args)
    assert count(db, "Change") == 0


def test_record_change_rejects_duplicate_id(db):
    record_change(db, change_id="CG-1", kind="deploy", system_id="SY13", changed_at=AT)
    with pytest.raises(RecordError, match="이미 있는"):
        record_change(db, change_id="CG-1", kind="deploy", system_id="SY12", changed_at=AT)
    assert count(db, "Change") == 1


def test_reset_event_layer_keeps_static_layer(db):
    before = {label: count(db, label) for label in STATIC_LABELS}
    record_change(db, change_id="CG-1", kind="deploy", system_id="SY13", changed_at=AT)
    record_change(db, change_id="CG-2", kind="table_change", system_id="SY17", changed_at=AT)
    assert reset_event_layer(db) == 2
    assert count(db, "Change") == 0
    assert {label: count(db, label) for label in STATIC_LABELS} == before


# --- 진단 에이전트가 쓰는 기록 함수 (PR #12) -------------------------------------------

from common.writers import events  # noqa: E402


def open_incident(db, incident_id="INC-1", at=AT):
    events.record_incident(db, incident_id=incident_id, severity_id="SEV3", opened_at=at)


def report(db, report_id="RP-1", reporter="U1", at=AT, symptom="S1", channel="CH02"):
    events.record_report(db, report_id=report_id, channel_id=channel, reporter_id=reporter, raw_text="안 들어와요",
                         received_at=at, symptom_id=symptom, scope="some_users" if symptom else None)


def test_report_is_linked_to_channel_and_symptom(db):
    report(db)
    report(db, report_id="RP-2", symptom=None)
    with db.session() as session:
        row = session.run(
            "MATCH (r:Report {id: 'RP-1'})-[:RECEIVED_VIA]->(c:Channel), (r)-[:INDICATES]->(s:SymptomType) "
            "RETURN c.id AS c, s.id AS s"
        ).single()
        assert (row["c"], row["s"]) == ("CH02", "S1")
        # 증상에 해당하지 않는 제보는 INDICATES가 없다
        assert session.run("MATCH (:Report {id: 'RP-2'})-[:INDICATES]->() RETURN count(*) AS n").single()["n"] == 0


@pytest.mark.parametrize("kwargs, message", [
    ({"channel": "CH99"}, "정의되지 않은 채널"),
    ({"symptom": "S99"}, "정의되지 않은 증상"),
    ({"at": datetime(2026, 12, 3, 19, 40)}, "시간대"),
])
def test_report_refuses_undefined_values(db, kwargs, message):
    with pytest.raises(RecordError, match=message):
        report(db, **kwargs)
    assert count(db, "Report") == 0


def test_part_of_counts_reports_and_duplicate_does_not(db):
    open_incident(db)
    report(db)
    report(db, report_id="RP-2", reporter="U2")
    events.link_report(db, report_id="RP-1", incident_id="INC-1", link="part_of")
    events.link_report(db, report_id="RP-2", incident_id="INC-1", link="suspected_duplicate")
    with db.session() as session:
        assert session.run("MATCH (i:Incident {id: 'INC-1'}) RETURN i.report_count AS n").single()["n"] == 1
    with pytest.raises(RecordError, match="접수 연결"):
        events.link_report(db, report_id="RP-1", incident_id="INC-1", link="merge")


def test_open_incidents_reads_window_and_reports(db):
    open_incident(db, "INC-OLD", at=AT - timedelta(hours=7))
    open_incident(db, "INC-NEW", at=AT)
    report(db)
    events.link_report(db, report_id="RP-1", incident_id="INC-NEW", link="part_of")
    events.record_affects(db, incident_id="INC-NEW", system_ids=["SY13"])
    rows = events.open_incidents(db, since=AT - timedelta(hours=6))
    assert [r["incident_id"] for r in rows] == ["INC-NEW"]
    assert rows[0]["systems"] == ["SY13"] and rows[0]["reports"][0]["reporter_id"] == "U1"


def test_affects_refuses_unknown_system_and_incident(db):
    open_incident(db)
    with pytest.raises(RecordError, match="정의되지 않은 시스템"):
        events.record_affects(db, incident_id="INC-1", system_ids=["SY13", "SY99"])
    with pytest.raises(RecordError, match="없는 Incident"):
        events.record_affects(db, incident_id="INC-9", system_ids=["SY13"])


def test_hypothesis_and_finding(db):
    open_incident(db)
    events.record_hypothesis(db, incident_id="INC-1", cause_id="C02", score=3, rank=1, status="candidate")
    events.record_hypothesis(db, incident_id="INC-1", cause_id="C02", score=3, rank=1, status="confirmed")
    events.record_finding(db, finding_id="FD-1", incident_id="INC-1", check_id="E02", cause_id="C02",
                          result="confirm", detail="지급 기록 없음", checked_at=AT)
    with db.session() as session:
        assert session.run("MATCH (:Incident)-[h:HAS_HYPOTHESIS]->(:Cause {id: 'C02'}) RETURN collect(h.status) AS s"
                           ).single()["s"] == ["confirmed"]
        assert session.run("MATCH (f:Finding)-[:CHECKS]->(:EvidenceCheck {id: 'E02'}) RETURN f.result AS r"
                           ).single()["r"] == "confirm"
    with pytest.raises(RecordError, match="가설 상태"):
        events.record_hypothesis(db, incident_id="INC-1", cause_id="C02", score=3, rank=1, status="maybe")
    with pytest.raises(RecordError, match="Finding.result"):
        events.record_finding(db, finding_id="FD-2", incident_id="INC-1", check_id="E02", cause_id="C02",
                              result="yes", detail="", checked_at=AT)


def test_started_after_and_severity_change(db):
    open_incident(db)
    record_change(db, change_id="CG-1", kind="deploy", system_id="SY13", changed_at=AT - timedelta(minutes=30))
    events.record_started_after(db, incident_id="INC-1", change_id="CG-1", gap_minutes=30)
    events.set_incident_severity(db, incident_id="INC-1", severity_id="SEV2")
    with db.session() as session:
        row = session.run("MATCH (i:Incident)-[r:STARTED_AFTER]->(:Change) RETURN r.gap_minutes AS g, i.severity_id AS s").single()
    assert (row["g"], row["s"]) == (30, "SEV2")
    with pytest.raises(RecordError, match="정의되지 않은 심각도"):
        events.set_incident_severity(db, incident_id="INC-1", severity_id="SEV9")


def test_linking_the_same_report_twice_changes_nothing(db):
    open_incident(db)
    report(db)
    for _ in range(2):
        events.link_report(db, report_id="RP-1", incident_id="INC-1", link="part_of")
    with db.session() as session:
        row = session.run("MATCH (i:Incident {id: 'INC-1'}) RETURN i.report_count AS n, "
                          "COUNT { (:Report)-[:PART_OF]->(i) } AS links").single()
    assert (row["n"], row["links"]) == (1, 1)
    assert len(events.open_incidents(db, since=AT - timedelta(hours=1))[0]["reports"]) == 1
