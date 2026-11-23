"""사건 계층 기록 함수 검증. 시드를 적재한 테스트용 Neo4j에서 돈다."""

from datetime import datetime, timedelta, timezone

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


@pytest.mark.parametrize("override,message", [
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
