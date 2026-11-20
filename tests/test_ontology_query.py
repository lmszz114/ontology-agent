"""온톨로지 조회 도구 검증. 시드를 적재한 Neo4j에서 돈다."""

import asyncio

import pytest
from mcp import Client
from neo4j.exceptions import ClientError

from mcp_servers.ontology_query import queries
from mcp_servers.ontology_query.server import build_server
from ontology.apply_schema import apply_schema
from ontology.load import load_static
from ontology.seed import load_seed
from test_seed import EXPECTED_OWNER

pytestmark = pytest.mark.integration

STARTED_AT = "2026-12-03T20:14+09:00"
TOOL_NAMES = {
    "get_symptom_types", "get_cause_candidates", "rank_causes", "get_checks",
    "get_owner_team", "get_procedures", "get_action_policy",
}


@pytest.fixture(scope="module")
def seed():
    return load_seed("org-a")


@pytest.fixture(scope="module")
def db(driver, seed):
    apply_schema(driver)
    load_static(driver, seed)
    return driver


@pytest.fixture
def changes(db):
    """변경 이력을 넣는 함수를 준다. 테스트가 끝나면 지운다."""
    def add(change_id: str, system_id: str, changed_at: str, kind: str = "deploy") -> None:
        with db.session() as session:
            session.run(
                "MATCH (sys:System {id: $system}) "
                "CREATE (:Change {id: $id, kind: $kind, changed_at: datetime($at)})-[:CHANGED]->(sys)",
                id=change_id, system=system_id, at=changed_at, kind=kind,
            ).consume()

    yield add
    with db.session() as session:
        session.run("MATCH (n:Change) WHERE n.id STARTS WITH '__test_' DETACH DELETE n").consume()


def test_get_symptom_types(db, seed):
    rows = queries.get_symptom_types(db)
    assert [r["id"] for r in rows] == [s["id"] for s in seed["symptoms"]]
    s1 = rows[0]
    assert s1["core_function"] is True and "구매 아이템 미지급" in s1["aliases"]
    assert [x["id"] for x in s1["observed_in"]] == ["SY12", "SY13"]


def test_get_cause_candidates(db, seed):
    for symptom in seed["symptoms"]:
        rows = queries.get_cause_candidates(db, symptom["id"])
        assert {r["cause_id"]: r["prior"] for r in rows} == {c["cause"]: c["prior"] for c in symptom["causes"]}
    c16 = next(r for r in queries.get_cause_candidates(db, "S6") if r["cause_id"] == "C16")
    assert c16["located_in"] == {
        "system_id": "SY17", "system_name": "게임 데이터 테이블", "scope": "data", "area": "probability",
    }


def test_rank_causes_without_changes_uses_prior_only(db, changes):
    rows = queries.rank_causes(db, "S1", STARTED_AT)
    assert [(r["rank"], r["cause_id"], r["score"]) for r in rows] == [(1, "C02", 3), (2, "C01", 2), (3, "C03", 1)]
    assert all(r["change_bonus"] == 0 and r["recent_changes"] == [] for r in rows)


def test_rank_causes_adds_bonus_for_recent_change(db, changes):
    """설계 4단계 3-4 예시: 증상 34분 전에 우편함 배포가 있었다."""
    changes("__test_1", "SY13", "2026-12-03T19:40+09:00")
    top = queries.rank_causes(db, "S1", STARTED_AT)[0]
    assert (top["cause_id"], top["prior_score"], top["change_bonus"], top["score"]) == ("C02", 3, 2, 5)
    assert top["recent_changes"] == [
        {"id": "__test_1", "kind": "deploy", "changed_at": "2026-12-03T19:40:00+09:00", "gap_minutes": 34}
    ]


def test_rank_causes_change_can_reorder(db, changes):
    """낮은 사전 가능성이라도 최근 변경이 있으면 순위가 올라간다."""
    changes("__test_1", "SY12", "2026-12-03T10:00+09:00", kind="product_config")
    rows = queries.rank_causes(db, "S1", STARTED_AT)
    assert [(r["cause_id"], r["score"]) for r in rows] == [("C02", 3), ("C03", 3), ("C01", 2)]


def test_rank_causes_ignores_changes_outside_window(db, changes):
    changes("__test_old", "SY13", "2026-12-02T20:13+09:00")    # 24시간 1분 전
    changes("__test_after", "SY13", "2026-12-03T20:15+09:00")  # 증상 시작 뒤
    changes("__test_other", "SY10", "2026-12-03T19:00+09:00")  # 후보와 상관없는 시스템
    rows = queries.rank_causes(db, "S1", STARTED_AT)
    assert all(r["change_bonus"] == 0 for r in rows)


def test_rank_causes_window_edge_and_timezone(db, changes):
    changes("__test_edge", "SY13", "2026-12-02T11:14Z")  # KST로 정확히 24시간 전
    top = queries.rank_causes(db, "S1", STARTED_AT)[0]
    assert top["cause_id"] == "C02" and top["recent_changes"][0]["gap_minutes"] == 24 * 60


def test_rank_causes_keeps_fractional_seconds(db, changes):
    """소수 초를 버리면 시작 직후의 변경이 시작 전으로 잡힌다."""
    changes("__test_just_after", "SY12", "2026-12-03T20:14:00.500+09:00")  # 시작 0.5초 뒤
    changes("__test_just_out", "SY02", "2026-12-02T20:13:59.500+09:00")    # 24시간 0.5초 전
    rows = queries.rank_causes(db, "S1", STARTED_AT)
    assert all(r["change_bonus"] == 0 for r in rows)
    changes("__test_just_in", "SY13", "2026-12-03T11:13:59.500Z")           # 시작 0.5초 전 (UTC 표기)
    top = queries.rank_causes(db, "S1", STARTED_AT)[0]
    assert (top["cause_id"], top["change_bonus"]) == ("C02", 2)


def test_rank_causes_requires_timezone(db):
    with pytest.raises(ValueError, match="시간대"):
        queries.rank_causes(db, "S1", "2026-12-03T20:14")
    with pytest.raises(ValueError, match="ISO 8601"):
        queries.rank_causes(db, "S1", "어제 저녁")


def test_get_checks(db, seed):
    for cause in seed["causes"]:
        rows = queries.get_checks(db, cause["id"])
        assert {r["check_id"]: r["expect"] for r in rows} == {c["check"]: c["expect"] for c in cause["checked_by"]}
    by_id = {r["check_id"]: r for r in queries.get_checks(db, "C13")}
    assert by_id["E10"]["judged_by_team"] == "시스템기획팀" and by_id["E10"]["tool"] == "query_currency_supply"
    assert by_id["E12"]["performed_by"] == "human" and by_id["E12"]["performed_by_team"] == "QA팀"
    assert by_id["E04"]["args"] == {"system": "SY10"} and by_id["E10"]["args"] == {}


@pytest.mark.parametrize("cause_id,team_name", EXPECTED_OWNER.items())
def test_get_owner_team_primary(db, cause_id, team_name):
    assert [t["name"] for t in queries.get_owner_team(db, cause_id)["primary"]] == [team_name]


def test_get_owner_team_related_teams(db):
    c10 = queries.get_owner_team(db, "C10")
    assert [t["name"] for t in c10["other_owners"]] == ["플랫폼팀"]  # 통합 런처 담당은 협업
    c16 = queries.get_owner_team(db, "C16")
    assert {t["name"] for t in c16["other_owners"]} == {"콘텐츠기획팀", "서버개발팀"}
    c18 = queries.get_owner_team(db, "C18")
    assert c18["other_owners"] == []
    assert [(t["name"], bool(t["condition"])) for t in c18["collaborators"]] == [("시스템기획팀", True)]


def test_get_procedures(db):
    result = queries.get_procedures(db, "C11", "SEV2")
    assert [(p["procedure_id"], p["condition"]) for p in result["resolved_by"]] == [
        ("P3", None), ("P6", "미지급"), ("P7", "과지급"), ("P10", None),
    ]
    p10 = result["resolved_by"][-1]
    assert p10["led_by"] == "개발PM팀"
    assert [(a["action_id"], a["risk"]) for a in p10["actions"]] == [("A14", "low"), ("A15", "high")]
    assert [p["procedure_id"] for p in result["triggered_by_severity"]] == ["P3"]
    assert queries.get_procedures(db, "C11", "SEV3")["triggered_by_severity"] == []
    assert "triggered_by_severity" not in queries.get_procedures(db, "C11")


def test_get_action_policy(db, seed):
    for action in seed["actions"]:
        policy = queries.get_action_policy(db, action["id"])
        assert policy["risk"] == action["risk"] and policy["auto_executable"] == action["auto_executable"]
        assert [t["team_id"] for t in policy["approvers"]] == sorted(a["team"] for a in action["approvers"])
    a10 = queries.get_action_policy(db, "A10")
    assert (a10["executed_by"], a10["uses_system"]) == ("게임서비스팀", "GM 툴")
    a15 = queries.get_action_policy(db, "A15")
    assert a15["executed_by"] is None and a15["executed_by_rule"] == "incident_primary_team"
    assert {t["name"]: t["condition"] for t in a15["approvers"]} == {
        "개발PM팀": "정기 점검 반영", "게임사업PM팀": "긴급 점검 반영", "인프라운영팀": "긴급 점검 반영",
    }
    assert queries.get_action_policy(db, "A17")["approver_rule"] == "candidate_system_owner"


@pytest.mark.parametrize("call", [
    lambda d: queries.get_cause_candidates(d, "S9"),
    lambda d: queries.rank_causes(d, "C01", STARTED_AT),
    lambda d: queries.get_checks(d, "C99"),
    lambda d: queries.get_owner_team(d, "S1"),
    lambda d: queries.get_procedures(d, "C99"),
    lambda d: queries.get_procedures(d, "C01", "SEV9"),
    lambda d: queries.get_action_policy(d, "A99"),
])
def test_unknown_id_is_rejected(db, call):
    with pytest.raises(queries.UnknownId):
        call(db)


def test_queries_run_in_read_only_transactions(db):
    """조회 도구가 쓰는 경로로는 그래프를 바꿀 수 없다."""
    with pytest.raises(ClientError, match="read access mode"):
        queries._read(db, "CREATE (:Team {id: '__test_write__'})")


def test_mcp_server_exposes_the_seven_tools(db):
    async def run():
        async with Client(build_server(db)) as client:
            listed = await client.list_tools()
            ranked = await client.call_tool("rank_causes", {"symptom_id": "S1", "started_at": STARTED_AT})
            unknown = await client.call_tool("get_checks", {"cause_id": "C99"})
            return listed, ranked, unknown

    listed, ranked, unknown = asyncio.run(run())
    assert {t.name for t in listed.tools} == TOOL_NAMES
    assert all(t.description for t in listed.tools)
    assert not ranked.is_error
    assert [r["cause_id"] for r in ranked.structured_content["result"]] == ["C02", "C01", "C03"]
    assert unknown.is_error and "C99" in unknown.content[0].text
