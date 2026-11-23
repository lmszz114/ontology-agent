"""적재기 검증.

build_graph는 Neo4j 없이 보고, load_static은 실행 중인 Neo4j에 적재해서 본다.
"""

import copy

import pytest

from ontology.apply_schema import apply_schema
from ontology.load import STATIC_LABELS, STATIC_RELATIONS, build_graph, load_static
from ontology.seed import load_seed

# 시드 키 -> 레이블
LABEL_BY_KEY = {
    "teams": "Team", "roles": "Role", "channels": "Channel", "systems": "System",
    "symptoms": "SymptomType", "causes": "Cause", "evidence_checks": "EvidenceCheck",
    "procedures": "Procedure", "actions": "Action", "severities": "Severity",
}


@pytest.fixture(scope="module")
def seed():
    return load_seed("org-a")


@pytest.fixture(scope="module")
def graph(seed):
    return build_graph(seed)


def test_every_seed_item_becomes_a_node(seed, graph):
    assert set(graph.nodes) == set(STATIC_LABELS)
    for key, label in LABEL_BY_KEY.items():
        assert [n["id"] for n in graph.nodes[label]] == [item["id"] for item in seed[key]]


def test_relation_counts_match_seed(seed, graph):
    expected = {
        "HANDLED_FIRST_BY": len(seed["channels"]),
        "ASSIGNED_TO": len(seed["role_assignments"]),
        "OWNS": sum(len(o["scopes"]) for s in seed["systems"] for o in s["owners"]),
        "COLLABORATES_ON": sum(len(s.get("collaborators", [])) for s in seed["systems"]),
        "DEPENDS_ON": sum(len(s["depends_on"]) for s in seed["systems"]),
        "OBSERVED_IN": sum(len(s["observed_in"]) for s in seed["symptoms"]),
        "MAY_BE_CAUSED_BY": sum(len(s["causes"]) for s in seed["symptoms"]),
        "LOCATED_IN": len(seed["causes"]),
        "CHECKED_BY": sum(len(c["checked_by"]) for c in seed["causes"]),
        "RESOLVED_BY": sum(len(c["resolved_by"]) for c in seed["causes"]),
        "HAS_ACTION": sum(len(p["actions"]) for p in seed["procedures"]),
        "LED_BY": sum(1 for p in seed["procedures"] if p["led_by"]),
        "EXECUTED_BY": sum(1 for a in seed["actions"] if a["executed_by"]),
        "REQUIRES_APPROVAL_FROM": sum(len(a["approvers"]) for a in seed["actions"]),
        "TRIGGERS": sum(len(s["triggers"]) for s in seed["severities"]),
    }
    assert set(expected) == set(STATIC_RELATIONS)
    assert {rel: len(rows) for rel, rows in graph.rels.items()} == expected


def test_relation_properties(graph):
    def props(rel_type, start, end):
        return [r["props"] for r in graph.rels[rel_type] if (r["start"], r["end"]) == (start, end)]

    # scope마다 관계 하나, area는 있는 쪽에만
    assert {p["scope"] for p in props("OWNS", "TM03", "SY11")} == {"logic", "data", "config"}
    assert props("OWNS", "TM02", "SY17") == [
        {"scope": "data", "area": "probability", "source": "manual", "source_doc_ids": []}
    ]
    assert "area" not in props("OWNS", "TM03", "SY17")[0]
    assert props("LOCATED_IN", "C16", "SY17")[0]["area"] == "probability"
    assert props("MAY_BE_CAUSED_BY", "S1", "C02")[0]["prior"] == "높음"
    assert props("CHECKED_BY", "C02", "E02")[0]["expect"] == "우편함 지급 기록이 없다"
    assert "args" not in props("CHECKED_BY", "C02", "E02")[0]
    assert props("CHECKED_BY", "C08", "E04")[0]["args"] == '{"system": "SY10"}'
    assert props("RESOLVED_BY", "C11", "P7")[0]["condition"] == "과지급"
    assert "condition" not in props("RESOLVED_BY", "C11", "P3")[0]
    assert [p["order"] for p in props("HAS_ACTION", "P9", "A16")] == [2]
    assert props("REQUIRES_APPROVAL_FROM", "A15", "TM05")[0]["condition"] == "정기 점검 반영"
    assert props("COLLABORATES_ON", "TM02", "SY14")[0]["condition"]


def test_all_relations_are_marked_manual(graph):
    for rows in graph.rels.values():
        for r in rows:
            assert r["props"]["source"] == "manual" and r["props"]["source_doc_ids"] == []


def test_null_properties_are_left_out(graph):
    for rows in graph.nodes.values():
        for node in rows:
            assert None not in node.values(), node["id"]
    channels = {n["id"]: n for n in graph.nodes["Channel"]}
    assert channels["CH01"]["source_system"] == "SY07"
    assert "source_system" not in channels["CH03"]


def static_counts(driver):
    with driver.session() as session:
        nodes = {
            label: session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            for label in STATIC_LABELS
        }
        rels = {
            rel: session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
            for rel in STATIC_RELATIONS
        }
    return nodes, rels


@pytest.fixture(scope="module")
def loaded(driver, seed):
    apply_schema(driver)
    return load_static(driver, seed)


@pytest.mark.integration
def test_load_writes_all_nodes_and_relations(driver, graph, loaded):
    nodes, rels = static_counts(driver)
    assert nodes == {label: len(rows) for label, rows in graph.nodes.items()}
    assert rels == {rel: len(rows) for rel, rows in graph.rels.items()}
    assert loaded["nodes"] == sum(nodes.values())
    assert loaded["relationships"] == sum(rels.values())


@pytest.mark.integration
def test_load_is_idempotent(driver, seed, loaded):
    before = static_counts(driver)
    again = load_static(driver, seed)
    assert static_counts(driver) == before
    assert again["removed_nodes"] == 0


@pytest.mark.integration
def test_node_properties_are_replaced_and_stale_nodes_removed(driver, seed, loaded):
    with driver.session() as session:
        session.run("MATCH (n:Team {id: 'TM01'}) SET n.leftover = 'x'").consume()
        session.run("CREATE (:Cause {id: '__stale__'})").consume()
        session.run("CREATE (:Cause {name: '__no_id__'})").consume()
    try:
        result = load_static(driver, seed)
        with driver.session() as session:
            team = session.run("MATCH (n:Team {id: 'TM01'}) RETURN properties(n) AS p").single()["p"]
            stale = session.run("MATCH (n:Cause {id: '__stale__'}) RETURN count(n) AS c").single()["c"]
            no_id = session.run("MATCH (n:Cause) WHERE n.id IS NULL RETURN count(n) AS c").single()["c"]
        assert "leftover" not in team and team["name"] == "콘텐츠기획팀"
        # 시드에 없는 id를 가진 노드와 id가 없는 노드를 모두 지운다
        assert stale == 0 and no_id == 0 and result["removed_nodes"] == 2
    finally:
        with driver.session() as session:
            session.run(
                "MATCH (n:Cause) WHERE n.id = '__stale__' OR n.name = '__no_id__' DETACH DELETE n"
            ).consume()


@pytest.mark.integration
def test_dangling_reference_fails_and_rolls_back(driver, seed, loaded):
    """없는 id를 가리키면 실패하고, 그래프는 실패 전 상태로 남는다."""
    before = static_counts(driver)
    broken = copy.deepcopy(seed)
    broken["causes"][0]["located_in"]["system"] = "SY99"
    with pytest.raises(ValueError, match="LOCATED_IN"):
        load_static(driver, broken)
    assert static_counts(driver) == before


@pytest.mark.integration
@pytest.mark.parametrize("breaker,message", [
    (lambda s: s["evidence_checks"][9].update(judged_by_team="TM99"), "judged_by_team"),
    (lambda s: s["actions"][6].update(uses_system="SY99"), "uses_system"),
    (lambda s: s["symptoms"][0]["causes"][0].update(prior="아주 높음"), "prior"),
    (lambda s: s["symptoms"][0]["causes"][0].update(prior=None), "prior"),
    (lambda s: s["causes"][7]["checked_by"][0].update(args={"system": "SY99"}), "SY99"),
    (lambda s: s["causes"][1]["checked_by"][0].update(when={"approval": "maybe"}), "maybe"),
])
def test_invalid_property_values_are_rejected_before_writing(driver, seed, loaded, breaker, message):
    """노드 속성으로 들어가는 참조와 확정 조건은 관계 수로 걸리지 않아 적재 전에 확인한다."""
    before = static_counts(driver)
    broken = copy.deepcopy(seed)
    breaker(broken)
    with pytest.raises(ValueError, match=message):
        load_static(driver, broken)
    assert static_counts(driver) == before
