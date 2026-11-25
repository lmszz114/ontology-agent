"""시드 YAML을 Neo4j 정적 지식 계층으로 적재한다.

실행: python -m ontology.load [조직]

Neo4j는 시드 파일의 사본이다 (설계 3단계 8장). 그래서 적재는 덮어쓰기다.
- 노드는 id 기준으로 만들거나 속성을 통째로 바꾼다
- 정적 관계는 모두 지우고 시드대로 다시 만든다
- 시드에서 빠진 정적 노드는 지운다
여러 번 실행해도 결과가 같다. 한 트랜잭션으로 처리해서 중간에 실패하면 되돌린다.

레이블과 관계 타입은 아래 상수에서만 나온다. 값은 모두 쿼리 매개변수로 넘긴다.
"""

import json
import sys
from dataclasses import dataclass, field

from neo4j import Driver, ManagedTransaction

from ontology.db import get_driver
from ontology.conditions import is_required
from ontology.seed import load_seed, seed_hash, validate_seed

# 정적 지식 계층의 노드 레이블 (설계 3단계 2-1)
STATIC_LABELS = (
    "Team", "Role", "Channel", "System", "SymptomType", "Cause",
    "EvidenceCheck", "Procedure", "Action", "Severity",
)

# 정적 지식 계층의 관계 타입 (설계 3단계 3-1): (시작 레이블, 끝 레이블)
STATIC_RELATIONS = {
    "HANDLED_FIRST_BY": ("Channel", "Role"),
    "ASSIGNED_TO": ("Role", "Team"),
    "OWNS": ("Team", "System"),
    "COLLABORATES_ON": ("Team", "System"),
    "DEPENDS_ON": ("System", "System"),
    "OBSERVED_IN": ("SymptomType", "System"),
    "MAY_BE_CAUSED_BY": ("SymptomType", "Cause"),
    "LOCATED_IN": ("Cause", "System"),
    "CHECKED_BY": ("Cause", "EvidenceCheck"),
    "RESOLVED_BY": ("Cause", "Procedure"),
    "HAS_ACTION": ("Procedure", "Action"),
    "LED_BY": ("Procedure", "Team"),
    "EXECUTED_BY": ("Action", "Team"),
    "REQUIRES_APPROVAL_FROM": ("Action", "Team"),
    "TRIGGERS": ("Severity", "Procedure"),
}


@dataclass
class Graph:
    """적재할 노드와 관계."""

    # 레이블 -> [속성 dict (id 포함)]
    nodes: dict[str, list[dict]] = field(default_factory=lambda: {label: [] for label in STATIC_LABELS})
    # 관계 타입 -> [{"start": id, "end": id, "props": dict}]
    rels: dict[str, list[dict]] = field(default_factory=lambda: {rel: [] for rel in STATIC_RELATIONS})

    def add_node(self, label: str, item: dict, keys: tuple[str, ...]) -> None:
        props = {key: item[key] for key in keys if item.get(key) is not None}
        self.nodes[label].append({"id": item["id"], **props})

    def add_rel(self, rel_type: str, start: str, end: str, **props) -> None:
        # 사람이 시드에 직접 넣은 관계다. 문서에서 추출해 승인된 관계는 extracted로 구분한다 (설계 3단계 3-1)
        clean = {key: value for key, value in props.items() if value is not None}
        clean |= {"source": "manual", "source_doc_ids": []}
        self.rels[rel_type].append({"start": start, "end": end, "props": clean})


def build_graph(seed: dict) -> Graph:
    """시드 dict를 노드와 관계 목록으로 바꾼다. Neo4j에 접속하지 않는다."""
    g = Graph()

    for team in seed["teams"]:
        g.add_node("Team", team, ("name", "aliases", "axis"))

    for role in seed["roles"]:
        g.add_node("Role", role, ("name", "description"))
    for ra in seed["role_assignments"]:
        g.add_rel("ASSIGNED_TO", ra["role"], ra["team"])

    for channel in seed["channels"]:
        g.add_node("Channel", channel, ("name", "kind", "source_system"))
        g.add_rel("HANDLED_FIRST_BY", channel["id"], channel["handled_first_by"])

    for system in seed["systems"]:
        g.add_node("System", system, ("name", "aliases", "category", "evidence_basis"))
        for owner in system["owners"]:
            # scope마다 관계 하나
            for scope in owner["scopes"]:
                g.add_rel("OWNS", owner["team"], system["id"], scope=scope, area=owner.get("area"))
        for collab in system.get("collaborators", []):
            g.add_rel("COLLABORATES_ON", collab["team"], system["id"], condition=collab["condition"])
        for dep in system["depends_on"]:
            g.add_rel("DEPENDS_ON", system["id"], dep)

    for symptom in seed["symptoms"]:
        g.add_node("SymptomType", symptom, ("name", "aliases", "core_function", "min_severity"))
        for system_id in symptom["observed_in"]:
            g.add_rel("OBSERVED_IN", symptom["id"], system_id)
        for ref in symptom["causes"]:
            g.add_rel("MAY_BE_CAUSED_BY", symptom["id"], ref["cause"], prior=ref["prior"])

    for cause in seed["causes"]:
        g.add_node("Cause", cause, ("name", "aliases"))
        loc = cause["located_in"]
        g.add_rel("LOCATED_IN", cause["id"], loc["system"], scope=loc["scope"], area=loc.get("area"))
        for ref in cause["checked_by"]:
            # 관계 속성에는 맵을 넣을 수 없어 인자는 JSON 문자열로 둔다.
            # when은 시드와 시나리오 검증용이라 그래프에 넣지 않는다 (ontology/conditions.py)
            args = json.dumps(ref["args"], ensure_ascii=False, sort_keys=True) if ref.get("args") else None
            # required가 false면 참고용 근거다. 확정 조건 설명(expect) 대신 쓰임(note)을 가진다
            g.add_rel(
                "CHECKED_BY", cause["id"], ref["check"],
                required=is_required(ref), expect=ref.get("expect"), note=ref.get("note"), args=args,
            )
        for ref in cause["resolved_by"]:
            # 조건의 판정 근거는 조회할 근거(condition_check)와 설명(condition_expect)만 넣는다. when은 검증기용이다
            cond_args = json.dumps(ref["args"], ensure_ascii=False, sort_keys=True) if ref.get("args") else None
            g.add_rel(
                "RESOLVED_BY", cause["id"], ref["procedure"], condition=ref.get("condition"),
                condition_check=ref.get("check"), condition_expect=ref.get("expect"), condition_args=cond_args,
            )

    for check in seed["evidence_checks"]:
        g.add_node(
            "EvidenceCheck", check,
            ("name", "tool", "performed_by", "judged_by_team", "performed_by_team", "source_systems"),
        )
        g.nodes["EvidenceCheck"][-1]["params"] = list(check.get("params", {}))

    for procedure in seed["procedures"]:
        g.add_node("Procedure", procedure, ("name", "steps", "human_only"))
        if procedure["led_by"]:
            g.add_rel("LED_BY", procedure["id"], procedure["led_by"])
        for order, action_id in enumerate(procedure["actions"], start=1):
            g.add_rel("HAS_ACTION", procedure["id"], action_id, order=order)

    for action in seed["actions"]:
        g.add_node(
            "Action", action,
            ("name", "kind", "risk", "auto_executable", "simulated",
             "executed_by_rule", "approver_rule", "uses_system"),
        )
        if action["executed_by"]:
            g.add_rel("EXECUTED_BY", action["id"], action["executed_by"])
        for approver in action["approvers"]:
            g.add_rel("REQUIRES_APPROVAL_FROM", action["id"], approver["team"], condition=approver.get("condition"))

    for severity in seed["severities"]:
        g.add_node("Severity", severity, ("criteria",))
        for procedure_id in severity["triggers"]:
            g.add_rel("TRIGGERS", severity["id"], procedure_id)

    _apply_provenance(g, seed.get("provenance") or [])
    return g


def _apply_provenance(g: Graph, provenance: list[dict]) -> None:
    """반영된 지식의 출처를 관계 속성에 넣는다 (3단계 3-1 source, source_doc_ids, R8-7).

    새로 들어온 관계는 source=extracted, 이미 있던 관계에 출처만 더한 것은 manual 그대로 둔다. 출처가 가리키는 관계가
    그래프에 없으면 시드를 고친 쪽과 출처가 어긋난 것이라 적재를 멈춘다. 별칭은 노드 속성이라 여기서 다루지 않는다.
    """
    for entry in provenance:
        if entry["rel"] == "ALIAS":
            continue
        props = entry.get("props") or {}
        rows = [r for r in g.rels.get(entry["rel"], []) if r["start"] == entry["start"] and r["end"] == entry["end"]
                and all(r["props"].get(k) == v for k, v in props.items())]
        if not rows:
            raise ValueError(f"출처가 가리키는 관계가 시드에 없다: {entry['rel']} {entry['start']} -> {entry['end']} {props}")
        for r in rows:
            if entry["source"] == "extracted":
                r["props"]["source"] = "extracted"
            r["props"]["source_doc_ids"] = sorted(set(r["props"]["source_doc_ids"]) | set(entry["docs"]))


def _write(tx: ManagedTransaction, graph: Graph) -> dict:
    for rel_type in STATIC_RELATIONS:
        tx.run(f"MATCH ()-[r:{rel_type}]->() DELETE r").consume()

    removed = 0
    for label in STATIC_LABELS:
        rows = graph.nodes[label]
        tx.run(
            f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n = row",
            rows=rows,
        ).consume()
        summary = tx.run(
            # id가 없는 노드는 NOT IN이 null이 되어 걸리지 않으므로 따로 잡는다
            f"MATCH (n:{label}) WHERE n.id IS NULL OR NOT n.id IN $ids DETACH DELETE n",
            ids=[row["id"] for row in rows],
        ).consume()
        removed += summary.counters.nodes_deleted

    created = 0
    for rel_type, (start_label, end_label) in STATIC_RELATIONS.items():
        rows = graph.rels[rel_type]
        summary = tx.run(
            f"UNWIND $rows AS row "
            f"MATCH (a:{start_label} {{id: row.start}}) "
            f"MATCH (b:{end_label} {{id: row.end}}) "
            f"CREATE (a)-[r:{rel_type}]->(b) SET r = row.props",
            rows=rows,
        ).consume()
        # 양 끝 노드가 없으면 관계가 만들어지지 않는다. 조용히 빠지지 않게 여기서 막는다
        if summary.counters.relationships_created != len(rows):
            raise ValueError(
                f"{rel_type}: 시드에는 {len(rows)}개인데 {summary.counters.relationships_created}개만 만들어졌다. "
                f"가리키는 id가 {start_label} 또는 {end_label}에 없다"
            )
        created += len(rows)

    return {
        "nodes": sum(len(rows) for rows in graph.nodes.values()),
        "relationships": created,
        "removed_nodes": removed,
    }


# 적재한 시드의 해시를 남기는 노드. 정적 지식이 아니라 적재 기록이라 STATIC_LABELS에 넣지 않는다
SEED_META = "SeedMeta"


def load_static(driver: Driver, seed: dict) -> dict:
    """정적 지식 계층을 시드대로 맞춘다. 적재한 노드, 관계 수와 지운 노드 수를 돌려준다.

    같은 트랜잭션에서 적재한 시드의 해시와 조직을 남긴다. 실행기가 시드 파일과 그래프가 같은지 이 값으로 본다
    """
    validate_seed(seed)
    graph = build_graph(seed)

    def write(tx: ManagedTransaction) -> dict:
        result = _write(tx, graph)
        tx.run(f"MERGE (m:{SEED_META} {{id: 'seed'}}) SET m.hash = $hash, m.org = $org",
               hash=seed_hash(seed), org=seed["org"]["id"]).consume()
        return result

    with driver.session() as session:
        return session.execute_write(write)


def loaded_seed(driver: Driver) -> dict | None:
    """그래프에 마지막으로 적재한 시드의 해시와 조직. 적재 기록이 없으면 None."""
    with driver.session() as session:
        record = session.run(f"MATCH (m:{SEED_META} {{id: 'seed'}}) RETURN m.hash AS hash, m.org AS org").single()
    return dict(record) if record else None


if __name__ == "__main__":
    org = sys.argv[1] if len(sys.argv) > 1 else "org-a"
    with get_driver() as drv:
        result = load_static(drv, load_seed(org))
    print(
        f"{org} 적재: 노드 {result['nodes']}개, 관계 {result['relationships']}개"
        f" (시드에 없어 지운 노드 {result['removed_nodes']}개)"
    )
