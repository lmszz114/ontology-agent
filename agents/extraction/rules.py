"""추출 후보의 형식, 기존 그래프와의 대조(R8-5), 검수 배정(R8-6). LLM을 부르지 않는다 (설계 3단계 4-8, M4 PR 3).

후보 하나는 아래 모양의 dict다. 이름 정규화(R8-4)를 거쳐 기존 노드를 찾았으면 id가 있고, 못 찾았으면 None이다.

    {"relation": "OWNS",
     "subject": {"type": "Team", "name": "서버팀", "id": "TM03"},
     "object": {"type": "System", "name": "메일함", "id": "SY13"},
     "props": {"scope": "logic"},
     "evidence_spans": ["메일함 지급 로직은 서버팀이 맡는다."],
     "confidence": 0.8}

별칭(ALIAS)은 object가 {"type": "Name", "name": 별칭, "id": None}이다.

대조와 배정의 기준 지식 K는 시드로 만든 그래프다(ontology.load.build_graph). Neo4j에 적재된 시드와 같은지는
seed_matches_graph로 확인한다. 같은 입력이면 같은 결과가 나온다.
"""

from dataclasses import dataclass

from common.writers.extraction import check_candidate
from neo4j import Driver

from ontology.load import Graph, build_graph, loaded_seed
from ontology.seed import resolve_owner_teams, seed_hash

# K와 값을 비교하는 속성. expect는 설명 문장이라 코드로 같은지 가릴 수 없어 비교하지 않는다
COMPARED = {"OWNS": ("scope", "area"), "LOCATED_IN": ("scope", "area"), "MAY_BE_CAUSED_BY": ("prior",),
            "CHECKED_BY": ("required",), "RESOLVED_BY": ("condition",)}

# 시스템이 없는 후보의 검수 팀 (3단계 R8-6, 가정)
DEFAULT_REVIEWER = "TM05"  # 개발PM팀


@dataclass
class Knowledge:
    """대조와 배정에 쓰는 기준 지식 K. 시드 하나로 만든다."""

    seed: dict
    graph: Graph

    @classmethod
    def from_seed(cls, seed: dict) -> "Knowledge":
        return cls(seed, build_graph(seed))

    def rels(self, rel: str, start: str | None = None, end: str | None = None) -> list[dict]:
        return [r for r in self.graph.rels.get(rel, [])
                if (start is None or r["start"] == start) and (end is None or r["end"] == end)]

    def node(self, label: str, node_id: str) -> dict | None:
        return next((n for n in self.graph.nodes.get(label, []) if n["id"] == node_id), None)

    def owners(self, system_id: str, scope: str | None = None, area: str | None = None) -> list[str]:
        """시스템을 맡은 팀. scope를 주면 scope와 area가 맞는 팀이고(R5와 같이 area가 없는 쪽은 없는 쪽끼리만 맞는다),
        맞는 팀이 없으면 그 시스템을 맡은 팀 전체다 (M4 결정 3)."""
        owns = self.rels("OWNS", end=system_id)
        if scope is not None:
            fit = sorted({r["start"] for r in owns if r["props"].get("scope") == scope
                          and r["props"].get("area") == area})
            if fit:
                return fit
        return sorted({r["start"] for r in owns})


def seed_matches_graph(driver: Driver, seed: dict) -> bool:
    """Neo4j에 적재된 시드가 대조에 쓰는 시드와 같은지. 다르면 대조 결과가 그래프와 어긋난다 (적재 기록 SeedMeta)."""
    meta = loaded_seed(driver)
    return meta is not None and meta["hash"] == seed_hash(seed)


def _same(cprops: dict, kprops: dict, keys: tuple[str, ...]) -> bool:
    # 후보에 없는 속성은 비교하지 않는다. 빠진 속성은 C-ext가 정해진 기본값으로 채운다 (M4 결정 2)
    return all(kprops.get(key) == cprops[key] for key in keys if key in cprops)


def match(c: dict, k: Knowledge) -> str:
    """후보를 K와 대조해 existing, new, conflict를 정한다 (3단계 R8-5, M4 PR 3 결정 3).

    - 같은 관계와 양끝이 K에 있고 비교하는 속성이 같거나 후보에 속성이 없으면 existing
    - 같은 관계와 양끝인데 속성이 다르면 conflict
    - OWNS에서 같은 시스템, 같은 scope와 area를 K에서는 다른 팀이 맡으면 conflict
    - 그 밖은 new. 양끝 중 기존 노드를 못 찾은 것이 있으면 new다(새 엔티티 후보)
    """
    rel, subject, obj = c["relation"], c["subject"], c["object"]
    props = c.get("props") or {}
    if subject.get("id") is None:
        return "new"
    if rel == "ALIAS":
        node = k.node(subject["type"], subject["id"])
        known = (node or {}).get("aliases") or []
        return "existing" if obj["name"] in known or obj["name"] == (node or {}).get("name") else "new"
    if obj.get("id") is None:
        return "new"
    same_ends = k.rels(rel, subject["id"], obj["id"])
    keys = COMPARED.get(rel, ())
    if rel == "OWNS":
        # OWNS는 scope마다 관계가 하나라 scope가 관계를 가른다. 같은 팀이 다른 scope를 맡는다는 말은 모순이 아니다
        if any(_same(props, r["props"], keys) for r in same_ends):
            return "existing"
        if "scope" in props and any(
            r["start"] != subject["id"] and r["props"].get("scope") == props["scope"]
            # area도 후보에 있을 때만 비교한다. 영역을 말하지 않은 후보는 그 scope 전체를 맡는다는 말이다 (PR #29 리뷰)
            and ("area" not in props or r["props"].get("area") == props["area"])
            for r in k.rels("OWNS", end=obj["id"])
        ):
            return "conflict"
        return "new"
    if same_ends:
        return "existing" if any(_same(props, r["props"], keys) for r in same_ends) else "conflict"
    return "new"


def _cause_owner(cause: dict, k: Knowledge) -> list[str]:
    # 원인 쪽 관계는 그 원인을 고치는 1차 담당 팀이 본다 (R5, M4 PR 3 결정 4)
    if cause.get("id") is None or k.node("Cause", cause["id"]) is None:
        return []
    return resolve_owner_teams(k.seed, cause["id"])


def reviewers(c: dict, k: Knowledge, match_result: str) -> list[str]:
    """후보의 검수 팀 (3단계 R8-6, M4 결정 3과 PR 3 결정 4). 정할 수 없으면 개발PM팀(가정)이다."""
    rel, subject, obj = c["relation"], c["subject"], c["object"]
    props = c.get("props") or {}
    teams: list[str] = []
    if rel == "OWNS" and obj.get("id"):
        teams = k.owners(obj["id"], props.get("scope"), props.get("area"))
        if match_result == "conflict" and subject.get("id"):
            # conflict는 K의 기존 팀과 후보가 말한 팀이 함께 본다
            teams = sorted(set(teams) | {subject["id"]})
    elif rel == "LOCATED_IN" and obj.get("id"):
        teams = k.owners(obj["id"], props.get("scope"), props.get("area"))
    elif rel == "OBSERVED_IN" and obj.get("id"):
        teams = k.owners(obj["id"])
    elif rel == "DEPENDS_ON":
        teams = sorted({t for end in (subject, obj) if end.get("id") for t in k.owners(end["id"])})
    elif rel == "MAY_BE_CAUSED_BY":
        teams = _cause_owner(obj, k)
    elif rel in ("CHECKED_BY", "RESOLVED_BY"):
        teams = _cause_owner(subject, k)
    elif rel == "ALIAS" and subject.get("id"):
        if subject["type"] == "Team":
            teams = [subject["id"]]
        elif subject["type"] == "System":
            teams = k.owners(subject["id"])
        elif subject["type"] == "Cause":
            teams = _cause_owner(subject, k)
        # 증상 별칭은 가리키는 시스템이 하나로 정해지지 않아 기본 검수 팀이 본다
    return teams or [DEFAULT_REVIEWER]


def prepare(candidates: list[dict], k: Knowledge) -> list[dict]:
    """형식을 확인하고 대조 결과와 검수 팀을 붙인다. 기록 함수에 넘기기 전 단계다 (5단계 5장 5, 6번)."""
    out = []
    for c in candidates:
        check_candidate(c)
        m = match(c, k)
        out.append(c | {"match": m, "reviewers": reviewers(c, k, m)})
    return out
