"""지식 추출 계층 기록 함수 (설계 3단계 2-3, 3-3, 4-8 R8, 5단계 1장, M4 PR 3).

LLM은 그래프에 직접 쓰지 않는다. 추출기는 이 모듈의 함수로 Document와 Candidate만 쓴다. 정적 지식 계층은 읽기만 한다.
쓰기 전에 거부하는 것: 정의되지 않은 관계와 타입과 속성 값(check_candidate), 원문에 글자 그대로
없는 근거 문장(R8-3), 없는 문서와 노드와 팀, 같은 실행에서 같은 문서의 후보를 두 번 쓰기.

Incident에서 Document로 가는 DOCUMENTED_IN은 만들지 않는다. 평가는 시나리오마다 사건 계층을 비우고 회고 문서에 맞는
과거 사건이 그래프에 없어 이 관계를 만들 곳이 없다 (M4 PR 3 결정 2).
"""

import json
import re

from neo4j import Driver

from common.writers.events import RecordError

# 추출 대상 관계와 양끝 노드 타입 (3단계 4-8 R8-3). ALIAS는 aliases 속성이다
ALIAS_TARGETS = ("Team", "System", "SymptomType", "Cause")
RELATION_TYPES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "OWNS": (("Team",), ("System",)),
    "DEPENDS_ON": (("System",), ("System",)),
    "OBSERVED_IN": (("SymptomType",), ("System",)),
    "MAY_BE_CAUSED_BY": (("SymptomType",), ("Cause",)),
    "LOCATED_IN": (("Cause",), ("System",)),
    "CHECKED_BY": (("Cause",), ("EvidenceCheck",)),
    "RESOLVED_BY": (("Cause",), ("Procedure",)),
    "ALIAS": (ALIAS_TARGETS, ("Name",)),
}

# 관계마다 후보가 가질 수 있는 속성과 허용 값 (3단계 3-1). None은 값을 따로 정하지 않은 자유 문장이다
SCOPES = {"logic", "data", "config", "operation", "execution"}
AREAS = {"content", "probability"}
PROPS: dict[str, dict[str, set | None]] = {
    "OWNS": {"scope": SCOPES, "area": AREAS},
    "LOCATED_IN": {"scope": SCOPES, "area": AREAS},
    "MAY_BE_CAUSED_BY": {"prior": {"높음", "중간", "낮음"}},
    "CHECKED_BY": {"required": {True, False}, "expect": None},
    "RESOLVED_BY": {"condition": {"미지급", "과지급"}},
}
MATCHES = ("existing", "new", "conflict")


class CandidateError(RecordError):
    """정의되지 않은 관계, 타입, 속성 값을 가진 후보."""


def check_candidate(c: dict) -> None:
    """후보가 정의된 관계, 노드 타입, 속성 값만 쓰는지 본다 (3단계 1장 '허용된 타입만 기록')."""
    rel = c.get("relation")
    if rel not in RELATION_TYPES:
        raise CandidateError(f"추출 대상이 아닌 관계다: {rel}")
    subject_types, object_types = RELATION_TYPES[rel]
    for side, allowed in (("subject", subject_types), ("object", object_types)):
        end = c.get(side) or {}
        if end.get("type") not in allowed:
            raise CandidateError(f"{rel}의 {side} 타입은 {allowed} 중 하나다: {end.get('type')}")
        if not isinstance(end.get("name"), str) or not end["name"].strip():
            raise CandidateError(f"{rel}의 {side} 이름이 비어 있다")
    if rel == "ALIAS" and c["object"].get("id") is not None:
        raise CandidateError("별칭은 노드가 아니라 이름이라 id가 없다")
    allowed_props = PROPS.get(rel, {})
    for key, value in (c.get("props") or {}).items():
        if key not in allowed_props:
            raise CandidateError(f"{rel}에 없는 속성이다: {key}")
        values = allowed_props[key]
        if values is None:
            if not isinstance(value, str):
                raise CandidateError(f"{rel}.{key}는 문장이다: {value!r}")
        elif value not in values or (key == "required" and not isinstance(value, bool)):
            raise CandidateError(f"{rel}.{key}에 쓸 수 없는 값이다: {value!r}")
    confidence = c.get("confidence")
    if confidence is not None and (isinstance(confidence, bool) or not 0 <= confidence <= 1):
        raise CandidateError(f"confidence는 0과 1 사이다: {confidence!r}")
    if not isinstance(c.get("evidence_span"), str) or not c["evidence_span"].strip():
        raise CandidateError("근거 문장이 비어 있다")


EXTRACTION_LABELS = ("Candidate", "Document")
# 그래프의 Document.kind (3단계 2-3). 방해 문서는 other(일반 문서)로 넣는다. 평가 설계표의 distractor를 그대로 넣으면
# 추출기가 방해 문서라는 답을 보게 된다 (M4 PR 3 결정 1)
DOCUMENT_KINDS = {"postmortem", "runbook", "chat", "meeting", "other"}
CANDIDATE_STATUSES = {"pending", "approved", "rejected"}


# 후보 id는 실행 ID, 문서 ID, 순번을 ':'로 잇는다. 두 ID에 ':'가 들어가면 다른 조합이 같은 id가 될 수 있어
# 쓸 수 있는 글자를 정해 둔다 (PR #29 리뷰)
ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _require_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise RecordError(f"{name}에는 영문, 숫자, '_', '.', '-'만 쓴다: {value!r}")


def _check_document(doc: dict) -> None:
    _require_id(doc["document_id"], "document_id")
    if doc["kind"] not in DOCUMENT_KINDS:
        raise RecordError(f"Document.kind에 쓸 수 없는 값이다: {doc['kind']}")
    if not doc["text"].strip() or not doc["title"].strip():
        raise RecordError(f"{doc['document_id']}: 제목과 원문이 있어야 한다")


_RESET = "MATCH (n) WHERE n:Candidate OR n:Document DETACH DELETE n"


def record_documents(driver: Driver, documents: list[dict], *, replace: bool = False) -> int:
    """원천 문서 여럿을 한 트랜잭션으로 쓴다. 하나라도 거부되면 아무것도 쓰지 않는다. 이미 있는 문서는 거부한다.

    replace면 같은 트랜잭션에서 지식 추출 계층(문서와 후보)을 먼저 비운다. 지우기와 쓰기가 한 번에 커밋되어
    중간에 실패해도 반쯤 지워지거나 반쯤 적재된 그래프가 남지 않는다 (PR #29 리뷰).
    평가용 정보(dev와 test 구분, 정답 사실)는 넣지 않는다. 각 문서는 document_id, kind, title, text, created_at,
    author_team(없을 수 있다)을 가진다.
    """
    ids = [d["document_id"] for d in documents]
    if len(set(ids)) != len(ids):
        raise RecordError("같은 문서가 두 번 있다")
    for doc in documents:
        _check_document(doc)

    def write(tx) -> int:
        if replace:
            tx.run(_RESET).consume()
        for doc in documents:
            if tx.run("MATCH (d:Document {id: $id}) RETURN d.id", id=doc["document_id"]).single():
                raise RecordError(f"이미 있는 Document다: {doc['document_id']}")
            team = doc.get("author_team")
            if team is not None and not tx.run("MATCH (t:Team {id: $id}) RETURN t.id", id=team).single():
                raise RecordError(f"정의되지 않은 팀이다: {team}")
            tx.run("CREATE (d:Document {id: $id, kind: $kind, title: $title, text: $text, created_at: $at}) "
                   "SET d.author_team = $team",
                   id=doc["document_id"], kind=doc["kind"], title=doc["title"], text=doc["text"],
                   at=doc["created_at"], team=team).consume()
        return len(documents)

    with driver.session() as session:
        return session.execute_write(write)


def record_document(driver: Driver, **document) -> None:
    """원천 문서 하나를 쓴다. 이미 있으면 거부한다."""
    record_documents(driver, [document])


def record_candidates(driver: Driver, *, run_id: str, document_id: str, candidates: list[dict]) -> list[str]:
    """한 실행에서 한 문서로 뽑은 후보를 한 트랜잭션에 쓰고 후보 id 목록을 돌려준다.

    후보는 agents.extraction.rules.prepare를 거쳐 match와 reviewers가 붙은 것이다. 상태는 pending으로 시작한다.
    후보 하나라도 거부되면 아무것도 쓰지 않는다.
    """
    _require_id(run_id, "run_id")
    _require_id(document_id, "document_id")
    for c in candidates:
        check_candidate(c)
        if c.get("match") not in MATCHES:
            raise RecordError(f"match에 쓸 수 없는 값이다: {c.get('match')}")
        if not c.get("reviewers"):
            raise RecordError("검수 팀이 없다")

    def write(tx) -> list[str]:
        doc = tx.run("MATCH (d:Document {id: $id}) RETURN d.text AS text", id=document_id).single()
        if doc is None:
            raise RecordError(f"없는 Document다: {document_id}")
        if tx.run("MATCH (c:Candidate {run_id: $run})-[:EXTRACTED_FROM]->(:Document {id: $doc}) RETURN c.id LIMIT 1",
                  run=run_id, doc=document_id).single():
            raise RecordError(f"이 실행에서 이미 후보를 쓴 문서다: {run_id} {document_id}")
        ids = []
        for i, c in enumerate(candidates, start=1):
            # 근거 문장이 원문에 글자 그대로 없으면 후보를 버린다 (R8-3). 추출 단계가 먼저 걸러야 하고 여기서는 막기만 한다
            if c["evidence_span"] not in doc["text"]:
                raise RecordError(f"근거 문장이 원문에 없다: {document_id} {c['evidence_span'][:40]}")
            cid = f"{run_id}:{document_id}:{i:03d}"
            subject, obj = c["subject"], c["object"]
            tx.run(
                "MATCH (d:Document {id: $doc}) "
                "CREATE (c:Candidate {id: $id, run_id: $run, relation: $rel, subject_type: $st, subject_name: $sn, "
                "object_type: $ot, object_name: $on, props: $props, evidence_span: $span, match: $match, "
                "status: 'pending'})-[:EXTRACTED_FROM]->(d) "
                "SET c.confidence = $conf, c.subject_id = $sid, c.object_id = $oid",
                doc=document_id, id=cid, run=run_id, rel=c["relation"], st=subject["type"], sn=subject["name"],
                ot=obj["type"], on=obj["name"], sid=subject.get("id"), oid=obj.get("id"),
                # 관계 속성은 맵을 노드 속성에 넣을 수 없어 JSON 문자열로 둔다 (Finding.args와 같은 방식)
                props=json.dumps(c.get("props") or {}, ensure_ascii=False, sort_keys=True),
                span=c["evidence_span"], match=c["match"], conf=c.get("confidence"),
            ).consume()
            for role, end in (("subject", subject), ("object", obj)):
                if end.get("id") is None:
                    continue  # 기존 노드를 못 찾은 새 엔티티 후보는 PROPOSES가 없다 (3-3)
                linked = tx.run(
                    f"MATCH (c:Candidate {{id: $id}}), (n:{end['type']} {{id: $nid}}) "
                    "CREATE (c)-[:PROPOSES {role: $role}]->(n) RETURN n.id",
                    id=cid, nid=end["id"], role=role).single()
                if linked is None:
                    raise RecordError(f"정의되지 않은 {end['type']}다: {end['id']}")
            for team in sorted(set(c["reviewers"])):
                linked = tx.run("MATCH (c:Candidate {id: $id}), (t:Team {id: $team}) "
                                "CREATE (c)-[:REVIEWED_BY]->(t) RETURN t.id", id=cid, team=team).single()
                if linked is None:
                    raise RecordError(f"정의되지 않은 팀이다: {team}")
            ids.append(cid)
        return ids

    with driver.session() as session:
        return session.execute_write(write)


def list_candidates(driver: Driver, *, run_id: str | None = None, status: str | None = None,
                    team_id: str | None = None, document_id: str | None = None) -> list[dict]:
    """조건에 맞는 후보를 id 순으로 돌려준다. 검수 화면과 추출 채점기가 읽는다."""
    if status is not None and status not in CANDIDATE_STATUSES:
        raise RecordError(f"Candidate.status에 쓸 수 없는 값이다: {status}")
    query = (
        "MATCH (c:Candidate)-[:EXTRACTED_FROM]->(d:Document) "
        "WHERE ($run IS NULL OR c.run_id = $run) AND ($status IS NULL OR c.status = $status) "
        "AND ($doc IS NULL OR d.id = $doc) "
        "AND ($team IS NULL OR EXISTS { (c)-[:REVIEWED_BY]->(:Team {id: $team}) }) "
        "OPTIONAL MATCH (c)-[:REVIEWED_BY]->(t:Team) "
        "WITH c, d, collect(t.id) AS reviewers "
        "RETURN c {.*, document_id: d.id, reviewers: reviewers} AS c ORDER BY c.id"
    )
    with driver.session() as session:
        rows = session.run(query, run=run_id, status=status, team=team_id, doc=document_id).data()
    out = []
    for row in rows:
        c = row["c"]
        c["props"] = json.loads(c["props"])
        c["reviewers"] = sorted(c["reviewers"])
        c.pop("_lock", None)
        out.append(c)
    return out


def delete_run(driver: Driver, run_id: str) -> int:
    """한 실행의 후보를 지우고 지운 수를 돌려준다. 문서는 남긴다."""
    with driver.session() as session:
        return session.run("MATCH (c:Candidate {run_id: $run}) DETACH DELETE c", run=run_id
                           ).consume().counters.nodes_deleted


def reset_extraction_layer(driver: Driver) -> int:
    """지식 추출 계층(Document, Candidate)을 모두 지우고 지운 수를 돌려준다. 정적, 사건 계층은 건드리지 않는다."""
    # 문서와 후보를 한 번에 지운다. 나눠 지우면 그 사이에 쓴 후보가 문서 연결 없이 남을 수 있다 (PR #29 리뷰)
    with driver.session() as session:
        return session.run(_RESET).consume().counters.nodes_deleted
