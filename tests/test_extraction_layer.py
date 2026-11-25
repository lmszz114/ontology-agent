"""지식 추출 계층: 후보 형식, 대조(R8-5), 검수 배정(R8-6), 기록 함수 (M4 PR 3)."""

import sys

import pytest

from agents.extraction.rules import Knowledge, match, prepare, reviewers, seed_matches_graph
from common.writers.extraction import (
    PROPS, CandidateError, RecordError, check_candidate, delete_run, list_candidates, record_candidates,
    record_document, reset_extraction_layer,
)
from eval.corpus.check import load_plan
from eval.corpus.load import document_fields, load_documents
from ontology.seed import load_seed

SEED = load_seed("org-a")
K = Knowledge.from_seed(SEED)
PLAN = load_plan()
PREFIX = {"TM": "Team", "SY": "System", "S": "SymptomType", "C": "Cause", "E": "EvidenceCheck", "P": "Procedure"}
NAMES = {i["id"]: i["name"] for key in ("teams", "systems", "symptoms", "causes", "evidence_checks", "procedures")
         for i in SEED[key]}


def _end(node_id: str) -> dict:
    kind = next(v for p, v in sorted(PREFIX.items(), key=lambda x: -len(x[0])) if node_id.startswith(p))
    return {"type": kind, "name": NAMES[node_id], "id": node_id}


def candidate(fact: dict, span: str = "근거 문장") -> dict:
    """설계표의 정답 사실을 이름 정규화까지 마친 후보 모양으로 바꾼다."""
    obj = {"type": "Name", "name": fact["object"], "id": None} if fact["rel"] == "ALIAS" else _end(fact["object"])
    return {"relation": fact["rel"], "subject": _end(fact["subject"]), "object": obj,
            "props": dict(fact.get("props") or {}), "evidence_span": span, "confidence": 0.9}


def test_check_candidate_rejects_undefined_values():
    good = candidate({"rel": "OWNS", "subject": "TM03", "object": "SY13", "props": {"scope": "logic"}})
    check_candidate(good)
    bad = [
        good | {"relation": "ASSIGNED_TO"},  # 추출 대상이 아니다 (R8-3)
        good | {"subject": {"type": "System", "name": "우편함", "id": "SY13"}},
        good | {"props": {"scope": "design"}},
        good | {"props": {"prior": "높음"}},  # OWNS에 없는 속성
        good | {"confidence": 1.5},
        good | {"evidence_span": "  "},
        candidate({"rel": "CHECKED_BY", "subject": "C02", "object": "E02", "props": {"required": "yes"}}),
        candidate({"rel": "ALIAS", "subject": "SY13", "object": "메일함"}) | {
            "object": {"type": "Name", "name": "메일함", "id": "SY13"}},
    ]
    for c in bad:
        with pytest.raises(CandidateError):
            check_candidate(c)


def test_condition_values_match_the_seed():
    conditions = {ref["condition"] for c in SEED["causes"] for ref in c["resolved_by"] if ref.get("condition")}
    assert PROPS["RESOLVED_BY"]["condition"] == conditions


def test_plan_facts_are_existing_and_stale_facts_are_conflicts():
    """설계표의 정답 사실은 K에 있으니 existing, 옛날 정보는 K의 담당과 모순이니 conflict다 (4단계 3-5 라벨)."""
    for doc in PLAN:
        for fact in doc.get("facts") or []:
            m = match(candidate(fact), K)
            if fact["rel"] == "ALIAS":
                aliases = K.node(_end(fact["subject"])["type"], fact["subject"]).get("aliases") or []
                assert m == ("existing" if fact["object"] in aliases else "new"), (doc["id"], fact)
            else:
                assert m == "existing", (doc["id"], fact)
        for fact in doc.get("stale") or []:
            assert match(candidate(fact), K) == "conflict", (doc["id"], fact)


def test_match_rules():
    owns = {"rel": "OWNS", "subject": "TM03", "object": "SY13"}
    assert match(candidate(owns), K) == "existing"  # 속성이 없으면 비교하지 않는다 (M4 결정 2)
    # 같은 팀이 다른 scope를 맡는다는 말은 다른 팀과 겹치지 않으면 new다
    assert match(candidate(owns | {"props": {"scope": "execution"}}), K) == "new"
    # 같은 양끝인데 속성이 다르면 conflict
    # 영역을 말하지 않은 OWNS 후보는 그 scope를 맡은 다른 팀과 겹친다 (PR #29 리뷰)
    data = {"rel": "OWNS", "subject": "TM03", "object": "SY17", "props": {"scope": "data"}}
    assert match(candidate(data), K) == "conflict"
    assert match(candidate(data | {"subject": "TM01"}), K) == "existing"
    prior = {"rel": "MAY_BE_CAUSED_BY", "subject": "S1", "object": "C02", "props": {"prior": "낮음"}}
    assert match(candidate(prior), K) == "conflict"
    # 기존 노드를 못 찾은 쪽이 있으면 새 엔티티 후보다
    new_end = candidate(owns) | {"object": {"type": "System", "name": "길드 시스템", "id": None}}
    assert match(new_end, K) == "new"
    # K에 없는 관계
    dep = {"rel": "DEPENDS_ON", "subject": "SY13", "object": "SY02"}
    assert match(candidate(dep), K) == ("existing" if K.rels("DEPENDS_ON", "SY13", "SY02") else "new")
    assert match(candidate({"rel": "ALIAS", "subject": "SY13", "object": "메일함"}), K) == "existing"
    assert match(candidate({"rel": "ALIAS", "subject": "SY13", "object": "우체통"}), K) == "new"


def test_reviewers():
    def rev(fact, m=None):
        c = candidate(fact)
        return reviewers(c, K, m or match(c, K))

    assert rev({"rel": "OWNS", "subject": "TM03", "object": "SY13", "props": {"scope": "logic"}}) == ["TM03"]
    # 게임 데이터 테이블의 data는 영역마다 담당이 다르다. area가 없으면 맞는 팀이 없어 담당 팀 전체가 본다
    assert rev({"rel": "LOCATED_IN", "subject": "C07", "object": "SY17", "props": {"scope": "data"}}) == \
        sorted({"TM01", "TM02"} | set(K.owners("SY17")))
    assert rev({"rel": "LOCATED_IN", "subject": "C07", "object": "SY17",
                "props": {"scope": "data", "area": "content"}}) == ["TM01"]
    # 옛날 정보는 K의 기존 팀과 후보가 말한 팀이 함께 본다
    stale = next(f for d in PLAN for f in d.get("stale") or [])
    teams = rev(stale)
    assert stale["subject"] in teams and set(K.owners(stale["object"], stale["props"]["scope"])) <= set(teams)
    # 원인 쪽 관계는 원인의 1차 담당 팀(R5), 새 원인과 증상 별칭은 개발PM팀
    assert rev({"rel": "CHECKED_BY", "subject": "C02", "object": "E02"}) == ["TM03"]
    new_cause = candidate({"rel": "MAY_BE_CAUSED_BY", "subject": "S1", "object": "C02"}) | {
        "object": {"type": "Cause", "name": "새 원인", "id": None}}
    assert reviewers(new_cause, K, "new") == ["TM05"]
    assert rev({"rel": "ALIAS", "subject": "S1", "object": "템 안 옴"}) == ["TM05"]
    dep = rev({"rel": "DEPENDS_ON", "subject": "SY13", "object": "SY02"})
    assert set(K.owners("SY13")) | set(K.owners("SY02")) == set(dep)


def test_document_fields_hide_evaluation_info():
    noise = next(d for d in PLAN if d["kind"] == "distractor")
    fields = document_fields(noise)
    assert fields["kind"] == "other"  # 방해 문서라는 답을 그래프에 넣지 않는다 (M4 PR 3 결정 1)
    assert set(fields) == {"document_id", "kind", "title", "text", "created_at", "author_team"}
    assert not fields["title"].startswith("#") and fields["text"].startswith("# ")


def test_loading_test_documents_needs_allow_test(monkeypatch, capsys):
    from eval.corpus import load

    monkeypatch.setattr(sys, "argv", ["load", "--split", "test"])
    with pytest.raises(SystemExit):
        load.main()
    assert "M8" in capsys.readouterr().err


# 아래는 테스트용 Neo4j(7688)에서 돈다


@pytest.fixture(scope="module")
def db(driver):
    from ontology.apply_schema import apply_schema
    from ontology.load import load_static

    apply_schema(driver)
    load_static(driver, SEED)
    return driver


@pytest.fixture
def clean(db):
    reset_extraction_layer(db)
    yield db
    reset_extraction_layer(db)


def _pm01(db) -> tuple[dict, list[dict]]:
    doc = next(d for d in PLAN if d["id"] == "PM-01")
    record_document(db, **document_fields(doc))
    text = document_fields(doc)["text"]
    span = next(line for line in text.splitlines() if len(line) > 20)
    return doc, prepare([candidate(f, span) for f in doc["facts"]] + [candidate(doc["facts"][0], span) | {
        "object": {"type": "Cause", "name": "새 원인", "id": None}}], K)


@pytest.mark.integration
def test_candidates_round_trip(clean):
    db = clean
    assert seed_matches_graph(db, SEED)
    doc, cands = _pm01(db)
    ids = record_candidates(db, run_id="x1", document_id="PM-01", candidates=cands)
    assert ids[0] == "x1:PM-01:001" and len(ids) == len(cands)
    rows = list_candidates(db, run_id="x1")
    assert [r["id"] for r in rows] == ids
    first = rows[0]
    assert first["status"] == "pending" and first["document_id"] == "PM-01" and first["props"] == cands[0]["props"]
    assert first["reviewers"] == sorted(cands[0]["reviewers"])
    with db.session() as s:
        proposes = s.run("MATCH (:Candidate {run_id: 'x1'})-[p:PROPOSES]->() RETURN count(p) AS n").single()["n"]
    # 새 엔티티 후보의 끝은 PROPOSES가 없고 별칭의 이름도 노드가 아니다
    expected = sum((c["subject"].get("id") is not None) + (c["object"].get("id") is not None) for c in cands)
    assert proposes == expected
    assert list_candidates(db, team_id="TM03", run_id="x1")
    assert list_candidates(db, status="approved") == []
    # 같은 실행에서 같은 문서를 두 번 쓰지 않는다. 다른 실행은 따로 쌓인다
    with pytest.raises(RecordError):
        record_candidates(db, run_id="x1", document_id="PM-01", candidates=cands)
    record_candidates(db, run_id="x2", document_id="PM-01", candidates=cands[:2])
    assert delete_run(db, "x1") == len(cands)
    assert len(list_candidates(db)) == 2


@pytest.mark.integration
def test_one_bad_candidate_writes_nothing(clean):
    db = clean
    _, cands = _pm01(db)
    bad = cands[:1] + [cands[1] | {"evidence_span": "원문에 없는 문장이다"}]
    with pytest.raises(RecordError):
        record_candidates(db, run_id="x1", document_id="PM-01", candidates=bad)
    unknown = cands[:1] + [cands[1] | {"subject": cands[1]["subject"] | {"id": "C99"}}]
    with pytest.raises(RecordError):
        record_candidates(db, run_id="x1", document_id="PM-01", candidates=unknown)
    assert list_candidates(db) == []
    with pytest.raises(RecordError):
        record_candidates(db, run_id="x1", document_id="PM-99", candidates=cands)
    # ':'가 들어간 ID는 후보 id가 겹칠 수 있어 거부한다 (PR #29 리뷰)
    for run, doc in (("x:1", "PM-01"), ("", "PM-01"), ("x1", "PM:01")):
        with pytest.raises(RecordError):
            record_candidates(db, run_id=run, document_id=doc, candidates=cands)


@pytest.mark.integration
def test_layers_are_reset_separately(clean):
    from common.writers.events import reset_event_layer

    db = clean
    assert load_documents(db, "dev")
    with pytest.raises(RecordError):
        record_document(db, **document_fields(PLAN[0]))  # 이미 있다
    reset_event_layer(db)  # 사건 계층 초기화는 추출 계층을 건드리지 않는다
    with db.session() as s:
        docs = s.run("MATCH (d:Document) RETURN count(d) AS n").single()["n"]
    assert docs == sum(1 for d in PLAN if d["split"] == "dev")
    reset_extraction_layer(db)
    with db.session() as s:
        assert s.run("MATCH (d:Document) RETURN count(d) AS n").single()["n"] == 0
        assert s.run("MATCH (t:Team) RETURN count(t) AS n").single()["n"] == len(SEED["teams"])
