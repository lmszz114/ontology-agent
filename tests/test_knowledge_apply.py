"""후보 검수, 지식 반영(A17) 승인 게이트, 시드 반영 (M4 PR 5)."""

import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from agents.extraction import apply as A
from agents.extraction.rules import Knowledge, prepare
from ontology.load import build_graph
from ontology.seed import SEED_DIR, load_seed

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 11, 25, 15, 0, tzinfo=KST)
SEED = load_seed("org-a")
K = Knowledge.from_seed(SEED)


def cand(cid, rel, s, o, props=None, s_type=None, o_type=None, doc="PM-01", o_name=None):
    types = {"TM": "Team", "SY": "System", "S": "SymptomType", "C": "Cause", "E": "EvidenceCheck", "P": "Procedure"}

    def t(x):
        return next(v for p, v in sorted(types.items(), key=lambda kv: -len(kv[0])) if x.startswith(p))

    obj = {"type": "Name", "name": o, "id": None} if rel == "ALIAS" else {"type": o_type or t(o), "name": o_name or o, "id": o}
    c = {"relation": rel, "subject": {"type": s_type or t(s), "name": s, "id": s}, "object": obj,
         "props": props or {}, "evidence_spans": ["근거"], "confidence": 0.9}
    return prepare([c], K)[0] | {"id": cid, "document_id": doc}


def _new_dependency():
    """K에 없는 시스템 의존 하나."""
    systems = [s["id"] for s in SEED["systems"]]
    return next((a, b) for a in systems for b in systems if a != b and not K.rels("DEPENDS_ON", a, b))


def test_classify():
    a, b = _new_dependency()
    assert A.classify(cand("x1", "DEPENDS_ON", a, b))[0]
    assert A.classify(cand("x2", "OWNS", "TM03", "SY13", {"scope": "logic"}))[0]  # existing, 출처만
    assert not A.classify(cand("x3", "CHECKED_BY", "C02", "E09"))[0]  # new CHECKED_BY는 사람이
    assert not A.classify(cand("x4", "MAY_BE_CAUSED_BY", "S8", "C02"))[0]  # prior 없음
    conflict = cand("x5", "OWNS", "TM03", "SY17", {"scope": "data"})
    assert conflict["match"] == "conflict" and not A.classify(conflict)[0]
    fresh = cand("x6", "OWNS", "TM03", "SY13") | {"object": {"type": "System", "name": "길드", "id": None}}
    assert not A.classify(fresh)[0]


@pytest.fixture
def repo(tmp_path):
    """시드를 복사한 임시 저장소. 실제 시드는 건드리지 않는다 (M4 PR 5 결정 5)."""
    seed_dir = tmp_path / "ontology" / "seed"
    shutil.copytree(SEED_DIR, seed_dir)
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"],
                ["add", "."], ["commit", "-qm", "seed"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True)
    return tmp_path, seed_dir


def test_apply_to_seed_edits_only_what_it_should(repo):
    root, seed_dir = repo
    a, b = _new_dependency()
    batch = [cand("x1", "DEPENDS_ON", a, b), cand("x1b", "DEPENDS_ON", a, b, doc="CH-01"),
             cand("x2", "OWNS", "TM03", "SY13", {"scope": "logic"}),
             cand("x3", "ALIAS", "TM03", "서버팀"), cand("x4", "CHECKED_BY", "C02", "E09")]
    result = A.apply_to_seed(seed_dir, batch)
    assert result.applied == ["x1", "x1b", "x2", "x3"] and [m["candidate_id"] for m in result.manual] == ["x4"]
    seed = load_seed("org-a", seed_dir=seed_dir)
    assert b in next(s for s in seed["systems"] if s["id"] == a)["depends_on"]
    assert "서버팀" in next(t for t in seed["teams"] if t["id"] == "TM03")["aliases"]
    g = build_graph(seed)
    dep = next(r for r in g.rels["DEPENDS_ON"] if r["start"] == a and r["end"] == b)
    # 새 관계는 extracted, 같은 사실의 후보 둘은 시드에 한 번만 들어가고 출처가 합쳐진다
    assert dep["props"]["source"] == "extracted" and dep["props"]["source_doc_ids"] == ["CH-01", "PM-01"]
    owns = next(r for r in g.rels["OWNS"] if r["start"] == "TM03" and r["end"] == "SY13" and r["props"]["scope"] == "logic")
    assert owns["props"]["source"] == "manual" and owns["props"]["source_doc_ids"] == ["PM-01"]
    # 시드 파일은 반영한 줄만 바뀐다 (주석과 표기는 그대로)
    diff = subprocess.run(["git", "diff", "--numstat"], cwd=root, capture_output=True, text=True).stdout
    changed = {line.split("\t")[2]: int(line.split("\t")[1]) for line in diff.splitlines()}
    assert all(removed <= 1 for removed in changed.values()), diff
    assert set(changed) == {f"ontology/seed/{f}" for f in ("systems.yaml", "teams.yaml", "provenance.yaml")}


def test_apply_rolls_back_when_seed_check_fails(repo, monkeypatch):
    _, seed_dir = repo
    before = {p.name: p.read_text(encoding="utf-8") for p in seed_dir.glob("*.yaml")}

    def broken(seed):
        raise ValueError("깨진 시드")

    monkeypatch.setattr(A, "validate_seed", broken)
    a, b = _new_dependency()
    with pytest.raises(A.ApplyError):
        A.apply_to_seed(seed_dir, [cand("x1", "DEPENDS_ON", a, b)])
    assert {p.name: p.read_text(encoding="utf-8") for p in seed_dir.glob("*.yaml")} == before


def test_commit_only_on_a_clean_non_main_branch(repo):
    root, seed_dir = repo
    with pytest.raises(A.ApplyError):
        A.check_branch(root)  # main
    subprocess.run(["git", "switch", "-qc", "kb-demo"], cwd=root, check=True)
    assert A.check_branch(root) == "kb-demo"
    (root / "stray.txt").write_text("x")
    with pytest.raises(A.ApplyError):
        A.check_branch(root)


def _new_cause_link():
    """K에 없는 증상과 원인 쌍 하나."""
    causes = [c["id"] for c in SEED["causes"]]
    return next((s["id"], c) for s in SEED["symptoms"] for c in causes if not K.rels("MAY_BE_CAUSED_BY", s["id"], c))


def _free_owner_slot():
    """아무 팀도 맡지 않은 (시스템, scope) 자리 하나."""
    for system in SEED["systems"]:
        taken = {sc for o in system["owners"] if o.get("area") is None for sc in o["scopes"]}
        free = sorted({"logic", "data", "config", "operation", "execution"} - taken)
        if free:
            return system["id"], free[0]
    raise AssertionError("빈 담당 자리가 없다")


def test_same_link_with_another_prior_in_one_batch_goes_to_a_person(repo):
    """prior는 출처 키에 들어가지 않는다. 같은 증상과 원인에 다른 prior가 오면 앞 값만 남고 뒤 후보가 성공으로 보고되던 문제."""
    _, seed_dir = repo
    s, c = _new_cause_link()
    batch = [cand("p1", "MAY_BE_CAUSED_BY", s, c, {"prior": "높음"}),
             cand("p2", "MAY_BE_CAUSED_BY", s, c, {"prior": "높음"}, doc="CH-01"),
             cand("p3", "MAY_BE_CAUSED_BY", s, c, {"prior": "낮음"}, doc="CH-02")]
    result = A.apply_to_seed(seed_dir, batch)
    assert result.applied == ["p1", "p2"] and [m["candidate_id"] for m in result.manual] == ["p3"]
    refs = [r for r in next(x for x in load_seed("org-a", seed_dir=seed_dir)["symptoms"] if x["id"] == s)["causes"]
            if r["cause"] == c]
    assert refs == [{"cause": c, "prior": "높음"}]


def test_two_teams_for_one_owner_slot_in_one_batch_goes_to_a_person(repo):
    _, seed_dir = repo
    system, scope = _free_owner_slot()
    batch = [cand("o1", "OWNS", "TM03", system, {"scope": scope}), cand("o2", "OWNS", "TM05", system, {"scope": scope})]
    assert all(c["match"] == "new" for c in batch)  # K와만 비교하면 둘 다 새 사실이다
    result = A.apply_to_seed(seed_dir, batch)
    assert result.applied == ["o1"] and [m["candidate_id"] for m in result.manual] == ["o2"]


def test_seed_check_rejects_two_owners_for_one_slot():
    import copy

    from ontology.seed import validate_seed

    seed = copy.deepcopy(SEED)
    system = next(s for s in seed["systems"] if s["owners"])
    owner = system["owners"][0]
    other = next(t["id"] for t in seed["teams"] if t["id"] != owner["team"])
    system["owners"].append({"team": other, "scopes": [owner["scopes"][0]], **({"area": owner["area"]} if owner.get("area") else {})})
    with pytest.raises(ValueError, match="담당이 둘"):
        validate_seed(seed)


def test_repo_lock_is_exclusive(repo):
    import fcntl

    root, _ = repo
    with A.repo_lock(root):
        path = root / ".git" / "kb-apply.lock"
        with open(path) as f, pytest.raises(BlockingIOError):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_find_commit_reads_back_the_manual_list(repo):
    root, seed_dir = repo
    subprocess.run(["git", "switch", "-qc", "kb-demo"], cwd=root, check=True)
    a, b = _new_dependency()
    result = A.apply_to_seed(seed_dir, [cand("x1", "DEPENDS_ON", a, b), cand("x4", "CHECKED_BY", "C02", "E09")])
    run = {"run_id": "AR-abc", "approved_by_teams": ["TM03"], "executed_by_team": "TM05"}
    sha = A.commit_apply(run, result, root, seed_dir)
    assert A.find_commit("AR-abc", root) == (sha, result.manual)
    assert A.find_commit("AR-zzz", root) is None


def test_empty_provenance_keeps_the_seed_hash():
    """출처 파일이 비어 있으면 시드 해시가 그대로라 이전 결과를 다시 채점할 수 있다."""
    assert "provenance" not in SEED
    assert yaml.safe_load((SEED_DIR / "provenance.yaml").read_text(encoding="utf-8"))["provenance"] == []


# 아래는 테스트용 Neo4j(7688)에서 돈다


@pytest.fixture
def db(driver):
    from common.writers.events import reset_event_layer
    from common.writers.extraction import reset_extraction_layer
    from ontology.apply_schema import apply_schema
    from ontology.load import load_static

    apply_schema(driver)
    load_static(driver, SEED)
    reset_extraction_layer(driver)
    yield driver
    reset_extraction_layer(driver)
    reset_event_layer(driver)
    load_static(driver, SEED)


def _write(db, cands):
    from common.writers.extraction import record_candidates, record_documents
    from eval.corpus.check import load_plan
    from eval.corpus.load import document_fields

    doc = next(d for d in load_plan() if d["id"] == "PM-01")
    fields = document_fields(doc)
    record_documents(db, [fields])
    span = next(line for line in fields["text"].splitlines() if len(line) > 20)
    return record_candidates(db, run_id="x1", document_id="PM-01",
                             candidates=[c | {"evidence_spans": [span]} for c in cands])


@pytest.mark.integration
def test_review_then_a17_gate_then_apply(db, repo):
    from common.gate import gate
    from common.writers.events import RecordError, reset_event_layer
    from common.writers.extraction import list_candidates, review_candidate
    from tests.memory_tracer import MemoryTracer

    a, b = _new_dependency()
    ids = _write(db, [cand("_", "DEPENDS_ON", a, b), cand("_", "OWNS", "TM03", "SY13", {"scope": "logic"})])
    dep_id, owns_id = ids
    dep = next(c for c in list_candidates(db) if c["id"] == dep_id)
    assert len(dep["reviewers"]) >= 1
    # 배정된 팀이 모두 승인해야 승인된다. 배정되지 않은 팀은 판단할 수 없다
    with pytest.raises(RecordError):
        review_candidate(db, candidate_id=dep_id, team_id="TM12" if "TM12" not in dep["reviewers"] else "TM11",
                         decision="approve", now=NOW)
    for team in dep["reviewers"]:
        state = review_candidate(db, candidate_id=dep_id, team_id=team, decision="approve", now=NOW)
    assert state["status"] == "approved"
    with pytest.raises(RecordError):
        review_candidate(db, candidate_id=dep_id, team_id=dep["reviewers"][0], decision="reject", now=NOW)
    # 반려는 한 팀으로 충분하고 후보는 지우지 않는다
    owns = next(c for c in list_candidates(db) if c["id"] == owns_id)
    assert review_candidate(db, candidate_id=owns_id, team_id=owns["reviewers"][0], decision="reject",
                            now=NOW)["status"] == "rejected"

    tracer = MemoryTracer()
    with pytest.raises(gate.GateError):
        gate.request_knowledge_apply(db, tracer=tracer, candidate_ids=[owns_id], now=NOW)  # 반려된 후보
    run = gate.request_knowledge_apply(db, tracer=tracer, candidate_ids=[dep_id], now=NOW)
    assert run["status"] == "pending_approval" and run["candidate_ids"] == [dep_id]
    assert run["required_approver_teams"] == dep["reviewers"] and run["executed_by_team"] == "TM05"
    with pytest.raises(gate.GateError):
        gate.request_knowledge_apply(db, tracer=tracer, candidate_ids=[dep_id], now=NOW)  # 이미 요청됨
    # 사건 경로로는 A17을 요청할 수 없다
    with pytest.raises(A.ApplyError):
        A.execute(db, run_id=run["run_id"], tracer=tracer, now=NOW)  # 아직 승인 전
    for team in run["required_approver_teams"]:
        run = gate.approve(db, tracer=tracer, run_id=run["run_id"], team_id=team, now=NOW)
    assert run["status"] == "approved"
    # 평가 실행의 사건 계층 초기화는 A17 기록을 지우지 않는다 (M4 PR 5 결정 1)
    reset_event_layer(db)
    root, seed_dir = repo
    subprocess.run(["git", "switch", "-qc", "kb-demo"], cwd=root, check=True)
    out = A.execute(db, run_id=run["run_id"], tracer=tracer, now=NOW, root=root, seed_dir=seed_dir)
    assert out["applied"] == [dep_id] and out["run"]["status"] == "done" and out["run"]["commit"] == out["commit"]
    log = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=root, capture_output=True, text=True).stdout
    assert run["run_id"] in log
    with db.session() as s:
        row = s.run("MATCH (:System {id: $a})-[r:DEPENDS_ON]->(:System {id: $b}) RETURN r.source AS source",
                    a=a, b=b).single()
    assert row["source"] == "extracted"  # 다시 적재한 그래프에 반영됐다


@pytest.mark.integration
def test_gate_refuses_candidates_a_person_must_apply(db):
    from common.gate import gate
    from common.writers.extraction import list_candidates, review_candidate
    from tests.memory_tracer import MemoryTracer

    (cid,) = _write(db, [cand("_", "CHECKED_BY", "C02", "E09")])
    for team in next(c for c in list_candidates(db) if c["id"] == cid)["reviewers"]:
        review_candidate(db, candidate_id=cid, team_id=team, decision="approve", now=NOW)
    with pytest.raises(gate.GateError, match="사람이 직접"):
        gate.request_knowledge_apply(db, tracer=MemoryTracer(), candidate_ids=[cid], now=NOW)


def _approved_run(db, cands):
    from common.gate import gate
    from common.writers.extraction import list_candidates, review_candidate
    from tests.memory_tracer import MemoryTracer

    ids = _write(db, cands)
    for c in list_candidates(db):
        for team in c["reviewers"]:
            review_candidate(db, candidate_id=c["id"], team_id=team, decision="approve", now=NOW)
    tracer = MemoryTracer()
    run = gate.request_knowledge_apply(db, tracer=tracer, candidate_ids=ids, now=NOW)
    for team in run["required_approver_teams"]:
        run = gate.approve(db, tracer=tracer, run_id=run["run_id"], team_id=team, now=NOW)
    return run, tracer, ids


@pytest.mark.integration
def test_execute_resumes_after_a_failure_past_the_commit(db, repo, monkeypatch):
    """커밋 뒤에 다시 적재가 실패해도 같은 실행을 다시 누르면 새 커밋 없이 적재와 완료 기록만 한다."""
    import ontology.load

    root, seed_dir = repo
    subprocess.run(["git", "switch", "-qc", "kb-demo"], cwd=root, check=True)
    a, b = _new_dependency()
    run, tracer, (dep_id,) = _approved_run(db, [cand("_", "DEPENDS_ON", a, b)])
    real = ontology.load.load_static

    def down(driver, seed):
        raise ConnectionError("DB 연결 끊김")

    monkeypatch.setattr(ontology.load, "load_static", down)
    with pytest.raises(ConnectionError):
        A.execute(db, run_id=run["run_id"], tracer=tracer, now=NOW, root=root, seed_dir=seed_dir)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    monkeypatch.setattr(ontology.load, "load_static", real)
    out = A.execute(db, run_id=run["run_id"], tracer=tracer, now=NOW, root=root, seed_dir=seed_dir)
    assert out["resumed"] and out["commit"] == head and out["applied"] == [dep_id]
    assert out["run"]["status"] == "done" and out["run"]["commit"] == head
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip() == head


@pytest.mark.integration
def test_execute_without_seed_change_completes_without_commit(db, repo):
    """반영할 때 모든 후보가 시드와 어긋나 사람에게 돌아가면 커밋 없이 완료하고 목록을 실행 기록에 남긴다."""
    root, seed_dir = repo
    subprocess.run(["git", "switch", "-qc", "kb-demo"], cwd=root, check=True)
    s, c = _new_cause_link()
    run, tracer, (cid,) = _approved_run(db, [cand("_", "MAY_BE_CAUSED_BY", s, c, {"prior": "낮음"})])
    # 요청 뒤에 시드에 같은 관계가 다른 prior로 들어갔다
    path = seed_dir / "symptoms.yaml"
    from ruamel.yaml.comments import CommentedMap

    y = A._yaml()
    data = y.load(path.read_text(encoding="utf-8"))
    next(x for x in data["symptoms"] if x["id"] == s)["causes"].append(CommentedMap({"cause": c, "prior": "높음"}))
    with path.open("w", encoding="utf-8") as f:
        y.dump(data, f)
    subprocess.run(["git", "commit", "-qam", "다른 반영"], cwd=root, check=True)
    out = A.execute(db, run_id=run["run_id"], tracer=tracer, now=NOW, root=root, seed_dir=seed_dir)
    assert out["commit"] is None and [m["candidate_id"] for m in out["manual"]] == [cid]
    assert out["run"]["status"] == "done" and out["run"]["manual_candidate_ids"] == [cid]
