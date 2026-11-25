"""현장 문서 설계표 검사 (eval/corpus/check.py, 4단계 3-5)."""

import copy

import pytest

from eval.corpus.check import check_plan, load_plan
from ontology.seed import load_seed

SEED = load_seed()
PLAN = load_plan()


def test_real_plan_is_valid():
    assert check_plan(SEED, PLAN) == []
    assert len(PLAN) == 74 and sum(d["split"] == "dev" for d in PLAN) == 30


def _doc(did):
    return next(d for d in PLAN if d["id"] == did)


def _broken(did, mutate):
    docs = copy.deepcopy(PLAN)
    mutate(next(d for d in docs if d["id"] == did))
    return check_plan(SEED, docs)


@pytest.mark.parametrize("did,mutate,message", [
    # K에 없는 사실: 담당 팀, 사전 확률, 범위, 조건이 틀린 경우
    ("PM-01", lambda d: next(f for f in d["facts"] if f["rel"] == "OWNS").update(subject="TM04"), "K에 없는 사실"),
    ("PM-01", lambda d: next(f for f in d["facts"] if f["rel"] == "MAY_BE_CAUSED_BY")["props"].update(prior="낮음"), "K에 없는 사실"),
    ("PM-04", lambda d: next(f for f in d["facts"] if f["rel"] == "LOCATED_IN")["props"].update(area="probability"), "K에 없는 사실"),
    ("PM-05", lambda d: next(f for f in d["facts"] if f["rel"] == "RESOLVED_BY" and f["object"] == "P7")["props"].update(condition="미지급"), "K에 없는 사실"),
    ("CH-01", lambda d: d["facts"].append({"rel": "CHECKED_BY", "subject": "C02", "object": "E09", "props": {"required": True}}), "K에 없는 사실"),
    ("CH-01", lambda d: d["facts"].append({"rel": "ASSIGNED_TO", "subject": "R01", "object": "TM07"}), "추출 대상이 아닌"),
    # 옛날 정보는 K와 달라야 하고 담당 모순이어야 한다
    ("RB-08", lambda d: d["stale"][0].update(subject="TM10"), "K와 같다"),
    ("RB-08", lambda d: d["stale"][0].update(rel="DEPENDS_ON", object="SY05", subject="SY18"), "모순되지 않는다"),
    # 별칭은 K에 있는 팀, 시스템, 증상, 원인에만 붙인다 (PR #27 리뷰)
    ("PM-01", lambda d: next(f for f in d["facts"] if f["rel"] == "ALIAS").update(subject="SY_DOES_NOT_EXIST"), "별칭을 붙일 수 없는"),
    ("PM-01", lambda d: next(f for f in d["facts"] if f["rel"] == "ALIAS").update(subject="E01"), "별칭을 붙일 수 없는"),
    # 방해 문서에는 사실이 없다
    ("NZ-01", lambda d: d["facts"].append({"rel": "ALIAS", "subject": "SY13", "object": "메일함"}), "방해 문서"),
    ("PM-01", lambda d: d.update(kind="chat"), "접두어"),
])
def test_plan_errors_are_reported(did, mutate, message):
    errors = _broken(did, mutate)
    assert errors and any(message in e for e in errors), errors


def test_every_cause_needs_a_postmortem():
    docs = [d for d in copy.deepcopy(PLAN) if d["id"] != "PM-11"]  # C01의 유일한 회고
    assert any("C01" in e and "회고가 없는" in e for e in check_plan(SEED, docs))


def test_relevant_docs_follow_cause_and_skip_stale():
    """검색 정답 문서는 정답 원인을 다루는 문서이고 옛날 정보 문서는 뺀다 (eval/corpus/relevant.py)."""
    from eval.corpus.relevant import relevant_docs, stale_docs
    from eval.validate import load_scenarios

    dev = {s["id"]: s for s in load_scenarios("dev")}
    test = {s["id"]: s for s in load_scenarios("test")}
    # C11(이벤트 설정 오류): 회고, 대화, 회의록, 런북. 이벤트 설정을 서버개발팀이 하던 옛날 회의록(MT-02)은 뺀다
    assert relevant_docs(dev["SC-S4-01"]) == [["CH-06", "MT-01", "PM-05", "RB-01", "RB-02"]]
    assert stale_docs(dev["SC-S4-01"]) == [["MT-02"]]
    # 사건 둘이면 사건마다, 원인 미확정과 범위 밖은 비어 있다
    assert len(relevant_docs(test["SC-S6-06"])) == 2
    assert relevant_docs(test["SC-S1-09"]) == [[]] and relevant_docs(test["SC-OOS-02"]) == []
    # 모든 원인 있는 사건은 정답 문서가 하나 이상 있다
    for s in list(dev.values()) + list(test.values()):
        from eval.labels import label_incidents
        for incident, docs in zip(label_incidents(s["expected"]), relevant_docs(s)):
            assert bool(docs) == bool(incident.get("root_cause")), s["id"]


def test_generated_docs_exist_for_every_planned_doc():
    from eval.corpus.generate import doc_path

    assert all(doc_path(d).exists() for d in PLAN)
