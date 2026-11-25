"""G1 점검: 문서가 K 문장을 그대로 옮긴 비율 (M4 점검)."""

from eval.corpus.copy_check import check, sentences
from ontology.seed import load_seed

SEED = load_seed("org-a")


def test_counts_same_and_contained_sentences(monkeypatch):
    expect = SEED["causes"][1]["checked_by"][1]["expect"]  # C02의 확정 조건 설명
    doc = {"id": "X-1", "split": "dev", "kind": "postmortem", "title": "t", "text": f"제목\n\n{expect}.\n로그를 보니 {expect}고 했다."}
    plan = [doc]
    import eval.corpus.copy_check as cc

    monkeypatch.setattr(cc, "document_fields", lambda d: {"text": d["text"]})  # 설계표 문서 대신 위 문장을 쓴다
    r = check(plan, SEED)["splits"]["dev"]
    assert r["same"] == 1 and r["contains"] == 1


def test_sentence_split():
    assert sentences("# 제목\n\n첫 문장이다. 두 번째다!\n- 목록") == ["제목", "첫 문장이다", "두 번째다", "목록"]


def test_keeps_leading_numbers_that_are_not_list_markers():
    assert sentences("1. 첫 항목\n10분 이상 지연이 발생한다.\n2) 둘째") == ["첫 항목", "10분 이상 지연이 발생한다", "둘째"]
