"""근거 문장 표본 확인 (M4 점검): 표본과 판정 입력. LLM은 부르지 않는다."""

from pathlib import Path

from eval.grader.evidence_sample import claim, load_rep, prompt, sample

EXTRACT_DEV = Path("eval/results/2026-11-25-m4-extract-dev")


def test_sample_is_stable_and_spread_over_relations():
    cands = load_rep(EXTRACT_DEV, "r1")
    a, b = sample(cands, 30), sample(cands, 30)
    assert [c["candidate"] for c in a] == [c["candidate"] for c in b]
    assert len({c["relation"] for c in a}) == 8 and len(a) == 30


def test_prompt_has_claim_spans_and_paragraph_but_no_gold():
    from eval.corpus.check import load_plan
    from eval.corpus.load import document_fields

    docs = {d["id"]: document_fields(d)["text"] for d in load_plan() if d["split"] == "dev"}
    c = next(c for c in load_rep(EXTRACT_DEV, "r1") if c["relation"] == "OWNS")
    text = prompt(c, docs)
    assert claim(c) in text and all(s in text for s in c["evidence_spans"])
    assert "facts" not in text and "match" not in text
