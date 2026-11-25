"""추출 에이전트 처리 단계와 추출 채점기 (M4 PR 4). 가짜 LLM으로 돌아 비용이 없다."""

import json
import sys
import types

import pytest
import yaml

from agents.extraction import pipeline
from agents.extraction.rules import Knowledge
from eval.corpus.check import load_plan
from eval.corpus.generate import doc_path
from ontology.seed import load_seed

SEED = load_seed("org-a")
K = Knowledge.from_seed(SEED)
PLAN = load_plan()
PM01 = next(d for d in PLAN if d["id"] == "PM-01")
TEXT = doc_path(PM01).read_text(encoding="utf-8")


class FakeLLM:
    """작업 이름마다 돌려줄 JSON을 순서대로 준다."""

    def __init__(self, replies: dict[str, list], stop="end_turn"):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls = []
        self.stop = stop

    def call(self, *, task, **kwargs):
        self.calls.append((task, kwargs))
        data = self.replies[task].pop(0)
        return types.SimpleNamespace(
            model="fake", usage={"input_tokens": 10, "output_tokens": 5}, cost={"total": 0.001},
            message=types.SimpleNamespace(stop_reason=self.stop), json=lambda: data)


def raw(rel, st, sn, ot, on, evidence, **props):
    base = {"relation": rel, "subject_type": st, "subject_name": sn, "object_type": ot, "object_name": on,
            "scope": "", "area": "", "prior": "", "required": "", "expect": "", "condition": "",
            "evidence": evidence, "confidence": 0.8}
    return base | props


def sentence_with(word: str) -> str:
    return next(line.strip() for line in TEXT.splitlines() if word in line)


def test_paragraphs_split_on_blank_lines():
    assert pipeline.paragraphs("가\n나\n\n\n다\n") == ["가\n나", "다"]


def test_pipeline_drops_bad_candidates_normalizes_and_makes_aliases():
    """원문에 없는 근거, 맞지 않는 타입은 버린다. 이름은 코드로 먼저 잇고 남은 것만 LLM에게 묻는다."""
    mail = sentence_with("메일함")
    replies = {
        "extraction": [{"candidates": [
            # 시드 별칭(메일함)과 팀 별칭(서버팀)은 코드가 잇는다. 서버팀은 시드에 없는 이름이라 LLM이 고른다
            raw("OWNS", "Team", "서버팀", "System", "메일함", [mail], scope="logic"),
            raw("OWNS", "Team", "서버팀", "System", "메일함", ["원문에 없는 문장"], scope="logic"),
            raw("OWNS", "System", "메일함", "Team", "서버팀", [mail]),
            raw("OBSERVED_IN", "SymptomType", "결제 후 아이템 미지급", "System", "메일함", [mail], prior="높음"),
        ]}],
        "normalization": [{"matches": [{"index": 0, "node_id": "TM03"}]}],
    }
    llm = FakeLLM(replies)
    r = pipeline.run_document(llm, K, "PM-01", TEXT)
    reasons = sorted(d["reason"] for d in r.dropped)
    assert reasons == ["관계와 노드 타입이 맞지 않는다", "근거 문장이 원문에 없다"]
    owns = next(c for c in r.candidates if c["relation"] == "OWNS")
    assert owns["subject"]["id"] == "TM03" and owns["object"]["id"] == "SY13" and owns["match"] == "existing"
    # 관계에 없는 속성(OBSERVED_IN의 prior)은 버린다
    observed = next(c for c in r.candidates if c["relation"] == "OBSERVED_IN")
    assert observed["props"] == {}
    # 서버팀은 TM03의 이름도 별칭도 아니어서 별칭 후보가 생긴다. 메일함은 시드 별칭이라 생기지 않는다
    aliases = [(c["subject"]["id"], c["object"]["name"]) for c in r.candidates if c["relation"] == "ALIAS"]
    assert aliases == [("TM03", "서버팀")]
    # LLM에게는 코드가 잇지 못한 이름만 묻는다
    asked = [kw["messages"][0]["content"] for task, kw in llm.calls if task == "normalization"]
    assert len(asked) == 1 and "서버팀" in asked[0] and "메일함" not in asked[0].split("문서가 쓴 이름:")[1]
    assert r.cost == 0.002 and r.stats["raw"] == 4


def test_normalization_ignores_ids_of_another_type():
    replies = {"extraction": [{"candidates": [raw("OWNS", "Team", "서버팀", "System", "우편함", [sentence_with("메일함")])]}],
               "normalization": [{"matches": [{"index": 0, "node_id": "SY13"}]}]}
    r = pipeline.run_document(FakeLLM(replies), K, "PM-01", TEXT)
    assert r.candidates[0]["subject"]["id"] is None and r.candidates[0]["match"] == "new"


def test_unfinished_response_is_not_used():
    with pytest.raises(pipeline.ExtractionError):
        pipeline.run_document(FakeLLM({"extraction": [{"candidates": []}]}, stop="max_tokens"), K, "PM-01", TEXT)


def test_run_rejects_test_documents_and_bad_folders(monkeypatch, capsys):
    from agents.extraction import run

    test_doc = next(d["id"] for d in PLAN if d["split"] == "test")
    assert run.main(["--out", "eval/results/x-extract", "--only", test_doc]) == 1
    assert "M8" in capsys.readouterr().err
    assert run.main(["--out", "/tmp/elsewhere"]) == 1
    assert run.main(["--out", "eval/results/a b"]) == 1


def _write_rep(tmp_path, rep, docs):
    d = tmp_path / rep
    d.mkdir(parents=True)
    for doc_id, cands, dropped, raw_n in docs:
        (d / f"{doc_id}.yaml").write_text(yaml.safe_dump({
            "document_id": doc_id, "candidates": cands, "dropped": dropped, "stats": {"raw": raw_n},
            "cost_usd": 0.001}, allow_unicode=True), encoding="utf-8")


def _cand(rel, s, o, props=None, match="existing", s_name=None, o_name=None):
    return {"relation": rel, "subject": {"type": "x", "name": s_name or s, "id": s},
            "object": {"type": "x", "name": o_name or o, "id": None if rel == "ALIAS" else o},
            "props": props or {}, "match": match}


def test_grader_counts_against_plan(tmp_path, monkeypatch):
    import hashlib

    from eval.corpus.check import PLAN_PATH
    from eval.grader import extraction as g

    (tmp_path / "run.yaml").write_text(yaml.safe_dump({
        "plan_hash": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(), "prompt_version": "t", "git_commit": "c",
        "git_dirty": False, "models": {}, "started_at": "s"}), encoding="utf-8")
    facts = [f for f in PM01["facts"] if f["rel"] != "ALIAS"]
    first, second = facts[0], facts[1]
    good = _cand(first["rel"], first["subject"], first["object"], first.get("props"))
    # 속성이 틀린 정답 하나, K에 있지만 정답에 없는 후보 하나, 별칭 하나
    wrong = dict(second.get("props") or {})
    for key in wrong:
        wrong[key] = "data" if key == "scope" else "낮음"
    half = _cand(second["rel"], second["subject"], second["object"], wrong)
    extra = _cand("OBSERVED_IN", "S8", "SY10")
    alias = next(f for f in PM01["facts"] if f["rel"] == "ALIAS")
    named = _cand("OWNS", alias["subject"], "SY13", s_name=alias["object"]) if alias["subject"].startswith("TM") \
        else _cand("OWNS", "TM03", alias["subject"], o_name=alias["object"])
    alias_c = _cand("ALIAS", alias["subject"], alias["object"], match="new")
    noise = next(d for d in PLAN if d["kind"] == "distractor" and d["split"] == "dev")
    _write_rep(tmp_path, "r1", [
        ("PM-01", [good, half, extra, named, alias_c], [{"reason": "근거 문장이 원문에 없다"}], 6),
        (noise["id"], [extra], [], 1)])
    _write_rep(tmp_path, "r2", [("PM-01", [good], [], 1)])
    result = g.grade(tmp_path)
    r1 = result["per_rep"]["r1"]
    gold_n = len(PM01["facts"]) + len(PM01.get("stale") or [])
    # 맞은 것: good, half(속성만 틀림), named(PM-01의 TM03 OWNS SY13), 별칭. extra는 K에 있지만 정답에 없다
    assert r1["overall"]["tp"] == 4 and r1["overall"]["fp"] == 2 and r1["overall"]["fn"] == gold_n - 4
    assert r1["distractor_candidates"] == 1 and r1["false_positive_in_k"] >= 2
    assert r1["evidence_match_rate"] == round(6 / 7, 4)  # 두 문서의 원래 후보 7개 중 근거가 없는 1개
    assert r1["normalization_accuracy"] == 1.0
    assert 0 < r1["props_accuracy"] < 1
    assert result["reps"] == 2 and result["summary"]["overall"]["f1"]["stdev"] is not None


def test_grader_refuses_changed_plan(tmp_path):
    from eval.grader import extraction as g

    (tmp_path / "run.yaml").write_text(yaml.safe_dump({"plan_hash": "old"}), encoding="utf-8")
    with pytest.raises(g.GradeError):
        g.grade(tmp_path)
