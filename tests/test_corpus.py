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


class _FakeResult:
    def __init__(self, data, stop="end_turn", text=""):
        import types
        self._data, self.cost = data, {"total": 0.001}
        self.message = types.SimpleNamespace(stop_reason=stop)
        self.text, self.model = text, "fake"

    def json(self):
        return self._data


class _FakeLLM:
    def __init__(self, result):
        self.result = result

    def call(self, **kwargs):
        return self.result


def test_distractor_violation_needs_a_quote_from_the_text():
    """방해 문서는 원문에 글자 그대로 있는 인용만 위반으로 본다. 빈 인용과 없는 인용은 응답 오류로 남긴다 (PR #28 리뷰)."""
    from eval.corpus.verify import verify_doc

    doc = _doc("NZ-01")
    text = "겨울 이벤트 일정 회의. 서버개발팀이 우편함을 맡는다."
    ok, _ = verify_doc(_FakeLLM(_FakeResult({"quotes": ["", "원문에 없는 문장"]})), doc, text, {})
    assert ok["ok"] and ok["unexpected"] == [] and ok["invalid_quotes"] == ["", "원문에 없는 문장"]
    bad, _ = verify_doc(_FakeLLM(_FakeResult({"quotes": ["서버개발팀이 우편함을 맡는다."]})), doc, text, {})
    assert not bad["ok"] and bad["unexpected"] == ["서버개발팀이 우편함을 맡는다."]


def test_alias_is_checked_in_text_not_by_the_model():
    from eval.corpus.verify import missing_aliases

    doc = _doc("PM-08")  # 거래소 = '경매장'
    assert missing_aliases(doc, "경매장 등록이 실패했다") == []
    assert [m["fact"]["object"] for m in missing_aliases(doc, "거래소 등록이 실패했다")] == ["경매장"]


@pytest.mark.parametrize("module", ["eval.corpus.generate", "eval.corpus.verify"])
def test_unknown_only_ids_are_rejected(module, monkeypatch, capsys):
    import importlib
    import sys

    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module, "--only", "PM-001"])
    with pytest.raises(SystemExit):
        mod.main()
    assert "설계표에 없는 문서 id" in capsys.readouterr().err


def test_truncated_generation_is_not_saved(tmp_path, monkeypatch):
    """끝나지 않은 응답(max_tokens)은 정식 경로에 쓰지 않고 비용만 기록한다 (PR #28 리뷰)."""
    import sys

    from eval.corpus import generate

    monkeypatch.setattr(generate, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(generate, "RECORD_PATH", tmp_path / "generation.yaml")
    monkeypatch.setattr(generate, "Tracer", type("T", (), {"from_env": staticmethod(lambda: _NullTracer())}))
    monkeypatch.setattr(generate, "make_anthropic_client", lambda: None)
    monkeypatch.setattr(generate, "LLM", lambda *a: _FakeLLM(_FakeResult(None, stop="max_tokens", text="# 잘린 문서")))
    monkeypatch.setattr(sys, "argv", ["generate", "--only", "NZ-01"])
    assert generate.main() == 1
    assert not (tmp_path / "docs" / "dev" / "NZ-01.md").exists()
    import yaml
    record = yaml.safe_load((tmp_path / "generation.yaml").read_text(encoding="utf-8"))
    assert record["documents"]["NZ-01"]["generations"] == 1 and record["documents"]["NZ-01"]["stop_reason"] == "max_tokens"


class _NullTracer:
    def run(self, *a, **k):
        import contextlib
        return contextlib.nullcontext()

    def flush(self):
        pass
