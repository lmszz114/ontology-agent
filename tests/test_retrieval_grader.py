"""검색 채점기 (M5 PR 2). Voyage를 부르지 않는다."""

import random
from pathlib import Path

import pytest
import yaml

from eval.corpus.check import load_plan
from eval.grader import retrieval as R

PLAN = load_plan()
DEV = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(Path("eval/scenarios/dev").glob("*.yaml"))]


def test_queries_skip_incidents_without_relevant_docs_and_use_report_text():
    qs = R.queries(DEV, PLAN)
    assert qs and all(q["relevant"] for q in qs)
    assert "SC-OOS-01" not in {q["scenario"] for q in qs}
    sc = next(s for s in DEV if s["id"] == qs[0]["scenario"])
    assert sc["reports"][0]["text"] in qs[0]["query"]
    # 검색어에 정답이나 원인 id가 들어가지 않는다
    assert not any(q["query"].count("C0") or q["query"].count("C1") for q in qs)


def test_score_recall_rr_and_stale():
    s = R.score(["K", "PM-01", "MT-02", "RB-01"], ["PM-01", "RB-01", "CH-06"], ["MT-02"])
    assert s == {"recall": pytest.approx(2 / 3), "rr": 0.5, "stale": True}
    assert R.score(["K"], ["PM-01"], []) == {"recall": 0.0, "rr": 0.0, "stale": False}


def test_random_recall_matches_the_expected_value():
    docs = ["A"] * 2 + ["B"] * 8  # 조각 10개 중 정답 문서 A의 조각 2개
    got = R.random_recall(docs, ["A"], 3, random.Random(1))
    assert got == pytest.approx(1 - (8 * 7 * 6) / (10 * 9 * 8), abs=0.03)


def test_evaluate_and_markdown_with_a_fake_search(monkeypatch):
    qs = [{"scenario": "S", "incident": 0, "query": "q", "relevant": ["PM-01"], "stale": ["MT-02"]}]
    order = [{"doc_id": d, "chunk_id": f"x:{d}:{i}", "text": "", "score": 1.0, "rank": i}
             for i, d in enumerate(["K", "MT-02", "K", "PM-01", "RB-01", "K", "K", "K", "K", "K"], start=1)]
    monkeypatch.setattr(R, "search", lambda *a, **kw: order[:kw["k"]])
    result = R.evaluate(None, None, qs, {"paragraph": ["K"] * 9 + ["PM-01"], "fixed": ["PM-01"] * 2 + ["K"] * 8})
    assert result["summary"]["paragraph/vector/k3"] == {"recall": 0.0, "mrr": 0.0, "stale_rate": 1.0}
    assert result["summary"]["fixed/rerank/k5"] == {"recall": 1.0, "mrr": round(1 / 3, 4), "stale_rate": 1.0}
    assert len(result["summary"]) == 12 and len(result["random_recall"]) == 6
    md = R.markdown({"run": {"date": "d", "embed_model": "e", "rerank_model": "r"}} | result)
    assert md.count("\n| ") == 14  # 머리 줄, 구분 줄, 설정 12줄


def test_main_refuses_out_of_results_dirs_and_existing_results(tmp_path, capsys, monkeypatch):
    """결과 폴더 규칙은 진단 실행기와 같다. 이미 있는 결과는 덮어쓰지 않는다 (M5 점검). Voyage를 부르기 전에 멈춘다."""
    from eval.grader import retrieval as R
    from eval.runner import runner

    assert R.main(["--out", str(tmp_path / "x")]) == 1
    assert "eval/results" in capsys.readouterr().err
    done = runner.ROOT / "eval" / "results" / "2026-11-26-m5-retrieval-dev"
    assert R.main(["--out", str(done)]) == 1
    assert "이미 결과가 있다" in capsys.readouterr().err
