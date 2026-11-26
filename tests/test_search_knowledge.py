"""구성 B(문서 검색 지식 공급자) 검증 (M5 PR 3). Voyage와 Neo4j 없이 가짜 검색 결과로 본다."""

import asyncio

import pytest

from common.knowledge import search as S
from common.knowledge import text_prompts as P
from common.knowledge.base import KnowledgeUnavailable
from common.tool_calls import ToolCallError
from ontology.seed import load_seed
from tests.memory_tracer import MemoryTracer
from tests.test_text_knowledge import CHANGES, Script

SEED = load_seed("org-a")
SETTINGS = S.SearchSettings(scheme="paragraph", rerank=False, k=5)
ROWS = [{"rank": 1, "doc_id": "PM-01", "chunk_id": "paragraph:PM-01:004", "text": "원인은 메일함 지급 누락이다.", "score": 0.9},
        {"rank": 2, "doc_id": "K", "chunk_id": "paragraph:K:050", "text": "## 원인 (Cause)\n- C02: name=우편함 지급 누락", "score": 0.8},
        {"rank": 3, "doc_id": "PM-01", "chunk_id": "paragraph:PM-01:005", "text": "메일함 로직은 서버팀이 맡는다.", "score": 0.7}]
RANK = {"candidates": [{"cause_id": "C02", "score": 3, "rank": 1, "change_ids": [], "reason": "검색 결과"}]}


def provider(script, monkeypatch, rows=ROWS, fail=False):
    from types import SimpleNamespace

    from common.models import LLM, load_model_config

    calls = []

    async def fake_call(client, tracer, *, server, name, args):
        calls.append((server, name, args))
        if fail:
            raise ToolCallError("doc-search.search_docs 오류: 색인 없음")
        return rows

    monkeypatch.setattr(S, "call_tool_data", fake_call)
    llm = LLM(load_model_config("org-a"), MemoryTracer(), SimpleNamespace(messages=script))
    k = S.SearchKnowledge(docs=None, tracer=MemoryTracer(), settings=SETTINGS, llm=llm, seed=SEED, results=[],
                          changes=lambda: list(CHANGES))
    return k, calls


def test_b_does_not_answer_before_its_incident_search(monkeypatch):
    k, _ = provider(Script(), monkeypatch)
    with pytest.raises(KnowledgeUnavailable, match="검색 결과가 없다"):
        asyncio.run(k.rank_causes("S1", "2026-12-03T20:14:00+09:00"))


def test_search_uses_report_text_and_settings_and_feeds_only_retrieved_chunks(monkeypatch):
    script = Script(RANK, RANK)
    k, calls = provider(script, monkeypatch)
    record = asyncio.run(k.begin_incident("IN-1", ["결제했는데 아이템이 안 와요", "캐시 샀는데 우편함이 비었어요"]))
    assert calls == [("doc-search", "search_docs", {"query": "결제했는데 아이템이 안 와요\n캐시 샀는데 우편함이 비었어요",
                                                     "scheme": "paragraph", "rerank": False, "k": 5})]
    assert record["docs"] == ["PM-01", "K"] and len(record["chunks"]) == 3 and record["settings"] == SETTINGS.as_dict()
    asyncio.run(k.rank_causes("S1", "2026-12-03T20:14:00+09:00"))
    system = script.params[0]["system"][0]["text"]
    assert system.startswith(P.SYSTEM_B) and "원인은 메일함 지급 누락이다." in system and "# ID 목록" in system
    # 기준 지식 전체가 아니라 검색한 조각만 넣는다. 관계(담당 시스템 등)는 검색 결과에 나온 것뿐이다
    assert "담당 시스템 ->" not in system
    # 사건이 바뀌면 앞 사건의 답을 다시 쓰지 않는다
    asyncio.run(k.begin_incident("IN-2", ["또 안 와요"]))
    asyncio.run(k.rank_causes("S1", "2026-12-03T20:14:00+09:00"))
    assert len(script.params) == 2 and k.prompt_version == P.PROMPT_VERSION_B


def test_search_failure_is_reported_not_answered_from_nothing(monkeypatch):
    k, _ = provider(Script(), monkeypatch, fail=True)
    with pytest.raises(KnowledgeUnavailable, match="문서 검색 실패"):
        asyncio.run(k.begin_incident("IN-1", ["로그인이 안 돼요"]))
    assert k.errors and k.system is None


def test_id_listing_has_names_but_no_relations():
    text = S.id_listing(SEED)
    assert "- C02: " in text and "- TM03: " in text and "- S1: " in text
    assert "->" not in text and "scope" not in text and "prior" not in text


def test_a_prompt_is_unchanged_and_b_reuses_its_rules():
    assert P.PROMPT_VERSION == "knowledge-a-v2" and P.SYSTEM.startswith("너는 게임 라이브 운영 조직의 지식 담당이다. 아래 기준 지식만")
    assert P.SYSTEM_B.endswith(P.SYSTEM[P.SYSTEM.index("# 판단 규칙"):])


def test_b_needs_chosen_settings(monkeypatch):
    from agents.diagnosis import run

    monkeypatch.setattr(S, "B_SETTINGS", None)
    with pytest.raises(ValueError, match="검색 설정"):
        run._b_settings()
    monkeypatch.setattr(S, "B_SETTINGS", SETTINGS)
    assert run._b_settings() == SETTINGS


def test_grader_requires_search_records_for_b():
    from common.retrieval.chunking import all_chunks, corpus_hash
    from eval.corpus.check import load_plan
    from eval.grader.grade import GradeError, check_basis

    run = {"split": "dev", "scenario": "SC-S1-01", "config": "B", "git_commit": "0" * 40}
    with pytest.raises(GradeError, match="검색 색인"):
        check_basis({"run": run}, SEED)
    bad = run | {"search_settings": SETTINGS.as_dict(), "retrieval_index": {"corpus_hash": "x", "model": "voyage-4"}}
    with pytest.raises(GradeError, match="다르다"):
        check_basis({"run": bad}, SEED)
    good = bad | {"retrieval_index": {"corpus_hash": corpus_hash(all_chunks(SEED, load_plan())), "model": "voyage-4"}}
    check_basis({"run": good}, SEED)


def test_questions_add_their_own_code_made_search(monkeypatch):
    script = Script("{}", "{}")  # 답의 모양은 이 테스트의 관심이 아니다. 두 번 틀리면 지식을 받지 못한 것으로 끝난다
    k, calls = provider(script, monkeypatch, rows=ROWS)
    asyncio.run(k.begin_incident("IN-1", ["아이템이 안 와요"]))
    extra = [{"rank": 1, "doc_id": "K", "chunk_id": "fixed:K:020", "text": "C02 판정 근거 -> E02 (required=True)", "score": 0.9},
             ROWS[0]]

    async def fake_call(client, tracer, *, server, name, args):
        calls.append((server, name, args))
        return extra

    monkeypatch.setattr(S, "call_tool_data", fake_call)
    with pytest.raises(KnowledgeUnavailable):
        asyncio.run(k.get_checks("C02"))
    query = calls[-1][2]["query"]
    assert query.startswith("C02 우편함 지급 누락") and "판정 근거" in query and calls[-1][2]["k"] == 5
    system = script.params[0]["system"][0]["text"]
    # 사건 검색 결과에 질문 검색 결과를 더하고, 이미 있는 조각은 다시 넣지 않는다
    assert "C02 판정 근거 -> E02" in system and system.count("원인은 메일함 지급 누락이다.") == 1
    assert k._record["extra"][0]["question"] == ["checks", "C02"]
    before = len(calls)
    with pytest.raises(KnowledgeUnavailable):
        asyncio.run(k.get_checks("C02"))  # 이미 물은 질문(실패도 기억한다)은 다시 검색하지 않는다
    assert len(calls) == before and len(script.params) == 2
