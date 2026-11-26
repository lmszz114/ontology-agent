"""문서 검색: 조각 나누기, Voyage 클라이언트, 색인 적재와 검색 (M5 PR 1). Voyage를 부르지 않는다."""

import json

import httpx
import pytest

from common.retrieval import chunking as C
from common.retrieval import voyage as V
from common.retrieval.search import SearchError, docs_of, search
from eval.corpus.check import load_plan
from ontology.seed import load_seed
from tests.fake_voyage import FakeVoyage

SEED = load_seed("org-a")
PLAN = load_plan()


def test_paragraph_chunks_keep_headings_with_the_next_paragraph():
    text = "# 제목\n\n작성: 팀\n\n## 원인\n\n원인은 우편함이다.\n\n## 조치\n1. 공지\n2. 보상"
    assert [c.text for c in C.chunk("X", text, "paragraph")] == ["# 제목\n작성: 팀", "## 원인\n원인은 우편함이다.",
                                                                  "## 조치\n1. 공지\n2. 보상"]


def test_k_text_is_chunked_per_item_with_its_section():
    parts = C.chunk(C.K_DOC_ID, "# 기준 지식\n\n## 팀 (Team)\n\n- TM01: a\n  - x\n- TM02: b\n\n## 역할 (Role)\n\n- RL01: c",
                    "paragraph")
    assert [p.text for p in parts] == ["## 팀 (Team)\n- TM01: a\n  - x", "## 팀 (Team)\n- TM02: b", "## 역할 (Role)\n- RL01: c"]


def test_fixed_chunks_overlap_and_cover_the_text():
    text = "가" * 1250
    parts = C.chunk("X", text, "fixed")
    assert [len(p.text) for p in parts] == [500, 500, 450]
    assert all(len(p.text) <= C.FIXED_SIZE for p in parts)
    assert parts[1].text[:C.FIXED_OVERLAP] == parts[0].text[-C.FIXED_OVERLAP:]


def test_all_chunks_cover_k_and_every_document_with_unique_ids():
    chunks = C.all_chunks(SEED, PLAN)
    assert len({c.id for c in chunks}) == len(chunks)
    for scheme in C.SCHEMES:
        assert {c.doc_id for c in chunks if c.scheme == scheme} == {C.K_DOC_ID} | {d["id"] for d in PLAN}
    # 검색 대상에는 문서 원문만 넣는다. 설계표의 정답이나 split 표시는 들어가지 않는다
    assert not any("stale" in c.text or "split:" in c.text for c in chunks)


def test_corpus_hash_follows_text():
    a = C.chunk("X", "하나\n\n둘", "paragraph")
    b = C.chunk("X", "하나\n\n셋", "paragraph")
    assert C.corpus_hash(a) != C.corpus_hash(b) and C.corpus_hash(a) == C.corpus_hash(list(a))


def _client(tmp_path, handler):
    v = V.Voyage(cache_dir=tmp_path)
    v._http = httpx.Client(transport=httpx.MockTransport(handler))
    return v


def test_voyage_embeds_in_batches_caches_and_retries(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(V.time, "sleep", lambda s: None)

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(len(t))]} for i, t in
                                                  enumerate(body["input"])], "usage": {"total_tokens": 7}})

    monkeypatch.setattr(V, "BATCH", 2)
    monkeypatch.setattr(V, "MAX_CHARS", 100)
    v = _client(tmp_path, handler)
    assert v.embed(["a", "bb", "ccc"], "document") == [[1.0], [2.0], [3.0]]
    assert [len(c["input"]) for c in calls] == [2, 2, 1]  # 429 뒤 같은 묶음을 다시 보낸다
    assert calls[-1]["model"] == V.EMBED_MODEL and calls[-1]["input_type"] == "document"
    # 같은 텍스트는 캐시에서 읽는다. 다른 프로세스(새 클라이언트)도 파일 캐시를 읽는다
    v2 = _client(tmp_path, handler)
    assert v2.embed(["bb", "a"], "document") == [[2.0], [1.0]] and len(calls) == 3
    # 입력 종류가 다르면 다른 임베딩이다
    v2.embed(["a"], "query")
    assert len(calls) == 4 and v.usage.embed_tokens == 14


def test_batches_respect_count_and_characters(monkeypatch):
    monkeypatch.setattr(V, "BATCH", 3)
    monkeypatch.setattr(V, "MAX_CHARS", 10)
    items = [(str(i), t) for i, t in enumerate(["aaaa", "bbbb", "cccc", "d" * 20, "e"])]
    assert [[k for k, _ in b] for b in V._batches(items)] == [["0", "1"], ["2"], ["3"], ["4"]]


def test_voyage_errors_are_reported(tmp_path):
    v = _client(tmp_path, lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(V.VoyageError, match="401"):
        v.embed(["a"], "query")
    with pytest.raises(ValueError):
        v.embed(["a"], "passage")


def test_search_rejects_bad_arguments():
    for kwargs in ({"scheme": "sentence", "k": 3}, {"scheme": "fixed", "k": 0}, {"scheme": "fixed", "k": 31}):
        with pytest.raises(SearchError):
            search(None, FakeVoyage(), "로그인", **kwargs)
    with pytest.raises(SearchError):
        search(None, FakeVoyage(), "  ", scheme="fixed", k=3)


def test_docs_of_keeps_first_appearance_order():
    rows = [{"doc_id": "PM-01"}, {"doc_id": "K"}, {"doc_id": "PM-01"}, {"doc_id": "RB-03"}]
    assert docs_of(rows) == ["PM-01", "K", "RB-03"]


@pytest.fixture
def indexed(driver):
    from common.retrieval.index import write_index
    from ontology.apply_schema import apply_schema

    apply_schema(driver)
    plan = [d for d in PLAN if d["id"] in ("PM-01", "PM-02", "PM-03", "RB-05")]
    chunks = C.all_chunks(SEED, plan)
    fake = FakeVoyage()
    write_index(driver, chunks, fake.embed([c.text for c in chunks], "document"))
    yield driver, chunks, fake
    with driver.session() as s:
        s.run("MATCH (n) WHERE n:Chunk OR n:ChunkIndexMeta DETACH DELETE n").consume()


@pytest.mark.integration
def test_index_and_search(indexed):
    from common.retrieval.index import check_index, loaded_index

    driver, chunks, fake = indexed
    assert loaded_index(driver)["chunks"] == len(chunks)
    check_index(driver, chunks)
    query = "메일함 배포 뒤 캐시 구매 아이템이 지급되지 않았어요"
    for scheme in C.SCHEMES:
        rows = search(driver, fake, query, scheme=scheme, k=3)
        assert [r["rank"] for r in rows] == [1, 2, 3] and {r["chunk_id"].split(":")[0] for r in rows} == {scheme}
        assert rows[0]["doc_id"] == "PM-01"
        assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    before = fake.reranked
    rows = search(driver, fake, query, scheme="paragraph", k=5, rerank=True)
    assert fake.reranked == before + 1 and len(rows) == 5 and rows[0]["doc_id"] == "PM-01"
    assert ("query", query) in fake.embedded


@pytest.mark.integration
def test_check_index_stops_on_a_different_corpus(indexed):
    from common.retrieval.index import IndexError_, check_index

    driver, chunks, _ = indexed
    with pytest.raises(IndexError_, match="임베딩 모델"):
        check_index(driver, chunks, model="voyage-4-lite")
    with pytest.raises(IndexError_, match="다르다"):
        check_index(driver, chunks[:-1])
    with driver.session() as s:
        s.run("MATCH (m:ChunkIndexMeta) DELETE m").consume()
    with pytest.raises(IndexError_, match="없다"):
        check_index(driver, chunks)


@pytest.mark.integration
def test_search_docs_tool_reports_bad_input(indexed):
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    from mcp_servers.doc_search.server import build_server

    driver, _, fake = indexed
    server = build_server(driver, fake)
    out = asyncio.run(server.call_tool("search_docs", {"query": "로그인이 안 돼요", "k": 2, "scheme": "fixed"}))
    text = str(out)
    assert '"results"' in text and '"usage"' in text  # 조각 목록과 이 호출의 Voyage 사용량을 함께 돌려준다
    with pytest.raises(ToolError, match="조각 방식"):
        asyncio.run(server.call_tool("search_docs", {"query": "로그인", "scheme": "sentence"}))


def test_cache_drops_a_cut_last_line_but_stops_on_a_broken_middle_line(tmp_path):
    path = tmp_path / f"{V.EMBED_MODEL}.jsonl"
    good = json.dumps({"key": "a", "embedding": [1.0]})
    path.write_text(good + "\n" + '{"key": "b", "embe', encoding="utf-8")
    v = V.Voyage(cache_dir=tmp_path)
    assert v._cache == {"a": [1.0]} and path.read_text(encoding="utf-8") == good + "\n"
    path.write_text('{"key": "b", "embe\n' + good + "\n", encoding="utf-8")
    with pytest.raises(V.VoyageError, match="1번째 줄"):
        V.Voyage(cache_dir=tmp_path)


def test_rerank_ties_follow_chunk_id(monkeypatch):
    from common.retrieval import search as Smod

    class Tie(FakeVoyage):
        def rerank(self, query, documents, top_k):
            assert top_k == len(documents)  # 후보 전체의 점수를 받는다
            return [(i, 0.5) for i in range(len(documents))]

    rows = [{"chunk_id": c, "doc_id": "X", "text": c, "score": 0.9} for c in ("f:X:003", "f:X:001", "f:X:002")]

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute_read(self, fn):
            return [dict(r) for r in rows]

    class Driver:
        def session(self):
            return Session()

    out = Smod.search(Driver(), Tie(), "q", scheme="fixed", k=2, rerank=True)
    assert [r["chunk_id"] for r in out] == ["f:X:001", "f:X:002"]
