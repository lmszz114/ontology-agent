"""조각 검색 (설계 4단계 2-5: 조각 방식, 리랭크, 가져오는 개수).

벡터 검색으로 가까운 조각을 고르고, 리랭크를 켜면 더 넓게(POOL개) 가져온 뒤 Voyage rerank-3로 다시 정렬해
k개를 남긴다. 같은 입력이면 같은 순서가 나오도록 점수가 같으면 조각 id 순으로 둔다.
"""

from neo4j import Driver

from common.retrieval.chunking import SCHEMES

POOL = 30  # 리랭크 전에 벡터 검색으로 가져오는 조각 수 (가정값)
TOP_KS = (3, 5, 10)

QUERY = (
    "CYPHER 25 MATCH (c:Chunk) SEARCH c IN (VECTOR INDEX chunk_embedding FOR $vector WHERE c.scheme = $scheme "
    "LIMIT $limit) SCORE AS score RETURN c.id AS chunk_id, c.doc_id AS doc_id, c.text AS text, score "
    "ORDER BY score DESC, chunk_id"
)


class SearchError(ValueError):
    pass


def search(driver: Driver, embedder, query: str, *, scheme: str, k: int, rerank: bool = False) -> list[dict]:
    """조각 k개를 순위대로 돌려준다. 각 조각: chunk_id, doc_id, text, score, rank."""
    if scheme not in SCHEMES:
        raise SearchError(f"정의되지 않은 조각 방식이다: {scheme} (가능: {SCHEMES})")
    if not isinstance(k, int) or not 1 <= k <= POOL:
        raise SearchError(f"가져오는 개수는 1~{POOL}이다: {k}")
    if not query or not query.strip():
        raise SearchError("검색어가 비어 있다")
    (vector,) = embedder.embed([query], "query")
    with driver.session() as session:
        rows = session.execute_read(lambda tx: tx.run(QUERY, vector=vector, scheme=scheme,
                                                      limit=POOL if rerank else k).data())
    if rerank:
        ranked = embedder.rerank(query, [r["text"] for r in rows], k)
        rows = [rows[i] | {"score": score} for i, score in ranked]
    return [r | {"rank": i} for i, r in enumerate(rows[:k], start=1)]


def docs_of(results: list[dict]) -> list[str]:
    """검색 결과에 나온 문서를 처음 나온 순서대로. 문서 단위 지표(Recall@k, MRR)에 쓴다."""
    seen: list[str] = []
    for r in results:
        if r["doc_id"] not in seen:
            seen.append(r["doc_id"])
    return seen
