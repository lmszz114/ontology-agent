"""검색 조각을 임베딩해 Neo4j에 적재한다 (설계 5단계 7장 문서 청크, 임베딩).

실행: python -m common.retrieval.index           # 개발용 DB에 두 조각 방식을 모두 적재
      python -m common.retrieval.index --dry-run # 조각 수와 글자 수만 본다 (Voyage를 부르지 않는다)

조각(Chunk)은 정적, 사건, 지식 추출 계층과 따로 둔 검색용 사본이다. 원본은 시드와 현장 문서 마크다운이다.
적재는 덮어쓰기다. 조각을 모두 지우고 다시 만들며 한 트랜잭션으로 한다. 적재 기록(ChunkIndexMeta)에
조각 목록의 해시와 임베딩 모델을 남겨, 검색할 때 지금 시드와 문서로 만든 색인인지 확인한다.
"""

import argparse
import sys
from datetime import UTC, datetime

from neo4j import Driver

from common.retrieval.chunking import SCHEMES, Chunk, all_chunks, corpus_hash
from common.retrieval.voyage import EMBED_MODEL


class IndexError_(RuntimeError):
    pass


def write_index(driver: Driver, chunks: list[Chunk], embeddings: list[list[float]], model: str = EMBED_MODEL) -> dict:
    if len(chunks) != len(embeddings):
        raise ValueError("조각 수와 임베딩 수가 다르다")
    meta = {"corpus_hash": corpus_hash(chunks), "model": model, "chunks": len(chunks),
            "schemes": sorted({c.scheme for c in chunks}), "built_at": datetime.now(UTC).isoformat()}
    rows = [{"id": c.id, "scheme": c.scheme, "doc_id": c.doc_id, "order": c.order, "text": c.text, "embedding": e}
            for c, e in zip(chunks, embeddings)]

    def work(tx):
        tx.run("MATCH (c:Chunk) DETACH DELETE c").consume()
        tx.run("MATCH (m:ChunkIndexMeta) DELETE m").consume()
        tx.run("UNWIND $rows AS r CREATE (c:Chunk {id: r.id, scheme: r.scheme, doc_id: r.doc_id, order: r.order, "
               "text: r.text}) WITH c, r CALL db.create.setNodeVectorProperty(c, 'embedding', r.embedding)",
               rows=rows).consume()
        tx.run("CREATE (m:ChunkIndexMeta) SET m = $meta", meta=meta).consume()

    with driver.session() as session:
        session.execute_write(work)
    return meta


def loaded_index(driver: Driver) -> dict | None:
    with driver.session() as session:
        rec = session.run("MATCH (m:ChunkIndexMeta) RETURN properties(m) AS m").single()
    return rec["m"] if rec else None


def check_index(driver: Driver, chunks: list[Chunk], model: str = EMBED_MODEL) -> dict:
    """적재된 색인이 지금 조각 목록과 같은지 본다. 다르면 멈춘다 (시드나 문서가 바뀌었는데 다시 적재하지 않음)."""
    meta = loaded_index(driver)
    if meta is None:
        raise IndexError_("검색 색인이 없다. python -m common.retrieval.index로 적재한다")
    if meta["corpus_hash"] != corpus_hash(chunks):
        raise IndexError_("검색 색인이 지금 시드, 문서와 다르다. 다시 적재한다")
    if meta["model"] != model:
        # 다른 모델로 만든 색인에 지금 모델의 검색어 벡터를 대면 순위가 틀린다 (PR #35 리뷰)
        raise IndexError_(f"검색 색인의 임베딩 모델({meta['model']})이 지금 모델({model})과 다르다. 다시 적재한다")
    return meta


def build(driver: Driver, embedder, seed: dict, plan: list[dict], schemes: tuple[str, ...] = SCHEMES) -> dict:
    chunks = all_chunks(seed, plan, schemes)
    embeddings = embedder.embed([c.text for c in chunks], "document")
    return write_index(driver, chunks, embeddings)


def main(argv=None) -> int:
    from eval.corpus.check import load_plan
    from ontology.seed import load_seed

    parser = argparse.ArgumentParser(description="검색 조각 임베딩과 적재")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--org", default="org-a")
    args = parser.parse_args(argv)
    seed, plan = load_seed(args.org), load_plan()
    chunks = all_chunks(seed, plan)
    for scheme in SCHEMES:
        part = [c for c in chunks if c.scheme == scheme]
        print(f"{scheme}: 조각 {len(part)}개, 글자 {sum(len(c.text) for c in part)}자, "
              f"가장 긴 조각 {max(len(c.text) for c in part)}자", file=sys.stderr)
    if args.dry_run:
        return 0
    from common.retrieval.voyage import Voyage
    from ontology.apply_schema import apply_schema
    from ontology.db import get_driver

    voyage = Voyage()
    with get_driver() as driver:
        apply_schema(driver)
        meta = write_index(driver, chunks, voyage.embed([c.text for c in chunks], "document"))
    print(f"적재 {meta['chunks']}개, 모델 {meta['model']}, 임베딩 토큰 {voyage.usage.embed_tokens}"
          f" (캐시에 있던 것은 세지 않는다)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
