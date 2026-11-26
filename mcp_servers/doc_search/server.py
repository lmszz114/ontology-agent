"""문서 검색 MCP 서버 (설계 5단계 6장 search_docs).

실행: python -m mcp_servers.doc_search.server

기준 지식 K와 현장 문서를 조각 단위로 검색한다. 읽기 전용이다. 색인이 지금 시드, 문서와 다르면 띄우지 않는다.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from neo4j import Driver

from common.retrieval.search import SearchError, search


def build_server(driver: Driver, embedder) -> MCPServer:
    server = MCPServer("doc-search", instructions="사내 기준 지식과 현장 문서(회고, 런북, 대화, 회의록)를 검색한다. 읽기 전용이다.")

    @server.tool()
    def search_docs(query: str, k: int = 5, scheme: str = "paragraph", rerank: bool = False) -> list[dict]:
        """검색어와 가까운 문서 조각을 순위대로 돌려준다.

        k는 가져오는 조각 수(1~30), scheme은 조각 방식(paragraph, fixed), rerank는 리랭크 사용 여부다.
        각 조각: chunk_id, doc_id(문서 id, 기준 지식은 K), text, score, rank.
        """
        try:
            return search(driver, embedder, query, scheme=scheme, k=k, rerank=rerank)
        except SearchError as exc:
            raise ToolError(str(exc)) from exc

    return server


def main() -> None:
    import os

    from common.retrieval.chunking import all_chunks
    from common.retrieval.index import check_index
    from common.retrieval.voyage import Voyage
    from eval.corpus.check import load_plan
    from ontology.db import get_driver
    from ontology.seed import load_seed

    driver = get_driver()
    check_index(driver, all_chunks(load_seed(os.environ.get("ORG", "org-a")), load_plan()))
    build_server(driver, Voyage()).run()


if __name__ == "__main__":
    main()
