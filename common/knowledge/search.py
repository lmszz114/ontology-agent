"""구성 B 지식 공급자: 문서 검색 (설계 4단계 2-2, 5단계 4-2).

사건마다 그 사건의 제보 원문으로 문서 검색(search_docs MCP)을 한 번 하고, 검색해 온 조각만 근거로
구성 A와 같은 질문(원인 순위, 판정 근거, 담당 팀, 절차)에 LLM이 답한다. 규칙 설명은 구성 A와 같다.
답은 정의된 ID로 내야 하므로 ID와 이름 목록을 함께 준다. 관계는 넣지 않는다.

검색 설정(조각 방식, 리랭크, 가져오는 개수)은 dev 검색 비교로 고른다 (4단계 2-5, M5 PR 2).
"""

from dataclasses import dataclass
from typing import Any

from mcp import Client

from common.knowledge import text_prompts as P
from common.knowledge.base import KnowledgeUnavailable
from common.knowledge.text import TextKnowledge
from common.tool_calls import ToolCallError, call_tool_data
from common.tracing import Tracer

SERVER = "doc-search"


@dataclass(frozen=True)
class SearchSettings:
    scheme: str
    rerank: bool
    k: int

    def as_dict(self) -> dict:
        return {"scheme": self.scheme, "rerank": self.rerank, "k": self.k}


# dev 검색 비교(eval/results/2026-11-26-m5-retrieval-dev)를 보고 사용자가 골랐다 (2026-11-26).
# 12가지 중 Recall@10 0.6321, MRR 0.8214로 가장 높고 옛날 문서가 들어오지 않았다. 사건 14개라 작은 차이는 우연일 수 있다
B_SETTINGS: SearchSettings | None = SearchSettings(scheme="fixed", rerank=True, k=10)


def id_listing(seed: dict) -> str:
    """ID와 이름만. 별칭, 관계, 속성은 넣지 않는다."""
    sections = [("증상", "symptoms"), ("원인", "causes"), ("판정 근거", "evidence_checks"), ("팀", "teams"),
                ("시스템", "systems"), ("절차", "procedures"), ("조치", "actions"), ("심각도", "severities")]
    lines = ["# ID 목록"]
    for title, key in sections:
        lines.append(f"## {title}")
        lines += [f"- {item['id']}: {item.get('name', item['id'])}" for item in seed[key]]
    return "\n".join(lines)


class SearchKnowledge(TextKnowledge):
    def __init__(self, *, docs: Client, tracer: Tracer, settings: SearchSettings, seed: dict, **kwargs):
        super().__init__(seed=seed, name="B", **kwargs)
        self.docs = docs
        self.tracer = tracer
        self.settings = settings
        self.prompt_version = P.PROMPT_VERSION_B
        self.retry = P.RETRY_B
        self.ids_text = id_listing(seed)
        self.system = None  # 사건마다 검색 결과로 만든다. 검색 전에 질문이 오면 막는다
        self.searches: list[dict] = []

    async def begin_incident(self, incident_id: str, reports: list[str]) -> dict[str, Any]:
        query = "\n".join(reports)  # 검색 채점(eval/grader/retrieval)과 같은 검색어 (M5 사용자 결정)
        try:
            rows = await call_tool_data(self.docs, self.tracer, server=SERVER, name="search_docs",
                                        args={"query": query} | self.settings.as_dict())
        except ToolCallError as exc:
            self.system = None
            self.errors.append(f"5.search.{incident_id}: {exc}")
            raise KnowledgeUnavailable(f"문서 검색 실패: {exc}") from exc
        chunks = "\n\n".join(f"[{r['rank']}] 문서 {r['doc_id']} ({r['chunk_id']})\n{r['text']}" for r in rows)
        # 검색 결과가 사건마다 달라서 캐시는 한 사건의 질문들 사이에서만 쓰인다
        self.system = [{"type": "text", "cache_control": {"type": "ephemeral"},
                        "text": P.SYSTEM_B + "\n" + self.ids_text + "\n\n# 검색 결과\n\n" + (chunks or "(없음)")}]
        self._memo.clear()  # 앞 사건의 검색 결과로 만든 답을 다시 쓰지 않는다
        docs: list[str] = []
        for r in rows:
            if r["doc_id"] not in docs:
                docs.append(r["doc_id"])
        record = {"incident_id": incident_id, "settings": self.settings.as_dict(), "docs": docs,
                  "chunks": [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "score": r["score"]} for r in rows]}
        self.searches.append(record)
        return record

    def _ask(self, *args, **kwargs):
        if self.system is None:
            # 검색 전이거나 이 사건의 검색이 실패했다. 근거 없이 답하지 않는다
            raise KnowledgeUnavailable("이 사건의 검색 결과가 없다")
        return super()._ask(*args, **kwargs)
