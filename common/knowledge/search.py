"""구성 B 지식 공급자: 문서 검색 (설계 4단계 2-2, 5단계 4-2).

사건마다 그 사건의 제보 원문으로 문서 검색(search_docs MCP)을 하고, 검색해 온 조각만 근거로
구성 A와 같은 질문(원인 순위, 판정 근거, 담당 팀, 절차)에 LLM이 답한다. 규칙 설명은 구성 A와 같다.
판정 근거, 담당 팀, 절차를 물을 때는 코드가 정한 검색어(원인이나 절차의 ID와 이름 + 묻는 것)로 한 번 더 검색해
그 질문의 조각을 더한다 (M5 사용자 결정). 제보 원문 검색 한 번만으로는 원인의 확정 조건이 거의 들어오지 않았다
(dev 첫 실행 2026-11-26, 사건 42개 중 확정 18개).
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
        self._base: list[dict] = []
        self._record: dict = {}
        self.voyage_usage = {"calls": 0, "embed_tokens": 0, "rerank_tokens": 0}

    async def _search(self, query: str, where: str) -> list[dict]:
        try:
            out = await call_tool_data(self.docs, self.tracer, server=SERVER, name="search_docs",
                                       args={"query": query} | self.settings.as_dict())
        except ToolCallError as exc:
            self.errors.append(f"{where}: {exc}")
            raise KnowledgeUnavailable(f"문서 검색 실패: {exc}") from exc
        for key in self.voyage_usage:  # 검색마다 쓴 Voyage 토큰을 실행 기록에 남긴다 (PR #35 리뷰)
            self.voyage_usage[key] += out["usage"][key]
        return out["results"]

    @staticmethod
    def _chunks_text(rows: list[dict], start: int) -> str:
        return "\n\n".join(f"[{i}] 문서 {r['doc_id']} ({r['chunk_id']})\n{r['text']}" for i, r in enumerate(rows, start=start))

    def _set_system(self, base: list[dict], extra: list[dict] = ()) -> None:
        """시스템 프롬프트를 바뀌는 주기대로 나눠 캐시한다 (M5 점검).

        규칙과 ID 목록은 실행 내내, 사건 검색 결과는 그 사건 안에서 같다. 질문 검색 결과는 질문마다 다르므로
        캐시 표시 뒤에 둔다. 한 블록에 모두 넣고 캐시하면 질문마다 새 캐시를 쓰기만 하고 읽지 못한다
        (dev B v2 3회: 캐시 쓰기 3,928,586토큰, 읽기 52,377토큰)
        """
        system = [{"type": "text", "text": P.SYSTEM_B + "\n" + self.ids_text, "cache_control": {"type": "ephemeral"}},
                  {"type": "text", "text": "# 검색 결과\n\n" + (self._chunks_text(base, 1) or "(없음)"),
                   "cache_control": {"type": "ephemeral"}}]
        if extra:
            system.append({"type": "text", "text": self._chunks_text(list(extra), len(base) + 1)})
        self.system = system

    async def begin_incident(self, incident_id: str, reports: list[str]) -> dict[str, Any]:
        query = "\n".join(reports)  # 검색 채점(eval/grader/retrieval)과 같은 검색어 (M5 사용자 결정)
        self.system, self._base = None, []
        self._memo.clear()  # 검색보다 먼저 비운다. 검색이 실패해도 앞 사건의 답이 남지 않는다 (PR #35 리뷰)
        rows = await self._search(query, f"5.search.{incident_id}")
        self._base = rows
        self._set_system(rows)
        docs: list[str] = []
        for r in rows:
            if r["doc_id"] not in docs:
                docs.append(r["doc_id"])
        self._record = {"incident_id": incident_id, "settings": self.settings.as_dict(), "docs": docs,
                        "chunks": [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "score": r["score"]} for r in rows],
                        "extra": []}
        self.searches.append(self._record)
        return self._record

    async def _with_question_search(self, key: tuple, query: str):
        """질문마다 코드가 정한 검색어로 한 번 더 검색해 사건 검색 결과에 더한다. 이미 답한 질문은 검색하지 않는다."""
        if self.system is None or key in self._memo:
            return
        extra = await self._search(query, f"{key[0]}.search")
        seen = {r["chunk_id"] for r in self._base}
        self._set_system(self._base, [r for r in extra if r["chunk_id"] not in seen])
        for r in extra:  # 이 사건에서 검색해 온 문서 목록(retrieved_docs)에 질문 검색의 문서도 더한다 (PR #35 리뷰)
            if r["doc_id"] not in self._record["docs"]:
                self._record["docs"].append(r["doc_id"])
        self._record["extra"].append({"question": list(key), "query": query,
                                      "chunks": [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "score": r["score"]}
                                                 for r in extra]})

    async def get_checks(self, cause_id: str) -> list[dict[str, Any]]:
        name = self.ids.causes.get(cause_id, "")
        await self._with_question_search(("checks", cause_id), f"{cause_id} {name} 원인을 확인하는 판정 근거와 확정 조건")
        return await super().get_checks(cause_id)

    async def owner_team(self, cause_id: str) -> dict[str, Any]:
        name = self.ids.causes.get(cause_id, "")
        await self._with_question_search(("owner", cause_id), f"{cause_id} {name} 원인의 위치 시스템과 담당 팀")
        return await super().owner_team(cause_id)

    async def cause_procedures(self, cause_id: str, severity_id: str) -> dict[str, Any]:
        name = self.ids.causes.get(cause_id, "")
        await self._with_question_search(("procedures", cause_id, severity_id),
                                         f"{cause_id} {name} 원인 확정 뒤 여는 대응 절차와 조치, {severity_id} 등급 절차")
        return await super().cause_procedures(cause_id, severity_id)

    async def procedure(self, procedure_id: str) -> dict[str, Any]:
        name = self.ids.procedures.get(procedure_id, "")
        await self._with_question_search(("procedure", procedure_id), f"{procedure_id} {name} 절차의 조치와 단계")
        return await super().procedure(procedure_id)

    def _ask(self, *args, **kwargs):
        if self.system is None:
            # 검색 전이거나 이 사건의 검색이 실패했다. 근거 없이 답하지 않는다
            raise KnowledgeUnavailable("이 사건의 검색 결과가 없다")
        return super()._ask(*args, **kwargs)
