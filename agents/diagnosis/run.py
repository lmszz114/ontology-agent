"""진단 에이전트로 시나리오 한 건을 돌린다 (평가 모드, 설계 5단계 8-1).

실행 명령은 eval/runner다 (python -m eval.runner). 이 모듈은 시나리오 한 건의 실행만 맡는다.
구성 선택, 반복, 결과 폴더, 커밋과 단가 기록은 실행기가 한다.

시나리오마다
1. 개발용 Neo4j의 사건 계층을 비우고 시나리오의 변경 이력을 기록한다 (5단계 7장, 원인 순위의 변경 가중 근거)
2. 온톨로지 조회와 모의 도구 MCP 서버를 별도 프로세스로 띄운다. 모의 도구는 시나리오의 환경을 읽는다
3. 구성에 맞는 지식 공급자를 붙여 시나리오의 제보를 그래프에 넣고 구조화 출력을 받는다
4. 결과와 실행 정보(모델, 프롬프트 버전, 토큰, 비용, trace ID)를 돌려준다
고위험 조치는 승인 대기 상태만 기록한다. 실제 실행은 없다 (모두 시뮬레이션).
"""

import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from agents.diagnosis.graph import KST, Deps, build_graph, initial_state
from agents.diagnosis.prompts import PROMPT_VERSION
from common.knowledge import GraphKnowledge, KnowledgeProvider
from common.knowledge.search import SearchKnowledge
from common.knowledge.text import TextKnowledge
from common.models import LLM, LLMResult
from common.models.config import USAGE_FIELDS
from common.tracing import Tracer
from common.writers import events
from mcp_servers.ontology_query import queries
from ontology.seed import load_seed

ROOT = Path(__file__).parent.parent.parent
# 모든 MCP 서버 프로세스에 넘기는 환경 변수. 비밀값 중 Neo4j 비밀번호(NEO4J_PASSWORD)가 들어 있다(Neo4j를 쓰지 않는
# 모의 도구 서버도 받는다). API 키는 넘기지 않고, Voyage 키만 문서 검색 서버에 넘긴다(아래 SEARCH_ENV) (PR #36 리뷰)
PASS_ENV = ("PATH", "HOME", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "LANG")
# 문서 검색 서버는 Voyage를 부른다. 키는 보통 프록시가 붙이므로(네트워크 시크릿) 프록시와 인증서 설정을 넘긴다.
# VOYAGE_API_KEY는 키를 환경 변수로 둔 곳(로컬 실행)에서만 있고, 그때는 이 서버 하나에만 넘긴다 (M5 점검)
SEARCH_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy", "SSL_CERT_FILE",
              "REQUESTS_CA_BUNDLE", "VOYAGE_API_KEY")

@dataclass(frozen=True)
class KnowledgeContext:
    """지식 공급자를 만들 때 쓰는 값. 구성마다 필요한 것만 꺼내 쓴다."""

    onto: Client                 # 온톨로지 조회 MCP 클라이언트 (구성 C)
    tracer: Tracer
    llm: LLM                     # 구성 A가 텍스트를 읽고 답할 때 부른다
    driver: Any                  # 사건 데이터(변경 이력)를 읽는다
    org: str
    results: list[LLMResult]     # 진단 실행의 LLM 호출 목록. 공급자의 호출도 비용과 토큰에 함께 센다
    docs: Client | None = None   # 문서 검색 MCP 클라이언트 (구성 B)


# 비교 구성별 지식 공급자 (4단계 2-2, 5단계 4-2). 에이전트 코드는 하나이고 지식 공급자만 바꿔 끼운다
KNOWLEDGE: dict[str, Callable[[KnowledgeContext], KnowledgeProvider]] = {
    "A": lambda ctx: TextKnowledge(llm=ctx.llm, seed=load_seed(ctx.org), results=ctx.results,
                                   changes=lambda: events.list_changes(ctx.driver), name="A"),
    # 검색 설정은 dev 검색 비교로 고른 값이다 (common/knowledge/search.B_SETTINGS, M5 PR 2)
    "B": lambda ctx: SearchKnowledge(docs=ctx.docs, tracer=ctx.tracer, settings=_b_settings(), llm=ctx.llm,
                                     seed=load_seed(ctx.org), results=ctx.results,
                                     changes=lambda: events.list_changes(ctx.driver)),
    "C": lambda ctx: GraphKnowledge(ctx.onto, ctx.tracer, name="C"),
    # C와 같은 공급자다. 실행기가 추출 후보로 만든 그래프 DB(7689)에 연결해 준다 (ontology/cext, M4 PR 6)
    "C-ext": lambda ctx: GraphKnowledge(ctx.onto, ctx.tracer, name="C-ext"),
}
# 개발용 그래프가 아니라 추출 그래프 DB에서 진단하는 구성
EXT_CONFIGS = {"C-ext"}
# 문서 검색 서버를 띄우는 구성
SEARCH_CONFIGS = {"B"}


def _b_settings():
    from common.knowledge.search import B_SETTINGS

    if B_SETTINGS is None:
        raise ValueError("구성 B의 검색 설정을 아직 고르지 않았다 (common/knowledge/search.B_SETTINGS, M5 PR 2)")
    return B_SETTINGS


def _server(module: str, extra: dict[str, str]) -> StdioServerParameters:
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env |= {"PYTHONPATH": str(ROOT)} | extra
    return StdioServerParameters(command=sys.executable, args=["-m", module], env=env, cwd=str(ROOT))


@asynccontextmanager
async def _optional_client(params: StdioServerParameters | None):
    if params is None:
        yield None
        return
    async with Client(params) as client:
        yield client


def prepare_event_layer(driver, scenario: dict) -> None:
    """사건 계층을 비우고 시나리오의 변경 이력을 기록한다."""
    events.reset_event_layer(driver)
    for index, change in enumerate(scenario["environment"].get("changes") or []):
        events.record_change(
            driver, change_id=f"CG-{index}", kind=change["kind"], system_id=change["system"],
            changed_at=datetime.fromisoformat(change["at"]),
        )


async def run_scenario(scenario: dict, scenario_path: Path, *, config: str, org: str, tracer: Tracer, llm: LLM,
                       driver, neo4j_uri: str | None = None) -> dict:
    """neo4j_uri를 주면 온톨로지 조회 서버가 그 DB를 읽는다. driver도 같은 DB에 연결돼 있어야 한다 (C-ext)."""
    if config not in KNOWLEDGE:
        raise ValueError(f"등록되지 않은 구성이다: {config} (구성: {sorted(KNOWLEDGE)})")
    prepare_event_layer(driver, scenario)
    onto_params = _server("mcp_servers.ontology_query.server",
                          {"ORG": org} | ({"NEO4J_URI": neo4j_uri} if neo4j_uri else {}))
    mock_params = _server("mcp_servers.mock_tools.server", {"ORG": org, "MOCK_ENV_FILE": str(scenario_path.resolve())})
    docs_params = (_server("mcp_servers.doc_search.server",
                           {"ORG": org} | {k: os.environ[k] for k in SEARCH_ENV if k in os.environ})
                   if config in SEARCH_CONFIGS else None)
    async with Client(onto_params) as onto, Client(mock_params) as mock, _optional_client(docs_params) as docs:
        listed = await mock.list_tools()
        mock_tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                      for t in listed.tools]
        results: list[LLMResult] = []
        knowledge = KNOWLEDGE[config](KnowledgeContext(onto=onto, tracer=tracer, llm=llm, driver=driver, org=org,
                                                       results=results, docs=docs))
        deps = Deps(
            driver=driver, llm=llm, tracer=tracer, knowledge=knowledge, llm_results=results,
            mock_client=mock, mock_tools=mock_tools, tool_checks=queries.get_evidence_tools(driver),
            symptoms=queries.get_symptom_types(driver), systems=queries.get_systems(driver),
        )
        graph = build_graph(deps)
        started = datetime.now(KST)
        with tracer.run(
            "diagnosis", session_id=scenario["id"], tags=[f"config:{config}", f"split:{scenario['split']}", "mode:eval"],
            metadata={"scenario": scenario["id"], "config": config, "org": org, "prompt_version": PROMPT_VERSION},
            input={"reports": scenario["reports"]},
        ) as run:
            trace_id = tracer.current_trace_id()
            state = await graph.ainvoke(initial_state(scenario["reports"]), config={"recursion_limit": 200})
            run.update(output={"outputs": state.get("outputs"), "unclassified": state.get("unclassified")})
        finished = datetime.now(KST)

    usage = {k: sum(r.usage[k] for r in deps.llm_results) for k in USAGE_FIELDS}
    return {
        "run": {
            "scenario": scenario["id"], "split": scenario["split"], "config": config, "org": org, "mode": "eval",
            "started_at": started.isoformat(), "finished_at": finished.isoformat(),
            "seconds": round((finished - started).total_seconds(), 1),
            "models": sorted({r.model for r in deps.llm_results}), "prompt_version": PROMPT_VERSION,
            "trace_id": trace_id,
            "llm_calls": len(deps.llm_results), "usage": usage,
            "cost_usd": sum(r.cost["total"] for r in deps.llm_results),
            # 지식 공급자의 답이 검증에 두 번 실패한 질문 (구성 A). 그래프 공급자는 없다
            "knowledge_errors": list(getattr(knowledge, "errors", [])),
            "knowledge_prompt_version": getattr(knowledge, "prompt_version", None),
            # 구성 B: 검색 설정. 색인 확인(조각 목록 해시)은 실행기가 남긴다
            "search_settings": knowledge.settings.as_dict() if config in SEARCH_CONFIGS else None,
            # 구성 B: 검색에 쓴 Voyage 토큰. 무료 토큰 안이라 비용(cost_usd)에는 더하지 않는다 (5단계 11-2)
            "voyage_usage": knowledge.voyage_usage if config in SEARCH_CONFIGS else None,
        },
        "outputs": state.get("outputs") or [],
        "unclassified": state.get("unclassified") or [],
    }
