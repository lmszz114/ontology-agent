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
from datetime import datetime
from pathlib import Path

from mcp import Client, StdioServerParameters

from agents.diagnosis.graph import KST, Deps, build_graph, initial_state
from agents.diagnosis.prompts import PROMPT_VERSION
from common.knowledge import GraphKnowledge, KnowledgeProvider
from common.models import LLM
from common.models.config import USAGE_FIELDS
from common.tracing import Tracer
from common.writers import events
from mcp_servers.ontology_query import queries

ROOT = Path(__file__).parent.parent.parent
# MCP 서버 프로세스에 넘기는 환경 변수. API 키 같은 비밀값은 넘기지 않는다
PASS_ENV = ("PATH", "HOME", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "LANG")

# 비교 구성별 지식 공급자 (4단계 2-2, 5단계 4-2). 에이전트 코드는 하나이고 지식 공급자만 바꿔 끼운다.
# 받는 값은 온톨로지 조회 MCP 클라이언트와 실행 기록기다. 구성 A는 M3 다음 PR에서 더한다
KNOWLEDGE: dict[str, Callable[[Client, Tracer], KnowledgeProvider]] = {
    "C": lambda onto, tracer: GraphKnowledge(onto, tracer, name="C"),
}


def _server(module: str, extra: dict[str, str]) -> StdioServerParameters:
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env |= {"PYTHONPATH": str(ROOT)} | extra
    return StdioServerParameters(command=sys.executable, args=["-m", module], env=env, cwd=str(ROOT))


def prepare_event_layer(driver, scenario: dict) -> None:
    """사건 계층을 비우고 시나리오의 변경 이력을 기록한다."""
    events.reset_event_layer(driver)
    for index, change in enumerate(scenario["environment"].get("changes") or []):
        events.record_change(
            driver, change_id=f"CG-{index}", kind=change["kind"], system_id=change["system"],
            changed_at=datetime.fromisoformat(change["at"]),
        )


async def run_scenario(scenario: dict, scenario_path: Path, *, config: str, org: str, tracer: Tracer, llm: LLM,
                       driver) -> dict:
    if config not in KNOWLEDGE:
        raise ValueError(f"등록되지 않은 구성이다: {config} (구성: {sorted(KNOWLEDGE)})")
    prepare_event_layer(driver, scenario)
    onto_params = _server("mcp_servers.ontology_query.server", {"ORG": org})
    mock_params = _server("mcp_servers.mock_tools.server", {"ORG": org, "MOCK_ENV_FILE": str(scenario_path.resolve())})
    async with Client(onto_params) as onto, Client(mock_params) as mock:
        listed = await mock.list_tools()
        mock_tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                      for t in listed.tools]
        deps = Deps(
            driver=driver, llm=llm, tracer=tracer, knowledge=KNOWLEDGE[config](onto, tracer),
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
        },
        "outputs": state.get("outputs") or [],
        "unclassified": state.get("unclassified") or [],
    }
