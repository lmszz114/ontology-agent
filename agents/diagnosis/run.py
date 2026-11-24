"""진단 에이전트 실행 (평가 모드, 설계 5단계 8-1).

실행: python -m agents.diagnosis.run --scenario eval/scenarios/dev/SC-S1-01.yaml [--out 결과폴더]
      python -m agents.diagnosis.run --split dev --out 결과폴더

평가 모드에서는 시나리오마다
1. 개발용 Neo4j의 사건 계층을 비우고 시나리오의 변경 이력을 기록한다 (5단계 7장, 원인 순위의 변경 가중 근거)
2. 온톨로지 조회와 모의 도구 MCP 서버를 별도 프로세스로 띄운다. 모의 도구는 시나리오의 환경을 읽는다
3. 시나리오의 제보를 그래프에 넣고 구조화 출력을 받는다
4. 결과와 실행 정보(모델, 프롬프트 버전, 커밋, 토큰, 비용, trace ID)를 파일로 남긴다
고위험 조치는 승인 대기 상태만 기록한다. 실제 실행은 없다 (모두 시뮬레이션).
채점은 하지 않는다. 채점기는 M3에서 만든다.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from mcp import Client, StdioServerParameters

from agents.diagnosis.graph import KST, Deps, build_graph, initial_state
from agents.diagnosis.prompts import PROMPT_VERSION
from common.knowledge import GraphKnowledge
from common.models import LLM, load_model_config, make_anthropic_client
from common.models.config import USAGE_FIELDS
from common.tracing import Tracer
from common.writers import events
from eval.validate import load_scenarios
from mcp_servers.ontology_query import queries
from ontology.db import get_driver

ROOT = Path(__file__).parent.parent.parent
# MCP 서버 프로세스에 넘기는 환경 변수. API 키 같은 비밀값은 넘기지 않는다
PASS_ENV = ("PATH", "HOME", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "LANG")


def _server(module: str, extra: dict[str, str]) -> StdioServerParameters:
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env |= {"PYTHONPATH": str(ROOT)} | extra
    return StdioServerParameters(command=sys.executable, args=["-m", module], env=env, cwd=str(ROOT))


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_event_layer(driver, scenario: dict) -> None:
    """사건 계층을 비우고 시나리오의 변경 이력을 기록한다."""
    events.reset_event_layer(driver)
    for index, change in enumerate(scenario["environment"].get("changes") or []):
        events.record_change(
            driver, change_id=f"CG-{index}", kind=change["kind"], system_id=change["system"],
            changed_at=datetime.fromisoformat(change["at"]),
        )


async def run_scenario(scenario: dict, scenario_path: Path, *, org: str, tracer: Tracer, llm: LLM, driver) -> dict:
    prepare_event_layer(driver, scenario)
    onto_params = _server("mcp_servers.ontology_query.server", {"ORG": org})
    mock_params = _server("mcp_servers.mock_tools.server", {"ORG": org, "MOCK_ENV_FILE": str(scenario_path.resolve())})
    async with Client(onto_params) as onto, Client(mock_params) as mock:
        listed = await mock.list_tools()
        mock_tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                      for t in listed.tools]
        deps = Deps(
            driver=driver, llm=llm, tracer=tracer, knowledge=GraphKnowledge(onto, tracer, name="C"),
            mock_client=mock, mock_tools=mock_tools, tool_checks=queries.get_evidence_tools(driver),
            symptoms=queries.get_symptom_types(driver), systems=queries.get_systems(driver),
        )
        graph = build_graph(deps)
        started = datetime.now(KST)
        with tracer.run(
            "diagnosis", session_id=scenario["id"], tags=["config:C", f"split:{scenario['split']}", "mode:eval"],
            metadata={"scenario": scenario["id"], "config": "C", "org": org, "prompt_version": PROMPT_VERSION},
            input={"reports": scenario["reports"]},
        ) as run:
            trace_id = tracer.current_trace_id()
            state = await graph.ainvoke(initial_state(scenario["reports"]), config={"recursion_limit": 200})
            run.update(output={"outputs": state.get("outputs"), "unclassified": state.get("unclassified")})
        finished = datetime.now(KST)

    usage = {k: sum(r.usage[k] for r in deps.llm_results) for k in USAGE_FIELDS}
    return {
        "run": {
            "scenario": scenario["id"], "split": scenario["split"], "config": "C", "org": org, "mode": "eval",
            "started_at": started.isoformat(), "finished_at": finished.isoformat(),
            "seconds": round((finished - started).total_seconds(), 1),
            "models": sorted({r.model for r in deps.llm_results}), "prompt_version": PROMPT_VERSION,
            "git_commit": _git_commit(), "trace_id": trace_id,
            "llm_calls": len(deps.llm_results), "usage": usage,
            "cost_usd": sum(r.cost["total"] for r in deps.llm_results),
        },
        "outputs": state.get("outputs") or [],
        "unclassified": state.get("unclassified") or [],
    }


async def main_async(args) -> int:
    if args.scenario:
        paths = [Path(p) for p in args.scenario]
    else:
        paths = sorted((ROOT / "eval" / "scenarios" / args.split).glob("*.yaml"))
        load_scenarios(args.split)  # 형식이 틀린 파일이 있으면 여기서 멈춘다
    tracer = Tracer.from_env()
    llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    with get_driver() as driver:
        try:
            for path in paths:
                scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
                result = await run_scenario(scenario, path, org=args.org, tracer=tracer, llm=llm, driver=driver)
                text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
                if out:
                    (out / f"{scenario['id']}.json").write_text(text + "\n", encoding="utf-8")
                    print(f"{scenario['id']}: {result['run']['llm_calls']} calls, ${result['run']['cost_usd']:.5f}")
                else:
                    print(text)
        finally:
            tracer.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="진단 에이전트 평가 모드 실행 (구성 C)")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--scenario", nargs="+", help="시나리오 파일")
    target.add_argument("--split", choices=["dev", "test"], help="이 세트의 시나리오 전체")
    parser.add_argument("--org", default="org-a")
    parser.add_argument("--out", help="결과 파일을 둘 폴더. 없으면 화면에 출력")
    args = parser.parse_args()
    if args.split == "test":
        # test 세트는 최종 실행(M8)에만 쓴다 (4단계 4-2). 실수로 돌리지 않게 막는다
        parser.error("test 세트는 최종 실행에서만 쓴다")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
