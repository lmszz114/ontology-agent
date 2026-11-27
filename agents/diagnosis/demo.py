"""데모 모드 실행 (설계 5단계 8-1, M7 PR 1).

실행: python -m agents.diagnosis.demo --scenario eval/scenarios/dev/SC-S1-01.yaml [--org org-a] [--config C]

평가 모드와 같은 에이전트 코드와 배포 패키지로 돈다. 다른 점은 승인이다. 고위험 조치에서 실행을 멈추고(LangGraph
interrupt) 승인권자의 승인, 반려를 받아 이어간다. 반려되면 같은 원인의 다른 해결 절차를 다시 고르고, 대안이 없으면
사람에게 넘긴다 (3단계 R6). 멈춘 상태는 SQLite 파일(.cache/demo/checkpoints.sqlite)에 남는다.

승인은 터미널(기본) 또는 슬랙 승인 버튼(--approve slack, apps/slack_app, M7 PR 2)으로 받는다. 둘 다 같은 그래프에
승인 판단 목록을 넘긴다. 슬랙은 SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_APPROVAL_CHANNEL 환경 변수가 필요하다.
JSM 접수(M7 PR 3)도 같은 그래프를 다른 입력으로 쓴다.
지금은 모의 도구가 시나리오 파일의 environment로 답한다. 데모용 가상 데이터는 M7 PR 3에서 만든다.
LLM을 부르므로 비용이 든다. test 세트는 돌릴 수 없다 (4단계 4-2).
"""

import argparse
import asyncio
import inspect
import json
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

import yaml
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from agents.diagnosis.graph import KST, build_graph, initial_state
from agents.diagnosis.prompts import PROMPT_VERSION
from apps.slack_app.approver import SlackConfigError, read_settings, slack_approver
from agents.diagnosis.run import KNOWLEDGE, agent_deps, prepare_event_layer, prompt_uses
from common.models import LLM, load_model_config, make_anthropic_client
from common.packages import PackageError, load_package
from common.tracing import Tracer
from eval.labels import scenario_org
from ontology.db import get_driver

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / ".cache" / "demo" / "checkpoints.sqlite"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="진단 에이전트 데모 모드 (터미널 승인)")
    p.add_argument("--scenario", required=True, help="제보와 environment가 있는 dev 시나리오 파일")
    p.add_argument("--org", default="org-a")
    p.add_argument("--config", default="C", choices=sorted(KNOWLEDGE))
    p.add_argument("--approve", default="terminal", choices=["terminal", "slack"], help="승인을 받는 곳")
    return p.parse_args(argv)


def ask_terminal(payload: dict) -> list[dict]:
    """멈춘 조치마다 아직 승인하지 않은 승인권자 팀의 판단을 터미널에서 받는다."""
    for error in payload.get("errors") or []:
        print(f"  (앞 판단을 처리하지 못했다: {error})")
    decisions = []
    for run in payload["pending"]:
        for team in run["required_approver_teams"]:
            if team in (run.get("approved_by_teams") or []):
                continue
            answer = ""
            while answer not in ("a", "r"):
                answer = input(f"  {run['action_id']} ({run['run_id']}) 승인권자 {team}: 승인 a / 반려 r > ").strip().lower()
            decisions.append({"run_id": run["run_id"], "team_id": team,
                              "decision": "approve" if answer == "a" else "reject"})
    return decisions


def load_names(driver) -> dict:
    """승인 메시지에 쓸 이름 (ID -> 이름). 팀, 조치, 원인, 절차, 증상."""
    records = driver.execute_query(
        "MATCH (n) WHERE n:Team OR n:Action OR n:Cause OR n:Procedure OR n:SymptomType "
        "RETURN n.id AS id, n.name AS name").records
    return {r["id"]: r["name"] for r in records if r["id"] and r["name"]}


async def run_demo(scenario: dict, scenario_path: Path, *, org: str, config: str, decide=None,
                   approve: str = "terminal", checkpoints: Path = CHECKPOINTS) -> dict:
    """decide(멈춘 내용) -> 승인 판단 목록. 주지 않으면 approve에 따라 터미널이나 슬랙에서 받는다."""
    load_package(org, uses=prompt_uses(config))
    if decide is None and approve == "slack":
        # 연결을 열기 전에 본다. 열린 뒤에 나면 MCP 연결의 작업 묶음에 싸여 이유가 가려진다
        read_settings()
    tracer = Tracer.from_env()
    llm = LLM(load_model_config(org), tracer, make_anthropic_client())
    thread = f"{scenario['id']}-{datetime.now(KST):%Y%m%d-%H%M%S}"
    checkpoints.parent.mkdir(parents=True, exist_ok=True)
    try:
        with get_driver() as driver:
            prepare_event_layer(driver, scenario)
            async with agent_deps(scenario_path, config=config, org=org, tracer=tracer, llm=llm, driver=driver,
                                  approval=True) as deps, \
                    AsyncSqliteSaver.from_conn_string(str(checkpoints)) as saver, AsyncExitStack() as stack:
                approver = None
                if decide is None and approve == "slack":
                    approver = await stack.enter_async_context(slack_approver(load_names(driver)))
                    decide = approver.decide
                decide = decide or ask_terminal
                graph = build_graph(deps, checkpointer=saver)
                run_config = {"configurable": {"thread_id": thread}, "recursion_limit": 200}
                with tracer.run("diagnosis", session_id=thread, tags=[f"config:{config}", "mode:demo"],
                                metadata={"scenario": scenario["id"], "config": config, "org": org,
                                          "prompt_version": PROMPT_VERSION}, input={"reports": scenario["reports"]}):
                    state = await graph.ainvoke(initial_state(scenario["reports"]), config=run_config)
                    while state.get("__interrupt__"):
                        payload = state["__interrupt__"][0].value
                        print(f"\n승인 대기: 사건 {payload['incident_id']}")
                        for run in payload["pending"]:
                            waiting = [t for t in run["required_approver_teams"]
                                       if t not in (run.get("approved_by_teams") or [])]
                            print(f"  {run['action_id']} ({run['run_id']}): 승인을 기다리는 팀 {waiting}", flush=True)
                        decisions = decide(payload)
                        if inspect.isawaitable(decisions):
                            decisions = await decisions
                        state = await graph.ainvoke(Command(resume=decisions), config=run_config)
                    if approver is not None:
                        await approver.finish(state.get("outputs") or [])
                cost = sum(r.cost["total"] for r in deps.llm_results)
    finally:
        tracer.flush()
    return {"thread": thread, "outputs": state.get("outputs") or [], "unclassified": state.get("unclassified") or [],
            "cost_usd": cost}


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.scenario)
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if scenario.get("split") != "dev":
        print("실행할 수 없다: dev 시나리오만 쓴다 (4단계 4-2)", file=sys.stderr)
        return 1
    if scenario_org(scenario) != args.org:
        print(f"실행할 수 없다: 시나리오의 조직({scenario_org(scenario)})과 --org가 다르다", file=sys.stderr)
        return 1
    try:
        result = asyncio.run(run_demo(scenario, path, org=args.org, config=args.config, approve=args.approve))
    except PackageError as exc:
        print(f"실행할 수 없다: 배포 패키지를 쓸 수 없다: {exc}", file=sys.stderr)
        return 1
    except SlackConfigError as exc:
        print(f"실행할 수 없다: {exc}", file=sys.stderr)
        return 1
    for out in result["outputs"]:
        print(f"\n사건 {out['incident_id']}: 원인 {out['root_cause']}, 담당 {out['primary_team']}, 절차 {out['procedures']}")
        for action in out["actions"]:
            print(f"  {action['action_id']}: {action['status']}")
        for reason in out["needs_human"]:
            print(f"  사람 확인: {reason}")
    print(f"\n스레드 {result['thread']}, 비용 ${result['cost_usd']:.5f}")
    print(json.dumps({"unclassified": result["unclassified"]}, ensure_ascii=False) if result["unclassified"] else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
