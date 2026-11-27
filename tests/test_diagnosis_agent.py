"""진단 에이전트 11단계 검증. 테스트용 Neo4j에서 돈다. API를 부르지 않는다.

LLM은 실제 LLM 클래스와 실행 기록을 그대로 거치고, API 응답만 가짜다. 가짜 응답은 단계 프롬프트로
단계를 알아보고 정해진 규칙대로 답한다 (규칙대로 판단하는 모델을 흉내 낸다). 응답 규칙을 바꿔
출력 검증, 다시 묻기, 게이트 차단 같은 경우를 만든다.
MCP 서버는 실제 서버를 프로세스 안에서 붙인다. 도구 호출은 MCP 프로토콜을 그대로 거친다.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from anthropic.types import Message
from mcp import Client

from agents.diagnosis import prompts
from agents.diagnosis.graph import Deps, build_graph, initial_state
from agents.diagnosis.run import prepare_event_layer
from common.knowledge import GraphKnowledge
from common.models import LLM, load_model_config
from eval.validate import expected_procedures
from mcp_servers.mock_tools.server import build_server as build_mock_server
from mcp_servers.ontology_query import queries
from mcp_servers.ontology_query.server import build_server as build_onto_server
from ontology.apply_schema import apply_schema
from ontology.conditions import Environment
from ontology.load import load_static
from ontology.rules import rules_of
from ontology.seed import load_seed
from tests.memory_tracer import MemoryTracer

pytestmark = pytest.mark.integration

KST = timezone(timedelta(hours=9))
SCENARIOS = Path(__file__).parent.parent / "eval" / "scenarios" / "dev"
TOOL_OF = {}  # 판정 근거 ID -> 도구 이름 (seed에서 채운다)


@pytest.fixture(scope="module")
def seed():
    data = load_seed("org-a")
    TOOL_OF.update({c["id"]: c["tool"] for c in data["evidence_checks"]})
    return data


@pytest.fixture(scope="module")
def db(driver, seed):
    apply_schema(driver)
    load_static(driver, seed)
    return driver


# --- 가짜 모델 -----------------------------------------------------------------

def message(content: list[dict], stop_reason: str = "end_turn") -> Message:
    return Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    })


def text(data: dict) -> Message:
    return message([{"type": "text", "text": json.dumps(data, ensure_ascii=False)}])


class RuleModel:
    """단계 프롬프트로 단계를 알아보고 규칙대로 답하는 가짜 모델.

    intake: 단계 1의 답. scenario의 정답(증상, 범위)을 쓴다
    overrides: 단계 이름 -> 답을 바꾸는 함수 목록. 호출될 때마다 앞에서 하나씩 꺼내 쓴다
    """

    def __init__(self, intake: dict, overrides: dict | None = None):
        self.intake_data = intake
        self.verdicts: dict = {}
        self.conditions: list = []
        self.overrides = {k: list(v) for k, v in (overrides or {}).items()}
        self.calls: list[str] = []

    def _override(self, step: str, params: dict):
        queue = self.overrides.get(step)
        return queue.pop(0)(params) if queue else None

    def create(self, **params) -> Message:
        system = params["system"]
        step = {prompts.INTAKE: "intake", prompts.DUPLICATE: "duplicate", prompts.CHECK: "check",
                prompts.COLLABORATION: "collaboration", prompts.PROCEDURE: "procedure"}[system]
        self.calls.append(step)
        custom = self._override(step, params)
        if custom is not None:
            return custom
        return getattr(self, step)(params)

    def duplicate(self, params):
        ids = params["output_config"]["format"]["schema"]["properties"]["judgments"]["items"]["properties"]["incident_id"]["enum"]
        return text({"judgments": [{"incident_id": i, "verdict": "same", "reason": "같은 증상"} for i in ids]})

    def check(self, params):
        first = params["messages"][0]["content"]
        cause = re.search(r"확인할 원인 가설: (C\d+)", first).group(1)
        checks = json.loads(first.split("판정 근거:\n", 1)[1].split("\n\n이미 조회한 결과", 1)[0])
        if len(params["messages"]) == 1:
            uses = [{"type": "tool_use", "id": f"tu_{i}", "name": c["tool"], "input": c["args"]}
                    for i, c in enumerate(checks) if c["required"]]
            return message(uses, stop_reason="tool_use")
        # 도구 결과를 읽고 판단한다. 가짜 모델은 판단 기준을 바깥(verdicts)에서 받는다
        verdict = self.verdicts.get(cause, "rule_out")
        return text({"result": verdict, "used_checks": [c["check_id"] for c in checks if c["required"]],
                     "reused_results": [], "reason": "규칙대로"})

    def collaboration(self, params):
        return text({"apply": [], "reason": "조건에 해당하지 않는다"})

    def procedure(self, params):
        content = params["messages"][0]["content"]
        severity = re.search(r"심각도 (SEV\d)", content).group(1)
        candidates = json.loads(content.split("후보 절차:\n", 1)[1])
        chosen = [p for p in candidates if not p["human_only"]
                  and (p["procedure_id"] not in ("P3", "P4") or severity in ("SEV1", "SEV2"))]
        conditions = sorted({c for p in chosen for c in p["conditions"]} & set(self.conditions))
        chosen = [p for p in chosen if not p["conditions"] or set(p["conditions"]) & set(conditions)]
        actions = {a["action_id"]: a["risk"] for p in chosen for a in p["actions"]}
        return text({
            "procedures": [p["procedure_id"] for p in chosen], "resolved_conditions": conditions,
            "actions": [{"action_id": a, "mode": "approval" if r == "high" else "auto"} for a, r in sorted(actions.items())],
            "reason": "규칙대로",
        })

    def intake(self, params):
        return text(self.intake_data)


class TextOverride:
    """모의 도구 클라이언트를 감싸 응답 문장을 바꾼다. 같은 조회가 다른 결과를 내는 경우를 만든다."""

    def __init__(self, client, texts: dict[str, list[str]]):
        self.client, self.texts = client, {name: iter(values) for name, values in texts.items()}

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def call_tool(self, name, args):
        result = await self.client.call_tool(name, args)
        replacement = next(self.texts.get(name, iter(())), None)
        if replacement is None or result.is_error:
            return result
        block = result.content[0].model_copy(update={"text": replacement})
        return result.model_copy(update={"content": [block], "structured_content": None})


def run_agent(db, seed, scenario: dict, model: RuleModel, verdicts: dict, conditions=(), tool_texts=None,
              knowledge=None):
    """knowledge: (onto, tracer, llm, results) -> 지식 공급자. 없으면 그래프 공급자(구성 C)를 쓴다."""
    model.verdicts = verdicts
    model.conditions = list(conditions)
    tracer = MemoryTracer()
    llm = LLM(load_model_config("org-a"), tracer, SimpleNamespace(messages=SimpleNamespace(create=model.create)))

    async def go():
        prepare_event_layer(db, scenario)
        env = Environment(seed, scenario["environment"]["checks"])
        async with Client(build_onto_server(db)) as onto, Client(build_mock_server(seed, env)) as mock:
            listed = await mock.list_tools()
            results = []
            provider = knowledge(onto, tracer, llm, results) if knowledge else GraphKnowledge(onto, tracer)
            deps = Deps(
                driver=db, llm=llm, tracer=tracer, knowledge=provider, llm_results=results,
                mock_client=TextOverride(mock, tool_texts) if tool_texts else mock,
                mock_tools=[{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                            for t in listed.tools],
                tool_checks=queries.get_evidence_tools(db),
                symptoms=queries.get_symptom_types(db), systems=queries.get_systems(db),
                rules=rules_of(seed),
            )
            return await build_graph(deps).ainvoke(initial_state(scenario["reports"]),
                                                   config={"recursion_limit": 200}), tracer

    return asyncio.run(go())


def load(name: str) -> dict:
    return yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text(encoding="utf-8"))


def intake_of(scenario: dict, systems=("SY12", "SY13")) -> dict:
    e = scenario["expected"]
    return {"symptom_id": e["symptom"], "scope": e["scope"], "system_ids": list(systems), "reason": "테스트"}


def graph_count(db, query: str, **params) -> int:
    with db.session() as session:
        return session.run(query, **params).single()[0]


# --- 시나리오 1건 전체 ------------------------------------------------------------

def test_one_scenario_produces_structured_output(db, seed):
    """M2 완료 기준: 시나리오 1건을 넣으면 구조화 출력이 나온다 (4단계 3-3 필드)."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    state, tracer = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})

    (out,) = state["outputs"]
    expected = scenario["expected"]
    assert out["symptom"] == "S1" and out["severity"] == expected["severity"]
    assert out["root_cause"] == "C02" and out["primary_team"] == expected["primary_team"]
    # 우편함 지급 누락(prior 높음)이 1순위라 한 번에 확정된다. 다른 가설은 확인하지 않는다
    assert [(h["cause_id"], h["status"]) for h in out["hypotheses"]][0] == ("C02", "confirmed")
    # 7단계 판정 근거 조회는 필수 근거와 같고, 1단계에서 영향 범위 근거(E15)를 제보마다 한 번 조회한다
    assert {c["check_id"] for c in out["tool_calls"] if c["step"] == "7.check"} == set(expected["required_checks"])
    assert [c["check_id"] for c in out["tool_calls"] if c["step"] == "1.intake"] == ["E15"] * len(scenario["reports"])
    # 규칙대로 고르는 모델이면 정답 라벨의 절차, 조치와 같아야 한다 (P1은 정답 라벨에 넣지 않는다)
    procedures = set(out["procedures"]) - {"P1"}
    assert procedures == expected_procedures(seed, scenario) == set(expected["procedures"])
    runs = {r["action_id"]: r for r in out["actions"]}
    assert sorted(a for a, r in runs.items() if r["status"] == "auto_done") == expected["auto_actions"]
    assert sorted(a for a, r in runs.items() if r["status"] == "pending_approval") == expected["approval_actions"]
    assert not any(r["gate_blocked"] for r in runs.values())
    assert out["first_handler"]["role_id"] == "RL02"
    assert "C02" in out["explanation"] and out["needs_human"] == []
    for key in ("incident_id", "hypotheses", "collaborators", "procedures", "actions", "explanation", "retrieved_docs"):
        assert key in out

    # 사건 계층 기록: 접수, 사건, 가설, 판정 근거 결과, 실행 기록
    assert graph_count(db, "MATCH (:Report)-[:PART_OF]->(:Incident) RETURN count(*)") == 1
    assert graph_count(db, "MATCH (:Incident)-[h:HAS_HYPOTHESIS {status: 'confirmed'}]->(:Cause {id: 'C02'}) RETURN count(*)") == 1
    assert graph_count(db, "MATCH (f:Finding)-[:CHECKS]->(:EvidenceCheck) RETURN count(f)") == 2
    assert graph_count(db, "MATCH (r:ActionRun) RETURN count(r)") == len(out["actions"])
    # 실행 기록: LLM 호출, 도구 호출, 게이트 처리가 남는다
    kinds = [s["type"] for s in tracer.spans()]
    assert kinds.count("generation") == len(model.calls)
    assert kinds.count("tool") >= 2 and kinds.count("guardrail") == len(out["actions"])


def test_change_weight_reorders_and_is_recorded(db, seed):
    """시나리오의 변경 이력이 그래프에 기록되어 원인 순위에 가중이 붙고 STARTED_AFTER로 남는다 (R4)."""
    scenario = load("SC-S2-02")
    model = RuleModel(intake_of(scenario, systems=("SY01", "SY09")))
    state, _ = run_agent(db, seed, scenario, model, verdicts={scenario["expected"]["root_cause"]: "confirm"})
    (out,) = state["outputs"]
    top = out["hypotheses"][0]
    assert top["cause_id"] == scenario["expected"]["root_cause"]
    assert graph_count(db, "MATCH (:Incident)-[:STARTED_AFTER]->(:Change) RETURN count(*)") >= 1


# --- 판정 근거가 확정되지 않는 경우 ---------------------------------------------------

def test_moves_to_next_hypothesis(db, seed):
    """1순위가 배제되면 다음 후보로 넘어간다 (R4)."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C01": "confirm"})
    (out,) = state["outputs"]
    assert [h["status"] for h in out["hypotheses"][:2]] == ["ruled_out", "confirmed"]
    assert out["root_cause"] == "C01"


def test_hypothesis_cap_stops_before_the_last_candidate(db, seed):
    """확정이 안 되면 상한(3개, 가정값)까지만 확인한다. 후보가 4개인 S3로 상한 자체를 본다."""
    scenario = load("SC-S3-01")
    model = RuleModel(intake_of(scenario, systems=("SY02", "SY17")))
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    assert len(out["hypotheses"]) == 4
    assert out["root_cause"] is None and model.calls.count("check") == 3 * 2
    assert [h["result"] is not None for h in out["hypotheses"]] == [True, True, True, False]


def test_unconfirmed_assigns_top_two_teams(db, seed):
    """원인이 확정되지 않으면 상위 2개 가설의 담당 팀을 함께 배정하고 사람 확인을 요청한다 (R5-4)."""
    scenario = load("SC-S1-01")
    state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario)), verdicts={"C02": "inconclusive", "C01": "inconclusive", "C03": "inconclusive"})
    (out,) = state["outputs"]
    assert out["root_cause"] is None
    assert out["primary_team"] == "TM03"  # 1순위 C02의 담당 (서버개발팀)
    assert [c["team_id"] for c in out["collaborators"]] == ["TM09"]  # 2순위 C01의 담당 (플랫폼팀)
    # 다음 확인은 1순위 C02의 필수 근거 중 아직 결과가 없는 것이다. 모두 조회했으면 사람이 결과를 판단한다
    assert out["next_check"] is None
    assert any("확정되지 않았다" in n for n in out["needs_human"])
    # 원인이 없으면 접수 절차만 연다
    assert out["procedures"] == ["P1"]


# --- 출력 검증과 안전 -----------------------------------------------------------------

def test_invalid_intake_is_asked_again(db, seed):
    """정의되지 않은 값이 나오면 문제를 알려 한 번 다시 묻는다 (5단계 4-1 출력 검증)."""
    scenario = load("SC-S1-01")
    bad = lambda p: text({"symptom_id": "S1", "scope": "some_users", "system_ids": [], "reason": "x"})  # noqa: E731
    model = RuleModel(intake_of(scenario), overrides={"intake": [bad]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert model.calls[:2] == ["intake", "intake"]
    assert state["outputs"][0]["root_cause"] == "C02"


def test_intake_failing_twice_goes_to_human(db, seed):
    scenario = load("SC-S1-01")
    bad = lambda p: text({"symptom_id": "S1", "scope": "some_users", "system_ids": [], "reason": "x"})  # noqa: E731
    model = RuleModel(intake_of(scenario), overrides={"intake": [bad, bad]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    assert state["outputs"] == []
    (item,) = state["unclassified"]
    assert "출력 검증 실패" in item["needs_human"][0]


def test_rollback_is_refused_and_asked_again(db, seed):
    """사람만 여는 절차(롤백 P5)를 고르면 코드가 거부하고 다시 묻는다 (4단계 3-4)."""
    scenario = load("SC-S3-01")  # C07: 롤백이 원인에 연결돼 있다
    model = RuleModel(intake_of(scenario, systems=("SY14", "SY17")))

    def with_rollback(params):
        answer = json.loads(RuleModel.procedure(model, params).content[0].text)
        answer["procedures"].append("P5")
        return text(answer)

    model.overrides["procedure"] = [with_rollback]
    state, _ = run_agent(db, seed, scenario, model, verdicts={scenario["expected"]["root_cause"]: "confirm"})
    assert model.calls.count("procedure") == 2
    (out,) = state["outputs"]
    assert "P5" not in out["procedures"] and "A09" not in {r["action_id"] for r in out["actions"]}


def test_high_risk_marked_auto_is_blocked(db, seed):
    """고위험 조치를 자동 실행으로 표시해도 게이트가 막고 승인 대기로 둔다 (Q3, 게이트 차단 건수)."""
    scenario = load("SC-S4-02")  # 제보에 보상을 바로 지급하라는 지시가 섞인 안전 시나리오
    model = RuleModel(intake_of(scenario, systems=("SY15", "SY13")))

    def all_auto(params):
        answer = json.loads(RuleModel.procedure(model, params).content[0].text)
        for a in answer["actions"]:
            a["mode"] = "auto"
        return text(answer)

    model.overrides["procedure"] = [all_auto]
    state, _ = run_agent(db, seed, scenario, model, verdicts={scenario["expected"]["root_cause"]: "confirm"},
                         conditions=scenario["expected"].get("resolved_conditions", []))
    runs = state["outputs"][0]["actions"]
    high = [r for r in runs if r["action_id"] in scenario["expected"]["approval_actions"]]
    assert high and all(r["status"] == "pending_approval" and r["gate_blocked"] for r in high)
    assert graph_count(db, "MATCH (r:ActionRun) WHERE r.status IN ['approved', 'done'] RETURN count(r)") == 0


def test_cause_fix_timing_follows_severity(db, seed):
    """원인 수정(A15)의 반영 시점은 심각도로 정한다 (R7). SEV3은 정기 점검 반영이라 개발PM팀이 승인한다."""
    scenario = load("SC-S1-01")
    state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario)), verdicts={"C02": "confirm"})
    with db.session() as session:
        teams = session.run("MATCH (r:ActionRun {action_id: 'A15'}) RETURN r.required_approver_teams AS t").single()["t"]
    assert teams == ["TM05"]


# --- 접수와 중복 -----------------------------------------------------------------------

def two_reports(scenario: dict, *, same_reporter: bool) -> dict:
    first = scenario["reports"][0]
    second = dict(first, channel="CH01", at=(datetime.fromisoformat(first["at"]) + timedelta(minutes=10)).isoformat(),
                  reporter=first["reporter"] if same_reporter else "U9999", text="상점에서 산 패키지가 안 들어와요")
    return dict(scenario, reports=[first, second])


def test_same_reporter_is_merged_without_llm(db, seed):
    """같은 요청자가 시간창 안에 다른 채널로 보낸 제보는 같은 사건이다. 중복 판정 LLM을 부르지 않는다 (R2)."""
    scenario = two_reports(load("SC-S1-01"), same_reporter=True)
    model = RuleModel(intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert "duplicate" not in model.calls
    (out,) = state["outputs"]
    assert len(out["report_ids"]) == 2
    assert graph_count(db, "MATCH (i:Incident) RETURN max(i.report_count)") == 2


def test_other_reporter_goes_to_duplicate_judgment(db, seed):
    scenario = two_reports(load("SC-S1-01"), same_reporter=False)
    model = RuleModel(intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert model.calls.count("duplicate") == 1
    assert len(state["outputs"]) == 1 and len(state["outputs"][0]["report_ids"]) == 2


def test_ambiguous_duplicate_opens_new_incident_and_flags(db, seed):
    scenario = two_reports(load("SC-S1-01"), same_reporter=False)
    ambiguous = lambda p: text({"judgments": [  # noqa: E731
        {"incident_id": i, "verdict": "ambiguous", "reason": "시스템만 겹친다"}
        for i in p["output_config"]["format"]["schema"]["properties"]["judgments"]["items"]["properties"]["incident_id"]["enum"]]})
    model = RuleModel(intake_of(scenario), overrides={"duplicate": [ambiguous]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert len(state["outputs"]) == 2
    assert any("중복 의심" in n for n in state["outputs"][1]["needs_human"])
    assert graph_count(db, "MATCH (:Report)-[:SUSPECTED_DUPLICATE_OF]->(:Incident) RETURN count(*)") == 1


def test_out_of_scope_report_goes_to_human(db, seed):
    """증상 8개에 없는 제보는 진단하지 않고 사람에게 넘긴다 (4단계 T6)."""
    scenario = load("SC-S1-01")
    model = RuleModel({"symptom_id": "none", "scope": "some_users", "system_ids": [], "reason": "사용법 문의"})
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    assert state["outputs"] == [] and len(state["unclassified"]) == 1
    assert model.calls == ["intake"]
    assert graph_count(db, "MATCH (r:Report) WHERE NOT (r)-[:INDICATES]->() RETURN count(r)") == 1


# --- 리뷰 반영 (PR #12) ---------------------------------------------------------------

def test_non_object_answer_is_asked_again(db, seed):
    """JSON을 읽었어도 객체가 아니거나 필드가 빠지면 다시 묻는다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario), overrides={"intake": [lambda p: text([])]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert model.calls[:2] == ["intake", "intake"] and state["outputs"][0]["root_cause"] == "C02"
    missing = lambda p: text({"symptom_id": "S1", "scope": "some_users", "system_ids": ["SY12"]})  # noqa: E731
    model = RuleModel(intake_of(scenario), overrides={"intake": [missing]})
    run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert model.calls[:2] == ["intake", "intake"]


def test_urgent_procedures_are_candidates_without_confirmed_cause(db, seed):
    """SEV1, SEV2면 원인이 확정되지 않아도 긴급 공지(P3)와 긴급 점검(P4)이 후보다 (R3 TRIGGERS, R7)."""
    scenario = load("SC-S2-01")  # SEV1
    model = RuleModel(intake_of(scenario, systems=("SY01", "SY09")))
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    assert out["root_cause"] is None and out["severity"] == "SEV1"
    assert out["procedures"] == ["P1", "P3", "P4"]


@pytest.mark.parametrize("change, message", [
    (lambda a: a["actions"].pop(), "조치가 빠졌다"),
    (lambda a: a["actions"].append(dict(a["actions"][0])), "두 번 요청"),
])
def test_actions_must_match_chosen_procedures(db, seed, change, message):
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))

    def broken(params):
        answer = json.loads(RuleModel.procedure(model, params).content[0].text)
        change(answer)
        return text(answer)

    model.overrides["procedure"] = [broken]
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert model.calls.count("procedure") == 2
    runs = [r["action_id"] for r in state["outputs"][0]["actions"]]
    assert len(runs) == len(set(runs)) and set(runs) == set(scenario["expected"]["auto_actions"]) | set(scenario["expected"]["approval_actions"])


def test_conditional_procedure_needs_reported_condition(db, seed):
    """조건이 붙은 절차(과지급이면 회수 P7)를 고르면 충족한 조건도 적어야 한다."""
    scenario = load("SC-S4-01")
    model = RuleModel(intake_of(scenario, systems=("SY15", "SY13")))

    def no_condition(params):
        answer = json.loads(RuleModel.procedure(model, params).content[0].text)
        answer["resolved_conditions"] = []
        return text(answer)

    model.overrides["procedure"] = [no_condition]
    state, _ = run_agent(db, seed, scenario, model, verdicts={scenario["expected"]["root_cause"]: "confirm"},
                         conditions=["과지급"])
    assert model.calls.count("procedure") == 2
    assert "P7" in state["outputs"][0]["procedures"]


def test_reused_evidence_is_linked_to_the_later_hypothesis(db, seed):
    """앞 가설에서 조회한 결과를 다시 부르지 않고 쓴 가설에도 근거가 남는다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))

    def answer_from_cache(params):
        # C01을 확인할 때 도구를 다시 부르지 않고 앞에서 조회한 E01 결과로 판단한다
        rid = re.search(r"^- (K\d+) E01 ", params["messages"][0]["content"], re.M).group(1)
        return text({"result": "confirm", "used_checks": ["E01"], "reused_results": [rid], "reason": "앞의 결제 기록 조회 결과"})

    original = model.check

    def check(params):
        first = params["messages"][0]["content"]
        if "확인할 원인 가설: C01" in first:
            return answer_from_cache(params)
        return original(params)

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "rule_out"})
    (out,) = state["outputs"]
    c01 = next(h for h in out["hypotheses"] if h["cause_id"] == "C01")
    assert [(f["check_id"], f["reused"]) for f in c01["findings"]] == [("E01", True)]
    assert graph_count(db, "MATCH (f:Finding {cause_id: 'C01'}) RETURN count(f)") == 1
    # 실제 호출 이력에는 C02를 확인할 때 부른 것만 있다
    assert {t["cause_id"] for t in out["tool_calls"] if t["step"] == "7.check"} == {"C02"}


def burst(scenario: dict, times: list[int]) -> dict:
    """같은 요청자가 보낸 제보 여러 건. times는 첫 제보로부터의 분이고, 이 순서로 넣는다."""
    first = scenario["reports"][0]
    start = datetime.fromisoformat(first["at"])
    return dict(scenario, reports=[dict(first, at=(start + timedelta(minutes=m)).isoformat()) for m in times])


def test_escalation_counts_reports_up_to_this_one(db, seed):
    """30분 안에 20건이 몰리면 한 단계 올리고 P2를 반드시 연다 (R3)."""
    scenario = burst(load("SC-S1-01"), list(range(20)))
    state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario)), verdicts={"C02": "confirm"})
    (out,) = state["outputs"]
    assert out["escalated"] and out["severity"] == "SEV2" and "P2" in out["procedures"]


def test_later_reports_do_not_count_for_an_earlier_one(db, seed):
    """입력 순서가 시각 순서와 달라도 이번 제보보다 늦게 접수된 제보는 세지 않는다."""
    scenario = burst(load("SC-S1-01"), [0] + [120] * 18 + [10])
    state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario)), verdicts={"C02": "confirm"})
    (out,) = state["outputs"]
    assert not out["escalated"] and out["severity"] == "SEV3"


def test_test_split_is_refused_before_any_call(tmp_path):
    from eval.runner.runner import RunnerError, check_dev_only
    path = tmp_path / "SC-T.yaml"
    path.write_text("split: test\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="dev 시나리오만"):
        check_dev_only([path])


def test_string_list_items_are_type_checked(db, seed):
    """문자열 목록에 다른 타입이 섞이면 오류로 멈추지 않고 다시 묻는다."""
    scenario = load("SC-S1-01")
    bad = lambda p: text({"symptom_id": "S1", "scope": "some_users", "system_ids": [{}], "reason": "x"})  # noqa: E731
    model = RuleModel(intake_of(scenario), overrides={"intake": [bad]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert model.calls[:2] == ["intake", "intake"] and state["outputs"][0]["root_cause"] == "C02"


def test_check_answer_with_bad_used_checks_is_inconclusive(db, seed):
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    original = model.check

    def check(params):
        if len(params["messages"]) > 1:
            return text({"result": "confirm", "used_checks": [1], "reused_results": [], "reason": "x"})
        return original(params)

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    assert state["outputs"][0]["hypotheses"][0]["result"] == "inconclusive"


@pytest.mark.parametrize("sid, systems", [("SC-S2-01", ("SY01", "SY09")), ("SC-S3-01", ("SY02", "SY17"))])
def test_urgent_procedures_cannot_be_left_out(db, seed, sid, systems):
    """SEV1(S2-01), SEV2(S3-01)에서 P3, P4를 빠뜨리면 다시 묻는다 (R3, R7)."""
    scenario = load(sid)
    model = RuleModel(intake_of(scenario, systems=systems))

    def without_urgent(params):
        answer = json.loads(RuleModel.procedure(model, params).content[0].text)
        answer["procedures"] = [p for p in answer["procedures"] if p not in ("P3", "P4")]
        answer["actions"] = [a for a in answer["actions"] if a["action_id"] not in ("A06", "A07", "A08")]
        return text(answer)

    model.overrides["procedure"] = [without_urgent]
    state, _ = run_agent(db, seed, scenario, model, verdicts={scenario["expected"]["root_cause"]: "confirm"})
    assert model.calls.count("procedure") == 2
    assert {"P3", "P4"} <= set(state["outputs"][0]["procedures"])


def test_reused_result_is_linked_by_result_id_not_check_id(db, seed):
    """같은 근거(E06)를 인자만 다르게 두 번 조회했으면, 다음 가설에는 결과 ID로 고른 결과만 근거로 잇는다.

    S3에서 C08은 E06(SY10), C09는 E06(SY18)을 본다. 첫 가설에서 둘 다 조회해 두고 C09가 SY18 결과를 인용한다.
    """
    scenario = load("SC-S3-01")
    model = RuleModel(intake_of(scenario, systems=("SY02", "SY17")))
    seen = {}

    def check(params):
        first = params["messages"][0]["content"]
        cause = re.search(r"확인할 원인 가설: (C\d+)", first).group(1)
        seen.setdefault("first", cause)
        if cause == seen["first"]:
            if len(params["messages"]) == 1:
                uses = [{"type": "tool_use", "id": f"tu_{s}", "name": "query_deploy_history", "input": {"system": s}}
                        for s in ("SY10", "SY18")]
                return message(uses, stop_reason="tool_use")
            return text({"result": "rule_out", "used_checks": [], "reused_results": [], "reason": "x"})
        if cause == "C09":
            seen["cached"] = [line for line in first.splitlines() if line.startswith("- K") and " E06 " in line]
            picked = next(line.split()[1] for line in seen["cached"] if '"SY18"' in line)
            return text({"result": "confirm", "used_checks": ["E06"], "reused_results": [picked],
                         "reason": "앞의 SY18 배포 이력"})
        return text({"result": "rule_out", "used_checks": [], "reused_results": [], "reason": "x"})

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    assert seen["first"] != "C09"
    assert len(seen["cached"]) == 2  # 같은 근거를 인자만 다르게 조회한 결과가 둘 있다
    (out,) = state["outputs"]
    c09 = next(h for h in out["hypotheses"] if h["cause_id"] == "C09")
    assert [(f["check_id"], f["reused"], f["args"]) for f in c09["findings"]] == [("E06", True, {"system": "SY18"})]
    # 그래프의 Finding에도 그 조회의 인자가 남는다
    with db.session() as session:
        stored = session.run("MATCH (f:Finding {cause_id: 'C09'}) RETURN f.tool AS t, f.args AS a").data()
    assert [(r["t"], json.loads(r["a"])) for r in stored] == [("query_deploy_history", {"system": "SY18"})]


def test_unknown_reused_result_id_is_inconclusive(db, seed):
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    model.check = lambda params: text({"result": "confirm", "used_checks": [], "reused_results": ["K99"], "reason": "x"})
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    assert state["outputs"][0]["hypotheses"][0]["result"] == "inconclusive"
    # 형식이 틀린 답은 출력 형식 오류로 사람에게 넘긴다. 채점기가 이 사유를 센다 (PR #39 리뷰)
    from agents.diagnosis.graph import OUTPUT_ERROR

    assert any(OUTPUT_ERROR in reason for reason in state["outputs"][0]["needs_human"])


def test_same_query_called_again_keeps_the_earlier_result(db, seed):
    """같은 도구를 같은 인자로 다시 불러 결과가 달라져도, 앞 결과 ID를 인용하면 앞 결과가 근거로 남는다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    original = model.check
    seen = {}

    def check(params):
        first = params["messages"][0]["content"]
        cause = re.search(r"확인할 원인 가설: (C\d+)", first).group(1)
        seen.setdefault("first", cause)
        payment = [{"type": "tool_use", "id": "tu_pay", "name": "query_payment_record", "input": {}}]
        if cause == seen["first"]:
            if len(params["messages"]) == 1:
                return message(payment, stop_reason="tool_use")
            return text({"result": "rule_out", "used_checks": ["E01"], "reused_results": [], "reason": "x"})
        if "picked" not in seen:
            if len(params["messages"]) == 1:
                # 앞 가설의 E01 결과(K1)가 보이는데도 같은 인자로 다시 조회한다
                seen["old"] = re.search(r"^- (K\d+) E01 ", first, re.M).group(1)
                return message(payment, stop_reason="tool_use")
            seen["picked"] = True
            return text({"result": "confirm", "used_checks": ["E01"], "reused_results": [seen["old"]], "reason": "앞 결과"})
        return original(params)

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={}, tool_texts={"query_payment_record": ["첫 조회 결과", "다시 조회한 결과"]})
    (out,) = state["outputs"]
    second = next(h for h in out["hypotheses"] if h["cause_id"] != seen["first"] and h["findings"])
    assert sorted((f["reused"], f["detail"]) for f in second["findings"]) == [(False, "다시 조회한 결과"), (True, "첫 조회 결과")]


def test_intake_sees_the_report_lookup(db, seed):
    """접수 정리에는 접수 내역 조회(E15) 결과가 함께 들어간다. 영향 범위의 근거다 (3단계 6장)."""
    scenario = load("SC-S3-01")
    detail = next(c["detail"] for c in scenario["environment"]["checks"] if c["check"] == "E15")
    seen = []
    model = RuleModel(intake_of(scenario), overrides={"intake": [lambda p: seen.append(p["messages"][0]["content"])]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={scenario["expected"]["root_cause"]: "confirm"})
    assert f"접수 내역 조회(E15) 결과: {detail}" in seen[0]
    (out,) = state["outputs"]
    assert [c for c in out["tool_calls"] if c["step"] == "1.intake"] == [
        {"step": "1.intake", "cause_id": None, "tool": "query_reports", "check_id": "E15", "args": {}, "is_error": False}]


def test_answer_without_required_checks_is_asked_again(db, seed):
    """필수 판정 근거를 조회하지 않고 답하면 한 번 다시 묻는다. 다시 물은 뒤 조회해서 확정한다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    original = model.check
    prompts_seen = []

    def check(params):
        prompts_seen.append(params["messages"][-1]["content"])
        if len(params["messages"]) == 1:
            return text({"result": "inconclusive", "used_checks": [], "reused_results": [], "reason": "조회 안 함"})
        if len(params["messages"]) == 3:
            # 다시 묻는 말을 받고 처음처럼 필수 근거를 조회한다
            return original({**params, "messages": params["messages"][:1]})
        return original(params)

    model.check = check
    root = scenario["expected"]["root_cause"]
    state, _ = run_agent(db, seed, scenario, model, verdicts={root: "confirm"})
    (out,) = state["outputs"]
    assert out["root_cause"] == root
    retry = next(p for p in prompts_seen if isinstance(p, str) and "필수 판정 근거" in p)
    assert all(c in retry for c in scenario["expected"]["required_checks"])


def test_required_check_retry_happens_once(db, seed):
    """다시 물어도 조회하지 않으면 그 답을 받는다. 무한히 다시 묻지 않는다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    model.check = lambda params: text({"result": "inconclusive", "used_checks": [], "reused_results": [],
                                        "reason": "조회 안 함"})
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    assert out["root_cause"] is None
    assert model.calls.count("check") == 2 * len(out["hypotheses"][:3])
    assert all(h["result"] == "inconclusive" for h in out["hypotheses"][:3])
    # 형식에 맞는 불명확 답은 출력 형식 오류가 아니다
    from agents.diagnosis.graph import OUTPUT_ERROR

    assert not any(OUTPUT_ERROR in reason for reason in out["needs_human"])


def test_confirm_without_required_checks_after_retry_is_inconclusive(db, seed):
    """다시 물어도 필수 근거 없이 확정하면 원인을 확정하지 않는다 (R4-4)."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    model.check = lambda params: text({"result": "confirm", "used_checks": [], "reused_results": [],
                                        "reason": "조회 없이 확정"})
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    assert out["root_cause"] is None
    first = out["hypotheses"][0]
    assert first["result"] == "inconclusive" and first["status"] == "candidate"
    assert graph_count(db, "MATCH (:Incident)-[h:HAS_HYPOTHESIS {status: 'confirmed'}]->() RETURN count(h)") == 0


def test_next_check_names_a_team_id_and_skips_checked_evidence(db, seed):
    """다음 확인은 아직 결과가 없는 필수 근거를 가리키고, 담당은 팀 ID로 적는다 (R5-4)."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    original = model.check

    def check(params):
        if "확인할 원인 가설: C02" in params["messages"][0]["content"]:
            if len(params["messages"]) == 1:
                # 필수 근거 둘 중 E01만 조회한다
                return message([{"type": "tool_use", "id": "tu_0", "name": "query_payment_record", "input": {}}],
                               stop_reason="tool_use")
            return text({"result": "inconclusive", "used_checks": ["E01"], "reused_results": [], "reason": "E02 미확인"})
        return original(params)

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C01": "inconclusive", "C03": "inconclusive"})
    (out,) = state["outputs"]
    assert out["next_check"]["cause_id"] == "C02" and out["next_check"]["check_id"] == "E02"
    assert out["next_check"]["by"] in {t["id"] for t in seed["teams"]}


def test_upper_inconclusive_hypothesis_is_flagged_when_lower_confirms(db, seed):
    """상위 가설이 불명확인 채로 하위 가설이 확정되면 사람 확인을 붙인다 (R4-4)."""
    scenario = load("SC-S1-01")
    state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario)),
                         verdicts={"C02": "inconclusive", "C01": "confirm"})
    (out,) = state["outputs"]
    assert out["root_cause"] == "C01"
    assert any("C02" in n and "불명확" in n for n in out["needs_human"])


def test_same_reporter_with_other_symptom_is_not_merged(db, seed):
    """같은 요청자라도 증상이 다르면 같은 사건으로 바로 묶지 않고 중복 판정으로 넘긴다 (R2)."""
    scenario = two_reports(load("SC-S1-01"), same_reporter=True)
    other = lambda p: text({"symptom_id": "S2", "scope": "some_users", "system_ids": ["SY01"], "reason": "x"})  # noqa: E731
    model = RuleModel(intake_of(scenario), overrides={"intake": [lambda p: None, other]})
    run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    assert "duplicate" in model.calls


def test_unrelated_query_is_not_recorded_as_evidence(db, seed):
    """가설의 판정 근거가 아닌 조회는 호출 이력에만 남고 그 가설의 근거(Finding)가 되지 않는다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    original = model.check

    def check(params):
        if len(params["messages"]) == 1 and "확인할 원인 가설: C02" in params["messages"][0]["content"]:
            uses = original(params).content
            extra = {"type": "tool_use", "id": "tu_extra", "name": "query_deploy_history", "input": {"system": "SY10"}}
            return message([b.model_dump() for b in uses] + [extra], stop_reason="tool_use")
        return original(params)

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
    (out,) = state["outputs"]
    c02 = next(h for h in out["hypotheses"] if h["cause_id"] == "C02")
    assert "E06" not in {f["check_id"] for f in c02["findings"]}
    assert "E06" in {t["check_id"] for t in out["tool_calls"]}
    assert graph_count(db, "MATCH (f:Finding {cause_id: 'C02', check_id: 'E06'}) RETURN count(f)") == 0


def test_required_check_with_wrong_arguments_does_not_count(db, seed):
    """원인이 인자를 정한 필수 근거는 그 인자로 조회해야 충족된다. 다른 시스템으로 조회하면 다시 묻는다."""
    scenario = load("SC-S7-01")
    model = RuleModel(intake_of(scenario, systems=("SY14",)))
    original = model.check
    asked = []

    def check(params):
        first = params["messages"][0]["content"]
        checks = json.loads(first.split("판정 근거:\n", 1)[1].split("\n\n이미 조회한 결과", 1)[0])
        if len(params["messages"]) == 1:
            uses = [{"type": "tool_use", "id": f"tu_{i}", "name": c["tool"],
                     "input": {k: "SY01" for k in c["args"]} if c["args"] else {}}
                    for i, c in enumerate(checks) if c["required"]]
            return message(uses, stop_reason="tool_use")
        asked.append(params["messages"][-1]["content"])
        return text({"result": "confirm", "used_checks": [c["check_id"] for c in checks if c["required"]],
                     "reused_results": [], "reason": "x"})

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    first = state["outputs"][0]["hypotheses"][0]
    assert first["result"] == "inconclusive"
    assert any(isinstance(a, str) and "필수 판정 근거" in a for a in asked)


def test_retry_drops_tool_use_blocks_from_the_earlier_answer(db, seed):
    """다시 묻는 대화에는 결과가 없는 도구 호출 블록을 남기지 않는다. 남기면 API가 요청을 거부한다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    seen = []

    def check(params):
        seen.append(params["messages"])
        if len(params["messages"]) == 1:
            answer = {"result": "confirm", "used_checks": [], "reused_results": [], "reason": "x"}
            return message([{"type": "text", "text": json.dumps(answer)},
                            {"type": "tool_use", "id": "tu_cut", "name": "query_payment_record", "input": {}}],
                           stop_reason="max_tokens")
        return text({"result": "inconclusive", "used_checks": [], "reused_results": [], "reason": "x"})

    model.check = check
    run_agent(db, seed, scenario, model, verdicts={})
    retry = next(m for m in seen if len(m) == 3)
    assert all(b["type"] != "tool_use" for b in retry[1]["content"])


def test_evidence_with_other_arguments_is_not_this_causes_finding(db, seed):
    """원인이 인자를 정한 근거는 그 인자의 결과만 근거다. C08(E06 SY10)에 SY18 결과가 붙지 않는다."""
    scenario = load("SC-S3-01")
    model = RuleModel(intake_of(scenario, systems=("SY02", "SY17")))
    original = model.check

    def check(params):
        first = params["messages"][0]["content"]
        if "확인할 원인 가설: C08" in first and len(params["messages"]) == 1:
            uses = [{"type": "tool_use", "id": f"tu_{s}", "name": "query_deploy_history", "input": {"system": s}}
                    for s in ("SY10", "SY18")]
            return message(uses, stop_reason="tool_use")
        if "확인할 원인 가설: C08" in first:
            return text({"result": "rule_out", "used_checks": ["E06"], "reused_results": [], "reason": "x"})
        return text({"result": "rule_out", "used_checks": [], "reused_results": [], "reason": "x"})

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    c08 = next(h for h in out["hypotheses"] if h["cause_id"] == "C08")
    assert [f["args"] for f in c08["findings"] if f["check_id"] == "E06"] == [{"system": "SY10"}]


def test_input_order_does_not_change_grouping(db, seed):
    """제보가 시각 순서와 다르게 들어와도 접수 시각 순서로 처리해 같은 사건으로 묶는다 (R2)."""
    forward = two_reports(load("SC-S1-01"), same_reporter=False)
    backward = dict(forward, reports=list(reversed(forward["reports"])))
    results = []
    for scenario in (forward, backward):
        model = RuleModel(intake_of(scenario))
        state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"})
        (out,) = state["outputs"]
        results.append(sorted(out["report_indexes"]))
        # 묶인 제보에는 처리 중 회신(A04)을 보낸다
        assert [r["action_id"] for r in out["merge_replies"]] == ["A04"]
        assert out["merge_replies"][0]["status"] == "auto_done"
        # 실행 기록에 대상 접수가 남는다. 묶인 제보는 사건의 두 번째 접수다
        run_id, target = out["merge_replies"][0]["run_id"], out["merge_replies"][0]["report_id"]
        assert target == out["report_ids"][1]
        assert graph_count(db, "MATCH (r:ActionRun {id: $id, target_report_id: $t}) RETURN count(r)", id=run_id, t=target) == 1
    assert results == [[0, 1], [0, 1]]


def test_runner_end_to_end_with_mcp_processes(db, seed, tmp_path, monkeypatch):
    """실행기가 실제 MCP 서버 프로세스와 테스트용 Neo4j로 시나리오를 돌리고, 채점기가 그 결과를 읽는다.

    LLM만 규칙 모델로 바꾼다. 서버 프로세스는 NEO4J_URI를 받으므로 테스트용 DB 주소로 바꿔 넘긴다.
    """
    from agents.diagnosis.run import run_scenario
    from eval.grader.grade import grade_folder
    from eval.runner.runner import GitState, plan, run_all
    from tests.conftest import test_db_uri

    monkeypatch.setenv("NEO4J_URI", test_db_uri())
    path = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "dev" / "SC-S1-01.yaml"
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = RuleModel(intake_of(scenario))
    model.verdicts, model.conditions = {"C02": "confirm"}, []
    tracer = MemoryTracer()
    llm = LLM(load_model_config("org-a"), tracer, SimpleNamespace(messages=SimpleNamespace(create=model.create)))
    git = GitState(commit="abc1234", dirty=False, changed=[])
    expected = {"config": "C", "org": "org-a", "split": "dev", "prompt_version": prompts.PROMPT_VERSION,
                "git_commit": git.commit, "git_dirty": False}
    todo = plan([path], out=tmp_path, reps=1, resume=False, expected=expected)

    async def run_one(s, p):
        return await run_scenario(s, p, config="C", org="org-a", tracer=tracer, llm=llm, driver=db)
    from ontology.seed import seed_hash
    asyncio.run(run_all(todo, out=tmp_path, git=git, run_one=run_one, prices=llm.config.price_record,
                        seed_hash=seed_hash(seed), log=lambda _: None))

    result = grade_folder(tmp_path, seed)
    assert result["meta"]["config"] == "C" and result["meta"]["git_dirty"] is False
    (g,) = result["reps"][0]["scenarios"]
    assert g["symptom"] and g["top1"] and g["team"] and g["required"] == {"total": 2, "covered": 2}
    # 시나리오 하나라 이어서 돌릴 것이 없다
    assert plan([path], out=tmp_path, reps=1, resume=True, expected=expected) == []


# --- 9단계 절차 조건과 원인이 여는 절차 (M3 점검) -------------------------------------------


@pytest.mark.integration
def test_condition_check_is_looked_up_before_procedure_choice(db, seed):
    """조건이 붙은 절차(C11의 P6 미지급, P7 과지급)는 9단계 전에 코드가 조건의 판정 근거(E02)를 조회한다.

    M3 dev 실행에서 E02를 아무 단계도 부르지 않아 과지급 근거 없이 P7을 고르지 못했다 (SC-S4-01).
    두 조건이 같은 조회를 쓰면 한 번만 부르고, 결과 문장이 9단계 입력에 들어가야 한다.
    """
    scenario = load("SC-S4-01")
    seen = []
    model = RuleModel(intake_of(scenario, systems=("SY15",)),
                      overrides={"procedure": [lambda p: seen.append(p["messages"][0]["content"])]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C11": "confirm"}, conditions=["과지급"])
    (out,) = state["outputs"]
    calls = [c for c in out["tool_calls"] if c["step"] == "9.condition"]
    assert [(c["check_id"], c["tool"], c["cause_id"]) for c in calls] == [("E02", "query_mailbox_log", "C11")]
    assert "조건 과지급: 판정 근거 E02" in seen[0] and "설정된 수량의 두 배" in seen[0]
    assert "P7" in out["procedures"] and out["resolved_conditions"] == ["과지급"]


@pytest.mark.integration
def test_no_condition_lookup_without_conditional_procedure(db, seed):
    scenario = load("SC-S1-01")
    state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario)), verdicts={"C02": "confirm"})
    (out,) = state["outputs"]
    assert not [c for c in out["tool_calls"] if c["step"] == "9.condition"]


@pytest.mark.integration
def test_unconditional_cause_procedure_is_always_opened(db, seed):
    """확정 원인에 조건 없이 연결된 절차(C14의 회수 P7)는 코드가 항상 여는 절차로 둔다 (4단계 3-4, M3 점검).

    M3 dev 실행에서 LLM이 "C14와 연결 근거가 없다"며 P7을 뺐다 (구성 A, SC-S5-01). 빼면 다시 묻는다.
    """
    scenario = load("SC-S5-01")

    def drop_p7(params):
        candidates = json.loads(params["messages"][0]["content"].split("후보 절차:\n", 1)[1])
        assert {p["procedure_id"]: p["always"] for p in candidates}["P7"] is True
        procs = [p for p in candidates if p["procedure_id"] != "P7" and not p["human_only"]]
        actions = sorted({a["action_id"]: a["risk"] for p in procs for a in p["actions"]}.items())
        return text({"procedures": [p["procedure_id"] for p in procs], "resolved_conditions": [],
                     "actions": [{"action_id": a, "mode": "approval" if r == "high" else "auto"} for a, r in actions],
                     "reason": "P7을 뺀다"})

    model = RuleModel(intake_of(scenario, systems=("SY22",)), overrides={"procedure": [drop_p7]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C14": "confirm"})
    (out,) = state["outputs"]
    assert model.calls.count("procedure") == 2
    assert "P7" in out["procedures"]


@pytest.mark.integration
def test_severity_procedures_stay_with_severity_rule(db, seed):
    """원인에 연결된 긴급 공지(P3)는 조건이 없어도 항상 여는 절차로 만들지 않는다. SEV3, SEV4면 열지 않는다 (R3)."""
    scenario = load("SC-S5-01")
    seen = []
    model = RuleModel(intake_of(scenario, systems=("SY22",)),
                      overrides={"procedure": [lambda p: seen.append(p["messages"][0]["content"])]})
    run_agent(db, seed, scenario, model, verdicts={"C14": "confirm"})
    candidates = {p["procedure_id"]: p for p in json.loads(seen[0].split("후보 절차:\n", 1)[1])}
    severity = re.search(r"심각도 (SEV\d)", seen[0]).group(1)
    assert candidates["P4"]["always"] is (severity in ("SEV1", "SEV2"))


@pytest.mark.integration
def test_failed_condition_lookup_goes_to_human(db, seed):
    """조건 판정 근거를 조회하지 못하면 사람 확인 사유로 남긴다 (PR #22 리뷰).

    조건 절차는 필수가 아니라 LLM이 빼도 검증을 통과한다. 근거 없이 빠진 것을 알 수 있어야 한다.
    """
    scenario = load("SC-S4-01")

    class BrokenConditionTool(GraphKnowledge):
        async def cause_procedures(self, cause_id, severity_id):
            found = await super().cause_procedures(cause_id, severity_id)
            for proc in found["resolved_by"]:
                if proc.get("condition_check"):
                    proc["condition_check"] = {**proc["condition_check"], "tool": "no_such_tool"}
            return found

    model = RuleModel(intake_of(scenario, systems=("SY15",)))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C11": "confirm"},
                         knowledge=lambda onto, tracer, llm, results: BrokenConditionTool(onto, tracer))
    (out,) = state["outputs"]
    calls = [c for c in out["tool_calls"] if c["step"] == "9.condition"]
    assert len(calls) == 1 and calls[0]["is_error"]
    assert any("조건 판정 근거 E02를 조회하지 못했다" in n for n in out["needs_human"])
    assert "P7" not in out["procedures"]


@pytest.mark.integration
def test_condition_check_id_follows_called_tool(db, seed):
    """조건 조회의 판정 근거 ID는 7단계처럼 부른 도구로 정한다. 공급자가 준 ID와 도구가 어긋나면 사람 확인으로 남긴다
    (M3 점검 2 리뷰: 구성 A의 답이 E02에 다른 도구를 짝지어도 형식 검증은 통과한다)."""
    scenario = load("SC-S4-01")

    class MismatchedTool(GraphKnowledge):
        async def cause_procedures(self, cause_id, severity_id):
            found = await super().cause_procedures(cause_id, severity_id)
            for proc in found["resolved_by"]:
                if proc.get("condition_check"):
                    proc["condition_check"] = {**proc["condition_check"], "tool": "query_config_change",
                                               "args": {"system": "SY15"}}
            return found

    model = RuleModel(intake_of(scenario, systems=("SY15",)))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C11": "confirm"},
                         knowledge=lambda onto, tracer, llm, results: MismatchedTool(onto, tracer))
    (out,) = state["outputs"]
    calls = [c for c in out["tool_calls"] if c["step"] == "9.condition"]
    assert [(c["tool"], c["check_id"]) for c in calls] == [("query_config_change", "E07")]
    assert any("판정 근거는 E02인데 도구 query_config_change는 E07다" in n for n in out["needs_human"])


def test_tool_round_limit_is_not_an_output_error(db, seed):
    """도구 호출만 이어가다 한도에 닿으면 불명확으로 두고 사람에게 넘기되, 출력 형식 오류로 세지 않는다 (PR #39 리뷰)."""
    from agents.diagnosis.graph import MAX_TOOL_ROUNDS, OUTPUT_ERROR

    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    model.check = lambda params: message(
        [{"type": "tool_use", "id": "tu_pay", "name": "query_payment_record", "input": {}}], stop_reason="tool_use")
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    assert out["hypotheses"][0]["result"] == "inconclusive"
    assert any(f"도구 호출 {MAX_TOOL_ROUNDS}회" in reason for reason in out["needs_human"])
    assert not any(OUTPUT_ERROR in reason for reason in out["needs_human"])


def test_answer_on_the_last_round_is_not_reported_as_no_answer(db, seed):
    """마지막 회차에 답했는데 필수 근거가 없어 다시 묻게 되면, 답이 없었다고 하지 않고 처음 답을 불명확으로 쓴다 (M6 점검)."""
    from agents.diagnosis.graph import MAX_TOOL_ROUNDS, OUTPUT_ERROR

    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    first = {}

    def check(params):
        cause = re.search(r"확인할 원인 가설: (C\d+)", params["messages"][0]["content"]).group(1)
        first.setdefault("cause", cause)
        rounds = (len(params["messages"]) + 1) // 2
        if cause == first["cause"] and rounds < MAX_TOOL_ROUNDS:
            return message([{"type": "tool_use", "id": f"tu_{rounds}", "name": "query_deploy_history",
                             "input": {"system": "SY10"}}], stop_reason="tool_use")
        return text({"result": "confirm", "used_checks": [], "reused_results": [], "reason": "조회 없이 답함"})

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    hyp = next(h for h in out["hypotheses"] if h["cause_id"] == first["cause"])
    assert hyp["result"] == "inconclusive"
    assert not any(f"도구 호출 {MAX_TOOL_ROUNDS}회" in r for r in out["needs_human"])
    assert not any(OUTPUT_ERROR in r for r in out["needs_human"])


def test_restored_answer_is_inconclusive_even_if_checks_were_called_after_the_retry(db, seed):
    """다시 물은 뒤 필수 근거를 조회했지만 한도 안에 답하지 않으면, 조회 전에 한 처음 확정 답을 쓰지 않는다 (PR #40 리뷰)."""
    from agents.diagnosis.graph import MAX_TOOL_ROUNDS, OUTPUT_ERROR

    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    first = {}

    def check(params):
        content = params["messages"][0]["content"]
        cause = re.search(r"확인할 원인 가설: (C\d+)", content).group(1)
        first.setdefault("cause", cause)
        if cause != first["cause"]:
            return RuleModel.check(model, params)
        if len(params["messages"]) == 1:
            return text({"result": "confirm", "used_checks": [], "reused_results": [], "reason": "조회 전 답"})
        checks = json.loads(content.split("판정 근거:\n", 1)[1].split("\n\n이미 조회한 결과", 1)[0])
        uses = [{"type": "tool_use", "id": f"tu_{len(params['messages'])}_{i}", "name": c["tool"], "input": c["args"]}
                for i, c in enumerate(checks) if c["required"]]
        return message(uses, stop_reason="tool_use")

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    hyp = next(h for h in out["hypotheses"] if h["cause_id"] == first["cause"])
    assert hyp["result"] == "inconclusive"
    assert out["root_cause"] != first["cause"]
    assert not any(OUTPUT_ERROR in r for r in out["needs_human"])
    assert MAX_TOOL_ROUNDS > 2


# ---------- 데모 모드: 승인 대기와 반려 뒤 다시 고르기 (M7 PR 1) ----------


def run_demo(db, seed, scenario: dict, model: RuleModel, verdicts: dict, decide, conditions=(), checkpointer=None):
    """데모 모드로 돌린다. 멈출 때마다 decide(멈춘 내용) -> 승인 판단 목록으로 이어간다. (최종 상태, 멈춘 내용 목록)."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    model.verdicts = verdicts
    model.conditions = list(conditions)
    tracer = MemoryTracer()
    llm = LLM(load_model_config("org-a"), tracer, SimpleNamespace(messages=SimpleNamespace(create=model.create)))

    async def go():
        prepare_event_layer(db, scenario)
        env = Environment(seed, scenario["environment"]["checks"])
        async with Client(build_onto_server(db)) as onto, Client(build_mock_server(seed, env)) as mock:
            listed = await mock.list_tools()
            deps = Deps(
                driver=db, llm=llm, tracer=tracer, knowledge=GraphKnowledge(onto, tracer), llm_results=[],
                mock_client=mock, tool_checks=queries.get_evidence_tools(db),
                mock_tools=[{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                            for t in listed.tools],
                symptoms=queries.get_symptom_types(db), systems=queries.get_systems(db), rules=rules_of(seed),
                approval=True,
            )
            graph = build_graph(deps, checkpointer=checkpointer or InMemorySaver())
            config = {"configurable": {"thread_id": scenario["id"]}, "recursion_limit": 200}
            state = await graph.ainvoke(initial_state(scenario["reports"]), config=config)
            paused = []
            while state.get("__interrupt__"):
                payload = state["__interrupt__"][0].value
                paused.append(payload)
                state = await graph.ainvoke(Command(resume=decide(payload)), config=config)
            return state, paused

    return asyncio.run(go())


def approve_all(payload):
    return [{"run_id": r["run_id"], "team_id": t, "decision": "approve"}
            for r in payload["pending"] for t in r["required_approver_teams"] if t not in (r["approved_by_teams"] or [])]


def test_demo_waits_for_approval_and_completes_approved_actions(db, seed):
    """데모 모드는 고위험 조치에서 멈추고, 승인권자가 모두 승인하면 완료(시뮬레이션)로 남긴다."""
    scenario = load("SC-S1-01")
    state, paused = run_demo(db, seed, scenario, RuleModel(intake_of(scenario)), {"C02": "confirm"}, approve_all)
    assert len(paused) == 1
    assert sorted(r["action_id"] for r in paused[0]["pending"]) == scenario["expected"]["approval_actions"]
    (out,) = state["outputs"]
    high = [a for a in out["actions"] if a["action_id"] in scenario["expected"]["approval_actions"]]
    assert {a["status"] for a in high} == {"done"}
    assert {a["status"] for a in out["actions"] if a not in high} == {"auto_done"}
    assert graph_count(db, "MATCH (r:ActionRun {status: 'done'}) RETURN count(r)") == len(high)


def test_every_approver_must_approve_before_the_action_runs(db, seed):
    """승인권자가 둘인 긴급 점검(A08)은 한 팀만 승인하면 다시 멈춘다."""
    scenario = load("SC-S4-01")
    seen = []

    def one_team_at_a_time(payload):
        seen.append({r["action_id"]: list(r["approved_by_teams"] or []) for r in payload["pending"]})
        return [{"run_id": r["run_id"], "team_id": next(t for t in r["required_approver_teams"]
                                                         if t not in (r["approved_by_teams"] or [])),
                 "decision": "approve"} for r in payload["pending"]]

    state, paused = run_demo(db, seed, scenario, RuleModel(intake_of(scenario, ("SY15",))), {"C11": "confirm"},
                             one_team_at_a_time, conditions=["과지급"])
    assert len(paused) == 2 and "A08" in seen[1] and len(seen[1]["A08"]) == 1
    (out,) = state["outputs"]
    assert next(a for a in out["actions"] if a["action_id"] == "A08")["status"] == "done"


def test_rejection_without_alternative_goes_to_human(db, seed):
    """반려된 조치의 절차를 빼고, 같은 원인의 다른 해결 절차가 없으면 LLM을 부르지 않고 사람에게 넘긴다 (R6)."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))

    def reject_compensation(payload):
        return [{"run_id": r["run_id"], "team_id": r["required_approver_teams"][0],
                 "decision": "reject" if r["action_id"] == "A10" else "approve"} for r in payload["pending"]]

    state, _ = run_demo(db, seed, scenario, model, {"C02": "confirm"}, reject_compensation)
    (out,) = state["outputs"]
    assert out["excluded_procedures"] == ["P6"]
    assert next(a for a in out["actions"] if a["action_id"] == "A10")["status"] == "rejected"
    assert any("사람이 대안을 정한다" in r for r in out["needs_human"])
    assert model.calls.count("procedure") == 1  # 긴급 공지 P3은 SEV3이라 대안이 아니다


def test_rejection_reselects_another_procedure_of_the_same_cause(db, seed):
    """회수(P7)가 반려되면 같은 원인의 다른 해결 절차(보상 P6)를 다시 고르고, 새 조치도 승인을 기다린다 (R6)."""
    scenario = load("SC-S4-01")
    alt = text({"procedures": ["P6"], "resolved_conditions": ["미지급"],
                "actions": [{"action_id": "A10", "mode": "approval"}], "reason": "회수 대신 보상"})
    model = RuleModel(intake_of(scenario, ("SY15",)), overrides={"procedure": [lambda p: None, lambda p: alt]})
    seen = []

    def reject_recall(payload):
        seen.append(sorted(r["action_id"] for r in payload["pending"]))
        return [{"run_id": r["run_id"], "team_id": t, "decision": "reject" if r["action_id"] == "A11" else "approve"}
                for r in payload["pending"] for t in r["required_approver_teams"]]

    state, paused = run_demo(db, seed, scenario, model, {"C11": "confirm"}, reject_recall, conditions=["과지급"])
    (out,) = state["outputs"]
    assert model.calls.count("procedure") == 2
    assert seen[-1] == ["A10"]
    assert out["excluded_procedures"] == ["P7"]
    assert "P6" in out["procedures"]
    assert {a["action_id"]: a["status"] for a in out["actions"]}["A10"] == "done"


def test_wrong_approver_is_reported_and_asked_again(db, seed):
    scenario = load("SC-S1-01")
    calls = []

    def first_wrong(payload):
        calls.append(payload)
        if len(calls) == 1:
            return [{"run_id": payload["pending"][0]["run_id"], "team_id": "TM12", "decision": "approve"}]
        return approve_all(payload)

    state, paused = run_demo(db, seed, scenario, RuleModel(intake_of(scenario)), {"C02": "confirm"}, first_wrong)
    assert len(paused) == 2 and paused[1]["errors"] and not paused[0]["errors"]


def test_demo_mode_needs_a_checkpointer():
    deps = SimpleNamespace(approval=True)
    with pytest.raises(ValueError, match="체크포인터"):
        build_graph(deps)


def test_paused_run_resumes_from_sqlite_in_a_new_graph(db, seed, tmp_path):
    """멈춘 상태를 SQLite에 저장하면 다른 그래프 객체(프로세스가 다시 뜬 경우)에서 이어갈 수 있다."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    scenario = load("SC-S1-01")
    path = tmp_path / "checkpoints.sqlite"

    # 첫 실행은 멈춘 채로 끝내고, 두 번째 그래프에서 같은 스레드를 이어간다
    from langgraph.types import Command

    model = RuleModel(intake_of(scenario))
    model.verdicts = {"C02": "confirm"}
    tracer = MemoryTracer()
    llm = LLM(load_model_config("org-a"), tracer, SimpleNamespace(messages=SimpleNamespace(create=model.create)))
    config = {"configurable": {"thread_id": "resume-test"}, "recursion_limit": 200}

    async def graph_with(saver, onto, mock, listed):
        deps = Deps(driver=db, llm=llm, tracer=tracer, knowledge=GraphKnowledge(onto, tracer), llm_results=[],
                    mock_client=mock, tool_checks=queries.get_evidence_tools(db),
                    mock_tools=[{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                                for t in listed.tools],
                    symptoms=queries.get_symptom_types(db), systems=queries.get_systems(db), rules=rules_of(seed),
                    approval=True)
        return build_graph(deps, checkpointer=saver)

    async def go():
        prepare_event_layer(db, scenario)
        env = Environment(seed, scenario["environment"]["checks"])
        async with Client(build_onto_server(db)) as onto, Client(build_mock_server(seed, env)) as mock:
            listed = await mock.list_tools()
            async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
                first = await (await graph_with(saver, onto, mock, listed)).ainvoke(initial_state(scenario["reports"]),
                                                                                   config=config)
            payload = first["__interrupt__"][0].value
            async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
                return await (await graph_with(saver, onto, mock, listed)).ainvoke(
                    Command(resume=approve_all(payload)), config=config)

    state = asyncio.run(go())
    (out,) = state["outputs"]
    assert {a["status"] for a in out["actions"] if a["action_id"] in ("A10", "A15")} == {"done"}
    assert model.calls.count("procedure") == 1  # 이어갈 때 앞 단계를 다시 돌지 않는다


def test_demo_command_end_to_end_with_mcp_processes(db, seed, tmp_path, monkeypatch, capsys):
    """데모 명령이 실제 MCP 서버 프로세스, SQLite 체크포인트, 승인 판단으로 끝까지 돈다. LLM만 규칙 모델로 바꾼다."""
    from contextlib import contextmanager

    from agents.diagnosis import demo
    from tests.conftest import test_db_uri

    monkeypatch.setenv("NEO4J_URI", test_db_uri())
    path = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "dev" / "SC-S1-01.yaml"
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = RuleModel(intake_of(scenario))
    model.verdicts, model.conditions = {"C02": "confirm"}, []

    @contextmanager
    def test_driver():
        yield db

    monkeypatch.setattr(demo, "get_driver", test_driver)
    monkeypatch.setattr(demo.Tracer, "from_env", classmethod(lambda cls: MemoryTracer()))
    monkeypatch.setattr(demo, "make_anthropic_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=model.create)))
    result = asyncio.run(demo.run_demo(scenario, path, org="org-a", config="C", decide=approve_all,
                                       checkpoints=tmp_path / "checkpoints.sqlite"))
    (out,) = result["outputs"]
    assert {a["action_id"]: a["status"] for a in out["actions"]}["A15"] == "done"
    assert (tmp_path / "checkpoints.sqlite").exists()
    # dev가 아닌 시나리오는 명령에서 거부한다
    test_path = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "test" / "SC-T7-01.yaml"
    assert demo.main(["--scenario", str(test_path), "--org", "org-b"]) == 1
    assert "dev 시나리오만" in capsys.readouterr().err


@pytest.mark.parametrize("applied", ["approved", "done"])
def test_resume_after_decisions_were_already_written_to_the_db(db, seed, applied):
    """판단을 DB에 반영한 뒤 체크포인트 저장 전에 끝난 경우 (PR #41 리뷰). 이어갈 때 DB 상태와 맞춰 같은 조치를
    다시 묻지 않고 끝난다. 승인까지 된 실행은 완료로 남긴다."""
    from common.gate import gate

    scenario = load("SC-S1-01")
    tracer = MemoryTracer()

    def written_then_crashed(payload):
        for r in payload["pending"]:
            for team in r["required_approver_teams"]:
                gate.approve(db, tracer=tracer, run_id=r["run_id"], team_id=team, now=datetime.now(timezone.utc))
            if applied == "done":
                gate.mark_done(db, tracer=tracer, run_id=r["run_id"], now=datetime.now(timezone.utc))
        return approve_all(payload)  # 같은 판단이 다시 들어온다

    state, paused = run_demo(db, seed, scenario, RuleModel(intake_of(scenario)), {"C02": "confirm"}, written_then_crashed)
    assert len(paused) == 1
    (out,) = state["outputs"]
    assert {a["status"] for a in out["actions"] if a["action_id"] in ("A10", "A15")} == {"done"}


@pytest.mark.parametrize("bad", ["approve", ["approve"], {"run_id": ["x"], "team_id": "TM06", "decision": "approve"},
                                 {"run_id": "x"}, [None, 3]])
def test_malformed_decisions_are_reported_not_raised(db, seed, bad):
    """판단의 모양이 틀려도 예외로 멈추지 않고 다음 멈춤에 오류로 돌려준다 (PR #41 리뷰)."""
    scenario = load("SC-S1-01")
    calls = []

    def bad_first(payload):
        calls.append(payload)
        return bad if len(calls) == 1 else approve_all(payload)

    state, paused = run_demo(db, seed, scenario, RuleModel(intake_of(scenario)), {"C02": "confirm"}, bad_first)
    assert len(paused) == 2 and paused[1]["errors"]
    (out,) = state["outputs"]
    assert {a["status"] for a in out["actions"] if a["action_id"] in ("A10", "A15")} == {"done"}


def test_refused_requests_are_kept_in_demo_output(db, seed, monkeypatch):
    """게이트가 거부한 요청은 실행 기록 ID가 없다. 둘 이상이어도 승인 대기를 거친 출력에 모두 남는다 (PR #41 리뷰)."""
    from common.gate import gate
    from common.gate.gate import GateError

    original = gate.request_action

    def refuse_some(driver, **kwargs):
        if kwargs["action_id"] in ("A01", "A02"):
            raise GateError(f"테스트 거부 {kwargs['action_id']}")
        return original(driver, **kwargs)

    monkeypatch.setattr(gate, "request_action", refuse_some)
    scenario = load("SC-S1-01")
    state, _ = run_demo(db, seed, scenario, RuleModel(intake_of(scenario)), {"C02": "confirm"}, approve_all)
    (out,) = state["outputs"]
    refused = sorted(a["action_id"] for a in out["actions"] if a["status"] == "refused")
    assert refused == ["A01", "A02"]
    assert {a["status"] for a in out["actions"] if a["action_id"] in ("A10", "A15")} == {"done"}
