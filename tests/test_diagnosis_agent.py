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


def run_agent(db, seed, scenario: dict, model: RuleModel, verdicts: dict, conditions=(), tool_texts=None):
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
                driver=db, llm=llm, tracer=tracer, knowledge=GraphKnowledge(onto, tracer),
                mock_client=TextOverride(mock, tool_texts) if tool_texts else mock,
                mock_tools=[{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                            for t in listed.tools],
                tool_checks=queries.get_evidence_tools(db),
                symptoms=queries.get_symptom_types(db), systems=queries.get_systems(db),
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

def test_moves_to_next_hypothesis_and_caps(db, seed):
    """1순위가 배제되면 다음 후보로 넘어가고, 확정이 안 되면 상한(3개)까지만 확인한다 (R4, 가정값)."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C01": "confirm"})
    (out,) = state["outputs"]
    assert [h["status"] for h in out["hypotheses"][:2]] == ["ruled_out", "confirmed"]
    assert out["root_cause"] == "C01"

    model = RuleModel(intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    (out,) = state["outputs"]
    # 후보 3개를 모두 확인하고 멈춘다. 가설마다 도구 요청 1번, 판단 1번
    assert out["root_cause"] is None and model.calls.count("check") == 3 * 2
    assert sum(1 for h in out["hypotheses"] if h["result"]) == 3


def test_unconfirmed_assigns_top_two_teams(db, seed):
    """원인이 확정되지 않으면 상위 2개 가설의 담당 팀을 함께 배정하고 사람 확인을 요청한다 (R5-4)."""
    scenario = load("SC-S1-01")
    state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario)), verdicts={"C02": "inconclusive", "C01": "inconclusive", "C03": "inconclusive"})
    (out,) = state["outputs"]
    assert out["root_cause"] is None
    assert out["primary_team"] == "TM03"  # 1순위 C02의 담당 (서버개발팀)
    assert [c["team_id"] for c in out["collaborators"]] == ["TM09"]  # 2순위 C01의 담당 (플랫폼팀)
    assert out["next_check"]["cause_id"] == "C02"
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
    from agents.diagnosis.run import check_dev_only
    path = tmp_path / "SC-T.yaml"
    path.write_text("split: test\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="dev 시나리오만"):
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
    """같은 근거(E06)를 인자만 다르게 두 번 조회했으면, 다음 가설에는 결과 ID로 고른 결과만 근거로 잇는다."""
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    original = model.check
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
            return text({"result": "rule_out", "used_checks": ["E06"], "reused_results": [], "reason": "x"})
        if seen.get("pick_cause", cause) == cause:
            # 이 가설의 필수 근거가 E06이 아니어서 한 번 다시 묻는다. 다시 물어도 같은 답을 한다
            seen["pick_cause"] = cause
            if "picked" not in seen:
                seen["cached"] = [line for line in first.splitlines() if line.startswith("- K") and " E06 " in line]
                seen["picked"] = next(line.split()[1] for line in seen["cached"] if '"SY18"' in line)
            return text({"result": "confirm", "used_checks": ["E06"], "reused_results": [seen["picked"]],
                         "reason": "앞의 SY18 배포 이력"})
        return original(params)

    model.check = check
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    assert len(seen["cached"]) == 2  # 같은 근거를 인자만 다르게 조회한 결과가 둘 있다
    (out,) = state["outputs"]
    second = next(h for h in out["hypotheses"] if h["cause_id"] != seen["first"] and h["findings"])
    assert [(f["check_id"], f["reused"], f["args"]) for f in second["findings"]] == [("E06", True, {"system": "SY18"})]
    # 그래프의 Finding에도 그 조회의 인자가 남는다
    with db.session() as session:
        stored = session.run("MATCH (f:Finding {cause_id: $c}) RETURN f.tool AS t, f.args AS a",
                             c=second["cause_id"]).data()
    assert [(r["t"], json.loads(r["a"])) for r in stored] == [("query_deploy_history", {"system": "SY18"})]


def test_unknown_reused_result_id_is_inconclusive(db, seed):
    scenario = load("SC-S1-01")
    model = RuleModel(intake_of(scenario))
    model.check = lambda params: text({"result": "confirm", "used_checks": [], "reused_results": ["K99"], "reason": "x"})
    state, _ = run_agent(db, seed, scenario, model, verdicts={})
    assert state["outputs"][0]["hypotheses"][0]["result"] == "inconclusive"


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
