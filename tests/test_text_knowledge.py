"""구성 A(전체 텍스트 지식 공급자) 검증.

- K 텍스트: 그래프 적재 데이터와 같은 사실, 검증기 전용 값(when) 제외, 같은 시드면 같은 문자열
- 공급자: 출력 검증, 다시 묻기, 실패 기록, 캐시 표시, 호출 기록 (가짜 LLM, Neo4j 없음)
- 통합: 그래프 쿼리 결과를 그대로 답하는 가짜 LLM이면 구성 A가 구성 C와 같은 진단 결과를 낸다 (Neo4j 필요)
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from common.knowledge import text_prompts as P
from common.knowledge.k_text import k_text
from common.knowledge.base import KnowledgeUnavailable
from common.knowledge.text import TextKnowledge
from common.models import LLM, load_model_config
from mcp_servers.ontology_query import queries
from ontology.load import build_graph
from ontology.seed import load_seed
from tests.memory_tracer import MemoryTracer
from tests.test_diagnosis_agent import RuleModel, db, intake_of, load, message, run_agent, seed, text  # noqa: F401

SEED = load_seed("org-a")


# --- K 텍스트 -----------------------------------------------------------------

def test_k_text_is_stable_and_covers_graph():
    first, second = k_text(SEED), k_text(load_seed("org-a"))
    assert first == second
    graph = build_graph(SEED)
    for rows in graph.nodes.values():
        for node in rows:
            assert f"- {node['id']}: " in first
    rel_count = sum(len(rows) for rows in graph.rels.values())
    assert first.count("  - ") == rel_count


def test_k_text_leaves_out_validator_only_values():
    """판정 근거의 when(관측값 조건)은 그래프에 넣지 않는 검증기 전용 값이다. C의 LLM도 보지 못한다."""
    body = k_text(SEED)
    assert "when" not in body and "source_doc_ids" not in body
    assert "expect=" in body and "required=True" in body


# --- 공급자 (가짜 LLM) ----------------------------------------------------------

class Script:
    """정해 둔 답을 차례로 돌려주는 가짜 Messages API."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.params: list[dict] = []

    def create(self, **params):
        self.params.append(params)
        answer = self.answers.pop(0)
        return message([{"type": "text", "text": answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)}])


def provider(script: Script, changes=()) -> tuple[TextKnowledge, list]:
    results: list = []
    llm = LLM(load_model_config("org-a"), MemoryTracer(), SimpleNamespace(messages=script))
    return TextKnowledge(llm=llm, seed=SEED, results=results, changes=lambda: list(changes)), results


CHANGES = [{"id": "CG-0", "kind": "deploy", "system_id": "SY13", "changed_at": "2026-12-03T19:40:00+09:00"}]


def test_rank_converts_answer_and_computes_gap_in_code():
    script = Script({"candidates": [
        {"cause_id": "C03", "score": 3, "rank": 2, "change_ids": [], "reason": "x"},
        {"cause_id": "C02", "score": 5, "rank": 1, "change_ids": ["CG-0"], "reason": "y"},
    ]})
    kp, results = provider(script, CHANGES)
    rows = asyncio.run(kp.rank_causes("S1", "2026-12-03T20:14:00+09:00"))
    assert [r["cause_id"] for r in rows] == ["C02", "C03"]
    assert rows[0]["recent_changes"] == [{"id": "CG-0", "kind": "deploy", "changed_at": CHANGES[0]["changed_at"],
                                          "gap_minutes": 34}]
    # 변경 이력은 질문에 넣고, K는 캐시를 건 system 블록 하나로 보낸다
    params = script.params[0]
    assert "CG-0: 시스템 SY13" in params["messages"][0]["content"]
    (block,) = params["system"]
    assert block["cache_control"] == {"type": "ephemeral"} and block["text"].startswith(P.SYSTEM)
    assert len(results) == 1 and kp.errors == []


def test_undefined_id_is_asked_again_then_recorded():
    bad = {"candidates": [{"cause_id": "C02", "score": 3, "rank": 1, "change_ids": ["CG-9"], "reason": "x"}]}
    kp, results = provider(Script(bad, bad))
    with pytest.raises(KnowledgeUnavailable):
        asyncio.run(kp.rank_causes("S1", "2026-12-03T20:14:00+09:00"))
    assert len(results) == 2 and "CG-9" in kp.errors[0]
    # 실패도 기억한다. 같은 질문을 다시 묻지 않는다
    with pytest.raises(KnowledgeUnavailable):
        asyncio.run(kp.rank_causes("S1", "2026-12-03T20:14:00+09:00"))
    assert len(results) == 2 and len(kp.errors) == 1


def test_wrong_rank_order_is_asked_again():
    script = Script(
        {"candidates": [{"cause_id": "C02", "score": 3, "rank": 2, "change_ids": [], "reason": "x"}]},
        {"candidates": [{"cause_id": "C02", "score": 3, "rank": 1, "change_ids": [], "reason": "x"}]},
    )
    kp, results = provider(script)
    rows = asyncio.run(kp.rank_causes("S1", "2026-12-03T20:14:00+09:00"))
    assert rows[0]["rank"] == 1 and len(results) == 2
    assert "순위" in script.params[1]["messages"][-1]["content"]


def test_checks_args_and_same_question_is_not_asked_twice():
    answer = {"checks": [{"check_id": "E04", "tool": "query_error_log", "required": True, "expect": "오류 로그",
                          "note": "", "performed_by": "", "judged_by_team_id": "", "performed_by_team_id": "",
                          "args": [{"name": "system", "value": "SY15"}]}]}
    tool = next(c["tool"] for c in SEED["evidence_checks"] if c["id"] == "E04")
    answer["checks"][0]["tool"] = tool
    kp, results = provider(Script(answer))
    first = asyncio.run(kp.get_checks("C12"))
    first[0]["args"]["system"] = "바뀜"  # 부르는 쪽이 고쳐도 저장한 값은 그대로다
    second = asyncio.run(kp.get_checks("C12"))
    assert second[0]["args"] == {"system": "SY15"} and second[0]["note"] is None
    assert len(results) == 1


def test_procedure_failure_is_not_an_empty_procedure():
    """조회에 실패한 절차를 조치가 빈 정상 절차로 돌려주지 않는다 (PR #18 리뷰)."""
    wrong = {"procedure_id": "P2", "human_only": False, "led_by_team_id": "", "steps": [], "actions": []}
    kp, _ = provider(Script(wrong, wrong))
    with pytest.raises(KnowledgeUnavailable, match="P1"):
        asyncio.run(kp.procedure("P1"))
    assert kp.errors and "P1" in kp.errors[0]


def test_malformed_answer_is_asked_again():
    kp, results = provider(Script({"wrong": 1}, "not json"))
    with pytest.raises(KnowledgeUnavailable):
        asyncio.run(kp.owner_team("C02"))
    assert len(results) == 2 and len(kp.errors) == 1


def test_missing_field_read_by_conversion_is_asked_again():
    """ID는 맞는데 변환 코드가 읽는 scope가 빠진 답. 예외로 멈추지 않고 다시 묻는다 (PR #18 리뷰)."""
    partial = {"system_id": "SY13", "area": "", "primary": ["TM03"], "other_owners": [], "collaborators": []}
    script = Script(partial, partial | {"scope": "logic"})
    kp, results = provider(script)
    owner = asyncio.run(kp.owner_team("C02"))
    assert owner["located_in"]["scope"] == "logic" and len(results) == 2
    assert "scope" in script.params[1]["messages"][-1]["content"]


# --- 통합: 그래프 결과를 그대로 답하는 가짜 LLM ---------------------------------------

class OracleModel(RuleModel):
    """지식 질문에는 구성 C의 고정 쿼리 결과를 그대로 답하고, 나머지 단계는 RuleModel로 답한다.

    텍스트를 완벽하게 읽은 LLM과 같다. 이때 구성 A의 결과가 구성 C와 같아야 공급자의 변환이 맞다.
    """

    def __init__(self, graph_db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = graph_db
        self.team_id = {t["name"]: t["id"] for t in SEED["teams"]}

    def create(self, **params):
        system = params["system"]
        if isinstance(system, list) and system[0]["text"].startswith(P.SYSTEM):
            self.calls.append("knowledge")
            return text(self.knowledge(params["messages"][0]["content"]))
        return super().create(**params)

    def _proc(self, p: dict, with_condition: bool) -> dict:
        row = {"procedure_id": p["procedure_id"], "human_only": p["human_only"],
               "led_by_team_id": self.team_id.get(p["led_by"], ""), "steps": p["steps"] or [],
               "actions": [{"action_id": a["action_id"], "risk": a["risk"]} for a in p["actions"]]}
        if with_condition:
            row["condition"] = p.get("condition") or ""
        return row

    def knowledge(self, question: str) -> dict:
        if m := re.match(r"증상 (S\d+), 증상 시작 (\S+)\.", question):
            rows = queries.rank_causes(self.db, m.group(1), m.group(2))
            return {"candidates": [{"cause_id": r["cause_id"], "score": r["score"], "rank": r["rank"],
                                    "change_ids": [c["id"] for c in r["recent_changes"]], "reason": "규칙"} for r in rows]}
        if m := re.match(r"원인 (C\d+)의 판정 근거", question):
            return {"checks": [{
                "check_id": c["check_id"], "tool": c["tool"], "required": c["required"], "expect": c["expect"] or "",
                "note": c["note"] or "", "performed_by": c["performed_by"] or "",
                "judged_by_team_id": c["judged_by_team_id"] or "", "performed_by_team_id": c["performed_by_team_id"] or "",
                "args": [{"name": k, "value": v} for k, v in c["args"].items()],
            } for c in queries.get_checks(self.db, m.group(1))]}
        if m := re.match(r"원인 (C\d+)의 담당 팀", question):
            o = queries.get_owner_team(self.db, m.group(1))
            return {"system_id": o["located_in"]["system_id"], "scope": o["located_in"]["scope"],
                    "area": o["located_in"]["area"] or "", "primary": [t["team_id"] for t in o["primary"]],
                    "other_owners": [{"team_id": t["team_id"], "scope": w["scope"], "area": w["area"] or ""}
                                     for t in o["other_owners"] for w in t["owns"]],
                    "collaborators": [{"team_id": t["team_id"], "condition": t["condition"]} for t in o["collaborators"]]}
        if m := re.match(r"원인 (C\d+)의 해결 절차와 심각도 (SEV\d)", question):
            found = queries.get_procedures(self.db, m.group(1), m.group(2))
            return {"resolved_by": [self._proc(p, True) for p in found["resolved_by"]],
                    "triggered_by_severity": [self._proc(p, False) for p in found["triggered_by_severity"]]}
        if m := re.match(r"절차 (P\d+)를", question):
            return self._proc(queries.get_procedure(self.db, m.group(1)), False)
        raise AssertionError(f"모르는 질문이다: {question}")


def _text_provider(graph_db):
    def make(onto, tracer, llm, results):
        from common.writers import events
        return TextKnowledge(llm=llm, seed=SEED, results=results, changes=lambda: events.list_changes(graph_db))
    return make


KEYS = ("symptom", "severity", "root_cause", "primary_team", "procedures", "resolved_conditions", "needs_human",
        "collaborators", "next_check")


@pytest.mark.integration
@pytest.mark.parametrize("name,systems,verdicts", [
    ("SC-S1-01", ("SY12", "SY13"), {"C02": "confirm"}),
    ("SC-S7-01", ("SY14",), {"C18": "confirm"}),
    ("SC-S1-01", ("SY12", "SY13"), {}),  # 확정되지 않으면 상위 2개 가설의 담당 팀 (R5-4)
])
def test_text_provider_matches_graph_provider_with_perfect_reading(db, seed, name, systems, verdicts):  # noqa: F811
    scenario = load(name)
    graph_state, _ = run_agent(db, seed, scenario, RuleModel(intake_of(scenario, systems=systems)), verdicts=verdicts)
    model = OracleModel(db, intake_of(scenario, systems=systems))
    text_state, _ = run_agent(db, seed, scenario, model, verdicts=verdicts, knowledge=_text_provider(db))

    (c_out,), (a_out,) = graph_state["outputs"], text_state["outputs"]
    for key in KEYS:
        assert a_out[key] == c_out[key], key
    strip = lambda hs: [(h["cause_id"], h["rank"], h["score"], h["status"]) for h in hs]  # noqa: E731
    assert strip(a_out["hypotheses"]) == strip(c_out["hypotheses"])
    assert [(a["action_id"], a["status"]) for a in a_out["actions"]] == [(a["action_id"], a["status"]) for a in c_out["actions"]]
    assert "knowledge" in model.calls


@pytest.mark.integration
def test_no_candidates_goes_to_human(db, seed):  # noqa: F811
    """원인 후보를 받지 못하면 근거 확인을 건너뛰고 사람에게 넘긴다."""
    scenario = load("SC-S1-01")

    def make(onto, tracer, llm, results):
        class Empty(TextKnowledge):
            async def rank_causes(self, symptom_id, started_at):
                return []
        return Empty(llm=llm, seed=SEED, results=results, changes=lambda: [])

    model = OracleModel(db, intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={}, knowledge=make)
    (out,) = state["outputs"]
    assert out["hypotheses"] == [] and out["root_cause"] is None and out["primary_team"] is None
    assert "원인 후보를 받지 못했다" in out["needs_human"]
    assert not any(c["step"] == "7.check" for c in out["tool_calls"])


def _failing(method: str, *, procedure_id: str | None = None):
    """한 메서드만 답하지 못하는 텍스트 공급자. 나머지는 가짜 LLM(OracleModel)이 그래프 결과로 답한다."""
    def make(onto, tracer, llm, results):
        from common.writers import events

        class Failing(TextKnowledge):
            pass

        original = getattr(TextKnowledge, method)

        async def fail(self, *args):
            if procedure_id is None or args[0] == procedure_id:
                raise KnowledgeUnavailable(f"{method} 실패")
            return await original(self, *args)
        setattr(Failing, method, fail)
        return Failing(llm=llm, seed=SEED, results=results, changes=lambda: events.list_changes(_DB[0]))
    return make


_DB: list = []


@pytest.mark.integration
def test_unavailable_checks_never_confirm(db, seed):  # noqa: F811
    """판정 근거를 받지 못하면 LLM이 확정하려 해도 확정하지 않고 사람에게 넘긴다 (PR #18 리뷰)."""
    _DB[:] = [db]
    scenario = load("SC-S1-01")
    model = OracleModel(db, intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"}, knowledge=_failing("get_checks"))
    (out,) = state["outputs"]
    assert out["root_cause"] is None
    assert all(h["status"] == "candidate" for h in out["hypotheses"] if h["result"])
    assert any("판정 근거를 받지 못했다" in n for n in out["needs_human"])
    assert not any(c["step"] == "7.check" for c in out["tool_calls"])
    assert "check" not in model.calls


@pytest.mark.integration
def test_no_required_check_never_confirms(db, seed):  # noqa: F811
    """필수 판정 근거가 하나도 없는 목록이면 확정 조건이 없다. 조회 없이 확정되지 않는다 (모든 구성 공통)."""
    _DB[:] = [db]
    scenario = load("SC-S1-01")

    def make(onto, tracer, llm, results):
        from common.writers import events

        class NoRequired(TextKnowledge):
            async def get_checks(self, cause_id):
                return [dict(c, required=False) for c in await TextKnowledge.get_checks(self, cause_id)]
        return NoRequired(llm=llm, seed=SEED, results=results, changes=lambda: events.list_changes(db))

    model = OracleModel(db, intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"}, knowledge=make)
    (out,) = state["outputs"]
    assert out["root_cause"] is None
    assert any("확정 조건" in n for n in out["needs_human"])


@pytest.mark.integration
def test_unavailable_procedure_goes_to_human(db, seed):  # noqa: F811
    """절차를 받지 못하면 후보에 넣지 않고 사람 확인 사유로 남긴다. 조치가 빈 절차로 검증을 통과하지 않는다."""
    _DB[:] = [db]
    scenario = load("SC-S1-01")
    model = OracleModel(db, intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"},
                         knowledge=_failing("procedure", procedure_id="P1"))
    (out,) = state["outputs"]
    # P1(접수)은 절차 조회로만 후보가 된다. 받지 못하면 항상 여는 절차여도 후보에서 빠진다
    assert out["root_cause"] == "C02" and "P1" not in out["procedures"]
    assert any("절차 P1를 받지 못했다" in n for n in out["needs_human"])


@pytest.mark.integration
def test_unavailable_owner_goes_to_human(db, seed):  # noqa: F811
    _DB[:] = [db]
    scenario = load("SC-S1-01")
    model = OracleModel(db, intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"}, knowledge=_failing("owner_team"))
    (out,) = state["outputs"]
    assert out["primary_team"] is None
    assert any("담당 팀을 받지 못했다" in n for n in out["needs_human"])


@pytest.mark.integration
def test_unavailable_rank_goes_to_human(db, seed):  # noqa: F811
    _DB[:] = [db]
    scenario = load("SC-S1-01")
    model = OracleModel(db, intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={}, knowledge=_failing("rank_causes"))
    (out,) = state["outputs"]
    assert out["hypotheses"] == [] and any("원인 후보를 받지 못했다" in n for n in out["needs_human"])


@pytest.mark.integration
def test_required_procedure_stays_required_through_cause_list(db, seed):  # noqa: F811
    """P10 조회만 실패하고 원인이 여는 절차 목록으로 P10이 들어와도 항상 여는 절차로 남는다 (PR #18 리뷰).

    필수 표시가 빠지면 LLM이 P10을 빼도 검증을 통과한다. 여기서는 P10을 빼는 답을 먼저 주고 다시 묻는지 본다.
    """
    _DB[:] = [db]
    scenario = load("SC-S1-01")

    def drop_p10(params):
        procs = [p for p in json.loads(params["messages"][0]["content"].split("후보 절차:\n", 1)[1])
                 if not p["human_only"] and p["procedure_id"] != "P10"]
        actions = sorted({a["action_id"]: a["risk"] for p in procs for a in p["actions"]}.items())
        return text({"procedures": [p["procedure_id"] for p in procs], "resolved_conditions": ["누락 보상"],
                     "actions": [{"action_id": a, "mode": "approval" if r == "high" else "auto"} for a, r in actions],
                     "reason": "P10을 뺀다"})

    model = OracleModel(db, intake_of(scenario), overrides={"procedure": [drop_p10]})
    state, _ = run_agent(db, seed, scenario, model, verdicts={"C02": "confirm"}, conditions=["누락 보상"],
                         knowledge=_failing("procedure", procedure_id="P10"))
    (out,) = state["outputs"]
    assert model.calls.count("procedure") == 2  # P10을 뺀 첫 답은 거부되고 다시 묻는다
    assert "P10" in out["procedures"]
    assert any("절차 P10를 받지 못했다" in n for n in out["needs_human"])


@pytest.mark.integration
def test_no_procedure_candidates_does_not_ask(db, seed):  # noqa: F811
    """원인이 확정되지 않고 접수 절차도 받지 못하면 후보가 없다. 빈 선택지로 묻지 않고 사람에게 넘긴다 (PR #18 리뷰)."""
    _DB[:] = [db]
    scenario = load("SC-S1-01")
    model = OracleModel(db, intake_of(scenario))
    state, _ = run_agent(db, seed, scenario, model, verdicts={}, knowledge=_failing("procedure"))
    (out,) = state["outputs"]
    assert "procedure" not in model.calls
    assert out["procedures"] == [] and out["actions"] == []
    assert any("절차 후보가 없어" in n for n in out["needs_human"])
