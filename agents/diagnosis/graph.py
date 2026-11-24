"""진단 에이전트 (설계 5단계 4-1, LangGraph).

11단계를 LangGraph 노드로 둔다. 해석이 필요한 단계(1, 4, 7, 8의 조건부 협업, 9)만 LLM이 맡고,
규칙 계산과 안전 장치(2, 3, 5, 6의 점수, 10, 11)는 코드가 한다 (5단계 1장).
회사 지식은 지식 공급자(common.knowledge)를 거쳐 얻는다. 구성마다 공급자만 바꾼다.
LLM은 그래프에 쓰지 않는다. 기록은 common.writers의 함수가 한다.
고위험 조치를 실행하는 함수는 LLM 도구 목록에 없다. LLM이 받는 도구는 판정 근거 조회(모의 도구)뿐이다.

제보 여러 건은 차례로 1~5단계를 거쳐 사건으로 묶이고, 새로 열린 사건마다 6~11단계를 돈다.
"""

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import Client
from neo4j import Driver

from agents.diagnosis import prompts
from common.gate import gate
from common.gate.gate import GateError
from common.knowledge import KnowledgeProvider, KnowledgeUnavailable
from common.models import LLM, LLMResult
from common.rules.severity import ESCALATION_COUNT, ESCALATION_WINDOW, SCOPES, base_severity, escalate
from common.tool_calls import call_tool
from common.tracing import Tracer
from common.writers import events
from mcp_servers.ontology_query import queries

KST = timezone(timedelta(hours=9))

# 가정값 (설계 3단계 4-2, 4-3, 11장). 4단계 평가에서 조정한다
DUPLICATE_WINDOW = timedelta(hours=6)          # R2 비교 시간창
# 가정값 (이 PR에서 정함, 설계에 값이 없다). 확정이 안 되면 다음 후보로 넘어가는 횟수의 상한
MAX_HYPOTHESES = 3
# 한 가설을 확인하며 도구를 부르는 왕복의 상한. 끝없는 도구 호출을 막는다
MAX_TOOL_ROUNDS = 8

URGENT_LEVELS = {"SEV1", "SEV2"}
INTAKE_PROCEDURE, ESCALATION_PROCEDURE = "P1", "P2"
URGENT_NOTICE, URGENT_MAINTENANCE, FIX_PROCEDURE = "P3", "P4", "P10"
MERGE_REPLY_ACTION = "A04"  # 처리 중 회신. 같은 사건으로 묶은 제보에 보낸다 (R2 표)
SCOPE_CHECK = "E15"  # 접수 내역 조회. 영향 범위 추정의 근거 (3단계 6장, R3)
CAUSE_FIX_ACTION = "A15"
# R7: 원인 수정(A15)의 반영 시점. 시드의 승인 조건 값과 같아야 한다
FIX_TIMING = {True: "긴급 점검 반영", False: "정기 점검 반영"}


@dataclass
class Deps:
    """그래프 노드가 쓰는 의존 객체. 실행마다 만든다."""

    driver: Driver
    llm: LLM
    tracer: Tracer
    knowledge: KnowledgeProvider
    mock_client: Client
    mock_tools: list[dict[str, Any]]       # LLM에 주는 판정 근거 조회 도구 (Anthropic 도구 형식)
    tool_checks: dict[str, str]            # 도구 이름 -> 판정 근거 ID
    symptoms: list[dict[str, Any]]
    systems: list[dict[str, Any]]
    now: Callable[[], datetime] = lambda: datetime.now(KST)
    llm_results: list[LLMResult] = field(default_factory=list)


class State(TypedDict, total=False):
    reports: list[dict]       # 입력 제보 (channel, reporter, at, text)
    index: int                # 지금 처리 중인 제보 번호
    report: dict              # 지금 처리 중인 제보의 작업 값
    incidents: dict           # incident_id -> 사건 요약
    queue: list               # 진단할 새 사건
    diag: dict                # 지금 진단 중인 사건의 작업 값
    outputs: list             # 사건별 구조화 출력 (4단계 3-3)
    unclassified: list        # 증상 8개에 해당하지 않아 사람에게 넘긴 제보


def initial_state(reports: list[dict]) -> State:
    """그래프 입력. 결과 목록은 빈 목록으로 시작한다 (사건이 하나도 열리지 않아도 키가 있다).

    제보는 접수 시각 순서로 처리한다. 실제 접수도 시각 순서로 들어오고, 중복 후보(R2)와 상향 창(R3)이
    이 순서를 전제로 한다. 입력 순서가 달라도 결과가 같아야 한다. 원래 순서의 번호(index)는 채점용으로 남긴다
    """
    ordered = sorted(({**r, "index": i} for i, r in enumerate(reports)), key=lambda r: (_parse_at(r["at"]), r["index"]))
    return {"reports": ordered, "index": 0, "incidents": {}, "queue": [], "outputs": [], "unclassified": []}


def _parse_at(value: str) -> datetime:
    at = datetime.fromisoformat(value)
    if at.utcoffset() is None:
        raise ValueError(f"시간대가 없는 시각이다: {value}")
    return at


def _report_block(report: dict) -> str:
    return f'<report channel="{report["channel"]}" reporter="{report["reporter"]}" at="{report["at"]}">\n{report["text"]}\n</report>'


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --- LLM 호출 도우미 ---------------------------------------------------------

def _ask_json(
    deps: Deps,
    *,
    task: str,
    name: str,
    system: str,
    content: str,
    schema: dict,
    validate: Callable[[dict], list[str]],
) -> tuple[dict | None, list[str]]:
    """JSON 형식 답을 받고 검증한다 (5단계 4-1 출력 검증).

    정의되지 않은 값이 나오면 문제를 알려 한 번 다시 묻는다. 그래도 틀리면 (None, 문제 목록)을 돌려준다.
    부르는 쪽은 이때 사람 확인으로 넘긴다.
    """
    messages: list[dict] = [{"role": "user", "content": content}]
    errors: list[str] = []
    for attempt in range(2):
        result = deps.llm.call(
            task=task, name=name if attempt == 0 else f"{name}.retry", system=system,
            messages=messages, output_schema=schema,
        )
        deps.llm_results.append(result)
        try:
            data = result.json()
        except ValueError:
            data, errors = None, [f"JSON으로 읽을 수 없는 답이다 (stop_reason={result.message.stop_reason})"]
        else:
            errors = _structure_errors(data, schema) or validate(data)
        if not errors:
            return data, []
        messages += [
            {"role": "assistant", "content": result.text or "(빈 답)"},
            {"role": "user", "content": prompts.RETRY.format(errors="\n".join(f"- {e}" for e in errors))},
        ]
    return None, errors


def _structure_errors(data: Any, schema: dict) -> list[str]:
    """JSON을 읽은 뒤 단계별 검증 전에 모양을 본다. 객체가 아니거나 필수 필드가 없거나 타입이 다르면 문제다.

    JSON 형식 출력은 API가 보장하지만, 읽기에 성공한 것만으로 필드와 타입이 맞다고 볼 수 없다.
    단계별 검증기가 모양이 틀린 값에서 예외를 내지 않게 여기서 먼저 막는다.
    """
    if not isinstance(data, dict):
        return [f"답은 JSON 객체여야 한다: {type(data).__name__}"]
    types = {"string": str, "array": list, "object": dict}
    errors = []
    for name in schema.get("required", []):
        spec = schema["properties"][name]
        if name not in data:
            errors.append(f"{name} 필드가 없다")
        elif spec.get("type") in types and not isinstance(data[name], types[spec["type"]]):
            errors.append(f"{name}은 {spec['type']}이어야 한다")
        elif spec.get("type") == "array":
            item_type = spec.get("items", {}).get("type")
            for item in data[name]:
                if item_type == "object":
                    errors += [f"{name}의 항목: {e}" for e in _structure_errors(item, spec["items"])]
                elif item_type in types and not isinstance(item, types[item_type]):
                    errors.append(f"{name}의 항목은 {item_type}이어야 한다: {item!r}")
    return errors


def _flag(state: dict, incident_id: str, reason: str) -> dict:
    """사건에 사람 확인 사유를 하나 더한 incidents를 돌려준다 (상태를 직접 고치지 않는다)."""
    incidents = dict(state["incidents"])
    incident = dict(incidents[incident_id])
    incident["needs_human"] = [*incident["needs_human"], reason]
    incidents[incident_id] = incident
    return incidents


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or list(properties),
        "additionalProperties": False,
    }


# --- 그래프 -------------------------------------------------------------------

def build_graph(deps: Deps):
    symptom_by_id = {s["id"]: s for s in deps.symptoms}
    system_ids = [s["id"] for s in deps.systems]

    # 1 접수 정리 (LLM, R1)
    async def intake(state: State) -> State:
        report = state["reports"][state["index"]]
        catalog = "\n".join(
            f"- {s['id']} {s['name']} (유사 표현: {', '.join(s['aliases'])}; 드러나는 시스템: "
            f"{', '.join(o['id'] for o in s['observed_in'])})"
            for s in deps.symptoms
        )
        systems = "\n".join(f"- {s['id']} {s['name']}" for s in deps.systems)
        schema = _schema({
            "symptom_id": {"type": "string", "enum": [*symptom_by_id, "none"]},
            "scope": {"type": "string", "enum": sorted(SCOPES)},
            "system_ids": {"type": "array", "items": {"type": "string", "enum": system_ids}},
            "reason": {"type": "string"},
        })

        def validate(data: dict) -> list[str]:
            errors = []
            if data.get("symptom_id") not in [*symptom_by_id, "none"]:
                errors.append(f"정의되지 않은 증상이다: {data.get('symptom_id')}")
            if data.get("scope") not in SCOPES:
                errors.append(f"정의되지 않은 영향 범위다: {data.get('scope')}")
            unknown = sorted(set(data.get("system_ids") or []) - set(system_ids))
            if unknown:
                errors.append(f"정의되지 않은 시스템이다: {unknown}")
            if data.get("symptom_id") != "none" and not data.get("system_ids"):
                errors.append("증상이 있으면 관련 시스템을 하나 이상 적는다")
            return errors

        # 영향 범위의 근거: 접수 내역 조회(E15). 판단이 아니라 조회라서 코드가 부르고, 범위는 LLM이 정한다 (3단계 6장, R3)
        lookup_tool = next(tool for tool, check in deps.tool_checks.items() if check == SCOPE_CHECK)
        lookup = await call_tool(deps.mock_client, deps.tracer, server="mock_tools", name=lookup_tool, args={})
        lookup_text = lookup.text if not lookup.is_error else "접수 내역을 조회하지 못했다"
        content = (f"증상 유형:\n{catalog}\n\n시스템:\n{systems}\n\n제보:\n{_report_block(report)}"
                   f"\n\n접수 내역 조회({SCOPE_CHECK}) 결과: {lookup_text}")
        data, errors = _ask_json(
            deps, task="intake", name="1.intake", system=prompts.INTAKE, content=content,
            schema=schema, validate=validate,
        )
        at = _parse_at(report["at"])
        work = {
            "report_id": _new_id("RP"), "report_index": report.get("index"),
            "channel": report["channel"], "reporter": report["reporter"],
            "at": at, "text": report["text"], "needs_human": [],
            "scope_lookup": {"cause_id": None, "step": "1.intake", "tool": lookup_tool, "check_id": SCOPE_CHECK,
                             "args": {}, "is_error": lookup.is_error, "text": lookup.text},
        }
        if data is None:
            work |= {"symptom_id": None, "scope": None, "system_ids": [], "intake_reason": None}
            work["needs_human"].append(f"접수 정리 출력 검증 실패: {errors}")
        else:
            symptom = None if data["symptom_id"] == "none" else data["symptom_id"]
            work |= {
                "symptom_id": symptom, "scope": data["scope"] if symptom else None,
                "system_ids": sorted(set(data["system_ids"])) if symptom else [], "intake_reason": data["reason"],
            }
        events.record_report(
            deps.driver, report_id=work["report_id"], channel_id=work["channel"], reporter_id=work["reporter"],
            raw_text=work["text"], received_at=at, symptom_id=work["symptom_id"], scope=work["scope"],
        )
        return {"report": work}

    # 2 1차 처리 역할 조회 (코드, R1)
    async def first_handler(state: State) -> State:
        work = dict(state["report"])
        work["first_handler"] = queries.get_first_handler(deps.driver, work["channel"])
        return {"report": work}

    # 3 중복 후보 찾기 (코드, R2 시간창과 같은 요청자 병합)
    async def duplicate_candidates(state: State) -> State:
        work = dict(state["report"])
        if work["symptom_id"] is None:
            work |= {"decision": "unclassified", "candidates": []}
            return {"report": work}
        candidates = [
            c for c in events.open_incidents(deps.driver, since=work["at"] - DUPLICATE_WINDOW)
            if datetime.fromisoformat(c["opened_at"]) <= work["at"]
        ]
        same_reporter = [c for c in candidates if any(
            r["reporter_id"] == work["reporter"] and r["symptom_id"] == work["symptom_id"] for r in c["reports"])]
        if same_reporter:
            # 같은 요청자가 시간창 안에 같은 증상으로 다시 보낸 제보는 채널이 달라도 같은 사건이다 (R2).
            # 증상이 다르면 다른 문제일 수 있어 묶지 않고 중복 판정으로 넘긴다
            work |= {"decision": "same", "target": same_reporter[0]["incident_id"], "decided_by": "same_reporter"}
        elif not candidates:
            work |= {"decision": "new"}
        work["candidates"] = candidates
        return {"report": work}

    # 4 중복 판정 (LLM, R2 기준). 판정할 후보가 없으면 부르지 않는다
    async def duplicate_judge(state: State) -> State:
        work = dict(state["report"])
        if "decision" in work:
            return {}
        ids = [c["incident_id"] for c in work["candidates"]]
        schema = _schema({
            "judgments": {"type": "array", "items": _schema({
                "incident_id": {"type": "string", "enum": ids},
                "verdict": {"type": "string", "enum": ["same", "ambiguous", "different"]},
                "reason": {"type": "string"},
            })},
        })

        def validate(data: dict) -> list[str]:
            got = [j["incident_id"] for j in data.get("judgments", [])]
            return [] if sorted(got) == sorted(ids) else [f"사건 {ids}마다 하나씩 판정한다. 받은 것: {got}"]

        listing = "\n\n".join(
            f"사건 {c['incident_id']} (열린 시각 {c['opened_at']}, 영향 시스템 {c['systems']}, "
            f"증상 {sorted({r['symptom_id'] for r in c['reports'] if r['symptom_id']})})\n"
            + "\n".join(_report_block({"channel": "-", "reporter": r["reporter_id"], "at": r["received_at"], "text": r["text"]})
                        for r in c["reports"])
            for c in work["candidates"]
        )
        content = (
            f"새 제보 (증상 {work['symptom_id']}, 영향 시스템 {work['system_ids']}):\n"
            f"{_report_block({'channel': work['channel'], 'reporter': work['reporter'], 'at': work['at'].isoformat(), 'text': work['text']})}"
            f"\n\n열린 사건:\n{listing}"
        )
        data, errors = _ask_json(
            deps, task="intake", name="4.duplicate", system=prompts.DUPLICATE, content=content,
            schema=schema, validate=validate,
        )
        if data is None:
            work |= {"decision": "ambiguous", "targets": ids}
            work["needs_human"].append(f"중복 판정 출력 검증 실패: {errors}")
            return {"report": work}
        same = [j["incident_id"] for j in data["judgments"] if j["verdict"] == "same"]
        ambiguous = [j["incident_id"] for j in data["judgments"] if j["verdict"] == "ambiguous"]
        if same:
            work |= {"decision": "same", "target": same[0], "decided_by": "llm"}
        elif ambiguous:
            work |= {"decision": "ambiguous", "targets": ambiguous}
        else:
            work |= {"decision": "new"}
        work["duplicate_judgments"] = data["judgments"]
        return {"report": work}

    # 5 심각도 계산 (코드, R3). 새 사건을 열거나 기존 사건에 묶는다
    async def severity(state: State) -> State:
        work = state["report"]
        incidents = dict(state.get("incidents") or {})
        queue = list(state.get("queue") or [])
        unclassified = list(state.get("unclassified") or [])
        decision = work["decision"]

        if decision == "unclassified":
            unclassified.append({
                "report_id": work["report_id"], "report_index": work["report_index"], "symptom": "none",
                "first_handler": work["first_handler"],
                "needs_human": work["needs_human"] or ["증상 8개에 해당하지 않아 사람에게 넘긴다"],
            })
        elif decision == "same":
            incident_id = work["target"]
            events.link_report(deps.driver, report_id=work["report_id"], incident_id=incident_id, link="part_of")
            summary = dict(incidents.get(incident_id) or {"incident_id": incident_id})
            # 같은 사건으로 묶은 제보에는 원래 채널로 "처리 중" 회신을 보낸다 (R2 표, A04). 저위험 자동 실행이다
            # 실행 기록에 대상 접수를 남긴다. 회신 채널은 그 접수의 채널이다
            reply = gate.request_action(deps.driver, tracer=deps.tracer, incident_id=incident_id, action_id=MERGE_REPLY_ACTION,
                                        requested_mode="auto", now=deps.now(), target_report_id=work["report_id"])
            summary["merge_replies"] = [*summary.get("merge_replies", []), {
                "report_id": reply["target_report_id"], "channel": work["channel"], "action_id": reply["action_id"],
                "status": reply["status"], "run_id": reply["run_id"]}]
            summary["report_ids"] = [*summary.get("report_ids", []), work["report_id"]]
            summary["reports"] = [*summary.get("reports", []), work]
            # 같은 사건에 짧은 시간 동안 접수가 몰리면 한 단계 상향한다 (R3, 상향은 한 번만).
            # 창은 이번 제보 시각을 끝으로 하는 30분이다. 입력 순서와 상관없이 이번 제보보다 늦은 접수는 세지 않는다
            recent = [
                r for c in work["candidates"] if c["incident_id"] == incident_id for r in c["reports"]
                if work["at"] - ESCALATION_WINDOW <= datetime.fromisoformat(r["received_at"]) <= work["at"]
            ]
            if len(recent) + 1 >= ESCALATION_COUNT and not summary.get("escalated") and "severity" in summary:
                summary["severity"] = escalate(summary["severity"])
                summary["escalated"] = True
                events.set_incident_severity(deps.driver, incident_id=incident_id, severity_id=summary["severity"])
            incidents[incident_id] = summary
        else:
            symptom = symptom_by_id[work["symptom_id"]]
            sev = base_severity(work["scope"], symptom["core_function"], symptom.get("min_severity"))
            incident_id = _new_id("INC")
            events.record_incident(deps.driver, incident_id=incident_id, severity_id=sev, opened_at=work["at"])
            events.link_report(deps.driver, report_id=work["report_id"], incident_id=incident_id, link="part_of")
            events.record_affects(deps.driver, incident_id=incident_id, system_ids=work["system_ids"])
            needs_human = list(work["needs_human"])
            if decision == "ambiguous":
                # 애매하면 새 사건으로 진단을 이어 가되 중복 의심으로 묶어 사람이 확인하게 한다 (R2)
                for target in work["targets"]:
                    events.link_report(deps.driver, report_id=work["report_id"], incident_id=target, link="suspected_duplicate")
                needs_human.append(f"중복 의심: {work['targets']}")
            incidents[incident_id] = {
                "incident_id": incident_id, "symptom_id": work["symptom_id"], "scope": work["scope"],
                "system_ids": work["system_ids"], "severity": sev, "escalated": False,
                "opened_at": work["at"], "report_ids": [work["report_id"]], "reports": [work],
                "first_handler": work["first_handler"], "needs_human": needs_human,
            }
            queue.append(incident_id)
        return {"incidents": incidents, "queue": queue, "unclassified": unclassified, "index": state["index"] + 1}

    def after_report(state: State) -> str:
        return "intake" if state["index"] < len(state["reports"]) else "next_incident"

    async def next_incident(state: State) -> State:
        queue = list(state.get("queue") or [])
        if not queue:
            return {"diag": None}
        incident_id = queue.pop(0)
        return {"queue": queue, "diag": {"incident_id": incident_id, "hyp_index": 0, "checked": 0,
                                          "tool_results": {}, "tool_calls": [], "verdicts": {}}}

    def after_next(state: State) -> str:
        return "rank" if state.get("diag") else END

    # 6 원인 후보와 순위 (지식 공급자, R4-1~3)
    async def rank(state: State) -> State:
        diag = dict(state["diag"])
        incident = state["incidents"][diag["incident_id"]]
        try:
            ranked = await deps.knowledge.rank_causes(incident["symptom_id"], incident["opened_at"].isoformat())
            problem = None
        except KnowledgeUnavailable as exc:
            ranked, problem = [], f"원인 후보를 받지 못했다: {exc}"
        for row in ranked:
            events.record_hypothesis(
                deps.driver, incident_id=diag["incident_id"], cause_id=row["cause_id"],
                score=row["score"], rank=row["rank"], status="candidate",
            )
            for change in row.get("recent_changes") or []:
                events.record_started_after(
                    deps.driver, incident_id=diag["incident_id"], change_id=change["id"],
                    gap_minutes=int(change["gap_minutes"]),
                )
        diag["ranked"] = ranked
        if not ranked:
            # 원인 후보가 하나도 없으면 근거 확인을 건너뛰고 사람에게 넘긴다.
            # 그래프 공급자는 증상마다 후보가 있어 생기지 않는다. 텍스트 공급자(구성 A)가 답하지 못하면 생긴다
            return {"diag": diag, "incidents": _flag(state, diag["incident_id"], problem or "원인 후보를 받지 못했다")}
        return {"diag": diag}

    def after_rank(state: State) -> str:
        return "check" if state["diag"]["ranked"] else "owner"

    # 7 판정 근거 조회 (LLM이 고르고 모의 도구 호출, R4-4)
    async def check(state: State) -> State:
        diag = dict(state["diag"])
        incident = state["incidents"][diag["incident_id"]]
        hypothesis = diag["ranked"][diag["hyp_index"]]
        cause_id = hypothesis["cause_id"]
        # 확정 조건(필수 판정 근거)이 없으면 이 가설은 확정도 배제도 할 근거가 없다. LLM에 묻지 않고 불명확으로 둔다.
        # 그래프 공급자는 시드 검증으로 원인마다 확정 조건이 있다. 텍스트 공급자(구성 A)가 답하지 못하거나
        # 필수 근거를 하나도 적지 않으면 생긴다. 빈 목록을 그대로 쓰면 조회 없이 확정될 수 있다 (R4-4)
        try:
            checks = await deps.knowledge.get_checks(cause_id)
            problem = None if any(c["required"] for c in checks) else f"{cause_id}의 확정 조건(필수 판정 근거)을 받지 못했다"
        except KnowledgeUnavailable as exc:
            checks, problem = [], f"{cause_id}의 판정 근거를 받지 못했다: {exc}"
        tool_results = dict(diag["tool_results"])
        tool_calls = list(diag["tool_calls"])

        # 앞 가설에서 조회해 둔 결과. 결과 ID로 구분해서, 같은 근거를 다른 인자로 조회한 결과가 섞이지 않게 한다
        # 조회 결과는 호출마다 따로 남긴다. 같은 도구, 같은 인자로 다시 불러도 앞 결과를 덮어쓰지 않는다
        prior_ids = set(tool_results)
        known = "\n".join(f"- {r['result_id']} {r['check_id']} {r['tool']} {json.dumps(r['args'], ensure_ascii=False)}: {r['text']}"
                           for r in tool_results.values())
        content = (
            "사건 제보:\n" + "\n".join(_report_block({**r, "at": r["at"].isoformat()}) for r in incident["reports"])
            + f"\n\n증상 {incident['symptom_id']}, 영향 범위 {incident['scope']}, 증상 시작 {incident['opened_at'].isoformat()}"
            + f"\n\n확인할 원인 가설: {cause_id} {hypothesis['name']} (순위 {hypothesis['rank']}, 점수 {hypothesis['score']}, "
            + f"최근 변경 {hypothesis.get('recent_changes') or '없음'})"
            + f"\n판정 근거:\n{json.dumps(checks, ensure_ascii=False, indent=1)}"
            + (f"\n\n이미 조회한 결과:\n{known}" if known else "")
        )
        schema = _schema({
            "result": {"type": "string", "enum": ["confirm", "rule_out", "inconclusive"]},
            "used_checks": {"type": "array", "items": {"type": "string"}},
            "reused_results": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        })
        messages: list[dict] = [{"role": "user", "content": content}]
        data, called, retried = None, [], False
        required = [(c["check_id"], c["args"]) for c in checks if c["required"]]
        for _ in range(0 if problem else MAX_TOOL_ROUNDS):
            result = deps.llm.call(
                task="diagnosis", name=f"7.check.{cause_id}", system=prompts.CHECK, messages=messages,
                tools=deps.mock_tools, output_schema=schema,
            )
            deps.llm_results.append(result)
            uses = [b for b in result.message.content if b.type == "tool_use"]
            if result.message.stop_reason != "tool_use" or not uses:
                try:
                    data = result.json()
                except ValueError:
                    data = None
                # 필수 판정 근거를 조회하지도, 앞 결과를 인용하지도 않고 답하면 한 번 다시 묻는다 (R4-4)
                missing = _unchecked(data, schema, required, called, tool_results, prior_ids)
                # 다시 물을 때 앞 답의 도구 호출 블록은 뺀다. 결과 없는 도구 호출 뒤에 글을 붙이면 API가 거부한다
                said = [b.model_dump(mode="json", exclude_none=True) for b in result.message.content if b.type != "tool_use"]
                if missing and not retried and said:
                    retried = True
                    messages.append({"role": "assistant", "content": said})
                    messages.append({"role": "user", "content": prompts.RETRY.format(
                        errors=f"- 필수 판정 근거 {missing}를 조회하지 않고 답했다. 조회하거나 이미 조회한 결과를 "
                               "reused_results로 인용한 뒤 다시 답한다")})
                    data = None
                    continue
                break
            messages.append({"role": "assistant", "content": [b.model_dump(mode="json", exclude_none=True)
                                                              for b in result.message.content]})
            answers = []
            for use in uses:
                out = await call_tool(deps.mock_client, deps.tracer, server="mock_tools", name=use.name, args=use.input)
                if not out.is_error:
                    result_id = f"K{len(tool_results) + 1}"
                    tool_results[result_id] = {"result_id": result_id, "tool": use.name, "args": use.input,
                                               "check_id": deps.tool_checks.get(use.name), "text": out.text}
                called.append({"step": "7.check", "tool": use.name, "args": use.input, "check_id": deps.tool_checks.get(use.name),
                               "is_error": out.is_error, "text": out.text})
                answers.append({"type": "tool_result", "tool_use_id": use.id, "content": out.text, "is_error": out.is_error})
            messages.append({"role": "user", "content": answers})

        if problem:
            verdict = {"result": "inconclusive", "reason": problem, "used_checks": [], "reused_results": []}
        elif (data is None or _structure_errors(data, schema)
                or data["result"] not in ("confirm", "rule_out", "inconclusive")
                or not set(data["reused_results"]) <= prior_ids):
            verdict = {"result": "inconclusive", "reason": "판정 근거 단계의 답을 읽을 수 없다",
                       "used_checks": [], "reused_results": []}
        else:
            verdict = data
            # 다시 물어도 필수 근거가 없으면 확정도 배제도 근거가 없다. 불명확으로 둔다 (R4-4)
            missing = _unchecked(data, schema, required, called, tool_results, prior_ids)
            if missing and verdict["result"] != "inconclusive":
                verdict = {**verdict, "result": "inconclusive",
                           "reason": f"필수 판정 근거 {missing}의 결과 없이 {data['result']}로 답했다. 답한 이유: {data['reason']}"}
        status = {"confirm": "confirmed", "rule_out": "ruled_out", "inconclusive": "candidate"}[verdict["result"]]
        events.record_hypothesis(deps.driver, incident_id=diag["incident_id"], cause_id=cause_id,
                                 score=hypothesis["score"], rank=hypothesis["rank"], status=status)
        # 이 가설의 근거: 이번에 새로 부른 성공 결과와, 판단에 썼다고 결과 ID로 답한 앞 가설의 조회 결과.
        # 실제 호출 이력(tool_calls)과 따로 둔다. 재사용한 결과도 이 가설의 Finding과 출력에 남긴다.
        # 근거 ID가 아니라 결과 ID로 잇는다. 같은 근거를 다른 인자로 조회한 결과, 같은 인자로 다시 조회해
        # 달라진 결과가 잘못 연결되지 않게 한다. 인용한 결과는 그대로 잇는다
        # 이 원인의 판정 근거가 아닌 조회는 근거로 남기지 않는다. 호출 이력(tool_calls)에는 그대로 남는다
        # 원인이 인자를 정한 근거는 그 인자로 조회한 결과만 근거다 (예: C08의 E06은 SY10 배포 이력만)
        def own(check_id: str, args: dict) -> bool:
            return any(c["check_id"] == check_id and _covered(check_id, c["args"], [(check_id, args)]) for c in checks)

        evidence = [{"check_id": c["check_id"], "tool": c["tool"], "args": c["args"], "text": c["text"], "reused": False}
                    for c in called if not c["is_error"] and own(c["check_id"], c["args"])]
        for result_id in dict.fromkeys(verdict["reused_results"]):
            cached = tool_results[result_id]
            if own(cached["check_id"], cached["args"]):
                evidence.append({k: cached[k] for k in ("check_id", "tool", "args", "text")} | {"reused": True})
        checked_at = deps.now()
        for e in evidence:
            events.record_finding(
                deps.driver, finding_id=_new_id("FD"), incident_id=diag["incident_id"], check_id=e["check_id"],
                cause_id=cause_id, result=verdict["result"], detail=e["text"], checked_at=checked_at,
                tool=e["tool"], args=e["args"],
            )
        tool_calls += [dict(c, cause_id=cause_id) for c in called]
        verdicts = dict(diag["verdicts"])
        verdicts[cause_id] = {**verdict, "status": status, "checks": checks, "evidence": evidence}
        diag |= {"tool_results": tool_results, "tool_calls": tool_calls, "verdicts": verdicts,
                 "checked": diag["checked"] + 1}
        if verdict["result"] == "confirm":
            diag["confirmed"] = cause_id
        diag["hyp_index"] += 1
        if problem:
            return {"diag": diag, "incidents": _flag(state, diag["incident_id"], problem)}
        return {"diag": diag}

    def after_check(state: State) -> str:
        diag = state["diag"]
        if diag.get("confirmed"):
            return "owner"
        if diag["hyp_index"] < len(diag["ranked"]) and diag["checked"] < MAX_HYPOTHESES:
            return "check"
        return "owner"

    async def _owner_or_none(cause_id: str, needs_human: list[str]) -> dict:
        """담당 팀을 받지 못하면 사람 확인 사유를 남기고 팀이 없는 값으로 둔다. 빈 값을 정상 결과로 쓰지 않는다."""
        try:
            return await deps.knowledge.owner_team(cause_id)
        except KnowledgeUnavailable as exc:
            needs_human.append(f"{cause_id}의 담당 팀을 받지 못했다: {exc}")
            return {"cause_id": cause_id, "located_in": None, "primary": [], "other_owners": [], "collaborators": []}

    # 8 담당 팀 결정 (지식 공급자 + 조건부 협업만 LLM, R5)
    async def owner(state: State) -> State:
        diag = dict(state["diag"])
        incident_id = diag["incident_id"]
        needs_human = list(state["incidents"][incident_id]["needs_human"])
        collaborators: list[dict] = []
        confirmed = diag.get("confirmed")
        if confirmed:
            # 상위 가설을 불명확으로 남긴 채 하위 가설이 확정되면 사람이 확인한다 (R4-4)
            above = []
            for h in diag["ranked"]:
                if h["cause_id"] == confirmed:
                    break
                if diag["verdicts"].get(h["cause_id"], {}).get("result") == "inconclusive":
                    above.append(h["cause_id"])
            if above:
                needs_human.append(f"상위 가설 {above}가 불명확인 채로 {confirmed}가 확정됐다")
            info = await _owner_or_none(confirmed, needs_human)
            primaries = [t["team_id"] for t in info["primary"]]
            if len(primaries) != 1:
                needs_human.append(f"1차 담당 팀이 하나로 정해지지 않는다: {primaries}")
            primary = primaries[0] if primaries else None
            # 같은 시스템의 다른 부분을 맡은 팀은 협업으로 붙인다 (R5-3)
            collaborators += [{"team_id": t["team_id"], "why": "같은 시스템의 다른 부분 담당"} for t in info["other_owners"]]
            conditional = info["collaborators"]
            if conditional:
                ids = [t["team_id"] for t in conditional]
                listing = "\n".join(f"- {t['team_id']} {t['name']}: 조건 {t['condition']}" for t in conditional)
                verdict = diag["verdicts"][confirmed]
                content = (
                    "사건 제보:\n" + "\n".join(_report_block({**r, "at": r["at"].isoformat()})
                                            for r in state["incidents"][incident_id]["reports"])
                    + f"\n\n확정 원인 {confirmed}, 판단 이유: {verdict['reason']}"
                    + "\n\n조회 결과:\n" + "\n".join(
                        f"- {r['check_id']} {r['tool']} {json.dumps(r['args'], ensure_ascii=False)}: {r['text']}"
                        for r in diag["tool_results"].values())
                    + f"\n\n조건부 협업 팀:\n{listing}"
                )
                data, errors = _ask_json(
                    deps, task="diagnosis", name="8.collaboration", system=prompts.COLLABORATION, content=content,
                    schema=_schema({"apply": {"type": "array", "items": {"type": "string", "enum": ids}},
                                    "reason": {"type": "string"}}),
                    validate=lambda d: [f"목록에 없는 팀이다: {sorted(set(d['apply']) - set(ids))}"]
                    if set(d.get("apply", [])) - set(ids) else [],
                )
                if data is None:
                    needs_human.append(f"조건부 협업 판단 출력 검증 실패: {errors}")
                else:
                    collaborators += [{"team_id": t, "why": "조건부 협업", "reason": data["reason"]} for t in data["apply"]]
            next_check = None
        else:
            # 확정되지 않으면 상위 2개 가설의 담당 팀을 함께 배정한다 (R5-4)
            open_hyps = [h for h in diag["ranked"] if diag["verdicts"].get(h["cause_id"], {}).get("status") != "ruled_out"][:2]
            teams = []
            for h in open_hyps:
                info = await _owner_or_none(h["cause_id"], needs_human)
                teams += [t["team_id"] for t in info["primary"] if t["team_id"] not in teams]
            primary = teams[0] if teams else None
            collaborators += [{"team_id": t, "why": "확정 전 상위 가설 담당"} for t in teams[1:]]
            needs_human.append(f"원인이 확정되지 않았다. 상위 가설: {[h['cause_id'] for h in open_hyps]}")
            next_check = None
            if open_hyps:
                # 다음 확인은 상위 가설의 필수 근거 중 아직 결과가 없는 것이다. 모두 조회했으면 사람이 결과를 판단한다
                target = open_hyps[0]["cause_id"]
                done = [(e["check_id"], e["args"]) for e in diag["verdicts"].get(target, {}).get("evidence", [])]
                try:
                    target_checks = await deps.knowledge.get_checks(target)
                except KnowledgeUnavailable as exc:
                    target_checks = []
                    needs_human.append(f"다음 확인할 판정 근거를 받지 못했다: {exc}")
                pending = [c for c in target_checks if c["required"] and not _covered(c["check_id"], c["args"], done)]
                if pending:
                    c = pending[0]
                    # 담당은 팀 ID로 적는다. 사람이 하는 조회는 수행 팀, 판단 팀이 따로 있으면 그 팀, 없으면 1차 담당
                    next_check = {"cause_id": target, "check_id": c["check_id"], "args": c["args"],
                                  "by": c.get("performed_by_team_id") or c.get("judged_by_team_id") or primary}
        # 같은 팀이 여러 번 붙지 않게 하고 1차 담당은 협업에서 뺀다
        seen, unique = {primary}, []
        for c in collaborators:
            if c["team_id"] not in seen:
                seen.add(c["team_id"])
                unique.append(c)
        if primary:
            events.assign_team(deps.driver, incident_id=incident_id, team_id=primary, role="primary")
        for c in unique:
            events.assign_team(deps.driver, incident_id=incident_id, team_id=c["team_id"], role="collaborator")
        diag |= {"primary_team": primary, "collaborators": unique, "next_check": next_check, "needs_human": needs_human}
        return {"diag": diag}

    # 9 절차, 조치 선택 (LLM, R6, R7)
    async def procedures(state: State) -> State:
        diag = dict(state["diag"])
        incident = state["incidents"][diag["incident_id"]]
        sev = incident["severity"]
        confirmed = diag.get("confirmed")
        candidates: dict[str, dict] = {}

        unavailable: list[str] = []

        async def known(procedure_id: str) -> dict | None:
            # 절차를 받지 못하면 후보에 넣지 않고 사람 확인 사유로 남긴다. 조치가 빈 절차를 정상 후보로 쓰면
            # 그 절차를 고르고 조치를 빼도 검증을 통과한다
            try:
                return await deps.knowledge.procedure(procedure_id)
            except KnowledgeUnavailable as exc:
                unavailable.append(f"절차 {procedure_id}를 받지 못했다: {exc}")
                return None

        # 반드시 여는 절차는 조회 결과와 상관없이 코드가 정한다. 어느 경로(절차 조회, 원인이 여는 절차)로
        # 후보에 들어오든 항상 여는 절차로 둔다. 절차 조회에 실패해도 원인 쪽 목록으로 들어오면 필수가 유지된다
        # 접수(P1)는 모든 사건, 등급이 상향되면 에스컬레이션(P2, R3), 긴급 공지(P3)는 등급에 걸린 절차(R3 TRIGGERS),
        # 긴급 점검(P4)도 SEV1, SEV2면 연다(R7). 원인이 확정되면 원인 수정(P10)을 연다
        required = {INTAKE_PROCEDURE}
        if incident.get("escalated"):
            required.add(ESCALATION_PROCEDURE)
        if sev in URGENT_LEVELS:
            required |= {URGENT_NOTICE, URGENT_MAINTENANCE}
        if confirmed:
            required.add(FIX_PROCEDURE)

        # 조건마다 그 조건을 판단할 판정 근거. 지식 공급자가 절차와 함께 준다 (M3 점검)
        condition_checks: dict[str, dict] = {}

        def add(proc: dict | None, **extra) -> None:
            if proc is None:
                return
            fields = {k: v for k, v in proc.items() if k not in ("condition", "condition_check")}
            entry = candidates.setdefault(proc["procedure_id"], {**fields, "always": False, "conditions": []})
            entry["always"] = entry["always"] or proc["procedure_id"] in required
            if extra.get("condition"):
                entry["conditions"].append(extra["condition"])
                if proc.get("condition_check"):
                    condition_checks.setdefault(extra["condition"], proc["condition_check"])

        for procedure_id in (INTAKE_PROCEDURE, ESCALATION_PROCEDURE, URGENT_NOTICE, URGENT_MAINTENANCE):
            if procedure_id in required:
                add(await known(procedure_id))
        if confirmed:
            try:
                found = await deps.knowledge.cause_procedures(confirmed, sev)
            except KnowledgeUnavailable as exc:
                found = {"resolved_by": [], "triggered_by_severity": []}
                unavailable.append(f"{confirmed}가 여는 절차를 받지 못했다: {exc}")
            add(await known(FIX_PROCEDURE))
            for proc in found["resolved_by"]:
                # 원인에 조건 없이 연결된 절차(보상, 회수, 확률 표기 정정 등)는 연다는 규칙이라 코드가 정한다
                # (4단계 3-4, M3 점검). 긴급 공지와 긴급 점검은 등급 규칙(R3, R7)을, 롤백은 사람 결정을 따른다
                if (not proc.get("condition") and not proc.get("human_only")
                        and proc["procedure_id"] not in (URGENT_NOTICE, URGENT_MAINTENANCE)):
                    required.add(proc["procedure_id"])
                add(proc, condition=proc.get("condition"))
            for proc in found.get("triggered_by_severity", []):
                add(proc)
        # 원인이 확정되지 않으면 원인에 걸린 절차를 고를 근거가 없다. 접수, 상향, 등급 절차만 후보로 둔다

        allowed_conditions = sorted({c for p in candidates.values() for c in p["conditions"]})
        action_ids = sorted({a["action_id"] for p in candidates.values() for a in p["actions"]})
        schema = _schema({
            "procedures": {"type": "array", "items": {"type": "string", "enum": sorted(candidates)}},
            "resolved_conditions": {"type": "array", "items": {"type": "string", "enum": allowed_conditions or [""]}},
            "actions": {"type": "array", "items": _schema({
                # 후보 절차에 조치가 없으면 빈 enum 대신 빈 문자열 하나를 둔다 (조건과 같은 방식). 검증이 거부한다
                "action_id": {"type": "string", "enum": action_ids or [""]},
                "mode": {"type": "string", "enum": ["auto", "approval"]},
            })},
            "reason": {"type": "string"},
        })

        def validate(data: dict) -> list[str]:
            errors = []
            chosen = data.get("procedures") or []
            unknown = sorted(set(chosen) - set(candidates))
            if unknown:
                errors.append(f"후보에 없는 절차다: {unknown}")
            human = sorted(p for p in chosen if candidates.get(p, {}).get("human_only"))
            if human:
                errors.append(f"사람만 여는 절차는 고르지 않는다: {human}")
            missing = sorted(p for p, c in candidates.items() if c["always"] and p not in chosen)
            if missing:
                errors.append(f"항상 여는 절차가 빠졌다: {missing}")
            in_chosen = {a["action_id"] for p in chosen if p in candidates for a in candidates[p]["actions"]}
            requested = [a["action_id"] for a in data.get("actions") or []]
            stray = sorted(set(requested) - in_chosen)
            if stray:
                errors.append(f"고른 절차에 없는 조치다: {stray}")
            missing_actions = sorted(in_chosen - set(requested))
            if missing_actions:
                errors.append(f"고른 절차의 조치가 빠졌다: {missing_actions}")
            duplicated = sorted({a for a in requested if requested.count(a) > 1})
            if duplicated:
                errors.append(f"같은 조치를 두 번 요청했다: {duplicated}")
            reported = set(data.get("resolved_conditions") or [])
            bad_cond = sorted(reported - set(allowed_conditions))
            if bad_cond:
                errors.append(f"후보 절차에 없는 조건이다: {bad_cond}")
            unmet = sorted(p for p in chosen if candidates.get(p, {}).get("conditions")
                           and not set(candidates[p]["conditions"]) & reported)
            if unmet:
                errors.append(f"조건이 붙은 절차인데 충족한 조건을 적지 않았다: {unmet}")
            return errors

        # 조건의 판정 근거는 판단이 아니라 조회라서 코드가 부른다. 조건이 맞는지는 LLM이 결과를 읽고 정한다
        # (접수 내역 조회 E15와 같은 방식, M3 점검). 같은 도구와 인자는 한 번만 부른다
        condition_calls: list[dict] = []
        condition_lines: list[str] = []
        done: dict[tuple[str, str], dict] = {}
        for name in sorted(condition_checks):
            check = condition_checks[name]
            args = check.get("args") or {}
            key = (check["tool"], json.dumps(args, ensure_ascii=False, sort_keys=True))
            if key not in done:
                result = await call_tool(deps.mock_client, deps.tracer, server="mock_tools", name=check["tool"], args=args)
                done[key] = {"step": "9.condition", "cause_id": confirmed, "tool": check["tool"],
                             "check_id": check["check_id"], "args": args, "is_error": result.is_error,
                             "text": result.text}
                condition_calls.append(done[key])
                if result.is_error:
                    # 조건 절차는 필수가 아니라 LLM이 빼도 검증을 통과한다. 근거 없이 빠진 것을 사람이 알게 남긴다
                    unavailable.append(f"조건 판정 근거 {check['check_id']}를 조회하지 못했다: {result.text}")
            call = done[key]
            text = call["text"] if not call["is_error"] else f"조회하지 못했다 ({call['text']})"
            condition_lines.append(f"- 조건 {name}: 판정 근거 {check['check_id']}, 기대 결과 \"{check['expect']}\". "
                                   f"조회 결과: {text}")

        verdict = diag["verdicts"].get(confirmed, {}) if confirmed else {}
        content = (
            f"사건: 증상 {incident['symptom_id']}, 심각도 {sev}, 등급 상향 {'예' if incident.get('escalated') else '아니오'}, "
            f"확정 원인 {confirmed or '없음'}"
            + (f" (판단 이유: {verdict.get('reason')})" if verdict else "")
            + "\n조회 결과:\n" + "\n".join(f"- {c['tool']}: {c['text']}" for c in diag["tool_calls"] if not c["is_error"])
            + ("\n\n조건 판정 근거 조회:\n" + "\n".join(condition_lines) if condition_lines else "")
            + f"\n\n후보 절차:\n{json.dumps(list(candidates.values()), ensure_ascii=False, indent=1)}"
        )
        needs_human = list(diag["needs_human"]) + unavailable
        if not candidates:
            # 고를 절차가 하나도 없으면 묻지 않는다. 선택지가 빈 스키마를 API에 보내지 않는다
            # (그래프 공급자는 접수 P1이 항상 있어 생기지 않는다. 텍스트 공급자가 절차를 하나도 주지 못하면 생긴다)
            data = None
            needs_human.append("고를 수 있는 절차 후보가 없어 절차와 조치를 고르지 않았다")
        else:
            data, errors = _ask_json(
                deps, task="diagnosis", name="9.procedures", system=prompts.PROCEDURE, content=content,
                schema=schema, validate=validate,
            )
            if data is None:
                needs_human.append(f"절차, 조치 선택 출력 검증 실패: {errors}")
        if data is None:
            data = {"procedures": [], "resolved_conditions": [], "actions": [], "reason": None}
        diag |= {"procedure_choice": data, "candidate_procedures": sorted(candidates), "needs_human": needs_human,
                 "tool_calls": list(diag["tool_calls"]) + condition_calls}
        return {"diag": diag}

    # 10 승인 게이트 (코드, R6)
    async def approval_gate(state: State) -> State:
        diag = dict(state["diag"])
        incident = state["incidents"][diag["incident_id"]]
        urgent = incident["severity"] in URGENT_LEVELS
        runs = []
        for request in diag["procedure_choice"]["actions"]:
            condition = FIX_TIMING[urgent] if request["action_id"] == CAUSE_FIX_ACTION else None
            try:
                run = gate.request_action(
                    deps.driver, tracer=deps.tracer, incident_id=diag["incident_id"],
                    action_id=request["action_id"], requested_mode=request["mode"], now=deps.now(),
                    condition=condition,
                )
                runs.append({"action_id": run["action_id"], "requested_mode": run["requested_mode"],
                             "status": run["status"], "gate_blocked": run["gate_blocked"], "run_id": run["run_id"]})
            except GateError as exc:
                runs.append({"action_id": request["action_id"], "requested_mode": request["mode"],
                             "status": "refused", "error": str(exc)})
        diag["runs"] = runs
        return {"diag": diag}

    # 11 기록, 출력 (코드). 사건 계층은 앞 단계에서 기록했다. 여기서는 구조화 출력을 만든다 (4단계 3-3)
    async def record(state: State) -> State:
        diag = state["diag"]
        incident = state["incidents"][diag["incident_id"]]
        hypotheses = []
        for h in diag["ranked"]:
            v = diag["verdicts"].get(h["cause_id"])
            hypotheses.append({
                "cause_id": h["cause_id"], "rank": h["rank"], "score": h["score"],
                "status": v["status"] if v else "candidate",
                "result": v["result"] if v else None,
                "findings": [{"check_id": e["check_id"], "tool": e["tool"], "args": e["args"], "result": v["result"],
                              "detail": e["text"],
                              "reused": e["reused"]} for e in v["evidence"]] if v else [],
            })
        output = {
            "symptom": incident["symptom_id"],
            "incident_id": diag["incident_id"],
            "report_ids": incident["report_ids"],
            # 시나리오 제보의 원래 순서 번호. 채점기가 정답 묶음(incident_groups)과 비교한다
            "report_indexes": [r["report_index"] for r in incident["reports"]],
            "scope": incident["scope"],
            "severity": incident["severity"],
            "escalated": incident.get("escalated", False),
            "first_handler": incident["first_handler"],
            "hypotheses": hypotheses,
            "root_cause": diag.get("confirmed"),
            "primary_team": diag["primary_team"],
            "collaborators": diag["collaborators"],
            "next_check": diag.get("next_check"),
            "procedures": sorted(diag["procedure_choice"]["procedures"], key=lambda p: int(p[1:])),
            "resolved_conditions": diag["procedure_choice"]["resolved_conditions"],
            "actions": diag["runs"],
            "merge_replies": incident.get("merge_replies", []),
            # 접수 정리의 접수 내역 조회(1단계)와 판정 근거 조회(7단계)를 단계와 함께 남긴다
            "tool_calls": [{k: c[k] for k in ("step", "cause_id", "tool", "check_id", "args", "is_error")}
                           for c in [*(r["scope_lookup"] for r in incident["reports"]), *diag["tool_calls"]]],
            "explanation": _explain(incident, diag, symptom_by_id),
            "needs_human": diag["needs_human"],
            "retrieved_docs": None,  # 문서 검색 구성(B, H)만 쓴다
        }
        return {"outputs": [*(state.get("outputs") or []), output], "diag": None}

    graph = StateGraph(State)
    for name, fn in [
        ("intake", intake), ("first_handler", first_handler), ("duplicate_candidates", duplicate_candidates),
        ("duplicate_judge", duplicate_judge), ("severity", severity), ("next_incident", next_incident),
        ("rank", rank), ("check", check), ("owner", owner), ("procedures", procedures),
        ("approval_gate", approval_gate), ("record", record),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "first_handler")
    graph.add_edge("first_handler", "duplicate_candidates")
    graph.add_edge("duplicate_candidates", "duplicate_judge")
    graph.add_edge("duplicate_judge", "severity")
    graph.add_conditional_edges("severity", after_report, ["intake", "next_incident"])
    graph.add_conditional_edges("next_incident", after_next, ["rank", END])
    graph.add_conditional_edges("rank", after_rank, ["check", "owner"])
    graph.add_conditional_edges("check", after_check, ["check", "owner"])
    graph.add_edge("owner", "procedures")
    graph.add_edge("procedures", "approval_gate")
    graph.add_edge("approval_gate", "record")
    graph.add_edge("record", "next_incident")
    return graph.compile()


def _unchecked(data, schema: dict, required: list[str], called: list[dict], tool_results: dict,
               prior_ids: set[str]) -> list[str]:
    """답에 쓸 근거가 없는 필수 판정 근거. 이번에 성공한 조회와 인용한 앞 결과를 근거로 본다.

    답을 읽을 수 없으면 빈 목록이다. 그 경우는 읽을 수 없는 답으로 따로 처리한다.
    """
    if data is None or _structure_errors(data, schema):
        return []
    have = [(c["check_id"], c["args"]) for c in called if not c["is_error"]]
    have += [(tool_results[r]["check_id"], tool_results[r]["args"]) for r in data["reused_results"] if r in prior_ids]
    return [_check_label(check_id, args) for check_id, args in required if not _covered(check_id, args, have)]


def _covered(check_id: str, args: dict, have: list[tuple[str, dict]]) -> bool:
    """필수 근거가 조회됐는지. 원인이 인자를 정해 둔 근거(예: 어느 시스템의 로그)는 그 인자로 조회해야 한다."""
    return any(c == check_id and all(a.get(k) == v for k, v in args.items()) for c, a in have)


def _check_label(check_id: str, args: dict) -> str:
    return f"{check_id}({', '.join(f'{k}={v}' for k, v in args.items())})" if args else check_id


def _explain(incident: dict, diag: dict, symptom_by_id: dict) -> str:
    """근거 경로 설명 (R4-5). 기록에서 경로를 조립하고 판정 근거 단계 LLM의 판단 이유를 붙인다."""
    symptom = symptom_by_id[incident["symptom_id"]]
    lines = [f"증상 {symptom['id']} {symptom['name']} (영향 범위 {incident['scope']}, 심각도 {incident['severity']})"]
    for h in diag["ranked"]:
        v = diag["verdicts"].get(h["cause_id"])
        if not v:
            continue
        used = [e["check_id"] for e in v["evidence"]]
        lines.append(f"-> 원인 {h['cause_id']} {h['name']} (점수 {h['score']}): {v['result']}, 근거 {used}. {v['reason']}")
    if diag.get("primary_team"):
        lines.append(f"-> 1차 담당 {diag['primary_team']}")
    return "\n".join(lines)
