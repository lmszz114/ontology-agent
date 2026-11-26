"""전체 텍스트 지식 공급자 (구성 A) (설계 4단계 2-2, 5단계 4-2).

기준 지식 K 전체를 프롬프트에 넣고, 원인 후보와 순위(R4), 판정 근거(R4-4), 담당 팀(R5), 절차(R7)를
LLM이 텍스트와 규칙 설명을 읽고 답한다. 구성 C의 그래프 공급자와 같은 모양의 값을 돌려준다.

코드는 형식과 정의된 ID까지만 검증한다. "이 증상의 후보 원인인가"처럼 그래프 관계로 답을 고치지 않는다.
그렇게 하면 구성 A에 온톨로지를 몰래 주는 셈이다. 틀리면 한 번 다시 묻고, 그래도 틀리면 errors에 남기고
KnowledgeUnavailable을 낸다. 빈 값으로 돌려주지 않는다. 빈 값은 "근거가 없다"와 구분되지 않기 때문이다.
받는 쪽(진단 그래프)은 이 예외를 사람 확인 사유로 남긴다.

K와 규칙 설명은 system 블록 하나로 두고 캐싱한다 (5분). 같은 질문은 한 실행 안에서 다시 묻지 않는다.
변경 경과 시간(gap_minutes)은 기록용 값이라 코드가 시각으로 계산한다. 가중을 줄지는 LLM이 정한다.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from common.knowledge import text_prompts as P
from common.knowledge.base import KnowledgeUnavailable
from common.knowledge.k_text import k_text
from common.models import LLM, LLMResult

RISKS = ["low", "high"]


@dataclass(frozen=True)
class KnowledgeIds:
    """정의된 ID와 이름. 답이 정의된 값인지 보는 데만 쓴다 (출력 검증)."""

    causes: dict[str, str]
    checks: dict[str, str]
    tools: list[str]
    teams: dict[str, str]
    systems: dict[str, str]
    procedures: dict[str, str]
    actions: dict[str, str]

    @classmethod
    def from_seed(cls, seed: dict) -> "KnowledgeIds":
        def names(key: str) -> dict[str, str]:
            return {item["id"]: item["name"] for item in seed[key]}
        return cls(causes=names("causes"), checks=names("evidence_checks"),
                   tools=sorted({c["tool"] for c in seed["evidence_checks"]}), teams=names("teams"),
                   systems=names("systems"), procedures=names("procedures"), actions=names("actions"))


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or list(properties),
            "additionalProperties": False}


def _enum(values) -> dict:
    return {"type": "string", "enum": sorted(values)}


def _shape_errors(data: Any, schema: dict, where: str = "답") -> list[str]:
    """스키마의 필수 필드와 타입을 본다. 변환 코드가 읽는 필드가 빠지거나 타입이 다르면 예외 대신 다시 묻는다."""
    kind = schema.get("type")
    checks = {"object": dict, "array": list, "string": str, "boolean": bool}
    if kind == "integer":
        ok = isinstance(data, int) and not isinstance(data, bool)
    else:
        ok = isinstance(data, checks[kind]) if kind in checks else True
    if not ok:
        return [f"{where}의 타입이 {kind}가 아니다"]
    if kind == "object":
        errors = []
        for name in schema.get("required", []):
            if name not in data:
                errors.append(f"{where}에 {name} 필드가 없다")
            else:
                errors += _shape_errors(data[name], schema["properties"][name], f"{where}.{name}")
        return errors
    if kind == "array":
        return [e for i, item in enumerate(data) for e in _shape_errors(item, schema["items"], f"{where}[{i}]")]
    if "enum" in schema and data not in schema["enum"]:
        return [f"{where}가 정의된 값이 아니다: {data!r}"]
    return []


def _number(pid: str) -> int:
    return int(pid[1:]) if pid[1:].isdigit() else 0


class TextKnowledge:
    def __init__(self, *, llm: LLM, seed: dict, results: list[LLMResult],
                 changes: Callable[[], list[dict[str, Any]]], name: str = "A"):
        self.llm = llm
        self.name = name
        self.prompt_version = P.PROMPT_VERSION
        self.ids = KnowledgeIds.from_seed(seed)
        self.results = results          # 진단 실행의 LLM 호출 목록. 비용과 토큰을 함께 센다
        self.changes = changes          # 시나리오의 변경 이력 (사건 데이터)
        self.errors: list[str] = []     # 검증에 두 번 실패한 질문
        self.system = [{"type": "text", "text": P.SYSTEM + "\n" + k_text(seed), "cache_control": {"type": "ephemeral"}}]
        self.retry = P.RETRY
        self._memo: dict[tuple, Any] = {}

    async def begin_incident(self, incident_id: str, reports: list[str]) -> dict | None:
        """구성 A는 기준 지식 전체를 쓰므로 사건마다 바꾸는 것이 없다."""
        return None

    # --- 공통 ---------------------------------------------------------------

    def _ask(self, name: str, content: str, schema: dict, validate: Callable[[dict], list[str]]) -> dict:
        messages: list[dict] = [{"role": "user", "content": content}]
        errors: list[str] = []
        for attempt in range(2):
            result = self.llm.call(
                task="diagnosis", name=f"knowledge.{self.name}.{name}" + (".retry" if attempt else ""),
                system=self.system, messages=messages, output_schema=schema,
                metadata={"config": self.name, "prompt_version": P.PROMPT_VERSION},
            )
            self.results.append(result)
            try:
                data = result.json()
            except ValueError:
                data, errors = None, [f"JSON으로 읽을 수 없는 답이다 (stop_reason={result.message.stop_reason})"]
            else:
                # 형식(필수 필드, 타입)을 먼저 보고 ID를 본다. 변환 코드가 읽는 필드가 빠지면 예외 대신 다시 묻는다
                errors = _shape_errors(data, schema) or validate(data)
            if not errors:
                return data
            messages += [{"role": "assistant", "content": result.text or "(빈 답)"},
                         {"role": "user", "content": self.retry.format(errors="\n".join(f"- {e}" for e in errors))}]
        self.errors.append(f"{name}: {errors}")
        raise KnowledgeUnavailable(f"{name}: {errors}")

    def _memoized(self, key: tuple, make: Callable[[], Any]) -> Any:
        if key not in self._memo:
            try:
                self._memo[key] = make()
            except KnowledgeUnavailable as exc:
                self._memo[key] = exc  # 실패도 기억한다. 같은 질문을 다시 물어 비용을 더 쓰지 않는다
        value = self._memo[key]
        if isinstance(value, KnowledgeUnavailable):
            raise value
        return json.loads(json.dumps(value))  # 부르는 쪽이 고쳐도 저장한 값이 바뀌지 않게 복사한다

    def _undefined(self, label: str, values, defined) -> list[str]:
        bad = sorted({v for v in values if v not in defined})
        return [f"정의되지 않은 {label}다: {bad}"] if bad else []

    # --- R4 원인 후보와 순위 -------------------------------------------------

    async def rank_causes(self, symptom_id: str, started_at: str) -> list[dict[str, Any]]:
        return self._memoized(("rank", symptom_id, started_at), lambda: self._rank(symptom_id, started_at))

    def _rank(self, symptom_id: str, started_at: str) -> list[dict[str, Any]]:
        changes = {c["id"]: c for c in self.changes()}
        listing = "\n".join(f"- {c['id']}: 시스템 {c['system_id']}, 종류 {c['kind']}, 시각 {c['changed_at']}"
                            for c in sorted(changes.values(), key=lambda c: c["id"])) or "- 없음"
        schema = _schema({"candidates": {"type": "array", "items": _schema({
            "cause_id": _enum(self.ids.causes), "score": {"type": "integer"}, "rank": {"type": "integer"},
            "change_ids": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"},
        })}})

        def validate(data: dict) -> list[str]:
            rows = data["candidates"]
            ids = [r["cause_id"] for r in rows]
            errors = self._undefined("원인", ids, self.ids.causes)
            if len(set(ids)) != len(ids):
                errors.append("같은 원인을 두 번 적었다")
            if sorted(r["rank"] for r in rows) != list(range(1, len(rows) + 1)):
                errors.append("순위는 1부터 빠짐없이 한 번씩 매긴다")
            errors += self._undefined("변경", [c for r in rows for c in r["change_ids"]], changes)
            return errors

        data = self._ask("6.rank", P.RANK.format(symptom_id=symptom_id, started_at=started_at, changes=listing),
                         schema, validate)
        start = datetime.fromisoformat(started_at)
        rows = []
        for r in sorted(data["candidates"], key=lambda r: r["rank"]):
            recent = [{"id": cid, "kind": changes[cid]["kind"], "changed_at": changes[cid]["changed_at"],
                       "gap_minutes": int((start - datetime.fromisoformat(changes[cid]["changed_at"])).total_seconds() // 60)}
                      for cid in dict.fromkeys(r["change_ids"])]
            rows.append({"cause_id": r["cause_id"], "name": self.ids.causes[r["cause_id"]], "score": r["score"],
                         "rank": r["rank"], "recent_changes": recent, "reason": r["reason"]})
        return rows

    # --- R4-4 판정 근거 -------------------------------------------------------

    async def get_checks(self, cause_id: str) -> list[dict[str, Any]]:
        return self._memoized(("checks", cause_id), lambda: self._checks(cause_id))

    def _checks(self, cause_id: str) -> list[dict[str, Any]]:
        team = {"type": "string", "enum": ["", *sorted(self.ids.teams)]}
        schema = _schema({"checks": {"type": "array", "items": _schema({
            "check_id": _enum(self.ids.checks), "tool": _enum(self.ids.tools), "required": {"type": "boolean"},
            "expect": {"type": "string"}, "note": {"type": "string"}, "performed_by": {"type": "string"},
            "judged_by_team_id": team, "performed_by_team_id": team,
            "args": {"type": "array", "items": _schema({"name": {"type": "string"}, "value": {"type": "string"}})},
        })}})

        def validate(data: dict) -> list[str]:
            rows = data["checks"]
            errors = self._undefined("판정 근거", [r["check_id"] for r in rows], self.ids.checks)
            errors += self._undefined("조회 도구", [r["tool"] for r in rows], self.ids.tools)
            errors += self._undefined("팀", [t for r in rows for t in (r["judged_by_team_id"], r["performed_by_team_id"]) if t],
                                      self.ids.teams)
            keys = [(r["check_id"], json.dumps(sorted((a["name"], a["value"]) for a in r["args"]))) for r in rows]
            if len(set(keys)) != len(keys):
                errors.append("같은 판정 근거를 같은 인자로 두 번 적었다")
            return errors

        data = self._ask(f"7.checks.{cause_id}", P.CHECKS.format(cause_id=cause_id), schema, validate)
        rows = []
        for r in sorted(data["checks"], key=lambda r: r["check_id"]):
            rows.append({
                "check_id": r["check_id"], "name": self.ids.checks[r["check_id"]], "tool": r["tool"],
                "performed_by": r["performed_by"] or None, "required": r["required"],
                "expect": r["expect"] or None, "note": r["note"] or None,
                "args": {a["name"]: a["value"] for a in r["args"]},
                "judged_by_team": self.ids.teams.get(r["judged_by_team_id"]),
                "performed_by_team": self.ids.teams.get(r["performed_by_team_id"]),
                "judged_by_team_id": r["judged_by_team_id"] or None,
                "performed_by_team_id": r["performed_by_team_id"] or None,
            })
        return rows

    # --- R5 담당 팀 ----------------------------------------------------------

    async def owner_team(self, cause_id: str) -> dict[str, Any]:
        return self._memoized(("owner", cause_id), lambda: self._owner(cause_id))

    def _owner(self, cause_id: str) -> dict[str, Any]:
        teams = _enum(self.ids.teams)
        schema = _schema({
            "system_id": _enum(self.ids.systems), "scope": {"type": "string"}, "area": {"type": "string"},
            "primary": {"type": "array", "items": teams},
            "other_owners": {"type": "array", "items": _schema({
                "team_id": teams, "scope": {"type": "string"}, "area": {"type": "string"}})},
            "collaborators": {"type": "array", "items": _schema({"team_id": teams, "condition": {"type": "string"}})},
        })

        def validate(data: dict) -> list[str]:
            errors = self._undefined("시스템", [data["system_id"]], self.ids.systems)
            used = data["primary"] + [o["team_id"] for o in data["other_owners"]] + [c["team_id"] for c in data["collaborators"]]
            return errors + self._undefined("팀", used, self.ids.teams)

        data = self._ask(f"8.owner.{cause_id}", P.OWNER.format(cause_id=cause_id), schema, validate)
        primary_ids = list(dict.fromkeys(data["primary"]))
        others: dict[str, dict] = {}
        for o in sorted(data["other_owners"], key=lambda o: (o["team_id"], o["scope"])):
            if o["team_id"] in primary_ids:
                continue
            entry = others.setdefault(o["team_id"], {"team_id": o["team_id"], "name": self.ids.teams[o["team_id"]], "owns": []})
            entry["owns"].append({"scope": o["scope"], "area": o["area"] or None})
        return {
            "cause_id": cause_id,
            "located_in": {"system_id": data["system_id"], "system_name": self.ids.systems[data["system_id"]],
                           "scope": data["scope"], "area": data["area"] or None},
            "primary": [{"team_id": t, "name": self.ids.teams[t]} for t in primary_ids],
            "other_owners": list(others.values()),
            "collaborators": sorted(({"team_id": c["team_id"], "name": self.ids.teams[c["team_id"]],
                                      "condition": c["condition"]} for c in data["collaborators"]),
                                    key=lambda t: t["team_id"]),
        }

    # --- R7 절차 -------------------------------------------------------------

    def _procedure_schema(self, with_condition: bool) -> dict:
        props = {
            "procedure_id": _enum(self.ids.procedures), "human_only": {"type": "boolean"},
            "led_by_team_id": {"type": "string", "enum": ["", *sorted(self.ids.teams)]},
            "steps": {"type": "array", "items": {"type": "string"}},
            "actions": {"type": "array", "items": _schema({"action_id": _enum(self.ids.actions), "risk": _enum(RISKS)})},
        }
        if with_condition:
            # 조건이 없으면 condition과 condition_check의 check_id, tool을 빈 문자열로 둔다
            props["condition"] = {"type": "string"}
            props["condition_check"] = _schema({
                "check_id": {"type": "string", "enum": ["", *sorted(self.ids.checks)]},
                "tool": {"type": "string", "enum": ["", *self.ids.tools]},
                "expect": {"type": "string"},
                "args": {"type": "array", "items": _schema({"name": {"type": "string"}, "value": {"type": "string"}})},
            })
        return _schema(props)

    def _procedure_errors(self, rows: list[dict]) -> list[str]:
        errors = self._undefined("절차", [r["procedure_id"] for r in rows], self.ids.procedures)
        errors += self._undefined("조치", [a["action_id"] for r in rows for a in r["actions"]], self.ids.actions)
        errors += self._undefined("위험도", [a["risk"] for r in rows for a in r["actions"]], RISKS)
        for r in rows:
            if "condition" not in r:
                continue
            check = r["condition_check"]
            if r["condition"] and not (check["check_id"] and check["tool"] and check["expect"]):
                errors.append(f"{r['procedure_id']}: 조건 {r['condition']}의 판정 근거(check_id, tool, expect)를 적지 않았다")
            if not r["condition"] and (check["check_id"] or check["tool"]):
                errors.append(f"{r['procedure_id']}: 조건이 없는데 조건의 판정 근거를 적었다")
            errors += self._undefined("판정 근거", [check["check_id"]] if check["check_id"] else [], self.ids.checks)
            errors += self._undefined("조회 도구", [check["tool"]] if check["tool"] else [], self.ids.tools)
        return errors + self._undefined("팀", [r["led_by_team_id"] for r in rows if r["led_by_team_id"]], self.ids.teams)

    def _procedure_row(self, r: dict) -> dict:
        row = {"procedure_id": r["procedure_id"], "name": self.ids.procedures[r["procedure_id"]], "steps": r["steps"],
               "human_only": r["human_only"], "led_by": self.ids.teams.get(r["led_by_team_id"]),
               "actions": [{"action_id": a["action_id"], "name": self.ids.actions[a["action_id"]], "risk": a["risk"]}
                           for a in r["actions"]]}
        if "condition" in r:
            check = r["condition_check"]
            condition_check = ({"check_id": check["check_id"], "tool": check["tool"], "expect": check["expect"],
                                "args": {a["name"]: a["value"] for a in check["args"]}}
                               if r["condition"] else None)
            row = {"condition": r["condition"] or None, "condition_check": condition_check, **row}
        return row

    async def cause_procedures(self, cause_id: str, severity_id: str) -> dict[str, Any]:
        return self._memoized(("procedures", cause_id, severity_id), lambda: self._cause_procedures(cause_id, severity_id))

    def _cause_procedures(self, cause_id: str, severity_id: str) -> dict[str, Any]:
        schema = _schema({
            "resolved_by": {"type": "array", "items": self._procedure_schema(True)},
            "triggered_by_severity": {"type": "array", "items": self._procedure_schema(False)},
        })

        def validate(data: dict) -> list[str]:
            return self._procedure_errors(data["resolved_by"] + data["triggered_by_severity"])

        data = self._ask(f"9.procedures.{cause_id}", P.CAUSE_PROCEDURES.format(cause_id=cause_id, severity_id=severity_id),
                         schema, validate)
        order = lambda rows: sorted(rows, key=lambda r: _number(r["procedure_id"]))  # noqa: E731
        return {"cause_id": cause_id,
                "resolved_by": [self._procedure_row(r) for r in order(data["resolved_by"])],
                "triggered_by_severity": [self._procedure_row(r) for r in order(data["triggered_by_severity"])]}

    async def procedure(self, procedure_id: str) -> dict[str, Any]:
        return self._memoized(("procedure", procedure_id), lambda: self._procedure(procedure_id))

    def _procedure(self, procedure_id: str) -> dict[str, Any]:
        schema = self._procedure_schema(False)

        def validate(data: dict) -> list[str]:
            errors = self._procedure_errors([data])
            if data["procedure_id"] != procedure_id:
                errors.append(f"물은 절차는 {procedure_id}다: {data['procedure_id']}")
            return errors

        data = self._ask(f"9.procedure.{procedure_id}", P.PROCEDURE.format(procedure_id=procedure_id), schema, validate)
        return self._procedure_row(data)
