"""판정 근거의 관측 값과 원인의 확정 조건을 다룬다 (설계 3단계 6장, R4).

Neo4j 없이 시드 dict만으로 돈다. 쓰는 곳은 셋이다.
- 시드 검증: 조건이 정의된 필드와 값만 쓰는지 (적재 전에 확인), 같은 증상의 원인끼리 구분되는지 (테스트)
- 모의 도구: 환경(가상 데이터)을 읽어 조회 결과 문장을 만든다
- 시나리오 검증: 환경을 조건에 대입했을 때 정답 원인만 확정되는지 (시나리오는 M2에서 만든다)

진단 에이전트의 판정을 대신하지 않는다. 에이전트에게는 문장만 가고, 확정과 배제는 LLM이 판단한다.
"""

import json
from dataclasses import dataclass, field
from itertools import product

CONFIRM = "confirm"
RULE_OUT = "rule_out"


class ConditionError(ValueError):
    """정의되지 않은 판정 근거, 인자, 필드, 값."""


def args_key(args: dict | None) -> str:
    """인자 dict를 비교할 수 있는 문자열로 바꾼다. 인자가 없으면 '{}'."""
    return json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def _checks(seed: dict) -> dict[str, dict]:
    return {c["id"]: c for c in seed["evidence_checks"]}


def validate_args(seed: dict, check_id: str, args: dict | None) -> dict:
    """판정 근거에 넘기는 인자를 확인하고 그대로 돌려준다."""
    check = _checks(seed).get(check_id)
    if check is None:
        raise ConditionError(f"정의되지 않은 판정 근거다: {check_id}")
    args = args or {}
    params = check.get("params", {})
    if set(args) != set(params):
        raise ConditionError(f"{check_id}의 인자는 {sorted(params)}이어야 한다: {sorted(args)}")
    systems = {s["id"] for s in seed["systems"]}
    for name, value in args.items():
        allowed = systems if params[name].get("kind") == "system" else set(params[name]["values"])
        if value not in allowed:
            raise ConditionError(f"{check_id}의 {name}에 쓸 수 없는 값이다: {value}")
    return args


def validate_observations(seed: dict, check_id: str, observations: dict) -> None:
    """관측 값이 정의된 필드와 값인지 확인한다."""
    check = _checks(seed).get(check_id)
    if check is None:
        raise ConditionError(f"정의되지 않은 판정 근거다: {check_id}")
    fields = check.get("observations", {})
    for name, value in observations.items():
        if name not in fields:
            raise ConditionError(f"{check_id}에 정의되지 않은 필드다: {name}")
        if value not in fields[name]["values"]:
            raise ConditionError(f"{check_id}.{name}에 쓸 수 없는 값이다: {value}")


def allowed_values(expected) -> list:
    """when의 값을 목록으로 맞춘다. 값 하나도 목록 하나로 본다."""
    return list(expected) if isinstance(expected, list) else [expected]


def is_required(ref: dict) -> bool:
    """확정과 배제에 쓰는 판정 근거면 참. when 대신 note가 있으면 참고용이다."""
    return "when" in ref


def required_checks(cause: dict) -> list[dict]:
    return [ref for ref in cause["checked_by"] if is_required(ref)]


def validate_cause_conditions(seed: dict) -> None:
    """모든 원인의 args와 when이 판정 근거 정의에 맞는지 확인한다.

    판정 근거 연결은 확정 조건(when, expect)이거나 참고용(note) 둘 중 하나여야 한다.
    """
    for cause in seed["causes"]:
        if not required_checks(cause):
            raise ConditionError(f"{cause['id']}에 확정 조건이 없다")
        for ref in cause["checked_by"]:
            validate_args(seed, ref["check"], ref.get("args"))
            where = f"{cause['id']}의 {ref['check']}"
            if not is_required(ref):
                # 금지된 필드는 값이 비어 있어도 거부한다. 있는지 자체를 본다
                if not ref.get("note") or "expect" in ref:
                    raise ConditionError(f"{where}: 참고용 근거는 note만 적는다")
                continue
            if not ref["when"] or not ref.get("expect") or "note" in ref:
                raise ConditionError(f"{where}: 확정 조건은 when과 expect를 적는다")
            for name, expected in ref["when"].items():
                for value in allowed_values(expected):
                    validate_observations(seed, ref["check"], {name: value})


@dataclass
class Environment:
    """판정 근거 조회가 돌려줄 가상 데이터.

    entries의 각 항목: {check, args(선택), observations(선택), detail(선택)}
    적지 않은 필드는 판정 근거의 default(이상 없음)로 본다.
    """

    seed: dict
    entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_key: dict[tuple[str, str], dict] = {}
        for entry in self.entries:
            args = validate_args(self.seed, entry["check"], entry.get("args"))
            validate_observations(self.seed, entry["check"], entry.get("observations", {}))
            key = (entry["check"], args_key(args))
            if key in self._by_key:
                raise ConditionError(f"환경에 같은 조회가 두 번 있다: {key}")
            self._by_key[key] = entry

    def observe(self, check_id: str, args: dict | None = None) -> dict:
        """조회 하나의 관측 값 전체 (기본값 포함)."""
        args = validate_args(self.seed, check_id, args)
        fields = _checks(self.seed)[check_id].get("observations", {})
        values = {name: spec["default"] for name, spec in fields.items()}
        values |= self._by_key.get((check_id, args_key(args)), {}).get("observations", {})
        return values

    def detail(self, check_id: str, args: dict | None = None) -> str:
        """진단 에이전트에게 돌려줄 조회 결과 문장."""
        args = validate_args(self.seed, check_id, args)
        entry = self._by_key.get((check_id, args_key(args)), {})
        if entry.get("detail"):
            return entry["detail"]
        check = _checks(self.seed)[check_id]
        fields = check.get("observations", {})
        if not fields:
            return check["default_detail"]
        observed = self.observe(check_id, args)
        labels = self._arg_labels(check, args)
        return ". ".join(fields[name]["values"][observed[name]].format(**labels) for name in fields)

    def _arg_labels(self, check: dict, args: dict) -> dict[str, str]:
        """응답 문장에 넣을 인자의 이름. 시스템은 시스템 이름, 그 밖은 params에 적힌 이름."""
        systems = {s["id"]: s["name"] for s in self.seed["systems"]}
        labels = {}
        for name, value in args.items():
            values = check["params"][name].get("values")
            labels[name] = values[value] if isinstance(values, dict) else systems.get(value, value)
        return labels


def evaluate_cause(seed: dict, env: Environment, cause_id: str) -> str:
    """환경을 원인의 확정 조건에 대입한다. 모두 맞으면 확정, 하나라도 다르면 배제."""
    cause = next(c for c in seed["causes"] if c["id"] == cause_id)
    for ref in required_checks(cause):
        observed = env.observe(ref["check"], ref.get("args"))
        for name, expected in ref["when"].items():
            if observed[name] not in allowed_values(expected):
                return RULE_OUT
    return CONFIRM


def confirmed_causes(seed: dict, env: Environment, symptom_id: str) -> list[str]:
    """증상의 원인 후보 중 이 환경에서 확정되는 것."""
    symptom = next(s for s in seed["symptoms"] if s["id"] == symptom_id)
    return [ref["cause"] for ref in symptom["causes"] if evaluate_cause(seed, env, ref["cause"]) == CONFIRM]


def environments_for(seed: dict, cause_id: str) -> list[Environment]:
    """원인의 확정 조건을 만족하고 나머지는 이상 없는 환경 전부.

    허용 값이 여럿인 조건이 있으면 그 조합마다 환경 하나를 만든다.
    """
    cause = next(c for c in seed["causes"] if c["id"] == cause_id)
    checks = required_checks(cause)
    slots = [
        (index, name, allowed_values(expected))
        for index, ref in enumerate(checks)
        for name, expected in ref["when"].items()
    ]
    environments = []
    for choice in product(*(values for _, _, values in slots)):
        entries = [
            {"check": ref["check"], "args": ref.get("args"), "observations": {}}
            for ref in checks
        ]
        for (index, name, _), value in zip(slots, choice):
            entries[index]["observations"][name] = value
        environments.append(Environment(seed, entries))
    return environments


def implies(seed: dict, strong_id: str, weak_id: str) -> bool:
    """strong이 확정되는 모든 경우에 weak도 확정되면 참.

    weak의 조건이 모두 strong의 조건에 들어 있고, 허용 값도 strong 쪽이 같거나 더 좁을 때다.
    이러면 판정 근거로 두 원인을 가를 수 없다.
    """
    def conditions(cause_id: str) -> dict[tuple[str, str, str], set]:
        cause = next(c for c in seed["causes"] if c["id"] == cause_id)
        return {
            (ref["check"], args_key(ref.get("args")), name): set(allowed_values(expected))
            for ref in required_checks(cause)
            for name, expected in ref["when"].items()
        }

    strong, weak = conditions(strong_id), conditions(weak_id)
    return all(key in strong and strong[key] <= values for key, values in weak.items())
