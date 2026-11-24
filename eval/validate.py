"""시나리오 파일을 읽고, 정답 라벨이 온톨로지와 맞는지 확인한다 (설계 4단계 3장).

실행: python -m eval.validate

시나리오를 에이전트에 넣기 전에 도는 검사다. 가상 데이터와 정답을 사람이 손으로 맞추기 때문에
어긋난 시나리오가 섞이면 틀린 기준으로 평가하게 된다. LLM을 부르지 않는다.

확인하는 것
- 형식과 참조: 정의된 채널, 증상, 원인, 팀, 절차, 조치인지
- 환경: 판정 근거의 필드와 값, 변경 이력의 종류와 시각
- 원인: 환경을 확정 조건에 대입했을 때 정답 원인만 확정되고 나머지 후보는 배제되는지
  (원인이 확정되지 않아야 정답인 사건은 모든 후보가 배제되는지)
- 규칙으로 계산되는 라벨: 필수 판정 근거, 담당 팀, 심각도, 절차, 조치가 계산한 값과 같은지

라벨은 사건 단위로 본다 (eval/labels.py). 사건이 여럿이면 사건마다 위 검사를 한다
"""

import sys
from datetime import datetime
from pathlib import Path

import yaml

from common.rules.severity import ESCALATION_COUNT, ESCALATION_WINDOW, SCOPES, base_severity, escalate
from common.writers.events import CHANGE_KINDS
from ontology.conditions import (
    CONFIRM, ConditionError, Environment, allowed_values, evaluate_cause, procedure_conditions, required_checks,
)
from ontology.seed import load_seed, resolve_owner_teams
from eval.labels import INCIDENT_FIELDS, is_multi, label_groups, label_incidents

SCENARIO_DIR = Path(__file__).parent / "scenarios"
SPLITS = ("dev", "test")
TYPES = {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
SOURCES = {"설계", "경험"}

# 접수와 분류(P1)는 모든 사건에서 연다
INTAKE_PROCEDURE = "P1"
# 원인 수정과 반영(P10)은 원인이 확정되면 항상 연다 (R7)
FIX_PROCEDURE = "P10"
URGENT_NOTICE, URGENT_MAINTENANCE = "P3", "P4"
# 등급이 상향된 사건은 에스컬레이션(P2)을 연다 (R3)
ESCALATION_PROCEDURE = "P2"

# 변경 이력의 종류마다 그 변경이 보여야 하는 판정 근거와 관측 값 (3단계 6장).
# 시스템 인자가 있는 근거는 변경된 시스템으로 조회한 결과를 본다
CHANGE_EVIDENCE = {
    "deploy": ("E06", "deploy", {"success", "partial", "failed"}, "system"),
    "patch": ("E06", "deploy", {"success", "partial", "failed"}, "system"),
    "table_change": ("E05", "changed", {"yes"}, None),
    "event_config": ("E07", "changed", {"yes"}, "system"),
    "product_config": ("E07", "changed", {"yes"}, "system"),
}
URGENT_LEVELS = {"SEV1", "SEV2"}


def load_scenarios(split: str) -> list[dict]:
    """한 묶음(dev 또는 test)의 시나리오를 파일 이름 순으로 읽는다."""
    return [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((SCENARIO_DIR / split).glob("*.yaml"))
    ]


def _aware(value) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.utcoffset() is not None else None


def started_at(scenario: dict) -> datetime:
    """증상 시작 시각. 사건으로 묶인 제보 중 가장 이른 시각으로 본다.

    분류하지 않는 제보(unclassified)는 증상과 관계가 없어 시작 시각에 넣지 않는다
    """
    grouped = {i for group in label_groups(scenario["expected"]) for i in group}
    return min(_aware(r["at"]) for i, r in enumerate(scenario["reports"]) if i in grouped)


def _condition_errors(seed: dict, expected: dict, env: Environment) -> list[str]:
    """절차 조건(미지급, 과지급) 라벨이 조건의 판정 근거 관측 값과 맞는지. 조건이 붙은 절차를 가진 원인만 본다.

    조건마다 판정 근거와 값은 시드의 해결 절차에 있다 (causes.yaml, M3 점검). 에이전트도 같은 판정 근거를 조회한다.
    expected는 사건 하나의 라벨이다
    """
    cause = next((c for c in seed["causes"] if c["id"] == expected.get("root_cause")), None)
    if cause is None:
        return []
    used = {ref["condition"] for ref in cause["resolved_by"] if ref.get("condition")}
    if not used:
        return []
    definitions = procedure_conditions(seed)

    def holds(name: str) -> bool:
        definition = definitions[name]
        observed = env.observe(definition["check"], definition.get("args"))
        return all(observed[k] in allowed_values(v) for k, v in definition["when"].items())

    derived = {name for name in used if holds(name)}
    labelled = set(expected.get("resolved_conditions") or []) & used
    # 어떤 조건도 성립하지 않으면(관측이 정상이면) 조건 절차를 열지 않는 것이 정답이다 (PR #22 리뷰)
    if labelled != derived:
        return [f"resolved_conditions는 조건의 판정 근거 관측 값으로 {sorted(derived)}여야 한다 (라벨 {sorted(labelled)})"]
    return []


def _escalation_evidence(scenario: dict, reports: list[int]) -> bool:
    """사건의 접수가 상향 기준(R3)을 넘는지. 창은 각 접수 시각을 끝으로 하는 시간이다."""
    times = sorted(_aware(scenario["reports"][i]["at"]) for i in reports)
    return any(sum(1 for t in times if end - ESCALATION_WINDOW <= t <= end) >= ESCALATION_COUNT for end in times)


def _change_errors(seed: dict, scenario: dict, env: Environment) -> list[str]:
    """변경 이력과 판정 근거의 관측 값이 서로 맞는지 (3단계 6장).

    증상 시작 전 변경이 있으면 그 변경을 보는 조회도 변경이 있다고 답해야 하고, 거꾸로도 같다.
    어긋나면 원인 점수의 변경 가중(R4)과 판정 근거가 서로 다른 말을 한다.
    """
    errors = []
    start = started_at(scenario)
    before = [c for c in scenario["environment"]["changes"] if _aware(c["at"]) <= start]
    for change in before:
        check_id, field_name, changed_values, arg = CHANGE_EVIDENCE[change["kind"]]
        args = {arg: change["system"]} if arg else None
        if arg is None:
            # 테이블 변경은 영역 인자로 조회한다. 어느 영역이든 변경이 보이면 맞다
            seen = any(e["check"] == check_id and (e.get("observations") or {}).get(field_name) in changed_values
                       for e in scenario["environment"]["checks"])
        else:
            try:
                seen = env.observe(check_id, args)[field_name] in changed_values
            except ConditionError as exc:
                errors.append(f"{change['kind']} 변경({change['system']})을 {check_id}로 조회할 수 없다: {exc}")
                continue
        if not seen:
            errors.append(f"증상 시작 전 {change['kind']} 변경({change['system']})이 있는데 {check_id} 조회는 변경이 없다고 답한다")
    for entry in scenario["environment"]["checks"]:
        matches = [(kind, spec) for kind, spec in CHANGE_EVIDENCE.items()
                   if entry["check"] == spec[0] and (entry.get("observations") or {}).get(spec[1]) in spec[2]]
        if not matches:
            continue
        system = (entry.get("args") or {}).get("system")
        if not any(c["kind"] == kind and (spec[3] is None or c["system"] == system) for kind, spec in matches for c in before):
            errors.append(f"{entry['check']} 조회는 증상 시작 전 변경이 있다고 답하는데 그런 변경 이력이 없다 ({entry.get('args') or {}})")
    return errors


def expected_procedures(seed: dict, labels: dict) -> set[str]:
    """정답 원인, 심각도, 환경으로 열어야 할 절차를 계산한다.

    labels는 사건 하나의 라벨이다. 시나리오를 주면 첫 사건의 라벨을 쓴다.
    원인이 확정되지 않아야 정답인 사건은 원인에 걸린 절차와 원인 수정(P10)을 열지 않는다.
    에이전트도 원인이 확정되지 않으면 접수, 상향, 등급 절차만 후보로 둔다 (5단계 4-1)

    설계에 적힌 것: P10은 항상 (R7), SEV1과 SEV2는 긴급 공지 P3 (R3 TRIGGERS)와 긴급 점검 P4 후보 (R7).
    아래는 설계에 규칙이 없어 정한 것이다 (2026-11-23 검토).
    - 원인에 P3, P4가 연결돼 있어도 SEV3, SEV4에서는 열지 않는다.
      가정이다. 일부 유저의 범위(한두 명인지 여러 명인지)를 가르는 기준이 없다.
      접수가 늘어 등급이 상향되면 SEV2가 되어 열린다
    - 사람만 여는 절차(human_only, 롤백 P5)는 원인에 연결돼 있어도 열지 않는다
    - 그 밖의 절차(보상 P6, 회수 P7, 확률 표기 정정 P8)는 원인에 연결돼 있으면 연다.
      조건이 붙어 있으면(미지급이면 P6, 과지급이면 P7) 시나리오의 resolved_conditions에 그 조건이 있을 때만 연다
    """
    expected = label_incidents(labels["expected"])[0] if "expected" in labels else labels
    severity = next(s for s in seed["severities"] if s["id"] == expected["severity"])
    human_only = {p["id"] for p in seed["procedures"] if p.get("human_only")}

    procedures = set(severity["triggers"])
    if expected.get("escalated"):
        procedures.add(ESCALATION_PROCEDURE)
    if expected["severity"] in URGENT_LEVELS:
        procedures.add(URGENT_MAINTENANCE)
    if expected.get("root_cause") is None:
        return procedures
    procedures.add(FIX_PROCEDURE)
    cause = next(c for c in seed["causes"] if c["id"] == expected["root_cause"])
    for ref in cause["resolved_by"]:
        procedure, condition = ref["procedure"], ref.get("condition")
        if procedure in (URGENT_NOTICE, URGENT_MAINTENANCE) or procedure in human_only:
            continue
        if condition is None or condition in expected.get("resolved_conditions", []):
            procedures.add(procedure)
    return procedures


LIST_LABELS = (
    "ruled_out", "required_checks", "resolved_conditions",
    "procedures", "auto_actions", "approval_actions", "forbidden_actions",
)
# 원인이 확정되지 않아야 정답인 사건은 이 두 라벨을 비운다(null). 사건 여럿 형식에서만 쓴다
UNCONFIRMED_LABELS = ("root_cause", "primary_team")


def _int_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in value)


def _label_shape_errors(labels: dict, where: str, allow_unconfirmed: bool) -> list[str]:
    """사건 하나의 라벨 모양. where는 메시지에 붙일 위치(expected 또는 expected.incidents[i])다."""
    errors = []
    for name in ("symptom", "scope", "severity", *UNCONFIRMED_LABELS):
        value = labels.get(name)
        if isinstance(value, str):
            continue
        if allow_unconfirmed and name in UNCONFIRMED_LABELS and name in labels and value is None:
            continue
        kind = "문자열이거나 null" if allow_unconfirmed and name in UNCONFIRMED_LABELS else "문자열"
        errors.append(f"{where}.{name}은 {kind}이어야 한다")
    # 목록의 원소 형식까지 본다. 뒤에서 정렬하고 집합으로 바꾸므로 형식이 섞이면 예외가 난다
    for name in LIST_LABELS:
        value = labels.get(name)
        if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            errors.append(f"{where}.{name}은 문자열의 목록이어야 한다")
    return errors


def _structure_errors(scenario) -> list[str]:
    """값의 모양을 본다. 모양이 틀리면 뒤의 검사가 예외로 멈추므로 먼저 걸러 낸다."""
    if not isinstance(scenario, dict):
        return ["(id 없음): 시나리오 파일의 최상위가 키와 값의 묶음이 아니다"]
    sid = scenario.get("id")
    if not isinstance(sid, str) or not sid:
        return ["(id 없음): id가 없거나 문자열이 아니다"]
    errors = []
    # 메타 정보도 허용 값 집합에서 찾으므로 문자열인지 먼저 본다. boundary는 없을 수 있다
    for name in ("type", "split", "source", "text_source"):
        if not isinstance(scenario.get(name), str):
            errors.append(f"{sid}: {name}은 문자열이어야 한다")
    if scenario.get("boundary") is not None and not isinstance(scenario["boundary"], str):
        errors.append(f"{sid}: boundary는 문자열이거나 비어 있어야 한다")

    def need(ok: bool, message: str) -> None:
        if not ok:
            errors.append(f"{sid}: {message}")

    def dicts(value) -> bool:
        return isinstance(value, list) and all(isinstance(item, dict) for item in value)

    def strings(items, names: tuple[str, ...], where: str) -> None:
        # ID와 시각 자리에 목록이나 묶음이 오면 뒤의 비교에서 예외가 난다
        for index, item in enumerate(items):
            for name in names:
                need(isinstance(item.get(name), str), f"{where} {index}: {name}은 문자열이어야 한다")

    reports = scenario.get("reports")
    if dicts(reports):
        strings(reports, ("channel", "reporter", "at", "text"), "제보")
    else:
        need(False, "reports는 제보 항목의 목록이어야 한다")
    environment = scenario.get("environment")
    if isinstance(environment, dict):
        need(dicts(environment.get("checks")), "environment.checks는 조회 항목의 목록이어야 한다")
        if dicts(environment.get("changes")):
            strings(environment["changes"], ("kind", "system", "at"), "변경")
        else:
            need(False, "environment.changes는 변경 항목의 목록이어야 한다")
    else:
        need(False, "environment가 없거나 키와 값의 묶음이 아니다")
    expected = scenario.get("expected")
    if not isinstance(expected, dict):
        need(False, "expected가 없거나 키와 값의 묶음이 아니다")
        return errors
    # 분류하지 않고 사람에게 넘길 제보(증상 8개 밖)의 번호. 없으면 빈 목록이다
    need(_int_list(expected.get("unclassified", [])), "expected.unclassified는 제보 번호(정수)의 목록이어야 한다")
    if is_multi(expected):
        # 사건 여럿 형식: 라벨은 사건마다 incidents 안에 적는다. 바깥에 같이 적으면 어느 쪽이 정답인지 모호하다
        mixed = sorted(name for name in (*INCIDENT_FIELDS, "incident_groups") if name in expected)
        need(not mixed, f"expected.incidents를 쓰면 라벨은 사건 안에만 적는다: {mixed}")
        incidents = expected["incidents"]
        if not dicts(incidents):
            need(False, "expected.incidents는 사건 라벨의 목록이어야 한다")
            return errors
        for index, incident in enumerate(incidents):
            where = f"expected.incidents[{index}]"
            need(_int_list(incident.get("reports")) and bool(incident.get("reports")),
                 f"{where}.reports는 제보 번호(정수)의 비지 않은 목록이어야 한다")
            for message in _label_shape_errors(incident, where, allow_unconfirmed=True):
                need(False, message)
    else:
        groups = expected.get("incident_groups")
        need(isinstance(groups, list) and all(_int_list(group) for group in groups),
             "expected.incident_groups는 제보 번호(정수) 묶음의 목록이어야 한다")
        # 사건 하나 형식은 묶음도 하나다. 사건이 여럿이면 사건마다 라벨이 달라서 incidents로 적는다
        if isinstance(groups, list) and all(_int_list(group) for group in groups):
            need(len(groups) == 1, "사건 하나 형식은 incident_groups에 묶음 하나를 적는다. 사건이 여럿이면 incidents를 쓴다")
        for message in _label_shape_errors(expected, "expected", allow_unconfirmed=False):
            need(False, message)
    return errors


def validate_scenario(seed: dict, scenario: dict) -> list[str]:
    """시나리오 하나의 문제를 문장 목록으로 돌려준다. 문제가 없으면 빈 목록."""
    errors: list[str] = _structure_errors(scenario)
    if errors:
        return errors
    sid = scenario["id"]

    def check(ok: bool, message: str) -> bool:
        if not ok:
            errors.append(f"{sid}: {message}")
        return ok

    ids = lambda key: {item["id"] for item in seed[key]}  # noqa: E731
    expected = scenario["expected"]
    environment = scenario["environment"]

    # 메타 정보
    check(scenario.get("type") in TYPES, f"type은 {sorted(TYPES)} 중 하나여야 한다")
    check(scenario.get("split") in SPLITS, "split은 dev 또는 test여야 한다")
    check(scenario.get("source") in SOURCES, f"source는 {sorted(SOURCES)} 중 하나여야 한다")
    check(bool(scenario.get("text_source")), "text_source(제보 문장을 만든 방법)가 없다")

    # 제보
    reports = scenario["reports"]
    if not check(bool(reports), "제보가 없다"):
        return errors
    for index, report in enumerate(reports):
        check(report.get("channel") in ids("channels"), f"제보 {index}: 정의되지 않은 채널이다")
        check(bool(report.get("reporter")) and bool(report.get("text")), f"제보 {index}: 요청자 또는 원문이 없다")
        check(_aware(report.get("at")) is not None, f"제보 {index}: at은 시간대가 붙은 시각이어야 한다")
    # 사건 묶음과 분류하지 않는 제보를 합쳐 모든 제보가 한 번씩 들어가야 한다
    groups = label_groups(expected)
    grouped = sorted([i for group in groups for i in group] + list(expected.get("unclassified") or []))
    check(grouped == list(range(len(reports))),
          "사건 묶음(incident_groups 또는 incidents의 reports)과 unclassified가 제보 전체를 한 번씩 담지 않는다")
    incidents = label_incidents(expected)
    # 한 시나리오의 사건끼리는 증상이 달라야 한다. 증상이 같으면 같은 환경에서 원인 후보가 겹쳐
    # 사건마다 정답 원인만 확정되는지 가릴 수 없다
    symptoms = [incident["symptom"] for incident in incidents]
    check(len(set(symptoms)) == len(symptoms), f"사건끼리 증상이 겹친다: {symptoms}")

    # 환경
    env = None
    try:
        env = Environment(seed, environment["checks"])
    except (ConditionError, KeyError, TypeError) as exc:
        check(False, f"environment.checks: {exc}")
    for index, change in enumerate(environment["changes"]):
        check(change.get("kind") in CHANGE_KINDS, f"변경 {index}: 정의되지 않은 종류다")
        check(change.get("system") in ids("systems"), f"변경 {index}: 정의되지 않은 시스템이다")
        check(_aware(change.get("at")) is not None, f"변경 {index}: at은 시간대가 붙은 시각이어야 한다")
    if errors:
        return errors
    # 사건이 없는 시나리오(범위 밖, T6)는 증상 시작 시각이 없어 변경 이력 대조를 하지 않는다
    if incidents:
        errors += _change_errors(seed, scenario, env)
    for index, incident in enumerate(incidents):
        prefix = f"사건 {index}: " if is_multi(expected) else ""
        errors += [f"{sid}: {prefix}{message}" for message in _incident_errors(seed, scenario, env, incident)]
    return errors


def _incident_errors(seed: dict, scenario: dict, env: Environment, expected: dict) -> list[str]:
    """사건 하나의 라벨이 환경과 규칙으로 계산한 값과 같은지. 문제를 문장 목록으로 돌려준다."""
    errors: list[str] = []

    def check(ok: bool, message: str) -> bool:
        if not ok:
            errors.append(message)
        return ok

    environment = scenario["environment"]
    root = expected.get("root_cause")
    if root is not None:
        errors += _condition_errors(seed, expected, env)

    # 증상과 원인
    symptom = next((s for s in seed["symptoms"] if s["id"] == expected.get("symptom")), None)
    if not check(symptom is not None, "정의되지 않은 증상이다"):
        return errors
    candidates = [ref["cause"] for ref in symptom["causes"]]
    confirmed = [c for c in candidates if evaluate_cause(seed, env, c) == CONFIRM]
    if root is None:
        # 원인이 확정되지 않아야 정답인 사건: 어떤 후보도 확정되지 않고 모두 배제된다 (2026-11-25).
        # 담당 팀은 정할 근거가 없어 비우고 사람이 판단한다. 판정 근거와 조건 절차도 없다
        check(confirmed == [], f"원인이 확정되지 않아야 정답인데 이 환경에서 {confirmed}가 확정된다")
        check(sorted(expected.get("ruled_out") or []) == sorted(candidates),
              f"원인이 확정되지 않는 사건의 ruled_out은 후보 전체 {sorted(candidates)}여야 한다")
        check(not expected.get("required_checks"), "원인이 확정되지 않는 사건의 required_checks는 비어 있어야 한다")
        check(expected.get("primary_team") is None, "원인이 확정되지 않는 사건의 primary_team은 null이어야 한다")
        check(not expected.get("resolved_conditions"), "원인이 확정되지 않는 사건의 resolved_conditions는 비어 있어야 한다")
        cause = None
    else:
        if not check(root in candidates, f"root_cause가 {symptom['id']}의 원인 후보가 아니다: {root}"):
            return errors
        check(confirmed == [root], f"이 환경에서 확정되는 원인이 정답과 다르다: 확정 {confirmed}, 정답 {root}")
        others = sorted(c for c in candidates if c != root)
        check(sorted(expected.get("ruled_out") or []) == others, f"ruled_out은 나머지 후보 {others}여야 한다")

        cause = next(c for c in seed["causes"] if c["id"] == root)
        needed = sorted(ref["check"] for ref in required_checks(cause))
        check(
            sorted(expected.get("required_checks") or []) == needed,
            f"required_checks는 정답 원인의 확정 조건 {needed}여야 한다",
        )
        owners = resolve_owner_teams(seed, root)
        check([expected.get("primary_team")] == owners, f"primary_team은 {owners}여야 한다")

    # 심각도
    escalated = expected.get("escalated", False)
    if check(isinstance(escalated, bool), "escalated는 true 또는 false여야 한다"):
        minutes = int(ESCALATION_WINDOW.total_seconds() // 60)
        # 상향 기준을 넘으면 상향이 정답이고, 넘지 않으면 상향하지 않는 것이 정답이다 (R3)
        check(escalated == _escalation_evidence(scenario, expected["reports"]),
              f"escalated는 같은 사건의 접수가 {minutes}분 안에 {ESCALATION_COUNT}건 이상일 때만 true다 (R3)")
    scope = expected.get("scope")
    if check(scope in SCOPES, f"scope는 {sorted(SCOPES)} 중 하나여야 한다"):
        severity = base_severity(scope, symptom["core_function"], symptom.get("min_severity"))
        if escalated is True:
            severity = escalate(severity)
        check(expected.get("severity") == severity, f"severity는 {severity}여야 한다 (R3)")
        # 영향 범위는 접수 내역 조회(E15) 결과로 추정한다 (3단계 6장). E15의 관측 값이 라벨의 범위와 같아야 한다.
        # 관측 값은 검증기만 읽고 에이전트는 문장만 받는다. E15는 인자가 없어 시나리오에 답이 하나뿐이다.
        # 그래서 사건이 여럿이면 모든 사건의 범위가 같아야 한다 (한계, 2026-11-25)
        observed_scope = env.observe("E15")["scope"]
        check(observed_scope == scope, f"scope가 {scope}인데 E15 관측 값은 {observed_scope}다")
        # 일부 유저보다 넓으면 에이전트가 읽을 문장도 있어야 한다
        lookup = [c for c in environment.get("checks") or [] if c.get("check") == "E15"
                  and isinstance(c.get("detail"), str) and c["detail"].strip()]
        check(scope == "some_users" or bool(lookup),
              f"scope가 {scope}면 그 근거인 E15(접수 내역 조회) detail이 환경에 있어야 한다")
    if errors:
        return errors

    # 절차와 조치
    if cause is not None:
        known_conditions = {ref["condition"] for ref in cause["resolved_by"] if ref.get("condition")}
        stray = set(expected.get("resolved_conditions") or []) - known_conditions
        check(not stray, f"resolved_conditions에 이 원인에 없는 조건이 있다: {sorted(stray)}")
    procedures = expected_procedures(seed, expected)
    check(
        set(expected.get("procedures") or []) == procedures,
        f"procedures는 {sorted(procedures, key=lambda p: int(p[1:]))}여야 한다",
    )
    actions = {a["id"]: a for a in seed["actions"]}
    in_scope = {
        action_id
        for procedure in seed["procedures"]
        if procedure["id"] in procedures | {INTAKE_PROCEDURE}
        for action_id in procedure["actions"]
    }
    low = sorted(a for a in in_scope if actions[a]["risk"] == "low")
    high = sorted(a for a in in_scope if actions[a]["risk"] == "high")
    check(sorted(expected.get("auto_actions") or []) == low, f"auto_actions는 {low}여야 한다")
    check(sorted(expected.get("approval_actions") or []) == high, f"approval_actions는 {high}여야 한다")
    check(
        sorted(expected.get("forbidden_actions") or []) == high,
        f"forbidden_actions는 승인 없이 실행하면 안 되는 조치 {high}여야 한다",
    )
    return errors


def validate_all(seed: dict) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for split in SPLITS:
        for path in sorted((SCENARIO_DIR / split).glob("*.yaml")):
            # 파일을 하나씩 읽는다. 문법이 틀린 파일이 있어도 나머지 파일은 검사한다
            try:
                scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                errors.append(f"{path.name}: 파일을 읽을 수 없다 ({type(exc).__name__})")
                continue
            try:
                errors.extend(validate_scenario(seed, scenario))
            except Exception as exc:  # noqa: BLE001
                # 모양 검사가 놓친 경우에도 한 파일 때문에 나머지 파일의 검사가 멈추지 않게 한다
                name = scenario.get("id") if isinstance(scenario, dict) else None
                errors.append(f"{name or '(id 없음)'}: 검증하다가 멈췄다 ({type(exc).__name__}: {exc})")
            # id가 없거나 모양이 틀린 파일은 위에서 보고했다. 중복과 폴더 검사는 id가 있을 때만 한다
            sid = scenario.get("id") if isinstance(scenario, dict) else None
            if not isinstance(sid, str) or not sid:
                continue
            if sid in seen:
                errors.append(f"{sid}: 같은 id가 두 번 있다")
            seen.add(sid)
            if scenario.get("split") != split:
                errors.append(f"{sid}: {split} 폴더에 있는데 split이 {scenario.get('split')}이다")
    return errors


def main() -> int:
    """모든 시나리오를 검증하고 결과를 출력한다. 문제가 있으면 1을 돌려준다."""
    problems = validate_all(load_seed("org-a"))
    for line in problems:
        print(line)
    # 파일 수는 파일을 다시 읽지 않고 센다. 읽을 수 없는 파일이 있어도 요약까지 출력한다
    total = sum(len(list((SCENARIO_DIR / split).glob("*.yaml"))) for split in SPLITS)
    print(f"시나리오 파일 {total}개, 문제 {len(problems)}건")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
