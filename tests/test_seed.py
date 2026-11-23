"""시드 YAML 검증. Neo4j 없이 돈다.

개수, 참조 무결성, 허용 값, 담당 팀 결정(R5), 구색 금지(설계 3단계 9장)를 본다.
"""

import copy

import pytest

from ontology.conditions import (
    RULE_OUT, ConditionError, Environment, confirmed_causes, environments_for,
    evaluate_cause, implies, validate_cause_conditions,
)
from ontology.seed import load_seed, resolve_owner_teams, validate_seed


@pytest.fixture(scope="module")
def seed():
    return load_seed("org-a")


def ids(items):
    return [item["id"] for item in items]


# 설계 3단계 2-1의 MVP 개수
EXPECTED_COUNTS = {
    "teams": 12,
    "roles": 6,
    "channels": 4,
    "systems": 22,
    "symptoms": 8,
    "causes": 21,
    "evidence_checks": 15,
    "procedures": 10,
    "actions": 17,
    "severities": 4,
}

# 설계 3단계 5장 표의 담당 팀
EXPECTED_OWNER = {
    "C01": "플랫폼팀", "C02": "서버개발팀", "C03": "게임사업PM팀",
    "C04": "플랫폼팀", "C05": "서버개발팀", "C06": "인프라운영팀",
    "C07": "콘텐츠기획팀", "C08": "서버개발팀", "C09": "클라이언트개발팀", "C10": "인프라운영팀",
    "C11": "콘텐츠기획팀", "C12": "서버개발팀",
    "C13": "서버개발팀", "C14": "보안팀",
    "C15": "게임사업PM팀", "C16": "시스템기획팀", "C17": "서버개발팀",
    "C18": "서버개발팀", "C19": "인프라운영팀",
    "C20": "인프라운영팀", "C21": "서버개발팀",
}

OWNS_SCOPES = {"logic", "data", "config", "operation", "execution"}
LOCATED_IN_SCOPES = {"logic", "data", "config", "operation", "execution"}


@pytest.mark.parametrize("key,count", EXPECTED_COUNTS.items())
def test_counts(seed, key, count):
    assert len(seed[key]) == count


@pytest.mark.parametrize("key", EXPECTED_COUNTS)
def test_ids_are_unique(seed, key):
    assert len(set(ids(seed[key]))) == len(seed[key])


def test_sequential_ids(seed):
    assert ids(seed["symptoms"]) == [f"S{i}" for i in range(1, 9)]
    assert ids(seed["causes"]) == [f"C{i:02d}" for i in range(1, 22)]
    assert ids(seed["evidence_checks"]) == [f"E{i:02d}" for i in range(1, 16)]
    assert ids(seed["procedures"]) == [f"P{i}" for i in range(1, 11)]
    assert ids(seed["actions"]) == [f"A{i:02d}" for i in range(1, 18)]
    assert ids(seed["severities"]) == ["SEV1", "SEV2", "SEV3", "SEV4"]


def test_references_exist(seed):
    teams, roles, systems = set(ids(seed["teams"])), set(ids(seed["roles"])), set(ids(seed["systems"]))
    causes, checks = set(ids(seed["causes"])), set(ids(seed["evidence_checks"]))
    procedures, actions = set(ids(seed["procedures"])), set(ids(seed["actions"]))

    for ch in seed["channels"]:
        assert ch["handled_first_by"] in roles
        assert ch["source_system"] is None or ch["source_system"] in systems
    for ra in seed["role_assignments"]:
        assert ra["role"] in roles and ra["team"] in teams
    for s in seed["systems"]:
        assert all(o["team"] in teams for o in s["owners"])
        assert all(d in systems and d != s["id"] for d in s["depends_on"])
        assert all(c["team"] in teams and c["condition"] for c in s.get("collaborators", []))
    for sym in seed["symptoms"]:
        assert all(x in systems for x in sym["observed_in"])
        assert all(c["cause"] in causes for c in sym["causes"])
    for c in seed["causes"]:
        assert c["located_in"]["system"] in systems
        assert all(x["check"] in checks for x in c["checked_by"])
        assert all(x["procedure"] in procedures for x in c["resolved_by"])
    for e in seed["evidence_checks"]:
        assert all(x in systems for x in e["source_systems"])
        for key in ("judged_by_team", "performed_by_team"):
            assert e.get(key) is None or e[key] in teams
    for p in seed["procedures"]:
        assert p["led_by"] is None or p["led_by"] in teams
        assert all(a in actions for a in p["actions"])
    for a in seed["actions"]:
        assert a["executed_by"] is None or a["executed_by"] in teams
        assert all(ap["team"] in teams for ap in a["approvers"])
        assert a.get("uses_system") is None or a["uses_system"] in systems
    for sev in seed["severities"]:
        assert all(p in procedures for p in sev["triggers"])
    assert seed["escalation_procedure"] in procedures


def test_allowed_values(seed):
    for t in seed["teams"]:
        assert t["axis"] in {"project", "common"}
    for ch in seed["channels"]:
        assert ch["kind"] in {"user", "internal"}
    for s in seed["systems"]:
        assert s["category"] in {"공통", "게임", "운영"}
        assert s["evidence_basis"] and set(s["evidence_basis"]) <= {"공개 자료 확인", "경험", "일반 가정"}
        for o in s["owners"]:
            assert o["scopes"] and set(o["scopes"]) <= OWNS_SCOPES
    for sym in seed["symptoms"]:
        assert isinstance(sym["core_function"], bool)
        assert all(c["prior"] in {"높음", "중간", "낮음"} for c in sym["causes"])
    for c in seed["causes"]:
        assert c["located_in"]["scope"] in LOCATED_IN_SCOPES
    for e in seed["evidence_checks"]:
        assert e["performed_by"] in {"agent", "human"}
    for a in seed["actions"]:
        assert a["kind"] in {"response", "fix", "knowledge"}
        assert a["risk"] in {"low", "high"}


def test_no_two_teams_own_same_scope_and_area(seed):
    """같은 시스템의 같은 scope, area를 두 팀이 맡으면 담당이 갈린다."""
    for s in seed["systems"]:
        seen = set()
        for o in s["owners"]:
            for scope in o["scopes"]:
                key = (scope, o.get("area"))
                assert key not in seen, (s["id"], key)
                seen.add(key)


@pytest.mark.parametrize("cause_id,team_name", EXPECTED_OWNER.items())
def test_cause_resolves_to_one_owner_team(seed, cause_id, team_name):
    """모든 원인은 1차 담당 팀 하나로 정해져야 한다 (R5, 담당 핑퐁 방지)."""
    names = {t["id"]: t["name"] for t in seed["teams"]}
    owners = resolve_owner_teams(seed, cause_id)
    assert [names[o] for o in owners] == [team_name]


def test_action_risk_rules(seed):
    """고위험 조치는 자동 실행하지 않고 승인권자가 있어야 한다 (R6)."""
    for a in seed["actions"]:
        if a["risk"] == "high":
            assert a["auto_executable"] is False
            assert a["approvers"] or a.get("approver_rule"), a["id"]
        else:
            assert a["auto_executable"] is True
            assert not a["approvers"] and a["executed_by"] is None, a["id"]
    high = {a["id"] for a in seed["actions"] if a["risk"] == "high"}
    assert high == {"A07", "A08", "A09", "A10", "A11", "A12", "A15", "A17"}
    real = {a["id"] for a in seed["actions"] if not a["simulated"]}
    assert real == {"A16", "A17"}


def test_every_cause_opens_p10(seed):
    """원인이 확정되면 P10을 공통으로 연다 (R7)."""
    for c in seed["causes"]:
        assert "P10" in [x["procedure"] for x in c["resolved_by"]], c["id"]


def test_every_action_belongs_to_a_procedure(seed):
    used = {a for p in seed["procedures"] for a in p["actions"]}
    assert used == set(ids(seed["actions"]))


def test_no_unused_nodes(seed):
    """구색 금지: 원인, 판정 근거, 절차는 모두 어딘가에서 쓰여야 한다."""
    used_causes = {c["cause"] for s in seed["symptoms"] for c in s["causes"]}
    assert used_causes == set(ids(seed["causes"]))

    used_checks = {x["check"] for c in seed["causes"] for x in c["checked_by"]}
    # E15는 원인 판정이 아니라 영향 범위 추정과 중복 판정에 공통으로 쓴다
    assert used_checks | {"E15"} == set(ids(seed["evidence_checks"]))

    used_procedures = {x["procedure"] for c in seed["causes"] for x in c["resolved_by"]}
    used_procedures |= {p for sev in seed["severities"] for p in sev["triggers"]}
    used_procedures |= {seed["escalation_procedure"], "P1", "P9"}  # 접수와 사후 리뷰는 모든 사건 공통
    assert used_procedures == set(ids(seed["procedures"]))


def test_all_teams_appear(seed):
    """구색 금지: 12개 팀 전원이 장애 경로에 등장한다 (설계 3단계 9-1)."""
    appearing = set()
    for c in seed["causes"]:
        appearing.update(resolve_owner_teams(seed, c["id"]))
    for a in seed["actions"]:
        appearing.update(ap["team"] for ap in a["approvers"])
        if a["executed_by"]:
            appearing.add(a["executed_by"])
    for e in seed["evidence_checks"]:
        appearing.update(t for t in (e.get("judged_by_team"), e.get("performed_by_team")) if t)
    appearing.update(ra["team"] for ra in seed["role_assignments"])
    appearing.update(p["led_by"] for p in seed["procedures"] if p["led_by"])
    assert appearing == set(ids(seed["teams"]))


def test_all_systems_appear(seed):
    """구색 금지: 22개 시스템 전원이 장애 경로에 등장한다 (설계 3단계 9-2)."""
    appearing = set()
    for sym in seed["symptoms"]:
        appearing.update(sym["observed_in"])
    for c in seed["causes"]:
        appearing.add(c["located_in"]["system"])
    for e in seed["evidence_checks"]:
        appearing.update(e["source_systems"])
    for a in seed["actions"]:
        if a.get("uses_system"):
            appearing.add(a["uses_system"])
    appearing.update(ch["source_system"] for ch in seed["channels"] if ch["source_system"])
    assert appearing == set(ids(seed["systems"]))


def test_game_systems_depend_on_logging_and_monitoring(seed):
    """모든 게임 시스템은 로그 수집과 모니터링에 의존한다 (설계 2단계 3-2)."""
    for s in seed["systems"]:
        if s["category"] == "게임":
            assert {"SY03", "SY04"} <= set(s["depends_on"]), s["id"]


def test_cause_conditions_use_defined_fields_and_values(seed):
    validate_cause_conditions(seed)
    validate_seed(seed)


def test_tool_names_are_unique(seed):
    tools = [e["tool"] for e in seed["evidence_checks"]]
    assert len(tools) == len(set(tools))


def test_each_cause_uses_a_check_once(seed):
    for c in seed["causes"]:
        checks = [x["check"] for x in c["checked_by"]]
        assert len(checks) == len(set(checks)), c["id"]


def test_observation_defaults_are_defined_values(seed):
    for e in seed["evidence_checks"]:
        fields = e.get("observations", {})
        assert fields or e.get("default_detail"), e["id"]
        for name, spec in fields.items():
            assert spec["default"] in spec["values"], (e["id"], name)
            assert all(isinstance(text, str) and text for text in spec["values"].values())


def test_every_observation_field_is_used_by_some_cause(seed):
    """구색 금지: 원인 판정에 쓰이지 않는 관측 필드를 두지 않는다."""
    used = {(x["check"], name) for c in seed["causes"] for x in c["checked_by"] for name in x.get("when", {})}
    defined = {(e["id"], name) for e in seed["evidence_checks"] for name in e.get("observations", {})}
    assert defined == used


def test_causes_of_same_symptom_are_distinguishable(seed):
    """같은 증상의 한 원인이 확정될 때 다른 원인도 반드시 확정되면 판정 근거로 가를 수 없다.

    조건이 같은 경우(이전의 C15, C16)와 한쪽 조건이 다른 쪽에 통째로 들어 있는 경우를 잡는다.
    """
    for sym in seed["symptoms"]:
        ids = [ref["cause"] for ref in sym["causes"]]
        for strong in ids:
            for weak in ids:
                if strong != weak:
                    assert not implies(seed, strong, weak), (sym["id"], strong, weak)


def test_nothing_is_confirmed_when_everything_is_normal(seed):
    env = Environment(seed)
    for sym in seed["symptoms"]:
        assert confirmed_causes(seed, env, sym["id"]) == [], sym["id"]


@pytest.mark.parametrize("cause_id", list(EXPECTED_OWNER))
def test_cause_is_the_only_one_confirmed_by_its_own_conditions(seed, cause_id):
    """원인의 조건만 만족하고 나머지가 이상 없으면, 같은 증상에서 그 원인만 확정된다.

    허용 값이 여럿인 조건은 모든 조합을 돌려 본다.
    """
    envs = environments_for(seed, cause_id)
    assert envs
    for env in envs:
        for sym in seed["symptoms"]:
            if cause_id in [ref["cause"] for ref in sym["causes"]]:
                assert confirmed_causes(seed, env, sym["id"]) == [cause_id], (sym["id"], env.entries)


def test_network_fault_alone_does_not_confirm_resource_shortage(seed):
    """네트워크만 이상하고 자원에 여유가 있으면 서버 자원 부족(C20)이 아니다."""
    def env(status):
        return Environment(seed, [
            {"check": "E09", "args": {"system": "SY05"}, "observations": {"status": status}},
            {"check": "E14", "observations": {"overlaps_large_content": "yes"}},
        ])

    assert evaluate_cause(seed, env("network_fault"), "C20") == RULE_OUT
    assert confirmed_causes(seed, env("resource_limit"), "S8") == ["C20"]
    # 서버 인프라 장애(C06)는 자원 한계와 네트워크 이상 어느 쪽이든 확정된다
    assert confirmed_causes(seed, env("network_fault"), "S2") == ["C06"]
    assert confirmed_causes(seed, env("resource_limit"), "S2") == ["C06"]


def test_design_example_scenario(seed):
    """설계 4단계 3-4 예시: 결제 승인 기록은 있고 우편함 지급 기록이 없다."""
    env = Environment(seed, [
        {"check": "E01", "observations": {"approval": "present"}, "detail": "두 유저 모두 결제 승인 기록 있음"},
        {"check": "E02", "observations": {"delivery": "absent"}},
        {"check": "E07", "args": {"system": "SY12"}, "observations": {"changed": "no"}},
    ])
    assert confirmed_causes(seed, env, "S1") == ["C02"]
    assert [evaluate_cause(seed, env, c) for c in ("C01", "C03")] == [RULE_OUT, RULE_OUT]
    assert env.detail("E01") == "두 유저 모두 결제 승인 기록 있음"
    assert env.detail("E02") == "우편함 지급 기록이 없다"
    assert env.detail("E07", {"system": "SY12"}) == "증상 시작 전에 캐시 상점 설정 변경 이력이 없다"


@pytest.mark.parametrize("entry", [
    {"check": "E99"},
    {"check": "E01", "observations": {"approval": "maybe"}},
    {"check": "E01", "observations": {"unknown_field": "present"}},
    {"check": "E04", "observations": {"error_log": "present"}},                      # 인자 누락
    {"check": "E04", "args": {"system": "SY99"}},                                   # 없는 시스템
    {"check": "E07", "args": {"system": "SY10"}},                                   # 허용 값 밖
    {"check": "E01", "args": {"system": "SY10"}},                                   # 인자가 없는 조회
])
def test_environment_rejects_undefined_entries(seed, entry):
    with pytest.raises(ConditionError):
        Environment(seed, [entry])


def test_environment_rejects_duplicate_entries(seed):
    with pytest.raises(ConditionError, match="두 번"):
        Environment(seed, [{"check": "E01"}, {"check": "E01"}])


def test_every_check_is_either_condition_or_reference(seed):
    for c in seed["causes"]:
        for x in c["checked_by"]:
            assert bool(x.get("when") and x.get("expect")) != bool(x.get("note")), (c["id"], x["check"])


def test_reference_checks_do_not_decide(seed):
    """참고용 근거는 확정과 배제에 쓰지 않는다 (설계 3단계 5장 S6, S7)."""
    reference = {(c["id"], x["check"]) for c in seed["causes"] for x in c["checked_by"] if "note" in x}
    assert reference == {("C14", "E10"), ("C17", "E06")}
    # 확률 모듈 로직 오류는 배포 이력이 없어도 수치 비교만으로 확정된다
    env = Environment(seed, [{"check": "E08", "observations": {"public_vs_table": "same", "table_vs_actual": "different"}}])
    assert confirmed_causes(seed, env, "S6") == ["C17"]
    # 어뷰징은 재화 유통량이 늘지 않아도 계정 집중만으로 확정된다
    env = Environment(seed, [{"check": "E11", "observations": {"concentration": "concentrated"}}])
    assert confirmed_causes(seed, env, "S7") == ["C14"] and confirmed_causes(seed, env, "S5") == ["C14"]


@pytest.mark.parametrize("breaker", [
    lambda ref: ref.pop("expect"),                    # 확정 조건인데 설명이 없다
    lambda ref: ref.update(note="참고"),              # 확정 조건과 참고용 표시가 함께 있다
    lambda ref: (ref.pop("when"), ref.pop("expect")),  # 둘 다 없다
    lambda ref: (ref.pop("when"), ref.update(note="참고")),  # 참고용인데 expect가 남았다
    lambda ref: ref.update(note=""),                                       # 금지된 필드는 빈 값이어도 안 된다
    lambda ref: (ref.pop("when"), ref.update(note="참고", expect="")),     # 참고용에 빈 expect
    lambda ref: (ref.pop("when"), ref.pop("expect"), ref.update(note="")),  # 참고용인데 쓰임이 비었다
])
def test_malformed_check_entries_are_rejected(seed, breaker):
    broken = copy.deepcopy(seed)
    breaker(broken["causes"][1]["checked_by"][0])
    with pytest.raises(ConditionError):
        validate_cause_conditions(broken)


def test_cause_without_conditions_is_rejected(seed):
    broken = copy.deepcopy(seed)
    c17 = next(c for c in broken["causes"] if c["id"] == "C17")
    c17["checked_by"] = [x for x in c17["checked_by"] if "note" in x]
    with pytest.raises(ConditionError, match="확정 조건이 없다"):
        validate_cause_conditions(broken)
