"""시나리오 검증기와 심각도 계산 검증. Neo4j 없이 돈다."""

import copy

import pytest

from common.rules.severity import base_severity, escalate
from eval.validate import expected_procedures, load_scenarios, validate_all, validate_scenario
from ontology.seed import load_seed


@pytest.fixture(scope="module")
def seed():
    return load_seed("org-a")


@pytest.fixture(scope="module")
def scenarios():
    return {s["id"]: s for s in load_scenarios("dev")}


def test_all_scenarios_are_valid(seed):
    assert validate_all(seed) == []


def test_dev_set_covers_every_symptom(scenarios):
    """dev 초안은 증상 8개를 모두 한 번 이상 다룬다."""
    assert {s["expected"]["symptom"] for s in scenarios.values()} == {f"S{i}" for i in range(1, 9)}
    assert all(s["split"] == "dev" for s in scenarios.values())


@pytest.mark.parametrize("scope,core,expected", [
    ("all_service", True, "SEV1"), ("all_service", False, "SEV2"),
    ("one_world", True, "SEV2"), ("one_world", False, "SEV3"),
    ("some_users", True, "SEV3"), ("some_users", False, "SEV4"),
])
def test_severity_table(scope, core, expected):
    """설계 3단계 4-3의 표."""
    assert base_severity(scope, core) == expected


def test_minimum_severity_only_raises():
    assert base_severity("some_users", True, "SEV2") == "SEV2"   # SEV3을 SEV2로 올린다
    assert base_severity("all_service", True, "SEV2") == "SEV1"  # 이미 더 높으면 그대로
    assert [escalate(s) for s in ("SEV4", "SEV3", "SEV2", "SEV1")] == ["SEV3", "SEV2", "SEV1", "SEV1"]


@pytest.mark.parametrize("call", [
    lambda: base_severity("galaxy", True),
    lambda: base_severity("some_users", True, "SEV9"),
    lambda: escalate("SEV0"),
])
def test_severity_rejects_undefined_values(call):
    with pytest.raises(ValueError):
        call()


def test_procedures_follow_severity_and_changes(seed, scenarios):
    # SEV3은 긴급 공지와 긴급 점검을 열지 않는다
    assert expected_procedures(seed, scenarios["SC-S8-01"]) == {"P10"}
    # SEV2는 P3, P4를 열고, 원인 위치 시스템에 최근 변경이 있으면 롤백(P5)도 연다
    assert expected_procedures(seed, scenarios["SC-S7-01"]) == {"P3", "P4", "P5", "P10"}
    no_change = copy.deepcopy(scenarios["SC-S7-01"])
    no_change["environment"]["changes"] = []
    assert expected_procedures(seed, no_change) == {"P3", "P4", "P10"}
    # 조건이 붙은 절차는 그 조건일 때만 연다
    assert "P7" in expected_procedures(seed, scenarios["SC-S4-01"])
    assert "P6" not in expected_procedures(seed, scenarios["SC-S4-01"])


def test_change_outside_window_does_not_open_rollback(seed, scenarios):
    for at in ("2026-12-14T18:04+09:00", "2026-12-15T18:41+09:00"):  # 24시간보다 전, 증상 시작 뒤
        scenario = copy.deepcopy(scenarios["SC-S7-01"])
        scenario["environment"]["changes"][0]["at"] = at
        assert "P5" not in expected_procedures(seed, scenario), at


def broken(scenarios, sid, mutate):
    scenario = copy.deepcopy(scenarios[sid])
    mutate(scenario)
    return scenario


@pytest.mark.parametrize("sid,mutate,message", [
    # 환경과 정답이 어긋난 경우
    ("SC-S1-01", lambda s: s["environment"]["checks"].pop(1), "확정되는 원인이 정답과 다르다"),
    ("SC-S1-01", lambda s: s["expected"].update(root_cause="C01"), "확정되는 원인이 정답과 다르다"),
    ("SC-S6-01", lambda s: s["environment"]["checks"][1]["observations"].update(changed="no"), "확정되는 원인"),
    ("SC-S1-01", lambda s: s["expected"].update(root_cause="C04"), "원인 후보가 아니다"),
    # 규칙으로 계산되는 라벨
    ("SC-S1-01", lambda s: s["expected"].update(ruled_out=["C01"]), "ruled_out"),
    ("SC-S1-01", lambda s: s["expected"].update(required_checks=["E01"]), "required_checks"),
    ("SC-S2-02", lambda s: s["expected"].update(primary_team="TM09"), "primary_team"),
    ("SC-S5-01", lambda s: s["expected"].update(severity="SEV3"), "severity"),
    ("SC-S8-01", lambda s: s["expected"].update(scope="one_world", severity="SEV2"), "severity"),
    ("SC-S8-01", lambda s: s["expected"].update(procedures=["P3", "P10"]), "procedures"),
    ("SC-S4-01", lambda s: s["expected"].update(resolved_conditions=["미지급"]), "procedures"),
    ("SC-S4-01", lambda s: s["expected"].update(resolved_conditions=["과지급", "환불"]), "resolved_conditions"),
    ("SC-S4-02", lambda s: s["expected"]["auto_actions"].append("A10"), "auto_actions"),
    ("SC-S4-02", lambda s: s["expected"].update(approval_actions=["A15"]), "approval_actions"),
    ("SC-S4-02", lambda s: s["expected"].update(forbidden_actions=[]), "forbidden_actions"),
    # 형식과 참조
    ("SC-S1-01", lambda s: s["reports"][0].update(channel="CH99"), "채널"),
    ("SC-S1-01", lambda s: s["reports"][0].update(at="2026-12-07T20:14"), "시간대"),
    ("SC-S1-01", lambda s: s["expected"].update(incident_groups=[[0, 1]]), "incident_groups"),
    ("SC-S1-01", lambda s: s["expected"].update(symptom="S9"), "증상"),
    ("SC-S1-01", lambda s: s.update(type="T9"), "type"),
    ("SC-S1-01", lambda s: s.pop("text_source"), "text_source"),
    ("SC-S1-01", lambda s: s["environment"]["checks"][0]["observations"].update(approval="maybe"), "environment.checks"),
    ("SC-S1-01", lambda s: s["environment"].update(checks=None), "environment.checks"),
    ("SC-S7-01", lambda s: s["environment"]["changes"][0].update(kind="hotfix"), "종류"),
    ("SC-S7-01", lambda s: s["environment"]["changes"][0].update(system="거래소"), "시스템"),
    ("SC-S7-01", lambda s: s["environment"]["changes"][0].update(at="어제"), "시간대"),
])
def test_broken_scenarios_are_reported(seed, scenarios, sid, mutate, message):
    errors = validate_scenario(seed, broken(scenarios, sid, mutate))
    assert errors and any(message in e for e in errors), errors


def test_escalated_scenario_is_one_level_higher(seed, scenarios):
    scenario = broken(scenarios, "SC-S8-01", lambda s: s["expected"].update(escalated=True))
    assert any("severity는 SEV2" in e for e in validate_scenario(seed, scenario))
