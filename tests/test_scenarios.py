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
    # SEV2는 P3, P4를 연다. 거래소 배포 직후지만 롤백(P5)은 사람만 여는 절차라 열지 않는다
    assert expected_procedures(seed, scenarios["SC-S7-01"]) == {"P3", "P4", "P10"}
    # 조건이 붙은 절차는 그 조건일 때만 연다
    assert "P7" in expected_procedures(seed, scenarios["SC-S4-01"])
    assert "P6" not in expected_procedures(seed, scenarios["SC-S4-01"])


def test_rollback_is_never_opened_by_the_agent(seed, scenarios):
    """롤백은 원인에 연결돼 있어도 어떤 시나리오의 정답에도 들어가지 않는다."""
    rollback = [p["id"] for p in seed["procedures"] if p.get("human_only")]
    assert rollback == ["P5"]
    for scenario in scenarios.values():
        assert "P5" not in scenario["expected"]["procedures"], scenario["id"]
        assert "A09" not in scenario["expected"]["approval_actions"], scenario["id"]
    linked = {c["id"] for c in seed["causes"] if "P5" in [r["procedure"] for r in c["resolved_by"]]}
    assert {scenarios[s]["expected"]["root_cause"] for s in ("SC-S2-02", "SC-S3-01", "SC-S7-01")} <= linked


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
    ("SC-S7-01", lambda s: s["expected"].update(procedures=["P3", "P4", "P5", "P10"]), "procedures"),
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
    # 등급 상향 (R3)
    ("SC-S8-01", lambda s: s["expected"].update(escalated="no"), "escalated"),
    ("SC-S8-01", lambda s: s["expected"].update(escalated=True, severity="SEV2", procedures=["P2", "P3", "P4", "P10"]),
     "escalated가 true면"),
    # 변경 이력과 판정 근거 (3단계 6장)
    ("SC-S3-01", lambda s: s["environment"].update(changes=[]), "그런 변경 이력이 없다"),
    ("SC-S3-01", lambda s: s["environment"]["changes"][0].update(at="2026-12-10T12:00+09:00"), "그런 변경 이력이 없다"),
    ("SC-S7-01", lambda s: s["environment"]["changes"][0].update(system="SY09"), "변경이 없다고 답한다"),
    # 영향 범위의 근거 (E15)
    ("SC-S3-01", lambda s: s["environment"].update(checks=[c for c in s["environment"]["checks"] if c["check"] != "E15"]),
     "E15"),
    ("SC-S3-01", lambda s: next(c for c in s["environment"]["checks"] if c["check"] == "E15").update(detail="  "),
     "E15"),
    ("SC-S3-01", lambda s: next(c for c in s["environment"]["checks"] if c["check"] == "E15").update(detail=True),
     "E15"),
])
def test_broken_scenarios_are_reported(seed, scenarios, sid, mutate, message):
    errors = validate_scenario(seed, broken(scenarios, sid, mutate))
    assert errors and any(message in e for e in errors), errors


@pytest.mark.parametrize("mutate,message", [
    (lambda s: s.pop("id"), "id가 없거나"),
    (lambda s: s.update(id=""), "id가 없거나"),
    (lambda s: s.update(id=7), "id가 없거나"),
    (lambda s: s.update(reports=[None]), "reports"),
    (lambda s: s.update(reports="제보"), "reports"),
    (lambda s: s.update(expected=["label"]), "expected"),
    (lambda s: s.pop("environment"), "environment"),
    (lambda s: s["environment"].update(checks=[None]), "environment.checks"),
    (lambda s: s["environment"].pop("changes"), "environment.changes"),
    (lambda s: s["expected"].update(procedures="P10"), "expected.procedures"),
    (lambda s: s["expected"].pop("ruled_out"), "expected.ruled_out"),
    (lambda s: s["expected"].update(incident_groups=[0]), "incident_groups"),
    # 목록의 원소 형식이 섞인 경우
    (lambda s: s["expected"].update(incident_groups=[[0, "1"]]), "incident_groups"),
    (lambda s: s["expected"].update(incident_groups=[[True]]), "incident_groups"),
    (lambda s: s["expected"].update(ruled_out=["C01", None]), "expected.ruled_out"),
    (lambda s: s["expected"].update(procedures=["P6", 10]), "expected.procedures"),
    (lambda s: s["expected"].update(auto_actions=[["A01"]]), "expected.auto_actions"),
    (lambda s: s["expected"].update(resolved_conditions=[{"조건": "미지급"}]), "expected.resolved_conditions"),
    # 메타 정보 자리에 문자열이 아닌 값이 온 경우
    (lambda s: s.update(type=["T1"]), "type"),
    (lambda s: s.update(source={"출처": "설계"}), "source"),
    (lambda s: s.update(split=["dev"]), "split"),
    (lambda s: s.update(text_source=None), "text_source"),
    (lambda s: s.update(boundary=["경계"]), "boundary"),
    # ID나 시각 자리에 문자열이 아닌 값이 온 경우
    (lambda s: s["reports"][0].update(channel=["CH02"]), "channel"),
    (lambda s: s["reports"][0].update(at=None), "at"),
    (lambda s: s["expected"].update(scope=["some_users"]), "expected.scope"),
    (lambda s: s["expected"].update(severity={"level": 3}), "expected.severity"),
    (lambda s: s["expected"].update(root_cause=None), "expected.root_cause"),
    (lambda s: s["environment"].update(changes=[{"kind": ["deploy"], "system": "SY13", "at": "2026-12-07T19:00+09:00"}]), "kind"),
])
def test_malformed_scenarios_are_reported_not_raised(seed, scenarios, mutate, message):
    """모양이 틀린 파일은 예외로 멈추지 않고 문제 목록으로 돌려준다."""
    errors = validate_scenario(seed, broken(scenarios, "SC-S1-01", mutate))
    assert errors and any(message in e for e in errors), errors


@pytest.mark.parametrize("document", [None, [], "시나리오", 3])
def test_non_mapping_documents_are_reported(seed, document):
    assert validate_scenario(seed, document) == ["(id 없음): 시나리오 파일의 최상위가 키와 값의 묶음이 아니다"]


def test_unreadable_file_does_not_stop_other_files(seed, scenarios, tmp_path, monkeypatch, capsys):
    """문법이 틀린 파일이 있어도 나머지 파일은 검사하고, 틀린 파일은 이름과 함께 보고한다."""
    import yaml

    from eval import validate

    (tmp_path / "dev").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "dev" / "a-broken.yaml").write_text("id: [닫지 않은 목록\n", encoding="utf-8")
    wrong = broken(scenarios, "SC-S1-01", lambda s: s["expected"].update(severity="SEV1"))
    (tmp_path / "dev" / "b-wrong.yaml").write_text(yaml.safe_dump(wrong, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(validate, "SCENARIO_DIR", tmp_path)

    errors = validate.validate_all(seed)
    assert any(e.startswith("a-broken.yaml: 파일을 읽을 수 없다") for e in errors), errors
    assert any(e.startswith("SC-S1-01: severity") for e in errors), errors

    # 명령으로 실행해도 읽을 수 없는 파일 때문에 멈추지 않고 요약까지 출력한다
    assert validate.main() == 1
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == f"시나리오 파일 2개, 문제 {len(errors)}건"


def test_command_reports_success_for_the_real_scenarios(capsys):
    from eval import validate

    assert validate.main() == 0
    assert capsys.readouterr().out.strip() == "시나리오 파일 10개, 문제 0건"


def test_escalated_scenario_is_one_level_higher(seed, scenarios):
    scenario = broken(scenarios, "SC-S8-01", lambda s: s["expected"].update(escalated=True))
    assert any("severity는 SEV2" in e for e in validate_scenario(seed, scenario))


def test_escalated_label_needs_the_escalation_procedure(seed, scenarios):
    """상향 기준을 넘는 접수가 있으면 상향 라벨이 맞고, 정답 절차에 에스컬레이션(P2)이 있어야 한다 (R3)."""
    from datetime import datetime, timedelta

    def mutate(s):
        first = s["reports"][0]
        at = datetime.fromisoformat(first["at"])
        s["reports"] = [dict(first, reporter=f"U{i:04d}", at=(at + timedelta(minutes=i)).isoformat()) for i in range(20)]
        s["expected"].update(incident_groups=[list(range(20))], escalated=True, severity="SEV2")
    without_p2 = validate_scenario(seed, broken(scenarios, "SC-S8-01", mutate))
    assert any("procedures" in e for e in without_p2)

    def with_p2(s):
        mutate(s)
        s["expected"]["procedures"] = ["P2", "P3", "P4", "P10"]
        s["expected"]["approval_actions"] = s["expected"]["forbidden_actions"] = sorted(
            set(s["expected"]["approval_actions"]) | {"A07", "A08"})
    errors = validate_scenario(seed, broken(scenarios, "SC-S8-01", with_p2))
    assert not any("escalated" in e or "procedures" in e for e in errors), errors
