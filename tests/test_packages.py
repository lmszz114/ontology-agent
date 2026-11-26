"""배포 패키지 로더와 가상 회사 B (설계 5단계 3장, 4단계 2-4, M6).

가상 회사 B는 채널과 역할 배정만 다르다. 같은 코드와 시드로 1차 처리 역할과 배정만 바뀌는지 본다.
접수 채널 -> 1차 처리 역할 -> 팀은 그래프 조회라 LLM 없이 확인할 수 있다.
"""

import copy

import pytest

from agents.diagnosis.run import SEARCH_CONFIGS, prompt_uses
from agents.extraction.prompts import PROMPT_VERSION as EXTRACTION_VERSION
from common.packages import PackageError, load_package
from eval.validate import validate_scenario
from ontology.seed import load_seed, seed_hash, validate_seed

ORGS = ("org-a", "org-b")
ORG_KEYS = ("org", "channels", "role_assignments")


@pytest.mark.parametrize("org", ORGS)
def test_every_package_matches_the_code_prompts(org):
    """프롬프트를 바꾸고 패키지를 고치지 않으면 여기서 걸린다. 모든 구성과 추출의 코드 버전이 패키지와 같아야 한다."""
    for config in ("A", "C", "C-ext", *SEARCH_CONFIGS):
        package = load_package(org, uses=prompt_uses(config))
        assert package.record() == {"id": org, "version": 1}
    load_package(org, uses={"extraction": EXTRACTION_VERSION})


def test_mismatched_or_unknown_prompt_is_refused():
    with pytest.raises(PackageError, match="diagnosis"):
        load_package("org-a", uses={"diagnosis": "diagnosis-v0"})
    with pytest.raises(PackageError, match="정의되지 않은 프롬프트"):
        load_package("org-a", uses={"summary": "x"})
    with pytest.raises(PackageError, match="배포 패키지가 없다"):
        load_package("org-z")


def test_org_b_differs_only_in_channels_and_role_assignments():
    a, b = load_seed("org-a"), load_seed("org-b")
    assert {k: v for k, v in a.items() if k not in ORG_KEYS} == {k: v for k, v in b.items() if k not in ORG_KEYS}
    assert seed_hash(a) != seed_hash(b)  # 조직 설정도 시드 해시에 들어간다. 실행기가 다른 조직의 그래프를 막는다
    assert "CH01" not in {c["id"] for c in b["channels"]}  # 공식 커뮤니티 없음
    intake = {r["role"]: r["team"] for r in b["role_assignments"]}["RL03"]
    assert intake == "TM07" and {r["role"]: r["team"] for r in a["role_assignments"]}["RL03"] == "TM05"


def test_channel_role_needs_a_team():
    seed = copy.deepcopy(load_seed("org-b"))
    seed["role_assignments"] = [r for r in seed["role_assignments"] if r["role"] != "RL03"]
    with pytest.raises(ValueError, match="맡은 팀이 없다"):
        validate_seed(seed)
    seed = copy.deepcopy(load_seed("org-b"))
    next(r for r in seed["role_assignments"] if r["role"] == "RL03")["team"] = None
    with pytest.raises(ValueError, match="비어 있다"):
        validate_seed(seed)
    seed = copy.deepcopy(load_seed("org-a"))
    seed["role_assignments"].append({"role": "RL03", "team": "TM07"})
    with pytest.raises(ValueError, match="두 팀"):
        validate_seed(seed)


def test_scenario_from_a_channel_org_b_does_not_have_is_rejected():
    """공식 커뮤니티 제보가 든 시나리오는 가상 회사 B 설정으로 검증하면 거부된다 (T7은 B에 있는 채널로 쓴다)."""
    from eval.validate import load_scenarios

    scenario = next(s for s in load_scenarios("dev") if any(r["channel"] == "CH01" for r in s["reports"]))
    assert not validate_scenario(load_seed("org-a"), scenario)
    assert any("정의되지 않은 채널" in e for e in validate_scenario(load_seed("org-b"), scenario))


@pytest.mark.integration
def test_first_handler_follows_the_loaded_package(driver):
    """같은 조회 코드가 적재한 조직 설정에 따라 다른 1차 처리 팀을 낸다. 끝나면 가상 회사 A로 되돌린다."""
    from mcp_servers.ontology_query import queries
    from ontology.load import load_static

    try:
        load_static(driver, load_seed("org-b"))
        assert queries.get_first_handler(driver, "CH03")["team_id"] == "TM07"
        assert queries.get_first_handler(driver, "CH02")["team_id"] == "TM07"
        with pytest.raises(Exception, match="CH01"):
            queries.get_first_handler(driver, "CH01")
        load_static(driver, load_seed("org-a"))
        assert queries.get_first_handler(driver, "CH03")["team_id"] == "TM05"
        assert queries.get_first_handler(driver, "CH01")["role_id"] == "RL01"
    finally:
        load_static(driver, load_seed("org-a"))


# ---------- T7 시나리오 (M6 PR 2) ----------

T7 = ("SC-T7-01", "SC-T7-02", "SC-T7-03")


def _t7(sid: str) -> dict:
    from eval.validate import load_scenarios

    return copy.deepcopy(next(s for s in load_scenarios("test") if s["id"] == sid))


@pytest.mark.parametrize("sid", T7)
def test_t7_is_valid_only_under_its_org(sid):
    """T7은 가상 회사 B 설정으로 검증하면 통과하고 A 설정으로 보면 거부된다. 원래 시나리오와 라벨은 1차 처리 팀만 다르다."""
    from eval.labels import label_incidents
    from eval.validate import load_scenarios

    t7 = _t7(sid)
    assert validate_scenario(load_seed("org-b"), t7) == []
    assert any("org-b 설정으로 검증한다" in e for e in validate_scenario(load_seed("org-a"), t7))
    original = next(s for s in load_scenarios("test") if s["id"] == t7["derived_from"])
    strip = lambda labels: [{k: v for k, v in i.items() if k not in ("first_handler", "reports")} for i in labels]  # noqa: E731
    assert strip(label_incidents(t7["expected"])) == strip(label_incidents(original["expected"]))
    assert all(i["first_handler"] == "TM07" for i in label_incidents(t7["expected"]))


def test_first_handler_label_follows_the_earliest_report():
    """1차 처리 팀은 사건의 가장 이른 제보 채널로 계산한다. 같은 제보를 A 설정으로 보면 개발PM팀이다."""
    from eval.validate import first_handler_team

    t7 = _t7("SC-T7-02")
    assert first_handler_team(load_seed("org-b"), t7, [0, 1, 2, 3]) == "TM07"
    assert first_handler_team(load_seed("org-a"), t7, [0, 1, 2, 3]) == "TM05"  # 사내 메신저(CH03) -> RL03 -> TM05
    wrong = _t7("SC-T7-02")
    wrong["expected"]["first_handler"] = "TM05"
    assert any("first_handler는" in e and "TM07" in e for e in validate_scenario(load_seed("org-b"), wrong))


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.pop("org"), "org-a가 아닌 조직"),
    (lambda s: s["expected"].pop("first_handler"), "사건마다 1차 처리 팀"),
    (lambda s: s.pop("derived_from"), "derived_from"),
    (lambda s: s.update(org=["org-b"]), "org은 문자열"),
])
def test_t7_needs_org_handler_and_origin(mutate, message):
    t7 = _t7("SC-T7-01")
    mutate(t7)
    seed = load_seed("org-a" if "org" not in t7 else "org-b")
    assert any(message in e for e in validate_scenario(seed, t7))


def test_runner_refuses_scenario_from_another_org():
    from pathlib import Path

    from eval.runner.__main__ import check_orgs
    from eval.runner.runner import RunnerError

    t7 = Path("eval/scenarios/test/SC-T7-01.yaml")
    dev = Path("eval/scenarios/dev/SC-S1-01.yaml")
    check_orgs([t7], "org-b")
    check_orgs([dev], "org-a")
    with pytest.raises(RunnerError, match="SC-T7-01.yaml\\(org-b\\)"):
        check_orgs([dev, t7], "org-a")
    with pytest.raises(RunnerError, match="SC-S1-01.yaml\\(org-a\\)"):
        check_orgs([dev], "org-b")


def _t7_result(team: str | None) -> dict:
    """SC-T7-01 정답 그대로의 가짜 결과. 1차 처리 팀만 바꿔 본다."""
    e = _t7("SC-T7-01")["expected"]
    output = {
        "symptom": e["symptom"], "report_indexes": [0], "scope": e["scope"], "severity": e["severity"],
        "first_handler": {"channel_id": "CH03", "role_id": "RL03", "team_id": team},
        "hypotheses": [{"cause_id": e["root_cause"], "rank": 1, "status": "confirmed", "result": "confirm"}],
        "root_cause": e["root_cause"], "primary_team": e["primary_team"], "procedures": e["procedures"],
        "actions": [], "tool_calls": [],
    }
    run = {"scenario": "SC-T7-01", "split": "test", "org": "org-b", "llm_calls": 1, "usage": {"input": 1, "output": 1},
           "cost_usd": 0.0, "seconds": 1.0}
    return {"run": run, "outputs": [output], "unclassified": []}


def test_grader_scores_org_setting_only_on_t7():
    """조직 설정 반영률은 1차 처리 팀 라벨이 있는 사건(T7)만 센다. 다른 시나리오는 해당 없음이다."""
    from eval.grader.grade import aggregate, grade_scenario, load_scenario

    seed = load_seed("org-b")
    scenario = load_scenario("test", "SC-T7-01")
    right = grade_scenario(_t7_result("TM07"), scenario, seed)
    wrong = grade_scenario(_t7_result("TM05"), scenario, seed)
    assert right["first_handler"] is True and wrong["first_handler"] is False
    assert wrong["first_failure"] == "배정: 1차 처리"
    assert aggregate([right, wrong])["first_handler"] == 0.5
    missing = _t7_result("TM07")
    missing["outputs"] = []
    assert grade_scenario(missing, scenario, seed)["first_handler"] is False

    from tests.test_grader import grade, perfect_s7_02

    other = grade(perfect_s7_02())
    assert other["first_handler"] is None and other["first_failure"] is None
    assert aggregate([other])["first_handler"] is None


def test_grader_refuses_result_run_under_another_org(tmp_path):
    import json

    from eval.grader.grade import GradeError, grade_folder
    from tests.test_grader import _run

    result = _t7_result("TM05")
    result["run"] = _run("SC-T7-01", split="test", org="org-a")
    (tmp_path / "SC-T7-01.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(GradeError, match="org-b 설정인데 org-a로 돌았다"):
        grade_folder(tmp_path)
