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
