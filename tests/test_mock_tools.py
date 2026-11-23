"""모의 도구 MCP 서버 검증. Neo4j 없이 돈다."""

import asyncio

import pytest
from mcp import Client

from mcp_servers.mock_tools.server import build_server, load_environment
from ontology.conditions import Environment
from ontology.seed import load_seed


@pytest.fixture(scope="module")
def seed():
    return load_seed("org-a")


def call(server, *calls):
    async def run():
        async with Client(server) as client:
            listed = await client.list_tools()
            return listed, [await client.call_tool(name, args) for name, args in calls]

    return asyncio.run(run())


def test_tools_match_evidence_checks(seed):
    listed, _ = call(build_server(seed, Environment(seed)))
    assert [t.name for t in listed.tools] == [e["tool"] for e in seed["evidence_checks"]]
    params = {t.name: sorted(t.input_schema.get("properties", {})) for t in listed.tools}
    assert params == {e["tool"]: sorted(e.get("params", {})) for e in seed["evidence_checks"]}
    assert all(t.description for t in listed.tools)


def test_default_answers_are_normal(seed):
    _, results = call(
        build_server(seed, Environment(seed)),
        ("query_payment_record", {}),
        ("query_server_log", {"system": "SY10"}),
        ("query_reports", {}),
    )
    assert [r.content[0].text for r in results] == [
        "결제 승인 기록이 있다",
        "월드 서버에 증상과 관련된 오류 로그가 없다. 월드 서버에 처리 지연 로그가 없다",
        "같은 증상의 다른 접수 내역이 없다",
    ]


def test_answers_follow_environment(seed):
    env = Environment(seed, [
        {"check": "E02", "observations": {"delivery": "absent"}, "detail": "월드3에서 19:40 이후 지급 기록 0건"},
        {"check": "E04", "args": {"system": "SY10"}, "observations": {"delay_log": "present"}},
        {"check": "E06", "args": {"system": "SY18"}, "observations": {"deploy": "partial"}},
        {"check": "E05", "args": {"area": "probability"}, "observations": {"changed": "yes"}},
    ])
    _, results = call(
        build_server(seed, env),
        ("query_mailbox_log", {}),
        ("query_server_log", {"system": "SY10"}),
        ("query_server_log", {"system": "SY14"}),  # 환경에 없는 시스템은 이상 없음
        ("query_deploy_history", {"system": "SY18"}),
        ("query_deploy_history", {"system": "SY10"}),  # 배포 이력이 없으면 배포 결과를 말하지 않는다
        ("query_table_change", {"area": "probability"}),
    )
    assert [r.content[0].text for r in results] == [
        "월드3에서 19:40 이후 지급 기록 0건",
        "월드 서버에 증상과 관련된 오류 로그가 없다. 월드 서버에 특정 콘텐츠 처리 지연 로그가 있다",
        "거래소에 증상과 관련된 오류 로그가 없다. 거래소에 처리 지연 로그가 없다",
        "증상 시작 전에 게임 클라이언트 배포 이력이 있다. 가장 최근 배포가 일부 대상에만 적용됐다",
        "증상 시작 전에 월드 서버 배포 이력이 없다",
        "증상 시작 전에 게임 데이터 테이블 확률 영역의 변경 이력이 있다",
    ]


def test_answers_do_not_expose_field_values(seed):
    """에이전트에게는 문장만 간다. 필드 이름과 값은 응답에 없다."""
    env = Environment(seed, [{"check": "E01", "observations": {"approval": "error"}}])
    _, (result,) = call(build_server(seed, env), ("query_payment_record", {}))
    assert result.content[0].text == "결제 승인 기록에 오류가 남아 있다"
    assert result.structured_content in (None, {"result": result.content[0].text})


def test_invalid_arguments_are_explained(seed):
    _, results = call(
        build_server(seed, Environment(seed)),
        ("query_server_log", {"system": "SY99"}),
        ("query_table_change", {"area": "event"}),
    )
    assert all(r.is_error for r in results)
    assert "SY99" in results[0].content[0].text and "event" in results[1].content[0].text


def test_load_environment_from_file(seed, tmp_path):
    path = tmp_path / "env.yaml"
    path.write_text(
        "environment:\n"
        "  checks:\n"
        "    - check: E09\n"
        "      args: {system: SY11}\n"
        "      observations: {status: resource_limit}\n"
        "  changes: []\n",
        encoding="utf-8",
    )
    env = load_environment(seed, path)
    assert env.observe("E09", {"system": "SY11"}) == {"status": "resource_limit"}
    assert env.observe("E09", {"system": "SY05"}) == {"status": "normal"}
    assert load_environment(seed, None).entries == []
    empty = tmp_path / "empty.yaml"
    empty.write_text("environment:\n  checks: []\n", encoding="utf-8")
    assert load_environment(seed, empty).entries == []


@pytest.mark.parametrize("text", [
    "enviroment:\n  checks: []\n", "environment:\n", "environment:\n  - check: E01\n", "",
    "environment:\n  checks:\n", "environment:\n  checks: false\n", "environment:\n  checks: {}\n",
])
def test_load_environment_rejects_files_without_checks(seed, tmp_path, text):
    """키 이름을 틀린 파일이 전부 이상 없음으로 읽히면 안 된다."""
    path = tmp_path / "env.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="environment.checks"):
        load_environment(seed, path)


def test_parameter_examples_are_valid(seed):
    """도구 설명에 거부되는 값을 예시로 적지 않는다."""
    listed, _ = call(build_server(seed, Environment(seed)))
    by_name = {t.name: t.description for t in listed.tools}
    assert "SY10" not in by_name["query_config_change"] and "SY10" not in by_name["query_resource_metrics"]
    assert "SY10" in by_name["query_server_log"]
