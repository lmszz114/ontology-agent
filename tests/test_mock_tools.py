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
        {"check": "E06", "args": {"system": "SY18"}, "observations": {"deployed": "yes", "result": "partial"}},
    ])
    _, results = call(
        build_server(seed, env),
        ("query_mailbox_log", {}),
        ("query_server_log", {"system": "SY10"}),
        ("query_server_log", {"system": "SY14"}),  # 환경에 없는 시스템은 이상 없음
        ("query_deploy_history", {"system": "SY18"}),
    )
    assert [r.content[0].text for r in results] == [
        "월드3에서 19:40 이후 지급 기록 0건",
        "월드 서버에 증상과 관련된 오류 로그가 없다. 월드 서버에 특정 콘텐츠 처리 지연 로그가 있다",
        "거래소에 증상과 관련된 오류 로그가 없다. 거래소에 처리 지연 로그가 없다",
        "증상 시작 전에 게임 클라이언트 배포 이력이 있다. 가장 최근 배포가 일부 대상에만 적용됐다",
    ]


def test_answers_do_not_expose_field_values(seed):
    """에이전트에게는 문장만 간다. 필드 이름과 값은 응답에 없다."""
    env = Environment(seed, [{"check": "E01", "observations": {"approval": "error"}}])
    _, (result,) = call(build_server(seed, env), ("query_payment_record", {}))
    assert result.structured_content in (None, {"result": result.content[0].text})
    assert "approval" not in result.content[0].text and "error" not in result.content[0].text


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
        "  - check: E09\n"
        "    args: {system: SY11}\n"
        "    observations: {status: resource_limit}\n",
        encoding="utf-8",
    )
    env = load_environment(seed, path)
    assert env.observe("E09", {"system": "SY11"}) == {"status": "resource_limit"}
    assert env.observe("E09", {"system": "SY05"}) == {"status": "normal"}
    assert load_environment(seed, None).entries == []
