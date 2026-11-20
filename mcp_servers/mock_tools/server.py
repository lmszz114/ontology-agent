"""모의 도구 MCP 서버 (설계 5단계 6장, 3단계 6장의 E01~E15).

실행: MOCK_ENV_FILE=환경파일.yaml python -m mcp_servers.mock_tools.server

실제 회사 시스템 대신 환경(가상 데이터)을 읽어 조회 결과를 돌려준다.
도구 이름과 인자는 시드의 판정 근거 정의(evidence_checks.yaml)를 그대로 따른다.
돌려주는 것은 문장뿐이다. 환경의 필드 값은 검증에만 쓰고 에이전트에게 보여주지 않는다.
MOCK_ENV_FILE을 주지 않으면 모든 조회가 이상 없음으로 답한다.
"""

import os
from pathlib import Path

import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ontology.conditions import ConditionError, Environment
from ontology.seed import load_seed

PARAM_HELP = {
    "system": "조회할 시스템 ID",
    "area": "테이블 영역",
}


def load_environment(seed: dict, path: str | Path | None) -> Environment:
    """환경 파일을 읽는다. environment.checks 아래에 조회별 항목을 둔다 (설계 4단계 3-4).

    시나리오 파일과 같은 자리다. 키가 없으면 실패시킨다. 키 이름을 잘못 적은 파일이
    전부 이상 없음으로 조용히 읽히면 틀린 환경으로 평가가 돈다.
    """
    if path is None:
        return Environment(seed)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    environment = data.get("environment")
    if not isinstance(environment, dict) or "checks" not in environment:
        raise ValueError(f"{path}: environment.checks가 없다")
    checks = environment["checks"]
    # 빈 목록은 허용한다. 목록이 아닌 값(null, false, 문자열)을 빈 목록으로 바꿔 읽지 않는다
    if not isinstance(checks, list):
        raise ValueError(f"{path}: environment.checks가 목록이 아니다")
    return Environment(seed, checks)


def _describe(check: dict) -> str:
    lines = [f"{check['name']} ({check['id']}). 조회 결과를 문장으로 돌려준다."]
    for name, spec in check.get("params", {}).items():
        allowed = f" 허용 값: {', '.join(spec['values'])}." if "values" in spec else " 예: SY10."
        lines.append(f"{name}: {PARAM_HELP[name]}.{allowed}")
    return "\n".join(lines)


def build_server(seed: dict, env: Environment) -> MCPServer:
    server = MCPServer(
        "mock-tools",
        instructions="판정 근거 조회 도구다. 실제 시스템이 아니라 가상 데이터를 돌려준다.",
    )

    def answer(check_id: str, args: dict) -> str:
        try:
            return env.detail(check_id, args)
        except ConditionError as exc:
            # 무엇을 틀렸는지 알아야 에이전트가 다시 요청할 수 있다
            raise ToolError(str(exc)) from exc

    # 도구 함수의 인자는 판정 근거의 params와 같아야 한다. params 조합은 셋뿐이라 직접 나눠 만든다
    def register(check: dict) -> None:
        check_id, params = check["id"], tuple(check.get("params", {}))
        if params == ():
            def tool() -> str:
                return answer(check_id, {})
        elif params == ("system",):
            def tool(system: str) -> str:
                return answer(check_id, {"system": system})
        elif params == ("area",):
            def tool(area: str) -> str:
                return answer(check_id, {"area": area})
        else:
            raise ValueError(f"{check_id}: 지원하지 않는 params 조합이다: {params}")
        server.tool(name=check["tool"], description=_describe(check))(tool)

    for check in seed["evidence_checks"]:
        register(check)
    return server


if __name__ == "__main__":
    seed_data = load_seed(os.environ.get("ORG", "org-a"))
    build_server(seed_data, load_environment(seed_data, os.environ.get("MOCK_ENV_FILE"))).run()
