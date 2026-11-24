"""심각도 판정 (설계 3단계 4-3, R3).

정해진 표를 찾는 계산이라 LLM에게 맡기지 않는다. 진단 에이전트의 심각도 계산 단계와
시나리오 검증이 같은 함수를 쓴다. 표의 값은 가정이다.
"""

from datetime import timedelta

# 영향 범위 (설계 3단계 4-3 표의 행)
SCOPES = {
    "all_service": "전체 서비스",
    "one_world": "특정 월드 전체",
    "some_users": "일부 유저",
}

# 등급 상향 기준 (R3 상향 규칙, 가정값). 같은 사건의 접수가 이 시간 안에 이 수 이상이면 한 단계 올린다.
# 시간창은 새 제보의 접수 시각을 끝으로 한다. 진단 에이전트와 시나리오 검증이 같은 값을 쓴다
ESCALATION_WINDOW = timedelta(minutes=30)
ESCALATION_COUNT = 20

# 낮은 등급부터 높은 등급 순
LEVELS = ("SEV4", "SEV3", "SEV2", "SEV1")

# (영향 범위, 핵심 기능 여부) -> 등급
TABLE = {
    ("all_service", True): "SEV1",
    ("all_service", False): "SEV2",
    ("one_world", True): "SEV2",
    ("one_world", False): "SEV3",
    ("some_users", True): "SEV3",
    ("some_users", False): "SEV4",
}


def base_severity(scope: str, core_function: bool, min_severity: str | None = None) -> str:
    """영향 범위와 핵심 기능 여부로 등급을 정한다.

    min_severity는 증상 유형이 정한 최소 등급이다 (재화 비정상 증식은 규모 확인 전까지 최소 SEV2).
    """
    if scope not in SCOPES:
        raise ValueError(f"정의되지 않은 영향 범위다: {scope}")
    severity = TABLE[(scope, bool(core_function))]
    if min_severity is not None:
        if min_severity not in LEVELS:
            raise ValueError(f"정의되지 않은 심각도다: {min_severity}")
        severity = max(severity, min_severity, key=LEVELS.index)
    return severity


def escalate(severity: str) -> str:
    """한 단계 상향한다. SEV1은 그대로다."""
    if severity not in LEVELS:
        raise ValueError(f"정의되지 않은 심각도다: {severity}")
    return LEVELS[min(LEVELS.index(severity) + 1, len(LEVELS) - 1)]
