"""정답 라벨을 사건 단위로 읽는다 (설계 4단계 3-2).

라벨은 두 가지 형식으로 쓴다. 검증기와 채점기는 이 함수로 같은 모양(사건 목록)을 받는다.

- 사건 하나: expected에 증상, 원인 같은 라벨을 바로 적고 incident_groups에 그 사건의 제보 묶음 하나를 적는다.
  M2, M3에서 만든 dev 시나리오가 이 형식이다. 파일을 바꾸면 이미 낸 결과의 시나리오 해시가 달라져
  다시 채점할 수 없으므로 그대로 둔다
- 사건 여럿 또는 없음: expected.incidents에 사건마다 제보 번호(reports)와 라벨을 적는다 (test 세트, 2026-11-25).
  다른 원인의 사건을 섞는 시나리오(T3)와 사건이 없는 범위 밖 시나리오(T6, incidents: [])에 쓴다

두 형식 모두 분류하지 않을 제보는 expected.unclassified에 적는다.
원인이 확정되지 않아야 정답인 사건은 root_cause와 primary_team을 비워(null) 둔다 (T5, 2026-11-25)
조직 설정 변경 시나리오(T7)는 사건마다 1차 처리 팀(first_handler)을 더 적는다 (M6, 2026-11-26)
"""

# 사건 하나의 라벨 이름. 사건 여럿 형식의 각 항목도 같은 이름을 쓴다
INCIDENT_FIELDS = (
    "symptom", "scope", "severity", "escalated", "root_cause", "ruled_out", "required_checks", "primary_team",
    "resolved_conditions", "procedures", "auto_actions", "approval_actions", "forbidden_actions", "first_handler",
)


# 시나리오에 조직(org)을 적지 않으면 가상 회사 A 설정으로 돈다. T7만 다른 조직을 적는다 (M6)
DEFAULT_ORG = "org-a"


def scenario_org(scenario: dict) -> str:
    """시나리오를 돌릴 조직 설정 (배포 패키지 id)."""
    return scenario.get("org") or DEFAULT_ORG


def is_multi(expected: dict) -> bool:
    """사건 여럿 형식(expected.incidents)인지."""
    return "incidents" in expected


def label_incidents(expected: dict) -> list[dict]:
    """사건마다 {reports, 라벨...}. 사건 하나 형식은 incident_groups의 첫 묶음을 그 사건의 제보로 본다.

    모양은 검증기(eval/validate.py)가 먼저 확인한다. 여기서는 확인하지 않는다
    """
    if is_multi(expected):
        return [dict(incident) for incident in expected["incidents"]]
    incident = {name: expected[name] for name in INCIDENT_FIELDS if name in expected}
    incident["reports"] = list(expected["incident_groups"][0])
    return [incident]


def label_groups(expected: dict) -> list[list[int]]:
    """정답 사건 묶음 (제보 번호 목록의 목록)."""
    if is_multi(expected):
        return [list(incident["reports"]) for incident in expected["incidents"]]
    return [list(group) for group in expected["incident_groups"]]
