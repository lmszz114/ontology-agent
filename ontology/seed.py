"""시드 YAML과 조직 설정을 읽는다.

정적 지식 계층의 원본은 이 YAML 파일이고, Neo4j는 이를 적재한 사본이다
(설계 3단계 8장, 5단계 7장).
"""

from pathlib import Path

import yaml

from ontology.conditions import validate_cause_conditions

ROOT = Path(__file__).parent.parent
SEED_DIR = Path(__file__).parent / "seed"
PACKAGES_DIR = ROOT / "packages"

# 파일 이름 -> 최상위 키
SEED_FILES = {
    "teams.yaml": "teams",
    "roles.yaml": "roles",
    "systems.yaml": "systems",
    "symptoms.yaml": "symptoms",
    "causes.yaml": "causes",
    "evidence_checks.yaml": "evidence_checks",
    "procedures.yaml": "procedures",
    "actions.yaml": "actions",
    "severities.yaml": "severities",
}


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_seed(org: str = "org-a") -> dict:
    """시드와 조직 설정을 하나의 dict로 돌려준다."""
    seed: dict = {}
    for filename, key in SEED_FILES.items():
        data = _read(SEED_DIR / filename)
        seed[key] = data[key]
        if key == "severities":
            seed["escalation_procedure"] = data["escalation_procedure"]

    org_data = _read(PACKAGES_DIR / org / "org.yaml")
    seed["org"] = org_data["org"]
    seed["channels"] = org_data["channels"]
    seed["role_assignments"] = org_data["role_assignments"]
    return seed


def validate_seed(seed: dict) -> None:
    """적재 전에 시드를 확인한다.

    관계로 들어가는 참조는 적재기가 관계 수로 확인한다. 여기서는 노드 속성으로 들어가서
    틀려도 그대로 적재되는 값을 본다.
    """
    teams = {t["id"] for t in seed["teams"]}
    systems = {s["id"] for s in seed["systems"]}

    def require(value, allowed: set, where: str) -> None:
        if value is not None and value not in allowed:
            raise ValueError(f"{where}: 정의되지 않은 값이다: {value}")

    for channel in seed["channels"]:
        require(channel.get("source_system"), systems, f"{channel['id']}.source_system")
    for check in seed["evidence_checks"]:
        for key in ("judged_by_team", "performed_by_team"):
            require(check.get(key), teams, f"{check['id']}.{key}")
        for system_id in check["source_systems"]:
            require(system_id, systems, f"{check['id']}.source_systems")
    for action in seed["actions"]:
        require(action.get("uses_system"), systems, f"{action['id']}.uses_system")
    # prior는 필수라서 비어 있어도 거부한다
    for symptom in seed["symptoms"]:
        for ref in symptom["causes"]:
            if ref.get("prior") not in {"높음", "중간", "낮음"}:
                raise ValueError(f"{symptom['id']}의 {ref['cause']} prior: 정의되지 않은 값이다: {ref.get('prior')}")
    validate_cause_conditions(seed)


def resolve_owner_teams(seed: dict, cause_id: str) -> list[str]:
    """원인의 1차 담당 팀 후보를 시드에서 구한다 (R5). Neo4j 없이 시드를 검증할 때 쓴다.

    운영 경로는 그래프를 읽는 mcp_servers/ontology_query의 get_owner_team이다.

    원인 위치(LOCATED_IN)의 scope와 area가 담당(OWNS)의 scope, area와 같은 팀을 고른다.
    area가 없는 쪽은 없는 쪽끼리만 맞는다.
    """
    cause = next(c for c in seed["causes"] if c["id"] == cause_id)
    loc = cause["located_in"]
    system = next(s for s in seed["systems"] if s["id"] == loc["system"])
    return [
        owner["team"]
        for owner in system["owners"]
        if loc["scope"] in owner["scopes"] and owner.get("area") == loc.get("area")
    ]
