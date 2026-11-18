"""시드 YAML과 조직 설정을 읽는다.

정적 지식 계층의 원본은 이 YAML 파일이고, Neo4j는 이를 적재한 사본이다
(설계 3단계 8장, 5단계 7장).
"""

from pathlib import Path

import yaml

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


def resolve_owner_teams(seed: dict, cause_id: str) -> list[str]:
    """원인의 1차 담당 팀 후보를 돌려준다 (R5).

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
