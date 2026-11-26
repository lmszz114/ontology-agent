"""판단 규칙이 여는 절차 (설계 3단계 R1, R3, R7).

절차 ID를 코드에 두지 않고 시드(ontology/seed/rules.yaml)에서 읽는다. 진단 에이전트, 시나리오 검증기,
채점기가 같은 값을 쓴다 (M6, 5단계 12장). 긴급 공지(P3)는 규칙이 아니라 등급의 TRIGGERS 관계라 시드의
severities에서 읽는다.
"""

from dataclasses import dataclass

RULE_KEYS = ("intake", "escalation", "urgent_maintenance", "fix")


@dataclass(frozen=True)
class Rules:
    intake: str               # 모든 사건 (R1)
    escalation: str           # 등급 상향 (R3)
    urgent_maintenance: str   # 긴급 등급 (R7)
    fix: str                  # 원인 확정 (R7)
    urgent_levels: frozenset[str]
    triggers: dict[str, tuple[str, ...]]  # 등급 -> 그 등급이 여는 절차 (TRIGGERS)

    def severity_procedures(self, severity: str) -> set[str]:
        """이 등급이면 원인과 상관없이 여는 절차: 등급의 TRIGGERS와 긴급 등급의 긴급 점검."""
        procedures = set(self.triggers.get(severity, ()))
        if severity in self.urgent_levels:
            procedures.add(self.urgent_maintenance)
        return procedures

    @property
    def by_severity(self) -> set[str]:
        """등급 규칙(R3 TRIGGERS, R7)이 여는 절차. 원인에 연결돼 있어도 원인 쪽으로는 열지 않는다 (4단계 3-4)."""
        return {p for procs in self.triggers.values() for p in procs} | {self.urgent_maintenance}


def rules_of(seed: dict) -> Rules:
    data = seed["rules"]
    procs = data["rule_procedures"]
    return Rules(**{k: procs[k] for k in RULE_KEYS}, urgent_levels=frozenset(data["urgent_levels"]),
                 triggers={s["id"]: tuple(s.get("triggers") or ()) for s in seed["severities"]})


def validate_rules(seed: dict) -> None:
    data = seed["rules"]
    procs = data.get("rule_procedures") or {}
    if set(procs) != set(RULE_KEYS):
        raise ValueError(f"rules.yaml의 rule_procedures는 {list(RULE_KEYS)}여야 한다: {sorted(procs)}")
    if len(set(procs.values())) != len(procs):
        # 규칙마다 다른 절차여야 한다. 같으면 한 절차가 두 규칙으로 열려 어느 규칙이 열었는지 가를 수 없다 (PR #37 리뷰)
        raise ValueError(f"rules.yaml: 두 규칙이 같은 절차를 가리킨다: {procs}")
    known = {p["id"] for p in seed["procedures"]}
    for key, procedure in procs.items():
        if procedure not in known:
            raise ValueError(f"rules.yaml {key}: 정의되지 않은 절차다: {procedure}")
    human_only = {p["id"] for p in seed["procedures"] if p.get("human_only")}
    if human_only & set(procs.values()):
        raise ValueError(f"rules.yaml: 사람만 여는 절차를 규칙이 열 수 없다: {sorted(human_only & set(procs.values()))}")
    # 등급이 여는 절차(TRIGGERS)는 규칙 절차와 겹치면 안 된다. 겹치면 그 절차를 어느 규칙이 여는지 가를 수 없고,
    # 원인 수정(P10)이 등급에 걸리면 원인 미확정 사건에서 조회되지 않는다 (PR #37 리뷰)
    for sev in seed["severities"]:
        overlap = set(sev.get("triggers") or []) & set(procs.values())
        if overlap:
            raise ValueError(f"{sev['id']}.triggers에 규칙 절차가 있다: {sorted(overlap)}")
    levels = {s["id"] for s in seed["severities"]}
    if not set(data.get("urgent_levels") or []) <= levels:
        raise ValueError(f"rules.yaml urgent_levels: 정의되지 않은 등급이 있다: {data.get('urgent_levels')}")
