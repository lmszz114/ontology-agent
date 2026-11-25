"""시나리오 사건마다 검색해 와야 하는 문서 (4단계 3-2 relevant_docs, 5-3 검색 지표).

시나리오 파일에 적지 않고 문서 설계표(plan.yaml)와 정답 라벨에서 규칙으로 계산한다 (2026-11-25).
시나리오 파일을 고치면 이미 낸 결과의 시나리오 해시가 바뀌어 다시 채점할 수 없고, test 세트는 잠겨 있기 때문이다.
규칙이 정해져 있으니 같은 입력이면 늘 같은 목록이 나온다.

규칙
- 사건의 정답 원인을 다루는 문서(설계표의 causes에 그 원인이 있는 문서)가 정답이다. 회고, 런북, 대화, 회의록 모두 포함한다
- 옛날 정보(stale)를 담은 문서는 넣지 않는다. 그 문서를 근거로 쓰면 틀린다. 옛날 문서 인용률(5-3)로 따로 본다
- 원인이 확정되지 않아야 정답인 사건과 범위 밖 시나리오는 정답 문서가 없다
"""

from eval.corpus.check import load_plan
from eval.labels import label_incidents


def relevant_docs(scenario: dict, plan: list[dict] | None = None) -> list[list[str]]:
    """사건마다 정답 문서 id 목록. 사건 순서는 label_incidents와 같다."""
    plan = load_plan() if plan is None else plan
    out = []
    for incident in label_incidents(scenario["expected"]):
        root = incident.get("root_cause")
        out.append(sorted(d["id"] for d in plan if root and root in (d.get("causes") or []) and not d.get("stale")))
    return out


def stale_docs(scenario: dict, plan: list[dict] | None = None) -> list[list[str]]:
    """사건마다 그 원인을 다루지만 옛날 정보를 담은 문서. 검색해 오면 옛날 문서 인용으로 센다."""
    plan = load_plan() if plan is None else plan
    out = []
    for incident in label_incidents(scenario["expected"]):
        root = incident.get("root_cause")
        out.append(sorted(d["id"] for d in plan if root and root in (d.get("causes") or []) and d.get("stale")))
    return out
