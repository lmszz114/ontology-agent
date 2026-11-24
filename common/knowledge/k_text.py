"""구성 A의 기준 지식 K 텍스트 (설계 4단계 2-1, 2-2, 5단계 4-2).

K는 1~3단계의 팀, 시스템, 원인, 근거, 절차 표 내용이다. 구성 C의 그래프와 같은 사실을 주려고
그래프 적재 데이터(ontology.load.build_graph)에서 텍스트를 만든다. 그래프에 넣지 않는 값
(판정 근거의 when 조건처럼 검증기만 쓰는 값)은 텍스트에도 넣지 않는다.

같은 시드에서는 바이트까지 같은 문자열이 나와야 한다. 프롬프트 캐싱은 앞부분이 한 글자라도 바뀌면 깨진다.
"""

import json

from ontology.load import STATIC_LABELS, Graph, build_graph

LABEL_TITLES = {
    "Team": "팀",
    "Role": "역할",
    "Channel": "접수 채널",
    "System": "시스템",
    "SymptomType": "증상",
    "Cause": "원인",
    "EvidenceCheck": "판정 근거",
    "Procedure": "절차",
    "Action": "조치",
    "Severity": "심각도",
}

# 관계를 시작 노드 아래에 적을 때 쓰는 이름. 화살표 방향은 그래프와 같다
RELATION_TITLES = {
    "HANDLED_FIRST_BY": "1차 처리 역할",
    "ASSIGNED_TO": "배정 팀",
    "OWNS": "담당 시스템",
    "COLLABORATES_ON": "조건부 협업 시스템",
    "DEPENDS_ON": "의존 시스템",
    "OBSERVED_IN": "나타나는 시스템",
    "MAY_BE_CAUSED_BY": "후보 원인",
    "LOCATED_IN": "위치",
    "CHECKED_BY": "판정 근거",
    "RESOLVED_BY": "해결 절차",
    "HAS_ACTION": "조치",
    "LED_BY": "주관 팀",
    "EXECUTED_BY": "실행 팀",
    "REQUIRES_APPROVAL_FROM": "승인 팀",
    "TRIGGERS": "여는 절차",
}


def _value(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _props(props: dict, skip: tuple[str, ...] = ()) -> str:
    items = [f"{k}={_value(v)}" for k, v in sorted(props.items()) if k not in skip]
    return ", ".join(items)


def render(graph: Graph) -> str:
    """노드마다 속성과 나가는 관계를 적는다. 정렬을 고정해 같은 그래프면 같은 문자열이 나온다."""
    outgoing: dict[str, list[tuple[str, str, dict]]] = {}
    for rel_type, rows in graph.rels.items():
        for row in rows:
            # source, source_doc_ids는 사람이 넣은 관계인지 표시하는 관리용 값이다. 지식이 아니라 뺀다
            props = {k: v for k, v in row["props"].items() if k not in ("source", "source_doc_ids")}
            outgoing.setdefault(row["start"], []).append((rel_type, row["end"], props))

    lines = ["# 기준 지식", ""]
    for label in STATIC_LABELS:
        lines += [f"## {LABEL_TITLES[label]} ({label})", ""]
        for node in sorted(graph.nodes[label], key=lambda n: n["id"]):
            lines.append(f"- {node['id']}: {_props(node, skip=('id',))}")
            rels = sorted(outgoing.get(node["id"], []), key=lambda r: (r[0], r[1], _props(r[2])))
            for rel_type, end, props in rels:
                extra = _props(props)
                lines.append(f"  - {RELATION_TITLES[rel_type]} -> {end}" + (f" ({extra})" if extra else ""))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def k_text(seed: dict) -> str:
    return render(build_graph(seed))
