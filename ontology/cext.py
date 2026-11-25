"""구성 C-ext의 그래프: 문서에서 추출한 후보를 검수 없이 모두 넣은 온톨로지 (설계 4단계 2-2, 5단계 4-2, M4 PR 6).

실행: python -m ontology.cext eval/results/<추출 결과 폴더> r1     # C-ext 그래프 DB(NEO4J_EXT_URI, 7689)에 적재

C와 같은 진단 에이전트, 같은 지식 공급자(그래프)로 진단하고 그래프만 다르다. 운영 흐름은 검수와 승인(A17)을 거친 후보만
반영한다. 이 그래프는 추출 품질이 진단에 주는 영향을 재는 실험용이라 그 원칙의 예외다 (3단계 R8).

K에서 가져오는 것
- 노드 전체와 노드 속성(이름, 별칭, 핵심 기능, 최소 심각도, 판정 근거 정의, 절차, 조치). 이름 정규화가 K 노드에 잇는
  방식이라 노드는 K가 기준이다. 추출된 별칭은 노드의 별칭에 더한다
- 추출 대상이 아닌 관계: 역할 배정, 채널, 조치의 위험도와 실행 팀과 승인권자, 절차의 조치와 주관 팀, 심각도가 여는
  절차, 조건부 협업 (3단계 R8-3)
- 미지급, 과지급 같은 조건의 정의(어느 판정 근거로 판단하는지와 그 설명). 원인마다의 사실이 아니라 전체 규칙이다

추출에서 가져오는 것: OWNS, DEPENDS_ON, OBSERVED_IN, MAY_BE_CAUSED_BY, LOCATED_IN, CHECKED_BY, RESOLVED_BY, 별칭.
K의 같은 관계는 하나도 넣지 않는다.

추출에 없는 값은 정해진 규칙으로 채운다 (M4 PR 6 결정, 사용자 승인). K를 보고 맞추지 않는다
- MAY_BE_CAUSED_BY의 prior가 없으면 중간
- OWNS, LOCATED_IN의 scope가 없으면 scope 없이 넣는다. 담당 조회에서 맞는 팀이 없게 되고 그것이 결과다
- CHECKED_BY의 required가 없으면 확정 조건(참)
- 판정 근거의 조회 인자(args)는 문서에서 뽑지 않는다. 원인 위치에서 채운다: 시스템을 받는 근거는 위치의 시스템,
  테이블 영역을 받는 근거는 위치의 영역. K의 인자와 다른 원인(C10, C15, C21)이 있고 그대로 결과에 남긴다
- 새 엔티티 후보(노드 id가 없는 후보)는 노드를 만들 수 없어 넣지 않고 수를 남긴다

스키마 규칙은 지킨다. 한 원인의 위치는 하나, 같은 증상과 원인 사이의 원인 후보 관계는 하나, 같은 원인과 판정 근거
사이의 관계는 하나다 (3단계 3-1). 조회 쿼리가 이 규칙에 기댄다. 후보가 같은 자리에 다른 값을 내면 가장 많이 나온 값을
쓰고, 같으면 가장 먼저 나온 후보(문서 id, 후보 순서)의 값을 쓴다. 바뀐 자리 수를 남긴다.
충돌 후보(옛날 정보)는 그대로 넣는다. 같은 담당 자리를 두 팀이 맡게 되면 진단이 사람 확인으로 남긴다.
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ontology.db import get_driver
from ontology.load import SEED_META, Graph, build_graph, write_graph
from ontology.seed import load_seed, seed_hash

BUILDER_VERSION = "cext-v1"
EXT_URI_DEFAULT = "bolt://localhost:7689"
EXTRACTED = ("OWNS", "DEPENDS_ON", "OBSERVED_IN", "MAY_BE_CAUSED_BY", "LOCATED_IN", "CHECKED_BY", "RESOLVED_BY")
DEFAULT_PRIOR = "중간"
# 같은 양끝에 하나만 있어야 하는 관계와 그 자리를 가르는 키. LOCATED_IN은 원인마다 하나다
SINGLE = {
    "LOCATED_IN": lambda s, e, p: (s,),
    "MAY_BE_CAUSED_BY": lambda s, e, p: (s, e),
    "CHECKED_BY": lambda s, e, p: (s, e),
    "RESOLVED_BY": lambda s, e, p: (s, e),
}


class CextError(RuntimeError):
    pass


def ext_uri() -> str:
    return os.environ.get("NEO4J_EXT_URI", EXT_URI_DEFAULT)


def load_candidates(folder: Path, rep: str) -> tuple[list[dict], dict]:
    """추출 결과 폴더의 한 반복에서 후보를 읽는다. 문서를 모두 돈 반복이어야 한다. (후보, 출처 정보)를 돌려준다."""
    folder = Path(folder)
    run = yaml.safe_load((folder / "run.yaml").read_text(encoding="utf-8"))
    if run.get("split") != "dev":
        raise CextError(f"{folder}: dev 문서 추출 결과만 쓴다 (split={run.get('split')})")
    reps = sorted(p for p in folder.glob("r*") if p.is_dir())
    found = {p.name: {f.stem for f in p.glob("*.yaml")} for p in reps}
    if rep not in found:
        raise CextError(f"{folder}에 반복 {rep}가 없다: {sorted(found)}")
    expected = set(run["documents"]) if run.get("documents") else set().union(*found.values())
    if found[rep] != expected:
        raise CextError(f"{folder}/{rep}: 문서를 모두 돌지 않은 반복이다. 빠진 문서 {sorted(expected - found[rep])}")
    candidates, digest = [], hashlib.sha256()
    for path in sorted((folder / rep).glob("*.yaml")):
        raw = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + raw)
        data = yaml.safe_load(raw.decode("utf-8"))
        for index, c in enumerate(data.get("candidates") or []):
            candidates.append(c | {"document_id": data["document_id"], "_order": (data["document_id"], index)})
    source = {"folder": folder.as_posix(), "rep": rep, "run_id": f"{folder.name}-{rep}",
              "extraction_seed_hash": run.get("seed_hash"), "files_hash": digest.hexdigest()}
    return candidates, source


def _conditions(seed: dict) -> dict[str, dict]:
    """조건 이름 -> 조건의 판정 근거 정의 (K의 전체 규칙). 같은 이름은 모든 원인에서 같다 (3단계 3-1)."""
    out = {}
    for cause in seed["causes"]:
        for ref in cause["resolved_by"]:
            if ref.get("condition"):
                out[ref["condition"]] = {"condition_check": ref.get("check"), "condition_expect": ref.get("expect"),
                                         "condition_args": json.dumps(ref["args"], ensure_ascii=False, sort_keys=True)
                                         if ref.get("args") else None}
    return out


def _vote(rows: list[tuple[tuple, dict, dict]]) -> tuple[dict, dict, bool]:
    """같은 자리에 들어온 (순서, 끝, 속성) 중 가장 많은 값. 같으면 먼저 나온 값. (끝, 속성, 바뀐 자리인지)."""
    def key(row):
        return json.dumps({"end": row[1], "props": row[2]}, ensure_ascii=False, sort_keys=True)

    counts = Counter(key(r) for r in rows)
    first = {}
    for r in sorted(rows, key=lambda r: r[0]):
        first.setdefault(key(r), r)
    best = min(counts, key=lambda k: (-counts[k], first[k][0]))
    return first[best][1], first[best][2], len(counts) > 1


def build_cext_graph(seed: dict, candidates: list[dict]) -> tuple[Graph, dict]:
    """K 시드와 추출 후보로 C-ext 그래프를 만든다. Neo4j에 접속하지 않는다. (그래프, 만든 기록)을 돌려준다."""
    g = build_graph(seed)
    for rel in EXTRACTED:
        g.rels[rel] = []
    params = {check["id"]: list(check.get("params", {})) for check in seed["evidence_checks"]}
    conditions = _conditions(seed)
    nodes = {label: {n["id"]: n for n in rows} for label, rows in g.nodes.items()}
    stats: dict = {"candidates": len(candidates), "dropped_new_entity": 0, "default_prior": 0, "no_scope": 0,
                   "default_required": 0, "unknown_condition": 0, "voted_slots": 0, "aliases_added": 0}

    # (관계, 시작, 끝, 속성 JSON) -> {순서, 문서 목록}
    facts: dict[tuple, dict] = {}
    expects: dict[tuple, str] = {}
    for c in sorted(candidates, key=lambda c: c["_order"]):
        rel, s, o = c["relation"], c["subject"], c["object"]
        if s.get("id") is None or (rel != "ALIAS" and o.get("id") is None):
            stats["dropped_new_entity"] += 1
            continue
        p = c.get("props") or {}
        if rel == "ALIAS":
            node = nodes[s["type"]][s["id"]]
            if o["name"] != node.get("name") and o["name"] not in (node.get("aliases") or []):
                node["aliases"] = [*(node.get("aliases") or []), o["name"]]
                stats["aliases_added"] += 1
            continue
        props: dict = {}
        if rel in ("OWNS", "LOCATED_IN"):
            props = {k: p[k] for k in ("scope", "area") if p.get(k)}
            stats["no_scope"] += "scope" not in props
        elif rel == "MAY_BE_CAUSED_BY":
            props = {"prior": p.get("prior") or DEFAULT_PRIOR}
            stats["default_prior"] += not p.get("prior")
        elif rel == "CHECKED_BY":
            required = p.get("required")
            stats["default_required"] += required is None
            # 확정 조건 설명(expect)은 문서마다 말이 달라 값 비교와 다수결에서 뺀다. 먼저 나온 후보의 문장을 쓴다
            props = {"required": True if required is None else bool(required)}
        elif rel == "RESOLVED_BY":
            if p.get("condition"):
                if p["condition"] not in conditions:
                    stats["unknown_condition"] += 1
                props = {"condition": p["condition"], **conditions.get(p["condition"], {})}
        key = (rel, s["id"], o["id"], json.dumps(props, ensure_ascii=False, sort_keys=True))
        entry = facts.setdefault(key, {"order": c["_order"], "docs": set()})
        entry["docs"].add(c["document_id"])
        if rel == "CHECKED_BY" and p.get("expect"):
            expects.setdefault((s["id"], o["id"]), p["expect"])

    # 하나만 있어야 하는 자리는 가장 많이 나온 값 하나로 줄인다
    slots: dict[tuple, list] = defaultdict(list)
    chosen = []
    for (rel, start, end, props_json), entry in facts.items():
        props = json.loads(props_json)
        if rel in SINGLE:
            # 같은 값이 여러 문서에서 나오면 그 수만큼 센다
            for doc in sorted(entry["docs"]):
                slots[(rel, *SINGLE[rel](start, end, props))].append((entry["order"], end, props, doc))
        else:
            chosen.append((rel, start, end, props, sorted(entry["docs"])))
    for slot, rows in sorted(slots.items()):
        end, props, voted = _vote([(r[0], r[1], r[2]) for r in rows])
        stats["voted_slots"] += voted
        docs = sorted({r[3] for r in rows if r[1] == end and r[2] == props})
        chosen.append((slot[0], slot[1], end, props, docs))

    located = {start: (end, props) for rel, start, end, props, _ in chosen if rel == "LOCATED_IN"}
    stats["args_from_location"] = 0
    for rel, start, end, props, docs in sorted(chosen, key=lambda x: (x[0], x[1], x[2])):
        if rel == "CHECKED_BY" and params.get(end):
            # 조회 인자는 문서에서 뽑지 않는다. 원인 위치로 채운다 (결정 3)
            system, loc = located.get(start, (None, {}))
            values = {"system": system, "area": loc.get("area")}
            args = {name: values.get(name) for name in params[end] if values.get(name)}
            if args:
                props = props | {"args": json.dumps(args, ensure_ascii=False, sort_keys=True)}
                stats["args_from_location"] += 1
        if rel == "CHECKED_BY" and (start, end) in expects:
            props = props | {"expect": expects[(start, end)]}
        clean = {k: v for k, v in props.items() if v is not None}
        g.rels[rel].append({"start": start, "end": end,
                            "props": clean | {"source": "extracted", "source_doc_ids": docs}})
    stats["relationships"] = {rel: len(g.rels[rel]) for rel in EXTRACTED}
    return g, stats


def cext_hash(seed: dict, source: dict) -> str:
    """C-ext 그래프의 기준 해시: K 시드, 추출 결과 파일, 빌더 버전. 실행 결과에 남겨 같은 그래프인지 본다."""
    text = json.dumps({"seed": seed_hash(seed), "files": source["files_hash"], "builder": BUILDER_VERSION},
                      sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def load_cext(driver, seed: dict, folder: Path, rep: str) -> dict:
    """추출 결과 한 반복으로 C-ext 그래프를 만들어 driver의 DB에 덮어쓴다. 사건 계층과 추출 계층은 건드리지 않는다.

    추출 때의 시드와 지금 시드가 다르면 멈춘다. 이름 정규화의 기준 K가 달라진다.
    """
    from ontology.apply_schema import apply_schema

    apply_schema(driver)  # C-ext DB는 데이터를 남기지 않아 다시 뜨면 제약이 없다. 여러 번 불러도 같다
    candidates, source = load_candidates(folder, rep)
    if source["extraction_seed_hash"] != seed_hash(seed):
        raise CextError(f"{folder}: 추출 때의 시드와 지금 시드가 다르다. 같은 K로 추출한 결과만 쓴다")
    graph, stats = build_cext_graph(seed, candidates)
    digest = cext_hash(seed, source)
    meta = {"hash": digest, "org": seed["org"]["id"], "kind": "C-ext", "source": source["run_id"]}
    counts = write_graph(driver, graph, meta)
    return {"hash": digest, "source": source, "builder": BUILDER_VERSION, "stats": stats, "counts": counts}


def loaded_meta(driver) -> dict | None:
    with driver.session() as session:
        record = session.run(f"MATCH (m:{SEED_META} {{id: 'seed'}}) RETURN properties(m) AS m").single()
    return record["m"] if record else None


if __name__ == "__main__":
    folder, rep = Path(sys.argv[1]), sys.argv[2]
    org = sys.argv[3] if len(sys.argv) > 3 else "org-a"
    with get_driver(ext_uri()) as drv:
        out = load_cext(drv, load_seed(org), folder, rep)
    print(json.dumps({k: out[k] for k in ("hash", "stats", "counts")}, ensure_ascii=False, indent=2))
