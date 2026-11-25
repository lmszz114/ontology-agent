"""현장 문서 설계표(eval/corpus/plan.yaml) 검사 (설계 4단계 3-5).

실행: python -m eval.corpus.check

설계표의 facts는 기준 지식 K(시드)에 실제로 있는 사실이어야 한다. 문서 묶음 D에는 K에 없는 새 사실을 넣지 않는다.
stale은 반대로 K와 어긋나는 옛날 정보여야 하고, 추출에서 conflict로 잡혀야 정답이다 (R8-5: 같은 시스템, 같은 scope와
area를 다른 팀이 OWNS). 별칭(ALIAS)은 K에 없는 이름을 쓸 수 있다. 문서마다 같은 대상을 부르는 표현이다.
LLM을 부르지 않는다.
"""

import sys
from pathlib import Path

import yaml

from ontology.load import build_graph
from ontology.seed import load_seed

PLAN_PATH = Path(__file__).parent / "plan.yaml"
KINDS = {"postmortem": "PM", "runbook": "RB", "chat": "CH", "meeting": "MT", "distractor": "NZ"}
# 추출 대상 관계 (3단계 R8 3번). ALIAS는 aliases 속성이다
RELATIONS = {"OWNS", "DEPENDS_ON", "OBSERVED_IN", "MAY_BE_CAUSED_BY", "LOCATED_IN", "CHECKED_BY", "RESOLVED_BY", "ALIAS"}
# 관계 속성 중 K와 비교하는 것. CHECKED_BY의 required는 시드의 when 유무로 정해진다
COMPARED_PROPS = {"OWNS": ("scope", "area"), "LOCATED_IN": ("scope", "area"), "MAY_BE_CAUSED_BY": ("prior",),
                  "CHECKED_BY": ("required", "expect"), "RESOLVED_BY": ("condition",)}


def load_plan(path: Path = PLAN_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["documents"]


def k_relations(seed: dict) -> dict[tuple[str, str, str], list[dict]]:
    """(관계, 시작 id, 끝 id)마다 K에 있는 관계 속성 목록. OWNS는 scope마다 관계가 하나씩 있다."""
    graph = build_graph(seed)
    out: dict = {}
    for rel, rows in graph.rels.items():
        for row in rows:
            out.setdefault((rel, row["start"], row["end"]), []).append(row["props"])
    return out


def _matches(fact: dict, props_list: list[dict]) -> bool:
    """사실의 속성이 K의 관계 하나와 같은지. 설명 문장(expect)은 적었을 때만 비교하고 나머지는 없음까지 같아야 한다."""
    wanted = fact.get("props") or {}
    keys = COMPARED_PROPS.get(fact["rel"], ())
    return any(all(p.get(key) == wanted.get(key) for key in keys if key != "expect" or "expect" in wanted)
               for p in props_list)


def _conflicts(fact: dict, k: dict) -> bool:
    """옛날 정보가 K의 OWNS와 모순되는지: 같은 시스템, 같은 scope와 area를 K에서는 다른 팀이 맡는다."""
    if fact["rel"] != "OWNS":
        return False
    p = fact.get("props") or {}
    for (rel, team, system), props_list in k.items():
        if rel == "OWNS" and system == fact["object"] and team != fact["subject"]:
            if any(q.get("scope") == p.get("scope") and q.get("area") == p.get("area") for q in props_list):
                return True
    return False


def check_plan(seed: dict, docs: list[dict]) -> list[str]:
    errors: list[str] = []
    k = k_relations(seed)
    ids = [d["id"] for d in docs]
    errors += [f"{i}: 같은 id가 두 번 있다" for i in sorted({i for i in ids if ids.count(i) > 1})]
    causes = {c["id"] for c in seed["causes"]}
    # aliases 속성이 있는 노드 (3단계 2-1): 팀, 시스템, 증상, 원인
    alias_targets = {item["id"] for key in ("teams", "systems", "symptoms", "causes") for item in seed[key]}
    for d in docs:
        where = d["id"]
        if d.get("kind") not in KINDS or not where.startswith(KINDS[d["kind"]] + "-"):
            errors.append(f"{where}: kind와 id 접두어가 맞지 않는다")
        if d.get("split") not in ("dev", "test"):
            errors.append(f"{where}: split은 dev 또는 test다")
        if not d.get("brief"):
            errors.append(f"{where}: brief(문서 요지)가 없다")
        if d["kind"] == "distractor" and (d.get("facts") or d.get("stale") or d.get("causes")):
            errors.append(f"{where}: 방해 문서에는 사실을 넣지 않는다")
        for c in d.get("causes") or []:
            if c not in causes:
                errors.append(f"{where}: 정의되지 않은 원인이다: {c}")
        for fact in d.get("facts") or []:
            if fact["rel"] not in RELATIONS:
                errors.append(f"{where}: 추출 대상이 아닌 관계다: {fact['rel']}")
            elif fact["rel"] == "ALIAS":
                if fact["subject"] not in alias_targets:
                    errors.append(f"{where}: 별칭을 붙일 수 없는 대상이다 (팀, 시스템, 증상, 원인의 id만): {fact['subject']}")
                if not isinstance(fact["object"], str) or not fact["object"].strip():
                    errors.append(f"{where}: 별칭이 비어 있다")
            elif not _matches(fact, k.get((fact["rel"], fact["subject"], fact["object"]), [])):
                errors.append(f"{where}: K에 없는 사실이다: {fact}")
        for fact in d.get("stale") or []:
            if (fact["rel"], fact["subject"], fact["object"]) in k and _matches(fact, k[(fact["rel"], fact["subject"], fact["object"])]):
                errors.append(f"{where}: 옛날 정보인데 K와 같다: {fact}")
            elif not _conflicts(fact, k):
                errors.append(f"{where}: 옛날 정보가 K의 담당과 모순되지 않는다 (conflict로 잡을 수 없다): {fact}")
    covered = {c for d in docs if d["kind"] == "postmortem" for c in d.get("causes") or []}
    missing = sorted(causes - covered)
    if missing:
        errors.append(f"장애 회고가 없는 원인이 있다: {missing}")
    return errors


def main() -> int:
    docs = load_plan()
    errors = check_plan(load_seed("org-a"), docs)
    for line in errors:
        print(line)
    counts = {kind: sum(1 for d in docs if d["kind"] == kind) for kind in KINDS}
    splits = {s: sum(1 for d in docs if d["split"] == s) for s in ("dev", "test")}
    print(f"문서 {len(docs)}개 {counts} {splits}, 사실 {sum(len(d.get('facts') or []) for d in docs)}개, "
          f"옛날 정보 {sum(len(d.get('stale') or []) for d in docs)}개, 문제 {len(errors)}건")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
