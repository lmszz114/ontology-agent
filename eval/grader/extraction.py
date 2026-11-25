"""추출 채점기 (설계 4단계 5-2, 5-5, M4 PR 4).

실행: python -m eval.grader.extraction eval/results/<폴더>

결과 파일(<폴더>/rN/*.yaml)과 문서 설계표(eval/corpus/plan.yaml)의 정답을 비교한다. 설계표가 실행 때와 다르면 멈춘다.
반복마다 지표를 내고 평균과 표본 표준편차(n-1)를 낸다. 결과는 <폴더>/GRADE.yaml에 쓴다.

정답과 후보는 (관계, 주어 id, 목적어 id)로 맞춘다. 별칭은 (노드 id, 별칭)이다. 이름 정규화로 기존 노드를 못 찾은 후보는
정답과 맞을 수 없어 틀린 후보로 센다. 정답은 설계표의 facts와 stale(옛날 정보)이다. 옛날 정보도 문서가 말한 사실이라
뽑은 것 자체는 맞고, conflict로 잡았는지는 충돌 탐지율로 따로 본다.
근거 문장이 관계를 뒷받침하는지는 이 채점기가 보지 않는다. 원문에 글자 그대로 있는지만 본다. 뒷받침 여부는 표본을 뽑아
사람과 Sonnet이 본다 (M4 PR 4 결정 4, 4단계 5-5).
"""

import hashlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from common.files import write_text_atomic
from eval.corpus.check import COMPARED_PROPS, PLAN_PATH, load_plan

RELATIONS = ("OWNS", "DEPENDS_ON", "OBSERVED_IN", "MAY_BE_CAUSED_BY", "LOCATED_IN", "CHECKED_BY", "RESOLVED_BY", "ALIAS")


class GradeError(RuntimeError):
    pass


def fact_key(rel: str, subject: str, obj: str) -> tuple:
    return (rel, subject, obj)


def candidate_key(c: dict) -> tuple:
    s, o = c["subject"], c["object"]
    if c["relation"] == "ALIAS":
        return ("ALIAS", s.get("id") or f"?{s['name']}", o["name"])
    return (c["relation"], s.get("id") or f"?{s['name']}", o.get("id") or f"?{o['name']}")


def gold_for(doc: dict) -> tuple[dict, dict]:
    """문서의 정답: (사실 키 -> 사실), (옛날 정보 키 -> 사실)."""
    facts = {fact_key(f["rel"], f["subject"], f["object"]): f for f in doc.get("facts") or []}
    stale = {fact_key(f["rel"], f["subject"], f["object"]): f for f in doc.get("stale") or []}
    return facts, stale


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def grade_rep(rep_dir: Path, plan: list[dict]) -> dict:
    by_id = {d["id"]: d for d in plan}
    counts = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in RELATIONS}
    props_total = props_ok = 0
    norm_total = norm_ok = 0
    raw_checked = evidence_bad = 0
    stale_total = stale_found = stale_conflict = 0
    noise = 0
    in_k_not_gold = 0
    cost = 0.0
    files = sorted(rep_dir.glob("*.yaml"))
    for path in files:
        r = yaml.safe_load(path.read_text(encoding="utf-8"))
        doc = by_id[r["document_id"]]
        cost += r["cost_usd"]
        # 근거 문장 일치율: 타입 검사를 통과한 원래 후보 중 근거가 원문에 있는 비율
        bad_ev = sum(1 for d in r["dropped"] if d["reason"] == "근거 문장이 원문에 없다")
        other = sum(1 for d in r["dropped"] if d["reason"] != "근거 문장이 원문에 없다")
        raw_checked += r["stats"].get("raw", 0) - other
        evidence_bad += bad_ev
        if doc["kind"] == "distractor":
            noise += len(r["candidates"])
        facts, stale = gold_for(doc)
        gold = facts | stale
        cands: dict[tuple, dict] = {}
        for c in r["candidates"]:
            cands.setdefault(candidate_key(c), c)
        for key, c in cands.items():
            rel = key[0]
            if key in gold:
                counts[rel]["tp"] += 1
                wanted = gold[key].get("props") or {}
                for prop in COMPARED_PROPS.get(rel, ()):
                    if prop == "expect" or prop not in wanted:
                        continue
                    props_total += 1
                    props_ok += (c.get("props") or {}).get(prop) == wanted[prop]
            else:
                counts[rel]["fp"] += 1
                in_k_not_gold += c.get("match") == "existing"
        for key in gold:
            if key not in cands:
                counts[key[0]]["fn"] += 1
        for key in stale:
            stale_total += 1
            if key in cands:
                stale_found += 1
                stale_conflict += cands[key].get("match") == "conflict"
        # 이름 정규화: 문서의 정답 별칭으로 쓰인 이름이 정답 노드에 이어졌는가
        aliases = {f["object"]: f["subject"] for f in doc.get("facts") or [] if f["rel"] == "ALIAS"}
        for c in r["candidates"]:
            if c["relation"] == "ALIAS":
                continue
            for end in (c["subject"], c["object"]):
                if end["name"] in aliases:
                    norm_total += 1
                    norm_ok += end.get("id") == aliases[end["name"]]
    per_rel = {}
    for rel, n in counts.items():
        p = _ratio(n["tp"], n["tp"] + n["fp"])
        rc = _ratio(n["tp"], n["tp"] + n["fn"])
        per_rel[rel] = n | {"precision": p, "recall": rc, "f1": _f1(p, rc)}
    tp = sum(n["tp"] for n in counts.values())
    fp = sum(n["fp"] for n in counts.values())
    fn = sum(n["fn"] for n in counts.values())
    p, rc = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "documents": len(files),
        "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc,
                    "f1": _f1(p, rc)},
        "relations": per_rel,
        "props_accuracy": _ratio(props_ok, props_total), "props_compared": props_total,
        "normalization_accuracy": _ratio(norm_ok, norm_total), "normalization_names": norm_total,
        "evidence_match_rate": _ratio(raw_checked - evidence_bad, raw_checked),
        "conflict_detection_rate": _ratio(stale_conflict, stale_total),
        "stale_extracted": stale_found, "stale_total": stale_total,
        "distractor_candidates": noise,
        "false_positive_in_k": in_k_not_gold,
        "cost_usd": round(cost, 6),
    }


def _summary(values: list) -> dict | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 4), "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else None}


def grade(out: Path) -> dict:
    run = yaml.safe_load((out / "run.yaml").read_text(encoding="utf-8"))
    if run["plan_hash"] != hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest():
        raise GradeError("문서 설계표(정답)가 실행 때와 다르다. 같은 정답으로 채점할 수 없다")
    plan = load_plan()
    reps = sorted(p for p in out.glob("r*") if p.is_dir() and p.name[1:].isdigit())
    if not reps:
        raise GradeError(f"반복 결과가 없다: {out}")
    results = {p.name: grade_rep(p, plan) for p in reps}
    keys = ["precision", "recall", "f1"]
    summary = {
        "overall": {k: _summary([r["overall"][k] for r in results.values()]) for k in keys},
        "relations": {rel: {k: _summary([r["relations"][rel][k] for r in results.values()]) for k in keys}
                      for rel in RELATIONS},
    }
    for metric in ("props_accuracy", "normalization_accuracy", "evidence_match_rate", "conflict_detection_rate",
                   "distractor_candidates", "false_positive_in_k", "cost_usd"):
        summary[metric] = _summary([r[metric] for r in results.values()])
    return {"run": {k: run[k] for k in ("prompt_version", "git_commit", "git_dirty", "models", "started_at")},
            "reps": len(results), "summary": summary, "per_rep": results}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("사용: python -m eval.grader.extraction eval/results/<폴더>", file=sys.stderr)
        return 2
    out = Path(argv[0])
    try:
        result = grade(out)
    except GradeError as exc:
        print(f"채점할 수 없다: {exc}", file=sys.stderr)
        return 1
    write_text_atomic(out / "GRADE.yaml", yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    s = result["summary"]
    print(f"반복 {result['reps']}회, F1 {s['overall']['f1']}, 정밀도 {s['overall']['precision']}, "
          f"재현율 {s['overall']['recall']}")
    for rel in RELATIONS:
        print(f"  {rel}: F1 {s['relations'][rel]['f1']}")
    for metric in ("props_accuracy", "normalization_accuracy", "evidence_match_rate", "conflict_detection_rate",
                   "distractor_candidates", "false_positive_in_k", "cost_usd"):
        print(f"  {metric}: {s[metric]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
