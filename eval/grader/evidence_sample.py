"""근거 문장 표본 확인: 추출 후보의 근거 문장이 그 관계를 뒷받침하는지 (설계 4단계 5-2, 5-5, M4 PR 4 결정 4, M4 점검).

실행: python -m eval.grader.evidence_sample eval/results/<추출 결과 폴더> r1 --n 30 --out eval/results/<점검 폴더>/evidence
      python -m eval.grader.evidence_sample ... --dry-run    # 표본과 예상 입력 크기만 본다 (LLM을 부르지 않는다)
      python -m eval.grader.evidence_sample --human <점검 폴더>/evidence/HUMAN.yaml <점검 폴더>/evidence/EVIDENCE.yaml
          # 사람 재확인 판정을 EVIDENCE.yaml의 human 칸에 넣고 일치 표를 계산한다 (LLM을 부르지 않는다)

추출 채점기는 근거 문장이 원문에 글자 그대로 있는지만 본다. 문장이 그 관계를 말하는지는 표본을 뽑아 Sonnet(채점 작업)이
판정하고 사람이 다시 확인한다. 판정은 셋이다: 뒷받침함, 일부만(관계는 맞지만 속성이나 한쪽 끝이 문장에 없음), 아님.
표본은 관계 종류마다 돌아가며 고른다(정해진 난수 씨앗). 판정 모델에는 문서 정답(설계표)을 주지 않는다. 근거 문장과 그
문장이 든 단락만 준다. 별칭 후보는 코드가 이름 정규화 결과로 만든 후보라 근거 문장이 이름이 나온 문장일 뿐이다(5단계 5장).
그래서 별칭은 "그 이름이 그 대상을 가리키는지"를 묻는다.
dev 문서 추출 결과만 쓴다 (4단계 4-2).
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from common.files import write_text_atomic
from common.models import LLM, load_model_config, make_anthropic_client
from common.tracing import Tracer
from eval.corpus.check import load_plan
from eval.corpus.load import document_fields

PROMPT_VERSION = "evidence-check-v1"
SEED = 20261125
VERDICTS = ("supported", "partial", "unsupported")
MEANING = {
    "OWNS": "{s} 팀이 {o} 시스템을 담당한다", "DEPENDS_ON": "{s} 시스템이 {o} 시스템에 의존한다",
    "OBSERVED_IN": "{s} 증상이 {o} 시스템에서 나타난다", "MAY_BE_CAUSED_BY": "{s} 증상의 원인 후보로 {o}가 있다",
    "LOCATED_IN": "{s} 원인은 {o} 시스템에 있다", "CHECKED_BY": "{s} 원인은 {o}로 확인한다",
    "RESOLVED_BY": "{s} 원인이 확정되면 {o} 절차를 연다", "ALIAS": "'{o}'는 {s}를 가리키는 다른 이름이다",
}
PROP_TEXT = {"scope": "담당 범위", "area": "영역", "prior": "가능성", "required": "확정 조건 여부", "condition": "조건",
             "expect": "확정 조건 설명"}
SYSTEM = (
    "너는 사내 문서에서 뽑은 지식 후보를 검수한다. 주장 하나와 그 근거로 든 문장, 문장이 든 단락을 받는다. "
    "근거 문장(필요하면 같은 단락의 앞뒤 문맥)이 주장을 말하는지 판정한다. 문서 밖의 상식이나 추측으로 채우지 않는다.\n"
    "- supported: 근거 문장이 주장의 관계와 속성을 모두 말한다\n"
    "- partial: 관계는 말하지만 속성(담당 범위, 가능성, 조건 등)이나 한쪽 대상이 근거 문장에 없거나 다르다\n"
    "- unsupported: 근거 문장이 이 관계를 말하지 않는다. 같은 단어가 나와도 다른 이야기다\n"
    "reason은 한국어 한두 문장으로 쓴다."
)
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}, "reason": {"type": "string"}},
    "required": ["verdict", "reason"], "additionalProperties": False,
}


def load_rep(folder: Path, rep: str) -> list[dict]:
    run = yaml.safe_load((folder / "run.yaml").read_text(encoding="utf-8"))
    if run.get("split") != "dev":
        raise SystemExit(f"{folder}: dev 문서 추출 결과만 쓴다 (split={run.get('split')})")
    out = []
    for path in sorted((folder / rep).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for i, c in enumerate(data.get("candidates") or []):
            out.append(c | {"document_id": data["document_id"], "candidate": f"{data['document_id']}:{i:03d}"})
    return out


def sample(candidates: list[dict], n: int) -> list[dict]:
    """관계 종류마다 돌아가며 고른다. 같은 입력이면 같은 표본이다."""
    rng = random.Random(SEED)
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_rel[c["relation"]].append(c)
    for rows in by_rel.values():
        rng.shuffle(rows)
    picked, rels = [], sorted(by_rel)
    while len(picked) < n and any(by_rel.values()):
        for rel in rels:
            if by_rel[rel] and len(picked) < n:
                picked.append(by_rel[rel].pop())
    return picked


def claim(c: dict) -> str:
    s, o = c["subject"], c["object"]
    text = MEANING[c["relation"]].format(s=s["name"], o=o["name"])
    props = {k: v for k, v in (c.get("props") or {}).items() if v not in (None, "")}
    if props:
        text += " (" + ", ".join(f"{PROP_TEXT.get(k, k)}: {v}" for k, v in props.items()) + ")"
    return text


def context(c: dict, docs: dict) -> str:
    paragraphs = [p for p in docs[c["document_id"]].split("\n\n") if any(span in p for span in c["evidence_spans"])]
    return "\n\n".join(paragraphs)


def prompt(c: dict, docs: dict) -> str:
    spans = "\n".join(f"- {s}" for s in c["evidence_spans"])
    return f"주장: {claim(c)}\n\n근거 문장:\n{spans}\n\n근거 문장이 든 단락:\n{context(c, docs)}"


def merge_human(evidence: dict, human: dict) -> dict:
    """사람 판정을 표본에 넣고 Sonnet 판정과의 일치를 센다. 사람이 보지 않은 표본은 human을 비워 둔다."""
    judged = human["judgments"]
    by_id = {s["candidate"]: s for s in evidence["samples"]}
    unknown = sorted(set(judged) - set(by_id))
    if unknown:
        raise ValueError(f"표본에 없는 후보: {unknown}")
    for cid, h in judged.items():
        if h["verdict"] not in VERDICTS:
            raise ValueError(f"{cid}: 판정은 {VERDICTS} 중 하나다 ({h['verdict']})")
    for s in evidence["samples"]:
        s["human"] = judged.get(s["candidate"])
    reviewed = [s for s in evidence["samples"] if s["human"]]
    matrix = {m: {h: 0 for h in VERDICTS} for m in VERDICTS}
    for s in reviewed:
        matrix[s["verdict"]][s["human"]["verdict"]] += 1
    evidence["human_review"] = {
        "reviewer": human["reviewer"], "date": human["date"], "how_picked": human["how_picked"],
        "reviewed": len(reviewed), "agree": sum(1 for s in reviewed if s["verdict"] == s["human"]["verdict"]),
        "human_summary": {v: sum(1 for s in reviewed if s["human"]["verdict"] == v) for v in VERDICTS},
        "model_by_human": matrix, "model_by_human_note": "바깥 키는 Sonnet 판정, 안쪽 키는 사람 판정이다",
    }
    return evidence


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--human":
        human_path, evidence_path = Path(argv[1]), Path(argv[2])
        evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
        merged = merge_human(evidence, yaml.safe_load(human_path.read_text(encoding="utf-8")))
        write_text_atomic(evidence_path, yaml.safe_dump(merged, allow_unicode=True, sort_keys=False))
        r = merged["human_review"]
        print(f"사람 재확인 {r['reviewed']}개, Sonnet과 일치 {r['agree']}개, 사람 판정 {r['human_summary']}")
        return 0
    parser = argparse.ArgumentParser(description="근거 문장 표본 확인 (Sonnet 판정, 사람 재확인)")
    parser.add_argument("folder")
    parser.add_argument("rep")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--out", required=False)
    parser.add_argument("--org", default="org-a")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    docs = {d["id"]: document_fields(d)["text"] for d in load_plan() if d["split"] == "dev"}
    picked = sample(load_rep(Path(args.folder), args.rep), args.n)
    prompts = [prompt(c, docs) for c in picked]
    if args.dry_run:
        counts = defaultdict(int)
        for c in picked:
            counts[c["relation"]] += 1
        chars = sum(len(p) for p in prompts) + len(SYSTEM) * len(prompts)
        print(f"표본 {len(picked)}개 {dict(counts)}, 입력 {chars}자 (호출 {len(prompts)}회)")
        return 0
    if not args.out:
        parser.error("--out이 필요하다")
    tracer = Tracer.from_env()
    llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
    rows, total = [], 0.0
    try:
        with tracer.run("evidence_sample", tags=["mode:audit"], metadata={"prompt_version": PROMPT_VERSION},
                        input={"folder": args.folder, "rep": args.rep, "n": len(picked)}):
            for c, text in zip(picked, prompts):
                result = llm.call(task="grading", name=f"evidence.{c['candidate']}", system=SYSTEM,
                                  messages=[{"role": "user", "content": text}], max_tokens=400, output_schema=SCHEMA,
                                  metadata={"candidate": c["candidate"], "prompt_version": PROMPT_VERSION})
                judged = result.json()
                total += result.cost["total"]
                rows.append({"candidate": c["candidate"], "relation": c["relation"], "match": c.get("match"),
                             "claim": claim(c), "evidence_spans": c["evidence_spans"], "verdict": judged["verdict"],
                             "reason": judged["reason"], "human": None, "model": result.model,
                             "cost_usd": round(result.cost["total"], 6)})
                print(f"{c['candidate']} {c['relation']}: {judged['verdict']}")
    finally:
        tracer.flush()
    summary = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
    by_rel = defaultdict(lambda: {v: 0 for v in VERDICTS})
    for r in rows:
        by_rel[r["relation"]][r["verdict"]] += 1
    out = {"folder": args.folder, "rep": args.rep, "prompt_version": PROMPT_VERSION, "seed": SEED, "n": len(rows),
           "summary": summary, "by_relation": dict(by_rel), "total_cost_usd": round(total, 5), "samples": rows}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out_dir / "EVIDENCE.yaml", yaml.safe_dump(out, allow_unicode=True, sort_keys=False))
    print(f"판정 {summary}, 비용 ${total:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
