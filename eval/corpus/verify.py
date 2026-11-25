"""생성한 문서에 설계표의 사실이 실제로 들어갔는지 확인한다 (M4 PR 2).

실행: python -m eval.corpus.verify [--only PM-01,CH-04]

문서마다 Sonnet 5.5(작업 이름 grading)에게 원문과 사실 목록을 주고, 사실마다 원문에 있는지와 근거 문장을 받는다.
근거 문장이 원문에 글자 그대로 없으면 코드가 '없음'으로 바꾼다. 방해 문서는 담당, 의존, 원인에 대한 사실을
담고 있지 않은지 본다. 결과는 eval/corpus/verify.yaml에 문서마다 남긴다.

없는 사실이 있는 문서는 다시 생성한다 (eval.corpus.generate --only --force). 설계표의 정답을 원문에 맞춰 고치지 않는다.
"""

import argparse
import sys
from pathlib import Path

import yaml

from common.files import write_text_atomic
from common.models import LLM, load_model_config, make_anthropic_client
from common.tracing import Tracer
from eval.corpus.check import load_plan
from eval.corpus.generate import describe, doc_path, names
from ontology.seed import load_seed

PROMPT_VERSION = "doc-verify-v1"
RESULT_PATH = Path(__file__).parent / "verify.yaml"

SYSTEM = """너는 사내 문서 검수자다. 문서 원문과 사실 목록을 받는다.
사실마다 원문이 그 사실을 말하고 있는지 판단한다. 다른 이름이나 돌려 말하기로 쓰였어도 같은 뜻이면 있다고 본다.
뜻이 다르거나 일부만 맞으면 없다고 본다. 있다고 볼 때는 그 사실을 보여주는 원문 문장을 글자 그대로 한 문장 옮긴다"""

NOISE_SYSTEM = """너는 사내 문서 검수자다. 이 문서는 장애 대응과 관계없는 문서여야 한다.
원문에 다음 중 하나라도 사실로 적혀 있으면 그 문장을 글자 그대로 옮긴다: 어떤 팀이 게임 시스템을 맡는다는 것,
게임 시스템끼리 의존한다는 것, 게임 장애의 원인이나 확인 방법이나 대응 절차. 없으면 빈 목록을 낸다"""

SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "present": {"type": "boolean"}, "quote": {"type": "string"}},
        "required": ["index", "present", "quote"], "additionalProperties": False}}},
    "required": ["facts"], "additionalProperties": False,
}
NOISE_SCHEMA = {
    "type": "object",
    "properties": {"quotes": {"type": "array", "items": {"type": "string"}}},
    "required": ["quotes"], "additionalProperties": False,
}


def checked_facts(doc: dict) -> list[tuple[str, dict]]:
    """확인할 사실: (구분, 사실). 별칭도 확인한다. 옛날 정보도 원문에 있어야 conflict로 잡을 수 있다."""
    return [("fact", f) for f in doc.get("facts") or []] + [("stale", f) for f in doc.get("stale") or []]


def verify_doc(llm: LLM, doc: dict, text: str, n: dict[str, str]) -> tuple[dict, float]:
    if doc["kind"] == "distractor":
        result = llm.call(task="grading", name=f"corpus.verify.{doc['id']}", system=NOISE_SYSTEM,
                          messages=[{"role": "user", "content": f"원문:\n{text}"}], output_schema=NOISE_SCHEMA,
                          max_tokens=1500, metadata={"doc": doc["id"], "prompt_version": PROMPT_VERSION})
        quotes = result.json()["quotes"]
        return {"ok": not quotes, "unexpected": quotes}, result.cost["total"]
    items = checked_facts(doc)
    listing = "\n".join(f"{i}. {describe(f, n)}" for i, (_, f) in enumerate(items))
    result = llm.call(task="grading", name=f"corpus.verify.{doc['id']}", system=SYSTEM,
                      messages=[{"role": "user", "content": f"원문:\n{text}\n\n사실 목록:\n{listing}"}],
                      output_schema=SCHEMA, max_tokens=3000,
                      metadata={"doc": doc["id"], "prompt_version": PROMPT_VERSION})
    answers = {a["index"]: a for a in result.json()["facts"]}
    missing = []
    for i, (kind, fact) in enumerate(items):
        a = answers.get(i)
        # 근거 문장이 원문에 없으면 있다고 믿지 않는다
        if not a or not a["present"] or not a["quote"].strip() or a["quote"].strip() not in text:
            missing.append({"kind": kind, "fact": fact, "said": describe(fact, n),
                            "reason": "답 없음" if not a else ("없다고 판단" if not a["present"] else "근거 문장이 원문에 없다")})
    return {"ok": not missing, "missing": missing}, result.cost["total"]


def main() -> int:
    parser = argparse.ArgumentParser(description="생성 문서의 사실 포함 확인 (Sonnet 5.5)")
    parser.add_argument("--only", help="쉼표로 나눈 문서 id. 주지 않으면 전부")
    parser.add_argument("--org", default="org-a")
    args = parser.parse_args()
    seed, plan = load_seed(args.org), load_plan()
    n = names(seed)
    wanted = set(args.only.split(",")) if args.only else None
    todo = [d for d in plan if wanted is None or d["id"] in wanted]
    absent = [d["id"] for d in todo if not doc_path(d).exists()]
    if absent:
        print(f"원문이 없는 문서가 있다: {absent}", file=sys.stderr)
        return 1
    record = yaml.safe_load(RESULT_PATH.read_text(encoding="utf-8")) if RESULT_PATH.exists() else {"documents": {}}
    tracer = Tracer.from_env()
    llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
    try:
        with tracer.run("verify_corpus", tags=["mode:generation"], metadata={"prompt_version": PROMPT_VERSION},
                        input={"documents": len(todo)}):
            for doc in todo:
                text = doc_path(doc).read_text(encoding="utf-8")
                outcome, cost = verify_doc(llm, doc, text, n)
                previous = record["documents"].get(doc["id"], {})
                record["documents"][doc["id"]] = outcome | {
                    "prompt_version": PROMPT_VERSION, "cost_usd": round(previous.get("cost_usd", 0) + cost, 5)}
                record["total_cost_usd"] = round(sum(r["cost_usd"] for r in record["documents"].values()), 5)
                write_text_atomic(RESULT_PATH, yaml.safe_dump(record, allow_unicode=True, sort_keys=True))
                print(f"{doc['id']}: {'통과' if outcome['ok'] else '부족'}")
    finally:
        tracer.flush()
    failed = sorted(i for i, r in record["documents"].items() if not r["ok"])
    print(f"확인 {len(todo)}개, 부족 {len(failed)}개 {failed}, 누적 ${record['total_cost_usd']:.5f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
