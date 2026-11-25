"""현장 문서 원문 생성 (설계 4단계 3-5, M4 PR 2).

실행: python -m eval.corpus.generate [--only PM-01,CH-04] [--force]

설계표(eval/corpus/plan.yaml)의 사실을 Sonnet 5.5(작업 이름 generation)가 현장 말투의 문서로 쓴다.
평가 대상(Haiku 5.5)과 다른 모델이다 (4단계 4-2, L15). 문서는 eval/corpus/docs/{split}/{id}.md에 쓰고,
생성 기록(모델, 프롬프트 버전, 비용)은 eval/corpus/generation.yaml에 문서마다 남긴다.

생성 지시에는 문서에 담을 사실만 넣는다. 추출 정답의 형식(id, 관계 이름)은 넣지 않고 이름으로 풀어 쓴다.
이미 있는 문서는 건너뛴다. 다시 쓰려면 --only와 --force를 함께 준다.
"""

import argparse
import sys
from pathlib import Path

import yaml

from common.files import write_text_atomic
from common.models import LLM, load_model_config, make_anthropic_client
from common.tracing import Tracer
from eval.corpus.check import load_plan
from ontology.seed import load_seed

PROMPT_VERSION = "doc-gen-v1"
CORPUS_DIR = Path(__file__).parent
DOCS_DIR = CORPUS_DIR / "docs"
RECORD_PATH = CORPUS_DIR / "generation.yaml"

SCOPES = {"logic": "로직", "data": "데이터", "config": "설정", "operation": "운영", "execution": "실행"}
AREAS = {"content": "콘텐츠 영역", "probability": "확률 영역"}
LENGTH = {"postmortem": "900자에서 1500자", "runbook": "600자에서 1000자", "chat": "메시지 15개에서 30개",
          "meeting": "600자에서 1000자", "distractor": "300자에서 700자"}
STYLE = {
    "postmortem": "사내 위키의 장애 회고 문서. 사건 개요, 타임라인, 확인한 근거, 원인, 담당, 조치, 재발 방지를 쓴다. 문장체",
    "runbook": "사내 위키의 대응 절차 문서(런북). 언제 쓰는지, 단계, 주의 사항을 쓴다. 문장체",
    "chat": "사내 메신저 채널 대화 기록. 한 줄에 '이름(팀): 메시지' 형식. 추측, 정정, 농담, 딴소리가 섞인다",
    "meeting": "회의록. 참석 팀, 안건, 논의, 결정, 후속 작업을 쓴다. 개조식과 문장이 섞인다",
    "distractor": "사내 문서. 주제에 맞는 일반 문서",
}

SYSTEM = """너는 가상 MMORPG 라이브 서비스 회사의 사내 문서를 쓰는 작가다.
- 주어진 사실을 모두 문서에 담는다. 사실을 빼거나 바꾸지 않는다. 주어진 것 말고 담당 팀, 시스템 관계, 원인에 대한 새 사실을 지어내지 않는다
- 사실을 표처럼 나열하지 않는다. 여러 문장에 흩어 쓰고, 돌려 말하기와 대명사를 섞는다. 다만 읽는 사람이 사실을 알아낼 수 있어야 한다
- '다른 이름'이 주어지면 그 대상을 문서의 대부분에서 그 이름으로 부른다
- '옛날 정보'는 문서가 쓰인 당시의 사실로 자연스럽게 쓴다. 지금은 다르다는 말을 넣지 않는다
- 실제 회사, 게임, 서비스, 사람 이름을 쓰지 않는다. 사람은 '김OO'처럼 성과 OO로 쓴다
- 마크다운 본문만 쓴다. 제목은 '# '로 시작하는 한 줄이다"""


def names(seed: dict) -> dict[str, str]:
    out = {}
    for key in ("teams", "systems", "symptoms", "causes", "evidence_checks", "procedures"):
        for item in seed[key]:
            out[item["id"]] = item["name"]
    return out


def describe(fact: dict, n: dict[str, str]) -> str:
    """사실 하나를 이름으로 풀어 쓴 문장. 생성 지시에만 쓴다."""
    rel, s, o, p = fact["rel"], fact["subject"], fact["object"], fact.get("props") or {}
    where = SCOPES.get(p.get("scope"), "") + (f"({AREAS[p['area']]})" if p.get("area") else "")
    if rel == "OWNS":
        return f"{n[s]}이 {n[o]}의 {where}을(를) 맡는다"
    if rel == "DEPENDS_ON":
        return f"{n[s]}은(는) {n[o]}에 의존한다 ({n[o]}에 문제가 생기면 {n[s]}에 영향이 간다)"
    if rel == "OBSERVED_IN":
        return f"'{n[s]}' 증상은 {n[o]}에서 나타난다"
    if rel == "MAY_BE_CAUSED_BY":
        return f"'{n[s]}' 증상의 원인 후보로 '{n[o]}'가 있다. 이런 증상에서 이 원인일 가능성은 {p['prior']} 편이다"
    if rel == "LOCATED_IN":
        return f"'{n[s]}'는 {n[o]}의 {where} 쪽 문제다"
    if rel == "CHECKED_BY":
        text = f"'{n[s]}'인지는 '{n[o]}'로 확인한다"
        return text + (f". 이 원인이면 확인 결과가 이렇다: {p['expect']}" if p.get("expect") else "")
    if rel == "RESOLVED_BY":
        cond = f" ({p['condition']}일 때)" if p.get("condition") else ""
        return f"'{n[s]}'가 원인이면 '{n[o]}' 절차를 연다{cond}"
    if rel == "ALIAS":
        return f"이 문서에서는 {n[s]}을(를) '{o}'라고 부른다"
    raise ValueError(rel)


def prompt(doc: dict, n: dict[str, str]) -> str:
    facts = [f for f in doc.get("facts") or [] if f["rel"] != "ALIAS"]
    aliases = [f for f in doc.get("facts") or [] if f["rel"] == "ALIAS"]
    lines = [f"문서 종류: {STYLE[doc['kind']]}", f"제목 요지: {doc['title']}", f"내용 요지: {doc['brief']}",
             f"작성 시점: {doc['created_at']}", f"분량: {LENGTH[doc['kind']]}"]
    if doc.get("author_team"):
        lines.append(f"작성 팀: {n[doc['author_team']]}")
    if facts:
        lines += ["", "문서에 담을 사실 (모두 담는다):"] + [f"- {describe(f, n)}" for f in facts]
    if doc.get("stale"):
        lines += ["", "옛날 정보 (문서 작성 당시의 사실로 쓴다):"] + [f"- {describe(f, n)}" for f in doc["stale"]]
    if aliases:
        lines += ["", "다른 이름:"] + [f"- {describe(f, n)}" for f in aliases]
    if doc["kind"] == "distractor":
        lines += ["", "장애 원인, 팀의 시스템 담당, 시스템 사이 의존 관계에 대한 사실은 쓰지 않는다. 비슷한 단어가 나와도 다른 뜻으로 쓴다"]
    return "\n".join(lines)


def doc_path(doc: dict) -> Path:
    return DOCS_DIR / doc["split"] / f"{doc['id']}.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="현장 문서 원문 생성 (Sonnet 5.5)")
    parser.add_argument("--only", help="쉼표로 나눈 문서 id. 주지 않으면 없는 문서 전부")
    parser.add_argument("--force", action="store_true", help="--only로 고른 문서를 다시 쓴다")
    parser.add_argument("--org", default="org-a")
    args = parser.parse_args()
    if args.force and not args.only:
        parser.error("--force는 --only와 함께 쓴다")
    seed, plan = load_seed(args.org), load_plan()
    n = names(seed)
    wanted = set(args.only.split(",")) if args.only else None
    todo = [d for d in plan if (wanted is None or d["id"] in wanted) and (args.force or not doc_path(d).exists())]
    record = yaml.safe_load(RECORD_PATH.read_text(encoding="utf-8")) if RECORD_PATH.exists() else {"documents": {}}

    tracer = Tracer.from_env()
    llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
    total = 0.0
    try:
        with tracer.run("generate_corpus", tags=["mode:generation"], metadata={"prompt_version": PROMPT_VERSION},
                        input={"documents": len(todo)}):
            for doc in todo:
                result = llm.call(task="generation", name=f"corpus.{doc['id']}", system=SYSTEM,
                                  messages=[{"role": "user", "content": prompt(doc, n)}], max_tokens=4000,
                                  metadata={"doc": doc["id"], "prompt_version": PROMPT_VERSION})
                text = result.text.strip() + "\n"
                path = doc_path(doc)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_text_atomic(path, text)
                cost = result.cost["total"]
                total += cost
                previous = record["documents"].get(doc["id"], {})
                # 다시 쓴 문서는 비용을 더한다. 누적 비용이 실제 지출이다
                record["documents"][doc["id"]] = {
                    "model": result.model, "prompt_version": PROMPT_VERSION,
                    "cost_usd": round(previous.get("cost_usd", 0) + cost, 5),
                    "generations": previous.get("generations", 0) + 1, "stop_reason": result.message.stop_reason}
                record["total_cost_usd"] = round(sum(r["cost_usd"] for r in record["documents"].values()), 5)
                # 문서마다 저장한다. 중간에 멈춰도 이미 비용을 낸 결과가 남는다
                write_text_atomic(RECORD_PATH, yaml.safe_dump(record, allow_unicode=True, sort_keys=True))
                print(f"{doc['id']}: {len(text)}자, ${cost:.5f}")
    finally:
        tracer.flush()
    print(f"문서 {len(todo)}개, 이번 비용 ${total:.5f}, 누적 ${record.get('total_cost_usd', 0):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
