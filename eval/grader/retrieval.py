"""검색 채점 (설계 4단계 5-3 검색 지표, 2-5 검색 설정 비교, 7장 G2, G4, G5).

실행: python -m eval.grader.retrieval --out eval/results/<폴더>      # dev 시나리오, 설정 12가지
      python -m eval.grader.retrieval --dry-run                     # 사건 수와 검색 횟수만 본다
      python -m eval.grader.retrieval --out ... --no-rerank         # 리랭크 없는 6가지만 (Voyage 속도 제한 때문에 나눠 돌릴 때)

진단 없이 검색 단계만 채점한다 (G4). LLM을 부르지 않는다. Voyage 임베딩과 리랭크만 부른다.
- 검색어: 정답 사건의 제보 원문을 이어 붙인 것. 구성 B 진단도 사건의 제보 원문으로 검색한다 (M5 사용자 결정).
  진단에서는 에이전트가 묶은 사건의 제보를 쓰므로, 묶음이 틀리면 검색어가 달라진다
- 정답 문서: eval/corpus/relevant.py. 정답 문서가 없는 사건(원인 미확정, 범위 밖)은 뺀다
- 설정: 조각 방식(paragraph, fixed) × 리랭크(없음, 있음) × top-k(3, 5, 10). top-k는 조각 수다.
  리랭크는 10개로 한 번 검색하고 잘라 쓰고, 벡터 검색은 k마다 다시 검색한다
- Recall@k: 사건의 정답 문서 중 상위 k개 조각의 문서에 들어온 비율의 평균
- MRR: 조각 순서대로 본 문서 중 첫 정답 문서 순위(문서 단위)의 역수 평균. k개 안에 없으면 0
- 옛날 문서 비율: 그 사건 원인의 옛날 정보 문서가 상위 k개에 들어온 사건의 비율
- 무작위 기준선 (G2): 같은 방식의 조각에서 k개를 무작위로 고를 때의 Recall@k 기댓값 (정해진 씨앗으로 2000번)
"""

import argparse
import random
import sys
from pathlib import Path

import yaml

from common.files import write_text_atomic
from common.retrieval.chunking import SCHEMES, all_chunks
from common.retrieval.search import TOP_KS, docs_of, search
from eval.corpus.check import load_plan
from eval.corpus.relevant import relevant_docs, stale_docs
from eval.labels import label_incidents

SEED = 20261126
RANDOM_DRAWS = 2000
MAX_K = max(TOP_KS)


def queries(scenarios: list[dict], plan: list[dict]) -> list[dict]:
    """정답 문서가 있는 사건마다 {scenario, incident, query, relevant, stale}."""
    out = []
    for sc in scenarios:
        for i, (incident, rel, stale) in enumerate(zip(label_incidents(sc["expected"]), relevant_docs(sc, plan),
                                                       stale_docs(sc, plan))):
            if not rel:
                continue
            text = "\n".join(sc["reports"][r]["text"] for r in incident["reports"])
            out.append({"scenario": sc["id"], "incident": i, "query": text, "relevant": rel, "stale": stale})
    return out


def score(docs: list[str], relevant: list[str], stale: list[str]) -> dict:
    hits = [d for d in docs if d in relevant]
    first = next((i for i, d in enumerate(docs, start=1) if d in relevant), None)
    return {"recall": len(set(hits)) / len(relevant), "rr": 1 / first if first else 0.0,
            "stale": any(d in stale for d in docs)}


def random_recall(chunk_docs: list[str], relevant: list[str], k: int, rng: random.Random) -> float:
    total = 0.0
    for _ in range(RANDOM_DRAWS):
        picked = {chunk_docs[i] for i in rng.sample(range(len(chunk_docs)), k)}
        total += len(picked & set(relevant)) / len(relevant)
    return total / RANDOM_DRAWS


def evaluate(driver, embedder, qs: list[dict], chunk_docs: dict[str, list[str]],
             reranks: tuple[bool, ...] = (False, True)) -> dict:
    rows, configs = [], {}
    for q in qs:
        for scheme in SCHEMES:
            for rerank in reranks:
                # 리랭크는 같은 후보를 정렬하므로 k개가 10개의 앞부분과 같다. 벡터 검색은 근사 검색이라 LIMIT이 다르면
                # 앞부분이 같다고 보장할 수 없어 k마다 다시 검색한다 (PR #35 리뷰, 검색어 임베딩은 캐시에서 읽는다)
                once = search(driver, embedder, q["query"], scheme=scheme, k=MAX_K, rerank=True) if rerank else None
                for k in TOP_KS:
                    found = once[:k] if rerank else search(driver, embedder, q["query"], scheme=scheme, k=k)
                    s = score(docs_of(found), q["relevant"], q["stale"])
                    key = f"{scheme}/{'rerank' if rerank else 'vector'}/k{k}"
                    configs.setdefault(key, []).append(s)
                    rows.append({"scenario": q["scenario"], "incident": q["incident"], "config": key,
                                 "docs": docs_of(found)} | s)
    rng = random.Random(SEED)
    summary = {}
    for key, scores in configs.items():
        scheme, _, k = key.split("/")
        k = int(k[1:])
        n = len(scores)
        summary[key] = {"recall": round(sum(s["recall"] for s in scores) / n, 4),
                        "mrr": round(sum(s["rr"] for s in scores) / n, 4),
                        "stale_rate": round(sum(s["stale"] for s in scores) / n, 4)}
    baseline = {}
    for scheme in SCHEMES:
        for k in TOP_KS:
            baseline[f"{scheme}/random/k{k}"] = round(
                sum(random_recall(chunk_docs[scheme], q["relevant"], k, rng) for q in qs) / len(qs), 4)
    return {"incidents": len(qs), "summary": summary, "random_recall": baseline, "rows": rows}


def markdown(result: dict) -> str:
    lines = [f"# 검색 설정 비교 (dev, 사건 {result['incidents']}개)", "",
             f"- 실행: {result['run']['date']} (KST), 임베딩 {result['run']['embed_model']}, 리랭크 {result['run']['rerank_model']}",
             "- Recall@k, MRR, 옛날 문서 비율은 사건 평균이다. top-k는 조각 수다. 무작위는 같은 방식의 조각에서 k개를 고른 기댓값이다",
             "- 채점기(eval/grader/retrieval)가 계산했다. LLM을 쓰지 않는다", "",
             "| 조각 방식 | 리랭크 | k | Recall@k | MRR | 옛날 문서 비율 | 무작위 Recall@k |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, s in result["summary"].items():
        scheme, mode, k = key.split("/")
        base = result["random_recall"][f"{scheme}/random/{k}"]
        lines.append(f"| {scheme} | {'있음' if mode == 'rerank' else '없음'} | {k[1:]} | {s['recall']:.4f} | {s['mrr']:.4f} "
                     f"| {s['stale_rate']:.4f} | {base:.4f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="검색 채점과 설정 비교 (dev)")
    parser.add_argument("--out")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--org", default="org-a")
    parser.add_argument("--no-rerank", action="store_true", help="리랭크 없는 설정만 채점한다")
    parser.add_argument("--allow-dirty", action="store_true", help="커밋하지 않은 변경이 있어도 돌리고 git_dirty를 남긴다")
    args = parser.parse_args(argv)
    from ontology.seed import load_seed

    seed, plan = load_seed(args.org), load_plan()
    scenario_paths = sorted(Path("eval/scenarios/dev").glob("*.yaml"))
    scenarios = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in scenario_paths]
    qs = queries(scenarios, plan)
    # 벡터 검색은 k마다 다시 하고, 리랭크 검색은 한 번 하고 자른다 (evaluate와 같은 계산)
    vector_calls = len(qs) * len(SCHEMES) * len(TOP_KS)
    rerank_calls = 0 if args.no_rerank else len(qs) * len(SCHEMES)
    print(f"사건 {len(qs)}개, 검색 {vector_calls + rerank_calls}번(리랭크 {rerank_calls}번)", file=sys.stderr)
    if args.dry_run:
        return 0
    if not args.out:
        parser.error("--out이 필요하다")
    from datetime import datetime, timedelta, timezone

    from eval.runner.runner import RunnerError, check_out_dir, git_state

    # 진단 실행기와 같은 규칙: 결과는 eval/results 아래에 두고, 어느 커밋의 코드로 낸 숫자인지 남긴다 (M5 점검)
    out = Path(args.out)
    try:
        check_out_dir(out)
        # 읽은 시나리오 파일이 커밋과 같은지도 본다. 무시 규칙에 걸린 파일은 git status에 나오지 않는다 (PR #36 리뷰)
        git = git_state(out, inputs=scenario_paths)
    except RunnerError as exc:
        print(f"채점할 수 없다: {exc}", file=sys.stderr)
        return 1
    if (out / "RETRIEVAL.yaml").exists():
        print(f"채점할 수 없다: {out}에 이미 결과가 있다. 새 폴더를 준다", file=sys.stderr)
        return 1
    if git.dirty and not args.allow_dirty:
        print("채점할 수 없다: 커밋하지 않은 변경이 있다 (--allow-dirty로 돌리면 git_dirty를 남긴다)\n  "
              + "\n  ".join(git.changed), file=sys.stderr)
        return 1

    from common.retrieval.index import check_index
    from common.retrieval.voyage import EMBED_MODEL, RERANK_MODEL, Voyage
    from ontology.db import get_driver

    chunks = all_chunks(seed, plan)
    chunk_docs = {s: [c.doc_id for c in chunks if c.scheme == s] for s in SCHEMES}
    voyage = Voyage()
    with get_driver() as driver:
        meta = check_index(driver, chunks)
        result = evaluate(driver, voyage, qs, chunk_docs, (False,) if args.no_rerank else (False, True))
    now = datetime.now(timezone(timedelta(hours=9)))
    result = {"run": {"date": now.date().isoformat(), "started_at": now.isoformat(), "split": "dev",
                      "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL, "corpus_hash": meta["corpus_hash"],
                      "embed_tokens": voyage.usage.embed_tokens, "rerank_tokens": voyage.usage.rerank_tokens,
                      "rate_waits": voyage.rate_waits, "rerank": not args.no_rerank,
                      "git_commit": git.commit, "git_dirty": git.dirty}} | result
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "RETRIEVAL.yaml", yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    write_text_atomic(out / "RETRIEVAL.md", markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
