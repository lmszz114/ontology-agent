"""추출 실행 명령 (설계 5단계 5장, 4단계 5-2, M4 PR 4).

실행: python -m agents.extraction.run --out eval/results/<폴더> [--reps 3] [--only PM-01,CH-04] [--resume] [--allow-dirty]

dev 문서만 돈다. test 문서는 M8 최종 실행에서 처음 추출한다 (4단계 4-2). 문서는 먼저 python -m eval.corpus.load --split dev로
개발용 Neo4j에 적재한다. 결과는 <폴더>/rN/<문서 id>.yaml에 문서마다 남기고, 후보는 실행 ID <폴더 이름>-rN으로
개발용 Neo4j에도 쓴다. 채점은 결과 파일로 한다 (python -m eval.grader.extraction <폴더>).
"""

import argparse
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from agents.extraction.pipeline import ExtractionError, run_document
from agents.extraction.prompts import PROMPT_VERSION
from agents.extraction.rules import Knowledge
from common.files import write_text_atomic
from common.models import LLM, load_model_config, make_anthropic_client
from common.packages import PackageError, load_package
from common.tracing import Tracer
from common.writers.extraction import ID_PATTERN, list_candidates, record_candidates
from eval.corpus.check import PLAN_PATH, load_plan
from eval.corpus.generate import doc_path
from eval.runner.__main__ import check_loaded_seed
from eval.runner.runner import RESULTS_DIR, ROOT, RunnerError, check_out_dir, git_state
from ontology.db import get_driver
from ontology.seed import load_seed, seed_hash

KST = timezone(timedelta(hours=9))
# 이어서 돌릴 때 앞 실행과 같아야 하는 값
SAME = ("org", "split", "prompt_version", "git_commit", "git_dirty", "seed_hash", "plan_hash", "models", "package")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dev 문서에서 지식 후보 추출 (Haiku 5.5)")
    p.add_argument("--out", required=True, help="eval/results 아래 새 폴더")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--only", help="쉼표로 나눈 문서 id")
    p.add_argument("--resume", action="store_true", help="결과 파일이 있는 문서는 건너뛴다")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--org", default="org-a")
    return p.parse_args(argv)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def documents(only: str | None) -> list[dict]:
    dev = [d for d in load_plan() if d["split"] == "dev"]
    if not only:
        return dev
    wanted = set(only.split(","))
    known = {d["id"] for d in load_plan()}
    if wanted - known:
        raise RunnerError(f"설계표에 없는 문서 id다: {sorted(wanted - known)}")
    test = sorted(i for i in wanted if next(d for d in load_plan() if d["id"] == i)["split"] != "dev")
    if test:
        raise RunnerError(f"test 문서는 M8 최종 실행에서만 추출한다 (4단계 4-2): {test}")
    return [d for d in dev if d["id"] in wanted]


def prepare(args) -> tuple[Path, dict, list[dict]]:
    """LLM을 부르기 전에 할 수 있는 확인을 모두 한다."""
    out = Path(args.out)
    check_out_dir(out)
    # 실행 ID는 폴더 이름에서 만든다. 결과 폴더를 eval/results 바로 아래로만 받아야 다른 결과가 같은 실행 ID를
    # 쓰지 않는다 (PR #30 리뷰)
    if out.resolve().parent != (ROOT / RESULTS_DIR).resolve():
        raise RunnerError(f"추출 결과 폴더는 {RESULTS_DIR.as_posix()} 바로 아래 폴더여야 한다: {out}")
    if not ID_PATTERN.fullmatch(out.name):
        raise RunnerError(f"폴더 이름은 실행 ID로 쓰여 영문, 숫자, '_', '.', '-'만 쓴다: {out.name}")
    if args.reps < 1:
        raise RunnerError("--reps는 1 이상이다")
    if args.resume and args.allow_dirty:
        raise RunnerError("--resume과 --allow-dirty는 함께 쓸 수 없다. 커밋하고 이어 돌린다")
    docs = documents(args.only)
    git = git_state(out, inputs=[PLAN_PATH, *(doc_path(d) for d in docs)])
    if git.dirty and not args.allow_dirty:
        raise RunnerError("커밋하지 않은 변경이 있다. 커밋하고 돌리거나 --allow-dirty를 준다:\n  " + "\n  ".join(git.changed))
    # 배포 패키지: 추출 프롬프트 버전이 패키지에 적힌 것과 같아야 한다 (5단계 3장, M6)
    try:
        package = load_package(args.org, uses={"extraction": PROMPT_VERSION})
    except (PackageError, ValueError) as exc:
        raise RunnerError(f"배포 패키지를 쓸 수 없다: {exc}") from exc
    config = package.models
    meta = {"org": args.org, "split": "dev", "prompt_version": PROMPT_VERSION, "git_commit": git.commit,
            "git_dirty": git.dirty, "seed_hash": seed_hash(package.seed), "plan_hash": _hash(PLAN_PATH),
            "models": {t: config.model_for(t) for t in ("extraction", "normalization")}, "package": package.record()}
    run_file = out / "run.yaml"
    if run_file.exists():
        before = yaml.safe_load(run_file.read_text(encoding="utf-8"))
        if not args.resume:
            raise RunnerError(f"이미 결과가 있는 폴더다. 이어 돌리려면 --resume: {out}")
        diff = [key for key in SAME if before.get(key) != meta[key]]
        if before.get("documents") and before["documents"] != [d["id"] for d in docs]:
            diff.append("documents")  # 이어 돌릴 때 문서 목록이 다르면 채점기가 완전한 반복을 가를 수 없다
        if diff:
            raise RunnerError(f"앞 실행과 조건이 다르다: {diff}")
    return out, meta, docs


def check_documents(driver, docs: list[dict]) -> None:
    """그래프의 문서가 원문 파일과 같은지 본다. 다르면 근거 문장 검사가 다른 원문으로 된다."""
    with driver.session() as s:
        rows = {r["id"]: r["text"] for r in s.run("MATCH (d:Document) RETURN d.id AS id, d.text AS text")}
    bad = [d["id"] for d in docs if rows.get(d["id"]) != doc_path(d).read_text(encoding="utf-8")]
    if bad:
        raise RunnerError(f"개발용 그래프의 문서가 원문과 다르거나 없다. python -m eval.corpus.load --split dev --reset: {bad}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        out, meta, docs = prepare(args)
        seed = load_seed(args.org)
        k = Knowledge.from_seed(seed)
        with get_driver() as driver:
            check_loaded_seed(driver, args.org, meta["seed_hash"])
            check_documents(driver, docs)
            tracer = Tracer.from_env()
            llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
            out.mkdir(parents=True, exist_ok=True)
            run_file = out / "run.yaml"
            if not run_file.exists():
                record = meta | {"started_at": datetime.now(KST).isoformat(timespec="seconds"), "reps": args.reps,
                                 "documents": [d["id"] for d in docs],
                                 "prices": llm.config.price_record(sorted(set(meta["models"].values())))}
                write_text_atomic(run_file, yaml.safe_dump(record, allow_unicode=True, sort_keys=False))
            else:
                # 이어 돌리며 반복을 늘리면 기록한 반복 수도 늘린다. 채점기는 rN 폴더를 센다
                record = yaml.safe_load(run_file.read_text(encoding="utf-8"))
                if args.reps > record.get("reps", 0):
                    record["reps"] = args.reps
                    write_text_atomic(run_file, yaml.safe_dump(record, allow_unicode=True, sort_keys=False))
            total = 0.0
            try:
                for rep in range(1, args.reps + 1):
                    run_id = f"{out.name}-r{rep}"
                    rep_dir = out / f"r{rep}"
                    rep_dir.mkdir(exist_ok=True)
                    for doc in docs:
                        path = rep_dir / f"{doc['id']}.yaml"
                        if path.exists():
                            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
                        else:
                            text = doc_path(doc).read_text(encoding="utf-8")
                            with tracer.run("extract_document", tags=["mode:extraction"],
                                            metadata={"doc": doc["id"], "run_id": run_id,
                                                      "prompt_version": PROMPT_VERSION}):
                                result = run_document(llm, k, doc["id"], text)
                            saved = {"document_id": doc["id"], "run_id": run_id, "candidates": result.candidates,
                                     "dropped": result.dropped, "stats": result.stats, "calls": result.calls,
                                     "cost_usd": result.cost}
                            # 비용을 낸 결과를 먼저 파일로 남기고 그래프에 쓴다
                            write_text_atomic(path, yaml.safe_dump(saved, allow_unicode=True, sort_keys=False))
                            total += result.cost
                            print(f"r{rep} {doc['id']}: 후보 {len(result.candidates)}, 버림 {len(result.dropped)}, "
                                  f"${result.cost:.5f}")
                        # 파일은 있는데 그래프에 없으면(앞 실행이 그 사이에 멈춤) 쓴다. 한 문서의 후보는 한 번에 쓰인다
                        if saved["candidates"] and not list_candidates(driver, run_id=run_id, document_id=doc["id"]):
                            record_candidates(driver, run_id=run_id, document_id=doc["id"],
                                              candidates=saved["candidates"])
            finally:
                tracer.flush()
            print(f"문서 {len(docs)}개 x {args.reps}회, 이번 비용 ${total:.5f}")
        return 0
    except (RunnerError, ExtractionError) as exc:
        print(f"실행할 수 없다: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
