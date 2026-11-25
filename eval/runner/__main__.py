"""평가 실행기.

실행: python -m eval.runner --config C --split dev --reps 3 --out eval/results/2026-11-24-m3-dev-C
      python -m eval.runner --config C --scenario eval/scenarios/dev/SC-S1-01.yaml --out 결과폴더

- --config: 비교 구성 (등록된 구성만, agents/diagnosis/run.py의 KNOWLEDGE)
- --reps: 반복 횟수 (기본 1). 결과는 <out>/r1 ... rN에 쓴다
- --out: eval/results 아래 폴더만 받는다
- --resume: 결과 폴더에 이미 있는 시나리오를 건너뛴다. 커밋, 구성, 프롬프트, 시드가 같아야 한다
- --allow-dirty: 커밋하지 않은 변경이 있어도 돌린다. 결과에 git_dirty: true가 남는다. --resume과 함께 쓸 수 없다
- 개발용 그래프에 적재한 시드가 지금 시드 파일과 다르면 돌리지 않는다 (python -m ontology.load로 다시 적재)
- --cext-from: 구성 C-ext의 추출 결과 폴더. 반복 N은 추출 반복 rN의 후보로 만든 그래프로 진단한다 (M4 PR 6 결정 1).
  그래프는 반복을 시작할 때 C-ext DB(NEO4J_EXT_URI, 7689)에 다시 적재한다. 개발용 DB는 건드리지 않는다
test 세트는 돌릴 수 없다 (4단계 4-2). 실행 전에 비용 승인 지점(CLAUDE.md 5장)을 확인한다.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agents.diagnosis.prompts import PROMPT_VERSION
from agents.diagnosis.run import EXT_CONFIGS, KNOWLEDGE, run_scenario
from common.models import LLM, load_model_config, make_anthropic_client
from common.tracing import Tracer
from eval.runner.runner import (
    ROOT,
    RunnerError,
    check_dev_only,
    check_out_dir,
    git_state,
    plan,
    run_all,
)
from eval.validate import load_scenarios
from ontology import cext
from ontology.db import get_driver
from ontology.load import loaded_seed
from ontology.seed import load_seed, seed_hash


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="진단 에이전트 평가 실행기")
    parser.add_argument("--config", required=True, choices=sorted(KNOWLEDGE), help="비교 구성")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--scenario", nargs="+", help="시나리오 파일")
    target.add_argument("--split", choices=["dev", "test"], help="이 세트의 시나리오 전체")
    parser.add_argument("--reps", type=int, default=1, help="반복 횟수 (기본 1)")
    parser.add_argument("--out", required=True, help="결과 폴더")
    parser.add_argument("--org", default="org-a")
    parser.add_argument("--resume", action="store_true", help="결과가 있는 시나리오는 건너뛴다")
    parser.add_argument("--allow-dirty", action="store_true", help="커밋하지 않은 변경이 있어도 돌린다")
    parser.add_argument("--cext-from", help="구성 C-ext의 추출 결과 폴더 (eval/results 아래)")
    args = parser.parse_args(argv)
    if (args.config in EXT_CONFIGS) != bool(args.cext_from):
        parser.error("--cext-from은 구성 C-ext에만, C-ext에는 반드시 준다")
    if args.split == "test":
        # test 세트는 최종 실행(M8)에만 쓴다 (4단계 4-2). 실수로 돌리지 않게 막는다
        parser.error("test 세트는 최종 실행에서만 쓴다")
    return args


def prepare(args: argparse.Namespace):
    """LLM을 부르기 전에 할 수 있는 확인을 모두 한다. 돌릴 목록과 Git 상태를 돌려준다."""
    if args.scenario:
        paths = [Path(p) for p in args.scenario]
    else:
        paths = sorted((ROOT / "eval" / "scenarios" / args.split).glob("*.yaml"))
        load_scenarios(args.split)  # 형식이 틀린 파일이 있으면 여기서 멈춘다
    check_dev_only(paths)
    out = Path(args.out)
    check_out_dir(out)
    if args.resume and args.allow_dirty:
        # 같은 커밋 위의 다른 미커밋 코드로 이어 돌리면 한 폴더에 섞여도 조건이 같아 보인다 (M3 점검 2)
        raise RunnerError("--resume과 --allow-dirty는 함께 쓸 수 없다. 커밋하고 이어 돌린다")
    git = git_state(out, inputs=paths)
    if git.dirty and not args.allow_dirty:
        # 커밋하지 않은 변경으로 돌리면 결과의 커밋으로 코드를 재현할 수 없다
        raise RunnerError("커밋하지 않은 변경이 있다. 커밋하고 돌리거나 --allow-dirty를 준다:\n  "
                          + "\n  ".join(git.changed))
    seed = seed_hash(load_seed(args.org))
    expected = {"config": args.config, "org": args.org, "split": "dev", "prompt_version": PROMPT_VERSION,
                "git_commit": git.commit, "git_dirty": git.dirty, "seed_hash": seed}
    if args.cext_from:
        check_cext(args, out)
        expected["cext_from"] = Path(args.cext_from).as_posix()
    todo = plan(paths, out=out, reps=args.reps, resume=args.resume, expected=expected)
    return out, git, todo, seed


def check_cext(args: argparse.Namespace, out: Path | None = None) -> None:
    """C-ext 실행 전 확인 (LLM 호출 전).

    - C-ext DB가 개발용 DB와 다른 서버인지
    - 추출 반복이 진단 반복 수만큼 있고 모두 온전한지
    - 이어서 돌릴 때 결과 폴더의 반복 rN에 이미 있는 결과가 지금 추출 결과로 만든 그래프와 같은지. 다르면 한 반복에
      두 그래프의 결과가 섞인다 (PR #32 리뷰). 섞지 않고 멈춘다. 새 폴더에 다시 돌린다
    """
    try:
        cext.check_ext_uri()
    except cext.CextError as exc:
        raise RunnerError(str(exc)) from exc
    seed = load_seed(args.org)
    for rep in range(1, args.reps + 1):
        try:
            _, source = cext.load_candidates(Path(args.cext_from), f"r{rep}")
        except (cext.CextError, OSError) as exc:
            raise RunnerError(f"C-ext 추출 결과를 쓸 수 없다: {exc}") from exc
        if source["extraction_seed_hash"] != seed_hash(seed):
            raise RunnerError(f"{args.cext_from}: 추출 때의 시드와 지금 시드가 다르다")
        if out is None or not (out / f"r{rep}").is_dir():
            continue
        now = cext.cext_hash(seed, source)
        for path in sorted((out / f"r{rep}").glob("*.json")):
            if path.name == "grades.json":
                continue
            graph = (json.loads(path.read_text(encoding="utf-8"))["run"].get("knowledge_graph") or {})
            if graph.get("hash") != now:
                raise RunnerError(f"{path}: 앞 결과의 C-ext 그래프({str(graph.get('hash'))[:12]})와 지금 추출 결과로 만든 "
                                  f"그래프({now[:12]})가 다르다. 한 반복에 섞지 않는다. 새 결과 폴더에 다시 돌린다")


def check_loaded_seed(driver, org: str, expected_hash: str) -> None:
    """개발용 그래프의 시드가 지금 시드 파일과 같은지 본다 (M3 점검 2).

    구성 A는 시드 파일에서, 구성 C와 승인 게이트는 그래프에서 지식을 읽는다. 다시 적재하지 않으면 둘이 갈린다
    """
    loaded = loaded_seed(driver)
    if loaded is None or loaded["hash"] != expected_hash or loaded["org"] != org:
        raise RunnerError(f"그래프에 적재한 시드가 지금 시드 파일({org})과 다르다. python -m ontology.load {org}로 다시 "
                          f"적재한다 (적재 기록: {loaded})")


async def main_async(args: argparse.Namespace) -> int:
    out, git, todo, seed = prepare(args)
    print(f"구성 {args.config}, 반복 {args.reps}, 실행 {len(todo)}건, 커밋 {git.commit[:7]}"
          + (" (커밋하지 않은 변경 있음)" if git.dirty else ""))
    if not todo:
        return 0
    ext = args.config in EXT_CONFIGS
    uri = cext.ext_uri() if ext else None
    with get_driver(uri) as driver:
        if not ext:
            check_loaded_seed(driver, args.org, seed)
        tracer = Tracer.from_env()
        llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
        try:
            # 반복마다 그 반복의 그래프로 돈다. C와 A는 반복 구분 없이 같은 그래프다
            for rep in sorted({r for r, _ in todo}):
                graph_info = None
                if ext:
                    graph_info = cext.load_cext(driver, load_seed(args.org), Path(args.cext_from), f"r{rep}")
                    print(f"r{rep} C-ext 그래프 적재: {graph_info['source']['run_id']} {graph_info['hash'][:12]} "
                          f"{graph_info['stats']['relationships']}")

                async def run_one(scenario: dict, path: Path, info=graph_info) -> dict:
                    result = await run_scenario(scenario, path, config=args.config, org=args.org, tracer=tracer,
                                                llm=llm, driver=driver, neo4j_uri=uri)
                    if info:
                        # 실행에 쓴 그래프: 추출 결과 반복, 그래프 해시, 빌더 버전, 채운 기본값 수 (채점기가 해시를 다시 본다)
                        result["run"] |= {"cext_from": Path(args.cext_from).as_posix(), "knowledge_graph": info}
                    return result
                await run_all([t for t in todo if t[0] == rep], out=out, git=git, run_one=run_one,
                              prices=llm.config.price_record, seed_hash=seed)
        finally:
            tracer.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except RunnerError as exc:
        print(f"실행할 수 없다: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
