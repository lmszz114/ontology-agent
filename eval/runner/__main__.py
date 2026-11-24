"""평가 실행기.

실행: python -m eval.runner --config C --split dev --reps 3 --out eval/results/2026-11-24-m3-dev-C
      python -m eval.runner --config C --scenario eval/scenarios/dev/SC-S1-01.yaml --out 결과폴더

- --config: 비교 구성 (등록된 구성만, agents/diagnosis/run.py의 KNOWLEDGE)
- --reps: 반복 횟수 (기본 1). 결과는 <out>/r1 ... rN에 쓴다
- --resume: 결과 폴더에 이미 있는 시나리오를 건너뛴다. 커밋, 구성, 프롬프트가 같아야 한다
- --allow-dirty: 커밋하지 않은 변경이 있어도 돌린다. 결과에 git_dirty: true가 남는다
test 세트는 돌릴 수 없다 (4단계 4-2). 실행 전에 비용 승인 지점(CLAUDE.md 5장)을 확인한다.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from agents.diagnosis.prompts import PROMPT_VERSION
from agents.diagnosis.run import KNOWLEDGE, run_scenario
from common.models import LLM, load_model_config, make_anthropic_client
from common.tracing import Tracer
from eval.runner.runner import ROOT, RunnerError, check_dev_only, git_state, plan, run_all
from eval.validate import load_scenarios
from ontology.db import get_driver


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
    args = parser.parse_args(argv)
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
    git = git_state(out, inputs=paths)
    if git.dirty and not args.allow_dirty:
        # 커밋하지 않은 변경으로 돌리면 결과의 커밋으로 코드를 재현할 수 없다
        raise RunnerError("커밋하지 않은 변경이 있다. 커밋하고 돌리거나 --allow-dirty를 준다:\n  "
                          + "\n  ".join(git.changed))
    expected = {"config": args.config, "org": args.org, "split": "dev", "prompt_version": PROMPT_VERSION,
                "git_commit": git.commit, "git_dirty": git.dirty}
    todo = plan(paths, out=out, reps=args.reps, resume=args.resume, expected=expected)
    return out, git, todo


async def main_async(args: argparse.Namespace) -> int:
    out, git, todo = prepare(args)
    print(f"구성 {args.config}, 반복 {args.reps}, 실행 {len(todo)}건, 커밋 {git.commit[:7]}"
          + (" (커밋하지 않은 변경 있음)" if git.dirty else ""))
    if not todo:
        return 0
    tracer = Tracer.from_env()
    llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
    with get_driver() as driver:
        try:
            async def run_one(scenario: dict, path: Path) -> dict:
                return await run_scenario(scenario, path, config=args.config, org=args.org, tracer=tracer, llm=llm,
                                          driver=driver)
            await run_all(todo, out=out, git=git, run_one=run_one, prices=llm.config.price_record)
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
