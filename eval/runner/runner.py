"""평가 실행기 (설계 5단계 10장, 12장 M3 계획 3번).

구성 하나를 골라 dev 시나리오를 정한 횟수만큼 돌리고 결과 파일을 남긴다. 채점은 eval/grader가 한다.

결과 폴더: <out>/r1 ... rN/<시나리오 ID>.json. 채점기가 같은 구조를 읽는다.
실행마다 결과의 run에 커밋, 작업 트리 변경 여부, 쓴 모델의 단가를 남긴다 (4단계 10장, 한계 L8, L14).
"""

import json
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from common.files import write_text_atomic
from common.models.config import price_record

ROOT = Path(__file__).resolve().parents[2]
# 이어서 돌릴 때 앞 결과와 같아야 하는 값. 다르면 같은 조건의 반복이 아니다
RESUME_SAME = ("config", "org", "split", "prompt_version", "git_commit", "git_dirty")


class RunnerError(Exception):
    """실행을 시작할 수 없다. 비용을 쓰기 전에 멈춘다."""


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool
    changed: list[str]


def git_state(out: Path, root: Path = ROOT) -> GitState:
    """커밋과 커밋하지 않은 변경. 결과 폴더 안의 파일은 변경으로 보지 않는다 (실행기가 쓰는 파일이다)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"Git 상태를 읽을 수 없다: {exc}") from exc
    out = out.resolve()
    changed = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1].strip('"')
        if not (root / path).resolve().is_relative_to(out):
            changed.append(path)
    return GitState(commit=commit, dirty=bool(changed), changed=changed)


def check_dev_only(paths: list[Path]) -> None:
    """dev 시나리오만 받는다. test 세트는 최종 실행(M8)에만 쓴다 (4단계 4-2). 파일을 직접 지정해도 막는다.

    키 확인과 LLM 호출 전에 부른다.
    """
    for path in paths:
        split = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("split")
        if split != "dev":
            raise RunnerError(f"{path}: dev 시나리오만 돌릴 수 있다 (split={split})")


def plan(paths: list[Path], *, out: Path, reps: int, resume: bool, expected: dict) -> list[tuple[int, Path]]:
    """돌릴 (반복 번호, 시나리오 파일) 목록. 이미 결과가 있는 것은 이어서 돌릴 때만 건너뛴다.

    expected는 이번 실행의 조건(RESUME_SAME 중 실행 전에 아는 값)이다. 앞 결과와 다르면 멈춘다.
    """
    if reps < 1:
        raise RunnerError(f"반복 횟수는 1 이상이어야 한다: {reps}")
    existing = sorted(out.glob("r*/*.json")) if out.exists() else []
    if existing and not resume:
        raise RunnerError(f"{out}에 결과가 이미 있다. 이어서 돌리려면 --resume을 준다")
    for path in existing:
        run = json.loads(path.read_text(encoding="utf-8"))["run"]
        diff = {k: (run.get(k), v) for k, v in expected.items() if run.get(k) != v}
        if diff:
            raise RunnerError(f"{path}: 이번 실행과 조건이 다르다 (앞 결과, 이번) {diff}")
    todo = []
    for rep in range(1, reps + 1):
        for path in paths:
            scenario_id = yaml.safe_load(path.read_text(encoding="utf-8"))["id"]
            if not (out / f"r{rep}" / f"{scenario_id}.json").exists():
                todo.append((rep, path))
    return todo


RunOne = Callable[[dict, Path], Awaitable[dict]]


async def run_all(todo: list[tuple[int, Path]], *, out: Path, git: GitState, run_one: RunOne,
                  log: Callable[[str], None] = print) -> list[Path]:
    """목록을 순서대로 돌린다. 실패하면 그 자리에서 멈춘다. 이미 쓴 결과는 남는다 (--resume으로 이어서 돌린다).

    자동으로 다시 시도하지 않는다. 다시 시도하면 실패가 가려진다. 일시적인 API 오류는 SDK가 재시도한다.
    """
    written = []
    for rep, path in todo:
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        result = await run_one(scenario, path)
        run = result["run"]
        run |= {"git_commit": git.commit, "git_dirty": git.dirty, "prices": price_record(run["models"])}
        folder = out / f"r{rep}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{scenario['id']}.json"
        write_text_atomic(target, json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
        written.append(target)
        log(f"r{rep} {scenario['id']}: {run['llm_calls']} calls, ${run['cost_usd']:.5f}")
    return written
