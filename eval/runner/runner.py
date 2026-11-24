"""평가 실행기 (설계 5단계 10장, 12장 M3 계획 3번).

구성 하나를 골라 dev 시나리오를 정한 횟수만큼 돌리고 결과 파일을 남긴다. 채점은 eval/grader가 한다.

결과 폴더: <out>/r1 ... rN/<시나리오 ID>.json. 채점기가 같은 구조를 읽는다.
실행마다 결과의 run에 커밋, 작업 트리 변경 여부, 쓴 모델의 단가를 남긴다 (4단계 10장, 한계 L8, L14).
"""

import json
import os
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from common.files import write_text_atomic

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


# 실행 결과를 두는 곳. 결과 파일은 실행에 영향을 주지 않아 커밋하지 않은 변경으로 보지 않는다.
# 앞 실행의 결과를 커밋하기 전에 다음 구성을 돌릴 수 있다 (비용 승인 지점 1, 2026-11-24 사용자 결정)
RESULTS_DIR = Path("eval") / "results"


def git_state(out: Path, root: Path = ROOT) -> GitState:
    """커밋과 커밋하지 않은 변경. 결과 폴더(이번 --out과 eval/results 전체) 안의 파일은 변경으로 보지 않는다."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        # -z: 경로를 따옴표나 이스케이프 없이 NUL로 구분해 준다 (한글 경로도 그대로 나온다)
        status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"Git 상태를 읽을 수 없다: {exc}") from exc
    # 심볼릭 링크를 따라가지 않고 경로 자체로 비교한다. 결과 폴더 밖의 파일을 결과 폴더 안을 가리키는
    # 링크로 바꾼 변경도 놓치지 않게 한다 (resolve는 링크 대상을 돌려준다)
    def absolute(path: Path) -> Path:
        return Path(os.path.abspath(path))

    ignored = (absolute(out), absolute(root / RESULTS_DIR))
    changed = []
    for paths in _status_entries(status):
        # 이동은 원래 경로와 새 경로를 모두 본다. 하나라도 결과 폴더 밖이면 변경이다
        outside = [p for p in paths if not any(absolute(root / p).is_relative_to(d) for d in ignored)]
        if outside:
            changed.append(" <- ".join(paths) if len(paths) > 1 else paths[0])
    return GitState(commit=commit, dirty=bool(changed), changed=changed)


def _status_entries(status: str) -> list[list[str]]:
    """git status --porcelain=v1 -z 출력을 항목별 경로 목록으로 나눈다.

    항목은 "XY 경로"이고 이동(R)과 복사(C)는 다음 NUL 칸에 원래 경로가 하나 더 온다.
    """
    fields = status.split("\0")
    entries, i = [], 0
    while i < len(fields) and fields[i]:
        code, path = fields[i][:2], fields[i][3:]
        if "R" in code or "C" in code:
            entries.append([path, fields[i + 1]])
            i += 2
        else:
            entries.append([path])
            i += 1
    return entries


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
    # 채점 파일(grades.json)은 실행 결과가 아니다. 채점기의 load_results와 같이 뺀다
    existing = sorted(p for p in out.glob("r*/*.json") if p.name != "grades.json") if out.exists() else []
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
                  prices: Callable[[list[str]], dict], log: Callable[[str], None] = print) -> list[Path]:
    """목록을 순서대로 돌린다. 실패하면 그 자리에서 멈춘다. 이미 쓴 결과는 남는다 (--resume으로 이어서 돌린다).

    자동으로 다시 시도하지 않는다. 다시 시도하면 실패가 가려진다. 일시적인 API 오류는 SDK가 재시도한다.
    prices는 비용 계산에 쓴 모델 설정의 단가 기록 함수다 (ModelConfig.price_record).
    """
    written = []
    for rep, path in todo:
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        result = await run_one(scenario, path)
        run = result["run"]
        run |= {"git_commit": git.commit, "git_dirty": git.dirty, "prices": prices(run["models"])}
        folder = out / f"r{rep}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{scenario['id']}.json"
        write_text_atomic(target, json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
        written.append(target)
        log(f"r{rep} {scenario['id']}: {run['llm_calls']} calls, ${run['cost_usd']:.5f}")
    return written
