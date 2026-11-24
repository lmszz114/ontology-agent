"""평가 실행기 (설계 5단계 10장, 12장 M3 계획 3번).

구성 하나를 골라 dev 시나리오를 정한 횟수만큼 돌리고 결과 파일을 남긴다. 채점은 eval/grader가 한다.

결과 폴더: <out>/r1 ... rN/<시나리오 ID>.json. 채점기가 같은 구조를 읽는다.
실행마다 결과의 run에 커밋, 작업 트리 변경 여부, 쓴 모델의 단가를 남긴다 (4단계 10장, 한계 L8, L14).
"""

import hashlib
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
RESUME_SAME = ("config", "org", "split", "prompt_version", "git_commit", "git_dirty", "seed_hash")


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


def check_out_dir(out: Path, root: Path = ROOT) -> None:
    """결과 폴더는 eval/results 아래만 받는다 (M3 점검 2).

    변경 검사는 이번 결과 폴더를 뺀다. 저장소 밖이나 코드 폴더(예: --out .)를 주면 고친 코드가 변경으로
    잡히지 않고 깨끗한 실행으로 기록된다.
    """
    results = Path(os.path.abspath(root / RESULTS_DIR))
    given = Path(os.path.abspath(out))
    if given == results or not given.is_relative_to(results):
        raise RunnerError(f"결과 폴더는 {RESULTS_DIR.as_posix()} 아래 새 폴더여야 한다: {out}")


def file_hash(path: Path) -> str:
    """시나리오 파일 바이트의 해시. 채점할 때 실행 때와 같은 라벨인지 본다."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state(out: Path, root: Path = ROOT, inputs: list[Path] | tuple[Path, ...] = ()) -> GitState:
    """커밋과 커밋하지 않은 변경. 결과 폴더(이번 --out과 eval/results 전체) 안의 파일은 변경으로 보지 않는다.

    inputs(실행에 넣는 시나리오 파일)는 git status와 따로, 넘긴 경로의 내용이 커밋(HEAD)의 같은 경로와 같은지
    직접 비교한다. 결과의 커밋으로 재현할 수 있는 입력이어야 하기 때문이다 (PR #20 리뷰). 저장소 밖이거나,
    심볼릭 링크를 거치거나, HEAD에 없거나, 내용이 다르면 변경이다. 추적하지 않은 파일, 무시 규칙에 걸린 파일,
    결과 폴더 안의 파일이 모두 여기에 걸린다.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        # -z: 경로를 따옴표나 이스케이프 없이 NUL로 구분해 준다 (한글 경로도 그대로 나온다)
        status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"Git 상태를 읽을 수 없다: {exc}") from exc
    # 저장소 파일의 변경은 심볼릭 링크를 따라가지 않고 경로 자체로 비교한다. 결과 폴더 밖의 파일을 결과 폴더 안을
    # 가리키는 링크로 바꾼 변경도 놓치지 않게 한다 (resolve는 링크 대상을 돌려준다)
    def absolute(path: Path) -> Path:
        return Path(os.path.abspath(path))

    ignored = (absolute(out), absolute(root / RESULTS_DIR))
    changed = []
    for paths in _status_entries(status):
        # 이동은 원래 경로와 새 경로를 모두 본다. 하나라도 결과 폴더 밖이면 변경이다
        outside = [p for p in paths if not any(absolute(root / p).is_relative_to(d) for d in ignored)]
        if outside:
            changed.append(" <- ".join(paths) if len(paths) > 1 else paths[0])
    changed += [p for p in (_input_change(path, root) for path in inputs) if p and p not in changed]
    return GitState(commit=commit, dirty=bool(changed), changed=changed)


def _input_change(path: Path, root: Path) -> str | None:
    """입력 파일이 커밋(HEAD)의 같은 경로 내용과 같으면 None, 다르면 변경 설명.

    비교 기준은 실행에 넘긴 경로다. 링크를 거치는 입력은 HEAD에서 같은 경로로 읽은 내용과 맞는지 확인하기
    어렵고 경우가 많아 구멍이 생기기 쉽다. 저장소에 링크로 된 시나리오가 없으므로 경로의 어느 단계에서든 링크를
    거친 입력은 변경으로 본다 (PR #20 리뷰: 커밋된 A.yaml을 다른 내용의 B.yaml을 가리키는 링크로 바꾸는 경우,
    alias/../A.yaml처럼 링크 뒤에 ..가 오는 경우).
    """
    root_abs = Path(os.path.abspath(root))
    given = Path(os.path.abspath(path))
    if not given.is_relative_to(root_abs):
        return f"{path} (저장소 밖의 입력)"
    rel = given.relative_to(root_abs)
    if _through_link(Path(path)):
        return f"{rel.as_posix()} (심볼릭 링크를 거친 입력)"
    shown = subprocess.run(["git", "show", f"HEAD:{rel.as_posix()}"], cwd=root, capture_output=True)
    if shown.returncode != 0:
        return f"{rel.as_posix()} (커밋에 없는 입력)"
    if shown.stdout != given.read_bytes():
        return rel.as_posix()
    return None


def _through_link(path: Path) -> bool:
    """넘겨받은 경로를 ..를 지우지 않고 앞에서부터 한 단계씩 따라가며 심볼릭 링크가 있는지 본다.

    abspath나 resolve 결과의 비교로는 못 잡는 경우가 있다. abspath는 링크를 풀기 전에 ..를 지워 다른 파일을
    가리킬 수 있고, alias가 같은 폴더의 real을 가리키면 alias/../A.yaml은 링크를 거쳐도 최종 경로가 같다
    (PR #20 리뷰). 상대 경로의 시작점인 os.getcwd()는 링크가 풀린 실제 경로라서 오탐이 생기지 않는다.
    """
    current = Path(path.anchor) if path.is_absolute() else Path(os.getcwd())
    for part in path.parts[1 if path.is_absolute() else 0:]:
        current = current / part
        if part not in (".", "..") and current.is_symlink():
            return True
    return False


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
                  prices: Callable[[list[str]], dict], seed_hash: str,
                  log: Callable[[str], None] = print) -> list[Path]:
    """목록을 순서대로 돌린다. 실패하면 그 자리에서 멈춘다. 이미 쓴 결과는 남는다 (--resume으로 이어서 돌린다).

    자동으로 다시 시도하지 않는다. 다시 시도하면 실패가 가려진다. 일시적인 API 오류는 SDK가 재시도한다.
    prices는 비용 계산에 쓴 모델 설정의 단가 기록 함수다 (ModelConfig.price_record).
    seed_hash는 실행에 쓴 시드의 해시다. 시나리오 파일 해시와 함께 남겨 채점기가 같은 기준으로 채점하는지 본다.
    """
    written = []
    for rep, path in todo:
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        result = await run_one(scenario, path)
        run = result["run"]
        run |= {"git_commit": git.commit, "git_dirty": git.dirty, "prices": prices(run["models"]),
                "seed_hash": seed_hash, "scenario_hash": file_hash(path)}
        folder = out / f"r{rep}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{scenario['id']}.json"
        write_text_atomic(target, json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
        written.append(target)
        log(f"r{rep} {scenario['id']}: {run['llm_calls']} calls, ${run['cost_usd']:.5f}")
    return written
