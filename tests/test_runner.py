"""평가 실행기 단위 테스트. 가짜 run_scenario로 돌린다 (LLM, Neo4j 없음)."""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from common.models import load_model_config
from eval.grader.grade import load_results
from eval.runner import __main__ as cli
from eval.runner.runner import GitState, RunnerError, git_state, plan, run_all

DEV = Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "dev"
PATHS = [DEV / "SC-S1-01.yaml", DEV / "SC-S7-02.yaml"]
GIT = GitState(commit="abc1234def", dirty=False, changed=[])
EXPECTED = {"config": "C", "org": "org-a", "split": "dev", "prompt_version": "diagnosis-v6",
            "git_commit": GIT.commit, "git_dirty": False}


def fake_result(scenario: dict) -> dict:
    return {"run": {"scenario": scenario["id"], "split": "dev", "config": "C", "org": "org-a", "mode": "eval",
                    "started_at": "2026-11-24T13:00:00+09:00", "finished_at": "2026-11-24T13:00:10+09:00",
                    "seconds": 10.0, "models": ["claude-haiku-5-5"], "prompt_version": "diagnosis-v6",
                    "trace_id": "t", "llm_calls": 4, "usage": {"input": 1000, "output": 100}, "cost_usd": 0.002},
            "outputs": [], "unclassified": []}


def run(todo, out, *, fail_on: str | None = None):
    calls = []

    async def run_one(scenario, path):
        calls.append(scenario["id"])
        if scenario["id"] == fail_on:
            raise RuntimeError("연결 실패")
        return fake_result(scenario)
    asyncio.run(run_all(todo, out=out, git=GIT, run_one=run_one, prices=load_model_config("org-a").price_record,
                        log=lambda _: None))
    return calls


def test_reps_write_rep_folders_with_run_record(tmp_path):
    todo = plan(PATHS, out=tmp_path, reps=2, resume=False, expected=EXPECTED)
    assert [(r, p.stem) for r, p in todo] == [(1, "SC-S1-01"), (1, "SC-S7-02"), (2, "SC-S1-01"), (2, "SC-S7-02")]
    run(todo, tmp_path)
    data = json.loads((tmp_path / "r2" / "SC-S7-02.json").read_text(encoding="utf-8"))
    assert data["run"]["git_commit"] == GIT.commit and data["run"]["git_dirty"] is False
    prices = data["run"]["prices"]
    assert prices["checked"] and prices["source"].startswith("https://")
    assert set(prices["models"]) == {"claude-haiku-5-5"} and "input" in prices["models"]["claude-haiku-5-5"]
    # 채점기가 같은 폴더를 읽을 수 있다
    assert [name for name, _ in load_results(tmp_path)] == ["r1", "r2"]


def test_existing_results_need_resume(tmp_path):
    run(plan(PATHS, out=tmp_path, reps=1, resume=False, expected=EXPECTED), tmp_path)
    with pytest.raises(RunnerError, match="--resume"):
        plan(PATHS, out=tmp_path, reps=1, resume=False, expected=EXPECTED)


def test_failure_stops_and_resume_skips_done(tmp_path):
    todo = plan(PATHS, out=tmp_path, reps=2, resume=False, expected=EXPECTED)
    with pytest.raises(RuntimeError):
        run(todo, tmp_path, fail_on="SC-S7-02")
    # 실패한 곳에서 멈췄다. 앞 결과는 남고 뒤는 돌지 않았다
    assert [p.relative_to(tmp_path).as_posix() for p in sorted(tmp_path.glob("r*/*.json"))] == ["r1/SC-S1-01.json"]
    todo = plan(PATHS, out=tmp_path, reps=2, resume=True, expected=EXPECTED)
    assert [(r, p.stem) for r, p in todo] == [(1, "SC-S7-02"), (2, "SC-S1-01"), (2, "SC-S7-02")]


@pytest.mark.parametrize("key,value", [("git_commit", "other"), ("prompt_version", "diagnosis-v7"),
                                       ("config", "A"), ("git_dirty", True)])
def test_resume_refuses_different_conditions(tmp_path, key, value):
    run(plan(PATHS, out=tmp_path, reps=1, resume=False, expected=EXPECTED), tmp_path)
    with pytest.raises(RunnerError, match=key):
        plan(PATHS, out=tmp_path, reps=1, resume=True, expected=EXPECTED | {key: value})


def test_reps_must_be_positive(tmp_path):
    with pytest.raises(RunnerError, match="반복"):
        plan(PATHS, out=tmp_path, reps=0, resume=False, expected=EXPECTED)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    return repo


def test_grades_file_is_not_a_result(tmp_path):
    """반복 폴더 하나를 채점하면 그 안에 grades.json이 생긴다. 이어서 돌릴 때 결과로 읽지 않는다."""
    run(plan(PATHS[:1], out=tmp_path, reps=1, resume=False, expected=EXPECTED), tmp_path)
    (tmp_path / "r1" / "grades.json").write_text("{}", encoding="utf-8")
    assert [(r, p.stem) for r, p in plan(PATHS, out=tmp_path, reps=1, resume=True, expected=EXPECTED)] == \
        [(1, "SC-S7-02")]


def test_price_record_uses_snapshot_read_with_costs(tmp_path):
    """단가 파일이 실행 중에 바뀌어도 비용 계산에 쓴 단가를 기록한다."""
    prices = tmp_path / "prices.yaml"
    source = Path(__file__).resolve().parents[1] / "common" / "models" / "prices.yaml"
    prices.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config = load_model_config("org-a", prices_path=prices)
    prices.write_text(source.read_text(encoding="utf-8").replace("input: 0.10", "input: 9.99"), encoding="utf-8")
    record = config.price_record(["claude-haiku-5-5"])
    assert record["models"]["claude-haiku-5-5"]["input"] == 0.10
    assert record["models"]["claude-haiku-5-5"]["input"] == config.prices["claude-haiku-5-5"].base.input


def test_git_state_ignores_result_folder(tmp_path):
    repo = _repo(tmp_path)
    out = repo / "eval" / "results" / "x"
    out.mkdir(parents=True)
    (out / "r1.json").write_text("{}", encoding="utf-8")
    state = git_state(out, root=repo)
    assert not state.dirty and len(state.commit) == 40
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "new.py").write_text("", encoding="utf-8")
    state = git_state(out, root=repo)
    assert state.dirty and sorted(state.changed) == ["a.py", "new.py"]


def test_git_state_korean_result_folder_and_moves(tmp_path):
    """한글 경로는 Git이 이스케이프해 출력한다. 결과 폴더가 한글이어도 그 안의 파일은 변경이 아니다.
    추적 중인 파일을 결과 폴더로 옮기면 원래 경로가 바뀐 것이므로 변경이다."""
    repo = _repo(tmp_path)
    out = repo / "결과폴더"
    (out / "r1").mkdir(parents=True)
    (out / "r1" / "SC-S1-01.json").write_text("{}", encoding="utf-8")
    assert not git_state(out, root=repo).dirty
    subprocess.run(["git", "mv", "a.py", "결과폴더/a.py"], cwd=repo, check=True, capture_output=True)
    state = git_state(out, root=repo)
    assert state.dirty and state.changed == ["결과폴더/a.py <- a.py"]


def test_cli_refuses_dirty_tree_and_test_split(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", lambda out, inputs=(): GitState("abc", True, ["a.py"]))
    assert cli.main(["--config", "C", "--scenario", str(PATHS[0]), "--out", str(tmp_path)]) == 1
    assert "--allow-dirty" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["--config", "C", "--split", "test", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli.main(["--config", "Z", "--split", "dev", "--out", str(tmp_path)])


def test_cli_allow_dirty_reaches_plan(tmp_path, monkeypatch):
    """--allow-dirty면 실행 목록까지 만든다. 실제 실행은 하지 않는다."""
    monkeypatch.setattr(cli, "git_state", lambda out, inputs=(): GitState("abc", True, ["a.py"]))
    args = cli.parse_args(["--config", "C", "--scenario", str(PATHS[0]), "--out", str(tmp_path), "--allow-dirty",
                           "--reps", "2"])
    out, git, todo = cli.prepare(args)
    assert git.dirty and len(todo) == 2


def test_git_state_ignores_other_result_folders(tmp_path):
    """앞 구성의 결과를 커밋하기 전에 다음 구성을 돌릴 수 있다. eval/results 밖의 변경은 그대로 잡는다."""
    repo = _repo(tmp_path)
    earlier = repo / "eval" / "results" / "2026-11-24-x-C" / "r1"
    earlier.mkdir(parents=True)
    (earlier / "SC-S1-01.json").write_text("{}", encoding="utf-8")
    out = repo / "eval" / "results" / "2026-11-24-x-A"
    assert not git_state(out, root=repo).dirty
    scenario = repo / "eval" / "scenarios" / "dev" / "SC-X.yaml"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("id: SC-X\n", encoding="utf-8")
    state = git_state(out, root=repo)
    assert state.dirty and state.changed == ["eval/scenarios/dev/SC-X.yaml"]


def test_git_state_does_not_follow_symlinks_into_results(tmp_path):
    """결과 폴더 밖의 추적 파일을 결과 폴더 안을 가리키는 링크로 바꿔도 변경으로 잡는다 (PR #20 리뷰)."""
    repo = _repo(tmp_path)
    target = repo / "eval" / "results" / "2026-11-24-x-C" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 9\n", encoding="utf-8")
    (repo / "a.py").unlink()
    (repo / "a.py").symlink_to(target)
    state = git_state(repo / "eval" / "results" / "2026-11-24-x-A", root=repo)
    assert state.dirty and state.changed == ["a.py"]


def test_git_state_counts_uncommitted_inputs_inside_results(tmp_path):
    """결과 폴더 안에 커밋하지 않은 시나리오를 두고 실행에 넣으면 변경으로 잡는다 (PR #20 리뷰)."""
    repo = _repo(tmp_path)
    scenario = repo / "eval" / "results" / "scratch" / "SC-X.yaml"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("id: SC-X\n", encoding="utf-8")
    out = repo / "eval" / "results" / "2026-11-24-x-A"
    assert not git_state(out, root=repo).dirty  # 입력이 아니면 결과 폴더 안의 파일일 뿐이다
    state = git_state(out, root=repo, inputs=[scenario])
    assert state.dirty and state.changed == ["eval/results/scratch/SC-X.yaml (커밋에 없는 입력)"]
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("id: SC-Y\n", encoding="utf-8")
    state = git_state(out, root=repo, inputs=[outside])
    assert state.dirty and "저장소 밖" in state.changed[0]


def _commit_all(repo: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "x"],
                   cwd=repo, check=True, capture_output=True)


def test_git_state_input_checks_content_against_head(tmp_path):
    """입력은 내용이 HEAD와 같아야 깨끗하다. 링크로 저장소 밖을 가리키거나 무시 규칙에 걸려도 잡는다 (PR #20 리뷰)."""
    repo = _repo(tmp_path)
    scenario = repo / "eval" / "scenarios" / "dev" / "SC-X.yaml"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("id: SC-X\n", encoding="utf-8")
    _commit_all(repo)
    out = repo / "eval" / "results" / "x"
    assert not git_state(out, root=repo, inputs=[scenario]).dirty
    # 저장소 밖 파일을 가리키는 링크를 커밋해 두고 입력으로 쓰면, 링크가 바뀌지 않았어도 재현할 수 없다
    outside = tmp_path / "outside.yaml"
    outside.write_text("id: SC-O\n", encoding="utf-8")
    link = repo / "eval" / "scenarios" / "dev" / "SC-L.yaml"
    link.symlink_to(outside)
    _commit_all(repo)
    state = git_state(out, root=repo, inputs=[link])
    assert state.dirty and state.changed == ["eval/scenarios/dev/SC-L.yaml (심볼릭 링크를 거친 입력)"]
    # 무시 규칙에 걸린 커밋하지 않은 입력은 git status에 나오지 않는다. 내용 비교로 잡는다
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    _commit_all(repo)
    hidden = repo / "ignored" / "SC-H.yaml"
    hidden.parent.mkdir()
    hidden.write_text("id: SC-H\n", encoding="utf-8")
    assert not git_state(out, root=repo).dirty
    state = git_state(out, root=repo, inputs=[hidden])
    assert state.dirty and state.changed == ["ignored/SC-H.yaml (커밋에 없는 입력)"]


def test_git_state_input_replaced_by_link_to_other_committed_file(tmp_path):
    """커밋된 A.yaml을 다른 내용의 B.yaml을 가리키는 링크로 바꾸면 변경이다. B.yaml이 HEAD와 같아도 그렇다 (PR #20 리뷰)."""
    repo = _repo(tmp_path)
    folder = repo / "eval" / "results" / "inputs"
    folder.mkdir(parents=True)
    a, b = folder / "A.yaml", folder / "B.yaml"
    a.write_text("id: A\n", encoding="utf-8")
    b.write_text("id: B\n", encoding="utf-8")
    _commit_all(repo)
    out = repo / "eval" / "results" / "x"
    assert not git_state(out, root=repo, inputs=[a]).dirty
    a.unlink()
    a.symlink_to(b)
    state = git_state(out, root=repo, inputs=[a])
    assert state.dirty and state.changed == ["eval/results/inputs/A.yaml (심볼릭 링크를 거친 입력)"]


def test_git_state_dotdot_through_link_is_a_link_input(tmp_path):
    """alias가 링크인 alias/../A.yaml은 abspath로는 inputs/A.yaml이지만 실제로는 다른 파일을 읽는다 (PR #20 리뷰)."""
    repo = _repo(tmp_path)
    inputs = repo / "eval" / "results" / "inputs"
    (inputs / "real" / "deep").mkdir(parents=True)
    (inputs / "A.yaml").write_text("id: A\n", encoding="utf-8")
    (inputs / "real" / "A.yaml").write_text("id: REAL\n", encoding="utf-8")
    (inputs / "alias").symlink_to(inputs / "real" / "deep")
    _commit_all(repo)
    state = git_state(repo / "eval" / "results" / "x", root=repo, inputs=[inputs / "alias" / ".." / "A.yaml"])
    assert state.dirty and state.changed == ["eval/results/inputs/A.yaml (심볼릭 링크를 거친 입력)"]


def test_git_state_dotdot_through_link_to_sibling_is_a_link_input(tmp_path):
    """alias가 같은 폴더의 real을 가리키면 alias/../A.yaml은 실제로도 inputs/A.yaml이다. 최종 경로가 같아도
    링크를 거쳤으므로 변경이다 (PR #20 리뷰). 경로를 한 단계씩 따라가며 링크를 찾아야 잡힌다."""
    repo = _repo(tmp_path)
    inputs = repo / "eval" / "results" / "inputs"
    (inputs / "real").mkdir(parents=True)
    (inputs / "A.yaml").write_text("id: A\n", encoding="utf-8")
    (inputs / "alias").symlink_to(inputs / "real")
    _commit_all(repo)
    out = repo / "eval" / "results" / "x"
    assert not git_state(out, root=repo, inputs=[inputs / "A.yaml"]).dirty
    assert not git_state(out, root=repo, inputs=[inputs / "real" / ".." / "A.yaml"]).dirty
    state = git_state(out, root=repo, inputs=[inputs / "alias" / ".." / "A.yaml"])
    assert state.dirty and state.changed == ["eval/results/inputs/A.yaml (심볼릭 링크를 거친 입력)"]
