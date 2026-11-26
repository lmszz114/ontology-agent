"""배포 전 회귀 테스트 (설계 5단계 3-2, M6 PR 3).

실행:
  python -m eval.regression paths              # 돌릴 시나리오 파일 경로 (실행기 --scenario에 넘긴다)
  python -m eval.regression check 결과폴더      # 채점한 결과(grades.json)가 통과 기준을 넘는지. 못 넘으면 1
  python -m eval.regression baseline 결과폴더   # 그 결과를 기준 값으로 eval/regression.yaml에 쓴다

GitHub Actions(.github/workflows/regression.yml)가 실행기, 채점기, check 순으로 부른다. LLM은 실행기만 부른다.
기준 값은 사람이 확인한 결과로만 쓴다. CI는 기준을 쓰지 않는다
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from eval.validate import SCENARIO_DIR

CONFIG_PATH = Path(__file__).parent / "regression.yaml"


class RegressionError(Exception):
    """설정이나 결과가 회귀 테스트의 조건과 맞지 않는다."""


def load_config(path: Path | None = None) -> dict:
    config = yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))
    paths = scenario_paths(config)
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise RegressionError(f"dev에 없는 회귀 시나리오다: {missing}")
    if len(set(config["scenarios"])) != len(config["scenarios"]):
        raise RegressionError("회귀 시나리오가 중복된다")
    return config


def scenario_paths(config: dict) -> list[Path]:
    return [SCENARIO_DIR / "dev" / f"{sid}.yaml" for sid in config["scenarios"]]


def _grades(folder: Path, config: dict) -> dict:
    """채점 결과를 읽고 회귀 설정과 같은 조건으로 돈 것인지 본다."""
    path = folder / "grades.json"
    if not path.exists():
        raise RegressionError(f"채점 결과가 없다. python -m eval.grader {folder}를 먼저 돌린다")
    grades = json.loads(path.read_text(encoding="utf-8"))
    meta = grades["meta"]
    wrong = {k: (meta[k], config[k]) for k in ("config", "org") if meta[k] != config[k]}
    if meta["split"] != "dev":
        wrong["split"] = (meta["split"], "dev")
    if sorted(meta["scenarios"]) != sorted(config["scenarios"]):
        wrong["scenarios"] = (sorted(meta["scenarios"]), sorted(config["scenarios"]))
    if meta["repetitions"] != 1:
        wrong["repetitions"] = (meta["repetitions"], 1)
    if wrong:
        raise RegressionError(f"회귀 설정과 다른 조건의 결과다 (결과, 설정): {wrong}")
    return grades


def failures(grades: dict, config: dict) -> list[str]:
    """통과 기준을 넘지 못한 항목. 비었으면 통과다."""
    metrics = grades["reps"][0]["metrics"]
    criteria = config["criteria"]
    baseline = config.get("baseline")
    found = []
    for key, label in (("approval_missing", "승인 요청 누락"), ("unauthorized", "무단 고위험 실행")):
        if metrics[key] > criteria[key]:
            found.append(f"{label} {metrics[key]}건 (기준 {criteria[key]}건 이하)")
    if not baseline:
        found.append("기준 값이 없다. 사람이 확인한 결과로 python -m eval.regression baseline을 먼저 쓴다")
    elif metrics["top1"] is None or metrics["top1"] < baseline["top1"] - criteria["top1_drop"] - 1e-9:
        found.append(f"원인 Top-1 {metrics['top1']} (기준 {baseline['top1']}에서 {criteria['top1_drop']}보다 많이 떨어짐)")
    return found


def summary_lines(grades: dict, config: dict) -> list[str]:
    metrics = grades["reps"][0]["metrics"]
    meta = grades["meta"]
    baseline = config.get("baseline") or {}
    failed = [f"{sid} ({s['first_failures'][0]})" for sid, s in sorted(grades["per_scenario"].items())
              if s["first_failures"][0]]
    return [
        f"구성 {meta['config']}, 커밋 {meta['git_commit'][:7]}, 시나리오 {len(meta['scenarios'])}건, 모델 {', '.join(meta['models'])}",
        f"원인 Top-1 {metrics['top1']} (기준 {baseline.get('top1')}, 기준 커밋 {str(baseline.get('git_commit'))[:7]})",
        f"승인 요청 누락 {metrics['approval_missing']}건, 무단 고위험 실행 {metrics['unauthorized']}건, "
        f"비용 {metrics['cost_usd']:.5f} USD",
        "틀린 시나리오 (처음 틀린 단계): " + (", ".join(failed) or "없음"),
    ]


def write_baseline(grades: dict, path: Path | None = None) -> dict:
    """결과를 기준 값으로 쓴다. 설정 파일의 주석을 지키려고 baseline 줄만 바꾼다."""
    meta, metrics = grades["meta"], grades["reps"][0]["metrics"]
    if meta.get("git_dirty"):
        raise RegressionError("커밋하지 않은 변경으로 돈 결과는 기준으로 쓰지 않는다")
    baseline = {"top1": metrics["top1"], "git_commit": meta["git_commit"], "started_at": meta["started_at"],
                "models": meta["models"], "prompt_version": meta["prompt_version"], "seed_hash": meta["seed_hash"]}
    path = path or CONFIG_PATH
    text = path.read_text(encoding="utf-8")
    head, sep, _ = text.partition("\nbaseline:")
    if not sep:
        raise RegressionError(f"{path.name}에 baseline 항목이 없다")
    dumped = yaml.safe_dump({"baseline": baseline}, allow_unicode=True, sort_keys=False)
    path.write_text(head + "\n" + dumped, encoding="utf-8")
    return baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="배포 전 회귀 테스트 (5단계 3-2)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("paths")
    for name in ("check", "baseline"):
        sub.add_parser(name).add_argument("folder")
    args = parser.parse_args(argv)
    try:
        config = load_config()
        if args.command == "paths":
            print(" ".join(p.relative_to(Path.cwd()).as_posix() if p.is_relative_to(Path.cwd()) else str(p)
                           for p in scenario_paths(config)))
            return 0
        grades = _grades(Path(args.folder), config)
        if args.command == "baseline":
            baseline = write_baseline(grades)
            print(f"기준 값을 썼다: 원인 Top-1 {baseline['top1']}, 커밋 {baseline['git_commit'][:7]}")
            return 0
        print("\n".join(summary_lines(grades, config)))
        found = failures(grades, config)
        for line in found:
            print(f"실패: {line}")
        print("회귀 테스트 통과" if not found else f"회귀 테스트 실패 {len(found)}건")
        return 1 if found else 0
    except RegressionError as exc:
        print(f"회귀 테스트를 할 수 없다: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
