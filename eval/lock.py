"""test 세트 정답 라벨 잠금 (설계 4단계 4-2).

정답 라벨은 에이전트 프롬프트를 고치기 전에 고정하고, test 세트는 최종 실행(M8)에만 쓴다.
잠금 파일에 test 시나리오 파일마다 내용 해시를 적어 두고, 검증기, 채점기, 테스트가 지금 파일과 비교한다.
파일 하나라도 바뀌거나 더해지거나 빠지면 거부한다. 잠금을 다시 쓰는 일은 잠금 파일의 변경으로 Git 이력에 남는다.

실행: python -m eval.lock           # 지금 파일이 잠금과 같은지 확인
      python -m eval.lock --write   # 잠금을 새로 쓴다. 사용자가 라벨을 확정했을 때만 쓴다
"""

import argparse
import hashlib
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

SCENARIO_DIR = Path(__file__).parent / "scenarios"
TEST_DIR = SCENARIO_DIR / "test"
LOCK_PATH = SCENARIO_DIR / "test-lock.yaml"
KST = timezone(timedelta(hours=9))


def file_hashes(directory: Path = TEST_DIR, failures: list[str] | None = None) -> dict[str, str]:
    """파일 이름마다 내용(바이트)의 sha256.

    failures를 주면 읽지 못한 파일을 거기에 적고 건너뛴다. 주지 않으면 예외를 그대로 낸다.
    잠금을 쓸 때는 읽지 못한 파일이 있으면 쓰지 않아야 해서 예외를 낸다
    """
    hashes = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            if failures is None:
                raise
            failures.append(f"{path.name}: 파일을 읽을 수 없다 ({type(exc).__name__})")
    return hashes


def read_lock(path: Path = LOCK_PATH) -> dict | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def lock_errors(directory: Path = TEST_DIR, path: Path = LOCK_PATH) -> list[str]:
    """지금 test 파일이 잠금과 다른 점. 같으면 빈 목록. 잠금 파일이 없으면 그 자체가 문제다."""
    lock = read_lock(path)
    if lock is None:
        return [f"test 세트 잠금 파일이 없다: {path.name}"]
    # 읽지 못한 파일은 따로 알리고 나머지 파일은 계속 비교한다. 검증기가 다른 시나리오 검사까지 멈추지 않게 한다
    unreadable: list[str] = []
    locked, now = lock.get("files") or {}, file_hashes(directory, unreadable)
    skipped = {line.split(":", 1)[0] for line in unreadable}
    errors = list(unreadable)
    errors += [f"{name}: 잠근 뒤에 내용이 바뀌었다" for name in sorted(locked) if name in now and now[name] != locked[name]]
    errors += [f"{name}: 잠근 파일인데 없다" for name in sorted(set(locked) - set(now) - skipped)]
    errors += [f"{name}: 잠그지 않은 파일이다" for name in sorted(set(now) - set(locked))]
    return errors


def write_lock(note: str, directory: Path = TEST_DIR, path: Path = LOCK_PATH) -> dict:
    lock = {"locked_at": datetime.now(KST).strftime("%Y-%m-%dT%H:%M+09:00"), "note": note,
            "files": file_hashes(directory)}
    header = "# test 세트 정답 라벨 잠금 (eval/lock.py). 손으로 고치지 않는다\n"
    path.write_text(header + yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description="test 세트 정답 라벨 잠금")
    parser.add_argument("--write", metavar="NOTE", help="잠금을 새로 쓴다. 왜 쓰는지 적는다")
    args = parser.parse_args()
    if args.write:
        lock = write_lock(args.write)
        print(f"잠금을 썼다: 파일 {len(lock['files'])}개")
        return 0
    errors = lock_errors()
    for line in errors:
        print(line)
    print("test 세트가 잠금과 같다" if not errors else f"잠금과 다른 점 {len(errors)}건")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
