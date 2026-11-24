"""채점기 실행.

실행: python -m eval.grader 결과폴더 [결과폴더 ...] [--compare 비교파일.md]

결과 폴더마다 grades.json과 GRADES.md를 쓴다. 폴더가 여럿이면 구성 비교표를 --compare 파일에 쓰거나 화면에 낸다.
결과 폴더는 반복 폴더(r1, r2 ...)를 담거나 결과 JSON을 바로 담는다.
"""

import argparse
import json
import sys
from pathlib import Path

from common.files import write_text_atomic
from eval.grader.grade import GradeError, grade_folder
from eval.grader.report import compare_markdown, folder_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="결과 파일 채점 (설계 4단계 5장)")
    parser.add_argument("folders", nargs="+", help="결과 폴더")
    parser.add_argument("--compare", help="여러 폴더의 비교표를 쓸 마크다운 파일")
    args = parser.parse_args(argv)

    results = []
    try:
        for folder in map(Path, args.folders):
            result = grade_folder(folder)
            write_text_atomic(folder / "grades.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            write_text_atomic(folder / "GRADES.md", folder_markdown(result))
            print(f"{folder}: grades.json, GRADES.md")
            results.append(result)
    except GradeError as exc:
        print(f"채점할 수 없다: {exc}", file=sys.stderr)
        return 1
    if len(results) > 1:
        text = compare_markdown(results)
        if args.compare:
            write_text_atomic(args.compare, text)
            print(f"비교표: {args.compare}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
