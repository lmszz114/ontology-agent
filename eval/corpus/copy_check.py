"""껍데기 방지 G1 점검: 현장 문서가 K의 표 문장을 그대로 옮겨 썼는지 센다 (설계 4단계 7장 G1, M4 점검).

실행: python -m eval.corpus.copy_check [--out 결과파일.yaml]

K의 문장은 시드에 사람이 쓴 설명 문장이다: 확정 조건 설명(expect), 참고 근거의 쓰임(note), 조건 설명, 절차 단계(steps),
심각도 기준(criteria), 역할 설명. 이름(팀, 시스템, 원인 이름)은 문서가 당연히 쓰는 말이라 세지 않는다.
문서 문장은 줄과 마침표(., ?, !, 다.)로 나눈다. 세는 것은 두 가지다.
- 같음: 문서 문장이 K 문장과 같다 (앞뒤 공백, 끝 마침표 무시)
- 포함: 문서 문장 안에 K 문장(8자 이상)이 글자 그대로 들어 있다
LLM을 쓰지 않는다. dev와 test 문서를 따로 낸다. 문서 내용을 읽기만 하고 프롬프트나 설정을 바꾸지 않아 test 사용 규칙(4-2)에 걸리지 않는다.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

from common.files import write_text_atomic
from eval.corpus.check import load_plan
from eval.corpus.load import document_fields
from ontology.seed import load_seed

MIN_LEN = 8


def k_sentences(seed: dict) -> set[str]:
    out: set[str] = set()

    def add(text):
        if isinstance(text, str) and len(_norm(text)) >= MIN_LEN:
            out.add(_norm(text))

    for cause in seed["causes"]:
        for ref in cause["checked_by"] + cause["resolved_by"]:
            add(ref.get("expect"))
            add(ref.get("note"))
    for proc in seed["procedures"]:
        for step in proc.get("steps") or []:
            add(step)
    for sev in seed["severities"]:
        add(sev.get("criteria"))
    for role in seed["roles"]:
        add(role.get("description"))
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".!?。").strip()


# 제목, 인용, 글머리표, 번호 목록 기호만 뗀다. "10분 이상"처럼 숫자로 시작하는 문장은 그대로 둔다 (PR #33 리뷰)
MARKER = re.compile(r"^\s*(?:#+|>|[-*]|\d+[.)])\s+")


def sentences(text: str) -> list[str]:
    # 줄마다 목록 기호를 먼저 뗀다. 문장으로 먼저 나누면 "1."이 따로 떨어진다
    parts = [p for line in text.splitlines() for p in re.split(r"(?<=[.!?])\s+", MARKER.sub("", line))]
    return [n for n in map(_norm, parts) if n]


def check(plan: list[dict], seed: dict) -> dict:
    ks = k_sentences(seed)
    result = {}
    for split in ("dev", "test"):
        total = same = contains = 0
        examples = []
        for doc in (d for d in plan if d["split"] == split):
            for s in sentences(document_fields(doc)["text"]):
                total += 1
                if s in ks:
                    same += 1
                    examples.append({"document": doc["id"], "sentence": s, "kind": "같음"})
                elif any(k in s for k in ks):
                    contains += 1
                    examples.append({"document": doc["id"], "sentence": s, "kind": "포함"})
        result[split] = {"documents": sum(1 for d in plan if d["split"] == split), "sentences": total,
                         "same": same, "contains": contains,
                         "same_ratio": round(same / total, 4) if total else None,
                         "contains_ratio": round((same + contains) / total, 4) if total else None,
                         "examples": examples}
    return {"k_sentences": len(ks), "min_len": MIN_LEN, "splits": result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="G1 점검: 문서가 K 문장을 그대로 옮긴 비율")
    parser.add_argument("--out", help="결과 YAML")
    args = parser.parse_args(argv)
    out = check(load_plan(), load_seed())
    text = yaml.safe_dump(out, allow_unicode=True, sort_keys=False)
    if args.out:
        write_text_atomic(Path(args.out), text)
    for split, r in out["splits"].items():
        print(f"{split}: 문서 {r['documents']}개, 문장 {r['sentences']}개, 같음 {r['same']}, 포함 {r['contains']}, "
              f"같음 비율 {r['same_ratio']}, 포함까지 비율 {r['contains_ratio']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
