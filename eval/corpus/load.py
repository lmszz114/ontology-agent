"""현장 문서 묶음 D를 지식 추출 계층의 Document로 적재한다 (설계 3단계 2-3, 4단계 3-5, M4 PR 3).

실행: python -m eval.corpus.load --split dev [--reset]

평가 설계표(plan.yaml)의 평가용 정보는 그래프에 넣지 않는다. dev와 test 구분, 정답 사실, 옛날 정보 표시, 원인 목록을
넣으면 추출기가 답을 보게 된다. 방해 문서는 일반 문서(other)로 넣는다 (M4 PR 3 결정 1).
test 문서는 M8에서 처음 추출한다 (4단계 4-2). 그래서 --split test는 --allow-test 없이는 거부한다.
"""

import argparse
import sys

from common.writers.extraction import record_documents
from eval.corpus.check import load_plan
from eval.corpus.generate import doc_path
from ontology.db import get_driver

KIND = {"postmortem": "postmortem", "runbook": "runbook", "chat": "chat", "meeting": "meeting", "distractor": "other"}


def document_fields(doc: dict) -> dict:
    """설계표의 문서 하나를 Document 속성으로 바꾼다. 제목은 설계표의 요지가 아니라 원문의 첫 줄이다."""
    text = doc_path(doc).read_text(encoding="utf-8")
    first = text.splitlines()[0]
    if not first.startswith("# "):
        raise ValueError(f"{doc['id']}: 원문 첫 줄이 '# '로 시작하는 제목이 아니다")
    return {"document_id": doc["id"], "kind": KIND[doc["kind"]], "title": first[2:].strip(), "text": text,
            "created_at": doc["created_at"], "author_team": doc.get("author_team")}


def load_documents(driver, split: str, plan: list[dict] | None = None, *, replace: bool = False) -> list[str]:
    """한 split의 문서를 한 트랜잭션으로 적재한다. replace면 같은 트랜잭션에서 지식 추출 계층을 먼저 비운다.

    중간에 실패하면 아무것도 바뀌지 않아 같은 명령으로 다시 돌리면 된다 (PR #29 리뷰).
    """
    plan = load_plan() if plan is None else plan
    docs = [document_fields(d) for d in plan if d["split"] == split]
    record_documents(driver, docs, replace=replace)
    return [d["document_id"] for d in docs]


def main() -> int:
    parser = argparse.ArgumentParser(description="현장 문서를 Document 노드로 적재")
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--reset", action="store_true",
                        help="이 DB의 지식 추출 계층(모든 문서와 후보)을 지우고 적재한다. 지우기와 적재는 한 트랜잭션이다")
    parser.add_argument("--allow-test", action="store_true", help="test 문서 적재 (M8 최종 실행에서만)")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("test 문서는 M8 최종 실행에서만 적재한다 (4단계 4-2). --allow-test가 필요하다")
    with get_driver() as driver:
        ids = load_documents(driver, args.split, replace=args.reset)
    print(f"{args.split} 문서 {len(ids)}개 적재")
    return 0


if __name__ == "__main__":
    sys.exit(main())
