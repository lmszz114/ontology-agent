"""검색 대상 문서와 조각 나누기 (설계 4단계 2-5 청크 방식).

검색 대상은 기준 지식 K 텍스트(구성 A와 같은 텍스트)와 현장 문서 74개 전부다 (M5 사용자 결정).
문서의 dev, test 구분은 추출 프롬프트 조정용이고, 검색 시스템은 회사 문서 전체를 대상으로 한다고 본다.
검색 대상에는 문서 원문만 넣는다. 설계표의 정답 사실, 옛날 정보 표시, split은 넣지 않는다.

조각 방식 두 가지
- paragraph: 단락 단위. 현장 문서는 빈 줄로 나누고 제목 줄은 다음 단락에 붙인다.
  K 텍스트는 항목 하나(예: '- TM01: ...'와 그 아래 줄)를 한 조각으로 하고 장 제목을 앞에 붙인다
- fixed: 고정 길이. 글자 FIXED_SIZE개 창을 FIXED_OVERLAP개씩 겹쳐 민다 (가정값)
"""

import hashlib
from dataclasses import dataclass

SCHEMES = ("paragraph", "fixed")
FIXED_SIZE = 500  # 가정값. 현장 문서 평균 길이(약 950자)의 절반 정도
FIXED_OVERLAP = 100
K_DOC_ID = "K"


@dataclass(frozen=True)
class Chunk:
    id: str
    scheme: str
    doc_id: str
    order: int
    text: str


def sources(seed: dict, plan: list[dict]) -> list[tuple[str, str]]:
    """(문서 id, 원문). K가 맨 앞이고 현장 문서는 설계표 순서다."""
    from common.knowledge.k_text import k_text
    from eval.corpus.load import document_fields

    return [(K_DOC_ID, k_text(seed))] + [(d["id"], document_fields(d)["text"]) for d in plan]


def _paragraphs(text: str) -> list[str]:
    out, pending = [], ""
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        if block.startswith("#") and "\n" not in block:
            pending = f"{pending}\n{block}".strip()  # 제목만 있는 단락은 다음 단락에 붙인다
            continue
        out.append(f"{pending}\n{block}".strip() if pending else block)
        pending = ""
    if pending:
        out.append(pending)
    return out


def _k_items(text: str) -> list[str]:
    out, section, item = [], "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if item:
                out.append("\n".join(item))
            section, item = line, []
        elif line.startswith("- "):
            if item:
                out.append("\n".join(item))
            item = [section, line]
        elif item and line.strip():
            item.append(line)
    if item:
        out.append("\n".join(item))
    return out


def _fixed(text: str) -> list[str]:
    step = FIXED_SIZE - FIXED_OVERLAP
    return [text[i:i + FIXED_SIZE] for i in range(0, max(len(text) - FIXED_OVERLAP, 1), step)]


def chunk(doc_id: str, text: str, scheme: str) -> list[Chunk]:
    if scheme == "paragraph":
        parts = _k_items(text) if doc_id == K_DOC_ID else _paragraphs(text)
    elif scheme == "fixed":
        parts = _fixed(text)
    else:
        raise ValueError(f"정의되지 않은 조각 방식이다: {scheme} (가능: {SCHEMES})")
    return [Chunk(f"{scheme}:{doc_id}:{i:03d}", scheme, doc_id, i, p) for i, p in enumerate(parts, start=1)]


def all_chunks(seed: dict, plan: list[dict], schemes: tuple[str, ...] = SCHEMES) -> list[Chunk]:
    return [c for scheme in schemes for doc_id, text in sources(seed, plan) for c in chunk(doc_id, text, scheme)]


def corpus_hash(chunks: list[Chunk]) -> str:
    """조각 목록의 해시. 색인이 지금 시드와 문서로 만든 것인지 확인하는 데 쓴다."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.id}\0{c.text}\0".encode())
    return h.hexdigest()
