"""테스트용 가짜 임베딩과 리랭크. Voyage를 부르지 않는다.

글자 2-gram을 해시해 1024차원에 더한 뒤 정규화한다. 글자가 많이 겹치는 문장끼리 가깝다.
리랭크는 검색어와 겹치는 2-gram 수로 점수를 매긴다.
"""

import hashlib
import math


def _grams(text: str) -> list[str]:
    t = "".join(text.split())
    return [t[i:i + 2] for i in range(len(t) - 1)]


class FakeVoyage:
    def __init__(self):
        from common.retrieval.voyage import Usage

        self.embedded: list[tuple[str, str]] = []
        self.reranked = 0
        self.usage = Usage()

    def embed(self, texts, input_type):
        out = []
        for text in texts:
            self.embedded.append((input_type, text))
            v = [0.0] * 1024
            for g in _grams(text):
                v[int(hashlib.md5(g.encode()).hexdigest(), 16) % 1024] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v] if any(v) else [1.0] + [0.0] * 1023)
        return out

    def rerank(self, query, documents, top_k):
        self.reranked += 1
        q = set(_grams(query))
        scored = sorted(((i, len(q & set(_grams(d))) / (len(q) or 1)) for i, d in enumerate(documents)),
                        key=lambda x: (-x[1], x[0]))
        return scored[:top_k]
