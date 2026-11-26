"""Voyage AI 임베딩과 리랭크 (설계 5단계 2장).

키는 클라우드 환경의 네트워크 시크릿으로 등록되어 프록시가 api.voyageai.com 요청에 붙인다.
코드는 VOYAGE_API_KEY가 있으면 그 값을, 없으면 자리표시 값을 보낸다. 키를 저장소에 적지 않는다.

임베딩은 텍스트마다 캐시한다(.cache/voyage, 저장소에 넣지 않음). 같은 모델, 같은 입력 종류, 같은 텍스트면
다시 부르지 않는다. 무료 토큰을 아끼고 색인을 다시 만들 때 같은 값을 쓰기 위해서다.
"""

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

API = "https://api.voyageai.com/v1"
EMBED_MODEL = "voyage-4"
RERANK_MODEL = "rerank-3"
BATCH = 128
# 결제 수단이 없는 계정은 분당 3회, 분당 1만 토큰으로 제한된다 (2026-11-26 실제 응답). 한 번에 보내는 글자 수를
# 줄이고, 제한에 걸리면 기다렸다가 같은 요청을 다시 보낸다. 한국어는 글자당 약 0.75토큰이다 (실측, 추정치)
MAX_CHARS = 4000
RATE_WAIT = 21.0
RATE_RETRIES = 12  # 요청 하나에서 속도 제한을 기다리는 최대 횟수 (약 4분)
SERVER_RETRIES = 3  # 5xx 응답과 연결 오류(시간 초과, 프록시 끊김)를 다시 보내는 횟수
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "voyage"


class VoyageError(RuntimeError):
    pass


@dataclass
class Usage:
    embed_tokens: int = 0
    rerank_tokens: int = 0
    calls: list[dict] = field(default_factory=list)


class Voyage:
    def __init__(self, cache_dir: Path | None = CACHE_DIR, timeout: float = 60.0):
        key = os.environ.get("VOYAGE_API_KEY") or "injected-by-proxy"
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"})
        self.usage = Usage()
        self.rate_waits = 0
        self._cache_path = cache_dir / f"{EMBED_MODEL}.jsonl" if cache_dir else None
        self._cache: dict[str, list[float]] = {}
        if self._cache_path and self._cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        """캐시를 읽는다. 쓰다가 끊겨 마지막 줄만 깨졌으면 그 줄을 잘라 내고, 중간 줄이 깨졌으면 멈춘다 (PR #35 리뷰).

        여러 프로세스(검색 채점기와 진단 실행의 검색 서버)가 같은 파일을 쓰므로 읽고 고치는 동안 파일을 잠근다.
        잠그지 않으면 다른 프로세스가 쓰는 중인 마지막 줄을 깨진 줄로 보고 잘라 낸다 (M5 점검)
        """
        with self._cache_path.open("r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                self._read_cache(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _read_cache(self, f) -> None:
        data = f.read()
        lines = data.split(b"\n")
        offset = 0  # 지금 줄이 시작하는 바이트 위치
        for i, line in enumerate(lines):
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    if any(rest for rest in lines[i + 1:]):
                        raise VoyageError(f"임베딩 캐시 {self._cache_path}의 {i + 1}번째 줄이 깨졌다") from exc
                    # 정상 기록을 다시 쓰지 않고 깨진 줄의 시작 위치에서 파일을 자른다 (PR #35 리뷰)
                    f.truncate(offset)
                    return
                self._cache[row["key"]] = row["embedding"]
            offset += len(line) + 1
        if data and not data.endswith(b"\n"):
            # 마지막 기록은 온전한데 줄바꿈 전에 끊겼다. 다음 기록이 같은 줄에 붙지 않게 줄바꿈을 더한다 (PR #35 리뷰)
            f.seek(0, 2)
            f.write(b"\n")

    def _post(self, path: str, body: dict) -> dict:
        # 기다린 횟수는 요청마다 센다. 실행 전체로 세면 오래 도는 검색 서버가 나중에는 바로 실패한다 (M5 점검)
        waits = errors = 0
        while True:
            try:
                resp = self._http.post(f"{API}/{path}", json=body)
            except httpx.TransportError as exc:
                if errors >= SERVER_RETRIES:
                    raise VoyageError(f"Voyage {path} 연결 실패: {exc}") from exc
                time.sleep(2 ** errors)
                errors += 1
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                if waits >= RATE_RETRIES:
                    raise VoyageError(f"Voyage {path}: 속도 제한이 풀리지 않는다")
                waits += 1
                self.rate_waits += 1  # 실행 기록에 남기는 전체 대기 횟수
                time.sleep(RATE_WAIT)  # 분당 제한이라 1분 창이 지나가기를 기다린다
                continue
            if resp.status_code in (500, 502, 503, 504) and errors < SERVER_RETRIES:
                time.sleep(2 ** errors)
                errors += 1
                continue
            raise VoyageError(f"Voyage {path} 실패: HTTP {resp.status_code} {resp.text[:200]}")

    @staticmethod
    def _key(text: str, input_type: str) -> str:
        return hashlib.sha256(f"{EMBED_MODEL}\0{input_type}\0{text}".encode()).hexdigest()

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """input_type은 document(색인할 조각) 또는 query(검색어)다."""
        if input_type not in ("document", "query"):
            raise ValueError(f"input_type은 document나 query다: {input_type}")
        keys = [self._key(t, input_type) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self._cache]
        for part in _batches(missing):
            data = self._post("embeddings", {"input": [t for _, t in part], "model": EMBED_MODEL,
                                             "input_type": input_type})
            tokens = data.get("usage", {}).get("total_tokens", 0)
            self.usage.embed_tokens += tokens
            self.usage.calls.append({"api": "embeddings", "inputs": len(part), "tokens": tokens})
            rows = sorted(data["data"], key=lambda r: r["index"])
            new = {k: r["embedding"] for (k, _), r in zip(part, rows)}
            self._cache.update(new)
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                lines = "".join(json.dumps({"key": k, "embedding": emb}) + "\n" for k, emb in new.items())
                with self._cache_path.open("a", encoding="utf-8") as f:
                    # 다른 프로세스의 기록과 줄이 섞이지 않게 잠그고 한 번에 쓴다 (M5 점검)
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        f.write(lines)
                        f.flush()
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
        return [self._cache[k] for k in keys]

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """(documents 안의 위치, 점수)를 점수 높은 순으로 top_k개 돌려준다."""
        if not documents:
            return []
        data = self._post("rerank", {"query": query, "documents": documents, "model": RERANK_MODEL,
                                     "top_k": min(top_k, len(documents))})
        tokens = data.get("usage", {}).get("total_tokens", 0)
        self.usage.rerank_tokens += tokens
        self.usage.calls.append({"api": "rerank", "inputs": len(documents), "tokens": tokens})
        return [(r["index"], r["relevance_score"]) for r in data["data"]]


def _batches(items: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """BATCH개, MAX_CHARS글자를 넘지 않게 묶는다. 한 텍스트가 MAX_CHARS보다 길면 혼자 한 묶음이다."""
    out, cur, size = [], [], 0
    for item in items:
        if cur and (len(cur) >= BATCH or size + len(item[1]) > MAX_CHARS):
            out.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += len(item[1])
    if cur:
        out.append(cur)
    return out
