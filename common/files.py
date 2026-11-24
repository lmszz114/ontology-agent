"""파일 쓰기 도움 함수."""

import os
import tempfile
from pathlib import Path


def write_text_atomic(path: str | Path, text: str) -> None:
    """같은 폴더의 임시 파일에 다 쓴 뒤 이름을 바꾼다.

    중간에 멈춰도 원래 파일이 반쯤 쓴 상태로 남지 않는다. 결과 파일과 채점 파일이 깨지면
    비용을 낸 실행을 다시 읽을 수 없기 때문이다.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
