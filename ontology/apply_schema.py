"""ontology/schema 폴더의 Cypher 파일을 Neo4j에 적용한다.

실행: python -m ontology.apply_schema
"""

from pathlib import Path

from neo4j import Driver

from ontology.db import get_driver

SCHEMA_DIR = Path(__file__).parent / "schema"


def read_statements(path: Path) -> list[str]:
    """주석 줄을 빼고 세미콜론으로 문장을 나눈다."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("//")
    ]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def apply_schema(driver: Driver) -> int:
    """적용한 문장 수를 돌려준다."""
    count = 0
    with driver.session() as session:
        for path in sorted(SCHEMA_DIR.glob("*.cypher")):
            for statement in read_statements(path):
                session.run(statement).consume()
                count += 1
    return count


if __name__ == "__main__":
    with get_driver() as driver:
        print(f"스키마 문장 {apply_schema(driver)}개 적용")
