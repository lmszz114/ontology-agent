import os

import pytest
from neo4j.exceptions import ServiceUnavailable

from ontology.db import get_driver


def test_db_uri() -> str:
    """테스트용 DB 주소. 값이 비어 있을 때도 테스트용 기본값을 쓴다.

    빈 값을 그대로 넘기면 get_driver가 개발용 DB(NEO4J_URI)로 연결한다.
    """
    return os.environ.get("NEO4J_TEST_URI") or "bolt://localhost:7688"


@pytest.fixture(scope="session")
def driver():
    """테스트용 Neo4j에 연결한다.

    테스트는 적재를 덮어쓰고 사건 계층을 지운다. 개발용 DB(NEO4J_URI)를 건드리지 않도록
    따로 띄운 테스트용 DB(NEO4J_TEST_URI, 기본 7688 포트)만 쓴다.
    연결할 수 없으면 integration 테스트를 건너뛴다.
    REQUIRE_NEO4J=1이면 (CI) 건너뛰지 않고 실패시킨다.
    """
    drv = get_driver(test_db_uri())
    try:
        drv.verify_connectivity()
    except ServiceUnavailable:
        drv.close()
        message = "테스트용 Neo4j에 연결할 수 없다. docker compose up -d neo4j-test 후 다시 실행"
        if os.environ.get("REQUIRE_NEO4J") == "1":
            pytest.fail(message)
        pytest.skip(message)
    yield drv
    drv.close()
