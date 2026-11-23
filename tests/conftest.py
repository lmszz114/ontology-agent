import os

import pytest
from neo4j.exceptions import ServiceUnavailable

from ontology.db import get_driver


@pytest.fixture(scope="session")
def driver():
    """테스트용 Neo4j에 연결한다.

    테스트는 적재를 덮어쓰고 사건 계층을 지운다. 개발용 DB(NEO4J_URI)를 건드리지 않도록
    따로 띄운 테스트용 DB(NEO4J_TEST_URI, 기본 7688 포트)만 쓴다.
    연결할 수 없으면 integration 테스트를 건너뛴다.
    REQUIRE_NEO4J=1이면 (CI) 건너뛰지 않고 실패시킨다.
    """
    drv = get_driver(os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7688"))
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
