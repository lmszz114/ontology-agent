import os

import pytest
from neo4j.exceptions import ServiceUnavailable

from ontology.db import get_driver


@pytest.fixture(scope="session")
def driver():
    """실행 중인 Neo4j에 연결한다.

    연결할 수 없으면 integration 테스트를 건너뛴다.
    REQUIRE_NEO4J=1이면 (CI) 건너뛰지 않고 실패시킨다.
    """
    drv = get_driver()
    try:
        drv.verify_connectivity()
    except ServiceUnavailable:
        drv.close()
        message = "Neo4j에 연결할 수 없다. docker compose up -d neo4j 후 다시 실행"
        if os.environ.get("REQUIRE_NEO4J") == "1":
            pytest.fail(message)
        pytest.skip(message)
    yield drv
    drv.close()
