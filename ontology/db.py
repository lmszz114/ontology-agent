"""Neo4j 연결 설정."""

import os

from neo4j import Driver, GraphDatabase


def get_driver(uri: str | None = None) -> Driver:
    """uri를 주지 않으면 NEO4J_URI(개발용 DB)에 연결한다."""
    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "localdev-password")
    return GraphDatabase.driver(uri, auth=(user, password))
