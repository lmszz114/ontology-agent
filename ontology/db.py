"""Neo4j 연결 설정."""

import os

from neo4j import Driver, GraphDatabase


def get_driver() -> Driver:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "localdev-password")
    return GraphDatabase.driver(uri, auth=(user, password))
