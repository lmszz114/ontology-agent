import pytest
from neo4j.exceptions import ConstraintError

from ontology.apply_schema import SCHEMA_DIR, apply_schema, read_statements

# 설계 3단계 2장의 노드 타입 17개
NODE_LABELS = {
    "Team", "Role", "Channel", "System", "SymptomType", "Cause",
    "EvidenceCheck", "Procedure", "Action", "Severity",
    "Report", "Incident", "Finding", "ActionRun", "Change",
    "Document", "Candidate",
}


def test_constraints_file_covers_all_node_labels():
    statements = read_statements(SCHEMA_DIR / "constraints.cypher")
    assert len(statements) == len(NODE_LABELS)
    for label in NODE_LABELS:
        assert any(f"(n:{label})" in s for s in statements), label


@pytest.mark.integration
def test_apply_schema_creates_unique_id_constraints(driver):
    apply_schema(driver)
    apply_schema(driver)  # 두 번 적용해도 오류가 없어야 한다
    with driver.session() as session:
        rows = session.run(
            "SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties "
            "WHERE type IN ['UNIQUENESS', 'NODE_PROPERTY_UNIQUENESS'] "
            "RETURN labelsOrTypes, properties"
        ).data()
    constrained = {r["labelsOrTypes"][0] for r in rows if r["properties"] == ["id"]}
    assert NODE_LABELS <= constrained


@pytest.mark.integration
def test_duplicate_id_is_rejected(driver):
    apply_schema(driver)
    with driver.session() as session:
        session.run("MATCH (n:Team {id: '__test_dup__'}) DETACH DELETE n").consume()
        session.run("CREATE (:Team {id: '__test_dup__'})").consume()
        try:
            with pytest.raises(ConstraintError):
                session.run("CREATE (:Team {id: '__test_dup__'})").consume()
        finally:
            session.run("MATCH (n:Team {id: '__test_dup__'}) DETACH DELETE n").consume()
