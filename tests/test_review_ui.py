"""에이전트 운영 화면 (사내 도구) 기본 동작 (M4 PR 5). 테스트용 Neo4j(7688)에서 Streamlit AppTest로 돈다.

반영 실행(A17 커밋)은 실제 저장소를 바꾸므로 여기서 누르지 않는다. 임시 저장소에서 tests/test_knowledge_apply.py가 본다.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests import conftest

APP = Path(__file__).resolve().parents[1] / "apps" / "review_ui" / "app.py"

pytestmark = pytest.mark.integration


@pytest.fixture
def app(driver, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from common.writers.events import reset_event_layer
    from common.writers.extraction import reset_extraction_layer
    from ontology.apply_schema import apply_schema
    from ontology.load import load_static
    from ontology.seed import load_seed
    from tests.memory_tracer import MemoryTracer
    from tests.test_knowledge_apply import SEED, _new_dependency, _write, cand

    monkeypatch.setenv("NEO4J_URI", conftest.test_db_uri())
    apply_schema(driver)
    load_static(driver, load_seed())
    reset_extraction_layer(driver)
    a, b = _new_dependency()
    ids = _write(driver, [cand("_", "OWNS", "TM03", "SY13", {"scope": "logic"}), cand("_", "DEPENDS_ON", a, b)])
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["tracer"] = MemoryTracer()
    yield at, ids
    reset_extraction_layer(driver)
    reset_event_layer(driver)
    load_static(driver, SEED)


def _choose_team(at, team):
    at.sidebar.selectbox(key="team").select(team)
    at.run()


def test_review_and_request_apply(app, driver):
    from common.writers.events import list_knowledge_runs
    from common.writers.extraction import list_candidates

    at, (owns_id, dep_id) = app
    at.run()
    assert not at.exception
    _choose_team(at, "TM03")
    # 서버개발팀(TM03)은 우편함 로직 담당이라 OWNS 후보의 검수 팀이다
    assert any(owns_id in e.label for e in at.expander)
    at.button(key=f"approve-{owns_id}").click()
    at.run()
    assert next(c for c in list_candidates(driver) if c["id"] == owns_id)["status"] == "approved"
    assert not any(owns_id in e.label for e in at.expander)  # 판단한 후보는 목록에서 빠진다

    # 실행 팀(개발PM팀)만 반영을 요청한다
    _choose_team(at, "TM05")
    at.multiselect[0].set_value([owns_id])
    at.button[[b.label for b in at.button].index("반영 요청")].click()
    at.run()
    runs = list_knowledge_runs(driver)
    assert len(runs) == 1 and runs[0]["candidate_ids"] == [owns_id] and runs[0]["status"] == "pending_approval"

    # 승인권자(검수 팀)는 실행에 든 후보 내용을 보고 승인한다
    _choose_team(at, "TM03")
    assert any(e.label == f"{runs[0]['run_id']} 후보 내용" for e in at.expander)
    assert any(owns_id in m.value for m in at.markdown)
    at.button(key=f"ok-{runs[0]['run_id']}").click()
    at.run()
    assert list_knowledge_runs(driver)[0]["status"] == "approved"
    assert not at.exception


NOW = datetime(2026, 11, 25, 15, 0, tzinfo=timezone(timedelta(hours=9)))


def test_failed_request_keeps_the_reason_on_screen(app, driver, monkeypatch):
    """게이트가 거부하면 이유가 화면에 남는다. 실패한 뒤 다시 그리면 오류 표시가 사라진다."""
    from common.gate import gate
    from common.writers.extraction import review_candidate

    at, (owns_id, _) = app
    review_candidate(driver, candidate_id=owns_id, team_id="TM03", decision="approve", now=NOW)

    def refuse(driver, **kwargs):
        raise gate.GateError("다른 세션이 먼저 요청했다")

    monkeypatch.setattr(gate, "request_knowledge_apply", refuse)
    at.run()
    _choose_team(at, "TM05")
    at.multiselect[0].set_value([owns_id])
    at.button[[b.label for b in at.button].index("반영 요청")].click()
    at.run()
    assert [e.value for e in at.error] == ["다른 세션이 먼저 요청했다"] and not at.exception


def test_candidates_a_person_must_apply_stay_listed(app, driver):
    from common.writers.extraction import (
        list_candidates,
        record_candidates,
        review_candidate,
    )
    from tests.test_knowledge_apply import cand

    span = next(c for c in list_candidates(driver))["evidence_spans"][0]
    (cid,) = record_candidates(driver, run_id="x2", document_id="PM-01",
                               candidates=[cand("_", "CHECKED_BY", "C02", "E09") | {"evidence_spans": [span]}])
    for team in next(c for c in list_candidates(driver) if c["id"] == cid)["reviewers"]:
        review_candidate(driver, candidate_id=cid, team_id=team, decision="approve", now=NOW)
    at, _ = app
    at.run()
    _choose_team(at, "TM05")
    assert not at.multiselect  # 자동 반영할 후보가 없어 요청 칸이 없다
    assert any(cid in m.value and "사람이 직접" in m.value for m in at.markdown)
