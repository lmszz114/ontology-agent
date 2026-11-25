"""에이전트 운영 화면 (사내 도구): 지식 후보 검수와 반영(A17) (설계 5단계 8-2, 3단계 R8-6, R8-7, M4 PR 5).

실행: streamlit run apps/review_ui/app.py   (개발용 Neo4j, NEO4J_URI)

인증은 없다. 화면에서 고른 팀으로 행동한다 (M4 결정 6). 멀티테넌시, 사용자 관리, 권한 체계는 만들지 않는다 (5단계 1장).
- 후보 검수: 그 팀에 배정된 후보를 원문 근거와 함께 보고 승인, 반려한다. conflict는 K의 기존 사실을 함께 보여준다
- 지식 반영(A17): 실행 팀(개발PM팀)이 승인된 후보의 반영을 요청하고, 승인권자(검수 팀) 전원이 승인하면 실행 팀이 반영한다.
  반영은 지금 브랜치에 시드 변경을 커밋하고 다시 적재한다. M8 전까지 이 커밋은 main에 머지하지 않는다 (M4 PR 5 결정 5)
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

# streamlit run은 스크립트 폴더만 모듈 경로에 넣는다. 저장소 루트를 넣어야 agents, common을 읽는다 (M4 점검에서 발견)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.extraction import apply as A
from agents.extraction.rules import Knowledge
from common.gate import gate
from common.tracing import Tracer
from common.writers.events import list_knowledge_runs
from common.writers.extraction import list_candidates, review_candidate
from common.writers.events import RecordError
from ontology.db import get_driver
from ontology.seed import load_seed

KST = timezone(timedelta(hours=9))
REL_TEXT = {"OWNS": "이(가) 맡는다", "DEPENDS_ON": "이(가) 의존한다", "OBSERVED_IN": "증상이 나타난다",
            "MAY_BE_CAUSED_BY": "의 원인 후보다", "LOCATED_IN": "에 있는 문제다", "CHECKED_BY": "(으)로 확인한다",
            "RESOLVED_BY": "절차를 연다", "ALIAS": "의 다른 이름이다"}


@st.cache_resource
def driver():
    return get_driver()


def tracer() -> Tracer:
    # 테스트는 session_state에 메모리 실행 기록을 넣는다. 화면은 Langfuse로 남긴다 (5단계 1장 '모든 실행을 남긴다')
    if "tracer" not in st.session_state:
        st.session_state["tracer"] = Tracer.from_env()
    return st.session_state["tracer"]


def now() -> datetime:
    return datetime.now(KST)


def documents() -> dict[str, dict]:
    with driver().session() as s:
        return {r["id"]: r for r in s.run("MATCH (d:Document) RETURN d.id AS id, d.title AS title, d.text AS text").data()}


def describe(c: dict) -> str:
    s, o = c["subject"], c["object"]
    props = ", ".join(f"{k}={v}" for k, v in (c.get("props") or {}).items())
    ids = f"{s.get('id') or '새 엔티티'} -> {o.get('id') or ('이름' if c['relation'] == 'ALIAS' else '새 엔티티')}"
    return f"**{s['name']}** → **{o['name']}** ({c['relation']}: {REL_TEXT[c['relation']]})  `{ids}` {props}"


def evidence(c: dict, docs: dict) -> None:
    text = docs.get(c["document_id"], {}).get("text", "")
    for span in c["evidence_spans"]:
        para = next((p for p in text.split("\n\n") if span in p), span)
        st.markdown("> " + para.replace(span, f"**{span}**").replace("\n", "\n> "))


def k_facts(c: dict, k: Knowledge) -> list[str]:
    """conflict를 판단할 때 볼 K의 기존 사실."""
    rel, s, o = c["relation"], c["subject"], c["object"]
    if rel == "OWNS" and o.get("id"):
        return [f"{r['start']} OWNS {o['id']} {r['props'].get('scope')} {r['props'].get('area') or ''}"
                for r in k.rels("OWNS", end=o["id"])]
    if s.get("id") and o.get("id"):
        return [f"{rel} {r['start']} -> {r['end']} {({k2: v for k2, v in r['props'].items() if k2 not in ('source', 'source_doc_ids')})}"
                for r in k.rels(rel, s["id"], o["id"])]
    return []


def review_tab(team: str, run_id: str | None, k: Knowledge) -> None:
    docs = documents()
    todo = [c for c in list_candidates(driver(), run_id=run_id, status="pending", team_id=team)
            if c["decisions"].get(team) is None]
    st.caption(f"이 팀이 판단할 후보 {len(todo)}개")
    for c in todo:
        with st.expander(f"{c['id']} · {c['relation']} · {c['match']}"):
            st.markdown(describe(c))
            st.caption(f"문서 {c['document_id']} {docs.get(c['document_id'], {}).get('title', '')} · "
                       f"신뢰도 {c.get('confidence')} · 검수 팀 {', '.join(c['reviewers'])} · 판단 {c['decisions'] or '없음'}")
            evidence(c, docs)
            if c["match"] == "conflict":
                st.warning("K의 기존 사실과 어긋난다:\n\n" + "\n\n".join(k_facts(c, k)))
            cols = st.columns(2)
            for col, (label, decision) in zip(cols, (("승인", "approve"), ("반려", "reject"))):
                if col.button(label, key=f"{decision}-{c['id']}"):
                    try:
                        review_candidate(driver(), candidate_id=c["id"], team_id=team, decision=decision, now=now())
                    except RecordError as exc:
                        st.error(str(exc))  # 다시 그리지 않아야 거부 이유가 남는다
                    else:
                        st.rerun()


def show_candidates(ids: list[str], by_id: dict, docs: dict) -> None:
    """실행에 든 후보의 내용과 근거. 승인권자는 이것을 보고 승인한다."""
    for cid in ids:
        c = by_id.get(cid)
        if c is None:
            st.markdown(f"- `{cid}` (후보를 찾을 수 없다)")
            continue
        st.markdown(f"- `{cid}` {describe(c)}")
        st.caption(f"문서 {c['document_id']} {docs.get(c['document_id'], {}).get('title', '')} · 검수 팀 {', '.join(c['reviewers'])}")
        evidence(c, docs)


def act(func, **kwargs):
    """게이트와 반영 함수를 부르고, 성공했을 때만 다시 그린다. 실패하면 이유를 화면에 남긴다."""
    try:
        out = func(**kwargs)
    except (gate.GateError, A.ApplyError) as exc:
        st.error(str(exc))
        return None
    st.rerun()
    return out


def apply_tab(team: str, run_id: str | None) -> None:
    runs = list_knowledge_runs(driver())
    busy = {cid for r in runs if r["status"] in ("pending_approval", "approved", "done") for cid in r["candidate_ids"]}
    executor = gate.queries.get_action_policy(driver(), gate.KNOWLEDGE_ACTION)["executed_by"]["team_id"]
    by_id = {c["id"]: c for c in list_candidates(driver())}
    docs = documents()
    st.info("반영은 지금 브랜치에 시드 변경을 커밋한다. M8 최종 실행 전까지 이 커밋은 main에 머지하지 않는다.")

    st.subheader("1. 반영 요청")
    ready = [c for c in list_candidates(driver(), run_id=run_id, status="approved") if c["id"] not in busy]
    auto = [c for c in ready if A.classify(c)[0]]
    for c in auto:
        st.markdown(f"- `{c['id']}` {describe(c)} · 자동 반영: {A.classify(c)[1]}")
    if not auto:
        st.caption("자동으로 반영할 승인된 후보가 없다")
    elif team != executor:
        st.caption(f"반영 요청과 실행은 실행 팀({executor})이 한다")
    else:
        chosen = st.multiselect("반영할 후보", [c["id"] for c in auto], default=[c["id"] for c in auto])
        if st.button("반영 요청") and chosen:
            act(gate.request_knowledge_apply, driver=driver(), tracer=tracer(), candidate_ids=chosen, now=now())

    st.subheader("사람이 직접 반영할 후보")
    st.caption("A17로 자동 반영하지 않는다. 시드에 직접 넣고 PR로 올린다 (M4 PR 5 결정 2)")
    manual = [(c, A.classify(c)[1]) for c in ready if not A.classify(c)[0]]
    manual += [(by_id[cid], f"{r['run_id']} 반영 중 돌려받음: {reason}") for r in runs if r["status"] == "done"
               for cid, reason in zip(r["manual_candidate_ids"], r["manual_reasons"]) if cid in by_id]
    for c, reason in manual:
        st.markdown(f"- `{c['id']}` {describe(c)} · {reason}")
    if not manual:
        st.caption("없음")

    st.subheader("2. 승인")
    for r in [r for r in runs if r["status"] == "pending_approval"]:
        st.markdown(f"`{r['run_id']}` 후보 {len(r['candidate_ids'])}개 · 승인권자 {', '.join(r['required_approver_teams'])} "
                    f"· 승인 {', '.join(r['approved_by_teams']) or '없음'}")
        with st.expander(f"{r['run_id']} 후보 내용"):
            show_candidates(r["candidate_ids"], by_id, docs)
        if team in r["required_approver_teams"] and team not in r["approved_by_teams"]:
            cols = st.columns(2)
            if cols[0].button("승인", key=f"ok-{r['run_id']}"):
                act(gate.approve, driver=driver(), tracer=tracer(), run_id=r["run_id"], team_id=team, now=now())
            if cols[1].button("반려", key=f"no-{r['run_id']}"):
                act(gate.reject, driver=driver(), tracer=tracer(), run_id=r["run_id"], team_id=team, now=now())

    st.subheader("3. 반영 실행")
    for r in [r for r in runs if r["status"] == "approved"]:
        st.markdown(f"`{r['run_id']}` 후보 {len(r['candidate_ids'])}개 · 전원 승인")
        with st.expander(f"{r['run_id']} 후보 내용"):
            show_candidates(r["candidate_ids"], by_id, docs)
        if team == executor and st.button("반영 실행", key=f"run-{r['run_id']}"):
            act(A.execute, driver=driver(), run_id=r["run_id"], tracer=tracer(), now=now())

    st.subheader("기록")
    for r in [r for r in runs if r["status"] in ("done", "rejected")]:
        if r["status"] == "rejected":
            tail = f" · 반려 {r.get('rejected_by_team')}"
        else:
            tail = f" · 커밋 {r['commit'][:7]}" if r.get("commit") else " · 커밋 없음(모두 사람에게 돌아감)"
            if r["manual_candidate_ids"]:
                tail += f" · 사람이 직접 반영할 것 {', '.join(r['manual_candidate_ids'])}"
        st.markdown(f"`{r['run_id']}` {r['status']} · 후보 {len(r['candidate_ids'])}개{tail}")


def main() -> None:
    st.set_page_config(page_title="에이전트 운영 화면 (사내 도구)", layout="wide")
    st.title("에이전트 운영 화면 (사내 도구)")
    seed = load_seed()
    teams = {t["id"]: t["name"] for t in seed["teams"]}
    team = st.sidebar.selectbox("팀", list(teams), format_func=lambda t: f"{t} {teams[t]}", key="team")
    with driver().session() as s:
        runs = s.run("MATCH (c:Candidate) RETURN DISTINCT c.run_id AS r ORDER BY r").value()
    run_id = st.sidebar.selectbox("추출 실행", [None, *runs], format_func=lambda r: r or "전체", key="run")
    tab1, tab2 = st.tabs(["후보 검수", "지식 반영 (A17)"])
    with tab1:
        review_tab(team, run_id, Knowledge.from_seed(seed))
    with tab2:
        apply_tab(team, run_id)


main()
