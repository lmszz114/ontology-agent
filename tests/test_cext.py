"""구성 C-ext 그래프: 추출 후보를 검수 없이 모두 넣은 온톨로지 (M4 PR 6)."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from ontology import cext
from ontology.load import build_graph
from ontology.seed import load_seed, seed_hash

SEED = load_seed("org-a")
EXTRACT_DEV = Path("eval/results/2026-11-25-m4-extract-dev")


def cand(rel, s, o, props=None, doc="PM-01", order=0, s_type=None, o_type="System", o_name=None):
    types = {"TM": "Team", "SY": "System", "S": "SymptomType", "C": "Cause", "E": "EvidenceCheck", "P": "Procedure"}

    def t(x):
        return next(v for p, v in sorted(types.items(), key=lambda kv: -len(kv[0])) if x and x.startswith(p))

    obj = {"type": "Name", "name": o, "id": None} if rel == "ALIAS" else {"type": t(o) if o else o_type, "name": o_name or o,
                                                                           "id": o}
    return {"relation": rel, "subject": {"type": s_type or t(s), "name": s, "id": s}, "object": obj,
            "props": props or {}, "document_id": doc, "_order": (doc, order)}


def rels(g, rel):
    return [(r["start"], r["end"], {k: v for k, v in r["props"].items() if k not in ("source", "source_doc_ids")})
            for r in g.rels[rel]]


def test_only_extracted_facts_replace_k_relations():
    g, _ = cext.build_cext_graph(SEED, [cand("DEPENDS_ON", "SY12", "SY13")])
    k = build_graph(SEED)
    assert rels(g, "DEPENDS_ON") == [("SY12", "SY13", {})]
    for rel in ("OWNS", "OBSERVED_IN", "MAY_BE_CAUSED_BY", "LOCATED_IN", "CHECKED_BY", "RESOLVED_BY"):
        assert g.rels[rel] == []  # K의 관계는 하나도 넣지 않는다
    # 추출 대상이 아닌 것은 K 그대로 (위험도, 승인권자, 역할 배정, 절차의 조치, 협업)
    for rel in ("REQUIRES_APPROVAL_FROM", "EXECUTED_BY", "ASSIGNED_TO", "HAS_ACTION", "TRIGGERS", "COLLABORATES_ON"):
        assert g.rels[rel] == k.rels[rel]
    assert {label: len(v) for label, v in g.nodes.items()} == {label: len(v) for label, v in k.nodes.items()}
    dep = g.rels["DEPENDS_ON"][0]["props"]
    assert dep["source"] == "extracted" and dep["source_doc_ids"] == ["PM-01"]


def test_defaults_and_dropped_candidates():
    batch = [cand("MAY_BE_CAUSED_BY", "S1", "C02"),  # prior 없음 -> 중간
             cand("CHECKED_BY", "C02", "E02", {"expect": "기록 없음"}),  # required 없음 -> 확정 조건
             cand("LOCATED_IN", "C02", "SY13"),  # scope 없음
             cand("OWNS", "TM03", None, {"scope": "logic"}, o_name="길드"),  # 새 엔티티
             cand("ALIAS", "TM03", "서버팀")]
    g, stats = cext.build_cext_graph(SEED, batch)
    assert rels(g, "MAY_BE_CAUSED_BY") == [("S1", "C02", {"prior": "중간"})]
    assert rels(g, "CHECKED_BY") == [("C02", "E02", {"required": True, "expect": "기록 없음"})]
    assert rels(g, "LOCATED_IN") == [("C02", "SY13", {})] and g.rels["OWNS"] == []
    assert stats["default_prior"] == 1 and stats["default_required"] == 1 and stats["no_scope"] == 1
    assert stats["dropped_new_entity"] == 1
    assert "서버팀" in next(n for n in g.nodes["Team"] if n["id"] == "TM03")["aliases"]


def test_check_args_come_from_the_extracted_location_not_from_k():
    # K의 C05 E04 인자는 SY09다. 추출된 위치가 다르면 그 위치로 채운다
    batch = [cand("LOCATED_IN", "C05", "SY10", {"scope": "logic"}), cand("CHECKED_BY", "C05", "E04", {"required": True}),
             cand("CHECKED_BY", "C05", "E03", {"required": True})]  # E03은 인자가 없는 근거
    g, _ = cext.build_cext_graph(SEED, batch)
    checks = {e: p for _, e, p in rels(g, "CHECKED_BY")}
    assert json.loads(checks["E04"]["args"]) == {"system": "SY10"} and "args" not in checks["E03"]
    # 위치가 없으면 인자를 채우지 않는다
    g, _ = cext.build_cext_graph(SEED, [cand("CHECKED_BY", "C05", "E04", {"required": True})])
    assert "args" not in rels(g, "CHECKED_BY")[0][2]


def test_condition_definition_comes_from_k_rule():
    g, _ = cext.build_cext_graph(SEED, [cand("RESOLVED_BY", "C01", "P6", {"condition": "미지급"}),
                                        cand("RESOLVED_BY", "C01", "P3")])
    by_proc = {e: p for _, e, p in rels(g, "RESOLVED_BY")}
    assert by_proc["P6"]["condition"] == "미지급" and by_proc["P6"]["condition_check"] == "E02"
    assert by_proc["P3"] == {}


def test_one_location_per_cause_by_majority_then_first():
    batch = [cand("LOCATED_IN", "C10", "SY18", {"scope": "execution"}, doc="RB-08"),  # 옛날 정보
             cand("LOCATED_IN", "C10", "SY08", {"scope": "execution"}, doc="PM-05"),
             cand("LOCATED_IN", "C10", "SY08", {"scope": "execution"}, doc="CH-03")]
    g, stats = cext.build_cext_graph(SEED, batch)
    assert rels(g, "LOCATED_IN") == [("C10", "SY08", {"scope": "execution"})] and stats["voted_slots"] == 1
    assert g.rels["LOCATED_IN"][0]["props"]["source_doc_ids"] == ["CH-03", "PM-05"]
    # 같은 수면 먼저 나온 후보(문서 id 순)
    g, _ = cext.build_cext_graph(SEED, [cand("MAY_BE_CAUSED_BY", "S1", "C02", {"prior": "낮음"}, doc="PM-02"),
                                        cand("MAY_BE_CAUSED_BY", "S1", "C02", {"prior": "높음"}, doc="CH-01")])
    assert rels(g, "MAY_BE_CAUSED_BY") == [("S1", "C02", {"prior": "높음"})]


def test_conflicting_owners_both_stay():
    """검수가 없다는 것이 C-ext다. 같은 담당 자리를 두 팀이 맡는 옛날 정보도 그대로 들어간다."""
    batch = [cand("OWNS", "TM04", "SY08", {"scope": "execution"}), cand("OWNS", "TM10", "SY08", {"scope": "execution"})]
    g, _ = cext.build_cext_graph(SEED, batch)
    assert sorted(s for s, _, _ in rels(g, "OWNS")) == ["TM04", "TM10"]


def _copy_extract(tmp_path, reps=("r1",)):
    out = tmp_path / "extract"
    out.mkdir()
    shutil.copy(EXTRACT_DEV / "run.yaml", out / "run.yaml")
    for rep in reps:
        shutil.copytree(EXTRACT_DEV / rep, out / rep)
    return out


def test_load_candidates_refuses_incomplete_rep_and_hash_tracks_files(tmp_path):
    out = _copy_extract(tmp_path, ("r1", "r2"))
    _, before = cext.load_candidates(out, "r1")
    assert before["extraction_seed_hash"] == seed_hash(SEED)
    first = min((out / "r2").glob("*.yaml"))
    first.unlink()
    with pytest.raises(cext.CextError, match="모두 돌지 않은"):
        cext.load_candidates(out, "r2")
    with pytest.raises(cext.CextError, match="r3"):
        cext.load_candidates(out, "r3")
    target = min((out / "r1").glob("*.yaml"))
    target.write_text(target.read_text(encoding="utf-8") + "\n# 바뀜\n", encoding="utf-8")
    _, after = cext.load_candidates(out, "r1")
    assert cext.cext_hash(SEED, before) != cext.cext_hash(SEED, after)


def test_load_candidates_refuses_test_split(tmp_path):
    out = _copy_extract(tmp_path)
    run = yaml.safe_load((out / "run.yaml").read_text(encoding="utf-8")) | {"split": "test"}
    (out / "run.yaml").write_text(yaml.safe_dump(run, allow_unicode=True), encoding="utf-8")
    with pytest.raises(cext.CextError, match="dev"):
        cext.load_candidates(out, "r1")


def test_runner_requires_cext_from_only_for_cext():
    from eval.runner.__main__ import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--config", "C-ext", "--split", "dev", "--out", "eval/results/x"])
    with pytest.raises(SystemExit):
        parse_args(["--config", "C", "--split", "dev", "--out", "eval/results/x", "--cext-from", str(EXTRACT_DEV)])
    args = parse_args(["--config", "C-ext", "--split", "dev", "--out", "eval/results/x", "--cext-from", str(EXTRACT_DEV)])
    assert args.cext_from == str(EXTRACT_DEV)


def test_runner_refuses_ext_db_same_as_dev_db(monkeypatch):
    from eval.runner.__main__ import check_cext, parse_args
    from eval.runner.runner import RunnerError

    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_EXT_URI", "bolt://localhost:7687/")
    args = parse_args(["--config", "C-ext", "--split", "dev", "--out", "eval/results/x", "--cext-from", str(EXTRACT_DEV)])
    with pytest.raises(RunnerError, match="같은 서버"):
        check_cext(args)
    monkeypatch.setenv("NEO4J_EXT_URI", "bolt://localhost:7689")
    args.reps = 4  # 추출은 3회뿐이다
    with pytest.raises(RunnerError, match="r4"):
        check_cext(args)


def test_grader_stops_when_the_graph_cannot_be_rebuilt(tmp_path):
    from eval.grader.grade import GradeError, check_basis

    out = _copy_extract(tmp_path)
    _, source = cext.load_candidates(out, "r1")
    run = {"split": "dev", "scenario": "SC-S1-01", "git_commit": "0" * 40, "cext_from": out.as_posix(),
           "knowledge_graph": {"hash": cext.cext_hash(SEED, source), "builder": cext.BUILDER_VERSION,
                               "source": source}}
    check_basis({"run": run}, SEED)
    target = min((out / "r1").glob("*.yaml"))
    target.write_text(target.read_text(encoding="utf-8") + "\n# 바뀜\n", encoding="utf-8")
    with pytest.raises(GradeError, match="C-ext"):
        check_basis({"run": run}, SEED)


# 아래는 테스트용 Neo4j(7688)에서 돈다. C-ext DB 대신 테스트 DB에 적재해 조회 쿼리가 C-ext 그래프에서 도는지 본다


@pytest.mark.integration
def test_load_cext_and_query_it(driver):
    from mcp_servers.ontology_query import queries
    from ontology.load import load_static

    try:
        info = cext.load_cext(driver, SEED, EXTRACT_DEV, "r1")
        meta = cext.loaded_meta(driver)
        assert meta["kind"] == "C-ext" and meta["hash"] == info["hash"] and meta["source"].endswith("-r1")
        assert info["counts"]["relationships"] > 0
        # 위치를 뽑지 못한 원인은 담당 팀이 없다고 답한다 (오류가 아니다)
        with driver.session() as s:
            no_loc = s.run("MATCH (c:Cause) WHERE NOT (c)-[:LOCATED_IN]->() RETURN c.id AS id ORDER BY id").value()
        assert no_loc, "dev 문서에 위치가 없는 원인이 있어야 한다 (C08, C12, C15, C21)"
        assert queries.get_owner_team(driver, no_loc[0])["primary"] == []
        ranked = queries.rank_causes(driver, "S1", "2026-11-25T10:00:00+09:00")
        assert ranked and all(r["prior"] in queries.PRIOR_SCORE for r in ranked)
        assert queries.get_action_policy(driver, "A10")["risk"] == "high"  # 위험도는 K에서
    finally:
        load_static(driver, SEED)


def test_same_server_written_differently_is_refused():
    with pytest.raises(cext.CextError, match="같은 서버"):
        cext.check_ext_uri("bolt://127.0.0.1:7687", "bolt://localhost:7687")
    with pytest.raises(cext.CextError):
        cext.check_ext_uri("neo4j://LOCALHOST:7687/", "bolt://localhost")  # 포트가 없으면 7687
    cext.check_ext_uri("bolt://localhost:7689", "bolt://127.0.0.1:7687")  # 기본 7689는 통과


def test_unresolvable_address_stops(monkeypatch):
    """한쪽 주소만 풀리지 않으면 이름과 IP를 비교하게 되어 같은 서버를 놓친다. 확인할 수 없으면 멈춘다."""
    import socket

    real = socket.getaddrinfo

    def fake(host, *args, **kwargs):
        if host == "dev-db":
            raise socket.gaierror("이름을 풀 수 없다")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    with pytest.raises(cext.CextError, match="주소를 풀 수 없어"):
        cext.check_ext_uri("bolt://127.0.0.1:7687", "bolt://dev-db:7687")
    with pytest.raises(cext.CextError, match="주소를 풀 수 없어"):
        cext.check_ext_uri("bolt://dev-db:7689", "bolt://localhost:7687")


def test_resume_refuses_results_from_another_graph(tmp_path, monkeypatch):
    """같은 추출 폴더라도 후보가 바뀌었으면 앞 결과와 한 반복에 섞지 않는다 (LLM 호출 전에 멈춘다)."""
    from eval.runner.__main__ import check_cext, parse_args
    from eval.runner.runner import RunnerError

    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_EXT_URI", "bolt://localhost:7689")
    src = _copy_extract(tmp_path)
    _, source = cext.load_candidates(src, "r1")
    out = tmp_path / "out"
    (out / "r1").mkdir(parents=True)
    result = out / "r1" / "SC-S1-01.json"
    result.write_text(json.dumps({"run": {"knowledge_graph": {"hash": cext.cext_hash(SEED, source)}}}), encoding="utf-8")
    args = parse_args(["--config", "C-ext", "--split", "dev", "--out", "eval/results/x", "--cext-from", str(src)])
    check_cext(args, out)  # 같은 그래프면 이어 돌린다
    target = min((src / "r1").glob("*.yaml"))
    target.write_text(target.read_text(encoding="utf-8") + "\n# 바뀜\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="섞지 않는다"):
        check_cext(args, out)


def test_grader_requires_graph_record_for_cext():
    from eval.grader.grade import GradeError, check_basis

    run = {"split": "dev", "scenario": "SC-S1-01", "config": "C-ext", "git_commit": "0" * 40}
    with pytest.raises(GradeError, match="그래프 기록"):
        check_basis({"run": run}, SEED)
    with pytest.raises(GradeError, match="그래프 기록"):
        check_basis({"run": run | {"cext_from": "x", "knowledge_graph": {}}}, SEED)


@pytest.mark.integration
def test_owner_query_with_owners_missing_scope(driver):
    """C-ext에는 scope 없는 담당이 들어올 수 있다. 같은 팀의 다른 담당과 함께 있어도 조회가 멈추지 않는다."""
    from mcp_servers.ontology_query import queries
    from ontology.load import load_static, write_graph

    batch = [cand("LOCATED_IN", "C02", "SY13", {"scope": "execution", "area": "content"}),
             cand("OWNS", "TM03", "SY13", {"area": "probability"}),
             cand("OWNS", "TM03", "SY13", {"scope": "logic", "area": "probability"})]
    g, _ = cext.build_cext_graph(SEED, batch)
    try:
        write_graph(driver, g, {"hash": "test", "org": "org-a", "kind": "C-ext"})
        owner = queries.get_owner_team(driver, "C02")
        assert owner["primary"] == [] and owner["other_owners"][0]["team_id"] == "TM03"
        assert {o["scope"] for o in owner["other_owners"][0]["owns"]} == {None, "logic"}
    finally:
        load_static(driver, SEED)


@pytest.mark.integration
def test_cause_without_location_stays_a_candidate(driver):
    """원인 후보 관계가 있으면 위치가 없어도 후보다 (R4-1). 위치는 변경 가중에만 쓰여서 가중만 0이다 (M4 점검)."""
    from datetime import datetime

    from common.writers.events import record_change, reset_event_layer
    from mcp_servers.ontology_query import queries
    from ontology.load import load_static, write_graph

    batch = [cand("MAY_BE_CAUSED_BY", "S2", "C04", {"prior": "중간"}), cand("MAY_BE_CAUSED_BY", "S2", "C05", {"prior": "중간"}),
             cand("LOCATED_IN", "C04", "SY01", {"scope": "operation"})]
    g, _ = cext.build_cext_graph(SEED, batch)
    try:
        write_graph(driver, g, {"hash": "test", "org": "org-a", "kind": "C-ext"})
        reset_event_layer(driver)
        record_change(driver, change_id="CG-0", kind="deploy", system_id="SY01",
                      changed_at=datetime.fromisoformat("2026-11-25T09:00:00+09:00"))
        ranked = queries.rank_causes(driver, "S2", "2026-11-25T10:00:00+09:00")
        assert [(r["cause_id"], r["system_id"], r["change_bonus"]) for r in ranked] == [("C04", "SY01", 2), ("C05", None, 0)]
        cands = {c["cause_id"]: c["located_in"] for c in queries.get_cause_candidates(driver, "S2")}
        assert cands["C05"] is None and cands["C04"]["system_id"] == "SY01"
    finally:
        reset_event_layer(driver)
        load_static(driver, SEED)
