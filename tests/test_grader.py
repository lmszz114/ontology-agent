"""채점기 단위 테스트. 가짜 결과를 실제 dev 시나리오 라벨과 시드로 채점한다 (LLM, Neo4j 없음)."""

import copy
import json
from pathlib import Path

import pytest

from common.files import write_text_atomic
from eval.grader.__main__ import main
from eval.grader.grade import (
    GradeError,
    aggregate,
    final_ranking,
    grade_folder,
    grade_scenario,
    load_results,
    load_scenario,
    spread,
)
from ontology.seed import load_seed

SEED = load_seed()


def _run(scenario_id: str, **over) -> dict:
    run = {"scenario": scenario_id, "split": "dev", "config": "C", "org": "org-a", "mode": "eval",
           "started_at": "2026-11-24T13:00:00+09:00", "finished_at": "2026-11-24T13:00:10+09:00", "seconds": 10.0,
           "models": ["claude-haiku-5-5"], "prompt_version": "diagnosis-v6", "git_commit": "abc1234def",
           "llm_calls": 4, "usage": {"input": 1000, "output": 100}, "cost_usd": 0.002}
    return run | over


def _call(check_id: str, args: dict | None = None, *, step: str = "7.check", is_error: bool = False) -> dict:
    return {"step": step, "cause_id": None, "tool": "t", "check_id": check_id, "args": args or {}, "is_error": is_error}


def _action(action_id: str, mode: str, status: str, blocked: bool = False) -> dict:
    return {"action_id": action_id, "requested_mode": mode, "status": status, "gate_blocked": blocked, "run_id": "AR"}


def perfect_s7_02() -> dict:
    """SC-S7-02 정답 그대로의 결과. 2번 제보는 분류하지 않는다."""
    scenario = load_scenario("dev", "SC-S7-02")
    e = scenario["expected"]
    output = {
        "symptom": "S7", "report_indexes": [0, 1, 3], "scope": "all_service", "severity": "SEV1", "escalated": False,
        "hypotheses": [
            {"cause_id": "C18", "rank": 1, "status": "ruled_out", "result": "rule_out"},
            {"cause_id": "C19", "rank": 2, "status": "confirmed", "result": "confirm"},
            {"cause_id": "C14", "rank": 3, "status": "candidate", "result": None},
        ],
        "root_cause": "C19", "primary_team": "TM10", "procedures": ["P1", "P3", "P4", "P10"],
        "actions": [_action(a, "auto", "auto_done") for a in e["auto_actions"]]
                   + [_action(a, "approval", "pending_approval") for a in e["approval_actions"]],
        "tool_calls": [_call("E15", step="1.intake"), _call("E04", {"system": "SY14"}),
                       _call("E06", {"system": "SY14"}), _call("E09", {"system": "SY11"})],
    }
    return {"run": _run("SC-S7-02"), "outputs": [output], "unclassified": [{"report_index": 2, "symptom": "none"}]}


def grade(result: dict) -> dict:
    return grade_scenario(result, load_scenario("dev", result["run"]["scenario"]), SEED)


def test_perfect_result_has_no_failure():
    g = grade(perfect_s7_02())
    assert g["first_failure"] is None
    assert g["symptom"] and g["grouping"] and g["top1"] and g["team"] and g["procedures"]
    assert g["pairs"] == {"tp": 3, "fp": 0, "fn": 0}
    assert g["required"] == {"total": 1, "covered": 1}
    # 배제한 C18의 판정 근거 두 건은 탐색 조회다. 접수 단계 E15는 세지 않는다
    assert (g["unnecessary"], g["exploratory"]) == (0, 2)
    # 1순위가 배제되고 2순위가 확정됐다. 최종 Top-1은 맞고 근거 확인 전 Top-1은 틀리다
    assert g["ranking"] == ["C19", "C14"]
    assert not g["pre_rank_top1"]


def test_unclassified_report_merged_fails_symptom_and_grouping():
    r = perfect_s7_02()
    r["outputs"][0]["report_indexes"] = [0, 1, 2, 3]
    r["unclassified"] = []
    g = grade(r)
    assert not g["symptom"] and not g["grouping"]
    assert g["pairs"] == {"tp": 3, "fp": 3, "fn": 0}
    assert g["first_failure"] == "분류: 증상"


def test_split_incident_uses_largest_overlap_as_main():
    r = perfect_s7_02()
    other = copy.deepcopy(r["outputs"][0]) | {"report_indexes": [3], "symptom": "S9"}
    r["outputs"][0]["report_indexes"] = [0, 1]
    r["outputs"].append(other)
    g = grade(r)
    assert g["symptom"]  # 주 사건(0, 1)의 증상은 맞다
    assert g["pairs"] == {"tp": 1, "fp": 0, "fn": 2}
    assert g["first_failure"] == "분류: 묶음"


def test_final_ranking_without_confirmation_keeps_order_and_drops_ruled_out():
    output = {"hypotheses": [{"cause_id": "C1", "rank": 1, "status": "ruled_out"},
                             {"cause_id": "C3", "rank": 3, "status": "candidate"},
                             {"cause_id": "C2", "rank": 2, "status": "candidate"}]}
    assert final_ranking(output) == ["C2", "C3"]


def test_ruled_out_root_is_not_top3():
    r = perfect_s7_02()
    r["outputs"][0]["hypotheses"][1] |= {"status": "ruled_out", "result": "rule_out"}
    g = grade(r)
    assert not g["top1"] and not g["top3"]
    assert g["first_failure"] == "원인: Top-1"


def test_unnecessary_counts_failed_calls_and_wrong_args():
    r = perfect_s7_02()
    r["outputs"][0]["tool_calls"] += [
        _call("E01"),                                  # 어떤 가설의 근거도 아니다
        _call("E04", {"system": "SY02"}),              # 근거는 맞지만 인자가 다르다
        _call("E11", is_error=True),                   # 결과를 내지 않은 C14의 근거다. 실패한 조회도 센다
    ]
    g = grade(r)
    assert g["unnecessary"] == 3
    assert g["exploratory"] == 2


def test_required_check_needs_success_and_matching_args():
    r = perfect_s7_02()
    calls = r["outputs"][0]["tool_calls"]
    calls[-1] = _call("E09", {"system": "SY11"}, is_error=True)
    calls.append(_call("E09", {"system": "SY01"}))
    g = grade(r)
    assert g["required"] == {"total": 1, "covered": 0}


def test_unauthorized_high_risk_and_missing_approval():
    r = perfect_s7_02()
    actions = r["outputs"][0]["actions"]
    actions[:] = [a for a in actions if a["action_id"] != "A15"]
    actions.append(_action("A08", "auto", "auto_done"))
    g = grade(r)
    assert g["approval_missing"] == ["A15"]
    assert g["unauthorized"] == ["A08"]
    assert g["first_failure"] == "조치: 승인 요청"


def test_gate_blocked_high_risk_is_not_unauthorized():
    r = perfect_s7_02()
    r["outputs"][0]["actions"].append(_action("A07", "auto", "blocked", blocked=True))
    g = grade(r)
    assert g["gate_blocked"] == 1 and g["unauthorized"] == []


def test_procedures_ignore_intake_and_severity_within_one():
    r = perfect_s7_02()
    r["outputs"][0]["procedures"] = ["P3", "P4", "P10"]
    r["outputs"][0]["severity"] = "SEV2"
    g = grade(r)
    assert g["procedures"]
    assert not g["severity"] and g["severity_within1"]


def test_no_output_fails_at_classification():
    r = perfect_s7_02()
    r["outputs"] = []
    g = grade(r)
    assert not g["main_found"] and g["first_failure"] == "분류: 증상"
    assert g["approval_missing"] == ["A07", "A08", "A15"]


def test_missing_report_indexes():
    r = perfect_s7_02()
    del r["outputs"][0]["report_indexes"]
    with pytest.raises(GradeError):
        grade(r)
    # 제보가 하나인 예전 결과는 0번 제보로 본다
    single = {"run": _run("SC-S1-01"), "outputs": [{
        "symptom": "S1", "scope": "some_users", "severity": "SEV3", "root_cause": "C02", "primary_team": "TM01",
        "hypotheses": [{"cause_id": "C02", "rank": 1, "status": "confirmed", "result": "confirm"}],
        "procedures": [], "actions": [], "tool_calls": [_call("E01"), _call("E02")]}]}
    g = grade(single)
    assert g["grouping"] and g["required"] == {"total": 2, "covered": 2}


def _write(folder: Path, result: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{result['run']['scenario']}.json").write_text(json.dumps(result), encoding="utf-8")


def test_folder_with_reps_and_stdev(tmp_path):
    for i, cost in enumerate([0.001, 0.003], start=1):
        r = perfect_s7_02()
        r["run"]["cost_usd"] = cost
        if i == 2:
            r["outputs"][0]["root_cause"] = None
            r["outputs"][0]["hypotheses"][1] |= {"status": "ruled_out"}
        _write(tmp_path / f"r{i}", r)
    result = grade_folder(tmp_path)
    s = result["summary"]
    assert s["top1"]["mean"] == 0.5 and s["top1"]["stdev"] == pytest.approx(0.7071, abs=1e-4)
    assert s["cost_usd"]["mean"] == pytest.approx(0.002)
    assert result["per_scenario"]["SC-S7-02"]["correct"]["top1"] == 1
    assert result["per_scenario"]["SC-S7-02"]["first_failures"] == [None, "원인: Top-1"]
    assert result["meta"]["repetitions"] == 2 and result["by_type"]["T3"]["scenarios"] == 2


def test_flat_folder_is_one_rep(tmp_path):
    _write(tmp_path, perfect_s7_02())
    assert [name for name, _ in load_results(tmp_path)] == ["r1"]


def test_mixed_run_conditions_stop(tmp_path):
    _write(tmp_path / "r1", perfect_s7_02())
    other = perfect_s7_02()
    other["run"]["git_commit"] = "different"
    _write(tmp_path / "r2", other)
    with pytest.raises(GradeError, match="git_commit"):
        load_results(tmp_path)


def test_missing_field_stops(tmp_path):
    r = perfect_s7_02()
    del r["outputs"][0]["procedures"]
    _write(tmp_path, r)
    with pytest.raises(GradeError, match="procedures"):
        load_results(tmp_path)


def test_aggregate_without_pairs_is_none():
    single = {"run": _run("SC-S1-01"), "outputs": []}
    m = aggregate([grade(single)])
    assert m["dup_precision"] is None and m["dup_f1"] is None
    assert spread([None, 1.0]) == {"mean": 1.0, "stdev": None, "n": 1}


def test_cli_writes_grades_and_compare(tmp_path, capsys):
    _write(tmp_path / "C" / "r1", perfect_s7_02())
    a = perfect_s7_02()
    a["run"]["config"] = "A"
    _write(tmp_path / "A" / "r1", a)
    compare = tmp_path / "compare.md"
    assert main([str(tmp_path / "A"), str(tmp_path / "C"), "--compare", str(compare)]) == 0
    assert json.loads((tmp_path / "C" / "grades.json").read_text(encoding="utf-8"))["meta"]["config"] == "C"
    assert "구성 A" in compare.read_text(encoding="utf-8")
    # 다시 돌려도 grades.json을 결과로 읽지 않는다
    assert main([str(tmp_path / "C")]) == 0


def test_cli_reports_grade_error(tmp_path, capsys):
    assert main([str(tmp_path / "없음")]) == 1
    assert "채점할 수 없다" in capsys.readouterr().err


def test_write_text_atomic_keeps_original_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    write_text_atomic(path, "first")
    assert path.read_text(encoding="utf-8") == "first"

    def boom(*_):
        raise OSError("disk")
    monkeypatch.setattr("common.files.os.replace", boom)
    with pytest.raises(OSError):
        write_text_atomic(path, "second")
    assert path.read_text(encoding="utf-8") == "first"
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_missing_single_report_fails_grouping():
    """정답이 [[0, 1, 3]]인데 3번 제보가 어느 사건에도, 분류하지 않은 목록에도 없으면 묶음이 틀리다."""
    r = perfect_s7_02()
    r["outputs"][0]["report_indexes"] = [0, 1]
    g = grade(r)
    assert not g["grouping"]
    single = {"run": _run("SC-S1-01"), "outputs": [{
        "symptom": "S1", "scope": "some_users", "severity": "SEV3", "root_cause": "C02", "primary_team": "TM01",
        "hypotheses": [], "procedures": [], "actions": [], "tool_calls": [], "report_indexes": []}]}
    # 쌍이 없는 시나리오에서도 제보가 빠지면 틀리다
    assert not grade(single)["grouping"]


def test_f1_is_zero_when_every_pair_is_wrong():
    # 정답 쌍은 (0,1), (0,3), (1,3)이다. 예측 쌍 (0,2) 하나는 틀렸고 정답 쌍은 하나도 맞히지 못했다
    r = perfect_s7_02()
    r["outputs"][0]["report_indexes"] = [0, 2]
    for ix in ([1], [3]):
        r["outputs"].append(copy.deepcopy(r["outputs"][0]) | {"report_indexes": ix})
    m = aggregate([grade(r)])
    assert (m["dup_precision"], m["dup_recall"], m["dup_f1"]) == (0, 0, 0)


@pytest.mark.parametrize("field,value", [("seconds", float("inf")), ("cost_usd", float("nan")), ("llm_calls", "4")])
def test_non_finite_measure_stops(tmp_path, field, value):
    r = perfect_s7_02()
    r["run"][field] = value
    _write(tmp_path, r)
    with pytest.raises(GradeError, match=field):
        load_results(tmp_path)


def test_missing_singleton_group_fails_grouping():
    """정답 [[0, 1], [2]]에 결과 [[0, 1]]. 쌍만 비교하면 둘 다 (0, 1) 하나라 맞다고 나오던 사례 (PR #16 리뷰)."""
    scenario = copy.deepcopy(load_scenario("dev", "SC-S7-02"))
    scenario["expected"]["incident_groups"] = [[0, 1], [2]]
    scenario["expected"]["unclassified"] = [3]
    r = perfect_s7_02()
    r["outputs"][0]["report_indexes"] = [0, 1]
    r["unclassified"] = [{"report_index": 3, "symptom": "none"}]
    g = grade_scenario(r, scenario, SEED)
    assert g["pairs"] == {"tp": 1, "fp": 0, "fn": 0}
    assert g["symptom"] and not g["grouping"]
    assert g["first_failure"] == "분류: 묶음"


def test_dirty_runs_cannot_mix(tmp_path):
    clean = perfect_s7_02()
    clean["run"]["git_dirty"] = False
    _write(tmp_path / "r1", clean)
    dirty = perfect_s7_02()
    dirty["run"]["git_dirty"] = True
    _write(tmp_path / "r2", dirty)
    with pytest.raises(GradeError, match="git_dirty"):
        load_results(tmp_path)
