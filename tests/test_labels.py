"""정답 라벨의 사건 단위 형식 (eval/labels.py). 사건 여럿, 원인 미확정, 범위 밖 시나리오의 검증과 채점.

test 세트에 쓸 형식이다 (2026-11-25). dev 시나리오 두 건을 합치거나 고쳐 만든 가짜 시나리오로 확인한다.
"""

import copy

import pytest

from eval.grader.grade import aggregate, grade_scenario
from eval.labels import label_groups, label_incidents
from eval.validate import expected_procedures, load_scenarios, validate_scenario
from ontology.seed import load_seed

SEED = load_seed()
DEV = {s["id"]: s for s in load_scenarios("dev")}


def _incident(scenario: dict, reports: list[int]) -> dict:
    label = {k: v for k, v in scenario["expected"].items() if k not in ("incident_groups", "unclassified")}
    return label | {"reports": reports}


def mixed() -> dict:
    """결제 미지급(S1, C02)과 아이템 복사 의심(S5, C14)이 같은 저녁에 들어온 시나리오. 두 사건 모두 일부 유저다."""
    s1, s5 = copy.deepcopy(DEV["SC-S1-01"]), copy.deepcopy(DEV["SC-S5-01"])
    s5["reports"][0]["at"] = "2026-12-07T20:30+09:00"
    checks = [c for c in s1["environment"]["checks"] + s5["environment"]["checks"] if c["check"] != "E15"]
    checks.append({"check": "E15", "observations": {"scope": "some_users"},
                   "detail": "결제 미지급 문의 1건, 거래소 복사 의심 제보 3건이 들어왔다. 모두 일부 유저다"})
    return {
        "id": "SC-MIX-01", "type": "T3", "boundary": None, "source": "설계", "split": "test", "text_source": "테스트",
        "reports": s1["reports"] + s5["reports"],
        "environment": {"checks": checks, "changes": []},
        "expected": {"incidents": [_incident(s1, [0]), _incident(s5, [1])], "unclassified": []},
    }


def out_of_scope() -> dict:
    """증상 8개에 없는 사용법 문의 하나. 사건이 없고 제보는 분류하지 않아야 정답이다 (T6)."""
    return {
        "id": "SC-OOS-01", "type": "T6", "boundary": None, "source": "설계", "split": "test", "text_source": "테스트",
        "reports": [{"channel": "CH02", "reporter": "U9001", "at": "2026-12-07T20:14+09:00",
                     "text": "길드 가입은 어디서 하나요?"}],
        "environment": {"checks": [], "changes": []},
        "expected": {"incidents": [], "unclassified": [0]},
    }


def unconfirmed() -> dict:
    """결제도 지급도 정상이고 상품 설정 변경도 없다. S1의 후보가 모두 배제돼 원인을 확정하지 않아야 정답이다 (T5)."""
    s1 = copy.deepcopy(DEV["SC-S1-01"])
    s1["environment"]["checks"] = [c for c in s1["environment"]["checks"] if c["check"] != "E02"]
    procedures = expected_procedures(SEED, {"severity": "SEV3", "root_cause": None})
    incident = {
        "reports": [0], "symptom": "S1", "scope": "some_users", "severity": "SEV3",
        "root_cause": None, "ruled_out": ["C01", "C02", "C03"], "required_checks": [], "primary_team": None,
        "resolved_conditions": [], "procedures": sorted(procedures),
        "auto_actions": ["A01", "A02", "A03", "A04", "A05"], "approval_actions": [], "forbidden_actions": [],
    }
    return s1 | {"id": "SC-UNC-01", "type": "T5", "split": "test",
                 "expected": {"incidents": [incident], "unclassified": []}}


@pytest.mark.parametrize("make", [mixed, out_of_scope, unconfirmed])
def test_new_label_forms_are_valid(make):
    assert validate_scenario(SEED, make()) == []


def test_flat_labels_read_as_one_incident():
    expected = DEV["SC-S7-02"]["expected"]
    [incident] = label_incidents(expected)
    assert incident["reports"] == [0, 1, 3] and incident["root_cause"] == "C19"
    assert label_groups(expected) == [[0, 1, 3]]
    assert label_groups(mixed()["expected"]) == [[0], [1]]
    assert label_incidents(out_of_scope()["expected"]) == []


def test_unconfirmed_incident_opens_no_cause_procedures():
    # 원인이 없으면 원인 수정(P10)과 원인에 걸린 절차(보상 P6)를 열지 않는다. SEV3은 등급 절차도 없다
    assert expected_procedures(SEED, {"severity": "SEV3", "root_cause": None}) == set()
    assert expected_procedures(SEED, {"severity": "SEV2", "root_cause": None}) == {"P3", "P4"}


def _broken(make, mutate) -> dict:
    scenario = make()
    mutate(scenario)
    return scenario


@pytest.mark.parametrize("make,mutate,message", [
    # 같은 증상의 사건 둘은 한 환경에서 원인을 가를 수 없다
    (mixed, lambda s: s["expected"]["incidents"][1].update(symptom="S1"), "증상이 겹친다"),
    # 라벨을 사건 안과 바깥에 같이 적으면 모호하다
    (mixed, lambda s: s["expected"].update(root_cause="C02"), "사건 안에만"),
    (mixed, lambda s: s["expected"].update(incident_groups=[[0], [1]]), "사건 안에만"),
    # 제보는 사건과 분류하지 않을 제보에 한 번씩
    (mixed, lambda s: s["expected"]["incidents"][1].update(reports=[0, 1]), "한 번씩 담지"),
    (mixed, lambda s: s["expected"]["incidents"][1].update(reports=[]), "비지 않은 목록"),
    # 사건마다 규칙 검사를 한다. 어느 사건인지 메시지에 붙는다
    (mixed, lambda s: s["expected"]["incidents"][1].update(primary_team="TM03"), "사건 1: primary_team"),
    # E15는 답이 하나라 사건마다 범위가 같아야 한다
    (mixed, lambda s: s["expected"]["incidents"][1].update(scope="all_service", severity="SEV1"), "E15 관측 값"),
    (mixed, lambda s: s["expected"].update(incidents="사건"), "incidents는 사건 라벨의 목록"),
    # 원인 미확정: 환경에서 원인이 확정되면 안 되고, 담당 팀과 판정 근거는 비운다
    (unconfirmed, lambda s: s["environment"]["checks"].append(
        {"check": "E02", "observations": {"delivery": "absent"}}), "확정되지 않아야 정답인데"),
    (unconfirmed, lambda s: s["expected"]["incidents"][0].update(primary_team="TM03"), "primary_team은 null"),
    (unconfirmed, lambda s: s["expected"]["incidents"][0].update(ruled_out=["C01"]), "후보 전체"),
    (unconfirmed, lambda s: s["expected"]["incidents"][0].update(required_checks=["E01"]), "required_checks"),
    (unconfirmed, lambda s: s["expected"]["incidents"][0].update(procedures=["P10"]), "procedures"),
    # 범위 밖: 분류하지 않을 제보가 빠지면 안 된다
    (out_of_scope, lambda s: s["expected"].update(unclassified=[]), "한 번씩 담지"),
])
def test_new_label_forms_report_errors(make, mutate, message):
    errors = validate_scenario(SEED, _broken(make, mutate))
    assert errors and any(message in e for e in errors), errors


def test_flat_form_keeps_one_group_and_string_cause():
    """사건 하나 형식은 예전 그대로다. 원인 미확정과 여러 묶음은 사건 여럿 형식으로만 적는다."""
    flat = copy.deepcopy(DEV["SC-S1-01"])
    flat["expected"]["root_cause"] = None
    assert any("expected.root_cause은 문자열이어야" in e for e in validate_scenario(SEED, flat))
    two = copy.deepcopy(DEV["SC-S7-02"])
    two["expected"]["incident_groups"] = [[0, 1], [3]]
    assert any("묶음 하나" in e for e in validate_scenario(SEED, two))


# ---------- 채점 ----------

def _run(scenario_id: str) -> dict:
    return {"scenario": scenario_id, "split": "test", "config": "C", "org": "org-a", "mode": "eval",
            "started_at": "2026-12-07T21:00:00+09:00", "seconds": 1.0, "models": ["m"], "prompt_version": "p",
            "git_commit": "abc1234", "llm_calls": 1, "usage": {"input": 1, "output": 1}, "cost_usd": 0.001}


def _output(label: dict, *, root: str | None = "label", team: str | None = "label") -> dict:
    root = label["root_cause"] if root == "label" else root
    team = label["primary_team"] if team == "label" else team
    hypotheses = [{"cause_id": c, "rank": i + 2, "status": "ruled_out", "result": "rule_out"}
                  for i, c in enumerate(label["ruled_out"])]
    if root:
        hypotheses.append({"cause_id": root, "rank": 1, "status": "confirmed", "result": "confirm"})
    return {
        "symptom": label["symptom"], "report_indexes": label["reports"], "scope": label["scope"],
        "severity": label["severity"], "hypotheses": hypotheses, "root_cause": root, "primary_team": team,
        "procedures": ["P1", *label["procedures"]],
        "actions": [{"action_id": a, "requested_mode": "approval", "status": "pending_approval"}
                    for a in label["approval_actions"]],
        "tool_calls": [],
    }


def _result(scenario: dict, outputs: list[dict], unclassified: list[int] = ()) -> dict:
    return {"run": _run(scenario["id"]), "outputs": outputs,
            "unclassified": [{"report_index": i} for i in unclassified]}


def test_mixed_incidents_are_matched_and_all_must_be_right():
    s = mixed()
    labels = s["expected"]["incidents"]
    # 결과 사건 순서가 라벨과 달라도 제보로 짝짓는다
    g = grade_scenario(_result(s, [_output(labels[1]), _output(labels[0])]), s, SEED)
    assert g["first_failure"] is None and g["main_found"] and g["grouping"]
    assert g["top1"] and g["team"] and g["procedures"] and g["symptom"]
    assert g["rankings"] == [["C02"], ["C14"]]
    assert g["approval_need"] == len(labels[0]["approval_actions"]) + len(labels[1]["approval_actions"])

    wrong = _output(labels[1], root="C13")
    g = grade_scenario(_result(s, [_output(labels[0]), wrong]), s, SEED)
    assert not g["top1"] and g["first_failure"] == "원인: Top-1"


def test_mixed_incidents_merged_into_one_fail_grouping():
    s = mixed()
    labels = s["expected"]["incidents"]
    merged = _output(labels[0]) | {"report_indexes": [0, 1]}
    g = grade_scenario(_result(s, [merged]), s, SEED)
    assert not g["main_found"] and not g["grouping"]
    # 짝이 없는 사건은 틀린 것으로 본다. 그 사건의 승인 요청은 모두 누락이다
    assert not g["top1"] and set(labels[1]["approval_actions"]) <= set(g["approval_missing"])


def test_out_of_scope_scenario_is_left_out_of_cause_metrics():
    s = out_of_scope()
    right = grade_scenario(_result(s, [], unclassified=[0]), s, SEED)
    assert right["symptom"] and right["grouping"] and right["first_failure"] is None
    assert right["top1"] is None and right["team"] is None and right["procedures"] is None
    assert right["required"] == {"total": 0, "covered": 0} and right["approval_need"] == 0

    # 범위 밖 제보로 사건을 열면 분류에서 틀린다
    label = _incident(DEV["SC-S1-01"], [0])
    wrong = grade_scenario(_result(s, [_output(label)]), s, SEED)
    assert not wrong["symptom"] and wrong["first_failure"] == "분류: 증상"

    # 해당 없는 시나리오는 분모에서 빠진다. 다른 시나리오의 Top-1만 센다
    other = mixed()
    labels = other["expected"]["incidents"]
    ok = grade_scenario(_result(other, [_output(labels[0]), _output(labels[1])]), other, SEED)
    m = aggregate([right, ok])
    assert m["scenarios"] == 2 and m["top1"] == 1.0 and m["symptom"] == 1.0


def test_unconfirmed_incident_needs_no_confirmation_and_no_team():
    s = unconfirmed()
    label = s["expected"]["incidents"][0]
    g = grade_scenario(_result(s, [_output(label)]), s, SEED)
    assert g["top1"] and g["top3"] and g["team"] and g["first_failure"] is None
    assert g["pre_rank_top1"] is None and g["required"] == {"total": 0, "covered": 0}

    # 근거 없이 원인을 확정하면 틀린다
    g = grade_scenario(_result(s, [_output(label, root="C02", team="TM03")]), s, SEED)
    assert not g["top1"] and not g["team"] and g["first_failure"] == "원인: Top-1"
