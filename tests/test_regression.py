"""배포 전 회귀 테스트의 통과 기준 확인 (eval/regression.py, 5단계 3-2). LLM 없이 가짜 채점 결과로 본다."""

import copy
import json
import shutil

import pytest
import yaml

from eval import regression
from eval.regression import RegressionError, failures, load_config, write_baseline
from eval.validate import load_scenarios

CONFIG = load_config()


def _grades(top1=1.0, approval_missing=0, unauthorized=0, **meta) -> dict:
    metrics = {"top1": top1, "approval_missing": approval_missing, "unauthorized": unauthorized, "cost_usd": 0.04}
    return {"meta": {"config": "C", "org": "org-a", "split": "dev", "scenarios": list(CONFIG["scenarios"]),
                     "repetitions": 1, "git_commit": "abc1234def", "git_dirty": False, "models": ["claude-haiku-5-5"],
                     "started_at": "2026-11-26T16:00:00+09:00", "prompt_version": "diagnosis-v7", "seed_hash": "s"}
            | meta,
            "reps": [{"metrics": metrics}],
            "per_scenario": {sid: {"first_failures": [None]} for sid in CONFIG["scenarios"]}}


def _config(top1=1.0) -> dict:
    return copy.deepcopy(CONFIG) | {"baseline": {"top1": top1, "git_commit": "abc1234def"}}


def test_scenarios_cover_every_symptom_and_type_except_t7():
    """증상 8개와 T1~T6이 모두 들어 있다. T7은 test 세트라 넣지 않는다."""
    from eval.labels import label_incidents

    picked = [s for s in load_scenarios("dev") if s["id"] in CONFIG["scenarios"]]
    assert len(picked) == len(CONFIG["scenarios"]) == 10
    assert {s["type"] for s in picked} == {"T1", "T2", "T3", "T4", "T5", "T6"}
    assert {i["symptom"] for s in picked for i in label_incidents(s["expected"])} == {f"S{n}" for n in range(1, 9)}
    assert all(s["split"] == "dev" for s in picked)


@pytest.mark.parametrize("grades, config, expected", [
    (_grades(), _config(), []),
    (_grades(top1=0.9), _config(), []),  # 10건 중 1건까지 허용
    (_grades(top1=0.8), _config(), ["원인 Top-1"]),
    (_grades(top1=None), _config(), ["원인 Top-1"]),
    (_grades(approval_missing=1), _config(), ["승인 요청 누락"]),
    (_grades(unauthorized=2), _config(), ["무단 고위험 실행"]),
    (_grades(), copy.deepcopy(CONFIG) | {"baseline": None}, ["기준 값이 없다"]),
])
def test_failures(grades, config, expected):
    found = failures(grades, config)
    assert len(found) == len(expected)
    assert all(word in line for word, line in zip(expected, found))


@pytest.mark.parametrize("meta", [{"config": "A"}, {"org": "org-b"}, {"repetitions": 3}, {"scenarios": ["SC-S1-01"]}])
def test_result_with_other_conditions_is_refused(tmp_path, meta):
    (tmp_path / "grades.json").write_text(json.dumps(_grades(**meta)), encoding="utf-8")
    with pytest.raises(RegressionError, match="다른 조건"):
        regression._grades(tmp_path, CONFIG)


def test_check_command_exit_code(tmp_path, monkeypatch, capsys):
    path = tmp_path / "regression.yaml"
    path.write_text(yaml.safe_dump(_config(), allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(regression, "CONFIG_PATH", path)
    (tmp_path / "grades.json").write_text(json.dumps(_grades(top1=0.7)), encoding="utf-8")
    assert regression.main(["check", str(tmp_path)]) == 1
    assert "회귀 테스트 실패 1건" in capsys.readouterr().out
    (tmp_path / "grades.json").write_text(json.dumps(_grades()), encoding="utf-8")
    assert regression.main(["check", str(tmp_path)]) == 0


def test_write_baseline_keeps_comments_and_refuses_dirty(tmp_path):
    path = tmp_path / "regression.yaml"
    shutil.copy(regression.CONFIG_PATH, path)
    write_baseline(_grades(top1=0.9), path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# 배포 전 회귀 테스트 설정")
    assert yaml.safe_load(text)["baseline"]["top1"] == 0.9
    assert yaml.safe_load(text)["scenarios"] == CONFIG["scenarios"]
    with pytest.raises(RegressionError, match="커밋하지 않은"):
        write_baseline(_grades(git_dirty=True), path)
