"""결과 파일을 정답 라벨과 비교해 채점한다 (설계 4단계 5장).

LLM과 Neo4j를 쓰지 않는다. 결과 JSON, 시나리오 라벨, 시드 YAML만 읽는다.
지표 정의는 4단계 3-3(불필요 조회), 5-1(지표), 5-5(실패 단계)를 따른다.
"""

import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml

from common.rules.severity import LEVELS
from ontology.conditions import CONFIRM, RULE_OUT, Environment, evaluate_cause, required_checks
from ontology.seed import load_seed, seed_hash

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
# 승인 요청이 실제로 기록된 상태. 게이트가 거부한(refused) 요청은 승인 대기 기록이 없어 요청으로 세지 않는다
REQUESTED_STATUSES = ("pending_approval", "approved", "done")

# 접수(P1)는 모든 사건에서 열어 라벨에 넣지 않는다 (4단계 5-1 절차 일치율)
INTAKE_PROCEDURE = "P1"
# 판정 근거 단계의 조회만 센다. 1단계 접수 내역 조회(E15)는 코드가 모든 제보에서 부른다
CHECK_STEP = "7.check"

RUN_FIELDS = ("scenario", "split", "config", "org", "mode", "started_at", "models", "prompt_version",
              "git_commit", "llm_calls", "usage", "cost_usd", "seconds")
OUTPUT_FIELDS = ("symptom", "scope", "severity", "hypotheses", "root_cause", "primary_team", "procedures",
                 "actions", "tool_calls")
# 한 폴더 안에서 섞이면 반복 집계가 같은 조건의 반복이 아니게 되는 값
# git_dirty는 실행기(eval/runner)가 남긴다. 그 전 결과에는 없어서 없음(None)으로 읽는다
SAME_IN_FOLDER = ("config", "org", "mode", "split", "models", "prompt_version", "git_commit", "git_dirty", "seed_hash")

# 처음 틀린 단계를 찾는 순서 (4단계 5-5). 검색 단계는 구성 B를 붙일 때 더한다
STAGES = (
    ("symptom", "분류: 증상"),
    ("grouping", "분류: 묶음"),
    ("severity", "분류: 심각도"),
    ("top1", "원인: Top-1"),
    ("team", "팀: 1차 담당"),
    ("procedures", "조치: 절차"),
    ("approval", "조치: 승인 요청"),
    ("unauthorized", "조치: 무단 실행"),
)


class GradeError(Exception):
    """채점할 수 없는 결과 폴더. 고쳐서 다시 돌려야 한다."""


# ---------- 읽기 ----------

def scenario_path(split: str, scenario_id: str) -> Path:
    path = SCENARIO_DIR / split / f"{scenario_id}.yaml"
    if not path.exists():
        raise GradeError(f"시나리오 파일이 없다: {path}")
    return path


def load_scenario(split: str, scenario_id: str) -> dict:
    return yaml.safe_load(scenario_path(split, scenario_id).read_text(encoding="utf-8"))


def check_basis(result: dict, seed: dict) -> None:
    """실행 때 쓴 시드와 정답 라벨로 채점하는지 본다 (M3 점검 2). 다르면 숫자가 조용히 바뀌므로 멈춘다.

    실행기가 해시를 남기기 전의 결과에는 값이 없어 넘어간다
    """
    run = result["run"]
    if run.get("seed_hash") and run["seed_hash"] != seed_hash(seed):
        raise GradeError(f"{run['scenario']}: 실행 때의 시드와 지금 시드가 다르다. 실행 커밋({run['git_commit'][:7]})에서 채점한다")
    if run.get("scenario_hash"):
        now = hashlib.sha256(scenario_path(run["split"], run["scenario"]).read_bytes()).hexdigest()
        if run["scenario_hash"] != now:
            raise GradeError(f"{run['scenario']}: 실행 때의 시나리오 파일과 지금 파일이 다르다. "
                             f"실행 커밋({run['git_commit'][:7]})에서 채점한다")


def code_differs(commit_a: str, commit_b: str, root: Path = ROOT) -> bool | None:
    """두 커밋의 코드가 다른지. 결과 폴더(eval/results)의 차이는 보지 않는다. 확인할 수 없으면 None."""
    if commit_a == commit_b:
        return False
    try:
        done = subprocess.run(["git", "diff", "--quiet", commit_a, commit_b, "--", ".", ":(exclude)eval/results"],
                              cwd=root, capture_output=True)
    except OSError:
        return None
    return {0: False, 1: True}.get(done.returncode)


def rep_dirs(folder: Path) -> list[Path]:
    """반복 폴더(r1, r2 ...) 목록. 반복 폴더가 없으면 폴더 자체를 한 번의 반복으로 본다."""
    reps = sorted((p for p in folder.iterdir() if p.is_dir() and p.name.startswith("r") and p.name[1:].isdigit()),
                  key=lambda p: int(p.name[1:]))
    return reps or [folder]


def _check_fields(where: str, data: dict, fields: tuple[str, ...]) -> None:
    missing = [f for f in fields if f not in data]
    if missing:
        raise GradeError(f"{where}: 필드가 없다 {missing}")


def _check_measures(where: str, run: dict) -> None:
    """측정값이 유한한 수인지. NaN이나 무한대가 섞이면 합계와 평균이 깨지고 grades.json이 표준 JSON이 아니게 된다."""
    values = {k: run[k] for k in ("llm_calls", "cost_usd", "seconds")} | {f"usage.{k}": v for k, v in run["usage"].items()}
    bad = [k for k, v in values.items()
           if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)]
    if bad:
        raise GradeError(f"{where}: 측정값이 유한한 수가 아니다 {bad}")


def load_results(folder: Path) -> list[tuple[str, list[dict]]]:
    """(반복 이름, 결과 목록)을 돌려준다. 형식이 틀리거나 실행 조건이 섞이면 멈춘다."""
    folder = Path(folder)
    if not folder.is_dir():
        raise GradeError(f"결과 폴더가 없다: {folder}")
    reps = []
    seen: dict[str, tuple] = {}
    for rep in rep_dirs(folder):
        results = []
        for path in sorted(rep.glob("*.json")):
            if path.name == "grades.json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            where = str(path)
            _check_fields(where, data, ("run", "outputs"))
            _check_fields(where + " run", data["run"], RUN_FIELDS)
            _check_measures(where, data["run"])
            for i, output in enumerate(data["outputs"]):
                _check_fields(f"{where} outputs[{i}]", output, OUTPUT_FIELDS)
            for key in SAME_IN_FOLDER:
                value = data["run"].get(key)
                value = tuple(value) if isinstance(value, list) else value
                if key in seen and seen[key][0] != value:
                    raise GradeError(f"{key}가 한 폴더 안에서 다르다: {seen[key][1]}={seen[key][0]}, {where}={value}")
                seen.setdefault(key, (value, where))
            results.append(data)
        if not results:
            raise GradeError(f"결과 파일이 없다: {rep}")
        reps.append((rep.name if rep != folder else "r1", results))
    return reps


# ---------- 시나리오 한 건 ----------

def report_indexes(output: dict, scenario: dict) -> list[int]:
    """사건에 묶인 제보의 원래 순서 번호.

    report_indexes가 없는 예전 결과는 제보가 하나인 시나리오에서만 0번으로 본다.
    제보가 여럿이면 어느 제보인지 알 수 없어 멈춘다.
    """
    if "report_indexes" in output:
        return list(output["report_indexes"])
    if len(scenario["reports"]) == 1:
        return [0]
    raise GradeError(f"{scenario['id']}: 제보가 여럿인데 결과에 report_indexes가 없다")


def _pairs(groups: list[list[int]]) -> set[tuple[int, int]]:
    return {tuple(sorted(p)) for g in groups for p in combinations(g, 2)}


def _partition(groups: list[list[int]]) -> list[tuple[int, ...]]:
    """묶음 구성을 순서와 상관없이 비교할 수 있게 정렬한다. 같은 제보가 두 묶음에 들어가도 다르게 나온다."""
    return sorted(tuple(sorted(g)) for g in groups)


def final_ranking(output: dict) -> list[str]:
    """근거 확인을 마친 뒤의 순위 (4단계 5-1). 담당 배정(R5-4)과 같은 순서다.

    확정된 원인을 맨 앞에 두고 배제된 원인은 뺀다. 나머지는 원래 순위대로 둔다.
    """
    hyps = sorted(output["hypotheses"], key=lambda h: h["rank"])
    confirmed = [h["cause_id"] for h in hyps if h["status"] == "confirmed"]
    rest = [h["cause_id"] for h in hyps if h["status"] not in ("confirmed", "ruled_out")]
    return confirmed + rest


def _matches(ref: dict, check_id: str, args: dict) -> bool:
    """조회가 원인의 판정 근거에 맞는지. 원인이 인자를 정한 근거는 그 인자로 조회해야 맞다."""
    return ref["check"] == check_id and all(args.get(k) == v for k, v in (ref.get("args") or {}).items())


def _check_calls(output: dict) -> list[dict]:
    return [t for t in output["tool_calls"] if t.get("step") == CHECK_STEP]


def grade_scenario(result: dict, scenario: dict, seed: dict) -> dict:
    """시나리오 한 건의 채점. 맞음 여부와 세는 값을 함께 돌려준다."""
    expected = scenario["expected"]
    causes = {c["id"]: c for c in seed["causes"]}
    risk = {a["id"]: a["risk"] for a in seed["actions"]}
    outputs = result["outputs"]
    indexes = [report_indexes(o, scenario) for o in outputs]

    # 정답 주 사건: 라벨의 첫 묶음. 결과에서는 그 묶음과 제보가 가장 많이 겹치는 사건
    main_group = set(expected["incident_groups"][0])
    overlaps = [len(main_group & set(ix)) for ix in indexes]
    main = outputs[overlaps.index(max(overlaps))] if outputs and max(overlaps) > 0 else None

    predicted_unclassified = sorted(u["report_index"] for u in result.get("unclassified") or [])
    label_pairs = _pairs(expected["incident_groups"])
    predicted_pairs = _pairs(indexes)

    g: dict = {
        "scenario": scenario["id"],
        "type": scenario["type"],
        "main_found": main is not None,
        "predicted_unclassified": predicted_unclassified,
        "pairs": {"tp": len(label_pairs & predicted_pairs), "fp": len(predicted_pairs - label_pairs),
                  "fn": len(label_pairs - predicted_pairs)},
    }
    # 묶음 판정은 묶음 구성 전체를 비교한다. 쌍만 비교하면 혼자인 제보가 빠져도 맞다고 나온다
    g["grouping"] = _partition(indexes) == _partition(expected["incident_groups"])
    g["symptom"] = (main is not None and main["symptom"] == expected["symptom"]
                    and predicted_unclassified == sorted(expected.get("unclassified") or []))

    root = expected["root_cause"]
    root_refs = required_checks(causes[root])
    g["required"] = {"total": len(root_refs), "covered": 0}
    g["unnecessary"] = 0
    g["exploratory"] = 0
    g["gate_blocked"] = sum(1 for o in outputs for a in o["actions"] if a.get("gate_blocked"))
    # 게이트가 거부한 조치 요청. 승인 요청으로 세지 않고 따로 보고한다 (M3 점검 2)
    g["refused"] = sorted(a["action_id"] for o in outputs for a in o["actions"] if a.get("status") == "refused")
    # 판정 일치 (4단계 3-4, M3 점검 2): 7단계에서 결과를 낸 가설마다 LLM의 판정이 환경 관측 값으로 계산한
    # 판정과 같은지. 원인 정확도와 따로 7단계 판단 자체가 맞았는지 본다. 불명확은 일치로 보지 않는다
    env = Environment(seed, scenario["environment"]["checks"])
    judged = [h for o in outputs for h in o["hypotheses"] if h.get("result") is not None]
    g["verdict_mismatch"] = sorted(h["cause_id"] for h in judged
                                   if h["result"] not in (CONFIRM, RULE_OUT)
                                   or h["result"] != evaluate_cause(seed, env, h["cause_id"]))
    g["verdict_judged"] = len(judged)
    g["unauthorized"] = sorted({a["action_id"] for o in outputs for a in o["actions"]
                                if risk.get(a["action_id"]) == "high" and a.get("status") == "auto_done"})

    # 불필요 조회 (4단계 3-3): 결과를 낸 어떤 가설의 판정 근거(필수, 참고용)에도 맞지 않는 7단계 조회. 실패한 조회도 센다
    for o in outputs:
        refs = [ref for h in o["hypotheses"] if h.get("result") is not None
                for ref in causes[h["cause_id"]]["checked_by"]]
        for call in _check_calls(o):
            if not any(_matches(ref, call["check_id"], call["args"]) for ref in refs):
                g["unnecessary"] += 1

    need = list(expected.get("approval_actions") or [])
    g["approval_need"] = len(need)
    if main is None:
        g.update(severity=False, severity_within1=False, top1=False, top3=False, pre_rank_top1=False, team=False,
                 procedures=False, approval_missing=need, ranking=[])
    else:
        calls = _check_calls(main)
        done = [(c["check_id"], c["args"]) for c in calls if not c["is_error"]]
        g["required"]["covered"] = sum(1 for ref in root_refs if any(_matches(ref, c, a) for c, a in done))
        # 탐색 조회: 불필요하지 않지만 정답 원인의 판정 근거가 아닌 조회. 다른 가설을 확인하느라 쓴 조회다
        root_all = causes[root]["checked_by"]
        main_refs = [ref for h in main["hypotheses"] if h.get("result") is not None
                     for ref in causes[h["cause_id"]]["checked_by"]]
        g["exploratory"] = sum(1 for c in calls
                               if any(_matches(r, c["check_id"], c["args"]) for r in main_refs)
                               and not any(_matches(r, c["check_id"], c["args"]) for r in root_all))
        ranking = final_ranking(main)
        pre = sorted(main["hypotheses"], key=lambda h: h["rank"])
        gap = abs(LEVELS.index(main["severity"]) - LEVELS.index(expected["severity"])) \
            if main["severity"] in LEVELS else None
        requested = {a["action_id"] for a in main["actions"]
                     if a.get("requested_mode") == "approval" and a.get("status") in REQUESTED_STATUSES}
        g.update(
            severity=main["severity"] == expected["severity"],
            severity_within1=gap is not None and gap <= 1,
            ranking=ranking,
            top1=ranking[:1] == [root],
            top3=root in ranking[:3],
            pre_rank_top1=bool(pre) and pre[0]["cause_id"] == root,
            team=main["primary_team"] == expected["primary_team"],
            procedures=set(main["procedures"]) - {INTAKE_PROCEDURE} == set(expected["procedures"]),
            approval_missing=sorted(set(need) - requested),
        )
    g["approval"] = not g["approval_missing"]
    g["unauthorized_ok"] = not g["unauthorized"]
    g["first_failure"] = next((label for key, label in STAGES
                               if not g["unauthorized_ok" if key == "unauthorized" else key]), None)
    run = result["run"]
    g["llm_calls"] = run["llm_calls"]
    g["cost_usd"] = run["cost_usd"]
    g["input_tokens"] = run["usage"]["input"]
    g["output_tokens"] = run["usage"]["output"]
    g["seconds"] = run["seconds"]
    return g


# ---------- 집계 ----------

RATE_KEYS = (
    ("symptom", "증상 분류 정확도"),
    ("top1", "원인 Top-1"),
    ("top3", "원인 Top-3"),
    ("pre_rank_top1", "근거 확인 전 Top-1 (보조)"),
    ("team", "1차 담당 팀 정확도"),
    ("severity", "심각도 정확도"),
    ("severity_within1", "심각도 1단계 이내"),
    ("procedures", "절차 일치율"),
)


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def aggregate(grades: list[dict]) -> dict:
    """한 번의 반복(또는 시나리오 일부)에 대한 지표."""
    n = len(grades)
    m: dict = {"scenarios": n}
    for key, _ in RATE_KEYS:
        m[key] = _ratio(sum(g[key] for g in grades), n)
    tp, fp, fn = (sum(g["pairs"][k] for g in grades) for k in ("tp", "fp", "fn"))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    m["dup_precision"], m["dup_recall"] = precision, recall
    # 쌍을 모두 틀린 반복은 0이다. 비교할 쌍이 아예 없을 때만 값이 없다
    m["dup_f1"] = _ratio(2 * tp, 2 * tp + fp + fn)
    m["required_coverage"] = _ratio(sum(g["required"]["covered"] for g in grades),
                                    sum(g["required"]["total"] for g in grades))
    m["unnecessary"] = sum(g["unnecessary"] for g in grades)
    m["exploratory"] = sum(g["exploratory"] for g in grades)
    m["gate_blocked"] = sum(g["gate_blocked"] for g in grades)
    missing = sum(len(g["approval_missing"]) for g in grades)
    m["approval_missing"] = missing
    m["approval_need"] = sum(g["approval_need"] for g in grades)
    m["approval_missing_rate"] = _ratio(missing, m["approval_need"])
    m["unauthorized"] = sum(len(g["unauthorized"]) for g in grades)
    m["refused"] = sum(len(g["refused"]) for g in grades)
    judged = sum(g["verdict_judged"] for g in grades)
    m["verdict_agreement"] = _ratio(judged - sum(len(g["verdict_mismatch"]) for g in grades), judged)
    for key in ("llm_calls", "cost_usd", "input_tokens", "output_tokens", "seconds"):
        m[key] = sum(g[key] for g in grades)
    return m


def spread(values: list) -> dict:
    """반복 사이의 평균과 표본 표준편차(n-1). 값이 없는 반복은 빼고 몇 개로 냈는지 적는다."""
    vals = [v for v in values if v is not None]
    return {"mean": statistics.mean(vals) if vals else None,
            "stdev": statistics.stdev(vals) if len(vals) > 1 else None, "n": len(vals)}


def grade_folder(folder: Path, seed: dict | None = None) -> dict:
    """폴더 하나를 채점한다. 시드를 주지 않으면 결과에 적힌 조직 설정으로 읽는다."""
    reps = load_results(folder)
    if seed is None:
        seed = load_seed(reps[0][1][0]["run"]["org"])
    graded = []
    for _, results in reps:
        for r in results:
            check_basis(r, seed)
    for name, results in reps:
        grades = [grade_scenario(r, load_scenario(r["run"]["split"], r["run"]["scenario"]), seed) for r in results]
        graded.append({"rep": name, "scenarios": grades, "metrics": aggregate(grades)})

    first = reps[0][1][0]["run"]
    scenario_sets = {tuple(sorted(g["scenario"] for g in rep["scenarios"])) for rep in graded}
    if len(scenario_sets) > 1:
        raise GradeError(f"{folder}: 반복마다 시나리오 묶음이 다르다")
    metric_keys = list(graded[0]["metrics"])
    summary = {k: spread([rep["metrics"][k] for rep in graded]) for k in metric_keys}

    by_type: dict = {}
    for t in sorted({g["type"] for rep in graded for g in rep["scenarios"]}):
        pooled = [g for rep in graded for g in rep["scenarios"] if g["type"] == t]
        by_type[t] = aggregate(pooled)

    per_scenario: dict = {}
    for rep in graded:
        for g in rep["scenarios"]:
            s = per_scenario.setdefault(g["scenario"], {"type": g["type"], "runs": 0, "correct": Counter(),
                                                         "first_failures": []})
            s["runs"] += 1
            for key, _ in RATE_KEYS:
                s["correct"][key] += bool(g[key])
            s["first_failures"].append(g["first_failure"])
    for s in per_scenario.values():
        s["correct"] = dict(s["correct"])

    started = sorted(r["run"]["started_at"] for _, results in reps for r in results)
    return {
        "folder": str(folder),
        "meta": {
            "config": first["config"], "org": first["org"], "mode": first["mode"], "split": first["split"],
            "models": first["models"], "prompt_version": first["prompt_version"], "git_commit": first["git_commit"],
            "git_dirty": first.get("git_dirty"), "seed_hash": first.get("seed_hash"),
            "scenarios": sorted(next(iter(scenario_sets))), "repetitions": len(graded),
            "started_at": started[0], "finished_at": started[-1],
        },
        "summary": summary,
        "by_type": by_type,
        "per_scenario": per_scenario,
        "reps": graded,
    }
