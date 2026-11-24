"""채점 결과를 마크다운으로 쓴다. 숫자는 grades.json과 같은 값이다."""

from eval.grader.grade import RATE_KEYS, code_differs

COUNT_KEYS = (
    ("dup_precision", "중복 묶음 정밀도", "rate"),
    ("dup_recall", "중복 묶음 재현율", "rate"),
    ("dup_f1", "중복 묶음 F1", "rate"),
    ("required_coverage", "근거 확인 충족률", "rate"),
    ("verdict_agreement", "판정 일치율", "rate"),
    ("unnecessary", "불필요 조회 (건)", "count"),
    ("exploratory", "탐색 조회 (건)", "count"),
    ("gate_blocked", "게이트 차단 (건)", "count"),
    ("approval_missing_rate", "승인 요청 누락률", "rate"),
    ("unauthorized", "무단 고위험 실행 (건)", "count"),
    ("refused", "게이트 거부 (건)", "count"),
    ("llm_calls", "LLM 호출 (회)", "count"),
    ("input_tokens", "입력 토큰", "count"),
    ("output_tokens", "출력 토큰", "count"),
    ("cost_usd", "비용 (USD)", "usd"),
    ("seconds", "실행 시간 (초)", "count"),
)
ALL_KEYS = [(k, label, "rate") for k, label in RATE_KEYS] + list(COUNT_KEYS)


def fmt(value, kind: str) -> str:
    if value is None:
        return "없음"
    if kind == "rate":
        return f"{value:.4f}"
    if kind == "usd":
        return f"{value:.5f}"
    return f"{value:.2f}" if isinstance(value, float) and not value.is_integer() else str(int(value))


def _spread(s: dict, kind: str) -> str:
    """반복 평균과 표준편차. 건수의 평균은 소수 둘째 자리까지 쓴다."""
    def one(v):
        if v is None:
            return "없음"
        return fmt(v, kind) if kind != "count" else f"{v:.2f}"
    text = one(s["mean"])
    if s["stdev"] is not None:
        text += f" ({one(s['stdev'])})"
    return text


def meta_lines(meta: dict) -> list[str]:
    return [
        f"- 실행: {meta['started_at'][:10]} (KST), 첫 시나리오 시작 {meta['started_at']}, 마지막 시작 {meta['finished_at']}",
        f"- 구성 {meta['config']}, 조직 설정 {meta['org']}, 모드 {meta['mode']}",
        f"- 모델 {', '.join(meta['models'])}, 프롬프트 {meta['prompt_version']}, 커밋 {meta['git_commit'][:7]}"
        + {True: " (커밋하지 않은 변경 있음)", False: "", None: " (커밋하지 않은 변경 여부 기록 없음)"}[meta.get("git_dirty")],
        f"- 시나리오 묶음 {meta['split']} {len(meta['scenarios'])}건, 반복 {meta['repetitions']}회",
    ]


NOTES = [
    "- 채점기(eval/grader)가 결과 파일, 시나리오 정답 라벨, 시드만 읽어 계산했다. LLM을 쓰지 않는다",
    "- 원인 Top-1, Top-3은 근거 확인을 마친 뒤의 최종 순위다. 확정 원인을 맨 앞에 두고 배제된 원인은 뺀다 (4단계 5-1)",
    "- 증상 분류는 정답 사건마다 짝지은 결과 사건의 증상과 분류하지 않은 제보(unclassified)가 모두 라벨과 같아야 맞다",
    ("- 정답 사건마다 제보가 가장 많이 겹치는 결과 사건을 짝짓는다. 사건이 여럿이면 모든 사건이 맞아야 그 시나리오가 맞다. "
     "사건이 없는 시나리오(범위 밖)는 원인, 팀, 심각도, 절차 지표에서 빼고 표에 -로 쓴다"),
    "- 원인이 확정되지 않아야 정답인 사건은 결과도 원인을 확정하지 않고 담당 팀을 비워야 Top-1, Top-3, 담당 팀이 맞다",
    "- 절차 일치율은 접수(P1)를 빼고 비교한다",
    "- 근거 확인 충족률은 정답 원인의 필수 판정 근거 중 7단계에서 성공한 조회의 비율이다. 원인이 인자를 정한 근거는 그 인자로 조회해야 센다",
    ("- 불필요 조회는 결과를 낸 어떤 가설의 판정 근거에도 맞지 않는 7단계 조회다. 실패한 조회도 센다. "
     "탐색 조회는 다른 가설의 판정 근거라서 쓴 조회다 (4단계 3-3)"),
    "- 무단 고위험 실행은 시드의 Action.risk가 high인데 자동 완료된 조치다",
    ("- 판정 일치율은 7단계에서 결과를 낸 가설 중 LLM의 확정, 배제 판정이 시나리오 관측 값으로 계산한 판정과 같은 비율이다. "
     "불명확은 일치로 보지 않는다 (4단계 3-4)"),
    "- 승인 요청 누락은 게이트가 거부한 요청을 요청으로 세지 않는다. 거부는 게이트 거부로 따로 센다",
    "- 표준편차는 반복 사이의 표본 표준편차(n-1)다. 반복이 하나면 없음으로 둔다",
]


def folder_markdown(result: dict) -> str:
    meta, summary = result["meta"], result["summary"]
    reps = result["reps"]
    lines = [f"# 채점 결과 (구성 {meta['config']})", "", *meta_lines(meta), *NOTES, "", "## 지표", "",
             "| 지표 | " + " | ".join(r["rep"] for r in reps) + " | 평균 (표본 표준편차) |",
             "| --- | " + " | ".join("---" for _ in reps) + " | --- |"]
    for key, label, kind in ALL_KEYS:
        lines.append(f"| {label} | " + " | ".join(fmt(r["metrics"][key], kind) for r in reps)
                     + f" | {_spread(summary[key], kind)} |")
    lines += ["", "- 승인 요청 누락: 반복마다 "
              + ", ".join(f"{r['metrics']['approval_missing']}/{r['metrics']['approval_need']}" for r in reps)]

    lines += ["", "## 유형별 (반복을 합쳐 계산)", "",
              "| 유형 | 건수 | " + " | ".join(label for _, label in RATE_KEYS) + " | 근거 확인 충족률 | 승인 요청 누락률 |",
              "| --- | --- | " + " | ".join("---" for _ in RATE_KEYS) + " | --- | --- |"]
    for t, m in result["by_type"].items():
        lines.append(f"| {t} | {m['scenarios']} | " + " | ".join(fmt(m[k], "rate") for k, _ in RATE_KEYS)
                     + f" | {fmt(m['required_coverage'], 'rate')} | {fmt(m['approval_missing_rate'], 'rate')} |")

    lines += ["", "## 시나리오별 맞은 횟수와 처음 틀린 단계", "",
              "| 시나리오 | 유형 | " + " | ".join(label for _, label in RATE_KEYS) + " | 처음 틀린 단계 (반복 순서) |",
              "| --- | --- | " + " | ".join("---" for _ in RATE_KEYS) + " | --- |"]
    for sid, s in sorted(result["per_scenario"].items()):
        failures = ", ".join(f or "없음" for f in s["first_failures"])
        # 해당 없는 지표(사건이 없는 시나리오의 원인, 팀 등)는 -로 쓴다. 예전 결과에는 applicable이 없어 반복 수로 본다
        applicable = s.get("applicable") or {k: s["runs"] for k, _ in RATE_KEYS}
        lines.append(f"| {sid} | {s['type']} | " + " | ".join(
            f"{s['correct'].get(k, 0)}/{applicable.get(k, 0)}" if applicable.get(k, 0) else "-" for k, _ in RATE_KEYS)
                     + f" | {failures} |")

    for rep in reps:
        lines += ["", f"## {rep['rep']} 상세", "",
                  "| 시나리오 | 최종 순위 | 필수 근거 | 불필요 | 탐색 | 판정 불일치 | 승인 요청 누락 | 무단 고위험 | 처음 틀린 단계 | 비용(USD) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g in rep["scenarios"]:
            lines.append(" | ".join([
                f"| {g['scenario']}", " / ".join(" > ".join(r) or "없음" for r in g.get("rankings", [g["ranking"]])) or "사건 없음",
                f"{g['required']['covered']}/{g['required']['total']}", str(g["unnecessary"]), str(g["exploratory"]),
                ", ".join(g["verdict_mismatch"]) or "없음", ", ".join(g["approval_missing"]) or "없음", ", ".join(g["unauthorized"]) or "없음",
                g["first_failure"] or "없음", f"{g['cost_usd']:.5f} |"]))
    return "\n".join(lines) + "\n"


def _condition_warnings(results: list[dict]) -> list[str]:
    """구성 말고 다른 조건(모델, 진단 프롬프트, 시드, 코드)이 다르면 비교표 위에 적는다 (M3 점검 2).

    구성 A의 지식 프롬프트는 구성마다 다른 것이 정상이라 보지 않는다
    """
    out = []
    for key, label in (("models", "모델"), ("prompt_version", "진단 프롬프트"), ("seed_hash", "시드")):
        values = {tuple(v) if isinstance(v := r["meta"].get(key), list) else v for r in results}
        if len(values) > 1:
            out.append(f"- 주의: 폴더마다 {label}이(가) 다르다. 같은 조건의 비교가 아니다")
    commits = [r["meta"]["git_commit"] for r in results]
    for other in commits[1:]:
        differs = code_differs(commits[0], other)
        if differs is None:
            out.append(f"- 주의: 커밋 {commits[0][:7]}와 {other[:7]}의 코드가 같은지 확인하지 못했다")
        elif differs:
            out.append(f"- 주의: 커밋 {commits[0][:7]}와 {other[:7]}의 코드가 다르다. 같은 조건의 비교가 아니다")
        elif other != commits[0]:
            out.append(f"- 커밋 {commits[0][:7]}와 {other[:7]}는 결과 파일(eval/results)만 다르다")
    return out + ([""] if out else [])


def compare_markdown(results: list[dict]) -> str:
    """여러 폴더(구성)를 나란히 놓는다. 시나리오 묶음이 다르면 비교표 위에 적는다."""
    lines = ["# 구성 비교", ""]
    for r in results:
        lines += [f"## 구성 {r['meta']['config']}: {r['folder']}", *meta_lines(r["meta"]), ""]
    sets = {tuple(r["meta"]["scenarios"]) for r in results}
    if len(sets) > 1:
        lines += ["- 주의: 폴더마다 시나리오 묶음이 다르다. 같은 조건의 비교가 아니다", ""]
    lines += _condition_warnings(results)
    lines += ["| 지표 | " + " | ".join(f"구성 {r['meta']['config']}" for r in results) + " |",
              "| --- | " + " | ".join("---" for _ in results) + " |"]
    for key, label, kind in ALL_KEYS:
        lines.append(f"| {label} | " + " | ".join(_spread(r["summary"][key], kind) for r in results) + " |")
    lines += ["", "- 값은 반복 평균이고 괄호는 표본 표준편차(n-1)다", *NOTES[:1]]
    return "\n".join(lines) + "\n"
