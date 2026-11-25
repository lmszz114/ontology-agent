"""승인된 지식 후보를 시드에 반영한다 (A17, 설계 3단계 R8-7, M4 PR 5).

흐름: 후보 검수 승인 -> A17 요청(승인 게이트, 승인 대기) -> 승인권자 전원 승인 -> 이 모듈이 시드 YAML을 고치고 검증한 뒤
지금 브랜치에 커밋 -> 개발용 Neo4j에 다시 적재 -> 실행 기록 완료(커밋 남김).

자동으로 반영하는 것 (M4 PR 5 결정 2)
- existing: 시드는 그대로 두고 출처(provenance.yaml)만 더한다
- new 중 OWNS(scope가 있을 때), DEPENDS_ON, OBSERVED_IN, MAY_BE_CAUSED_BY(prior가 있을 때), 별칭
사람이 직접 반영하는 것: CHECKED_BY, RESOLVED_BY, LOCATED_IN의 new(판정 값 when 같은 문서에서 뽑을 수 없는 값이 필요하다),
모든 conflict(기존 사실을 바꿔야 한다), 새 엔티티 후보(노드 id와 필수 속성이 없다). 분류 규칙은 classify(공통 기록 함수)에 있고
승인 게이트가 이들을 A17 요청에서 받지 않는다. 운영 화면은 이들을 사람이 직접 반영할 목록으로 계속 보여준다.
반영할 때 시드의 같은 자리와 값이 어긋나는 후보(prior가 다른 원인 후보, 다른 팀이 맡은 담당 자리)는 넣지 않고
사람이 직접 반영할 목록으로 돌려 완료 기록과 커밋 메시지에 남긴다.

M8 최종 실행 전까지 이 반영으로 바뀐 시드는 main에 머지하지 않는다. K가 바뀌면 정답 라벨과 이전 결과의 기준이 흔들린다
(M4 PR 5 결정 5). 그래서 커밋은 지금 브랜치에만 하고 푸시하지 않는다 (결정 4).
"""

import fcntl
import io
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from common.writers.extraction import classify
from ontology.seed import PROVENANCE_FILE, SEED_DIR, load_seed, validate_seed

ROOT = Path(__file__).resolve().parents[2]
NODE_FILES = {"Team": ("teams.yaml", "teams"), "System": ("systems.yaml", "systems"),
              "SymptomType": ("symptoms.yaml", "symptoms"), "Cause": ("causes.yaml", "causes")}
SEED_EDIT_FILES = ("teams.yaml", "systems.yaml", "symptoms.yaml", "causes.yaml", PROVENANCE_FILE)


class ApplyError(RuntimeError):
    pass


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _flow(data: dict) -> CommentedMap:
    m = CommentedMap(data)
    m.fa.set_flow_style()
    return m


def _flow_list(items: list) -> CommentedSeq:
    s = CommentedSeq(items)
    s.fa.set_flow_style()
    return s


@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)
    manual: list[dict] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)


def _prov_key(rel: str, start: str, end: str, props: dict) -> tuple:
    return (rel, start, end, tuple(sorted(props.items())))


def _clash(c: dict, props: dict, node) -> str | None:
    """후보가 지금 시드(이 배치에서 앞서 넣은 값 포함)의 같은 자리와 다른 값을 가지면 그 이유를 돌려준다.

    - MAY_BE_CAUSED_BY: 같은 증상과 원인에 prior가 이미 있고 값이 다르다. prior는 출처 키에 넣지 않는다.
      넣으면 같은 관계가 두 번 생긴다. 같으면 기존 출처에만 합친다
    - OWNS: 같은 시스템의 같은 scope, area 자리를 다른 팀이 맡고 있다. 1차 담당 조회가 두 팀을 돌려주게 된다
    """
    rel, s, o = c["relation"], c["subject"], c["object"]
    if rel == "MAY_BE_CAUSED_BY":
        prior = (c.get("props") or {}).get("prior")
        held = next((ref.get("prior") for ref in node("SymptomType", s["id"])["causes"] if ref["cause"] == o["id"]), None)
        if prior and held and prior != held:
            return f"시드의 prior({held})와 후보의 prior({prior})가 다르다. 어느 값이 맞는지 사람이 정한다"
    if rel == "OWNS" and props.get("scope"):
        holders = {e["team"] for e in node("System", o["id"])["owners"]
                   if props["scope"] in e["scopes"] and e.get("area") == props.get("area")}
        if holders - {s["id"]}:
            return f"{o['id']}의 {props['scope']} {props.get('area') or ''} 담당이 이미 {sorted(holders)}다. 사람이 정한다"
    return None


def apply_to_seed(seed_dir: Path, candidates: list[dict], org: str = "org-a") -> ApplyResult:
    """후보를 seed_dir의 시드 파일에 반영하고 시드 검증을 거친다. 검증에 실패하면 파일을 되돌리고 멈춘다."""
    y = _yaml()
    paths = {name: seed_dir / name for name in SEED_EDIT_FILES}
    originals = {name: p.read_text(encoding="utf-8") for name, p in paths.items()}
    docs = {name: y.load(text) for name, text in originals.items()}
    result = ApplyResult()
    if docs[PROVENANCE_FILE].get("provenance") is None:
        docs[PROVENANCE_FILE]["provenance"] = CommentedSeq()
    prov = docs[PROVENANCE_FILE]["provenance"]
    prov.fa.set_block_style()  # 빈 목록 '[]'은 한 줄 표기로 읽힌다. 항목을 더하면 여러 줄로 쓴다
    index = {_prov_key(e["rel"], e["start"], e["end"], dict(e.get("props") or {})): e for e in prov}

    def node(label: str, node_id: str):
        filename, key = NODE_FILES[label]
        found = next((n for n in docs[filename][key] if n["id"] == node_id), None)
        if found is None:
            raise ApplyError(f"시드에 없는 {label}다: {node_id}")
        return found

    for c in candidates:
        auto, reason = classify(c)
        if not auto:
            result.manual.append({"candidate_id": c["id"], "reason": reason})
            continue
        rel, s, o = c["relation"], c["subject"], c["object"]
        props = {k: v for k, v in (c.get("props") or {}).items() if rel in ("OWNS", "LOCATED_IN") and k in ("scope", "area")}
        end = o["name"] if rel == "ALIAS" else o["id"]
        key = _prov_key(rel, s["id"], end, props)
        # 후보의 match는 반영 전 K와 비교한 값이다. 앞 후보가 이 배치에서 시드에 넣은 값과 어긋나는지는 여기서 본다
        clash = _clash(c, props, node)
        if clash:
            result.manual.append({"candidate_id": c["id"], "reason": clash})
            continue
        new = c["match"] == "new" and key not in index
        if new:
            # 같은 사실을 담은 후보가 여럿이면(반복 추출) 시드에는 한 번만 넣고 출처만 합친다
            if rel == "OWNS":
                owners = node("System", o["id"])["owners"]
                entry = {"team": s["id"], "scopes": _flow_list([props["scope"]])}
                if props.get("area"):
                    entry["area"] = props["area"]
                owners.append(_flow(entry))
            elif rel == "DEPENDS_ON":
                node("System", s["id"])["depends_on"].append(o["id"])
            elif rel == "OBSERVED_IN":
                node("SymptomType", s["id"])["observed_in"].append(o["id"])
            elif rel == "MAY_BE_CAUSED_BY":
                node("SymptomType", s["id"])["causes"].append(_flow({"cause": o["id"], "prior": c["props"]["prior"]}))
            elif rel == "ALIAS":
                node(s["type"], s["id"])["aliases"].append(o["name"])
        if key in index:
            entry = index[key]
            entry["docs"] = _flow_list(sorted(set(entry["docs"]) | {c["document_id"]}))
            entry["candidates"] = _flow_list(sorted(set(entry["candidates"]) | {c["id"]}))
        else:
            entry = CommentedMap({"rel": rel, "start": s["id"], "end": end})
            if props:
                entry["props"] = _flow(props)
            entry["source"] = "extracted" if c["match"] == "new" else "manual"
            entry["docs"] = _flow_list([c["document_id"]])
            entry["candidates"] = _flow_list([c["id"]])
            prov.append(entry)
            index[key] = entry
        result.applied.append(c["id"])

    for name, data in docs.items():
        out = io.StringIO()
        y.dump(data, out)
        if out.getvalue() != originals[name]:
            paths[name].write_text(out.getvalue(), encoding="utf-8")
            result.changed_files.append(name)
    try:
        seed = load_seed(org, seed_dir=seed_dir)
        validate_seed(seed)
        from ontology.load import build_graph  # 출처가 시드의 관계와 맞는지 본다
        build_graph(seed)
    except Exception as exc:
        for name, text in originals.items():
            paths[name].write_text(text, encoding="utf-8")
        raise ApplyError(f"반영한 시드가 검증을 통과하지 못해 되돌렸다: {exc}") from exc
    return result


def _git(*args: str, root: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def check_branch(root: Path = ROOT) -> str:
    """커밋할 수 있는 브랜치인지 본다. main이거나 커밋하지 않은 변경이 있으면 멈춘다 (M4 PR 5 결정 4)."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", root=root)
    if branch in ("main", "HEAD"):
        raise ApplyError(f"main이나 분리된 HEAD에는 반영 커밋을 하지 않는다: {branch}")
    if _git("status", "--porcelain", root=root):
        raise ApplyError("커밋하지 않은 변경이 있다. 반영 커밋에 다른 변경이 섞이지 않게 먼저 정리한다")
    return branch


MANUAL_LINE = "  - "


def commit_apply(run: dict, result: ApplyResult, root: Path = ROOT, seed_dir: Path = SEED_DIR) -> str:
    """시드 변경을 지금 브랜치에 커밋하고 커밋 해시를 돌려준다. 푸시하지 않는다.

    커밋 메시지에 실행 id와 사람이 직접 반영할 후보와 이유를 남긴다. 커밋 뒤에 실패한 실행은 이 메시지로 이어서 한다.
    """
    if not result.changed_files:
        raise ApplyError("시드에 바뀐 것이 없다")
    rel_dir = seed_dir.resolve().relative_to(root.resolve())
    _git("add", *(str(rel_dir / f) for f in result.changed_files), root=root)
    lines = [f"지식 반영 A17: 후보 {len(result.applied)}개 ({run['run_id']})", "",
             f"승인: {', '.join(run['approved_by_teams'])} / 실행: {run['executed_by_team']}",
             "반영: " + ", ".join(result.applied)]
    if result.manual:
        lines.append("사람이 직접 반영할 것:")
        lines += [f"{MANUAL_LINE}{m['candidate_id']}: {m['reason']}" for m in result.manual]
    _git("commit", "-q", "-m", "\n".join(lines), root=root)
    return _git("rev-parse", "HEAD", root=root)


def find_commit(run_id: str, root: Path = ROOT) -> tuple[str, list[dict]] | None:
    """지금 브랜치에 이 실행의 반영 커밋이 있으면 (커밋, 수동 목록)을 돌려준다.

    커밋은 시드 변경과 함께 남는 기록이라 따로 단계를 저장하지 않고 이것으로 재개 지점을 정한다.
    커밋 뒤에 다시 적재나 완료 기록이 실패해도 같은 실행을 다시 누르면 새 커밋 없이 그 뒤부터 한다.
    """
    sha = _git("log", "--fixed-strings", f"--grep=({run_id})", "--format=%H", "-1", root=root)
    if not sha:
        return None
    body = _git("log", "-1", "--format=%B", sha, root=root)
    manual = [dict(zip(("candidate_id", "reason"), line[len(MANUAL_LINE):].split(": ", 1)))
              for line in body.splitlines() if line.startswith(MANUAL_LINE)]
    return sha, manual


@contextmanager
def repo_lock(root: Path = ROOT):
    """반영 실행 하나만 저장소 작업 트리를 쓰게 한다. 브랜치 확인부터 완료 기록까지 이 잠금 안에서 한다.

    둘이 함께 돌면 나중 실행이 앞 실행의 시드 변경을 덮거나, 앞 실행의 커밋에 다른 실행의 변경이 섞인다.
    운영 화면은 한 머신의 한 저장소에서 돈다고 가정하고 파일 잠금(.git 안)을 쓴다. 다른 머신은 막지 못한다.
    """
    path = Path(_git("rev-parse", "--absolute-git-dir", root=root)) / "kb-apply.lock"
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def execute(driver, *, run_id: str, tracer, now: datetime, org: str = "org-a", root: Path = ROOT,
            seed_dir: Path = SEED_DIR) -> dict:
    """승인된 A17 실행 기록을 반영하고 완료로 기록한다. 반영 결과와 실행 기록을 돌려준다.

    잠금을 얻은 뒤에 실행 상태와 브랜치를 본다. 이 실행의 커밋이 이미 있으면 시드를 다시 고치지 않고
    다시 적재와 완료 기록만 한다. 모든 후보가 사람에게 돌아가면 커밋 없이 완료하고 목록만 남긴다.
    """
    from common.gate.gate import KNOWLEDGE_ACTION, mark_done
    from common.writers.events import read_action_run
    from common.writers.extraction import list_candidates
    from ontology.load import load_static

    with repo_lock(root):
        run = read_action_run(driver, run_id)
        if run is None or run["action_id"] != KNOWLEDGE_ACTION:
            raise ApplyError(f"지식 반영 실행 기록이 아니다: {run_id}")
        if run["status"] != "approved":
            raise ApplyError(f"{run_id}: 승인된 반영만 실행한다: {run['status']}")
        check_branch(root)
        found = find_commit(run_id, root)
        if found:
            commit, manual = found
            applied = sorted(set(run["candidate_ids"]) - {m["candidate_id"] for m in manual})
        else:
            wanted = set(run["candidate_ids"])
            result = apply_to_seed(seed_dir, [c for c in list_candidates(driver) if c["id"] in wanted], org)
            commit = commit_apply(run, result, root, seed_dir) if result.applied else None
            applied, manual = result.applied, result.manual
        load_static(driver, load_seed(org, seed_dir=seed_dir))
        done = mark_done(driver, tracer=tracer, run_id=run_id, now=now, commit=commit, manual=manual)
    return {"applied": applied, "manual": manual, "commit": commit, "run": done, "resumed": bool(found)}
