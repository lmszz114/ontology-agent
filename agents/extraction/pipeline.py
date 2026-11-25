"""추출 처리 단계 1~5 (설계 5단계 5장, 3단계 R8, M4 PR 4).

1 분할(코드) -> 2 후보 추출(LLM, 문서마다 한 번) -> 3 근거 문장 검사(코드) -> 4 이름 정규화(코드, 남은 이름은 LLM)
-> 5 대조와 검수 배정(코드, rules.prepare). 저장(6)은 실행 명령이 기록 함수로 한다.

LLM이 낸 것은 그대로 믿지 않는다. 관계와 노드 타입이 맞지 않는 후보, 원문에 글자 그대로 없는 근거를 가진 후보는
버리고 버린 이유를 남긴다. 버린 수는 추출 채점의 근거 문장 일치율에 쓴다 (4단계 5-2).
"""

import json
import re
from dataclasses import dataclass, field

from agents.extraction import prompts as P
from agents.extraction.rules import Knowledge, prepare
from common.models import LLM
from common.writers.extraction import ALIAS_TARGETS, MAX_SPANS, RELATION_TYPES

LABEL_KEYS = {"Team": "teams", "System": "systems", "SymptomType": "symptoms", "Cause": "causes",
              "EvidenceCheck": "evidence_checks", "Procedure": "procedures"}


class ExtractionError(RuntimeError):
    """LLM 응답을 쓸 수 없다 (끝나지 않은 응답, 깨진 JSON). 결과를 남기지 않는다."""


@dataclass
class DocResult:
    candidates: list[dict]
    dropped: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)  # LLM 호출마다 작업, 모델, 토큰, 비용

    @property
    def cost(self) -> float:
        return round(sum(c["cost"] for c in self.calls), 6)


def paragraphs(text: str) -> list[str]:
    """빈 줄로 단락을 나눈다 (R8-2). 근거 문장은 단락 번호가 아니라 원문 문장으로 받는다."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _call(llm: LLM, *, task: str, name: str, doc_id: str, system, content: str, schema: dict,
          max_tokens: int, result: DocResult) -> dict:
    r = llm.call(task=task, name=name, system=system, messages=[{"role": "user", "content": content}],
                 output_schema=schema, max_tokens=max_tokens,
                 metadata={"doc": doc_id, "prompt_version": P.PROMPT_VERSION})
    result.calls.append({"task": task, "model": r.model, "usage": dict(r.usage), "cost": r.cost["total"],
                         "stop_reason": r.message.stop_reason})
    if r.message.stop_reason != "end_turn":
        raise ExtractionError(f"{doc_id} {task}: 응답이 끝나지 않았다 ({r.message.stop_reason})")
    try:
        return r.json()
    except ValueError as exc:
        raise ExtractionError(f"{doc_id} {task}: JSON을 읽을 수 없다") from exc


def extract_raw(llm: LLM, doc_id: str, text: str, result: DocResult) -> list[dict]:
    """2 후보 추출. 문서 전체를 단락 번호와 함께 한 번에 준다 (M4 PR 4 결정 1). 사실이 단락을 넘어 흩어져 있어서다."""
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs(text), start=1))
    # 시스템 프롬프트는 문서마다 같아 캐시를 건다
    system = [{"type": "text", "text": P.system_prompt(), "cache_control": {"type": "ephemeral"}}]
    data = _call(llm, task="extraction", name=f"extract.{doc_id}", doc_id=doc_id, system=system,
                 content=f"문서:\n{numbered}", schema=P.EXTRACT_SCHEMA, max_tokens=8000, result=result)
    return data["candidates"]


def _props(raw: dict) -> dict:
    out = {}
    for key in ("scope", "area", "prior", "condition"):
        if raw.get(key):
            out[key] = raw[key]
    if raw.get("required"):
        out["required"] = raw["required"] == "true"
    if raw.get("expect", "").strip():
        out["expect"] = raw["expect"].strip()
    allowed = {"OWNS": {"scope", "area"}, "LOCATED_IN": {"scope", "area"}, "MAY_BE_CAUSED_BY": {"prior"},
               "CHECKED_BY": {"required", "expect"}, "RESOLVED_BY": {"condition"}}.get(raw["relation"], set())
    # 관계에 없는 속성은 버린다. 형식은 맞지만 그 관계에 쓸 수 없는 값이다
    return {k: v for k, v in out.items() if k in allowed}


def check_raw(raw_list: list[dict], text: str, result: DocResult) -> list[dict]:
    """3 근거 검사와 형식 정리. 남는 후보는 이름 정규화 전이라 id가 없다."""
    kept = []
    for raw in raw_list:
        rel = raw["relation"]
        s_types, o_types = RELATION_TYPES[rel]
        reason = None
        if raw["subject_type"] not in s_types or raw["object_type"] not in o_types:
            reason = "관계와 노드 타입이 맞지 않는다"
        elif not raw["subject_name"].strip() or not raw["object_name"].strip():
            reason = "이름이 비어 있다"
        spans = list(dict.fromkeys(x.strip() for x in raw["evidence"] if x.strip()))
        if reason is None and not spans:
            reason = "근거 문장이 없다"
        elif reason is None and any(x not in text for x in spans):
            # 근거 문장이 원문에 글자 그대로 없으면 후보를 버린다 (R8-3)
            reason = "근거 문장이 원문에 없다"
        if reason:
            result.dropped.append({"reason": reason, "raw": raw})
            continue
        kept.append({
            "relation": rel,
            "subject": {"type": raw["subject_type"], "name": raw["subject_name"].strip(), "id": None},
            "object": {"type": raw["object_type"], "name": raw["object_name"].strip(), "id": None},
            "props": _props(raw),
            "evidence_spans": spans[:MAX_SPANS],
            "confidence": min(max(float(raw["confidence"]), 0.0), 1.0),
        })
    return kept


def _key(name: str) -> str:
    return re.sub(r"\s+", "", name).casefold()


def name_index(seed: dict) -> dict[str, dict[str, str]]:
    """타입마다 이름과 별칭 -> id. 공백과 대소문자를 무시한 정확 일치만 본다 (R8-4의 코드 단계)."""
    index: dict[str, dict[str, str]] = {}
    for label, key in LABEL_KEYS.items():
        names = index.setdefault(label, {})
        for item in seed[key]:
            for name in [item["name"], *(item.get("aliases") or [])]:
                names.setdefault(_key(name), item["id"])
    return index


def normalize(llm: LLM, k: Knowledge, doc_id: str, cands: list[dict], result: DocResult) -> None:
    """4 이름 정규화. 코드가 정확 일치로 잇고, 남은 이름은 문서마다 한 번 LLM에게 같은 타입의 노드 목록에서 고르게 한다."""
    index = name_index(k.seed)
    unresolved: dict[tuple[str, str], None] = {}
    for c in cands:
        for end in (c["subject"], c["object"]):
            end["id"] = index[end["type"]].get(_key(end["name"]))
            if end["id"] is None:
                unresolved[(end["type"], end["name"])] = None
    result.stats["names_by_code"] = sum(1 for c in cands for e in (c["subject"], c["object"]) if e["id"])
    result.stats["names_to_llm"] = len(unresolved)
    if not unresolved:
        return
    items = list(unresolved)
    types = sorted({t for t, _ in items})
    catalog = []
    ids = []
    for label in types:
        catalog.append(f"[{label}]")
        for node in k.seed[LABEL_KEYS[label]]:
            aliases = ", ".join(node.get("aliases") or [])
            catalog.append(f"- {node['id']}: {node['name']}" + (f" (다른 이름: {aliases})" if aliases else ""))
            ids.append(node["id"])
    asked = "\n".join(f"{i}. [{t}] {n}" for i, (t, n) in enumerate(items))
    data = _call(llm, task="normalization", name=f"normalize.{doc_id}", doc_id=doc_id, system=P.NORMALIZE_SYSTEM,
                 content=f"노드 목록:\n" + "\n".join(catalog) + f"\n\n문서가 쓴 이름:\n{asked}",
                 schema=P.normalize_schema(ids), max_tokens=2000, result=result)
    by_label = {label: {n["id"] for n in k.seed[LABEL_KEYS[label]]} for label in types}
    chosen = {}
    for m in data["matches"]:
        i, node_id = m["index"], m["node_id"]
        # 다른 타입의 노드를 고르면 쓰지 않는다. 형식(enum)은 모든 타입의 id를 받기 때문이다
        if 0 <= i < len(items) and node_id and node_id in by_label[items[i][0]]:
            chosen[items[i]] = node_id
    for c in cands:
        for end in (c["subject"], c["object"]):
            if end["id"] is None:
                end["id"] = chosen.get((end["type"], end["name"]))
    result.stats["names_by_llm"] = len(chosen)


def alias_candidates(cands: list[dict], k: Knowledge, text: str) -> list[dict]:
    """문서가 쓴 이름이 K 노드로 이어졌는데 그 노드의 원래 이름과 다르면 별칭 후보를 만든다 (R8-3 aliases)."""
    out, seen = [], set()
    for c in cands:
        for end in (c["subject"], c["object"]):
            if end["type"] not in ALIAS_TARGETS or end["id"] is None:
                continue
            node = k.node(end["type"], end["id"])
            # 시드에 이미 있는 별칭도 후보로 낸다. 대조에서 existing이 된다. 설계표의 별칭 정답에 시드 별칭(우편함 = 메일함)이
            # 있어서다. 원래 이름 그대로 쓴 것만 별칭이 아니다
            if _key(end["name"]) == _key(node["name"]) or (end["id"], end["name"]) in seen:
                continue
            seen.add((end["id"], end["name"]))
            spans = [x for x in c["evidence_spans"] if end["name"] in x]
            if not spans:
                # 대명사를 풀어 적은 이름처럼 근거 문장에 그 이름이 없으면 원문에서 그 이름이 나오는 줄을 찾는다
                spans = [line.strip() for line in text.splitlines() if end["name"] in line][:1]
            if not spans:
                continue  # 원문에 없는 이름은 별칭 근거가 없다
            out.append({"relation": "ALIAS", "subject": {"type": end["type"], "name": node["name"], "id": end["id"]},
                        "object": {"type": "Name", "name": end["name"], "id": None}, "props": {},
                        "evidence_spans": spans[:1], "confidence": c["confidence"]})
    return out


def _dedupe(cands: list[dict], result: DocResult) -> list[dict]:
    out, seen = [], set()
    for c in cands:
        key = (c["relation"], c["subject"]["id"] or c["subject"]["name"], c["object"]["id"] or c["object"]["name"],
               json.dumps(c["props"], ensure_ascii=False, sort_keys=True))
        if key in seen:
            result.stats["duplicates"] = result.stats.get("duplicates", 0) + 1
            continue
        seen.add(key)
        out.append(c)
    return out


def run_document(llm: LLM, k: Knowledge, doc_id: str, text: str) -> DocResult:
    result = DocResult(candidates=[])
    raw = extract_raw(llm, doc_id, text, result)
    result.stats["raw"] = len(raw)
    cands = check_raw(raw, text, result)
    normalize(llm, k, doc_id, cands, result)
    cands = _dedupe(cands + alias_candidates(cands, k, text), result)
    result.candidates = prepare(cands, k)
    result.stats["kept"] = len(result.candidates)
    return result
