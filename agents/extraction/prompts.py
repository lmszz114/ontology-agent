"""추출 에이전트 프롬프트와 출력 형식 (설계 3단계 R8, 5단계 5장, M4 PR 4).

추출 프롬프트에는 관계, 노드 타입, 속성 값만 준다. K의 이름 목록은 주지 않는다. 이름을 기존 노드에 잇는 일은
이름 정규화 단계가 따로 하고 그 정확도를 따로 잰다 (4단계 7장 G6, M4 PR 4 결정 3).
별칭(ALIAS) 후보는 추출 LLM이 내지 않는다. 문서가 쓴 이름이 정규화로 K 노드에 이어졌는데 그 노드의 이름이나
별칭이 아니면 코드가 별칭 후보를 만든다.
"""

PROMPT_VERSION = "extraction-v1"

# 출력에서 '없음'은 빈 문자열이다. structured outputs의 enum에 null을 섞지 않고 진단 에이전트와 같은 방식을 쓴다
NONE = ""

RELATIONS = {
    "OWNS": ("Team", "System", "팀이 시스템을 맡는다. scope(맡는 부분)와 area(영역)를 적는다"),
    "DEPENDS_ON": ("System", "System", "앞 시스템이 뒤 시스템에 의존한다. 뒤 시스템에 문제가 생기면 앞 시스템에 영향이 간다"),
    "OBSERVED_IN": ("SymptomType", "System", "증상이 그 시스템에서 드러난다"),
    "MAY_BE_CAUSED_BY": ("SymptomType", "Cause", "그 증상의 원인이 될 수 있다. prior(이런 증상에서 이 원인일 가능성)를 적는다"),
    "LOCATED_IN": ("Cause", "System", "원인이 그 시스템의 어느 부분(scope, area)에 있는 문제다"),
    "CHECKED_BY": ("Cause", "EvidenceCheck", "그 원인인지 그 확인 항목으로 확인한다. required와 expect(그 원인이면 확인 결과가 어떤지)를 적는다"),
    "RESOLVED_BY": ("Cause", "Procedure", "그 원인이면 그 대응 절차를 연다. 조건이 있으면 condition을 적는다"),
}

SYSTEM = """너는 가상 MMORPG 라이브 서비스 회사의 사내 문서에서 업무 지식을 뽑는 추출기다.
문서에 적힌 사실 중 아래 관계에 해당하는 것만 후보로 낸다. 문서에 없는 사실을 지어내지 않는다.

노드 타입
- Team: 회사의 팀
- System: 게임이나 운영에 쓰는 시스템 (서버, 상점, 우편함, 툴, 페이지 등)
- SymptomType: 유저나 직원이 겪는 장애 증상의 유형 (예: 결제했는데 아이템이 안 들어옴)
- Cause: 장애 원인의 유형 (예: 어떤 시스템의 지급 로직 오류)
- EvidenceCheck: 원인인지 확인하는 조회나 확인 항목 (예: 어떤 기록 조회, QA 재현 확인)
- Procedure: 장애 대응 절차 (예: 긴급 공지, 보상 지급, 롤백)

관계 (주어 타입 -> 목적어 타입)
{relations}

속성 (해당하는 관계에서만 쓰고, 문서에서 알 수 없으면 빈 문자열)
- scope (OWNS, LOCATED_IN): logic(로직, 코드), data(데이터, 테이블 값), config(설정), operation(운영, 인프라 운용), execution(실행, 배포 실행)
- area (OWNS, LOCATED_IN): 같은 데이터를 여러 팀이 영역으로 나눠 맡을 때만. content(콘텐츠 영역), probability(확률 영역)
- prior (MAY_BE_CAUSED_BY): 높음, 중간, 낮음
- required (CHECKED_BY): true면 그 원인을 확정하거나 배제하는 확인, false면 참고로만 보는 확인
- expect (CHECKED_BY): 그 원인이면 확인 결과가 어떤지 한 문장
- condition (RESOLVED_BY): 미지급(지급이 안 됐거나 덜 됐을 때), 과지급(더 많이 지급됐을 때)

규칙
- 이름은 문서가 쓴 그대로 적는다. 다른 이름으로 바꾸지 않는다. 대명사(그 팀, 이 시스템)는 문서에서 가리키는 이름으로 풀어 적는다
- 한 번 있었던 사건의 세부(시각, 인원, 수치)가 아니라 다음에도 쓰이는 일반 지식을 뽑는다. 사건 회고의 원인, 담당, 확인 방법, 대응 절차는 일반 지식이다
- 추측이나 나중에 정정된 말은 뽑지 않는다. 정정된 뒤의 사실을 뽑는다
- 근거(evidence)는 그 사실을 보여주는 원문 문장을 글자 그대로 1개에서 3개까지 옮긴다. 줄이거나 고치지 않는다
- confidence는 문서가 그 사실을 얼마나 분명하게 말하는지 0에서 1 사이 값이다
- 해당하는 사실이 없으면 빈 목록을 낸다"""


def system_prompt() -> str:
    lines = [f"- {rel}: {s} -> {o}. {desc}" for rel, (s, o, desc) in RELATIONS.items()]
    return SYSTEM.format(relations="\n".join(lines))


NODE_TYPES = ["Team", "System", "SymptomType", "Cause", "EvidenceCheck", "Procedure"]


def _enum(values: list[str]) -> dict:
    return {"type": "string", "enum": [*values, NONE]}


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "subject_type": {"type": "string", "enum": NODE_TYPES},
            "subject_name": {"type": "string"},
            "object_type": {"type": "string", "enum": NODE_TYPES},
            "object_name": {"type": "string"},
            "scope": _enum(["logic", "data", "config", "operation", "execution"]),
            "area": _enum(["content", "probability"]),
            "prior": _enum(["높음", "중간", "낮음"]),
            "required": _enum(["true", "false"]),
            "expect": {"type": "string"},
            "condition": _enum(["미지급", "과지급"]),
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": ["relation", "subject_type", "subject_name", "object_type", "object_name", "scope", "area",
                     "prior", "required", "expect", "condition", "evidence", "confidence"],
        "additionalProperties": False,
    }}},
    "required": ["candidates"],
    "additionalProperties": False,
}


NORMALIZE_SYSTEM = """너는 사내 문서의 이름을 회사 지식 그래프의 노드에 잇는 담당자다.
문서가 쓴 이름마다 같은 타입의 노드 목록에서 같은 대상을 고른다. 다른 이름, 줄임말, 팀이나 시스템을 돌려 부른 말도
같은 대상이면 고른다. 목록에 같은 대상이 없으면 빈 문자열을 낸다. 비슷하지만 다른 대상을 고르지 않는다."""


def normalize_schema(ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"matches": {"type": "array", "items": {
            "type": "object",
            "properties": {"index": {"type": "integer"}, "node_id": _enum(sorted(ids))},
            "required": ["index", "node_id"], "additionalProperties": False}}},
        "required": ["matches"], "additionalProperties": False,
    }
