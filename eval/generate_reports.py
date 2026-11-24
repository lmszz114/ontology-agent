"""제보 원문 후보 생성 (설계 4단계 4-1).

실행: python -m eval.generate_reports 요청파일.yaml --out 후보파일.yaml

정답 라벨을 먼저 정한 뒤, 제보 문장만 평가 대상(Haiku 5.5)과 다른 모델(Sonnet 5.5, 작업 이름 generation)로
여러 개 만든다. 사람이 후보를 골라 시나리오 파일에 옮긴다. 이 스크립트는 시나리오 파일을 직접 고치지 않는다.

요청 파일 형식 (목록):
  - slot: SC-S6-02/0           # 시나리오와 제보 자리
    channel: 공식 커뮤니티      # 채널 이름 (말투를 정하는 데만 쓴다)
    writer: 유저                # 유저 또는 직원
    situation: "..."            # 제보자가 겪은 일. 원인과 정답을 적지 않는다
    count: 3                    # 만들 후보 수

생성 지시에는 원인, 담당 팀, 정답 라벨을 넣지 않는다. 제보자가 아는 것만 넣는다.
모든 호출은 실행 기록(Langfuse)에 남는다.
"""

import argparse
import sys
from pathlib import Path

import yaml

from common.models import LLM, load_model_config, make_anthropic_client
from common.tracing import Tracer

PROMPT_VERSION = "report-gen-v1"

SYSTEM = """너는 가상 MMORPG 게임의 장애 제보 문장을 쓰는 작가다.
주어진 상황을 겪은 사람이 실제로 보낼 법한 제보를 쓴다.
- 제보자가 직접 보고 겪은 것만 쓴다. 원인을 단정하지 않는다. 추측은 추측처럼 쓴다
- 채널에 맞는 말투를 쓴다. 공식 커뮤니티는 글 제목 없이 본문만, 1:1 문의는 정중하게, 사내 메신저는 직원끼리 짧게
- 실제 회사, 게임, 서비스 이름을 쓰지 않는다
- 한두 문장에서 세 문장 사이로 쓴다
- 후보끼리 표현과 길이를 다르게 한다"""

SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {"type": "string"}}},
    "required": ["candidates"],
    "additionalProperties": False,
}


def generate(llm: LLM, request: dict):
    content = (f"채널: {request['channel']}\n쓰는 사람: {request['writer']}\n상황: {request['situation']}\n"
               f"후보 {request['count']}개를 쓴다.")
    result = llm.call(task="generation", name=f"generate.{request['slot']}", system=SYSTEM,
                      messages=[{"role": "user", "content": content}], output_schema=SCHEMA,
                      max_tokens=400 * request["count"] + 500, metadata={"slot": request["slot"], "prompt_version": PROMPT_VERSION})
    candidates = result.json()["candidates"]
    if len(candidates) != request["count"]:
        # 요청한 수와 다르면 멈춘다. 모자란 후보를 사람이 채우거나 조용히 넘어가지 않는다
        raise SystemExit(f"{request['slot']}: 후보 {request['count']}개를 요청했는데 {len(candidates)}개가 왔다")
    return candidates, result


def main() -> int:
    parser = argparse.ArgumentParser(description="제보 원문 후보 생성 (Sonnet 5.5)")
    parser.add_argument("requests", help="요청 파일 (YAML 목록)")
    parser.add_argument("--org", default="org-a")
    parser.add_argument("--out", required=True, help="후보를 쓸 YAML 파일")
    args = parser.parse_args()
    requests = yaml.safe_load(Path(args.requests).read_text(encoding="utf-8"))

    tracer = Tracer.from_env()
    llm = LLM(load_model_config(args.org), tracer, make_anthropic_client())
    out, cost = [], 0.0
    try:
        with tracer.run("generate_reports", tags=["mode:generation"], metadata={"prompt_version": PROMPT_VERSION},
                        input={"requests": len(requests)}):
            for request in requests:
                candidates, result = generate(llm, request)
                cost += result.cost["total"]
                out.append({"slot": request["slot"], "model": result.model, "candidates": candidates})
    finally:
        tracer.flush()
    Path(args.out).write_text(yaml.safe_dump({"prompt_version": PROMPT_VERSION, "cost_usd": round(cost, 5),
                                              "slots": out}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"{len(out)} slots, ${cost:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
