"""슬랙 승인 요청 메시지 (설계 5단계 3-1, 8-2, M7 PR 2). 슬랙을 부르지 않는 순수 함수다.

승인권자 팀마다 메시지 하나를 올린다. 버튼을 누른 사람이 그 팀의 승인권자로 행동한다. 슬랙 사용자와 팀의 대응표는
두지 않는다 (데모용 가정, 운영 화면과 같은 기준, 2026-11-27 사용자 결정).
"""

import json

APPROVE, REJECT = "approval_approve", "approval_reject"
DECISIONS = {APPROVE: "approve", REJECT: "reject"}


def _named(names: dict, item_id) -> str:
    if not item_id:
        return "없음"
    name = names.get(item_id)
    return f"{name} ({item_id})" if name else str(item_id)


def request_text(payload: dict, run: dict, team_id: str, names: dict) -> str:
    """승인 요청의 본문. 알림과 화면에 같은 내용이 나온다."""
    summary = payload.get("summary") or {}
    lines = [
        f"*승인 요청: {_named(names, run['action_id'])}*",
        f"승인권자: {_named(names, team_id)}",
        f"사건 {payload['incident_id']}",
    ]
    if summary:
        procedures = ", ".join(_named(names, p) for p in summary.get("procedures") or []) or "없음"
        lines += [
            f"증상 {_named(names, summary.get('symptom'))}, 심각도 {summary.get('severity')}",
            f"원인 {_named(names, summary.get('root_cause'))}, 담당 {_named(names, summary.get('primary_team'))}",
            f"절차 {procedures}",
        ]
    others = [t for t in run["required_approver_teams"] if t != team_id]
    if others:
        lines.append("함께 승인해야 하는 팀: " + ", ".join(_named(names, t) for t in others))
    lines.append(f"실행 기록 {run['run_id']}")
    return "\n".join(lines)


def request_blocks(payload: dict, run: dict, team_id: str, names: dict) -> list[dict]:
    value = json.dumps({"run_id": run["run_id"], "team_id": team_id})
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": request_text(payload, run, team_id, names)}},
        {"type": "actions", "elements": [
            {"type": "button", "action_id": APPROVE, "style": "primary", "value": value,
             "text": {"type": "plain_text", "text": "승인"}},
            {"type": "button", "action_id": REJECT, "style": "danger", "value": value,
             "text": {"type": "plain_text", "text": "반려"}},
        ]},
    ]


def closed_blocks(text: str, note: str) -> list[dict]:
    """버튼을 뗀 메시지. 요청 본문은 남기고 아래에 처리 결과를 붙인다."""
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": note}]},
    ]


def decision_note(decision: str, team_id: str, user_id: str | None, names: dict) -> str:
    word = "승인" if decision == "approve" else "반려"
    who = f"<@{user_id}>" if user_id else "알 수 없는 사용자"
    return f"{_named(names, team_id)} {word}. 누른 사람 {who}"


def parse_click(body: dict) -> dict | None:
    """버튼 누름(block_actions)에서 승인 판단을 꺼낸다. 이 앱의 버튼이 아니거나 값이 깨졌으면 None."""
    try:
        action = body["actions"][0]
        value = json.loads(action["value"])
        decision = DECISIONS[action["action_id"]]
        run_id, team_id = value["run_id"], value["team_id"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(run_id, str) or not isinstance(team_id, str):
        return None
    return {"run_id": run_id, "team_id": team_id, "decision": decision,
            "user_id": (body.get("user") or {}).get("id")}
