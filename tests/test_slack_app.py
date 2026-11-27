"""슬랙 승인 버튼 (apps/slack_app, 5단계 3-1, M7 PR 2). 슬랙을 부르지 않고 가짜 클라이언트로 본다."""

import asyncio
import json

import pytest

from apps.slack_app import messages
from apps.slack_app.approver import SlackApprover, SlackConfigError, read_settings

NAMES = {"A10": "보상 지급", "TM05": "게임사업PM팀", "TM09": "인프라운영팀", "C02": "원인 이름", "P6": "보상 지급 절차"}


class FakeSlack:
    """올린 메시지와 고친 메시지를 적어 둔다. on_post를 주면 메시지를 올릴 때마다 부른다 (버튼 누름 흉내)."""

    def __init__(self, on_post=None):
        self.posted, self.updated, self.on_post = [], [], on_post

    async def chat_postMessage(self, **kwargs):
        ts = f"{len(self.posted) + 1}.000"
        self.posted.append(kwargs | {"ts": ts})
        if self.on_post:
            self.on_post(kwargs)
        return {"ts": ts}

    async def chat_update(self, **kwargs):
        self.updated.append(kwargs)
        return {"ok": True}


def payload(*runs, errors=()):
    return {"incident_id": "INC-1", "pending": list(runs), "errors": list(errors),
            "summary": {"symptom": "S1", "severity": "SEV2", "root_cause": "C02", "primary_team": "TM05",
                        "procedures": ["P1", "P6"]}}


def run(run_id="AR-1", teams=("TM05",), approved=()):
    return {"run_id": run_id, "action_id": "A10", "required_approver_teams": list(teams),
            "approved_by_teams": list(approved)}


def click(run_id, team_id, action_id=messages.APPROVE, user="U1"):
    return {"user": {"id": user}, "actions": [{"action_id": action_id,
                                               "value": json.dumps({"run_id": run_id, "team_id": team_id})}]}


def buttons(post) -> list[dict]:
    return [e for block in post.get("blocks") or [] if block["type"] == "actions" for e in block["elements"]]


def test_request_message_names_what_is_approved_and_carries_the_decision_target():
    text = messages.request_text(payload(), run(teams=("TM05", "TM09")), "TM05", NAMES)
    assert "보상 지급 (A10)" in text and "승인권자: 게임사업PM팀 (TM05)" in text
    assert "함께 승인해야 하는 팀: 인프라운영팀 (TM09)" in text
    # 사건 정보는 요약 메시지에만 쓴다
    assert "INC-1" not in text and "SEV2" not in text
    summary = messages.summary_text(payload(), NAMES)
    assert "INC-1" in summary and "SEV2" in summary and "원인 이름 (C02)" in summary
    assert "보상 지급 절차 (P6)" in summary and "게임사업PM팀 (TM05)" in summary
    assert "INC-1" in messages.summary_text({"incident_id": "INC-1"}, {})
    approve, reject = buttons({"blocks": messages.request_blocks(payload(), run(), "TM05", NAMES)})
    assert (approve["action_id"], reject["action_id"]) == (messages.APPROVE, messages.REJECT)
    assert json.loads(approve["value"]) == json.loads(reject["value"]) == {"run_id": "AR-1", "team_id": "TM05"}
    # 이름을 모르는 ID는 ID만 쓴다. 요약이 없어도 만든다
    assert "승인권자: TM77" in messages.request_text({"incident_id": "INC-1"}, run(teams=("TM77",)), "TM77", {})


@pytest.mark.parametrize("body", [
    {}, {"actions": []}, {"actions": [{"action_id": "other", "value": "{}"}]},
    {"actions": [{"action_id": messages.APPROVE, "value": "깨진 값"}]},
    {"actions": [{"action_id": messages.APPROVE, "value": json.dumps({"run_id": 1, "team_id": "TM05"})}]},
    {"actions": [{"action_id": messages.APPROVE, "value": json.dumps(["AR-1"])}]},
])
def test_broken_click_is_not_a_decision(body):
    assert messages.parse_click(body) is None


def test_posts_one_request_per_approver_team_and_returns_the_clicked_decision():
    async def go():
        slack = FakeSlack()
        approver = SlackApprover(slack, "C123", NAMES)
        waiting = asyncio.create_task(approver.decide(payload(run(teams=("TM05", "TM09"), approved=("TM09",)),
                                                              run("AR-2", teams=("TM05", "TM09")))))
        await asyncio.sleep(0)
        # 이미 승인한 팀(AR-1의 TM09)에는 올리지 않는다
        # 사건 요약을 먼저 올리고 요청은 그 스레드에 올린다
        head, *requests = slack.posted
        assert "INC-1" in head["text"] and not buttons(head) and "thread_ts" not in head
        assert [json.loads(buttons(p)[0]["value"]) for p in requests] == [
            {"run_id": "AR-1", "team_id": "TM05"}, {"run_id": "AR-2", "team_id": "TM05"},
            {"run_id": "AR-2", "team_id": "TM09"}]
        assert all(p["thread_ts"] == head["ts"] for p in requests)
        assert all(p["channel"] == "C123" for p in slack.posted)
        assert await approver.handle(click("AR-2", "TM09", messages.REJECT, user="U7"))
        decisions = await waiting
        assert decisions == [{"run_id": "AR-2", "team_id": "TM09", "decision": "reject"}]
        # 누른 메시지는 버튼이 사라지고 누가 무엇을 했는지 남는다
        (update,) = slack.updated
        assert update["ts"] == "4.000" and not buttons(update)
        assert "인프라운영팀 (TM09) 반려" in update["text"] and "<@U7>" in update["text"]
        # 같은 버튼을 다시 누르거나 모르는 요청을 누르면 판단이 되지 않는다
        assert not await approver.handle(click("AR-2", "TM09"))
        assert not await approver.handle(click("AR-9", "TM05"))
        assert approver.clicks.empty()
        return approver, slack

    approver, slack = asyncio.run(go())
    assert set(approver.open) == {("AR-1", "TM05"), ("AR-2", "TM05")}


def test_next_pause_keeps_open_requests_closes_stale_ones_and_reports_errors():
    async def go():
        slack = FakeSlack()
        approver = SlackApprover(slack, "C123", NAMES)
        first = asyncio.create_task(approver.decide(payload(run(teams=("TM05", "TM09")))))
        await asyncio.sleep(0)
        await approver.handle(click("AR-1", "TM09", messages.REJECT))
        await first
        # TM09가 반려해 AR-1은 끝났고, 다시 고른 조치 AR-2가 승인 대기다. 앞 판단 오류도 함께 온다
        second = asyncio.create_task(approver.decide(payload(run("AR-2"), errors=["승인권자가 아니다"])))
        await asyncio.sleep(0)
        # 같은 사건이면 요약을 다시 올리지 않고, 오류도 스레드에 올린다
        assert sum("승인 요청*" in p["text"] and "thread_ts" not in p for p in slack.posted) == 1
        error = next(p for p in slack.posted if "앞 판단을 처리하지 못했다: 승인권자가 아니다" in p["text"])
        assert error["thread_ts"] == slack.posted[0]["ts"]
        stale = slack.updated[-1]
        assert stale["ts"] == "2.000" and "더 받지 않는다" in stale["text"] and not buttons(stale)
        assert set(approver.open) == {("AR-2", "TM05")}
        await approver.handle(click("AR-2", "TM05"))
        assert await second == [{"run_id": "AR-2", "team_id": "TM05", "decision": "approve"}]
        # 같은 요청이 그대로 다시 오면(다른 팀 판단을 기다리는 중) 새로 올리지 않는다
        third = asyncio.create_task(approver.decide(payload(run("AR-3", teams=("TM05", "TM09")))))
        await asyncio.sleep(0)
        await approver.handle(click("AR-3", "TM05"))
        await third
        count = len(slack.posted)
        fourth = asyncio.create_task(approver.decide(payload(run("AR-3", teams=("TM05", "TM09"), approved=("TM05",)))))
        await asyncio.sleep(0)
        assert len(slack.posted) == count and set(approver.open) == {("AR-3", "TM09")}
        fourth.cancel()
        output = {"incident_id": "INC-1", "needs_human": ["대안을 사람이 정한다"], "actions": [
            {"action_id": "A01", "status": "auto_done", "required_approver_teams": []},
            {"action_id": "A10", "status": "rejected", "required_approver_teams": ["TM05"]},
            {"action_id": "A15", "status": "done", "required_approver_teams": ["TM02"]}]}
        await approver.finish([output, {"incident_id": "INC-없음", "actions": [], "needs_human": []}])
        assert not approver.open and "더 받지 않는다" in slack.updated[-2]["text"]
        # 끝나면 사건 요약에 승인이 필요했던 조치의 결과와 사람 확인 사유를 붙인다. 올린 적 없는 사건은 건너뛴다
        result = slack.updated[-1]
        assert result["ts"] == slack.posted[0]["ts"] and "INC-1" in result["text"]
        assert "보상 지급 (A10): 반려" in result["text"] and "A15: 승인되어 완료" in result["text"]
        assert "A01" not in result["text"] and "사람 확인: 대안을 사람이 정한다" in result["text"]

    asyncio.run(go())


def test_request_that_failed_at_the_gate_is_posted_again():
    """누른 판단을 게이트가 거부하면(오류로 돌아옴) 그 요청은 여전히 승인 대기다. 버튼을 다시 올린다."""
    async def go():
        slack = FakeSlack()
        approver = SlackApprover(slack, "C123", NAMES)
        first = asyncio.create_task(approver.decide(payload(run())))
        await asyncio.sleep(0)
        await approver.handle(click("AR-1", "TM05"))
        await first
        second = asyncio.create_task(approver.decide(payload(run(), errors=["기록하지 못했다"])))
        await asyncio.sleep(0)
        assert buttons(slack.posted[-1]) and set(approver.open) == {("AR-1", "TM05")}
        second.cancel()

    asyncio.run(go())


def test_missing_settings_are_named_without_values():
    with pytest.raises(SlackConfigError, match="SLACK_APP_TOKEN, SLACK_APPROVAL_CHANNEL"):
        read_settings({"SLACK_BOT_TOKEN": "x"})
    assert set(read_settings({"SLACK_BOT_TOKEN": "a", "SLACK_APP_TOKEN": "b", "SLACK_APPROVAL_CHANNEL": "c"})) == {
        "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_APPROVAL_CHANNEL"}
