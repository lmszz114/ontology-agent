"""슬랙 승인 버튼 (설계 5단계 3-1, 8-1, M7 PR 2).

데모 모드의 진단 그래프가 승인 대기로 멈추면 승인 채널에 요청을 올리고, 버튼 누름을 Socket Mode로 받아 승인 판단으로
돌려준다. Socket Mode는 앱이 먼저 슬랙에 연결하므로 바깥에서 들어오는 요청을 받을 수 없는 환경에서도 돈다.

여기서는 승인 여부를 정하지 않는다. 누른 판단을 그래프에 넘길 뿐이고, 승인권자 확인과 기록은 승인 게이트가 한다.
비밀값은 환경 변수로만 받는다: SLACK_BOT_TOKEN(xoxb), SLACK_APP_TOKEN(xapp, connections:write),
SLACK_APPROVAL_CHANNEL(채널 ID). 저장소와 문서에 값을 적지 않는다.
"""

import asyncio
import os
import re
from contextlib import asynccontextmanager

from apps.slack_app import messages

ENV_NAMES = ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_APPROVAL_CHANNEL")


class SlackConfigError(Exception):
    """슬랙 연결에 필요한 설정이 없다."""


class SlackApprover:
    """승인 요청을 올리고 버튼 누름을 승인 판단으로 바꾼다. client는 chat_postMessage, chat_update를 가진 슬랙 클라이언트."""

    def __init__(self, client, channel: str, names: dict | None = None):
        self.client, self.channel, self.names = client, channel, names or {}
        self.clicks: asyncio.Queue = asyncio.Queue()
        # 버튼이 살아 있는 요청. (실행 기록 ID, 팀 ID) -> 메시지 시각(ts)과 본문
        self.open: dict[tuple[str, str], dict] = {}

    async def decide(self, payload: dict) -> list[dict]:
        """멈춘 내용(interrupt)을 받아 승인 판단 목록을 돌려준다. 판단이 하나 이상 올 때까지 기다린다."""
        for error in payload.get("errors") or []:
            await self.client.chat_postMessage(channel=self.channel, text=f"앞 판단을 처리하지 못했다: {error}")
        wanted = {(run["run_id"], team): run for run in payload["pending"]
                  for team in run["required_approver_teams"] if team not in (run.get("approved_by_teams") or [])}
        # 더 받을 필요가 없어진 요청(다른 팀이 반려한 조치 등)은 버튼을 뗀다
        for key in [k for k in self.open if k not in wanted]:
            await self._close(key, "더 받지 않는다. 조치가 이미 반려되었거나 끝났다")
        for key, run in wanted.items():
            if key in self.open:
                continue
            text = messages.request_text(payload, run, key[1], self.names)
            sent = await self.client.chat_postMessage(
                channel=self.channel, text=text, blocks=messages.request_blocks(payload, run, key[1], self.names))
            self.open[key] = {"ts": sent["ts"], "text": text}
        decisions = [await self.clicks.get()]
        while not self.clicks.empty():
            decisions.append(self.clicks.get_nowait())
        return decisions

    async def handle(self, body: dict) -> bool:
        """버튼 누름 하나를 처리한다. 살아 있는 요청의 누름이면 메시지를 고치고 판단을 넘긴다."""
        click = messages.parse_click(body)
        if click is None:
            return False
        key = (click["run_id"], click["team_id"])
        if key not in self.open:
            return False  # 이미 처리한 요청을 다시 눌렀거나 이 실행의 요청이 아니다
        await self._close(key, messages.decision_note(click["decision"], click["team_id"], click["user_id"], self.names))
        await self.clicks.put({k: click[k] for k in ("run_id", "team_id", "decision")})
        return True

    async def finish(self) -> None:
        """진단이 끝났다. 남은 요청의 버튼을 뗀다."""
        for key in list(self.open):
            await self._close(key, "더 받지 않는다. 조치가 이미 반려되었거나 끝났다")

    async def _close(self, key: tuple[str, str], note: str) -> None:
        message = self.open.pop(key)
        await self.client.chat_update(channel=self.channel, ts=message["ts"], text=f"{message['text']}\n{note}",
                                      blocks=messages.closed_blocks(message["text"], note))


def read_settings(env=None) -> dict:
    env = os.environ if env is None else env
    missing = [name for name in ENV_NAMES if not env.get(name)]
    if missing:
        raise SlackConfigError(f"슬랙 설정이 없다: {', '.join(missing)}. 클라우드 환경의 환경 변수로 등록한다")
    return {name: env[name] for name in ENV_NAMES}


@asynccontextmanager
async def slack_approver(names: dict | None = None, env=None):
    """Socket Mode로 연결한 SlackApprover. 블록을 나가면 연결을 닫는다."""
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_bolt.async_app import AsyncApp

    settings = read_settings(env)
    app = AsyncApp(token=settings["SLACK_BOT_TOKEN"])
    approver = SlackApprover(app.client, settings["SLACK_APPROVAL_CHANNEL"], names)

    @app.action(re.compile(f"^({messages.APPROVE}|{messages.REJECT})$"))
    async def on_click(ack, body):
        await ack()
        await approver.handle(body)

    handler = AsyncSocketModeHandler(app, settings["SLACK_APP_TOKEN"])
    await handler.connect_async()
    try:
        yield approver
    finally:
        await handler.close_async()
