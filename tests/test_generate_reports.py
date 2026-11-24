"""제보 후보 생성기. API를 부르지 않고 가짜 LLM으로 저장과 개수 검사를 본다."""

import sys
from types import SimpleNamespace

import yaml

from eval import generate_reports


class FakeLLM:
    def __init__(self, counts):
        self.counts = iter(counts)

    def call(self, **kwargs):
        n = next(self.counts)
        return SimpleNamespace(json=lambda: {"candidates": [f"문장 {i}" for i in range(n)]},
                               cost={"total": 0.01}, model="claude-sonnet-5-5")


class FakeTracer:
    def run(self, *args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()

    def flush(self):
        pass


def test_short_slot_is_recorded_and_fails(tmp_path, monkeypatch):
    requests = [{"slot": f"S/{i}", "channel": "c", "writer": "w", "situation": "s", "count": 3} for i in range(2)]
    (tmp_path / "req.yaml").write_text(yaml.safe_dump(requests, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(generate_reports.Tracer, "from_env", classmethod(lambda cls: FakeTracer()))
    monkeypatch.setattr(generate_reports, "LLM", lambda *a: FakeLLM([2, 3]))
    monkeypatch.setattr(generate_reports, "make_anthropic_client", lambda: None)
    monkeypatch.setattr(generate_reports, "load_model_config", lambda org: None)
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "req.yaml"), "--out", str(tmp_path / "out.yaml")])
    assert generate_reports.main() == 1
    saved = yaml.safe_load((tmp_path / "out.yaml").read_text(encoding="utf-8"))
    # 모자란 슬롯도 뒤의 슬롯도 모두 저장된다
    assert [(s["slot"], len(s["candidates"]), s.get("requested")) for s in saved["slots"]] == [("S/0", 2, 3), ("S/1", 3, None)]
    assert saved["cost_usd"] == 0.02
