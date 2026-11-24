"""모델 선택 규칙, 비용 계산, LLM 호출 기록 검증. API를 부르지 않는다."""

from types import SimpleNamespace

import pytest
import yaml
from anthropic.types import Message

from common.models import LLM, LLMConfigError, ModelConfigError, PricingError, load_model_config
from common.models.config import TASKS, load_prices, usage_from_response
from common.models.llm import API_BASE_URL, API_KEY_ENV, make_anthropic_client
from tests.memory_tracer import MemoryTracer


@pytest.fixture(scope="module")
def config():
    return load_model_config("org-a")


def usage(input=0, output=0, cache_read=0, cache_write_5m=0, cache_write_1h=0):
    return {"input": input, "output": output, "cache_read": cache_read,
            "cache_write_5m": cache_write_5m, "cache_write_1h": cache_write_1h}


# --- 모델 선택 규칙 -----------------------------------------------------------

def test_default_models_follow_design(config):
    """9장: 진단과 추출은 가장 싼 모델, 원문 생성과 채점은 다른 모델."""
    assert set(config.tasks) == TASKS
    for task in ("intake", "diagnosis", "extraction", "normalization"):
        assert config.model_for(task) == "claude-haiku-5-5"
    for task in ("generation", "grading"):
        assert config.model_for(task) == "claude-sonnet-5-5"


def test_unknown_task_is_refused(config):
    with pytest.raises(ModelConfigError, match="정의되지 않은 작업"):
        config.model_for("summary")


def write_package(tmp_path, tasks):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "models.yaml").write_text(yaml.safe_dump({"tasks": tasks}), encoding="utf-8")
    return tmp_path


def test_package_must_cover_exactly_the_design_tasks(tmp_path):
    tasks = {t: "claude-haiku-5-5" for t in TASKS}
    del tasks["grading"]
    with pytest.raises(ModelConfigError, match="빠짐 \\['grading'\\]"):
        load_model_config("pkg", packages_dir=write_package(tmp_path, tasks))
    tasks = {t: "claude-haiku-5-5" for t in TASKS} | {"summary": "claude-haiku-5-5"}
    with pytest.raises(ModelConfigError, match="모름 \\['summary'\\]"):
        other = tmp_path / "other"
        other.mkdir()
        load_model_config("pkg", packages_dir=write_package(other, tasks))


def test_model_without_price_is_refused(tmp_path):
    """단가가 없으면 비용을 남길 수 없으므로 설정을 읽을 때 거부한다."""
    tasks = {t: "claude-haiku-5-5" for t in TASKS} | {"grading": "claude-opus-5-5"}
    with pytest.raises(ModelConfigError, match="단가가 없다"):
        load_model_config("pkg", packages_dir=write_package(tmp_path, tasks))


def test_price_file_shape_is_checked(tmp_path):
    path = tmp_path / "prices.yaml"
    path.write_text(yaml.safe_dump({"checked": "x", "source": "y", "models": {"m": {"input": 1, "output": 1}}}))
    with pytest.raises(ModelConfigError, match="단가 항목이 없다"):
        load_prices(path)
    path.write_text(yaml.safe_dump({"checked": "x", "source": "y", "models": {"m": {
        "input": -1, "output": 1, "cache_write_5m": 1, "cache_write_1h": 1, "cache_read": 1}}}))
    with pytest.raises(ModelConfigError, match="0 이상의 숫자"):
        load_prices(path)
    path.write_text(yaml.safe_dump({"models": {}}))
    with pytest.raises(ModelConfigError, match="checked가 없다"):
        load_prices(path)


# --- 비용 계산 -----------------------------------------------------------------

def test_cost_uses_per_million_rates(config):
    """Haiku 5.5: 입력 0.10, 출력 0.50, 캐시 읽기 0.01, 5분 쓰기 0.125, 1시간 쓰기 0.20 (100만 토큰당)."""
    cost = config.cost("claude-haiku-5-5", usage(
        input=10_000, output=2_000, cache_read=50_000, cache_write_5m=4_000, cache_write_1h=1_000))
    assert cost["input"] == pytest.approx(0.001)
    assert cost["output"] == pytest.approx(0.001)
    assert cost["cache_read"] == pytest.approx(0.0005)
    assert cost["cache_write_5m"] == pytest.approx(0.0005)
    assert cost["cache_write_1h"] == pytest.approx(0.0002)
    assert cost["total"] == pytest.approx(0.0032)


def test_haiku_long_prompt_rate_applies_to_whole_request(config):
    """프롬프트(입력, 캐시 읽기, 캐시 쓰기 합)가 10만을 넘으면 요청 전체가 높은 단가다. 출력도 포함."""
    at_limit = config.cost("claude-haiku-5-5", usage(input=40_000, cache_read=60_000, output=1_000))
    assert at_limit["input"] == pytest.approx(0.004) and at_limit["output"] == pytest.approx(0.0005)
    over = config.cost("claude-haiku-5-5", usage(input=40_000, cache_read=60_001, output=1_000))
    assert over["input"] == pytest.approx(0.02)
    assert over["cache_read"] == pytest.approx(60_001 * 0.05 / 1e6)
    assert over["output"] == pytest.approx(0.0025)


def test_sonnet_has_no_long_prompt_rate(config):
    cost = config.cost("claude-sonnet-5-5", usage(input=500_000, output=1_000, cache_read=100_000))
    assert cost["input"] == pytest.approx(1.0)
    assert cost["cache_read"] == pytest.approx(0.01)
    assert cost["output"] == pytest.approx(0.01)


def test_cost_refuses_unknown_model_or_usage_shape(config):
    with pytest.raises(PricingError, match="단가가 없는 모델"):
        config.cost("claude-opus-5-5", usage())
    with pytest.raises(PricingError, match="사용량 항목"):
        config.cost("claude-haiku-5-5", {"input": 1})


def api_usage(**overrides):
    base = dict(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                cache_creation_input_tokens=0, cache_creation=None, inference_geo="global")
    return SimpleNamespace(**(base | overrides))


def test_usage_from_response_splits_cache_writes():
    detail = SimpleNamespace(ephemeral_5m_input_tokens=30, ephemeral_1h_input_tokens=70)
    got = usage_from_response(api_usage(cache_creation_input_tokens=100, cache_creation=detail, cache_read_input_tokens=5))
    assert got == usage(input=100, output=20, cache_read=5, cache_write_5m=30, cache_write_1h=70)


def test_usage_without_cache_detail_counts_writes_as_5m():
    got = usage_from_response(api_usage(cache_creation_input_tokens=40, cache_read_input_tokens=None))
    assert got == usage(input=100, output=20, cache_write_5m=40)


def test_usage_refuses_unpriced_or_inconsistent_values():
    with pytest.raises(PricingError, match="inference_geo"):
        usage_from_response(api_usage(inference_geo="us"))
    detail = SimpleNamespace(ephemeral_5m_input_tokens=1, ephemeral_1h_input_tokens=1)
    with pytest.raises(PricingError, match="합이 맞지 않는다"):
        usage_from_response(api_usage(cache_creation_input_tokens=5, cache_creation=detail))


# --- 클라이언트 ----------------------------------------------------------------

def test_client_uses_project_key_and_official_url(monkeypatch):
    """ANTHROPIC_BASE_URL이 있어도 공식 주소로 간다. 키는 프로젝트 전용 이름에서만 읽는다."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://claude-code-proxy.invalid")
    monkeypatch.setenv(API_KEY_ENV, "sk-test-key")
    client = make_anthropic_client()
    assert str(client.base_url).rstrip("/") == API_BASE_URL
    assert client.api_key == "sk-test-key"


def test_client_does_not_emit_its_own_traces(monkeypatch):
    """SDK 자체 기록이 켜져 있으면 Langfuse에 호출이 두 번 남고 비용이 두 배로 보인다."""
    monkeypatch.setenv(API_KEY_ENV, "sk-test-key")
    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY", raising=False)
    client = make_anthropic_client()
    assert client._tracer is None
    assert client._open_telemetry["propagation"] is False


def test_client_requires_project_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-other")
    with pytest.raises(LLMConfigError, match=API_KEY_ENV):
        make_anthropic_client()


# --- LLM 호출 기록 ---------------------------------------------------------------

class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **params):
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.response


def fake_message(stop_reason="end_turn", **usage_fields):
    return Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "원인 후보는 C03이다"}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1_000, "output_tokens": 200, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0} | usage_fields,
    })


def test_call_picks_model_by_task_and_records_usage_and_cost(config):
    tracer = MemoryTracer()
    messages = FakeMessages(fake_message())
    llm = LLM(config, tracer, SimpleNamespace(messages=messages))
    with tracer.run("diagnosis", session_id="SC-S1-01", metadata={"config": "C"}):
        result = llm.call(task="diagnosis", name="cause_ranking", system="규칙",
                          messages=[{"role": "user", "content": "제보"}],
                          tools=[{"name": "get_checks", "input_schema": {"type": "object"}}])

    sent = messages.calls[0]
    assert sent["model"] == "claude-haiku-5-5" and sent["system"] == "규칙"
    assert result.text == "원인 후보는 C03이다"
    assert result.usage == usage(input=1_000, output=200)
    assert result.cost["total"] == pytest.approx(0.0002)

    gen, run = tracer.spans()
    assert (gen["name"], gen["type"], gen["model"]) == ("cause_ranking", "generation", "claude-haiku-5-5")
    assert gen["usage"] == result.usage and gen["cost"] == pytest.approx(result.cost)
    assert gen["input"]["tools"] == ["get_checks"]
    assert gen["metadata"]["task"] == "diagnosis" and gen["metadata"]["stop_reason"] == "end_turn"
    assert gen["session_id"] == "SC-S1-01" and gen["trace_metadata"]["config"] == "C"
    assert gen["trace_id"] == run["trace_id"] and run["type"] == "agent"


def test_optional_params_are_not_sent_when_absent(config):
    messages = FakeMessages(fake_message())
    LLM(config, MemoryTracer(), SimpleNamespace(messages=messages)).call(
        task="intake", name="intake", messages=[{"role": "user", "content": "x"}])
    assert "system" not in messages.calls[0] and "tools" not in messages.calls[0]


def test_refusal_is_returned_and_flagged(config):
    tracer = MemoryTracer()
    llm = LLM(config, tracer, SimpleNamespace(messages=FakeMessages(fake_message(stop_reason="refusal"))))
    result = llm.call(task="intake", name="intake", messages=[{"role": "user", "content": "x"}])
    assert result.message.stop_reason == "refusal"
    assert tracer.spans()[0]["level"] == "WARNING"


def test_failed_call_is_recorded_as_error(config):
    tracer = MemoryTracer()
    llm = LLM(config, tracer, SimpleNamespace(messages=FakeMessages(error=RuntimeError("연결 끊김"))))
    with pytest.raises(RuntimeError):
        llm.call(task="intake", name="intake", messages=[{"role": "user", "content": "x"}])
    (span,) = tracer.spans()
    assert span["level"] == "ERROR" and "연결 끊김" in span["status_message"]


def test_unknown_task_fails_before_calling_api(config):
    messages = FakeMessages(fake_message())
    with pytest.raises(ModelConfigError):
        LLM(config, MemoryTracer(), SimpleNamespace(messages=messages)).call(
            task="summary", name="x", messages=[])
    assert messages.calls == []
