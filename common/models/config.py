"""모델 선택 규칙과 비용 계산 (설계 5단계 9장, 11-1).

작업 이름을 받아 쓸 모델을 정하고, 응답의 토큰 사용량에 단가를 곱해 비용을 계산한다.
비용은 코드로 계산한다. LLM에게 시키지 않는다 (5단계 1장).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
PACKAGES_DIR = ROOT / "packages"
PRICES_PATH = Path(__file__).parent / "prices.yaml"

# 설계 5단계 9장 표의 행. 에이전트 코드는 이 이름으로 모델을 고른다.
TASKS = {
    "intake",         # 접수 정리, 증상 분류, 중복 판정
    "diagnosis",      # 원인 판단, 담당 팀 판단, 절차 선택
    "extraction",     # 지식 후보 추출
    "normalization",  # 이름 정규화
    "generation",     # 제보, 문서 원문 생성
    "grading",        # LLM 채점
}

PRICE_FIELDS = ("input", "output", "cache_write_5m", "cache_write_1h", "cache_read")
USAGE_FIELDS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")


class ModelConfigError(ValueError):
    """모델 설정이나 단가 파일이 틀렸다."""


class PricingError(ValueError):
    """이 사용량의 비용을 계산할 수 없다."""


@dataclass(frozen=True)
class Rates:
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


@dataclass(frozen=True)
class ModelPrice:
    base: Rates
    long_threshold: int | None = None
    long: Rates | None = None

    def rates_for(self, prompt_tokens: int) -> Rates:
        if self.long is not None and prompt_tokens > self.long_threshold:
            return self.long
        return self.base


def _read(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelConfigError(f"{path}: 최상위가 매핑이 아니다")
    return data


def _rates(where: str, raw: dict) -> Rates:
    missing = [f for f in PRICE_FIELDS if f not in raw]
    if missing:
        raise ModelConfigError(f"{where}: 단가 항목이 없다: {missing}")
    values = {}
    for field in PRICE_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ModelConfigError(f"{where}.{field}: 0 이상의 유한한 숫자여야 한다: {value!r}")
        values[field] = float(value)
    return Rates(**values)


def load_prices(path: Path = PRICES_PATH) -> dict[str, ModelPrice]:
    """단가 파일을 읽는다. 모델 ID -> 단가."""
    data = _read(path)
    for key in ("checked", "source", "models"):
        if key not in data:
            raise ModelConfigError(f"{path}: {key}가 없다")
    prices = {}
    for model, raw in data["models"].items():
        if not isinstance(raw, dict):
            raise ModelConfigError(f"{model}: 단가가 매핑이 아니다")
        extra = set(raw) - set(PRICE_FIELDS) - {"long_prompt"}
        if extra:
            raise ModelConfigError(f"{model}: 모르는 항목: {sorted(extra)}")
        long_raw = raw.get("long_prompt")
        if long_raw is None:
            prices[model] = ModelPrice(base=_rates(model, raw))
            continue
        threshold = long_raw.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
            raise ModelConfigError(f"{model}.long_prompt.threshold: 양의 정수여야 한다: {threshold!r}")
        long_rates = {k: v for k, v in long_raw.items() if k != "threshold"}
        prices[model] = ModelPrice(
            base=_rates(model, raw),
            long_threshold=threshold,
            long=_rates(f"{model}.long_prompt", long_rates),
        )
    return prices


def price_record(models: list[str], path: Path = PRICES_PATH) -> dict:
    """실행 결과에 남길 단가. 쓴 모델의 단가와 단가 파일의 확인 날짜, 출처다 (4단계 10장, 한계 L14).

    단가 파일이 나중에 바뀌어도 그 실행의 비용을 어떤 단가로 계산했는지 결과 파일만 보고 알 수 있다.
    """
    data = _read(path)
    load_prices(path)  # 형식이 틀린 단가를 남기지 않는다
    missing = [m for m in models if m not in data["models"]]
    if missing:
        raise PricingError(f"단가가 없는 모델이다: {missing}")
    return {"checked": str(data["checked"]), "source": data["source"],
            "models": {m: data["models"][m] for m in models}}


def usage_from_response(usage) -> dict[str, int]:
    """Messages API 응답의 usage를 비용 계산에 쓰는 다섯 값으로 바꾼다.

    캐시 쓰기는 5분, 1시간으로 나뉜 값이 있으면 그것을 쓴다. 나뉜 값이 없으면 기본 유지 시간인 5분으로 본다.
    """
    if getattr(usage, "inference_geo", None) == "us":
        # 미국 전용 추론은 1.1배 단가인데 단가 파일에 없다. 쓰지 않는 설정이라 계산하지 않고 멈춘다
        raise PricingError("inference_geo us 요청의 단가가 단가 파일에 없다")
    server_tools = getattr(usage, "server_tool_use", None)
    if server_tools is not None and any(
        (count or 0) > 0 for count in server_tools.model_dump(exclude_none=True).values() if isinstance(count, int)
    ):
        # 웹 검색 같은 서버 도구는 토큰 외 요금이 붙는데 단가 파일에 없다
        raise PricingError(f"서버 도구 사용량의 단가가 단가 파일에 없다: {server_tools}")
    write_total = usage.cache_creation_input_tokens or 0
    detail = getattr(usage, "cache_creation", None)
    if detail is not None:
        write_5m = detail.ephemeral_5m_input_tokens or 0
        write_1h = detail.ephemeral_1h_input_tokens or 0
    else:
        write_5m, write_1h = write_total, 0
    if write_5m + write_1h != write_total:
        raise PricingError(f"캐시 쓰기 합이 맞지 않는다: 5분 {write_5m} + 1시간 {write_1h} != {write_total}")
    return {
        "input": usage.input_tokens or 0,
        "output": usage.output_tokens or 0,
        "cache_read": usage.cache_read_input_tokens or 0,
        "cache_write_5m": write_5m,
        "cache_write_1h": write_1h,
    }


@dataclass(frozen=True)
class ModelConfig:
    """한 배포 패키지의 작업별 모델과 단가."""

    package: str
    tasks: dict[str, str]
    prices: dict[str, ModelPrice]

    def model_for(self, task: str) -> str:
        if task not in self.tasks:
            raise ModelConfigError(f"정의되지 않은 작업이다: {task} (작업: {sorted(self.tasks)})")
        return self.tasks[task]

    def cost(self, model: str, usage: dict[str, int]) -> dict[str, float]:
        """사용량의 비용(USD)을 항목별로 돌려준다. total은 합계다."""
        if model not in self.prices:
            raise PricingError(f"단가가 없는 모델이다: {model}")
        if set(usage) != set(USAGE_FIELDS):
            raise PricingError(f"사용량 항목이 맞지 않는다: {sorted(usage)}")
        prompt = usage["input"] + usage["cache_read"] + usage["cache_write_5m"] + usage["cache_write_1h"]
        rates = self.prices[model].rates_for(prompt)
        parts = {
            "input": usage["input"] * rates.input,
            "output": usage["output"] * rates.output,
            "cache_read": usage["cache_read"] * rates.cache_read,
            "cache_write_5m": usage["cache_write_5m"] * rates.cache_write_5m,
            "cache_write_1h": usage["cache_write_1h"] * rates.cache_write_1h,
        }
        cost = {k: v / 1_000_000 for k, v in parts.items()}
        cost["total"] = sum(cost.values())
        return cost


def load_model_config(
    package: str = "org-a",
    *,
    packages_dir: Path = PACKAGES_DIR,
    prices_path: Path = PRICES_PATH,
) -> ModelConfig:
    """배포 패키지의 모델 설정을 읽고 검증한다.

    9장의 작업이 모두 있어야 하고, 고른 모델은 단가 파일에 있어야 한다.
    단가가 없는 모델을 고르면 비용을 남길 수 없으므로 읽을 때 거부한다.
    """
    path = packages_dir / package / "models.yaml"
    data = _read(path)
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        raise ModelConfigError(f"{path}: tasks가 매핑이 아니다")
    missing = TASKS - set(tasks)
    extra = set(tasks) - TASKS
    if missing or extra:
        raise ModelConfigError(f"{path}: 작업이 설계와 다르다. 빠짐 {sorted(missing)}, 모름 {sorted(extra)}")
    prices = load_prices(prices_path)
    for task, model in tasks.items():
        if model not in prices:
            raise ModelConfigError(f"{path}: {task}의 모델 {model!r}에 단가가 없다")
    return ModelConfig(package=package, tasks=dict(tasks), prices=prices)
