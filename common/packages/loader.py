"""배포 패키지 로더 (설계 5단계 3장).

packages/<조직>/ 폴더 하나가 배포 패키지다. 조직 설정(org.yaml), 모델 설정(models.yaml), 이 패키지가 쓰는
프롬프트 버전(package.yaml)을 한 묶음으로 읽어 에이전트에 넘긴다. 두 에이전트(진단, 추출)가 같은 로더를 쓴다.

- 시드와 조직 설정은 ontology.seed.load_seed, 모델 설정은 common.models.load_model_config가 읽고 검증한다
- 프롬프트 버전: 실행하는 코드의 버전이 패키지에 적힌 버전과 다르면 멈춘다. 프롬프트를 바꾸고 패키지를
  고치지 않으면 어느 패키지 버전으로 돈 결과인지 알 수 없다. 프롬프트 자체는 코드에 있다
- 버전: package.version과 Git 태그(<조직>/v<버전>)로 관리한다. 실행 기록에 id와 version을 남긴다
- 온톨로지 시드는 저장소에 하나다. 패키지에 시드 해시를 적지 않는다. 지식 반영(A17)이 시드를 바꿀 때마다
  패키지를 고쳐야 해서다. 대신 실행 기록에 시드 해시가 남는다 (M6 결정)
"""

from dataclasses import dataclass
import yaml

from common.models import ModelConfig, load_model_config
from common.models.config import PACKAGES_DIR

PROMPT_KEYS = ("diagnosis", "knowledge-a", "knowledge-b", "extraction")


class PackageError(ValueError):
    """배포 패키지를 읽을 수 없거나 실행하는 코드와 맞지 않는다."""


@dataclass(frozen=True)
class Package:
    id: str
    version: int
    prompts: dict[str, str]
    seed: dict
    models: ModelConfig

    def record(self) -> dict:
        """실행 기록에 남기는 값."""
        return {"id": self.id, "version": self.version}


def load_package(org: str, *, uses: dict[str, str] | None = None) -> Package:
    """배포 패키지를 읽는다. uses는 이 실행이 쓰는 프롬프트 {이름: 코드의 버전}이다. 패키지와 다르면 멈춘다."""
    from ontology.seed import load_seed, validate_seed

    folder = PACKAGES_DIR / org
    path = folder / "package.yaml"
    if not path.exists():
        raise PackageError(f"배포 패키지가 없다: {folder}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meta = data.get("package") or {}
    if meta.get("id") != org:
        raise PackageError(f"{path}: package.id({meta.get('id')})가 폴더 이름({org})과 다르다")
    # bool은 int의 하위 타입이라 true가 1로 통과하지 않게 따로 막는다 (M6 점검)
    if not isinstance(meta.get("version"), int) or isinstance(meta["version"], bool) or meta["version"] < 1:
        raise PackageError(f"{path}: package.version은 1 이상의 정수다: {meta.get('version')}")
    prompts = data.get("prompts") or {}
    if set(prompts) != set(PROMPT_KEYS):
        raise PackageError(f"{path}: prompts는 {list(PROMPT_KEYS)}여야 한다: {sorted(prompts)}")
    for name, version in (uses or {}).items():
        if name not in prompts:
            raise PackageError(f"{path}: 정의되지 않은 프롬프트다: {name}")
        if prompts[name] != version:
            raise PackageError(f"{path}: 프롬프트 {name}이 패키지는 {prompts[name]}, 코드는 {version}이다. "
                               "프롬프트를 바꿨으면 패키지의 버전도 고치고 version을 올린다")
    seed = load_seed(org)
    if seed["org"]["id"] != org:
        raise PackageError(f"{folder}/org.yaml: org.id({seed['org']['id']})가 패키지({org})와 다르다")
    validate_seed(seed)
    return Package(id=org, version=meta["version"], prompts=dict(prompts), seed=seed,
                   models=load_model_config(org))
