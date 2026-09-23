"""
Provider — 10번.

"특정 서비스 하나에 의존하지 않는다" 가 문서의 요구다.

    MYVOCAL Native / Suno / 기타 Music API / Local Music Model / 향후 추가

이걸 제대로 하려면 '무엇을 하는가' 와 '누가 하는가' 를 나눠야 한다.
작곡기는 "가사를 만들어 달라" 고만 하고, 그것을 Claude 가 하든 로컬 규칙이
하든 신경 쓰지 않는다. Provider 를 바꿔도 프로젝트 편집기는 그대로다.

바깥 서비스를 부르는 코드에는 반드시 있어야 하는 것들이 있다.

    시간 제한   응답이 안 오면 프로그램이 영원히 멈춘다
    재시도      일시적인 실패로 작업을 잃으면 안 된다
    비용 표시   유료 서비스는 얼마 나갈지 미리 알려야 한다
    되돌아갈 곳 서비스가 죽어도 프로그램은 돌아가야 한다

그래서 그 처리를 여기 한 번만 쓰고, 각 Provider 는 실제 호출만 구현한다.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from ..voice.korean import attach_particle


class ProviderError(RuntimeError):
    """Provider 관련 오류."""


class ProviderUnavailable(ProviderError):
    """지금은 쓸 수 없다. 키가 없거나, 꺼져 있거나, 서비스가 죽었다."""


class ProviderTimeout(ProviderError):
    """정해진 시간 안에 응답이 오지 않았다."""


class Capability(Enum):
    """Provider 가 할 수 있는 일."""

    LYRICS = "lyrics"                # 작사
    SONG_PLAN = "song_plan"          # 곡 기획 (구조, 장르, 분위기)
    MUSIC_AUDIO = "music_audio"      # 완성된 음원 생성
    VOICE_CLONE = "voice_clone"      # 목소리 학습
    SINGING = "singing"              # 노래 합성
    IMAGE = "image"                  # 이미지 생성 (캐릭터)
    VIDEO = "video"                  # 영상 생성
    STORYBOARD = "storyboard"        # 장면 구성

    @property
    def korean(self) -> str:
        return {
            "lyrics": "작사", "song_plan": "곡 기획", "music_audio": "음원 생성",
            "voice_clone": "목소리 학습", "singing": "노래 합성",
            "image": "이미지 생성", "video": "영상 생성", "storyboard": "장면 구성",
        }[self.value]


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Provider 한 개의 정보."""

    name: str
    display_name: str
    capabilities: frozenset[Capability]
    needs_key: bool = True
    needs_internet: bool = True
    cost_note: str = ""
    homepage: str = ""
    description: str = ""

    def can(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def describe(self) -> str:
        lines = [f"{self.display_name} ({self.name})"]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(
            "  할 수 있는 일: "
            + ", ".join(sorted(c.korean for c in self.capabilities))
        )
        needs = []
        if self.needs_key:
            needs.append("API 키")
        if self.needs_internet:
            needs.append("인터넷")
        lines.append(f"  필요한 것: {', '.join(needs) if needs else '없음'}")
        if self.cost_note:
            lines.append(f"  비용: {self.cost_note}")
        return "\n".join(lines)


@dataclass(slots=True)
class ProviderRequest:
    """Provider 에 넘길 요청."""

    capability: Capability
    payload: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ProviderError(f"시간 제한은 양수여야 합니다: {self.timeout_seconds}")
        if self.max_retries < 0:
            raise ProviderError(f"재시도 횟수는 0 이상이어야 합니다: {self.max_retries}")


@dataclass(slots=True)
class ProviderResult:
    """Provider 가 돌려준 것."""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    elapsed_seconds: float = 0.0
    cost_note: str = ""
    message: str = ""
    fell_back: bool = False          # 원래 쓰려던 것이 안 돼서 다른 걸 썼는가

    def require(self, key: str) -> Any:
        if not self.ok:
            raise ProviderError(self.message or "요청이 실패했습니다.")
        if key not in self.data:
            raise ProviderError(
                f"응답에 '{key}' 가 없습니다. 받은 항목: {sorted(self.data)}"
            )
        return self.data[key]


class Provider(ABC):
    """Provider 하나."""

    info: ProviderInfo

    @abstractmethod
    def is_ready(self) -> tuple[bool, str]:
        """지금 쓸 수 있는가와 그 이유."""

    @abstractmethod
    def execute(self, request: ProviderRequest) -> ProviderResult:
        """실제 호출. 시간 제한과 재시도는 run() 이 처리하므로 여기서는 안 해도 된다."""

    def estimate_cost(self, request: ProviderRequest) -> str:
        """이 요청에 얼마나 들지. 모르면 빈 문자열."""
        return self.info.cost_note

    def run(self, request: ProviderRequest,
            on_progress: Callable[[str], None] | None = None) -> ProviderResult:
        """시간 제한과 재시도를 붙여 실행한다."""
        ready, reason = self.is_ready()
        if not ready:
            raise ProviderUnavailable(f"{self.info.display_name}: {reason}")

        last_error: Exception | None = None
        for attempt in range(request.max_retries + 1):
            if on_progress is not None and attempt > 0:
                on_progress(f"{self.info.display_name} 다시 시도 중 ({attempt + 1}번째)")
            started = time.monotonic()
            try:
                result = self.execute(request)
                result.provider = self.info.name
                result.elapsed_seconds = time.monotonic() - started
                if result.ok:
                    return result
                last_error = ProviderError(result.message or "알 수 없는 실패")
            except ProviderUnavailable:
                raise
            except Exception as error:
                last_error = error
            if attempt < request.max_retries:
                # 잠시 기다렸다 다시 한다. 갈수록 더 오래 기다린다.
                time.sleep(min(8.0, 1.5 ** attempt))
        raise ProviderError(
            f"{self.info.display_name} 요청이 {request.max_retries + 1}번 모두 "
            f"실패했습니다: {last_error}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.info.name})"


class ProviderRegistry:
    """쓸 수 있는 Provider 들을 모아 두고 골라 준다."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._preferred: dict[Capability, str] = {}

    def register(self, provider: Provider) -> Provider:
        self._providers[provider.info.name] = provider
        return provider

    def unregister(self, name: str) -> bool:
        return self._providers.pop(name, None) is not None

    def get(self, name: str) -> Provider:
        if name not in self._providers:
            raise ProviderError(
                f"등록되지 않은 Provider 입니다: {name!r}\n"
                f"등록된 것: {', '.join(sorted(self._providers)) or '없음'}"
            )
        return self._providers[name]

    def all(self) -> list[Provider]:
        return list(self._providers.values())

    def prefer(self, capability: Capability, name: str) -> None:
        """이 일은 이 Provider 로 하겠다고 정한다."""
        self.get(name)      # 없으면 여기서 걸린다
        self._preferred[capability] = name

    def providers_for(self, capability: Capability,
                      ready_only: bool = True) -> list[Provider]:
        """이 일을 할 수 있는 것들. 정해 둔 것이 맨 앞에 온다."""
        found = [p for p in self._providers.values() if p.info.can(capability)]
        if ready_only:
            found = [p for p in found if p.is_ready()[0]]
        preferred = self._preferred.get(capability)
        if preferred:
            found.sort(key=lambda p: (p.info.name != preferred,
                                      p.info.needs_key, p.info.needs_internet))
        else:
            # 키도 인터넷도 필요 없는 것을 먼저 (항상 되니까)
            found.sort(key=lambda p: (p.info.needs_key, p.info.needs_internet))
        return found

    def run(self, request: ProviderRequest,
            on_progress: Callable[[str], None] | None = None,
            allow_fallback: bool = True) -> ProviderResult:
        """이 요청을 할 수 있는 Provider 로 실행한다.

        앞의 것이 실패하면 다음 것으로 넘어간다. 바깥 서비스가 죽어도
        프로그램이 멈추지 않으려면 이게 필요하다.
        """
        candidates = self.providers_for(request.capability)
        if not candidates:
            unavailable = [
                f"  · {p.info.display_name}: {p.is_ready()[1]}"
                for p in self._providers.values()
                if p.info.can(request.capability)
            ]
            detail = ("\n" + "\n".join(unavailable)) if unavailable else ""
            raise ProviderUnavailable(
                f"{attach_particle(request.capability.korean, '을/를')} 할 수 있는 것이 "
                f"없습니다.{detail}"
            )

        errors: list[str] = []
        for index, provider in enumerate(candidates):
            if on_progress is not None:
                # 이름이 영어일 수도 있어서 조사를 붙이지 않는 모양으로 쓴다
                on_progress(f"{request.capability.korean} 중: {provider.info.display_name}")
            try:
                result = provider.run(request, on_progress)
                result.fell_back = index > 0
                return result
            except Exception as error:
                errors.append(f"{provider.info.display_name}: {error}")
                if not allow_fallback:
                    raise
        raise ProviderError(
            f"{attach_particle(request.capability.korean, '을/를')} 아무도 해내지 못했습니다:\n"
            + "\n".join(f"  · {e}" for e in errors)
        )

    def status_report(self) -> str:
        """지금 무엇을 쓸 수 있는지. 설정 화면에 보여준다."""
        if not self._providers:
            return "등록된 Provider 가 없습니다."
        lines: list[str] = []
        for provider in sorted(self._providers.values(), key=lambda p: p.info.name):
            ready, reason = provider.is_ready()
            mark = "사용 가능" if ready else "사용 불가"
            lines.append(f"[{mark}] {provider.info.display_name}")
            if not ready:
                lines.append(f"    {reason}")
            lines.append(
                "    "
                + ", ".join(sorted(c.korean for c in provider.info.capabilities))
            )
        lines.append("")
        for capability in Capability:
            available = self.providers_for(capability)
            if available:
                names = ", ".join(p.info.display_name for p in available)
                lines.append(f"{capability.korean}: {names}")
            else:
                lines.append(f"{capability.korean}: 없음")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._providers)

    def __repr__(self) -> str:
        return f"ProviderRegistry({len(self._providers)}개 등록)"
