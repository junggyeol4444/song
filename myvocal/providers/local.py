"""
로컬 Provider — 키도 인터넷도 필요 없다.

API 키가 없어도 프로그램의 모든 흐름은 끝까지 가야 한다. 그래서 바깥 서비스가
하는 일마다 여기 로컬 판이 하나씩 있다. 품질은 바깥 서비스보다 낮을 수 있지만,
멈추지는 않는다.
"""

from __future__ import annotations

from ..voice.korean import attach_particle
from .base import Capability, Provider, ProviderInfo, ProviderRequest, ProviderResult


class LocalProvider(Provider):
    """MYVOCAL Native. 프로그램 안에서 규칙으로 처리한다."""

    info = ProviderInfo(
        name="local",
        display_name="MYVOCAL 내장",
        capabilities=frozenset({Capability.LYRICS}),
        needs_key=False,
        needs_internet=False,
        cost_note="무료",
        description="프로그램에 들어 있는 규칙으로 처리합니다. 인터넷이 없어도 됩니다.",
    )

    def is_ready(self) -> tuple[bool, str]:
        return True, "항상 사용 가능"

    def execute(self, request: ProviderRequest) -> ProviderResult:
        if request.capability is Capability.LYRICS:
            # 순환 import 를 피하려고 여기서 부른다 (lyrics 가 providers 를 쓴다)
            from ..music.lyrics import local_lyrics
            return ProviderResult(ok=True, data=local_lyrics(request.payload),
                                  cost_note="무료")
        return ProviderResult(
            ok=False,
            message=f"내장 기능은 {attach_particle(request.capability.korean, '을/를')} 하지 못합니다.",
        )
