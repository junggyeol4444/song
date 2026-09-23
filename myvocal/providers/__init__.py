"""
Provider — 바깥 서비스와 내장 기능을 같은 모양으로 다룬다 (10번).
"""

from __future__ import annotations

from .base import (
    Capability, Provider, ProviderError, ProviderInfo, ProviderRegistry, ProviderRequest,
    ProviderResult, ProviderTimeout, ProviderUnavailable,
)
from .settings import Settings, shared_settings


def default_registry(settings: Settings | None = None) -> ProviderRegistry:
    """프로그램이 쓰는 등록기. 키가 있는 것은 자동으로 쓸 수 있게 된다."""
    from .anthropic_provider import AnthropicProvider
    from .local import LocalProvider

    settings = settings or shared_settings()
    registry = ProviderRegistry()
    registry.register(LocalProvider())
    registry.register(AnthropicProvider(settings))
    # 키가 있으면 작사는 Claude 가 먼저 한다. 실패하면 내장으로 넘어간다.
    preferred = settings.get("preferred_providers", {}) or {}
    lyrics = preferred.get(Capability.LYRICS.value, "anthropic")
    if lyrics in ("anthropic", "local"):
        registry.prefer(Capability.LYRICS, lyrics)
    return registry


__all__ = [
    "Capability", "Provider", "ProviderError", "ProviderInfo", "ProviderRegistry",
    "ProviderRequest", "ProviderResult", "ProviderTimeout", "ProviderUnavailable",
    "Settings", "shared_settings", "default_registry",
]
