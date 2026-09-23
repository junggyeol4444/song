"""
Claude (Anthropic) Provider.

키가 있으면 작사를 Claude 가 한다. 규칙으로 쓰는 로컬 작사보다 훨씬 낫다.
키가 없거나, 인터넷이 끊겼거나, 서비스가 응답하지 않으면 등록기가 로컬로
넘어간다. 그래서 이 파일이 실패해도 프로그램은 멈추지 않는다.

어느 모델을 쓸지는 설정에서 고른다. 고르지 않았으면 계정에서 쓸 수 있는
모델 목록을 받아 가장 최근 것을 쓴다. 모델 이름을 코드에 박아 두면
그 모델이 은퇴하는 날 작사가 멈춘다.

가사는 '줄마다 몇 음절' 이 정해져 있다. 언어 모델은 글자 수를 정확히 세는 데
약하다. 그래서 받은 뒤 직접 세어 보고, 틀린 줄이 있으면 그 줄들만 짚어서
한 번 더 고쳐 달라고 한다. 그래도 남는 차이는 멜로디 쪽에서 맞춘다.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .base import (
    Capability, Provider, ProviderError, ProviderInfo, ProviderRequest, ProviderResult,
    ProviderUnavailable,
)
from .settings import Settings, shared_settings
from ..voice.korean import attach_particle

MODEL_SETTING = "anthropic_model"

_SYSTEM = """당신은 한국어 대중가요 작사가입니다.

주어진 멜로디의 틀에 맞춰 가사를 씁니다. 틀은 구간별로 '줄마다 몇 음절인가' 입니다.
음 하나에 한글 한 글자가 붙어서 불리므로, 각 줄의 한글 글자 수(띄어쓰기와 문장부호 제외)가
지정된 수와 정확히 같아야 합니다. 쓰고 나서 한 줄씩 글자를 세어 확인하세요.

지켜야 할 것:
- 한글만 씁니다. 영어, 숫자, 문장부호를 넣지 않습니다.
- 소리 내어 부를 수 있는 자연스러운 말이어야 합니다.
- 후렴은 곡의 핵심 감정을 담고, 벌스는 이야기를 진행시킵니다.
- 같은 틀을 가진 벌스끼리도 가사는 서로 달라야 합니다.
- 설명(subject)에 담긴 이야기와 장면을 가사에 반영합니다.
- 구간에 direction 이 있으면 그 지시를 따릅니다.
"""


def _schema(section_keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "enum": section_keys},
                        "lines": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["key", "lines"],
                    "additionalProperties": False,
                },
            },
            "note": {"type": "string"},
        },
        "required": ["sections", "note"],
        "additionalProperties": False,
    }


def _brief_text(payload: dict) -> str:
    """요청을 사람이 읽는 글로. 모델도 이걸 읽는다."""
    lines = [
        f"제목: {payload.get('title') or '(없음)'}",
        f"설명: {payload.get('subject') or '(없음)'}",
        f"장르: {payload.get('genre') or '(없음)'}",
    ]
    if payload.get("mood"):
        lines.append(f"분위기: {payload['mood']}")
    if payload.get("extra"):
        lines.append(f"추가 지시: {payload['extra']}")
    lines.append("")
    lines.append("써야 할 구간 (lines 는 줄마다 음절 수):")
    for item in payload["sections"]:
        if item["same_as"]:
            continue
        entry = {"key": item["key"], "label": item["label"], "kind": item["kind"],
                 "lines": item["lines"]}
        if item.get("direction"):
            entry["direction"] = item["direction"]
        lines.append(json.dumps(entry, ensure_ascii=False))
    repeated = [f"{i['label']} = {i['same_as']}" for i in payload["sections"] if i["same_as"]]
    if repeated:
        lines.append("")
        lines.append("다음 구간은 앞 구간과 같은 가사를 다시 부르므로 쓰지 않습니다: "
                     + ", ".join(repeated))
    return "\n".join(lines)


class AnthropicProvider(Provider):
    """Claude 로 작사한다."""

    info = ProviderInfo(
        name="anthropic",
        display_name="Claude (Anthropic)",
        capabilities=frozenset({Capability.LYRICS}),
        needs_key=True,
        needs_internet=True,
        cost_note="쓴 토큰만큼 Anthropic 계정에 청구됩니다. 작업이 끝나면 쓴 양을 보여 줍니다.",
        homepage="https://console.anthropic.com",
        description="Claude 가 멜로디 틀에 맞춰 가사를 씁니다.",
    )

    def __init__(self, settings: Settings | None = None,
                 client_factory: Callable[[str, float], Any] | None = None) -> None:
        self.settings = settings or shared_settings()
        # 시험할 때는 가짜 클라이언트를 넣는다. 평소에는 SDK 를 쓴다.
        self._client_factory = client_factory
        self._discovered_model = ""

    # ---- 준비 ----

    @staticmethod
    def sdk_available() -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def is_ready(self) -> tuple[bool, str]:
        if self._client_factory is None and not self.sdk_available():
            return False, "anthropic 패키지가 없습니다 (pip install anthropic)"
        if not self.settings.has_api_key("anthropic"):
            return False, "API 키가 없습니다 (설정에서 등록하거나 ANTHROPIC_API_KEY 환경 변수)"
        return True, "사용 가능"

    def _client(self, timeout: float):
        key = self.settings.api_key("anthropic")
        if self._client_factory is not None:
            return self._client_factory(key, timeout)
        import anthropic
        return anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=2)

    def model(self, client) -> str:
        """쓸 모델. 설정에 있으면 그것, 없으면 계정에서 쓸 수 있는 가장 최근 것."""
        chosen = str(self.settings.get(MODEL_SETTING, "") or "").strip()
        if chosen:
            return chosen
        if not self._discovered_model:
            page = client.models.list(limit=20)
            models = list(getattr(page, "data", page))
            if not models:
                raise ProviderUnavailable("이 계정에서 쓸 수 있는 Claude 모델이 없습니다.")
            self._discovered_model = models[0].id
        return self._discovered_model

    # ---- 실행 ----

    def execute(self, request: ProviderRequest) -> ProviderResult:
        if request.capability is not Capability.LYRICS:
            return ProviderResult(
                ok=False,
                message=f"{attach_particle(request.capability.korean, '은/는')} 지원하지 않습니다.")
        try:
            return self._lyrics(request)
        except ImportError as error:
            raise ProviderUnavailable(str(error)) from error
        except Exception as error:
            # 키가 틀렸거나 권한이 없으면 다시 해도 안 된다. 바로 다음 Provider 로.
            name = type(error).__name__
            if name in ("AuthenticationError", "PermissionDeniedError"):
                raise ProviderUnavailable(f"API 키가 거부됐습니다: {error}") from error
            if name == "NotFoundError":
                raise ProviderUnavailable(
                    f"모델을 찾을 수 없습니다. 설정의 모델 이름을 확인해 주세요: {error}"
                ) from error
            raise

    def _ask(self, client, model: str, messages: list[dict], keys: list[str]) -> tuple[dict, Any]:
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=_SYSTEM,
            messages=messages,
            output_config={"format": {"type": "json_schema", "schema": _schema(keys)}},
        )
        if response.stop_reason == "refusal":
            raise ProviderError("요청이 거절됐습니다. 설명을 바꿔 다시 시도해 주세요.")
        if response.stop_reason == "max_tokens":
            raise ProviderError("응답이 너무 길어 중간에 끊겼습니다.")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ProviderError(f"응답을 읽지 못했습니다: {error}") from error
        return data, response

    def _lyrics(self, request: ProviderRequest) -> ProviderResult:
        from ..music.lyrics import check_response

        payload = request.payload
        keys = [i["key"] for i in payload["sections"] if not i["same_as"]]
        if not keys:
            raise ProviderError("쓸 구간이 없습니다.")
        client = self._client(request.timeout_seconds)
        model = self.model(client)

        messages: list[dict] = [{"role": "user", "content": _brief_text(payload)}]
        data, response = self._ask(client, model, messages, keys)
        sections = {item["key"]: list(item["lines"]) for item in data["sections"]}
        problems = check_response(payload, sections)
        rounds = 1

        if problems:
            # 틀린 줄만 짚어서 한 번 더. 대화를 이어 가므로 앞의 가사를 기억한다.
            messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            messages.append({"role": "user", "content": (
                "음절 수가 맞지 않는 줄이 있습니다. 해당 줄만 고치고, 나머지는 그대로 두고, "
                "전체를 같은 형식으로 다시 주세요.\n" + "\n".join(problems)
            )})
            try:
                retry, response = self._ask(client, model, messages, keys)
                retried = {item["key"]: list(item["lines"]) for item in retry["sections"]}
                if len(check_response(payload, retried)) < len(problems):
                    data, sections = retry, retried
                rounds = 2
            except ProviderError:
                pass    # 첫 결과로 간다. 남은 차이는 멜로디 쪽에서 맞춘다.

        # 구간이 빠졌거나 줄 수가 다르면 멜로디에 붙일 수 없다. 실패로 돌려서
        # 다시 시도하거나 다음 Provider 로 넘어가게 한다.
        # 음절 수만 다른 것은 받아들인다. 그건 멜로디 쪽에서 맞출 수 있다.
        structural = check_response(payload, sections, syllables=False)
        if structural:
            raise ProviderError("가사 틀이 맞지 않습니다: " + " / ".join(structural))

        usage = getattr(response, "usage", None)
        tokens = ""
        if usage is not None:
            tokens = f" / 입력 {usage.input_tokens:,} · 출력 {usage.output_tokens:,} 토큰"
        return ProviderResult(
            ok=True,
            data={"sections": sections, "note": str(data.get("note", "")), "model": model},
            cost_note=f"모델 {model}, 요청 {rounds}번{tokens}",
        )
