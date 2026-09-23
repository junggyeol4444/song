"""
설정과 API 키 보관.

키는 프로젝트 파일에 넣지 않는다. 프로젝트를 남에게 주면 키도 같이 넘어간다.
저장소에도 넣지 않는다. 한 번 올라간 키는 지워도 기록에 남는다.

그래서 사용자 폴더의 설정 파일에만 둔다. 윈도우면
    C:\\Users\\<이름>\\AppData\\Roaming\\MYVOCAL Studio\\settings.json

환경 변수가 있으면 그쪽을 먼저 쓴다. 설정 파일에 안 적어도 되게 하려는 것이다.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SettingsError(ValueError):
    """설정 관련 오류."""


# Provider 이름 -> 환경 변수 이름
ENVIRONMENT_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "suno": "SUNO_API_KEY",
    "runway": "RUNWAY_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
}


def settings_folder() -> Path:
    """설정을 두는 곳."""
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "MYVOCAL Studio"
    elif os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
        return Path.home() / "Library" / "Application Support" / "MYVOCAL Studio"
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "myvocal-studio"


class Settings:
    """프로그램 설정. API 키도 여기 있다."""

    FILENAME = "settings.json"

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or settings_folder()
        self.path = self.folder / self.FILENAME
        self._data: dict[str, Any] = {}
        self.load()

    # ---------------------------------------------------------------- 읽고 쓰기

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            # 설정이 깨졌다고 프로그램이 안 켜지면 안 된다. 기본값으로 간다.
            self._data = {"_load_error": str(error)}

    def save(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self._data.items() if not k.startswith("_")}
        temporary = self.path.with_suffix(".tmp")
        # 키가 들어 있으므로 나만 읽을 수 있게 한다. 다 쓴 뒤에 권한을 바꾸면
        # 그 사이에 남이 읽을 수 있으니, 처음 만들 때부터 막아 둔다.
        # (윈도우는 이 권한 값을 쓰지 않는다. 사용자 폴더 자체가 그 사람 것이다.)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                             stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=1))
        temporary.replace(self.path)
        try:
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return self.path

    # ---------------------------------------------------------------- 값

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            raise SettingsError(f"'_' 로 시작하는 이름은 쓸 수 없습니다: {key!r}")
        self._data[key] = value

    # ---------------------------------------------------------------- API 키

    def api_key(self, provider: str) -> str:
        """키를 찾는다. 환경 변수가 우선이다.

        환경 변수를 먼저 보는 이유: 설정 파일에 안 적고도 쓸 수 있어야 하고,
        여럿이 쓰는 컴퓨터에서 각자 자기 키를 쓸 수 있어야 한다.
        """
        variable = ENVIRONMENT_KEYS.get(provider)
        if variable:
            value = os.environ.get(variable, "").strip()
            if value:
                return value
        keys = self._data.get("api_keys", {})
        return str(keys.get(provider, "")).strip()

    def set_api_key(self, provider: str, key: str) -> None:
        keys = dict(self._data.get("api_keys", {}))
        if key.strip():
            keys[provider] = key.strip()
        else:
            keys.pop(provider, None)
        self._data["api_keys"] = keys

    def has_api_key(self, provider: str) -> bool:
        return bool(self.api_key(provider))

    def key_source(self, provider: str) -> str:
        """키가 어디서 왔는지. 설정 화면에 보여준다."""
        variable = ENVIRONMENT_KEYS.get(provider)
        if variable and os.environ.get(variable, "").strip():
            return f"환경 변수 {variable}"
        if str(self._data.get("api_keys", {}).get(provider, "")).strip():
            return f"설정 파일 ({self.path})"
        return "없음"

    def masked_key(self, provider: str) -> str:
        """키를 화면에 보여줄 때 쓴다. 전체를 보여주면 어깨너머로 새어 나간다."""
        key = self.api_key(provider)
        if not key:
            return ""
        if len(key) <= 10:
            return "*" * len(key)
        return f"{key[:6]}{'*' * 8}{key[-4:]}"

    # ---------------------------------------------------------------- 기타

    @property
    def load_error(self) -> str:
        return str(self._data.get("_load_error", ""))

    def describe(self) -> str:
        lines = [f"설정 파일: {self.path}"]
        if self.load_error:
            lines.append(f"  (읽지 못해 기본값을 씁니다: {self.load_error})")
        lines.append("API 키:")
        for provider in sorted(ENVIRONMENT_KEYS):
            source = self.key_source(provider)
            if source == "없음":
                lines.append(f"  {provider}: 없음")
            else:
                lines.append(f"  {provider}: {self.masked_key(provider)}  ({source})")
        return "\n".join(lines)


_shared: Settings | None = None


def shared_settings() -> Settings:
    """프로그램 전체가 함께 쓰는 설정."""
    global _shared
    if _shared is None:
        _shared = Settings()
    return _shared
