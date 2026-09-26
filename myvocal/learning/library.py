"""
학습실 라이브러리 — 63번 '학습 자료 권한 관리'.

자료를 넣을 때 반드시 고른다.

    ○ 내가 만든 자료
    ○ 내 모습 / 내 목소리
    ○ 사용 허가를 받은 자료
    ○ Reference 분석용

앞의 셋은 학습(내 스타일 만들기)에 쓸 수 있다. 'Reference 분석용' 은 분석만 하고,
그 곡 하나를 참고로 새 곡을 만들 때만 쓴다. 내 스타일에는 절대 섞이지 않는다.

원본 파일은 복사하지 않는다. 분석 결과(숫자)만 사용자 폴더에 저장한다.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..providers.settings import settings_folder
from .analysis import MusicAnalysis, MusicAnalysisError, analyze_midi, analyze_project
from .style import MusicStyleProfile, ReferenceHints, StyleError, build_style, reference_hints, register_style

RIGHTS: dict[str, str] = {
    "own": "내가 만든 자료",
    "likeness": "내 모습 / 내 목소리",
    "licensed": "사용 허가를 받은 자료",
    "reference": "Reference 분석용",
}
LEARNABLE = {"own", "likeness", "licensed"}

MIDI_SUFFIXES = {".mid", ".midi"}
PROJECT_SUFFIXES = {".mvp"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}


def music_folder() -> Path:
    return settings_folder() / "music"


def analyze_file(path: str | Path) -> MusicAnalysis:
    """파일 종류에 맞게 분석한다."""
    path = Path(path)
    suffix = path.suffix.lower()
    if not path.exists():
        raise MusicAnalysisError(f"파일이 없습니다: {path}")
    if suffix in MIDI_SUFFIXES:
        return analyze_midi(path)
    if suffix in PROJECT_SUFFIXES:
        from ..core.project import Project
        return analyze_project(Project.load(path), path.name, "project")
    if suffix in AUDIO_SUFFIXES:
        from .audio_analysis import analyze_audio
        return analyze_audio(path)
    raise MusicAnalysisError(
        f"분석할 수 없는 파일입니다: {path.name}\n"
        f"가능: MIDI({', '.join(sorted(MIDI_SUFFIXES))}), 프로젝트(.mvp), "
        f"오디오({', '.join(sorted(AUDIO_SUFFIXES))})"
    )


@dataclass(slots=True)
class MusicItem:
    item_id: str
    name: str
    path: str
    rights: str
    added: float
    analysis: MusicAnalysis

    @property
    def learnable(self) -> bool:
        return self.rights in LEARNABLE

    def to_dict(self) -> dict:
        return {"id": self.item_id, "name": self.name, "path": self.path, "rights": self.rights,
                "added": self.added, "analysis": self.analysis.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "MusicItem":
        return cls(data["id"], data["name"], data["path"], data["rights"], data.get("added", 0.0),
                   MusicAnalysis.from_dict(data["analysis"]))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


class MusicLibrary:
    """내 음악과 참고 음악, 내 스타일들."""

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder or music_folder()

    @property
    def items_folder(self) -> Path:
        return self.folder / "items"

    @property
    def styles_folder(self) -> Path:
        return self.folder / "styles"

    # ---- 곡 ----

    def add(self, path: str | Path, rights: str, name: str = "") -> MusicItem:
        if rights not in RIGHTS:
            raise StyleError(f"자료 권한을 골라 주세요: {', '.join(RIGHTS.values())}")
        analysis = analyze_file(path)
        item = MusicItem(uuid.uuid4().hex[:10], name.strip() or Path(path).stem, str(path),
                         rights, time.time(), analysis)
        _write(self.items_folder / f"{item.item_id}.json", item.to_dict())
        return item

    def items(self) -> list[MusicItem]:
        if not self.items_folder.exists():
            return []
        result = []
        for path in self.items_folder.glob("*.json"):
            try:
                result.append(MusicItem.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda i: i.added)

    def get(self, item_id: str) -> MusicItem | None:
        return next((i for i in self.items() if i.item_id == item_id), None)

    def delete(self, item_id: str) -> bool:
        path = self.items_folder / f"{item_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    # ---- 내 스타일 ----

    def build_style(self, name: str, item_ids: list[str]) -> MusicStyleProfile:
        items = [self.get(i) for i in item_ids]
        missing = [i for i, item in zip(item_ids, items) if item is None]
        if missing:
            raise StyleError(f"없는 곡이 있습니다: {', '.join(missing)}")
        refused = [item.name for item in items if not item.learnable]
        if refused:
            raise StyleError(
                "Reference 분석용 자료는 학습(내 스타일)에 쓸 수 없습니다 (63번): "
                + ", ".join(refused)
                + "\n참고로만 쓰려면 'Reference 로 새 곡 만들기' 를 쓰세요.")
        style = build_style(name, [item.analysis for item in items])
        _write(self.styles_folder / f"{style.style_id}.json", style.to_dict())
        register_style(style)
        return style

    def styles(self) -> list[MusicStyleProfile]:
        if not self.styles_folder.exists():
            return []
        result = []
        for path in self.styles_folder.glob("*.json"):
            try:
                result.append(MusicStyleProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda s: s.name)

    def delete_style(self, style_id: str) -> bool:
        path = self.styles_folder / f"{style_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def register_all(self) -> list[str]:
        """저장된 내 스타일을 전부 작곡기에 등록한다. 프로그램을 켤 때 부른다."""
        names = []
        for style in self.styles():
            try:
                names.append(register_style(style))
            except Exception:
                continue
        return names

    # ---- Reference ----

    def reference(self, item_id: str) -> ReferenceHints:
        item = self.get(item_id)
        if item is None:
            raise StyleError("없는 곡입니다.")
        return reference_hints(item.analysis)


def load_style_genre(name: str) -> bool:
    """작곡기가 모르는 'mystyle...' 장르를 만나면 라이브러리에서 찾아 등록한다."""
    if not name.startswith("mystyle"):
        return False
    style_id = name[len("mystyle"):]
    path = music_folder() / "styles" / f"{style_id}.json"
    if not path.exists():
        return False
    try:
        register_style(MusicStyleProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    except Exception:
        return False
    return True
