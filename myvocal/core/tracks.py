"""
트랙 — 소리 한 줄.

5번 "각 요소가 실제 편집 가능한 트랙으로 생성된다" 가 이 파일의 이유다.
AI 가 만든 결과가 완성된 음원 한 덩어리로 오면 사용자는 아무것도 못 고친다.
드럼만 빼거나, 베이스 한 음만 바꾸거나, 스트링을 다시 만들 수 없다.

그래서 AI 가 만드는 것도 사람이 만드는 것과 똑같은 트랙이어야 한다.
'AI 전용 트랙' 같은 건 두지 않는다. 그런 게 생기는 순간 편집이 막힌다.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal

from ..audio.buffer import AudioError, db_to_linear
from .notes import Note, NoteList
from .units import TimeRange

TrackKind = Literal["instrument", "vocal", "drum", "audio", "bus"]

TRACK_KIND_NAMES: dict[str, str] = {
    "instrument": "악기",
    "vocal": "보컬",
    "drum": "드럼",
    "audio": "오디오",
    "bus": "버스",
}

# 화면에서 트랙을 구분하는 색. 역할별로 관습적인 색이 있다.
DEFAULT_COLORS: dict[str, str] = {
    "vocal": "#e8556d", "instrument": "#4a90d9", "drum": "#e8a33d",
    "audio": "#7aba5a", "bus": "#9b6bd6",
}


class TrackError(ValueError):
    """트랙 관련 오류."""


_track_id_counter = itertools.count(1)
_note_id_counter = itertools.count(1)


def next_track_id() -> int:
    return next(_track_id_counter)


def next_note_id() -> int:
    return next(_note_id_counter)


@dataclass(slots=True)
class EffectSlot:
    """트랙에 걸린 이펙트 하나.

    설정만 들고 있고 실제 처리는 하지 않는다. 렌더러가 이 설정으로
    dsp 모듈의 함수를 부른다. 이렇게 나눠야 프로젝트 파일을 저장할 수 있다.
    """

    effect_type: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    wet: float = 1.0        # 0 이면 원음만, 1 이면 처리한 소리만

    KNOWN_TYPES: tuple[str, ...] = (
        "eq", "compressor", "limiter", "gate", "saturate",
        "reverb", "delay", "stereo_width", "mono_below", "haas",
        "highpass", "lowpass",
    )

    def __post_init__(self) -> None:
        if self.effect_type not in self.KNOWN_TYPES:
            raise TrackError(
                f"모르는 이펙트입니다: {self.effect_type!r}\n"
                f"사용 가능: {', '.join(self.KNOWN_TYPES)}"
            )
        if not (0.0 <= self.wet <= 1.0):
            raise TrackError(f"wet 은 0~1 이어야 합니다: {self.wet}")

    def to_dict(self) -> dict:
        return {"type": self.effect_type, "params": dict(self.params),
                "enabled": self.enabled, "wet": self.wet}

    @classmethod
    def from_dict(cls, data: dict) -> "EffectSlot":
        return cls(data["type"], dict(data.get("params", {})),
                   bool(data.get("enabled", True)), float(data.get("wet", 1.0)))


@dataclass(slots=True)
class Automation:
    """시간에 따라 변하는 값. '후렴에서만 리버브를 늘려줘' 같은 요구가 이걸 쓴다.

    점들을 tick 순으로 들고, 사이는 선형 보간한다.
    """

    parameter: str                              # "volume_db", "pan", "eq.0.gain_db" 처럼
    points: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points.sort(key=lambda p: p[0])

    def set_point(self, tick: int, value: float) -> None:
        if tick < 0:
            raise TrackError(f"위치는 0 이상이어야 합니다: {tick}")
        self.points = [p for p in self.points if p[0] != tick]
        self.points.append((tick, value))
        self.points.sort(key=lambda p: p[0])

    def value_at(self, tick: int) -> float | None:
        """그 시점의 값. 점이 하나도 없으면 None (오토메이션 없음)."""
        if not self.points:
            return None
        if tick <= self.points[0][0]:
            return self.points[0][1]
        if tick >= self.points[-1][0]:
            return self.points[-1][1]
        for index in range(len(self.points) - 1):
            left_tick, left_value = self.points[index]
            right_tick, right_value = self.points[index + 1]
            if left_tick <= tick <= right_tick:
                if right_tick == left_tick:
                    return right_value
                ratio = (tick - left_tick) / (right_tick - left_tick)
                return left_value + (right_value - left_value) * ratio
        return self.points[-1][1]

    def to_dict(self) -> dict:
        return {"parameter": self.parameter, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> "Automation":
        return cls(data["parameter"], [(int(t), float(v)) for t, v in data.get("points", [])])


@dataclass(slots=True)
class Track:
    """트랙 하나.

    악기 트랙, 보컬 트랙, 드럼 트랙이 같은 구조다. kind 로 구분하고 필요한
    항목만 채운다. 종류마다 클래스를 나누면 '보컬 트랙을 악기 트랙으로 바꾸기'
    같은 흔한 조작이 어려워진다.
    """

    name: str
    kind: TrackKind = "instrument"
    track_id: int = field(default_factory=next_track_id)

    # 음악 내용
    notes: NoteList = field(default_factory=NoteList)

    # 악기 / 목소리
    instrument: str = "acoustic_piano"
    instrument_params: dict[str, Any] = field(default_factory=dict)
    voice_model: str = ""           # 보컬 트랙에서 쓸 목소리 모델 이름
    singing_style: str = ""         # 7번의 Singing Style

    # 믹서
    volume_db: float = 0.0
    pan: float = 0.0
    muted: bool = False
    soloed: bool = False
    effects: list[EffectSlot] = field(default_factory=list)
    automation: list[Automation] = field(default_factory=list)
    output_bus: str = "master"

    # 화면
    color: str = ""
    height: int = 80
    collapsed: bool = False

    # 어디서 왔는가. 51번 자연어 수정이 '드럼을 다시 만들어줘' 를 처리할 때 쓴다.
    generated_by: str = ""          # "" (사람) / "ai_composer" / "ai_arranger" 등
    locked: bool = False            # 켜면 AI 가 못 고친다

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise TrackError("트랙 이름이 비어 있습니다.")
        if self.kind not in TRACK_KIND_NAMES:
            raise TrackError(
                f"모르는 트랙 종류입니다: {self.kind!r} "
                f"(가능: {', '.join(TRACK_KIND_NAMES)})"
            )
        if not (-1.0 <= self.pan <= 1.0):
            raise TrackError(f"팬은 -1~1 이어야 합니다: {self.pan}")
        if not (-96.0 <= self.volume_db <= 12.0):
            raise TrackError(
                f"음량은 -96~+12dB 이어야 합니다: {self.volume_db} "
                f"(더 올려야 한다면 소리 자체를 키우는 게 맞습니다)"
            )
        if not self.color:
            self.color = DEFAULT_COLORS.get(self.kind, "#888888")

    # ---- 조회 ----

    @property
    def kind_name(self) -> str:
        return TRACK_KIND_NAMES[self.kind]

    @property
    def is_empty(self) -> bool:
        return len(self.notes) == 0

    @property
    def start_tick(self) -> int:
        return self.notes.start_tick

    @property
    def end_tick(self) -> int:
        return self.notes.end_tick

    @property
    def gain(self) -> float:
        return db_to_linear(self.volume_db)

    def pitch_range(self) -> tuple[int, int] | None:
        return self.notes.pitch_range

    def notes_in(self, time_range: TimeRange) -> list[Note]:
        return self.notes.in_range(time_range)

    def effect(self, effect_type: str) -> EffectSlot | None:
        for slot in self.effects:
            if slot.effect_type == effect_type:
                return slot
        return None

    def automation_for(self, parameter: str) -> Automation | None:
        for item in self.automation:
            if item.parameter == parameter:
                return item
        return None

    def volume_db_at(self, tick: int) -> float:
        """오토메이션을 반영한 그 시점의 음량."""
        item = self.automation_for("volume_db")
        if item is None:
            return self.volume_db
        value = item.value_at(tick)
        return self.volume_db if value is None else value

    def pan_at(self, tick: int) -> float:
        item = self.automation_for("pan")
        if item is None:
            return self.pan
        value = item.value_at(tick)
        return self.pan if value is None else value

    # ---- 편집 ----

    def add_note(self, note: Note) -> Note:
        if self.locked:
            raise TrackError(f"'{self.name}' 트랙이 잠겨 있습니다.")
        if note.note_id == 0:
            note.note_id = next_note_id()
        return self.notes.add(note)

    def add_notes(self, notes: Iterable[Note]) -> None:
        for note in notes:
            self.add_note(note)

    def remove_note(self, note: Note) -> bool:
        if self.locked:
            raise TrackError(f"'{self.name}' 트랙이 잠겨 있습니다.")
        return self.notes.remove(note)

    def clear_notes(self) -> None:
        if self.locked:
            raise TrackError(f"'{self.name}' 트랙이 잠겨 있습니다.")
        self.notes.clear()

    def add_effect(self, effect_type: str, **params) -> EffectSlot:
        slot = EffectSlot(effect_type, params)
        self.effects.append(slot)
        return slot

    def set_automation(self, parameter: str, points: Iterable[tuple[int, float]]) -> Automation:
        existing = self.automation_for(parameter)
        if existing is not None:
            self.automation.remove(existing)
        item = Automation(parameter, list(points))
        self.automation.append(item)
        return item

    def lyrics_text(self) -> str:
        """이 트랙의 가사를 순서대로 이어붙인다."""
        return "".join(n.lyric for n in self.notes if n.lyric)

    # ---- 저장 ----

    def to_dict(self) -> dict:
        data: dict = {
            "id": self.track_id,
            "name": self.name,
            "kind": self.kind,
            "instrument": self.instrument,
            "volume_db": self.volume_db,
            "pan": self.pan,
            "notes": self.notes.to_list(),
        }
        for key, value, default in (
            ("instrument_params", dict(self.instrument_params), {}),
            ("voice_model", self.voice_model, ""),
            ("singing_style", self.singing_style, ""),
            ("muted", self.muted, False),
            ("soloed", self.soloed, False),
            ("output_bus", self.output_bus, "master"),
            ("color", self.color, ""),
            ("height", self.height, 80),
            ("collapsed", self.collapsed, False),
            ("generated_by", self.generated_by, ""),
            ("locked", self.locked, False),
        ):
            if value != default:
                data[key] = value
        if self.effects:
            data["effects"] = [e.to_dict() for e in self.effects]
        if self.automation:
            data["automation"] = [a.to_dict() for a in self.automation]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        try:
            track = cls(
                name=data["name"],
                kind=data.get("kind", "instrument"),
                track_id=int(data.get("id", next_track_id())),
                notes=NoteList.from_list(data.get("notes", [])),
                instrument=data.get("instrument", "acoustic_piano"),
                instrument_params=dict(data.get("instrument_params", {})),
                voice_model=data.get("voice_model", ""),
                singing_style=data.get("singing_style", ""),
                volume_db=float(data.get("volume_db", 0.0)),
                pan=float(data.get("pan", 0.0)),
                muted=bool(data.get("muted", False)),
                soloed=bool(data.get("soloed", False)),
                effects=[EffectSlot.from_dict(e) for e in data.get("effects", [])],
                automation=[Automation.from_dict(a) for a in data.get("automation", [])],
                output_bus=data.get("output_bus", "master"),
                color=data.get("color", ""),
                height=int(data.get("height", 80)),
                collapsed=bool(data.get("collapsed", False)),
                generated_by=data.get("generated_by", ""),
                locked=bool(data.get("locked", False)),
            )
        except KeyError as error:
            raise TrackError(f"트랙 자료에 {error} 항목이 없습니다.") from error
        return track

    def __repr__(self) -> str:
        return (
            f"Track('{self.name}', {self.kind_name}, {self.instrument}, "
            f"음 {len(self.notes)}개)"
        )


class TrackList:
    """프로젝트의 트랙들. 순서가 곧 화면 순서다."""

    __slots__ = ("_tracks",)

    def __init__(self, tracks: Iterable[Track] = ()) -> None:
        self._tracks: list[Track] = list(tracks)
        self._check_unique_ids()

    def _check_unique_ids(self) -> None:
        seen: set[int] = set()
        for track in self._tracks:
            if track.track_id in seen:
                raise TrackError(f"트랙 번호가 겹칩니다: {track.track_id} ('{track.name}')")
            seen.add(track.track_id)

    def __len__(self) -> int:
        return len(self._tracks)

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks)

    def __getitem__(self, index) -> Track:
        return self._tracks[index]

    def __bool__(self) -> bool:
        return bool(self._tracks)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def add(self, track: Track) -> Track:
        if any(t.track_id == track.track_id for t in self._tracks):
            track.track_id = next_track_id()
        self._tracks.append(track)
        return track

    def remove(self, track: Track) -> bool:
        try:
            self._tracks.remove(track)
            return True
        except ValueError:
            return False

    def remove_id(self, track_id: int) -> bool:
        for index, track in enumerate(self._tracks):
            if track.track_id == track_id:
                del self._tracks[index]
                return True
        return False

    def move(self, from_index: int, to_index: int) -> None:
        if not (0 <= from_index < len(self._tracks)):
            raise TrackError(f"트랙 위치가 범위를 벗어났습니다: {from_index}")
        to_index = max(0, min(len(self._tracks) - 1, to_index))
        track = self._tracks.pop(from_index)
        self._tracks.insert(to_index, track)

    def find(self, track_id: int) -> Track | None:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def by_name(self, name: str) -> Track | None:
        for track in self._tracks:
            if track.name == name:
                return track
        return None

    def of_kind(self, kind: TrackKind) -> list[Track]:
        return [t for t in self._tracks if t.kind == kind]

    def audible(self) -> list[Track]:
        """실제로 소리가 나는 트랙들. 솔로가 하나라도 있으면 솔로만 들린다."""
        soloed = [t for t in self._tracks if t.soloed]
        candidates = soloed if soloed else self._tracks
        return [t for t in candidates if not t.muted]

    @property
    def end_tick(self) -> int:
        return max((t.end_tick for t in self._tracks), default=0)

    def to_list(self) -> list[dict]:
        return [t.to_dict() for t in self._tracks]

    @classmethod
    def from_list(cls, data: Iterable[dict]) -> "TrackList":
        return cls(Track.from_dict(item) for item in data)

    def __repr__(self) -> str:
        return f"TrackList({len(self._tracks)}개 트랙)"
