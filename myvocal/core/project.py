"""
프로젝트 — 모든 것이 들어 있는 하나의 문서.

이 파일이 MYVOCAL Studio 를 '하나의 프로그램'으로 만든다.

음악, 목소리, 캐릭터, 영상이 각자 자기 파일과 자기 시간축을 가지면, 그건
네 개의 프로그램을 한 창에 띄운 것일 뿐이다. 61번의 AI Creative Director 가
"후렴부터 밤거리로 나가게 해줘" 를 처리하려면, 후렴이 몇 초인지와 그 시각에
어떤 장면이 있는지를 같은 자리에서 알아야 한다.

그래서 여기 한 문서에 전부 담는다.

    메타          제목, 만든 사람, 만든 날짜
    시간          템포, 박자표          <- 음악과 영상이 함께 쓴다
    곡            조성, 장르, 구조, 트랙
    출연          캐릭터 / 사람        <- 14~27번
    영상          Storyboard, 장면     <- 36~41번
    자원          쓰고 있는 라이브러리 항목들

한 군데서 무엇을 바꾸면 나머지가 같은 값을 본다. 동기화할 일이 없다.
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..music.genre import GenreBlend, GenreError
from ..music.structure import Section, SongStructure, StructureError, build_structure
from ..music.theory import Key, Pitch, TheoryError
from .notes import Note, NoteError, NoteList
from .tracks import Track, TrackError, TrackList, next_track_id
from .units import (
    DEFAULT_SAMPLE_RATE, MeterChange, MeterMap, TempoChange, TempoMap,
    TimeError, TimeRange, TimeSignature,
)

FILE_FORMAT = "myvocal-project"
FILE_VERSION = 1


class ProjectError(ValueError):
    """프로젝트 관련 오류."""


@dataclass(slots=True)
class ProjectMeta:
    """프로젝트 정보."""

    title: str = "제목 없음"
    artist: str = ""
    created_at: str = ""
    modified_at: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.created_at:
            self.created_at = now
        if not self.modified_at:
            self.modified_at = now

    def touch(self) -> None:
        self.modified_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> dict:
        return {
            "title": self.title, "artist": self.artist,
            "created_at": self.created_at, "modified_at": self.modified_at,
            "description": self.description, "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMeta":
        return cls(
            title=data.get("title", "제목 없음"),
            artist=data.get("artist", ""),
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
        )


@dataclass(slots=True)
class AssetReference:
    """라이브러리 항목을 가리키는 참조 (59번).

    프로젝트 안에 목소리 모델이나 캐릭터를 통째로 넣지 않는다. 용량도 크고,
    같은 목소리를 쓰는 곡이 열 개면 열 벌이 생긴다. 대신 라이브러리의 항목을
    가리키고, 없으면 열 때 알린다.
    """

    asset_type: str         # "voice" / "character" / "music_style" / "video_style" / "person"
    asset_id: str
    display_name: str = ""
    role: str = ""          # "lead_vocal", "main_character" 처럼 이 프로젝트에서의 역할

    KNOWN_TYPES: tuple[str, ...] = (
        "voice", "character", "person", "music_style", "video_style",
        "singing_style", "motion", "outfit",
    )

    def __post_init__(self) -> None:
        if self.asset_type not in self.KNOWN_TYPES:
            raise ProjectError(
                f"모르는 자원 종류입니다: {self.asset_type!r}\n"
                f"사용 가능: {', '.join(self.KNOWN_TYPES)}"
            )
        if not self.asset_id.strip():
            raise ProjectError("자원 번호가 비어 있습니다.")

    def to_dict(self) -> dict:
        return {"type": self.asset_type, "id": self.asset_id,
                "name": self.display_name, "role": self.role}

    @classmethod
    def from_dict(cls, data: dict) -> "AssetReference":
        return cls(data["type"], data["id"], data.get("name", ""), data.get("role", ""))

    def __str__(self) -> str:
        return f"{self.display_name or self.asset_id} ({self.asset_type})"


class Project:
    """프로젝트 문서 하나."""

    __slots__ = (
        "meta", "sample_rate", "key", "genre", "tempo", "meter",
        "structure", "tracks", "assets", "video", "settings", "_path",
    )

    def __init__(
        self,
        title: str = "제목 없음",
        key: Key | str = "C",
        bpm: float = 120.0,
        time_signature: TimeSignature | str = "4/4",
        genre: GenreBlend | str = "pop",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.meta = ProjectMeta(title=title)
        self.sample_rate = int(sample_rate)
        if self.sample_rate < 8000:
            raise ProjectError(f"샘플레이트가 너무 낮습니다: {sample_rate}")

        self.key: Key = key if isinstance(key, Key) else Key.parse(key)
        self.genre: GenreBlend = (
            genre if isinstance(genre, GenreBlend) else GenreBlend.parse(genre)
        )
        if isinstance(time_signature, str):
            numerator, _, denominator = time_signature.partition("/")
            time_signature = TimeSignature(int(numerator), int(denominator))

        self.tempo = TempoMap([TempoChange(0, bpm)])
        self.meter = MeterMap([MeterChange(0, time_signature)])
        self.structure = SongStructure()
        self.tracks = TrackList()
        self.assets: list[AssetReference] = []
        self.video: dict[str, Any] = {}      # Storyboard 등. video 모듈이 채운다.
        self.settings: dict[str, Any] = {}
        self._path: Path | None = None

    # ------------------------------------------------------------------ 시간

    @property
    def bpm(self) -> float:
        return self.tempo.initial_bpm

    @bpm.setter
    def bpm(self, value: float) -> None:
        self.tempo = TempoMap([TempoChange(0, value)] + [
            c for c in self.tempo.changes if c.tick != 0
        ])
        self.meta.touch()

    @property
    def time_signature(self) -> TimeSignature:
        return self.meter.signature_at_tick(0)

    @property
    def bar_ticks(self) -> int:
        return self.time_signature.ticks_per_bar

    @property
    def end_tick(self) -> int:
        """곡의 끝. 구조와 트랙 중 더 긴 쪽을 따른다."""
        from_structure = (
            self.meter.bar_to_tick(self.structure.total_bars + 1) if self.structure else 0
        )
        return max(from_structure, self.tracks.end_tick)

    @property
    def duration_seconds(self) -> float:
        return self.tempo.tick_to_seconds(self.end_tick)

    def add_tempo_change(self, bar: int, bpm: float, curve: str = "step") -> None:
        tick = self.meter.bar_to_tick(bar)
        self.tempo = self.tempo.with_change(TempoChange(tick, bpm, curve))
        self.meta.touch()

    def add_meter_change(self, bar: int, signature: TimeSignature | str) -> None:
        if isinstance(signature, str):
            numerator, _, denominator = signature.partition("/")
            signature = TimeSignature(int(numerator), int(denominator))
        tick = self.meter.bar_to_tick(bar)
        self.meter = MeterMap(list(self.meter.changes) + [MeterChange(tick, signature)])
        self.meta.touch()

    def bar_to_seconds(self, bar: int) -> float:
        return self.tempo.tick_to_seconds(self.meter.bar_to_tick(bar))

    def seconds_to_bar(self, seconds: float) -> int:
        bar, _ = self.meter.tick_to_bar_beat(self.tempo.seconds_to_tick(seconds))
        return bar

    # ------------------------------------------------------------------ 구조

    def set_structure(self, structure: SongStructure) -> None:
        self.structure = structure
        self.meta.touch()

    def build_default_structure(self, repeat_final_chorus: bool = False) -> SongStructure:
        self.structure = build_structure(self.genre.primary.name, repeat_final_chorus)
        self.meta.touch()
        return self.structure

    def section_at_tick(self, tick: int) -> Section | None:
        return self.structure.at_tick(tick, self.meter)

    def section_range(self, section: Section) -> TimeRange:
        return section.tick_range(self.meter)

    def effective_genre(self, tick: int) -> GenreBlend:
        """그 시점에 적용되는 장르. 구간 설정이 있으면 그것을, 없으면 곡 전체."""
        section = self.section_at_tick(tick)
        if section is not None and section.genre is not None:
            return section.genre
        return self.genre

    def effective_key(self, tick: int) -> Key:
        """그 시점의 조성. 11번 '마지막 후렴만 반키 올려' 가 여기로 반영된다."""
        section = self.section_at_tick(tick)
        if section is not None and section.key is not None:
            return section.key
        return self.key

    # ------------------------------------------------------------------ 트랙

    def add_track(self, name: str, kind: str = "instrument", **kwargs) -> Track:
        track = Track(name=name, kind=kind, **kwargs)
        self.tracks.add(track)
        self.meta.touch()
        return track

    def remove_track(self, track: Track) -> bool:
        removed = self.tracks.remove(track)
        if removed:
            self.meta.touch()
        return removed

    def track(self, name_or_id: str | int) -> Track | None:
        if isinstance(name_or_id, int):
            return self.tracks.find(name_or_id)
        return self.tracks.by_name(name_or_id)

    def vocal_tracks(self) -> list[Track]:
        return self.tracks.of_kind("vocal")

    def notes_in_section(self, section: Section, track: Track | None = None) -> list[Note]:
        span = self.section_range(section)
        if track is not None:
            return track.notes_in(span)
        result: list[Note] = []
        for item in self.tracks:
            result.extend(item.notes_in(span))
        return result

    def transpose_section(self, section: Section, semitones: int,
                          tracks: Iterable[Track] | None = None) -> int:
        """구간 안의 음들을 옮긴다. 11번 '마지막 후렴만 반키 올려줘'.

        구간의 조성 표시도 같이 바꾼다. 음만 옮기고 조성 표시를 안 바꾸면
        나중에 화면과 분석이 다른 말을 한다.
        """
        span = self.section_range(section)
        targets = list(tracks) if tracks is not None else list(self.tracks)
        moved = 0
        for track in targets:
            if track.locked or track.kind == "drum":
                continue    # 드럼은 음높이가 악기 종류를 뜻하므로 옮기면 안 된다
            for note in list(track.notes.in_range(span)):
                track.notes.replace_note(note, note.transposed(semitones))
                moved += 1
        base = section.key or self.key
        section.key = base.transpose(semitones)
        self.meta.touch()
        return moved

    # ------------------------------------------------------------------ 자원

    def add_asset(self, asset_type: str, asset_id: str, display_name: str = "",
                  role: str = "") -> AssetReference:
        reference = AssetReference(asset_type, asset_id, display_name, role)
        for existing in self.assets:
            if existing.asset_type == asset_type and existing.asset_id == asset_id \
                    and existing.role == role:
                return existing
        self.assets.append(reference)
        self.meta.touch()
        return reference

    def assets_of(self, asset_type: str) -> list[AssetReference]:
        return [a for a in self.assets if a.asset_type == asset_type]

    def asset_by_role(self, role: str) -> AssetReference | None:
        for asset in self.assets:
            if asset.role == role:
                return asset
        return None

    # ------------------------------------------------------------------ 검사

    def problems(self) -> list[str]:
        """열거나 렌더링하기 전에 확인할 것들. 비어 있으면 정상이다."""
        found: list[str] = []
        if self.structure:
            for start, end in self.structure.gaps():
                found.append(f"{start}~{end}마디에 구간이 지정되지 않았습니다.")
        for track in self.tracks:
            if track.kind == "vocal" and not track.voice_model:
                found.append(f"보컬 트랙 '{track.name}' 에 목소리 모델이 없습니다.")
            if track.kind == "vocal":
                without_lyric = [n for n in track.notes if not n.lyric]
                if without_lyric and len(without_lyric) < len(track.notes):
                    found.append(
                        f"보컬 트랙 '{track.name}' 의 음 {len(without_lyric)}개에 "
                        f"가사가 없습니다."
                    )
            if track.kind != "drum":
                overlaps = track.notes.overlapping_pairs()
                if overlaps:
                    found.append(
                        f"'{track.name}' 트랙에 같은 음높이가 겹치는 곳이 "
                        f"{len(overlaps)}군데 있습니다."
                    )
        soloed = [t for t in self.tracks if t.soloed]
        if soloed and len(soloed) < len(self.tracks):
            found.append(
                f"솔로가 켜져 있어 {len(soloed)}개 트랙만 들립니다: "
                f"{', '.join(t.name for t in soloed)}"
            )
        return found

    def summary(self) -> str:
        lines = [
            f"{self.meta.title}" + (f" — {self.meta.artist}" if self.meta.artist else ""),
            f"  {self.key.display_name} / {self.bpm:.0f} BPM / {self.time_signature} / {self.genre}",
            f"  {self.structure.total_bars}마디, {self.duration_seconds:.1f}초"
            f" ({int(self.duration_seconds // 60)}:{self.duration_seconds % 60:04.1f})",
            f"  트랙 {len(self.tracks)}개"
            + (f" — {', '.join(t.name for t in self.tracks)}" if self.tracks else ""),
        ]
        if self.assets:
            lines.append(f"  자원: {', '.join(str(a) for a in self.assets)}")
        problems = self.problems()
        if problems:
            lines.append("  확인 필요:")
            lines.extend(f"    - {p}" for p in problems)
        return "\n".join(lines)

    # ------------------------------------------------------------------ 저장

    def to_dict(self) -> dict:
        return {
            "format": FILE_FORMAT,
            "version": FILE_VERSION,
            "meta": self.meta.to_dict(),
            "sample_rate": self.sample_rate,
            "key": str(self.key),
            "genre": self.genre.to_dict(),
            "tempo": [
                {"tick": c.tick, "bpm": c.bpm, "curve": c.curve} for c in self.tempo.changes
            ],
            "meter": [
                {"tick": c.tick, "numerator": c.signature.numerator,
                 "denominator": c.signature.denominator} for c in self.meter.changes
            ],
            "structure": self.structure.to_list(),
            "tracks": self.tracks.to_list(),
            "assets": [a.to_dict() for a in self.assets],
            "video": self.video,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        if data.get("format") != FILE_FORMAT:
            raise ProjectError(
                f"MYVOCAL 프로젝트 파일이 아닙니다 (format={data.get('format')!r})."
            )
        version = int(data.get("version", 0))
        if version > FILE_VERSION:
            raise ProjectError(
                f"이 파일은 더 새로운 형식(버전 {version})입니다. "
                f"이 프로그램은 버전 {FILE_VERSION} 까지 읽습니다."
            )
        project = cls(sample_rate=int(data.get("sample_rate", DEFAULT_SAMPLE_RATE)))
        project.meta = ProjectMeta.from_dict(data.get("meta", {}))
        project.key = Key.parse(data.get("key", "C"))
        project.genre = GenreBlend.from_dict(data.get("genre", {"weights": {"pop": 1.0}}))
        project.tempo = TempoMap([
            TempoChange(int(c["tick"]), float(c["bpm"]), c.get("curve", "step"))
            for c in data.get("tempo", [])
        ] or [TempoChange(0, 120.0)])
        project.meter = MeterMap([
            MeterChange(int(c["tick"]),
                        TimeSignature(int(c["numerator"]), int(c["denominator"])))
            for c in data.get("meter", [])
        ] or [MeterChange(0, TimeSignature(4, 4))])
        project.structure = SongStructure.from_list(data.get("structure", []))
        project.tracks = TrackList.from_list(data.get("tracks", []))
        project.assets = [AssetReference.from_dict(a) for a in data.get("assets", [])]
        project.video = dict(data.get("video", {}))
        project.settings = dict(data.get("settings", {}))
        return project

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ProjectError("저장할 경로가 없습니다.")
        if target.suffix != ".mvp":
            target = target.with_suffix(".mvp")
        self.meta.touch()
        target.parent.mkdir(parents=True, exist_ok=True)
        # 임시 파일에 먼저 쓰고 바꿔치운다. 저장 도중에 멈춰도 원본이 안 깨진다.
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        temporary.replace(target)
        self._path = target
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        source = Path(path)
        if not source.exists():
            raise ProjectError(f"파일이 없습니다: {source}")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ProjectError(f"파일이 손상됐습니다: {source} ({error})") from error
        project = cls.from_dict(data)
        project._path = source
        return project

    @property
    def path(self) -> Path | None:
        return self._path

    def __repr__(self) -> str:
        return (
            f"Project('{self.meta.title}', {self.key}, {self.bpm:.0f}BPM, "
            f"트랙 {len(self.tracks)}개, {self.duration_seconds:.1f}초)"
        )
