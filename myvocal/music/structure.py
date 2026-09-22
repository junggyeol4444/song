"""
곡 구조 — Intro / Verse / Chorus / Bridge ...

이게 왜 별도로 있어야 하는가:

  4번   구간마다 다른 장르를 지정한다
  11번  "마지막 후렴만 반키 올려줘"
  36번  Storyboard 가 구간 경계에서 장면을 바꾼다
  38번  구간마다 캐릭터 등장 강도를 다르게 한다

이 요구들이 전부 '구간'을 가리킨다. 구간이 없으면 '마지막 후렴'을 지목할
방법이 없어서, 사용자가 마디 번호를 직접 세야 한다.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

from ..core.units import MeterMap, TimeRange, TimeError
from .genre import GenreBlend, GenreError
from .theory import Key, TheoryError

SectionKind = Literal[
    "intro", "verse", "pre_chorus", "chorus", "post_chorus", "bridge",
    "instrumental", "solo", "breakdown", "drop", "build", "interlude", "outro",
]

SECTION_NAMES: dict[str, str] = {
    "intro": "인트로", "verse": "벌스", "pre_chorus": "프리코러스",
    "chorus": "후렴", "post_chorus": "포스트코러스", "bridge": "브릿지",
    "instrumental": "간주", "solo": "솔로", "breakdown": "브레이크다운",
    "drop": "드롭", "build": "빌드업", "interlude": "인터루드", "outro": "아웃트로",
}

# 구간별 기본 에너지 (0~1). 편곡 밀도와 음량의 기준이 된다.
DEFAULT_ENERGY: dict[str, float] = {
    "intro": 0.3, "verse": 0.45, "pre_chorus": 0.6, "chorus": 0.9,
    "post_chorus": 0.75, "bridge": 0.5, "instrumental": 0.7, "solo": 0.8,
    "breakdown": 0.25, "drop": 1.0, "build": 0.7, "interlude": 0.35, "outro": 0.35,
}

SECTION_COLORS: dict[str, str] = {
    "intro": "#7a8a99", "verse": "#4a90d9", "pre_chorus": "#d98c4a",
    "chorus": "#e8556d", "post_chorus": "#e87f9a", "bridge": "#9b6bd6",
    "instrumental": "#5aba9a", "solo": "#bad95a", "breakdown": "#6b7a8a",
    "drop": "#ff4d4d", "build": "#ffa64d", "interlude": "#8a99a8", "outro": "#7a8a99",
}


class StructureError(ValueError):
    """곡 구조 관련 오류."""


_section_id_counter = itertools.count(1)


@dataclass(slots=True)
class Section:
    """곡의 한 구간.

    위치는 마디로 잡는다. tick 으로 잡으면 박자표가 바뀔 때 어긋난다.
    사용자도 "8마디짜리 후렴"이라고 생각하지 "30720틱"이라고 생각하지 않는다.
    """

    kind: SectionKind
    start_bar: int              # 1부터
    length_bars: int
    label: str = ""             # "Verse 1", "Final Chorus" 처럼 사람이 붙이는 이름
    section_id: int = field(default_factory=lambda: next(_section_id_counter))

    genre: GenreBlend | None = None     # 구간별 장르 (4번). None 이면 곡 전체 장르를 따름
    key: Key | None = None              # 구간별 조성 (11번 반키 올리기)
    energy: float | None = None         # None 이면 종류별 기본값
    character_presence: str = ""        # 38번. "" 이면 곡 전체 설정을 따름
    notes_for_ai: str = ""              # "마지막 후렴은 웅장하게" 같은 지시

    def __post_init__(self) -> None:
        if self.kind not in SECTION_NAMES:
            raise StructureError(
                f"모르는 구간 종류입니다: {self.kind!r}\n"
                f"사용 가능: {', '.join(SECTION_NAMES)}"
            )
        if self.start_bar < 1:
            raise StructureError(f"시작 마디는 1 이상이어야 합니다: {self.start_bar}")
        if self.length_bars < 1:
            raise StructureError(f"길이는 1마디 이상이어야 합니다: {self.length_bars}")
        if self.energy is not None and not (0.0 <= self.energy <= 1.0):
            raise StructureError(f"에너지는 0~1 이어야 합니다: {self.energy}")
        if not self.label:
            self.label = SECTION_NAMES[self.kind]

    @property
    def end_bar(self) -> int:
        """이 구간 다음 마디 번호 (포함하지 않음)."""
        return self.start_bar + self.length_bars

    @property
    def kind_name(self) -> str:
        return SECTION_NAMES[self.kind]

    @property
    def color(self) -> str:
        return SECTION_COLORS.get(self.kind, "#888888")

    def effective_energy(self) -> float:
        return DEFAULT_ENERGY[self.kind] if self.energy is None else self.energy

    def tick_range(self, meter: MeterMap) -> TimeRange:
        return TimeRange(meter.bar_to_tick(self.start_bar), meter.bar_to_tick(self.end_bar))

    def seconds(self, meter: MeterMap, tempo) -> tuple[float, float]:
        span = self.tick_range(meter)
        return tempo.tick_to_seconds(span.start_tick), tempo.tick_to_seconds(span.end_tick)

    def contains_bar(self, bar: int) -> bool:
        return self.start_bar <= bar < self.end_bar

    def to_dict(self) -> dict:
        data: dict = {
            "kind": self.kind, "start_bar": self.start_bar,
            "length_bars": self.length_bars, "label": self.label, "id": self.section_id,
        }
        if self.genre is not None:
            data["genre"] = self.genre.to_dict()
        if self.key is not None:
            data["key"] = str(self.key)
        if self.energy is not None:
            data["energy"] = self.energy
        if self.character_presence:
            data["character_presence"] = self.character_presence
        if self.notes_for_ai:
            data["notes_for_ai"] = self.notes_for_ai
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        return cls(
            kind=data["kind"],
            start_bar=int(data["start_bar"]),
            length_bars=int(data["length_bars"]),
            label=data.get("label", ""),
            section_id=int(data.get("id", next(_section_id_counter))),
            genre=GenreBlend.from_dict(data["genre"]) if "genre" in data else None,
            key=Key.parse(data["key"]) if "key" in data else None,
            energy=float(data["energy"]) if "energy" in data else None,
            character_presence=data.get("character_presence", ""),
            notes_for_ai=data.get("notes_for_ai", ""),
        )

    def __str__(self) -> str:
        return f"{self.label} ({self.start_bar}~{self.end_bar - 1}마디)"


class SongStructure:
    """구간들의 모음. 빈틈 없이 이어지고 겹치지 않아야 한다.

    겹치거나 빈틈이 있으면 '이 마디는 어느 구간인가'에 답할 수 없고,
    그러면 구간별 장르·조성·캐릭터 설정이 전부 무너진다. 그래서 넣을 때 막는다.
    """

    __slots__ = ("_sections",)

    def __init__(self, sections: Iterable[Section] = ()) -> None:
        self._sections: list[Section] = sorted(sections, key=lambda s: s.start_bar)
        self._validate()

    def _validate(self) -> None:
        for index in range(len(self._sections) - 1):
            current, following = self._sections[index], self._sections[index + 1]
            if current.end_bar > following.start_bar:
                raise StructureError(
                    f"구간이 겹칩니다: '{current.label}'({current.start_bar}~"
                    f"{current.end_bar - 1}) 과 '{following.label}'"
                    f"({following.start_bar}~{following.end_bar - 1})"
                )

    def __len__(self) -> int:
        return len(self._sections)

    def __iter__(self) -> Iterator[Section]:
        return iter(self._sections)

    def __getitem__(self, index) -> Section:
        return self._sections[index]

    def __bool__(self) -> bool:
        return bool(self._sections)

    @property
    def sections(self) -> list[Section]:
        return list(self._sections)

    @property
    def total_bars(self) -> int:
        return max((s.end_bar for s in self._sections), default=1) - 1

    def add(self, section: Section) -> Section:
        for existing in self._sections:
            if (section.start_bar < existing.end_bar
                    and existing.start_bar < section.end_bar):
                raise StructureError(
                    f"'{section.label}'({section.start_bar}~{section.end_bar - 1}마디) 이 "
                    f"'{existing.label}'({existing.start_bar}~{existing.end_bar - 1}마디) 과 "
                    f"겹칩니다."
                )
        self._sections.append(section)
        self._sections.sort(key=lambda s: s.start_bar)
        return section

    def append(self, kind: SectionKind, length_bars: int, **kwargs) -> Section:
        """맨 뒤에 이어 붙인다. 곡을 처음 짤 때 가장 많이 쓴다."""
        start = self._sections[-1].end_bar if self._sections else 1
        return self.add(Section(kind, start, length_bars, **kwargs))

    def remove(self, section: Section) -> bool:
        try:
            self._sections.remove(section)
            return True
        except ValueError:
            return False

    def find(self, section_id: int) -> Section | None:
        for section in self._sections:
            if section.section_id == section_id:
                return section
        return None

    def at_bar(self, bar: int) -> Section | None:
        for section in self._sections:
            if section.contains_bar(bar):
                return section
        return None

    def at_tick(self, tick: int, meter: MeterMap) -> Section | None:
        bar, _ = meter.tick_to_bar_beat(tick)
        return self.at_bar(bar)

    def of_kind(self, kind: SectionKind) -> list[Section]:
        return [s for s in self._sections if s.kind == kind]

    def last_of_kind(self, kind: SectionKind) -> Section | None:
        """'마지막 후렴' 을 가리킬 때 쓴다 (11번)."""
        matching = self.of_kind(kind)
        return matching[-1] if matching else None

    def gaps(self) -> list[tuple[int, int]]:
        """구간 사이의 빈 마디들. 있으면 곡에 구멍이 있다는 뜻이다."""
        result: list[tuple[int, int]] = []
        if not self._sections:
            return result
        if self._sections[0].start_bar > 1:
            result.append((1, self._sections[0].start_bar - 1))
        for index in range(len(self._sections) - 1):
            end = self._sections[index].end_bar
            start = self._sections[index + 1].start_bar
            if end < start:
                result.append((end, start - 1))
        return result

    def renumber_labels(self) -> None:
        """같은 종류가 여러 번 나오면 번호를 붙인다. Verse 1, Verse 2 ...

        마지막 후렴은 'Final Chorus' 로 부른다. 사용자가 그렇게 부르기 때문이다.
        """
        counts: dict[str, int] = {}
        totals: dict[str, int] = {}
        for section in self._sections:
            totals[section.kind] = totals.get(section.kind, 0) + 1
        for section in self._sections:
            counts[section.kind] = counts.get(section.kind, 0) + 1
            index = counts[section.kind]
            total = totals[section.kind]
            base = SECTION_NAMES[section.kind]
            if total == 1:
                section.label = base
            elif index == total and total >= 2 and section.kind == "chorus":
                # '마지막 후렴' 은 사람이 실제로 쓰는 말이다 (11번의 예시가 그렇다).
                # 벌스는 그렇게 부르지 않는다. '벌스 2' 라고 한다.
                section.label = f"마지막 {base}"
            else:
                section.label = f"{base} {index}"

    def timeline_text(self, meter: MeterMap | None = None, tempo=None) -> str:
        """구조를 한눈에 보는 글자 표. 화면과 로그에 쓴다."""
        lines = []
        for section in self._sections:
            time_text = ""
            if meter is not None and tempo is not None:
                start, end = section.seconds(meter, tempo)
                time_text = f"  {int(start // 60)}:{start % 60:04.1f} ~ {int(end // 60)}:{end % 60:04.1f}"
            extra = []
            if section.genre is not None:
                extra.append(str(section.genre))
            if section.key is not None:
                extra.append(f"조성 {section.key}")
            if section.notes_for_ai:
                extra.append(f'"{section.notes_for_ai}"')
            suffix = ("  | " + " | ".join(extra)) if extra else ""
            lines.append(
                f"{section.label:<14} {section.start_bar:>3}~{section.end_bar - 1:<3}마디"
                f"  에너지 {section.effective_energy():.0%}{time_text}{suffix}"
            )
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self._sections]

    @classmethod
    def from_list(cls, data: Iterable[dict]) -> "SongStructure":
        return cls(Section.from_dict(item) for item in data)

    def __repr__(self) -> str:
        return f"SongStructure({len(self._sections)}구간, 총 {self.total_bars}마디)"


# 장르별로 흔한 곡 구조. AI 작곡이 시작점으로 쓴다.
STRUCTURE_TEMPLATES: dict[str, list[tuple[str, int]]] = {
    "pop": [("intro", 4), ("verse", 8), ("pre_chorus", 4), ("chorus", 8),
            ("verse", 8), ("pre_chorus", 4), ("chorus", 8), ("bridge", 8),
            ("chorus", 8), ("outro", 4)],
    "kpop": [("intro", 4), ("verse", 8), ("pre_chorus", 4), ("chorus", 8),
             ("post_chorus", 4), ("verse", 8), ("pre_chorus", 4), ("chorus", 8),
             ("bridge", 8), ("chorus", 8), ("outro", 4)],
    "ballad": [("intro", 4), ("verse", 8), ("pre_chorus", 4), ("chorus", 8),
               ("verse", 8), ("pre_chorus", 4), ("chorus", 8), ("bridge", 8),
               ("chorus", 8), ("outro", 6)],
    "rock": [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8),
             ("chorus", 8), ("solo", 8), ("chorus", 8), ("outro", 4)],
    "edm": [("intro", 8), ("build", 8), ("drop", 16), ("breakdown", 8),
            ("build", 8), ("drop", 16), ("outro", 8)],
    "anime": [("intro", 8), ("verse", 8), ("pre_chorus", 4), ("chorus", 8),
              ("instrumental", 4), ("verse", 8), ("pre_chorus", 4), ("chorus", 8),
              ("bridge", 8), ("chorus", 8), ("outro", 4)],
    "hiphop": [("intro", 4), ("verse", 16), ("chorus", 8), ("verse", 16),
               ("chorus", 8), ("verse", 8), ("chorus", 8), ("outro", 4)],
    "jazz": [("intro", 4), ("verse", 16), ("chorus", 16), ("solo", 32),
             ("chorus", 16), ("outro", 8)],
    "orchestral": [("intro", 8), ("verse", 16), ("build", 8), ("chorus", 16),
                   ("breakdown", 8), ("build", 8), ("chorus", 16), ("outro", 8)],
}


def build_structure(genre_name: str = "pop", repeat_final_chorus: bool = False) -> SongStructure:
    """장르에 맞는 기본 구조를 만든다. AI 작곡의 출발점이다."""
    from .genre import get_genre

    profile = get_genre(genre_name)
    template = STRUCTURE_TEMPLATES.get(profile.name)
    if template is None:
        # 등록된 틀이 없으면 가장 가까운 성격의 것을 쓴다
        if profile.drum_density > 0.8:
            template = STRUCTURE_TEMPLATES["edm"]
        elif profile.bpm_typical < 85:
            template = STRUCTURE_TEMPLATES["ballad"]
        else:
            template = STRUCTURE_TEMPLATES["pop"]
    structure = SongStructure()
    for kind, bars in template:
        structure.append(kind, bars)
    if repeat_final_chorus:
        last = structure.last_of_kind("chorus")
        if last is not None:
            outro = [s for s in structure if s.kind == "outro"]
            insert_at = outro[0].start_bar if outro else structure.total_bars + 1
            for section in structure:
                if section.start_bar >= insert_at:
                    section.start_bar += last.length_bars
            structure.add(Section("chorus", insert_at, last.length_bars))
    structure.renumber_labels()
    return structure
