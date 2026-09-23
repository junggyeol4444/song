"""
음 하나와 그 묶음.

여기가 프로그램 전체에서 가장 자주 만져지는 자료다. AI 작곡이 만들고,
편곡이 고치고, 노래 합성이 읽고, 립싱크가 시간을 가져가고, 사용자가 화면에서
끌어 옮긴다. 그래서 두 가지를 처음부터 넣는다.

1. 가사와 발음
   나중에 붙이려고 하면 음과 가사를 따로 관리하게 되고, 그때부터 둘이 어긋난다.
   음을 옮기면 가사도 같이 가야 하고, 그 시간이 그대로 입 모양 시간이 된다.

2. 표현
   세기만으로는 노래가 안 된다. 어디서 숨을 쉬고, 어디서 미끄러지고, 어디서
   떨림을 넣는지가 사람 목소리를 사람처럼 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Sequence

from ..music.theory import Pitch, TheoryError
from .units import PPQ, TimeRange, TimeError


class NoteError(ValueError):
    """음 관련 오류."""


# 노래 표현. 어느 것도 켜지 않으면 밋밋한 기계 소리가 된다.
EXPRESSION_KEYS: dict[str, str] = {
    "vibrato_depth": "비브라토 깊이(센트)",
    "vibrato_rate": "비브라토 속도(Hz)",
    "vibrato_onset": "비브라토가 들어오기까지(초)",
    "portamento": "앞 음에서 미끄러져 오는 시간(초)",
    "breathiness": "숨 섞임(0~1)",
    "tension": "성대 긴장(0~1). 높으면 날카롭고 낮으면 부드럽다",
    "attack_style": "음 시작 방식 (normal / soft / accent / scoop / fall)",
    "falsetto": "가성 정도(0~1)",
    "brightness": "음색 밝기(-1~1)",
    "loudness_curve": "음 안에서의 음량 변화 (flat / crescendo / diminuendo / swell)",
}

ATTACK_STYLES: tuple[str, ...] = ("normal", "soft", "accent", "scoop", "fall")
LOUDNESS_CURVES: tuple[str, ...] = ("flat", "crescendo", "diminuendo", "swell")


@dataclass(slots=True)
class Note:
    """음 하나.

    start_tick 과 duration_ticks 는 악보상의 시간이다. 실제 초는 TempoMap 이
    정한다. 템포를 바꿔도 음의 위치가 안 흔들리게 하려면 이래야 한다.
    """

    midi: int
    start_tick: int
    duration_ticks: int
    velocity: int = 90
    lyric: str = ""                 # 이 음에 붙는 가사 (보통 한 음절)
    phonemes: tuple[str, ...] = ()  # 발음 기호. 립싱크가 이걸 쓴다.
    expression: dict[str, float | str] = field(default_factory=dict)
    tied_to_next: bool = False      # 다음 음과 이어져 한 소리로 들린다
    word_end: bool = False          # 이 음의 가사 뒤에 띄어쓰기가 온다 (가사를 다시 보여줄 때)
    note_id: int = 0                # 프로젝트 안에서 고유. 편집 명령이 대상을 가리킬 때 쓴다.

    def __post_init__(self) -> None:
        if not (0 <= self.midi <= 127):
            raise NoteError(f"음높이는 0~127 이어야 합니다: {self.midi}")
        if self.start_tick < 0:
            raise NoteError(f"시작 위치는 0 이상이어야 합니다: {self.start_tick}")
        if self.duration_ticks <= 0:
            raise NoteError(
                f"길이는 0보다 커야 합니다: {self.duration_ticks} "
                f"(길이 0 인 음은 소리가 나지 않습니다)"
            )
        if not (0 <= self.velocity <= 127):
            raise NoteError(f"세기는 0~127 이어야 합니다: {self.velocity}")
        for key, value in self.expression.items():
            if key not in EXPRESSION_KEYS:
                raise NoteError(
                    f"알 수 없는 표현입니다: {key!r}\n"
                    f"사용 가능: {', '.join(sorted(EXPRESSION_KEYS))}"
                )
        style = self.expression.get("attack_style")
        if style is not None and style not in ATTACK_STYLES:
            raise NoteError(f"알 수 없는 시작 방식입니다: {style!r} (가능: {ATTACK_STYLES})")
        curve = self.expression.get("loudness_curve")
        if curve is not None and curve not in LOUDNESS_CURVES:
            raise NoteError(f"알 수 없는 음량 곡선입니다: {curve!r} (가능: {LOUDNESS_CURVES})")

    # ---- 조회 ----

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_ticks

    @property
    def range(self) -> TimeRange:
        return TimeRange(self.start_tick, self.end_tick)

    @property
    def pitch(self) -> Pitch:
        return Pitch.from_midi(self.midi)

    @property
    def beats(self) -> float:
        return self.duration_ticks / PPQ

    def seconds(self, tempo) -> tuple[float, float]:
        """실제 시작/끝 시각. 영상 쪽(립싱크, 자막)이 이걸 쓴다."""
        return tempo.tick_to_seconds(self.start_tick), tempo.tick_to_seconds(self.end_tick)

    def overlaps(self, other: "Note") -> bool:
        return self.start_tick < other.end_tick and other.start_tick < self.end_tick

    def expression_value(self, key: str, default: float | str = 0.0) -> float | str:
        if key not in EXPRESSION_KEYS:
            raise NoteError(f"알 수 없는 표현입니다: {key!r}")
        return self.expression.get(key, default)

    # ---- 변형 (원본을 바꾸지 않고 새로 만든다) ----

    def transposed(self, semitones: int) -> "Note":
        target = self.midi + semitones
        if not (0 <= target <= 127):
            raise NoteError(
                f"{self.pitch} 를 {semitones:+d}반음 옮기면 MIDI 범위를 벗어납니다 ({target})."
            )
        return replace(self, midi=target)

    def moved(self, delta_ticks: int) -> "Note":
        if self.start_tick + delta_ticks < 0:
            raise NoteError(f"{delta_ticks} 만큼 옮기면 0 앞으로 갑니다.")
        return replace(self, start_tick=self.start_tick + delta_ticks)

    def resized(self, duration_ticks: int) -> "Note":
        return replace(self, duration_ticks=duration_ticks)

    def with_velocity(self, velocity: int) -> "Note":
        return replace(self, velocity=velocity)

    def with_lyric(self, lyric: str, phonemes: Sequence[str] = (),
                   word_end: bool = False) -> "Note":
        return replace(self, lyric=lyric, phonemes=tuple(phonemes),
                       word_end=bool(lyric) and word_end)

    def with_expression(self, **values: float | str) -> "Note":
        merged = dict(self.expression)
        merged.update(values)
        return replace(self, expression=merged)

    def copy(self) -> "Note":
        return replace(self, expression=dict(self.expression))

    # ---- 저장 ----

    def to_dict(self) -> dict:
        data: dict = {
            "midi": self.midi,
            "start": self.start_tick,
            "length": self.duration_ticks,
            "velocity": self.velocity,
        }
        if self.lyric:
            data["lyric"] = self.lyric
        if self.phonemes:
            data["phonemes"] = list(self.phonemes)
        if self.expression:
            data["expression"] = dict(self.expression)
        if self.tied_to_next:
            data["tied"] = True
        if self.word_end:
            data["word_end"] = True
        if self.note_id:
            data["id"] = self.note_id
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        try:
            return cls(
                midi=int(data["midi"]),
                start_tick=int(data["start"]),
                duration_ticks=int(data["length"]),
                velocity=int(data.get("velocity", 90)),
                lyric=str(data.get("lyric", "")),
                phonemes=tuple(data.get("phonemes", ())),
                expression=dict(data.get("expression", {})),
                tied_to_next=bool(data.get("tied", False)),
                word_end=bool(data.get("word_end", False)),
                note_id=int(data.get("id", 0)),
            )
        except KeyError as error:
            raise NoteError(f"음 자료에 {error} 항목이 없습니다: {data}") from error

    def __str__(self) -> str:
        text = f"{self.pitch} @{self.start_tick} ({self.beats:g}박)"
        if self.lyric:
            text += f" '{self.lyric}'"
        return text


class NoteList:
    """한 구간 안의 음들. 항상 시작 위치 순으로 정렬돼 있다.

    정렬을 유지하는 이유: 렌더러와 화면이 매번 다시 정렬하면 음이 수천 개일 때
    느려진다. 넣을 때 한 번만 자리를 찾으면 된다.
    """

    __slots__ = ("_notes",)

    def __init__(self, notes: Iterable[Note] = ()) -> None:
        self._notes: list[Note] = sorted(notes, key=lambda n: (n.start_tick, n.midi))

    # ---- 조회 ----

    def __len__(self) -> int:
        return len(self._notes)

    def __iter__(self) -> Iterator[Note]:
        return iter(self._notes)

    def __getitem__(self, index):
        return self._notes[index]

    def __bool__(self) -> bool:
        return bool(self._notes)

    @property
    def notes(self) -> list[Note]:
        return list(self._notes)

    @property
    def start_tick(self) -> int:
        return self._notes[0].start_tick if self._notes else 0

    @property
    def end_tick(self) -> int:
        return max((n.end_tick for n in self._notes), default=0)

    @property
    def length_ticks(self) -> int:
        return max(0, self.end_tick - self.start_tick)

    @property
    def pitch_range(self) -> tuple[int, int] | None:
        if not self._notes:
            return None
        return min(n.midi for n in self._notes), max(n.midi for n in self._notes)

    def in_range(self, time_range: TimeRange) -> list[Note]:
        """구간에 걸치는 음들. 구간 안에서 시작하지 않아도 걸쳐 있으면 포함한다."""
        return [n for n in self._notes if n.start_tick < time_range.end_tick
                and n.end_tick > time_range.start_tick]

    def at_tick(self, tick: int) -> list[Note]:
        """그 순간 울리고 있는 음들. 코드 분석과 화면 재생 위치에 쓴다."""
        return [n for n in self._notes if n.start_tick <= tick < n.end_tick]

    def find(self, note_id: int) -> Note | None:
        for note in self._notes:
            if note.note_id == note_id:
                return note
        return None

    def overlapping_pairs(self) -> list[tuple[Note, Note]]:
        """같은 음높이가 겹치는 쌍. 단선율 트랙에서 이러면 소리가 이상해진다."""
        result: list[tuple[Note, Note]] = []
        for index, first in enumerate(self._notes):
            for second in self._notes[index + 1 :]:
                if second.start_tick >= first.end_tick:
                    break
                if first.midi == second.midi and first.overlaps(second):
                    result.append((first, second))
        return result

    # ---- 편집 ----

    def add(self, note: Note) -> Note:
        """정렬을 유지하며 넣는다."""
        import bisect

        key = (note.start_tick, note.midi)
        keys = [(n.start_tick, n.midi) for n in self._notes]
        self._notes.insert(bisect.bisect_left(keys, key), note)
        return note

    def extend(self, notes: Iterable[Note]) -> None:
        self._notes.extend(notes)
        self._notes.sort(key=lambda n: (n.start_tick, n.midi))

    def remove(self, note: Note) -> bool:
        try:
            self._notes.remove(note)
            return True
        except ValueError:
            return False

    def remove_id(self, note_id: int) -> bool:
        for index, note in enumerate(self._notes):
            if note.note_id == note_id:
                del self._notes[index]
                return True
        return False

    def replace_note(self, old: Note, new: Note) -> bool:
        if not self.remove(old):
            return False
        self.add(new)
        return True

    def clear(self) -> None:
        self._notes.clear()

    # ---- 일괄 변형 ----

    def transposed(self, semitones: int) -> "NoteList":
        return NoteList(n.transposed(semitones) for n in self._notes)

    def moved(self, delta_ticks: int) -> "NoteList":
        return NoteList(n.moved(delta_ticks) for n in self._notes)

    def quantized(self, grid_ticks: int, strength: float = 1.0) -> "NoteList":
        """박자에 맞춘다. strength 1.0 이면 완전히, 0.5 면 절반만 당긴다.

        완전히 맞추면 기계적으로 들린다. 사람 연주의 미세한 어긋남이 리듬의
        느낌을 만들기 때문이다. 그래서 정도를 고를 수 있게 한다.
        """
        if grid_ticks <= 0:
            raise NoteError(f"격자는 0보다 커야 합니다: {grid_ticks}")
        if not (0.0 <= strength <= 1.0):
            raise NoteError(f"강도는 0~1 이어야 합니다: {strength}")
        result: list[Note] = []
        for note in self._notes:
            target = int(round(note.start_tick / grid_ticks)) * grid_ticks
            moved = int(round(note.start_tick + (target - note.start_tick) * strength))
            result.append(replace(note, start_tick=max(0, moved)))
        return NoteList(result)

    def with_velocity_scaled(self, factor: float) -> "NoteList":
        return NoteList(
            n.with_velocity(max(1, min(127, int(round(n.velocity * factor)))))
            for n in self._notes
        )

    def legato(self, gap_ticks: int = 0) -> "NoteList":
        """다음 음까지 길이를 늘려 끊김 없이 이어지게 한다."""
        if gap_ticks < 0:
            raise NoteError("간격은 0 이상이어야 합니다.")
        ordered = sorted(self._notes, key=lambda n: n.start_tick)
        result: list[Note] = []
        for index, note in enumerate(ordered):
            if index + 1 < len(ordered):
                next_start = ordered[index + 1].start_tick
                new_length = max(1, next_start - note.start_tick - gap_ticks)
                result.append(note.resized(new_length))
            else:
                result.append(note)
        return NoteList(result)

    def humanized(self, timing_ticks: int = 8, velocity_spread: int = 8,
                  seed: int | None = None) -> "NoteList":
        """사람이 친 것처럼 미세하게 흐트러뜨린다.

        완벽하게 정렬된 연주는 사람이 바로 알아챈다. 이 함수는 그 어긋남을
        만들되, 음의 순서가 뒤바뀌지 않을 만큼만 움직인다.
        """
        import random

        generator = random.Random(seed)
        result: list[Note] = []
        for note in self._notes:
            shift = generator.randint(-timing_ticks, timing_ticks)
            start = max(0, note.start_tick + shift)
            velocity = max(1, min(127, note.velocity
                                  + generator.randint(-velocity_spread, velocity_spread)))
            result.append(replace(note, start_tick=start, velocity=velocity))
        return NoteList(result)

    # ---- 저장 ----

    def to_list(self) -> list[dict]:
        return [n.to_dict() for n in self._notes]

    @classmethod
    def from_list(cls, data: Sequence[dict]) -> "NoteList":
        return cls(Note.from_dict(item) for item in data)

    def __repr__(self) -> str:
        if not self._notes:
            return "NoteList(비어 있음)"
        low, high = self.pitch_range or (0, 0)
        return (
            f"NoteList({len(self._notes)}개, "
            f"{Pitch.from_midi(low)}~{Pitch.from_midi(high)}, "
            f"{self.start_tick}~{self.end_tick}tick)"
        )
