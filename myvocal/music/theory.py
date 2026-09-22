"""
음악 이론 엔진.

여기가 틀리면 그 위의 작곡·편곡·전조·코드 분석이 전부 틀린다.
그래서 관습이 아니라 이론대로 구현한다.

  - 음높이는 MIDI 노트 번호(0~127)로 다룬다. C4 = 60, A4 = 69 = 440Hz.
  - 이명동음(C# / Db)을 구분한다. Key 에 따라 올바른 표기를 고른다.
  - 코드는 '근음 + 인터벌 집합'으로 저장한다. 심볼은 거기서 생성/역파싱한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

# --------------------------------------------------------------------------
# 음이름
# --------------------------------------------------------------------------

# 온음계 7음의 피치클래스 (C=0 기준)
NATURAL_PC: dict[str, int] = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NATURAL_ORDER: tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")

_ACCIDENTAL_VALUE: dict[str, int] = {"#": 1, "♯": 1, "b": -1, "♭": -1, "x": 2, "𝄪": 2, "": 0}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#♯b♭x𝄪]*)(-?\d+)?$")

MIDDLE_C: int = 60          # C4
CONCERT_A: int = 69         # A4
CONCERT_A_HZ: float = 440.0


class TheoryError(ValueError):
    """음악 이론 관련 오류."""


@dataclass(frozen=True, slots=True, order=True)
class Pitch:
    """이명동음을 구분하는 음높이.

    midi 만으로는 C#4 와 Db4 를 구분할 수 없다. 조성에 맞는 악보 표기와
    성부 진행(voice leading) 판단을 위해 철자(letter+alter)를 따로 들고 있다.
    """

    midi: int
    letter: str = "C"
    alter: int = 0          # -2 겹내림 ~ +2 겹올림

    def __post_init__(self) -> None:
        if not (0 <= self.midi <= 127):
            raise TheoryError(f"MIDI 음높이는 0~127 범위여야 합니다: {self.midi}")
        if self.letter not in NATURAL_PC:
            raise TheoryError(f"음이름은 A~G 여야 합니다: {self.letter!r}")
        if not (-2 <= self.alter <= 2):
            raise TheoryError(f"임시표는 -2~+2 범위여야 합니다: {self.alter}")
        # 철자와 midi 가 실제로 같은 음을 가리키는지 확인한다.
        expected_pc = (NATURAL_PC[self.letter] + self.alter) % 12
        if self.midi % 12 != expected_pc:
            raise TheoryError(
                f"철자와 음높이가 어긋납니다: midi={self.midi}(pc {self.midi % 12}) "
                f"인데 {self.letter}{'#' * self.alter or 'b' * -self.alter} 는 pc {expected_pc}"
            )

    # ---- 생성 ----

    @classmethod
    def from_midi(cls, midi: int, prefer_flat: bool = False) -> "Pitch":
        """MIDI 번호로부터. 검은건반은 prefer_flat 에 따라 #/b 를 고른다."""
        if not (0 <= midi <= 127):
            raise TheoryError(f"MIDI 음높이는 0~127 범위여야 합니다: {midi}")
        pc = midi % 12
        sharp_spell = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
                       ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]
        flat_spell = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
                      ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0)]
        letter, alter = (flat_spell if prefer_flat else sharp_spell)[pc]
        return cls(midi, letter, alter)

    @classmethod
    def parse(cls, text: str) -> "Pitch":
        """'C4', 'Bb3', 'F#5', 'Ebb2' 같은 표기를 읽는다. 옥타브 생략 시 4옥타브."""
        match = _NOTE_RE.match(text.strip())
        if not match:
            raise TheoryError(f"음높이 표기를 읽을 수 없습니다: {text!r}")
        letter = match.group(1).upper()
        alter = sum(_ACCIDENTAL_VALUE[ch] for ch in match.group(2))
        octave = int(match.group(3)) if match.group(3) is not None else 4
        midi = (octave + 1) * 12 + NATURAL_PC[letter] + alter
        if not (0 <= midi <= 127):
            raise TheoryError(f"{text!r} 는 MIDI 범위를 벗어납니다 (계산값 {midi})")
        return cls(midi, letter, alter)

    # ---- 조회 ----

    @property
    def pitch_class(self) -> int:
        return self.midi % 12

    @property
    def octave(self) -> int:
        """과학적 음높이 표기. C4 = 60."""
        return self.midi // 12 - 1

    @property
    def frequency(self) -> float:
        """평균율 기준 주파수(Hz). A4 = 440Hz."""
        return CONCERT_A_HZ * (2.0 ** ((self.midi - CONCERT_A) / 12.0))

    @property
    def accidental_text(self) -> str:
        if self.alter > 0:
            return "#" * self.alter
        if self.alter < 0:
            return "b" * -self.alter
        return ""

    @property
    def name(self) -> str:
        """옥타브 없는 이름. 'C#', 'Bb'."""
        return f"{self.letter}{self.accidental_text}"

    @property
    def diatonic_step(self) -> int:
        """온음계 계단 번호. C0=0 부터 단조 증가. 인터벌의 도수 계산에 쓴다."""
        return self.octave * 7 + NATURAL_ORDER.index(self.letter)

    # ---- 연산 ----

    def transpose(self, semitones: int, prefer_flat: bool | None = None) -> "Pitch":
        """반음 단위 이조. 철자는 다시 계산한다 (11번 '반키 올려줘')."""
        if prefer_flat is None:
            prefer_flat = self.alter < 0
        return Pitch.from_midi(self.midi + semitones, prefer_flat=prefer_flat)

    def transpose_diatonic(self, steps: int, scale: "Scale") -> "Pitch":
        """음계 안에서 n도 이동. 조성을 유지한 멜로디 변형에 쓴다."""
        return scale.step(self, steps)

    def with_octave(self, octave: int) -> "Pitch":
        midi = (octave + 1) * 12 + NATURAL_PC[self.letter] + self.alter
        return Pitch(midi, self.letter, self.alter)

    def enharmonic_equals(self, other: "Pitch") -> bool:
        """철자는 달라도 같은 소리인가."""
        return self.midi == other.midi

    def __str__(self) -> str:
        return f"{self.name}{self.octave}"

    def __repr__(self) -> str:
        return f"Pitch({self})"


def spell_interval(root: Pitch, semitones: int, degree: int) -> Pitch:
    """근음에서 semitones 떨어진 음을 '몇 도인가'로 읽어 철자를 정한다.

    이게 없으면 Cm7 의 7음이 A#(증6도)로 나온다. 실제로는 Bb(단7도)다.
    소리는 같지만 악보와 전조가 깨지고, 코드 심볼 생성도 틀린다.

        spell_interval(C4, 3, 3)  -> Eb4   (단3도)
        spell_interval(C4, 10, 7) -> Bb4   (단7도)
        spell_interval(C4, 6, 5)  -> Gb4   (감5도)
        spell_interval(C4, 6, 4)  -> F#4   (증4도)
        spell_interval(C4, 18, 11)-> F#5   (#11)
    """
    midi = root.midi + semitones
    if not (0 <= midi <= 127):
        raise TheoryError(f"{root} 에서 {semitones}반음은 MIDI 범위를 벗어납니다 ({midi}).")
    letter = NATURAL_ORDER[(NATURAL_ORDER.index(root.letter) + (degree - 1)) % 7]
    alter = (midi % 12) - NATURAL_PC[letter]
    if alter > 6:
        alter -= 12
    elif alter < -6:
        alter += 12
    if -2 <= alter <= 2:
        return Pitch(midi, letter, alter)
    # 겹올림/겹내림을 넘어가면 (예: 이론상 Fbb) 철자를 포기하고 소리 기준으로 적는다
    return Pitch.from_midi(midi, prefer_flat=alter < 0)


def shift_octave(pitch: Pitch, octaves: int) -> Pitch:
    """철자는 그대로 두고 옥타브만 옮긴다. 전위/보이싱에서 쓴다."""
    return Pitch(pitch.midi + 12 * octaves, pitch.letter, pitch.alter)


def midi_to_frequency(midi: float) -> float:
    """정수가 아닌 값도 받는다 (피치 벤드, 마이크로톤)."""
    return CONCERT_A_HZ * (2.0 ** ((midi - CONCERT_A) / 12.0))


def frequency_to_midi(hz: float) -> float:
    """주파수 -> MIDI 번호(소수). 목소리 피치 분석(6번)에서 쓴다."""
    if hz <= 0:
        raise TheoryError(f"주파수는 양수여야 합니다: {hz}")
    import math

    return 12.0 * math.log2(hz / CONCERT_A_HZ) + CONCERT_A


def cents_between(hz_a: float, hz_b: float) -> float:
    """두 주파수의 센트 차이. 음정 정확도 측정에 쓴다."""
    import math

    if hz_a <= 0 or hz_b <= 0:
        raise TheoryError("주파수는 양수여야 합니다.")
    return 1200.0 * math.log2(hz_b / hz_a)


# --------------------------------------------------------------------------
# 인터벌
# --------------------------------------------------------------------------

# 도수별 완전/장음정 반음 수 (1도~8도)
_PERFECT_DEGREES = {1, 4, 5, 8}
_BASE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 8: 12}


@dataclass(frozen=True, slots=True)
class Interval:
    """음정. 도수와 성질을 구분한다 (증4도와 감5도는 다른 음정이다)."""

    degree: int      # 1 = 완전1도, 3 = 3도, 8 = 옥타브
    quality: str     # "P" 완전 / "M" 장 / "m" 단 / "A" 증 / "d" 감

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise TheoryError(f"도수는 1 이상이어야 합니다: {self.degree}")
        simple = ((self.degree - 1) % 7) + 1
        if simple in _PERFECT_DEGREES and self.quality in ("M", "m"):
            raise TheoryError(f"{self.degree}도에는 장/단이 없습니다 (완전/증/감만 가능)")
        if simple not in _PERFECT_DEGREES and self.quality == "P":
            raise TheoryError(f"{self.degree}도에는 완전이 없습니다 (장/단/증/감만 가능)")
        if self.quality not in ("P", "M", "m", "A", "d"):
            raise TheoryError(f"알 수 없는 음정 성질: {self.quality!r}")

    @property
    def semitones(self) -> int:
        simple = ((self.degree - 1) % 7) + 1
        octaves = (self.degree - 1) // 7
        base = _BASE_SEMITONES[simple]
        if simple in _PERFECT_DEGREES:
            offset = {"P": 0, "A": 1, "d": -1}[self.quality]
        else:
            offset = {"M": 0, "m": -1, "A": 1, "d": -2}[self.quality]
        return base + offset + 12 * octaves

    @classmethod
    def between(cls, lower: Pitch, upper: Pitch) -> "Interval":
        """두 음 사이의 실제 음정을 철자까지 고려해 구한다."""
        step_diff = upper.diatonic_step - lower.diatonic_step
        semi_diff = upper.midi - lower.midi
        if step_diff < 0:
            raise TheoryError("위 음이 아래 음보다 낮습니다.")
        degree = step_diff + 1
        simple = ((degree - 1) % 7) + 1
        octaves = (degree - 1) // 7
        base = _BASE_SEMITONES[simple] + 12 * octaves
        offset = semi_diff - base
        if simple in _PERFECT_DEGREES:
            quality = {0: "P", 1: "A", -1: "d"}.get(offset)
        else:
            quality = {0: "M", -1: "m", 1: "A", -2: "d"}.get(offset)
        if quality is None:
            raise TheoryError(
                f"표현할 수 없는 음정입니다: {lower} -> {upper} ({degree}도, 오차 {offset}반음)"
            )
        return cls(degree, quality)

    def __str__(self) -> str:
        return f"{self.quality}{self.degree}"


# --------------------------------------------------------------------------
# 음계
# --------------------------------------------------------------------------

# 이름 -> 근음으로부터의 반음 간격
SCALE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major":            (0, 2, 4, 5, 7, 9, 11),
    "natural_minor":    (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor":   (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor":    (0, 2, 3, 5, 7, 9, 11),
    "dorian":           (0, 2, 3, 5, 7, 9, 10),
    "phrygian":         (0, 1, 3, 5, 7, 8, 10),
    "lydian":           (0, 2, 4, 6, 7, 9, 11),
    "mixolydian":       (0, 2, 4, 5, 7, 9, 10),
    "locrian":          (0, 1, 3, 5, 6, 8, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues":            (0, 3, 5, 6, 7, 10),
    "whole_tone":       (0, 2, 4, 6, 8, 10),
    "chromatic":        tuple(range(12)),
    "harmonic_major":   (0, 2, 4, 5, 7, 8, 11),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),   # 록/메탈, 애니송에서 자주 쓴다
    "lydian_dominant":  (0, 2, 4, 6, 7, 9, 10),
    "altered":          (0, 1, 3, 4, 6, 8, 10),
    "hirajoshi":        (0, 2, 3, 7, 8),           # 일본 음계 (J-Pop / Anime)
    "in_sen":           (0, 1, 5, 7, 10),
    "yo":               (0, 2, 5, 7, 9),
}

# 각 음계음을 '몇 도로 읽을 것인가'. 철자 결정에 쓴다.
# 예: 단5음계 C 는 1, b3, 4, 5, b7 도 -> C Eb F G Bb (D#/A# 이 아니다)
SCALE_DEGREES: dict[str, tuple[int, ...] | None] = {
    "major":             (1, 2, 3, 4, 5, 6, 7),
    "natural_minor":     (1, 2, 3, 4, 5, 6, 7),
    "harmonic_minor":    (1, 2, 3, 4, 5, 6, 7),
    "melodic_minor":     (1, 2, 3, 4, 5, 6, 7),
    "dorian":            (1, 2, 3, 4, 5, 6, 7),
    "phrygian":          (1, 2, 3, 4, 5, 6, 7),
    "lydian":            (1, 2, 3, 4, 5, 6, 7),
    "mixolydian":        (1, 2, 3, 4, 5, 6, 7),
    "locrian":           (1, 2, 3, 4, 5, 6, 7),
    "major_pentatonic":  (1, 2, 3, 5, 6),
    "minor_pentatonic":  (1, 3, 4, 5, 7),
    "blues":             (1, 3, 4, 5, 5, 7),      # b5 와 5 가 함께 있다
    "whole_tone":        (1, 2, 3, 4, 5, 6),
    "chromatic":         None,                     # 반음계는 도수 철자가 성립하지 않는다
    "harmonic_major":    (1, 2, 3, 4, 5, 6, 7),
    "phrygian_dominant": (1, 2, 3, 4, 5, 6, 7),
    "lydian_dominant":   (1, 2, 3, 4, 5, 6, 7),
    "altered":           (1, 2, 3, 4, 5, 6, 7),
    "hirajoshi":         (1, 2, 3, 5, 6),
    "in_sen":            (1, 2, 4, 5, 7),
    "yo":                (1, 2, 4, 5, 6),
}

SCALE_DISPLAY_NAMES: dict[str, str] = {
    "major": "장음계", "natural_minor": "자연 단음계", "harmonic_minor": "화성 단음계",
    "melodic_minor": "가락 단음계", "dorian": "도리안", "phrygian": "프리지안",
    "lydian": "리디안", "mixolydian": "믹솔리디안", "locrian": "로크리안",
    "major_pentatonic": "장5음계", "minor_pentatonic": "단5음계", "blues": "블루스",
    "whole_tone": "온음음계", "chromatic": "반음계", "harmonic_major": "화성 장음계",
    "phrygian_dominant": "프리지안 도미넌트", "lydian_dominant": "리디안 도미넌트",
    "altered": "얼터드", "hirajoshi": "히라조시", "in_sen": "인센", "yo": "요나누키",
}


@dataclass(frozen=True, slots=True)
class Scale:
    """근음 + 음계 종류. 철자까지 올바르게 만들어낸다."""

    tonic: Pitch
    kind: str = "major"

    def __post_init__(self) -> None:
        if self.kind not in SCALE_INTERVALS:
            raise TheoryError(
                f"알 수 없는 음계입니다: {self.kind!r} "
                f"(사용 가능: {', '.join(sorted(SCALE_INTERVALS))})"
            )

    @property
    def intervals(self) -> tuple[int, ...]:
        return SCALE_INTERVALS[self.kind]

    @property
    def size(self) -> int:
        return len(self.intervals)

    @property
    def display_name(self) -> str:
        return f"{self.tonic.name} {SCALE_DISPLAY_NAMES.get(self.kind, self.kind)}"

    def pitches(self, octave: int | None = None) -> list[Pitch]:
        """한 옥타브분 음들. 각 음을 몇 도로 읽는지에 따라 철자를 정한다."""
        base = self.tonic if octave is None else self.tonic.with_octave(octave)
        degrees = SCALE_DEGREES.get(self.kind)
        if degrees is None:
            # 반음계처럼 도수 철자가 없는 음계는 소리 기준으로 적는다
            return [
                Pitch.from_midi(base.midi + s, prefer_flat=base.alter < 0)
                for s in self.intervals
            ]
        if len(degrees) != len(self.intervals):
            raise TheoryError(f"{self.kind} 음계의 도수표 길이가 음 개수와 다릅니다.")
        return [
            spell_interval(base, semitones, degree)
            for semitones, degree in zip(self.intervals, degrees)
        ]

    def contains(self, pitch: Pitch) -> bool:
        """이 음이 음계에 속하는가 (옥타브 무시)."""
        return (pitch.midi - self.tonic.midi) % 12 in self.intervals

    def degree_of(self, pitch: Pitch) -> int | None:
        """음계 안에서 몇 번째 음인가 (1부터). 음계 밖이면 None."""
        offset = (pitch.midi - self.tonic.midi) % 12
        if offset in self.intervals:
            return self.intervals.index(offset) + 1
        return None

    def pitch_at_degree(self, degree: int, octave: int | None = None) -> Pitch:
        """n번째 음계음. degree 는 1부터, 음계 크기를 넘으면 옥타브가 올라간다."""
        if degree < 1:
            raise TheoryError(f"음계 도수는 1 이상이어야 합니다: {degree}")
        index = (degree - 1) % self.size
        octave_shift = (degree - 1) // self.size
        base_pitches = self.pitches(octave)
        pitch = base_pitches[index]
        return Pitch.from_midi(pitch.midi + 12 * octave_shift, prefer_flat=pitch.alter < 0) \
            if octave_shift else pitch

    def step(self, pitch: Pitch, steps: int) -> Pitch:
        """음계를 따라 n칸 이동. 음계 밖의 음이면 가장 가까운 음계음으로 먼저 붙인다."""
        current = self.degree_of(pitch)
        if current is None:
            pitch = self.snap(pitch)
            current = self.degree_of(pitch)
            assert current is not None
        octave_of_pitch = pitch.octave
        target_degree = current + steps
        octave_shift = (target_degree - 1) // self.size
        index = (target_degree - 1) % self.size
        # 근음을 pitch 와 같은 옥타브 기준으로 놓고 계산
        tonic_here = self.tonic.with_octave(octave_of_pitch)
        if tonic_here.midi > pitch.midi:
            tonic_here = Pitch.from_midi(tonic_here.midi - 12, prefer_flat=tonic_here.alter < 0)
        spelled = Scale(tonic_here, self.kind).pitches()
        result = spelled[index]
        return Pitch.from_midi(result.midi + 12 * octave_shift, prefer_flat=result.alter < 0) \
            if octave_shift else result

    def snap(self, pitch: Pitch) -> Pitch:
        """가장 가까운 음계음으로 보정. 위아래 같은 거리면 아래를 택한다."""
        for distance in range(0, 7):
            for direction in (-1, 1) if distance else (0,):
                candidate = pitch.midi + direction * distance
                if 0 <= candidate <= 127 and (candidate - self.tonic.midi) % 12 in self.intervals:
                    return Pitch.from_midi(candidate, prefer_flat=self.tonic.alter < 0)
        raise TheoryError(f"음계음을 찾지 못했습니다: {pitch}")

    def transpose(self, semitones: int) -> "Scale":
        return Scale(self.tonic.transpose(semitones), self.kind)

    @property
    def is_minor(self) -> bool:
        return 3 in self.intervals and 4 not in self.intervals

    def __str__(self) -> str:
        return self.display_name


# --------------------------------------------------------------------------
# 코드
# --------------------------------------------------------------------------

# 코드 종류 -> 근음으로부터의 반음 간격.
# 순서는 심볼 파싱 우선순위에 쓰이므로 긴 이름이 앞에 오도록 아래에서 정렬한다.
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    # 3화음
    "":          (0, 4, 7),           # 메이저
    "m":         (0, 3, 7),
    "dim":       (0, 3, 6),
    "aug":       (0, 4, 8),
    "5":         (0, 7),              # 파워코드 (Rock / Metal / Punk)
    "sus2":      (0, 2, 7),
    "sus4":      (0, 5, 7),
    # 6화음
    "6":         (0, 4, 7, 9),
    "m6":        (0, 3, 7, 9),
    "6/9":       (0, 4, 7, 9, 14),
    # 7화음
    "maj7":      (0, 4, 7, 11),
    "7":         (0, 4, 7, 10),
    "m7":        (0, 3, 7, 10),
    "mMaj7":     (0, 3, 7, 11),
    "m7b5":      (0, 3, 6, 10),       # 하프 디미니시
    "dim7":      (0, 3, 6, 9),
    "aug7":      (0, 4, 8, 10),
    "maj7#5":    (0, 4, 8, 11),
    "7sus4":     (0, 5, 7, 10),
    "7sus2":     (0, 2, 7, 10),
    # 텐션
    "add9":      (0, 4, 7, 14),
    "madd9":     (0, 3, 7, 14),
    "add11":     (0, 4, 7, 17),
    "9":         (0, 4, 7, 10, 14),
    "maj9":      (0, 4, 7, 11, 14),
    "m9":        (0, 3, 7, 10, 14),
    "7b9":       (0, 4, 7, 10, 13),
    "7#9":       (0, 4, 7, 10, 15),
    "7#11":      (0, 4, 7, 10, 18),
    "7b13":      (0, 4, 7, 10, 20),
    "11":        (0, 4, 7, 10, 14, 17),
    "m11":       (0, 3, 7, 10, 14, 17),
    "maj11":     (0, 4, 7, 11, 14, 17),
    "13":        (0, 4, 7, 10, 14, 21),
    "maj13":     (0, 4, 7, 11, 14, 21),
    "m13":       (0, 3, 7, 10, 14, 21),
    "alt":       (0, 4, 10, 13, 20),  # 7alt: b9 #5 생략형
}

# 각 구성음을 '몇 도로 읽을 것인가'. CHORD_INTERVALS 와 길이가 같아야 한다.
# 이게 있어야 Cm7 의 7음이 Bb(단7도)로 나온다. A#(증6도)가 아니다.
# 9/11/13 도는 옥타브 위의 2/4/6 도다.
CHORD_DEGREES: dict[str, tuple[int, ...]] = {
    "":          (1, 3, 5),
    "m":         (1, 3, 5),
    "dim":       (1, 3, 5),              # C Eb Gb
    "aug":       (1, 3, 5),              # C E G#
    "5":         (1, 5),
    "sus2":      (1, 2, 5),
    "sus4":      (1, 4, 5),
    "6":         (1, 3, 5, 6),
    "m6":        (1, 3, 5, 6),
    "6/9":       (1, 3, 5, 6, 9),
    "maj7":      (1, 3, 5, 7),
    "7":         (1, 3, 5, 7),           # C E G Bb
    "m7":        (1, 3, 5, 7),           # C Eb G Bb
    "mMaj7":     (1, 3, 5, 7),
    "m7b5":      (1, 3, 5, 7),           # C Eb Gb Bb
    "dim7":      (1, 3, 5, 7),           # C Eb Gb Bbb  (감7도)
    "aug7":      (1, 3, 5, 7),
    "maj7#5":    (1, 3, 5, 7),
    "7sus4":     (1, 4, 5, 7),
    "7sus2":     (1, 2, 5, 7),
    "add9":      (1, 3, 5, 9),
    "madd9":     (1, 3, 5, 9),
    "add11":     (1, 3, 5, 11),
    "9":         (1, 3, 5, 7, 9),
    "maj9":      (1, 3, 5, 7, 9),
    "m9":        (1, 3, 5, 7, 9),
    "7b9":       (1, 3, 5, 7, 9),        # 9음이 Db
    "7#9":       (1, 3, 5, 7, 9),        # 9음이 D#
    "7#11":      (1, 3, 5, 7, 11),       # 11음이 F#
    "7b13":      (1, 3, 5, 7, 13),       # 13음이 Ab
    "11":        (1, 3, 5, 7, 9, 11),
    "m11":       (1, 3, 5, 7, 9, 11),
    "maj11":     (1, 3, 5, 7, 9, 11),
    "13":        (1, 3, 5, 7, 9, 13),
    "maj13":     (1, 3, 5, 7, 9, 13),
    "m13":       (1, 3, 5, 7, 9, 13),
    "alt":       (1, 3, 7, 9, 13),
}

CHORD_DISPLAY_NAMES: dict[str, str] = {
    "": "메이저", "m": "마이너", "dim": "디미니시", "aug": "어그먼트", "5": "파워코드",
    "sus2": "서스2", "sus4": "서스4", "6": "6th", "m6": "마이너6th", "6/9": "6/9",
    "maj7": "메이저7", "7": "도미넌트7", "m7": "마이너7", "mMaj7": "마이너메이저7",
    "m7b5": "하프디미니시", "dim7": "디미니시7", "aug7": "어그먼트7",
    "7sus4": "7서스4", "add9": "애드9", "9": "9th", "maj9": "메이저9", "m9": "마이너9",
    "11": "11th", "13": "13th", "alt": "얼터드",
}

# 파싱할 때 'maj7' 이 'maj' 보다, 'm7b5' 가 'm7' 보다 먼저 시도되어야 한다.
_CHORD_SUFFIXES: tuple[str, ...] = tuple(
    sorted((k for k in CHORD_INTERVALS if k), key=len, reverse=True)
)

# 같은 뜻의 다른 표기
_SUFFIX_ALIASES: dict[str, str] = {
    "M": "", "maj": "", "major": "", "Δ": "maj7", "M7": "maj7", "Ma7": "maj7",
    "min": "m", "minor": "m", "-": "m", "mi": "m", "min7": "m7", "-7": "m7",
    "o": "dim", "°": "dim", "o7": "dim7", "°7": "dim7",
    "ø": "m7b5", "ø7": "m7b5", "min7b5": "m7b5", "m7-5": "m7b5", "half-dim": "m7b5",
    "+": "aug", "#5": "aug", "+7": "aug7",
    "dom7": "7", "dom": "7", "sus": "sus4", "2": "sus2",
    "mmaj7": "mMaj7", "mM7": "mMaj7", "-Maj7": "mMaj7",
    "69": "6/9", "7alt": "alt",
}

_CHORD_RE = re.compile(r"^([A-Ga-g][#♯b♭x]*)(.*?)(?:/([A-Ga-g][#♯b♭x]*))?$")


@dataclass(frozen=True, slots=True)
class Chord:
    """화음. 근음 + 종류 + (선택) 베이스음.

    보이싱(어느 옥타브에 어떻게 벌려 놓을지)은 여기 없다. 그건 편곡 단계 일이다.
    여기서는 '어떤 화음인가'만 다룬다.
    """

    root: Pitch
    kind: str = ""
    bass: Pitch | None = None      # 분수코드 C/G 의 G
    inversion: int = 0             # 0=기본, 1=1전위, ...

    def __post_init__(self) -> None:
        if self.kind not in CHORD_INTERVALS:
            raise TheoryError(
                f"알 수 없는 코드 종류입니다: {self.kind!r} "
                f"(사용 가능 예: {', '.join(sorted(k for k in CHORD_INTERVALS if k)[:12])} ...)"
            )
        if not (0 <= self.inversion < len(CHORD_INTERVALS[self.kind])):
            raise TheoryError(
                f"{self.symbol_base} 는 {len(CHORD_INTERVALS[self.kind])}개 음이라 "
                f"{self.inversion}전위가 불가능합니다."
            )

    # ---- 구성음 ----

    @property
    def intervals(self) -> tuple[int, ...]:
        return CHORD_INTERVALS[self.kind]

    @property
    def size(self) -> int:
        return len(self.intervals)

    def pitches(self, octave: int | None = None) -> list[Pitch]:
        """구성음. 각 음을 몇 도로 읽는지에 따라 철자를 정하고, 전위와 분수코드를 반영한다."""
        root = self.root if octave is None else self.root.with_octave(octave)
        degrees = CHORD_DEGREES[self.kind]
        if len(degrees) != len(self.intervals):
            raise TheoryError(f"{self.kind} 코드의 도수표 길이가 구성음 개수와 다릅니다.")
        notes = [
            spell_interval(root, semitones, degree)
            for semitones, degree in zip(self.intervals, degrees)
        ]

        if self.inversion:
            moved = notes[: self.inversion]
            notes = notes[self.inversion :] + [shift_octave(n, 1) for n in moved]
        if self.bass is not None:
            bass = self.bass
            while bass.midi >= notes[0].midi:
                if bass.midi < 12:
                    break
                bass = shift_octave(bass, -1)
            if bass.midi < notes[0].midi:
                notes.insert(0, bass)
        return notes

    @property
    def pitch_classes(self) -> frozenset[int]:
        """옥타브 무시한 구성음 집합. 코드 비교/분석에 쓴다."""
        return frozenset((self.root.midi + i) % 12 for i in self.intervals)

    @property
    def bass_pitch(self) -> Pitch:
        """실제로 가장 낮게 울리는 음."""
        return self.pitches()[0]

    # ---- 심볼 ----

    @property
    def symbol_base(self) -> str:
        return f"{self.root.name}{self.kind}"

    @property
    def symbol(self) -> str:
        """'Cmaj7', 'Am7', 'G7/B' 형태."""
        text = self.symbol_base
        bass = self.bass_pitch
        if bass.pitch_class != self.root.pitch_class:
            text += f"/{bass.name}"
        return text

    @property
    def display_name(self) -> str:
        return f"{self.root.name} {CHORD_DISPLAY_NAMES.get(self.kind, self.kind)}"

    @classmethod
    def parse(cls, symbol: str) -> "Chord":
        """'Cmaj7', 'F#m7b5', 'Bb7', 'C/G', 'Dsus4' 를 읽는다."""
        text = symbol.strip()
        if not text:
            raise TheoryError("빈 코드 심볼입니다.")
        match = _CHORD_RE.match(text)
        if not match:
            raise TheoryError(f"코드 심볼을 읽을 수 없습니다: {symbol!r}")
        root_text, suffix, bass_text = match.groups()
        root = Pitch.parse(root_text)

        suffix = suffix.strip()
        kind = _SUFFIX_ALIASES.get(suffix, suffix)
        if kind not in CHORD_INTERVALS:
            # 별칭 표에도 없고 정식 이름도 아니면, 알려진 접미사로 시작하는지 본다
            resolved = None
            for candidate in _CHORD_SUFFIXES:
                if suffix == candidate:
                    resolved = candidate
                    break
            if resolved is None:
                lowered = suffix.lower()
                for alias, target in _SUFFIX_ALIASES.items():
                    if alias.lower() == lowered:
                        resolved = target
                        break
            if resolved is None:
                raise TheoryError(
                    f"알 수 없는 코드 접미사입니다: {suffix!r} (전체 심볼: {symbol!r})"
                )
            kind = resolved

        bass = Pitch.parse(bass_text) if bass_text else None
        if bass is not None and bass.pitch_class == root.pitch_class:
            bass = None
        return cls(root, kind, bass)

    # ---- 변형 ----

    def transpose(self, semitones: int) -> "Chord":
        """11번 '마지막 후렴만 반키 올려줘' 가 이걸 쓴다."""
        return Chord(
            self.root.transpose(semitones),
            self.kind,
            self.bass.transpose(semitones) if self.bass else None,
            self.inversion,
        )

    def inverted(self, inversion: int) -> "Chord":
        return Chord(self.root, self.kind, self.bass, inversion)

    def with_bass(self, bass: Pitch | None) -> "Chord":
        return Chord(self.root, self.kind, bass, self.inversion)

    def as_kind(self, kind: str) -> "Chord":
        return Chord(self.root, kind, self.bass, 0)

    # ---- 성질 ----

    @property
    def is_major(self) -> bool:
        return 4 in self.intervals and 3 not in self.intervals

    @property
    def is_minor(self) -> bool:
        return 3 in self.intervals and 4 not in self.intervals

    @property
    def is_dominant(self) -> bool:
        """딸림화음 성격(장3도 + 단7도). 해결을 원하는 화음."""
        return 4 in self.intervals and 10 in self.intervals

    @property
    def has_tritone(self) -> bool:
        """구성음 안에 트라이톤이 있는가. 긴장도 판단에 쓴다."""
        classes = sorted(self.pitch_classes)
        return any((b - a) % 12 == 6 for a in classes for b in classes)

    @property
    def tension_level(self) -> int:
        """0~5. 편곡 AI 가 '후렴이 심심해'(11번)를 판단할 때 쓴다."""
        score = 0
        if self.size >= 4:
            score += 1
        if self.size >= 5:
            score += 1
        if self.has_tritone:
            score += 1
        if any(i >= 13 for i in self.intervals):
            score += 1
        if self.kind in ("alt", "7b9", "7#9", "7#11", "7b13", "dim7", "aug", "aug7"):
            score += 1
        return min(5, score)

    def common_tones(self, other: "Chord") -> int:
        """두 코드의 공통음 개수. 코드 진행 자연스러움 판단에 쓴다."""
        return len(self.pitch_classes & other.pitch_classes)

    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:
        return f"Chord({self.symbol})"


def identify_chord(midi_notes: Sequence[int]) -> list[Chord]:
    """울린 음들로부터 코드 후보를 찾는다. 8번 '내 음악 분석'에서 쓴다.

    가능한 근음을 전부 시도해 구성음 집합이 정확히 일치하는 코드를 모으고,
    실제 최저음이 근음인 것을 앞에 둔다.
    """
    if not midi_notes:
        return []
    classes = frozenset(n % 12 for n in midi_notes)
    lowest_pc = min(midi_notes) % 12
    found: list[tuple[int, Chord]] = []
    for root_pc in range(12):
        for kind, intervals in CHORD_INTERVALS.items():
            if frozenset((root_pc + i) % 12 for i in intervals) != classes:
                continue
            root = Pitch.from_midi(60 + root_pc)
            bass = None if root_pc == lowest_pc else Pitch.from_midi(48 + lowest_pc)
            chord = Chord(root, kind, bass)
            # 근음이 최저음이고 구성음이 적을수록 단순한 해석이므로 우선한다
            rank = (0 if root_pc == lowest_pc else 1, len(intervals))
            found.append((rank, chord))
    found.sort(key=lambda pair: pair[0])
    return [chord for _, chord in found]


# --------------------------------------------------------------------------
# 조성
# --------------------------------------------------------------------------

# 5도권. 조표의 샵/플랫 개수 -> 장조 으뜸음
_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")

_MAJOR_KEY_ACCIDENTALS: dict[str, int] = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}

# 도수 표기: 장조 기준 각 음계도의 기본 3화음/7화음 성질
_MAJOR_DIATONIC_TRIADS = ("", "m", "m", "", "", "m", "dim")
_MAJOR_DIATONIC_SEVENTHS = ("maj7", "m7", "m7", "maj7", "7", "m7", "m7b5")
_MINOR_DIATONIC_TRIADS = ("m", "dim", "", "m", "m", "", "")
_MINOR_DIATONIC_SEVENTHS = ("m7", "m7b5", "maj7", "m7", "m7", "maj7", "7")

_ROMAN_UPPER = ("I", "II", "III", "IV", "V", "VI", "VII")


@dataclass(frozen=True, slots=True)
class Key:
    """조성. 으뜸음 + 선법. 조표 계산과 도수 분석을 담당한다."""

    tonic: Pitch
    mode: str = "major"      # SCALE_INTERVALS 의 키. 보통 major / natural_minor

    def __post_init__(self) -> None:
        if self.mode not in SCALE_INTERVALS:
            raise TheoryError(f"알 수 없는 선법입니다: {self.mode!r}")

    @classmethod
    def parse(cls, text: str) -> "Key":
        """'C', 'Am', 'F# minor', 'Bb major', 'D dorian' 을 읽는다."""
        raw = text.strip()
        if not raw:
            raise TheoryError("빈 조성 표기입니다.")
        lowered = raw.lower()
        for suffix, mode in (
            (" natural minor", "natural_minor"), (" harmonic minor", "harmonic_minor"),
            (" melodic minor", "melodic_minor"), (" minor", "natural_minor"),
            (" major", "major"), (" dorian", "dorian"), (" phrygian", "phrygian"),
            (" lydian", "lydian"), (" mixolydian", "mixolydian"), (" locrian", "locrian"),
            (" maj", "major"), (" min", "natural_minor"),
        ):
            if lowered.endswith(suffix):
                return cls(Pitch.parse(raw[: -len(suffix)].strip()), mode)
        if raw.endswith("m") and len(raw) > 1 and not raw.endswith("bm") is False:
            pass
        # 'Am', 'F#m' 처럼 끝이 m 인 경우
        if len(raw) >= 2 and raw[-1] == "m" and _NOTE_RE.match(raw[:-1]):
            return cls(Pitch.parse(raw[:-1]), "natural_minor")
        return cls(Pitch.parse(raw), "major")

    @property
    def scale(self) -> Scale:
        return Scale(self.tonic, self.mode)

    @property
    def is_minor(self) -> bool:
        return self.scale.is_minor

    @property
    def relative_major_tonic(self) -> Pitch:
        """나란한 장조의 으뜸음. 조표 계산에 쓴다."""
        if not self.is_minor:
            return self.tonic
        return self.tonic.transpose(3, prefer_flat=True)

    @property
    def accidental_count(self) -> int:
        """조표의 샵(+) / 플랫(-) 개수."""
        name = self.relative_major_tonic.name
        if name in _MAJOR_KEY_ACCIDENTALS:
            return _MAJOR_KEY_ACCIDENTALS[name]
        # 이명동음으로 다시 시도
        alternative = Pitch.from_midi(
            self.relative_major_tonic.midi, prefer_flat=self.relative_major_tonic.alter >= 0
        ).name
        if alternative in _MAJOR_KEY_ACCIDENTALS:
            return _MAJOR_KEY_ACCIDENTALS[alternative]
        raise TheoryError(f"조표를 계산할 수 없는 조성입니다: {self}")

    @property
    def key_signature(self) -> list[str]:
        """조표에 붙는 음이름 목록. ['F#','C#'] 처럼."""
        count = self.accidental_count
        if count > 0:
            return [f"{n}#" for n in _SHARP_ORDER[:count]]
        return [f"{n}b" for n in _FLAT_ORDER[:-count]]

    @property
    def prefers_flat(self) -> bool:
        return self.accidental_count < 0

    # ---- 도수 화음 ----

    def diatonic_chord(self, degree: int, seventh: bool = False) -> Chord:
        """음계 위에 쌓은 화음. degree 는 1~7."""
        if not (1 <= degree <= 7):
            raise TheoryError(f"음계 도수는 1~7 이어야 합니다: {degree}")
        scale_pitches = self.scale.pitches()
        if len(scale_pitches) != 7:
            raise TheoryError(f"7음계에서만 도수 화음을 만들 수 있습니다: {self.mode}")
        root = scale_pitches[degree - 1]
        if self.is_minor:
            table = _MINOR_DIATONIC_SEVENTHS if seventh else _MINOR_DIATONIC_TRIADS
        else:
            table = _MAJOR_DIATONIC_SEVENTHS if seventh else _MAJOR_DIATONIC_TRIADS
        return Chord(root, table[degree - 1])

    def diatonic_chords(self, seventh: bool = False) -> list[Chord]:
        return [self.diatonic_chord(d, seventh) for d in range(1, 8)]

    def degree_of_chord(self, chord: Chord) -> int | None:
        """이 코드가 몇 번째 도수인가. 조성 밖이면 None."""
        offset = (chord.root.midi - self.tonic.midi) % 12
        intervals = self.scale.intervals
        if offset in intervals:
            return intervals.index(offset) + 1
        return None

    def roman_numeral(self, chord: Chord) -> str:
        """코드를 도수 표기로. 'I', 'vi', 'V7', 'bVII', 'V/V' 형태."""
        degree = self.degree_of_chord(chord)
        if degree is not None:
            numeral = _ROMAN_UPPER[degree - 1]
            if chord.is_minor:
                numeral = numeral.lower()
            elif chord.kind in ("dim", "dim7", "m7b5"):
                numeral = numeral.lower() + "°"
            suffix = _roman_suffix(chord.kind)
            text = numeral + suffix
        else:
            # 조성 밖의 근음: 변화표를 붙인다
            offset = (chord.root.midi - self.tonic.midi) % 12
            below = max((i for i in self.scale.intervals if i < offset), default=None)
            if below is None:
                below = self.scale.intervals[0]
            base_degree = self.scale.intervals.index(below) + 1
            alteration = "b" if offset - below == 1 else "#"
            if alteration == "b":
                # 위쪽 음계음에서 내린 것으로 보는 게 자연스러운 경우
                above = min((i for i in self.scale.intervals if i > offset), default=None)
                if above is not None and above - offset == 1:
                    base_degree = self.scale.intervals.index(above) + 1
            numeral = _ROMAN_UPPER[base_degree - 1]
            if chord.is_minor:
                numeral = numeral.lower()
            text = alteration + numeral + _roman_suffix(chord.kind)

        bass = chord.bass_pitch
        if bass.pitch_class != chord.root.pitch_class:
            bass_degree = self.degree_of_chord(Chord(bass))
            text += f"/{_ROMAN_UPPER[bass_degree - 1]}" if bass_degree else f"/{bass.name}"
        return text

    def chord_from_roman(self, numeral: str) -> Chord:
        """'V7', 'bVII', 'iv' 같은 도수 표기로부터 실제 코드를 만든다."""
        text = numeral.strip()
        if not text:
            raise TheoryError("빈 도수 표기입니다.")
        alteration = 0
        while text and text[0] in "b#♭♯":
            alteration += -1 if text[0] in "b♭" else 1
            text = text[1:]
        match = re.match(r"^([ivxIVX]+)(.*)$", text)
        if not match:
            raise TheoryError(f"도수 표기를 읽을 수 없습니다: {numeral!r}")
        roman, suffix = match.groups()
        upper = roman.upper()
        if upper not in _ROMAN_UPPER:
            raise TheoryError(f"알 수 없는 로마 숫자입니다: {roman!r}")
        degree = _ROMAN_UPPER.index(upper) + 1

        scale_pitches = self.scale.pitches()
        root = scale_pitches[degree - 1]
        if alteration:
            root = root.transpose(alteration, prefer_flat=alteration < 0)

        suffix = suffix.replace("°", "dim").strip("/")
        if suffix in CHORD_INTERVALS:
            kind = suffix
        elif suffix in _SUFFIX_ALIASES:
            kind = _SUFFIX_ALIASES[suffix]
        elif suffix == "":
            kind = "m" if roman.islower() else ""
        elif suffix == "7":
            kind = "m7" if roman.islower() else "7"
        else:
            raise TheoryError(f"도수 표기의 접미사를 읽을 수 없습니다: {suffix!r}")
        if roman.islower() and kind == "":
            kind = "m"
        return Chord(root, kind)

    def secondary_dominant(self, target_degree: int) -> Chord:
        """부속화음 V/x. 'V/V', 'V/vi' 를 만든다. K-Pop/발라드에서 흔하다."""
        target = self.diatonic_chord(target_degree)
        return Chord(target.root.transpose(7), "7")

    def transpose(self, semitones: int) -> "Key":
        return Key(self.tonic.transpose(semitones, prefer_flat=self.prefers_flat), self.mode)

    @property
    def relative(self) -> "Key":
        """나란한조. 장조 <-> 단조."""
        if self.is_minor:
            return Key(self.tonic.transpose(3), "major")
        return Key(self.tonic.transpose(-3), "natural_minor")

    @property
    def parallel(self) -> "Key":
        """같은 으뜸음 장/단조."""
        return Key(self.tonic, "natural_minor" if not self.is_minor else "major")

    @property
    def display_name(self) -> str:
        mode_text = "단조" if self.is_minor else "장조"
        if self.mode not in ("major", "natural_minor"):
            mode_text = SCALE_DISPLAY_NAMES.get(self.mode, self.mode)
        return f"{self.tonic.name} {mode_text}"

    def __str__(self) -> str:
        return f"{self.tonic.name}{'m' if self.is_minor else ''}"


def _roman_suffix(kind: str) -> str:
    """코드 종류를 도수 표기용 접미사로."""
    if kind in ("", "m"):
        return ""
    if kind in ("dim", "m7b5"):
        return "" if kind == "dim" else "7b5"
    return kind.replace("maj7", "M7").replace("m7", "7").replace("m9", "9")


def detect_key(midi_notes: Sequence[int], weights: Sequence[float] | None = None) -> list[tuple[Key, float]]:
    """울린 음들로부터 조성을 추정한다. 8번 '내 음악 분석'의 Key 판정.

    Krumhansl-Schmuckler 조성 프로파일과의 상관계수를 쓴다. 24개 조 전부에 대해
    점수를 내고 높은 순으로 돌려준다. 점수는 -1 ~ 1 의 상관계수다.
    """
    if not midi_notes:
        return []
    if weights is None:
        weights = [1.0] * len(midi_notes)
    if len(weights) != len(midi_notes):
        raise TheoryError("음과 가중치 개수가 다릅니다.")

    histogram = [0.0] * 12
    for note, weight in zip(midi_notes, weights):
        histogram[note % 12] += weight
    total = sum(histogram)
    if total <= 0:
        return []

    # Krumhansl & Kessler (1982) 조성 프로파일
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def correlate(sample: Sequence[float], profile: Sequence[float]) -> float:
        n = len(sample)
        mean_s = sum(sample) / n
        mean_p = sum(profile) / n
        num = sum((sample[i] - mean_s) * (profile[i] - mean_p) for i in range(n))
        den_s = sum((sample[i] - mean_s) ** 2 for i in range(n)) ** 0.5
        den_p = sum((profile[i] - mean_p) ** 2 for i in range(n)) ** 0.5
        if den_s == 0 or den_p == 0:
            return 0.0
        return num / (den_s * den_p)

    results: list[tuple[Key, float]] = []
    for tonic_pc in range(12):
        rotated = histogram[tonic_pc:] + histogram[:tonic_pc]
        for profile, mode in ((major_profile, "major"), (minor_profile, "natural_minor")):
            score = correlate(rotated, profile)
            prefer_flat = tonic_pc in (1, 3, 8, 10) or (tonic_pc == 5)
            key = Key(Pitch.from_midi(60 + tonic_pc, prefer_flat=prefer_flat), mode)
            results.append((key, score))
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


# 두 표가 어긋나면 철자가 조용히 틀어지므로, 불러오는 시점에 바로 알린다.
_missing_degrees = set(CHORD_INTERVALS) - set(CHORD_DEGREES)
if _missing_degrees:
    raise TheoryError(f"CHORD_DEGREES 에 빠진 코드 종류: {sorted(_missing_degrees)}")
for _kind, _iv in CHORD_INTERVALS.items():
    if len(_iv) != len(CHORD_DEGREES[_kind]):
        raise TheoryError(
            f"{_kind!r}: 구성음 {len(_iv)}개인데 도수표는 {len(CHORD_DEGREES[_kind])}개입니다."
        )
_missing_scale_degrees = set(SCALE_INTERVALS) - set(SCALE_DEGREES)
if _missing_scale_degrees:
    raise TheoryError(f"SCALE_DEGREES 에 빠진 음계: {sorted(_missing_scale_degrees)}")
for _kind, _iv in SCALE_INTERVALS.items():
    _deg = SCALE_DEGREES[_kind]
    if _deg is not None and len(_iv) != len(_deg):
        raise TheoryError(
            f"{_kind!r} 음계: 음 {len(_iv)}개인데 도수표는 {len(_deg)}개입니다."
        )
del _missing_degrees, _missing_scale_degrees
