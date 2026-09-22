"""
선율 생성 — 5번의 Lead Melody.

무작위 음을 코드에 맞춰 뿌리면 선율이 안 된다. 사람이 '멜로디'로 듣는 것에는
구조가 있다.

  1. 프레이즈
     선율은 숨을 쉬는 단위로 나뉜다. 보통 2마디 또는 4마디다. 프레이즈 끝에는
     긴 음이 오고 그 뒤에 쉼이 온다. 이게 없으면 음이 끝없이 이어져서
     사람이 따라 부를 수 없다.

  2. 윤곽
     프레이즈 하나는 대개 하나의 모양을 그린다. 올라갔다 내려오거나(아치),
     계속 올라가거나, 내려오거나. 모양이 없으면 음이 제자리에서 맴돈다.

  3. 화음음과 지나가는 음
     강박에는 코드 구성음이 온다. 약박에는 그 사이를 잇는 음이 올 수 있다.
     반대로 하면 화음과 부딪히는 소리가 난다.

  4. 도약 뒤에는 반대 방향 순차진행
     큰 도약 뒤에 같은 방향으로 또 도약하면 부르기 어렵고 어색하다.
     이건 400년 된 작곡 규칙이고 지금도 유효하다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..core.notes import Note
from ..core.units import PPQ, note_value_to_ticks
from .genre import GenreProfile, get_genre
from .harmony import ChordEvent, ChordProgression, HarmonyError
from .theory import Chord, Key, Pitch, Scale, TheoryError


class MelodyError(ValueError):
    """선율 생성 관련 오류."""


# 프레이즈의 윤곽. 각 값은 프레이즈 안에서의 상대 높이 곡선이다.
CONTOURS: dict[str, tuple[float, ...]] = {
    "arch":        (0.0, 0.4, 0.8, 1.0, 0.8, 0.4, 0.1),   # 올라갔다 내려옴. 가장 흔하다.
    "ascending":   (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0),   # 계속 올라감. 고조될 때.
    "descending":  (1.0, 0.85, 0.7, 0.5, 0.35, 0.2, 0.0), # 내려옴. 가라앉을 때.
    "wave":        (0.3, 0.7, 0.4, 0.8, 0.45, 0.75, 0.3), # 물결. 벌스에 어울린다.
    "valley":      (0.9, 0.5, 0.2, 0.0, 0.25, 0.6, 0.95), # 내려갔다 올라옴.
    "static":      (0.4, 0.45, 0.4, 0.5, 0.4, 0.45, 0.4), # 거의 제자리. 랩/찬트.
    "terraced":    (0.1, 0.1, 0.5, 0.5, 0.85, 0.85, 1.0), # 계단식. 후렴 고조에 쓴다.
}

# 구간 종류별로 어울리는 윤곽
SECTION_CONTOURS: dict[str, tuple[str, ...]] = {
    "intro": ("descending", "static"),
    "verse": ("wave", "arch", "static"),
    "pre_chorus": ("ascending", "terraced"),
    "chorus": ("arch", "terraced", "ascending"),
    "post_chorus": ("arch", "wave"),
    "bridge": ("valley", "descending", "wave"),
    "instrumental": ("arch", "wave"),
    "solo": ("ascending", "arch"),
    "breakdown": ("static", "descending"),
    "drop": ("terraced", "ascending"),
    "build": ("ascending",),
    "interlude": ("wave", "static"),
    "outro": ("descending", "valley"),
}

# 음 길이 후보 (박 단위) 와 기본 가중치.
RHYTHM_VALUES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


@dataclass(slots=True)
class VocalRange:
    """부를 수 있는 음역. 6번의 목소리 분석 결과가 여기로 들어온다."""

    lowest: int
    highest: int
    comfortable_low: int = 0
    comfortable_high: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.lowest < self.highest <= 127):
            raise MelodyError(f"음역이 잘못됐습니다: {self.lowest} ~ {self.highest}")
        span = self.highest - self.lowest
        if not self.comfortable_low:
            # 가장 낮은 음과 가장 높은 음은 부르기 힘들다. 안쪽을 편한 음역으로 본다.
            self.comfortable_low = self.lowest + max(1, span // 6)
        if not self.comfortable_high:
            self.comfortable_high = self.highest - max(1, span // 6)
        if self.comfortable_low >= self.comfortable_high:
            self.comfortable_low = self.lowest
            self.comfortable_high = self.highest

    @property
    def semitones(self) -> int:
        return self.highest - self.lowest

    @property
    def center(self) -> int:
        return (self.comfortable_low + self.comfortable_high) // 2

    def contains(self, midi: int) -> bool:
        return self.lowest <= midi <= self.highest

    def is_comfortable(self, midi: int) -> bool:
        return self.comfortable_low <= midi <= self.comfortable_high

    def clamp(self, midi: int) -> int:
        """음역 밖이면 옥타브를 옮겨 안으로 넣는다. 그래도 안 되면 경계로 자른다."""
        while midi < self.lowest and midi + 12 <= self.highest:
            midi += 12
        while midi > self.highest and midi - 12 >= self.lowest:
            midi -= 12
        return max(self.lowest, min(self.highest, midi))

    def describe(self) -> str:
        return (
            f"{Pitch.from_midi(self.lowest)} ~ {Pitch.from_midi(self.highest)} "
            f"(편한 음역 {Pitch.from_midi(self.comfortable_low)} ~ "
            f"{Pitch.from_midi(self.comfortable_high)}, {self.semitones}반음)"
        )

    @classmethod
    def typical(cls, voice_type: str = "alto") -> "VocalRange":
        """대표적인 음역. 목소리 학습 전 기본값으로 쓴다."""
        table = {
            "bass": (40, 64), "baritone": (45, 69), "tenor": (48, 72),
            "countertenor": (53, 77), "alto": (53, 77), "mezzo": (55, 79),
            "soprano": (60, 84),
        }
        if voice_type not in table:
            raise MelodyError(
                f"모르는 성종입니다: {voice_type!r} (가능: {', '.join(table)})"
            )
        low, high = table[voice_type]
        return cls(low, high)


@dataclass(slots=True)
class Phrase:
    """선율의 한 호흡. 시작 박, 길이, 윤곽."""

    start_beat: float
    length_beats: float
    contour: str = "arch"
    peak_ratio: float = 1.0      # 이 프레이즈가 얼마나 높이 올라가는가 (0~1)
    rest_at_end_beats: float = 0.0

    def __post_init__(self) -> None:
        if self.length_beats <= 0:
            raise MelodyError(f"프레이즈 길이는 0보다 커야 합니다: {self.length_beats}")
        if self.contour not in CONTOURS:
            raise MelodyError(
                f"모르는 윤곽입니다: {self.contour!r} (가능: {', '.join(CONTOURS)})"
            )

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.length_beats

    @property
    def singing_beats(self) -> float:
        """실제로 노래하는 길이. 끝의 쉼을 뺀다."""
        return max(0.25, self.length_beats - self.rest_at_end_beats)

    def height_at(self, progress: float) -> float:
        """프레이즈 안 위치(0~1)에서의 상대 높이(0~1)."""
        curve = CONTOURS[self.contour]
        position = max(0.0, min(1.0, progress)) * (len(curve) - 1)
        index = int(position)
        if index >= len(curve) - 1:
            return curve[-1] * self.peak_ratio
        fraction = position - index
        value = curve[index] + (curve[index + 1] - curve[index]) * fraction
        return value * self.peak_ratio


def build_phrases(
    total_beats: float, beats_per_bar: int = 4, bars_per_phrase: int = 2,
    section_kind: str = "verse", generator: random.Random | None = None,
    rest_ratio: float = 0.2,
) -> list[Phrase]:
    """구간을 프레이즈로 나눈다.

    프레이즈 끝에 쉼을 둔다. 쉼이 없으면 숨 쉴 자리가 없어서 사람이 부를 수 없고,
    합성한 목소리도 기계처럼 들린다.
    """
    if total_beats <= 0:
        raise MelodyError(f"길이는 0보다 커야 합니다: {total_beats}")
    generator = generator or random.Random()
    phrase_beats = beats_per_bar * bars_per_phrase
    options = SECTION_CONTOURS.get(section_kind, ("arch", "wave"))

    phrases: list[Phrase] = []
    beat = 0.0
    index = 0
    count = max(1, int(round(total_beats / phrase_beats)))
    while beat < total_beats - 1e-9:
        length = min(phrase_beats, total_beats - beat)
        contour = options[index % len(options)]
        # 뒤로 갈수록 조금씩 더 높이 올라간다. 구간이 고조되는 느낌을 만든다.
        peak = 0.75 + 0.25 * (index / max(1, count - 1)) if count > 1 else 1.0
        rest = round(length * rest_ratio * 4) / 4 if length >= phrase_beats else 0.0
        phrases.append(Phrase(beat, length, contour, peak, rest))
        beat += length
        index += 1
    return phrases


class MelodyGenerator:
    """코드 진행 위에 선율을 얹는다."""

    def __init__(
        self, key: Key, genre: GenreProfile | str = "pop",
        vocal_range: VocalRange | None = None, seed: int | None = None,
    ) -> None:
        self.key = key
        self.genre = get_genre(genre) if isinstance(genre, str) else genre
        self.range = vocal_range or VocalRange.typical("alto")
        self.random = random.Random(seed)
        self.scale = key.scale

    # ---- 리듬 ----

    def _rhythm(self, beats: float, energy: float) -> list[float]:
        """프레이즈를 채울 음 길이들을 고른다.

        에너지가 높고 엇박이 많은 장르일수록 짧은 음이 많아진다.
        마지막 음은 길게 잡아 프레이즈가 닫히는 느낌을 준다.
        """
        if beats <= 0:
            return []
        density = 0.35 + 0.5 * energy + 0.25 * self.genre.syncopation
        # 밀도가 높을수록 짧은 음에 가중치가 간다
        weights = {
            0.25: max(0.0, density - 0.65) * 2.0,
            0.5: max(0.0, density - 0.35) * 1.8,
            0.75: max(0.0, density - 0.5) * 0.5,
            1.0: 1.0,
            1.5: 0.55,
            2.0: max(0.1, 1.2 - density),
            3.0: max(0.05, 0.5 - density * 0.4),
            4.0: max(0.02, 0.35 - density * 0.3),
        }
        pool = [(value, weight) for value, weight in weights.items() if weight > 0]
        if not pool:
            pool = [(1.0, 1.0)]

        result: list[float] = []
        remaining = beats
        while remaining > 1e-9:
            # 마지막에 남은 길이가 짧으면 그대로 채운다
            options = [(v, w) for v, w in pool if v <= remaining + 1e-9]
            if not options:
                result.append(round(remaining * 4) / 4 or 0.25)
                break
            total = sum(w for _, w in options)
            threshold = self.random.random() * total
            running = 0.0
            chosen = options[-1][0]
            for value, weight in options:
                running += weight
                if running >= threshold:
                    chosen = value
                    break
            result.append(chosen)
            remaining -= chosen
        # 프레이즈 마지막 음을 길게 만든다
        if len(result) >= 2 and result[-1] < 1.0:
            merged = result[-1] + result[-2]
            if merged <= 4.0:
                result = result[:-2] + [merged]
        return result

    # ---- 음높이 ----

    def _candidates(self, chord: Chord, strong: bool) -> list[int]:
        """이 시점에 놓을 수 있는 음높이(피치 클래스) 후보.

        강박에는 화음음만. 약박에는 음계음까지 허용한다.
        """
        chord_classes = sorted(chord.pitch_classes)
        if strong:
            return chord_classes
        scale_classes = sorted(
            (self.key.tonic.midi + i) % 12 for i in self.scale.intervals
        )
        # 화음음을 더 자주 고르도록 두 번 넣는다
        return chord_classes + chord_classes + scale_classes

    def _pick_pitch(
        self, target_midi: int, chord: Chord, strong: bool, previous: int | None,
    ) -> int:
        """목표 높이에 가장 가까우면서 화성에 맞는 실제 음을 고른다."""
        classes = self._candidates(chord, strong)
        best: tuple[float, int] | None = None
        for pitch_class in set(classes):
            # 목표 높이 주변 옥타브들을 본다
            base = target_midi - (target_midi % 12) + pitch_class
            for candidate in (base - 12, base, base + 12):
                if not self.range.contains(candidate):
                    continue
                cost = abs(candidate - target_midi)
                if not self.range.is_comfortable(candidate):
                    cost += 3.0
                if previous is not None:
                    leap = abs(candidate - previous)
                    if leap == 0:
                        cost += 1.5             # 같은 음이 계속되면 지루하다
                    elif leap > 12:
                        cost += (leap - 12) * 4.0   # 옥타브 넘는 도약은 부르기 힘들다
                    elif leap > 7:
                        cost += (leap - 7) * 1.2
                    # 화음음 쪽에 무게를 둔다
                if pitch_class in chord.pitch_classes:
                    cost -= 1.0 if strong else 0.4
                cost += self.random.random() * 0.8   # 같은 답만 나오지 않게 흔든다
                if best is None or cost < best[0]:
                    best = (cost, candidate)
        if best is None:
            return self.range.clamp(target_midi)
        return best[1]

    def _fix_leaps(self, midis: list[int]) -> list[int]:
        """큰 도약 뒤에는 반대 방향으로 순차진행하게 고친다.

        도약 뒤에 같은 방향으로 또 도약하면 부르기 어렵고 선율이 흩어진다.
        """
        if len(midis) < 3:
            return midis
        result = list(midis)
        for index in range(1, len(result) - 1):
            leap = result[index] - result[index - 1]
            if abs(leap) < 5:
                continue
            following = result[index + 1] - result[index]
            if following * leap <= 0 and abs(following) <= 2:
                continue    # 이미 반대 방향 순차진행이다
            direction = -1 if leap > 0 else 1
            target = result[index] + direction * 2
            snapped = self.scale.snap(Pitch.from_midi(target)).midi
            if self.range.contains(snapped):
                result[index + 1] = snapped
        return result

    # ---- 생성 ----

    def generate(
        self, progression: ChordProgression, beats_per_bar: int = 4,
        section_kind: str = "verse", energy: float = 0.5,
        bars_per_phrase: int = 2, start_tick: int = 0,
    ) -> list[Note]:
        """코드 진행 위에 선율을 만든다."""
        total_beats = progression.total_beats
        if total_beats <= 0:
            raise MelodyError("코드 진행이 비어 있습니다.")
        if not (0.0 <= energy <= 1.0):
            raise MelodyError(f"에너지는 0~1 이어야 합니다: {energy}")

        phrases = build_phrases(
            total_beats, beats_per_bar, bars_per_phrase, section_kind, self.random,
            rest_ratio=0.25 - 0.12 * energy,
        )

        # 에너지가 높으면 더 높은 음역을 쓴다. 후렴이 벌스보다 높은 이유다.
        low_bound = self.range.comfortable_low
        high_bound = self.range.comfortable_high
        span = high_bound - low_bound
        base = low_bound + span * (0.1 + 0.35 * energy)
        top = low_bound + span * (0.55 + 0.45 * energy)

        notes: list[Note] = []
        previous: int | None = None

        for phrase in phrases:
            durations = self._rhythm(phrase.singing_beats, energy)
            if not durations:
                continue
            positions: list[float] = []
            beat = phrase.start_beat
            for duration in durations:
                positions.append(beat)
                beat += duration

            targets: list[int] = []
            for index, position in enumerate(positions):
                progress = index / max(1, len(positions) - 1)
                height = phrase.height_at(progress)
                targets.append(int(round(base + (top - base) * height)))

            phrase_midis: list[int] = []
            for index, (position, duration) in enumerate(zip(positions, durations)):
                chord_event = progression.at_beat(position)
                chord = chord_event.chord if chord_event else self.key.diatonic_chord(1)
                # 마디의 1박과 3박을 강박으로 본다
                in_bar = position % beats_per_bar
                strong = (abs(in_bar) < 1e-6 or abs(in_bar - beats_per_bar / 2) < 1e-6
                          or index == 0 or index == len(positions) - 1)
                midi = self._pick_pitch(targets[index], chord, strong, previous)
                phrase_midis.append(midi)
                previous = midi

            phrase_midis = self._fix_leaps(phrase_midis)

            # 프레이즈 마지막 음은 화음음으로 맞춘다. 안 그러면 공중에 뜬 채 끝난다.
            last_chord_event = progression.at_beat(positions[-1])
            if last_chord_event is not None:
                chord_classes = last_chord_event.chord.pitch_classes
                if phrase_midis[-1] % 12 not in chord_classes:
                    for offset in (-1, 1, -2, 2, -3, 3):
                        candidate = phrase_midis[-1] + offset
                        if candidate % 12 in chord_classes and self.range.contains(candidate):
                            phrase_midis[-1] = candidate
                            break

            for position, duration, midi in zip(positions, durations, phrase_midis):
                notes.append(Note(
                    midi=midi,
                    start_tick=start_tick + int(round(position * PPQ)),
                    duration_ticks=max(1, int(round(duration * PPQ))),
                    velocity=self._velocity(energy, position, beats_per_bar),
                ))
            previous = phrase_midis[-1]

        return notes

    def _velocity(self, energy: float, position: float, beats_per_bar: int) -> int:
        """세기. 강박이 세고, 에너지가 높으면 전체가 세다."""
        base = 62 + int(38 * energy)
        in_bar = position % beats_per_bar
        if abs(in_bar) < 1e-6:
            base += 8
        elif abs(in_bar - beats_per_bar / 2) < 1e-6:
            base += 4
        base += self.random.randint(-4, 4)
        return max(1, min(127, base))
