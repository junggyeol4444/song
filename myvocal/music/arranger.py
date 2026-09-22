"""
편곡 — 5번의 Bass / Drums / Piano / Guitar / Strings, 그리고 AI 편곡.

선율과 코드만 있으면 곡이 아니다. 누가 무엇을 언제 연주하는지가 편곡이다.
여기서 정하는 것은 세 가지다.

  1. 어떤 악기를 쓸 것인가      장르의 악기 가중치 + 구간 에너지
  2. 각 악기가 무엇을 칠 것인가  베이스 패턴, 드럼 패턴, 반주 형태
  3. 얼마나 촘촘하게 할 것인가   에너지가 낮으면 빼고, 높으면 더한다

중요한 원칙: 구간마다 악기가 달라야 한다. 처음부터 끝까지 같은 악기가 같은
밀도로 나오면 곡에 기승전결이 없다. 인트로에서 하나둘 들어오고, 후렴에서
전부 나오고, 브릿지에서 빠지는 것이 '편곡'이다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..core.notes import Note
from ..core.units import PPQ
from .genre import GenreProfile, get_genre
from .harmony import ChordEvent, ChordProgression
from .instruments import DRUM_NOTES
from .theory import Chord, Key, Pitch, TheoryError


class ArrangeError(ValueError):
    """편곡 관련 오류."""


# 드럼 이름 -> MIDI 번호 (일반 MIDI)
DRUM_MIDI: dict[str, int] = {
    "kick": 36, "snare": 38, "rim": 37, "clap": 39, "snare_tight": 40,
    "hihat_closed": 42, "hihat_pedal": 44, "hihat_open": 46,
    "tom_low": 41, "tom_mid": 45, "tom_high": 48,
    "crash": 49, "ride": 51, "ride_bell": 53, "china": 52, "splash": 55,
    "cowbell": 56, "tambourine": 54, "shaker": 69, "clave": 75, "woodblock": 76,
}


# ==========================================================================
# 베이스
# ==========================================================================

BASS_PATTERNS: dict[str, str] = {
    "root_whole": "근음을 온음표로. 발라드 인트로, 앰비언트.",
    "root_quarter": "근음을 4분음표로. 가장 기본.",
    "root_fifth": "근음과 5음을 번갈아. 록, 컨트리.",
    "octave": "근음과 한 옥타브 위를 번갈아. 디스코, 신스팝.",
    "eighth_drive": "8분음표로 밀어붙인다. 록, 펑크, 메탈.",
    "walking": "4분음표로 코드 사이를 걸어간다. 재즈.",
    "syncopated": "엇박을 섞는다. 펑크, R&B, 시티팝.",
    "sustained": "코드 길이만큼 한 음. 오케스트라, 시네마틱.",
    "sub_drop": "코드 첫 박에 긴 저음. 힙합, 트랩.",
}


def _pattern_for_genre(profile: GenreProfile, energy: float) -> str:
    """장르와 에너지에 맞는 베이스 패턴을 고른다."""
    if profile.swing > 0.4:
        return "walking"
    if profile.low_end > 0.7 and profile.drum_density < 0.75:
        return "sub_drop"
    if profile.arrangement_density > 0.85 and profile.drum_density < 0.4:
        return "sustained"
    if energy < 0.3:
        return "root_whole" if profile.bpm_typical < 100 else "root_quarter"
    if profile.syncopation > 0.6:
        return "syncopated"
    # 밴드 편성(기타+드럼이 둘 다 센 장르)은 8분음표로 민다.
    # 옥타브 왕복보다 이쪽이 록·메탈·펑크의 실제 주법이다.
    band = (profile.instrument_weights.get("electric_guitar", 0.0) > 0.6
            or profile.instrument_weights.get("acoustic_guitar", 0.0) > 0.6)
    if profile.drum_density >= 0.7 and energy > 0.55 and band:
        return "eighth_drive"
    # 옥타브 왕복은 디스코/신스팝의 주법이다. 신스 베이스를 쓰는 장르에만 준다.
    if (profile.bpm_typical > 115 and energy > 0.5
            and profile.instrument_weights.get("synth_bass", 0.0) > 0.5):
        return "octave"
    if profile.drum_density >= 0.7 and energy > 0.6:
        return "eighth_drive"
    if energy > 0.6:
        return "root_fifth"
    return "root_quarter"


class BassGenerator:
    """베이스 라인을 만든다."""

    def __init__(self, key: Key, genre: GenreProfile | str = "pop",
                 low_midi: int = 28, high_midi: int = 55, seed: int | None = None) -> None:
        self.key = key
        self.genre = get_genre(genre) if isinstance(genre, str) else genre
        self.low = low_midi
        self.high = high_midi
        if self.low >= self.high:
            raise ArrangeError(f"음역이 잘못됐습니다: {low_midi} ~ {high_midi}")
        self.random = random.Random(seed)

    def _place(self, pitch_class: int, near: int) -> int:
        """음역 안에서 near 에 가장 가까운 위치에 놓는다."""
        base = near - (near % 12) + pitch_class
        best = None
        for candidate in (base - 12, base, base + 12):
            if self.low <= candidate <= self.high:
                cost = abs(candidate - near)
                if best is None or cost < best[0]:
                    best = (cost, candidate)
        if best is None:
            return max(self.low, min(self.high, base))
        return best[1]

    def generate(
        self, progression: ChordProgression, beats_per_bar: int = 4,
        energy: float = 0.5, pattern: str | None = None, start_tick: int = 0,
    ) -> list[Note]:
        if pattern is None:
            pattern = _pattern_for_genre(self.genre, energy)
        if pattern not in BASS_PATTERNS:
            raise ArrangeError(
                f"모르는 베이스 패턴입니다: {pattern!r}\n"
                f"사용 가능: {', '.join(BASS_PATTERNS)}"
            )
        maker = getattr(self, f"_pattern_{pattern}")
        notes: list[Note] = []
        center = (self.low + self.high) // 2
        previous = center
        for index, event in enumerate(progression):
            following = progression[index + 1] if index + 1 < len(progression) else None
            made, previous = maker(event, following, beats_per_bar, energy, previous)
            notes.extend(made)
        for note in notes:
            note.start_tick += start_tick
        return notes

    # ---- 패턴들 ----

    def _velocity(self, energy: float, accent: bool = False) -> int:
        base = 72 + int(30 * energy) + (8 if accent else 0)
        return max(1, min(127, base + self.random.randint(-3, 3)))

    def _pattern_root_whole(self, event, following, beats_per_bar, energy, previous):
        midi = self._place(event.chord.bass_pitch.pitch_class, previous)
        return [Note(midi, int(event.start_beat * PPQ),
                     max(1, int(event.duration_beats * PPQ)),
                     self._velocity(energy, True))], midi

    def _pattern_sustained(self, event, following, beats_per_bar, energy, previous):
        return self._pattern_root_whole(event, following, beats_per_bar, energy, previous)

    def _pattern_root_quarter(self, event, following, beats_per_bar, energy, previous):
        midi = self._place(event.chord.bass_pitch.pitch_class, previous)
        notes = []
        beat = event.start_beat
        while beat < event.end_beat - 1e-9:
            length = min(1.0, event.end_beat - beat)
            notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                              self._velocity(energy, abs(beat % beats_per_bar) < 1e-6)))
            beat += 1.0
        return notes, midi

    def _pattern_root_fifth(self, event, following, beats_per_bar, energy, previous):
        root = self._place(event.chord.bass_pitch.pitch_class, previous)
        fifth_class = (event.chord.root.pitch_class + 7) % 12
        fifth = self._place(fifth_class, root)
        notes = []
        beat = event.start_beat
        index = 0
        while beat < event.end_beat - 1e-9:
            length = min(1.0, event.end_beat - beat)
            midi = root if index % 2 == 0 else fifth
            notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                              self._velocity(energy, index % 2 == 0)))
            beat += 1.0
            index += 1
        return notes, root

    def _pattern_octave(self, event, following, beats_per_bar, energy, previous):
        root = self._place(event.chord.bass_pitch.pitch_class, previous)
        upper = root + 12 if root + 12 <= self.high else root
        notes = []
        beat = event.start_beat
        index = 0
        while beat < event.end_beat - 1e-9:
            length = min(0.5, event.end_beat - beat)
            midi = root if index % 2 == 0 else upper
            notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                              self._velocity(energy, index % 2 == 0)))
            beat += 0.5
            index += 1
        return notes, root

    def _pattern_eighth_drive(self, event, following, beats_per_bar, energy, previous):
        midi = self._place(event.chord.bass_pitch.pitch_class, previous)
        notes = []
        beat = event.start_beat
        index = 0
        while beat < event.end_beat - 1e-9:
            length = min(0.5, event.end_beat - beat)
            notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                              self._velocity(energy, index % 2 == 0)))
            beat += 0.5
            index += 1
        return notes, midi

    def _pattern_syncopated(self, event, following, beats_per_bar, energy, previous):
        """엇박. 1박, 1박 반, 2박 반 자리에 놓는다. 펑크와 R&B 의 느낌."""
        root = self._place(event.chord.bass_pitch.pitch_class, previous)
        fifth = self._place((event.chord.root.pitch_class + 7) % 12, root)
        offsets = [(0.0, root, 0.5), (1.5, root, 0.5), (2.5, fifth, 0.5), (3.5, root, 0.5)]
        notes = []
        for offset, midi, length in offsets:
            beat = event.start_beat + offset
            if beat >= event.end_beat - 1e-9:
                continue
            if self.random.random() > 0.55 + 0.4 * energy and offset > 0:
                continue
            actual = min(length, event.end_beat - beat)
            notes.append(Note(midi, int(beat * PPQ), max(1, int(actual * PPQ)),
                              self._velocity(energy, offset == 0.0)))
        if not notes:
            notes.append(Note(root, int(event.start_beat * PPQ),
                              max(1, int(event.duration_beats * PPQ)),
                              self._velocity(energy, True)))
        return notes, root

    def _pattern_sub_drop(self, event, following, beats_per_bar, energy, previous):
        """코드 첫 박에 긴 저음 하나. 힙합/트랩의 808."""
        midi = self._place(event.chord.bass_pitch.pitch_class, previous)
        if midi - 12 >= self.low:
            midi -= 12      # 더 낮게 깐다
        length = event.duration_beats * (0.65 + 0.3 * energy)
        return [Note(midi, int(event.start_beat * PPQ), max(1, int(length * PPQ)),
                     self._velocity(energy, True))], midi

    def _pattern_walking(self, event, following, beats_per_bar, energy, previous):
        """워킹 베이스. 4분음표로 다음 코드 근음을 향해 걸어간다.

        마지막 박은 다음 코드 근음의 반음 또는 온음 아래/위에 놓는다.
        그래야 다음 코드로 자연스럽게 연결된다. 재즈 베이스의 핵심이다.
        """
        root = self._place(event.chord.bass_pitch.pitch_class, previous)
        count = max(1, int(round(event.duration_beats)))
        target_class = (following.chord.bass_pitch.pitch_class
                        if following is not None else event.chord.root.pitch_class)
        scale_classes = sorted(
            (self.key.tonic.midi + i) % 12 for i in self.key.scale.intervals
        )
        chord_classes = sorted(event.chord.pitch_classes)

        midis = [root]
        for step in range(1, count):
            if step == count - 1 and following is not None:
                # 마지막 박: 다음 근음에 반음/온음으로 접근한다
                target = self._place(target_class, root)
                approach = target + self.random.choice((-1, 1, -2, 2))
                midis.append(max(self.low, min(self.high, approach)))
            else:
                pool = chord_classes if step % 2 == 0 else scale_classes
                previous_midi = midis[-1]
                candidates = []
                for pitch_class in pool:
                    for candidate in (previous_midi - 12, previous_midi,
                                      previous_midi + 12):
                        placed = candidate - (candidate % 12) + pitch_class
                        for option in (placed - 12, placed, placed + 12):
                            if self.low <= option <= self.high:
                                distance = abs(option - previous_midi)
                                if 1 <= distance <= 5:
                                    candidates.append(option)
                midis.append(self.random.choice(candidates) if candidates else previous_midi)

        notes = []
        for index, midi in enumerate(midis):
            beat = event.start_beat + index
            if beat >= event.end_beat - 1e-9:
                break
            length = min(1.0, event.end_beat - beat)
            notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                              self._velocity(energy, index == 0)))
        return notes, midis[-1]


# ==========================================================================
# 드럼
# ==========================================================================

@dataclass(frozen=True, slots=True)
class DrumHit:
    """드럼 한 방. 마디 안에서의 박 위치."""

    beat: float
    instrument: str
    velocity: int = 100

    def __post_init__(self) -> None:
        if self.instrument not in DRUM_MIDI:
            raise ArrangeError(
                f"모르는 드럼입니다: {self.instrument!r}\n"
                f"사용 가능: {', '.join(sorted(DRUM_MIDI))}"
            )
        if self.beat < 0:
            raise ArrangeError(f"박 위치는 0 이상이어야 합니다: {self.beat}")


# 장르별 한 마디 패턴 (4/4 기준). 실제 연주 관습을 따른다.
DRUM_PATTERNS: dict[str, list[DrumHit]] = {
    "basic_rock": [
        DrumHit(0.0, "kick", 112), DrumHit(1.0, "snare", 104),
        DrumHit(2.0, "kick", 106), DrumHit(3.0, "snare", 104),
    ],
    "pop": [
        DrumHit(0.0, "kick", 112), DrumHit(1.0, "snare", 102),
        DrumHit(2.5, "kick", 98), DrumHit(3.0, "snare", 102),
    ],
    "four_on_floor": [
        DrumHit(0.0, "kick", 115), DrumHit(1.0, "kick", 108),
        DrumHit(2.0, "kick", 112), DrumHit(3.0, "kick", 108),
        DrumHit(1.0, "clap", 100), DrumHit(3.0, "clap", 100),
    ],
    "ballad": [
        DrumHit(0.0, "kick", 96), DrumHit(2.0, "snare", 88),
    ],
    "hiphop": [
        DrumHit(0.0, "kick", 118), DrumHit(1.0, "snare", 104),
        DrumHit(1.75, "kick", 96), DrumHit(2.5, "kick", 102),
        DrumHit(3.0, "snare", 104),
    ],
    "rnb": [
        DrumHit(0.0, "kick", 108), DrumHit(1.0, "snare", 100),
        DrumHit(2.25, "kick", 94), DrumHit(3.0, "snare", 100),
        DrumHit(3.5, "kick", 88),
    ],
    "jazz_swing": [
        DrumHit(0.0, "ride", 92), DrumHit(1.0, "ride", 80),
        DrumHit(1.66, "ride", 72), DrumHit(2.0, "ride", 88),
        DrumHit(3.0, "ride", 80), DrumHit(3.66, "ride", 72),
        DrumHit(1.0, "hihat_pedal", 70), DrumHit(3.0, "hihat_pedal", 70),
    ],
    "metal": [
        DrumHit(0.0, "kick", 118), DrumHit(0.5, "kick", 110),
        DrumHit(1.0, "snare", 112), DrumHit(1.5, "kick", 110),
        DrumHit(2.0, "kick", 116), DrumHit(2.5, "kick", 110),
        DrumHit(3.0, "snare", 112), DrumHit(3.5, "kick", 110),
    ],
    "punk": [
        DrumHit(0.0, "kick", 116), DrumHit(0.5, "kick", 104),
        DrumHit(1.0, "snare", 110), DrumHit(2.0, "kick", 114),
        DrumHit(2.5, "kick", 104), DrumHit(3.0, "snare", 110),
    ],
    "half_time": [
        DrumHit(0.0, "kick", 114), DrumHit(2.0, "snare", 106),
        DrumHit(3.5, "kick", 96),
    ],
    "minimal": [
        DrumHit(0.0, "kick", 100), DrumHit(2.0, "snare", 92),
    ],
    "none": [],
}


def _drum_pattern_for(profile: GenreProfile, energy: float) -> str:
    """장르와 에너지에 맞는 드럼 패턴 이름."""
    if profile.drum_density < 0.12:
        return "none"
    if profile.swing > 0.4:
        return "jazz_swing"
    if energy < 0.25:
        return "minimal"
    if profile.drum_density > 0.85 and profile.bpm_typical > 140:
        return "metal"
    if profile.bpm_typical > 150 and profile.drum_density > 0.8:
        return "punk"
    # 포온더플로어(모든 박에 킥)는 전자 댄스 음악의 주법이다. 기타 밴드 편성인
    # 장르에 얹으면 애니송·록이 클럽 음악처럼 들린다.
    electronic = (profile.instrument_weights.get("synth_bass", 0.0) > 0.5
                  and profile.instrument_weights.get("electric_guitar", 0.0) < 0.6)
    if electronic and profile.drum_density > 0.7 and profile.bpm_typical >= 115:
        return "four_on_floor"
    if profile.syncopation > 0.7 and profile.low_end > 0.6:
        return "hiphop"
    if profile.syncopation > 0.6:
        return "rnb"
    if profile.drum_density < 0.4:
        return "ballad"
    # 느린 곡에 록 드럼을 얹으면 안 맞는다. 하프타임이 맞다.
    if profile.bpm_typical < 95:
        return "half_time" if profile.drum_density > 0.5 else "ballad"
    # 기타가 중심인 편성은 정박 록 드럼이 맞는다. 'pop' 패턴은 2박 반에 킥이
    # 들어가는 댄스 계열 느낌이라 기타 밴드와는 어긋난다.
    guitar_led = profile.instrument_weights.get("electric_guitar", 0.0) >= 0.8
    if guitar_led:
        return "basic_rock"
    if profile.arrangement_density >= 0.6 and profile.brightness > 0.2:
        return "pop"
    return "basic_rock"


class DrumGenerator:
    """드럼 트랙을 만든다."""

    def __init__(self, genre: GenreProfile | str = "pop", seed: int | None = None) -> None:
        self.genre = get_genre(genre) if isinstance(genre, str) else genre
        self.random = random.Random(seed)

    def generate(
        self, bars: int, beats_per_bar: int = 4, energy: float = 0.5,
        pattern: str | None = None, start_tick: int = 0,
        fill_at_end: bool = True, crash_at_start: bool = False,
    ) -> list[Note]:
        if bars < 1:
            raise ArrangeError(f"마디 수는 1 이상이어야 합니다: {bars}")
        if pattern is None:
            pattern = _drum_pattern_for(self.genre, energy)
        if pattern not in DRUM_PATTERNS:
            raise ArrangeError(
                f"모르는 드럼 패턴입니다: {pattern!r}\n"
                f"사용 가능: {', '.join(DRUM_PATTERNS)}"
            )
        base = DRUM_PATTERNS[pattern]
        if not base:
            return []

        notes: list[Note] = []
        for bar in range(bars):
            bar_beat = bar * beats_per_bar
            is_last = bar == bars - 1
            # 마지막 마디에는 필을 넣는다. 구간이 바뀐다는 신호가 된다.
            if is_last and fill_at_end and bars >= 2 and energy > 0.3:
                notes.extend(self._fill(bar_beat, beats_per_bar, energy))
            else:
                for hit in base:
                    if hit.beat >= beats_per_bar:
                        continue
                    velocity = self._adjust(hit.velocity, energy)
                    notes.append(Note(
                        DRUM_MIDI[hit.instrument], int((bar_beat + hit.beat) * PPQ),
                        max(1, int(0.25 * PPQ)), velocity,
                    ))
                notes.extend(self._hihats(bar_beat, beats_per_bar, energy, pattern))
            if bar == 0 and crash_at_start:
                notes.append(Note(DRUM_MIDI["crash"], int(bar_beat * PPQ),
                                  int(PPQ), self._adjust(112, energy)))
        for note in notes:
            note.start_tick += start_tick
        return notes

    def _adjust(self, velocity: int, energy: float) -> int:
        scaled = int(velocity * (0.68 + 0.42 * energy))
        return max(1, min(127, scaled + self.random.randint(-4, 4)))

    def _hihats(self, bar_beat: float, beats_per_bar: int, energy: float,
                pattern: str) -> list[Note]:
        """하이햇. 밀도가 리듬의 느낌을 정한다."""
        if pattern in ("jazz_swing", "none"):
            return []
        density = self.genre.drum_density * (0.5 + 0.5 * energy)
        if density < 0.2:
            return []
        step = 1.0 if density < 0.45 else (0.5 if density < 0.8 else 0.25)
        notes: list[Note] = []
        beat = 0.0
        index = 0
        while beat < beats_per_bar - 1e-9:
            # 정박이 더 세다. 이 강약이 없으면 하이햇이 기계 소리가 된다.
            on_beat = abs(beat % 1.0) < 1e-6
            velocity = self._adjust(96 if on_beat else 74, energy)
            # 가끔 오픈 하이햇을 섞는다
            name = "hihat_closed"
            if energy > 0.55 and index % 8 == 7 and self.random.random() < 0.35:
                name = "hihat_open"
            notes.append(Note(DRUM_MIDI[name], int((bar_beat + beat) * PPQ),
                              max(1, int(step * PPQ)), velocity))
            beat += step
            index += 1
        return notes

    def _fill(self, bar_beat: float, beats_per_bar: int, energy: float) -> list[Note]:
        """드럼 필. 마지막 한두 박에 탐을 굴린다."""
        notes: list[Note] = []
        # 앞부분은 평소대로
        for hit in DRUM_PATTERNS["basic_rock"]:
            if hit.beat < beats_per_bar - 2:
                notes.append(Note(DRUM_MIDI[hit.instrument],
                                  int((bar_beat + hit.beat) * PPQ),
                                  max(1, int(0.25 * PPQ)), self._adjust(hit.velocity, energy)))
        toms = ["snare", "tom_high", "tom_mid", "tom_low"]
        step = 0.25 if energy > 0.6 else 0.5
        beat = beats_per_bar - 2.0
        index = 0
        while beat < beats_per_bar - 1e-9:
            name = toms[min(len(toms) - 1, index * len(toms) // max(1, int(2 / step)))]
            velocity = self._adjust(92 + index * 3, energy)
            notes.append(Note(DRUM_MIDI[name], int((bar_beat + beat) * PPQ),
                              max(1, int(step * PPQ)), velocity))
            beat += step
            index += 1
        return notes


# ==========================================================================
# 반주
# ==========================================================================

ACCOMPANIMENT_STYLES: dict[str, str] = {
    "block": "코드를 통째로 짚는다. 가장 기본.",
    "arpeggio_up": "낮은 음부터 차례로. 발라드 피아노, 어쿠스틱 기타.",
    "arpeggio_updown": "올라갔다 내려온다.",
    "broken": "근음과 나머지를 번갈아. 왈츠, 포크.",
    "pad": "코드 길이만큼 길게. 스트링, 패드.",
    "comp": "엇박으로 짧게 끊어 친다. 재즈, 펑크, 시티팝.",
    "eighth_chords": "8분음표로 계속 친다. 록, 펑크.",
    "strum": "기타 스트로크 흉내. 아래위로 훑는다.",
}


class AccompanimentGenerator:
    """반주를 만든다."""

    def __init__(self, genre: GenreProfile | str = "pop", seed: int | None = None) -> None:
        self.genre = get_genre(genre) if isinstance(genre, str) else genre
        self.random = random.Random(seed)

    def style_for(self, energy: float) -> str:
        if self.genre.arrangement_density > 0.85 and self.genre.drum_density < 0.4:
            return "pad"
        if self.genre.swing > 0.4 or self.genre.syncopation > 0.65:
            return "comp"
        if energy < 0.35:
            return "arpeggio_up" if self.genre.bpm_typical < 110 else "pad"
        if self.genre.drum_density > 0.75 and energy > 0.6:
            return "eighth_chords"
        if self.genre.bpm_typical < 95:
            return "arpeggio_updown"
        return "block"

    def generate(
        self, progression: ChordProgression, beats_per_bar: int = 4,
        energy: float = 0.5, style: str | None = None, start_tick: int = 0,
        low_midi: int = 48, high_midi: int = 79,
    ) -> list[Note]:
        if style is None:
            style = self.style_for(energy)
        if style not in ACCOMPANIMENT_STYLES:
            raise ArrangeError(
                f"모르는 반주 형태입니다: {style!r}\n"
                f"사용 가능: {', '.join(ACCOMPANIMENT_STYLES)}"
            )
        voicings = progression.voice_lead(low_midi, high_midi)
        maker = getattr(self, f"_style_{style}")
        notes: list[Note] = []
        for event, voicing in zip(progression, voicings):
            notes.extend(maker(event, [p.midi for p in voicing], beats_per_bar, energy))
        for note in notes:
            note.start_tick += start_tick
        return notes

    def _velocity(self, energy: float, accent: bool = False) -> int:
        base = 58 + int(34 * energy) + (7 if accent else 0)
        return max(1, min(127, base + self.random.randint(-4, 4)))

    def _style_block(self, event, midis, beats_per_bar, energy):
        notes = []
        beat = event.start_beat
        while beat < event.end_beat - 1e-9:
            length = min(1.0, event.end_beat - beat)
            accent = abs(beat % beats_per_bar) < 1e-6
            for midi in midis:
                notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                                  self._velocity(energy, accent)))
            beat += 1.0
        return notes

    def _style_pad(self, event, midis, beats_per_bar, energy):
        length = max(1, int(event.duration_beats * PPQ))
        return [Note(midi, int(event.start_beat * PPQ), length, self._velocity(energy))
                for midi in midis]

    def _style_eighth_chords(self, event, midis, beats_per_bar, energy):
        notes = []
        beat = event.start_beat
        index = 0
        while beat < event.end_beat - 1e-9:
            length = min(0.5, event.end_beat - beat)
            accent = index % 2 == 0
            for midi in midis:
                notes.append(Note(midi, int(beat * PPQ), max(1, int(length * 0.85 * PPQ)),
                                  self._velocity(energy, accent)))
            beat += 0.5
            index += 1
        return notes

    def _style_arpeggio_up(self, event, midis, beats_per_bar, energy):
        return self._arpeggio(event, midis, energy, updown=False)

    def _style_arpeggio_updown(self, event, midis, beats_per_bar, energy):
        return self._arpeggio(event, midis, energy, updown=True)

    def _arpeggio(self, event, midis, energy, updown: bool):
        order = list(midis)
        if updown and len(order) > 2:
            order = order + order[-2:0:-1]
        step = 0.5 if event.duration_beats >= 2 else 0.25
        notes = []
        beat = event.start_beat
        index = 0
        while beat < event.end_beat - 1e-9:
            length = min(step * 1.6, event.end_beat - beat)
            midi = order[index % len(order)]
            notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                              self._velocity(energy, index % len(order) == 0)))
            beat += step
            index += 1
        return notes

    def _style_broken(self, event, midis, beats_per_bar, energy):
        if not midis:
            return []
        root, rest = midis[0], midis[1:] or midis
        notes = []
        beat = event.start_beat
        index = 0
        while beat < event.end_beat - 1e-9:
            length = min(0.5, event.end_beat - beat)
            if index % 2 == 0:
                notes.append(Note(root, int(beat * PPQ), max(1, int(length * PPQ)),
                                  self._velocity(energy, True)))
            else:
                for midi in rest:
                    notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                                      self._velocity(energy)))
            beat += 0.5
            index += 1
        return notes

    def _style_comp(self, event, midis, beats_per_bar, energy):
        """엇박으로 짧게 끊어 친다. 매번 같은 자리에 치면 기계 같으므로 흔든다."""
        offsets = [0.0, 0.75, 1.5, 2.25, 3.0, 3.5]
        notes = []
        for offset in offsets:
            beat = event.start_beat + offset
            if beat >= event.end_beat - 1e-9:
                continue
            if offset > 0 and self.random.random() > 0.35 + 0.4 * energy:
                continue
            length = min(0.4, event.end_beat - beat)
            for midi in midis:
                notes.append(Note(midi, int(beat * PPQ), max(1, int(length * PPQ)),
                                  self._velocity(energy, offset == 0.0)))
        if not notes:
            return self._style_pad(event, midis, beats_per_bar, energy)
        return notes

    def _style_strum(self, event, midis, beats_per_bar, energy):
        """스트로크. 줄을 훑으므로 음이 아주 조금씩 늦게 울린다.

        이 시간차가 없으면 기타가 아니라 오르간처럼 들린다.
        """
        notes = []
        beat = event.start_beat
        index = 0
        while beat < event.end_beat - 1e-9:
            down = index % 2 == 0
            order = midis if down else list(reversed(midis))
            spread = 0.018 if down else 0.012
            length = min(0.5, event.end_beat - beat)
            for position, midi in enumerate(order):
                offset = position * spread
                notes.append(Note(
                    midi, int((beat + offset) * PPQ),
                    max(1, int((length - offset) * PPQ)),
                    self._velocity(energy, down and index % 4 == 0),
                ))
            beat += 0.5
            index += 1
        return notes
