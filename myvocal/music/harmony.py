"""
코드 진행 생성 — 5번의 Chord.

무작위로 뽑으면 안 된다. 화성에는 방향이 있다. 사람이 음악을 들을 때
'긴장이 쌓였다가 풀린다'고 느끼는 건 화음의 기능 때문이다.

    으뜸 (T)      I, vi, iii        집. 안정. 여기서 출발하고 여기로 돌아온다
    버금딸림 (S)  IV, ii            집을 떠남. 긴장이 조금 쌓임
    딸림 (D)      V, vii°          가장 긴장. 으뜸으로 가고 싶어 한다

이 순서(T -> S -> D -> T)가 화성의 기본 흐름이다. D 에서 S 로 되돌아가는
움직임은 어색하게 들려서 거의 쓰지 않는다. 그래서 전이 확률로 그 성질을
그대로 담는다.

여기에 장르별 성향(7화음 사용률, 텐션, 조성 밖 화음)을 곱한다.
재즈는 7화음이 95%, 펑크는 5% 다. 같은 규칙에 값만 다르다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .genre import GenreProfile, get_genre
from .theory import Chord, Key, Pitch, TheoryError


class HarmonyError(ValueError):
    """화성 생성 관련 오류."""


# 음계도별 기능. 장조 기준.
MAJOR_FUNCTION: dict[int, str] = {1: "T", 2: "S", 3: "T", 4: "S", 5: "D", 6: "T", 7: "D"}
MINOR_FUNCTION: dict[int, str] = {1: "T", 2: "D", 3: "T", 4: "S", 5: "D", 6: "S", 7: "S"}

# 기능 사이의 전이 가중치. D -> S 를 낮게 잡는 것이 핵심이다.
FUNCTION_TRANSITIONS: dict[str, dict[str, float]] = {
    "T": {"T": 0.20, "S": 0.45, "D": 0.35},
    "S": {"T": 0.20, "S": 0.20, "D": 0.60},
    "D": {"T": 0.75, "S": 0.05, "D": 0.20},
}

# 같은 기능 안에서 어느 화음을 고를지. 1도와 5도가 가장 흔하다.
DEGREE_WEIGHTS_MAJOR: dict[str, dict[int, float]] = {
    "T": {1: 0.65, 6: 0.28, 3: 0.07},
    "S": {4: 0.60, 2: 0.40},
    "D": {5: 0.88, 7: 0.12},
}
DEGREE_WEIGHTS_MINOR: dict[str, dict[int, float]] = {
    "T": {1: 0.70, 3: 0.30},
    "S": {4: 0.50, 6: 0.30, 2: 0.20},
    "D": {5: 0.60, 7: 0.40},
}

# 장르마다 자주 쓰는 진행. 처음 만들 때 이 중에서 고르면 훨씬 자연스럽다.
#
# 장조와 단조를 따로 둔다. 장조 표기를 단조에 그대로 쓰면 안 된다.
# 단조는 6도와 7도가 이미 내려가 있어서, 거기에 또 b 를 붙이면 겹내림이 된다
# (A단조에서 bVI 는 Fb = E 가 되어 전혀 다른 화음이 나온다).
# 단조에서는 VI, VII 로 쓰고, 딸림화음만 화성단조의 장3도를 써서 V 로 쓴다.
COMMON_PROGRESSIONS: dict[str, dict[str, list[list[str]]]] = {
    "pop": {
        "major": [["I", "V", "vi", "IV"], ["vi", "IV", "I", "V"],
                  ["I", "IV", "V", "I"], ["I", "vi", "IV", "V"]],
        "minor": [["i", "VI", "III", "VII"], ["i", "VII", "VI", "V"],
                  ["i", "iv", "VII", "III"], ["VI", "VII", "i", "i"]],
    },
    "kpop": {
        "major": [["I", "V", "vi", "IV"], ["vi", "IV", "I", "V"],
                  ["IV", "V", "iii", "vi"], ["I", "iii", "IV", "V"],
                  ["vi", "V", "IV", "V"]],
        "minor": [["i", "VII", "VI", "VII"], ["iv", "V", "III", "VI"],
                  ["i", "VI", "III", "VII"], ["i", "v", "VI", "VII"]],
    },
    "ballad": {
        "major": [["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
                  ["I", "vi", "IV", "V"],
                  ["IV", "V", "iii", "vi", "ii", "V", "I"],
                  ["I", "IV", "vi", "V"]],
        "minor": [["i", "VII", "VI", "V"], ["i", "iv", "VII", "III"],
                  ["i", "VI", "iv", "V"], ["iv", "V", "i", "i"]],
    },
    "rock": {
        "major": [["I", "bVII", "IV", "I"], ["I", "IV", "V", "IV"],
                  ["vi", "IV", "I", "V"], ["I", "V", "IV", "IV"]],
        "minor": [["i", "VI", "III", "VII"], ["i", "VII", "VI", "VII"],
                  ["i", "iv", "VII", "i"], ["i", "III", "VII", "iv"]],
    },
    "pop_rock": {
        "major": [["I", "V", "vi", "IV"], ["I", "bVII", "IV", "I"]],
        "minor": [["i", "VI", "III", "VII"]],
    },
    "rnb": {
        "major": [["ii7", "V7", "Imaj7", "vi7"], ["Imaj7", "iii7", "vi7", "ii7"],
                  ["Imaj7", "vi7", "ii7", "V7"]],
        "minor": [["i7", "iv7", "VII7", "IIImaj7"], ["i7", "VI", "ii7", "V7"]],
    },
    "jazz": {
        "major": [["ii7", "V7", "Imaj7", "vi7"], ["Imaj7", "vi7", "ii7", "V7"],
                  ["iii7", "vi7", "ii7", "V7", "Imaj7"]],
        # 단조의 2도는 반감화음이다. ii7 로 쓰면 Bm7 이 되어 조성 밖으로 나간다.
        "minor": [["ii7b5", "V7", "i7", "i7"], ["i7", "iv7", "VII7", "IIImaj7"],
                  ["ii7b5", "V7", "i7", "iv7"]],
    },
    "citypop": {
        "major": [["Imaj7", "iii7", "vi7", "V7"], ["IVmaj7", "V7", "iii7", "vi7"],
                  ["ii7", "V7", "Imaj7", "IVmaj7"]],
        "minor": [["i7", "VII", "VImaj7", "V7"], ["iv7", "V7", "IIImaj7", "VImaj7"]],
    },
    "anime": {
        "major": [["IV", "V", "iii", "vi"],
                  ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
                  ["vi", "V", "IV", "V"], ["IV", "V", "I", "vi"]],
        "minor": [["iv", "V", "III", "VI"], ["i", "VII", "VI", "V"],
                  ["VI", "VII", "i", "i"], ["i", "VI", "III", "VII"]],
    },
    "jpop": {
        "major": [["IV", "V", "iii", "vi"], ["I", "V", "vi", "IV"],
                  ["IV", "V", "I", "I"]],
        "minor": [["iv", "V", "III", "VI"], ["i", "VII", "VI", "V"]],
    },
    "vocalsynth": {
        "major": [["IV", "V", "iii", "vi"], ["I", "V", "vi", "IV"],
                  ["vi", "V", "IV", "V"]],
        "minor": [["i", "VII", "VI", "VII"], ["iv", "V", "III", "VI"]],
    },
    "edm": {
        "major": [["vi", "IV", "I", "V"], ["I", "V", "vi", "IV"]],
        "minor": [["i", "VI", "III", "VII"], ["i", "VII", "VI", "V"],
                  ["i", "VI", "VII", "VII"]],
    },
    "house": {
        "major": [["Imaj7", "vi7", "ii7", "V7"], ["I", "V", "vi", "IV"]],
        "minor": [["i7", "VI", "III", "VII"], ["i7", "iv7", "VII", "III"]],
    },
    "techno": {
        "major": [["I", "I", "IV", "IV"]],
        "minor": [["i", "i", "VI", "VI"], ["i", "VII", "i", "VII"]],
    },
    "hiphop": {
        "major": [["I", "vi", "IV", "V"], ["Imaj7", "vi7", "ii7", "V7"]],
        "minor": [["i", "VI", "VII", "i"], ["i7", "iv7", "i7", "v7"],
                  ["i", "VII", "VI", "VII"]],
    },
    "lofi": {
        "major": [["ii7", "V7", "Imaj7", "vi7"], ["Imaj7", "vi7", "ii7", "V7"]],
        "minor": [["i7", "iv7", "VII7", "IIImaj7"], ["i7", "VI", "iv7", "V7"]],
    },
    "orchestral": {
        "major": [["I", "IV", "V", "I"], ["I", "vi", "ii", "V", "I"],
                  ["I", "V", "vi", "iii", "IV", "I", "IV", "V"]],
        "minor": [["i", "iv", "V", "i"], ["i", "VI", "iv", "V"]],
    },
    "cinematic": {
        "major": [["I", "IV", "I", "V"], ["I", "vi", "IV", "V"]],
        "minor": [["i", "VI", "III", "VII"], ["i", "VII", "VI", "V"],
                  ["i", "iv", "VI", "V"]],
    },
    "synthwave": {
        "major": [["I", "V", "vi", "IV"]],
        "minor": [["i", "VI", "III", "VII"], ["i", "VII", "VI", "VII"]],
    },
    "metal": {
        "major": [["I", "bVII", "IV", "I"]],
        "minor": [["i", "VI", "VII", "i"], ["i", "bII", "i", "VII"],
                  ["i", "VII", "VI", "V"]],
    },
    "punk": {
        "major": [["I", "IV", "V", "V"], ["I", "V", "vi", "IV"]],
        "minor": [["i", "VI", "VII", "i"]],
    },
    "acoustic": {
        "major": [["I", "V", "vi", "IV"], ["I", "IV", "I", "V"]],
        "minor": [["i", "VII", "VI", "V"], ["i", "iv", "VII", "III"]],
    },
    "ambient": {
        "major": [["Imaj7", "IVmaj7", "Imaj7", "IVmaj7"]],
        "minor": [["i7", "VImaj7", "i7", "VImaj7"]],
    },
}

# 종지. 구간 끝에서 쓴다. 종지가 없으면 곡이 안 끝난 느낌이 든다.
CADENCES: dict[str, list[str]] = {
    "authentic": ["V", "I"],        # 정격종지. 가장 확실하게 끝난다.
    "plagal": ["IV", "I"],          # 변격종지. 부드럽게 끝난다.
    "half": ["I", "V"],             # 반종지. 이어진다는 느낌. 벌스 끝에 쓴다.
    "deceptive": ["V", "vi"],       # 위종지. 끝날 듯 안 끝난다.
    "minor_authentic": ["V", "i"],
    "minor_plagal": ["iv", "i"],
    "minor_half": ["i", "V"],
}


@dataclass(slots=True)
class ChordEvent:
    """진행 안의 코드 하나. 몇 마디 몇 박부터 얼마 동안인가."""

    chord: Chord
    start_beat: float
    duration_beats: float
    roman: str = ""
    function: str = ""

    def __post_init__(self) -> None:
        if self.duration_beats <= 0:
            raise HarmonyError(f"코드 길이는 0보다 커야 합니다: {self.duration_beats}")
        if self.start_beat < 0:
            raise HarmonyError(f"시작 박은 0 이상이어야 합니다: {self.start_beat}")

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.duration_beats

    def contains_beat(self, beat: float) -> bool:
        return self.start_beat <= beat < self.end_beat

    def __str__(self) -> str:
        return f"{self.chord.symbol}({self.roman}) {self.start_beat:g}~{self.end_beat:g}박"


class ChordProgression:
    """한 구간의 코드 진행."""

    __slots__ = ("events", "key")

    def __init__(self, events: Iterable[ChordEvent], key: Key) -> None:
        self.events: list[ChordEvent] = sorted(events, key=lambda e: e.start_beat)
        self.key = key

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, index) -> ChordEvent:
        return self.events[index]

    @property
    def total_beats(self) -> float:
        return max((e.end_beat for e in self.events), default=0.0)

    def at_beat(self, beat: float) -> ChordEvent | None:
        for event in self.events:
            if event.contains_beat(beat):
                return event
        return None

    def chord_at(self, beat: float) -> Chord | None:
        event = self.at_beat(beat)
        return event.chord if event else None

    def transposed(self, semitones: int) -> "ChordProgression":
        return ChordProgression(
            [ChordEvent(e.chord.transpose(semitones), e.start_beat, e.duration_beats,
                        e.roman, e.function) for e in self.events],
            self.key.transpose(semitones),
        )

    def symbols(self) -> list[str]:
        return [e.chord.symbol for e in self.events]

    def romans(self) -> list[str]:
        return [e.roman for e in self.events]

    def describe(self) -> str:
        return " | ".join(f"{e.roman}({e.chord.symbol})" for e in self.events)

    def voice_lead(self, low_midi: int = 48, high_midi: int = 72) -> list[list[Pitch]]:
        """코드마다 어느 옥타브에 어떻게 벌려 놓을지 정한다.

        앞 코드와 공통음을 최대한 유지하고 각 성부가 적게 움직이도록 고른다.
        이걸 안 하면 코드마다 손이 건반 위를 껑충껑충 뛰어다니는 소리가 난다.
        """
        if low_midi >= high_midi:
            raise HarmonyError(f"음역이 잘못됐습니다: {low_midi} ~ {high_midi}")
        result: list[list[Pitch]] = []
        previous: list[int] | None = None
        for event in self.events:
            candidates: list[tuple[float, list[int]]] = []
            for inversion in range(event.chord.size):
                voiced = event.chord.inverted(inversion)
                for octave in range(2, 7):
                    midis = [p.midi for p in voiced.pitches(octave=octave)]
                    if not midis or min(midis) < low_midi or max(midis) > high_midi:
                        continue
                    if previous is None:
                        # 첫 코드는 음역 가운데에 놓는다
                        center = (low_midi + high_midi) / 2
                        cost = abs(sum(midis) / len(midis) - center)
                    else:
                        # 각 성부가 가장 가까운 앞 음에서 얼마나 움직였는가
                        cost = sum(min(abs(m - p) for p in previous) for m in midis)
                        cost += abs(len(midis) - len(previous)) * 2.0
                    candidates.append((cost, midis))
            if not candidates:
                # 음역에 안 들어가면 그냥 기본 위치를 쓴다
                midis = [p.midi for p in event.chord.pitches(octave=4)]
                candidates.append((0.0, midis))
            candidates.sort(key=lambda pair: pair[0])
            chosen = candidates[0][1]
            previous = chosen
            prefer_flat = event.chord.root.alter < 0
            result.append([Pitch.from_midi(m, prefer_flat=prefer_flat) for m in chosen])
        return result


def _weighted_choice(weights: dict, generator: random.Random):
    """가중치대로 하나 고른다."""
    items = [(k, v) for k, v in weights.items() if v > 0]
    if not items:
        raise HarmonyError("고를 수 있는 항목이 없습니다.")
    total = sum(v for _, v in items)
    threshold = generator.random() * total
    running = 0.0
    for key, value in items:
        running += value
        if running >= threshold:
            return key
    return items[-1][0]


class ProgressionGenerator:
    """코드 진행을 만든다."""

    def __init__(self, key: Key, genre: GenreProfile | str = "pop", seed: int | None = None) -> None:
        self.key = key
        self.genre = get_genre(genre) if isinstance(genre, str) else genre
        self.random = random.Random(seed)

    # ---- 화음 하나 꾸미기 ----

    def _decorate(self, degree: int, roman: str, is_last: bool) -> tuple[Chord, str]:
        """도수를 실제 화음으로 바꾸면서 장르에 맞게 7화음/텐션을 붙인다."""
        use_seventh = self.random.random() < self.genre.seventh_chords
        chord = self.key.diatonic_chord(degree, seventh=use_seventh)
        text = roman
        if use_seventh:
            text = roman + ("7" if not chord.kind.startswith("maj") else "M7")
        # 텐션은 7화음 위에만 얹는다. 3화음에 9도만 붙이면 붕 뜬다.
        if use_seventh and self.random.random() < self.genre.tension_notes:
            upgrade = {"maj7": "maj9", "7": "9", "m7": "m9"}.get(chord.kind)
            if upgrade is not None:
                chord = chord.as_kind(upgrade)
                text = text.replace("7", "9")
        return chord, text

    def _borrowed(self) -> tuple[Chord, str] | None:
        """조성 밖 화음. 장조에서 단조 화음을 빌려오는 것이 가장 흔하다."""
        if self.key.is_minor:
            options = [("bII", 1, ""), ("IV", 5, ""), ("V", 7, "")]
        else:
            options = [("bVII", 10, ""), ("bVI", 8, ""), ("iv", 5, "m"), ("bIII", 3, "")]
        label, semitones, kind = self.random.choice(options)
        root = self.key.tonic.transpose(semitones, prefer_flat=True)
        return Chord(root, kind), label

    # ---- 진행 만들기 ----

    def generate(
        self, bars: int, beats_per_bar: int = 4, chords_per_bar: float | None = None,
        cadence: str | None = None, start_degree: int = 1,
    ) -> ChordProgression:
        """기능 화성 전이로 진행을 만든다.

        cadence 를 주면 마지막 두 화음을 그 종지로 놓는다. 구간 끝에서 쓴다.
        """
        if bars < 1:
            raise HarmonyError(f"마디 수는 1 이상이어야 합니다: {bars}")
        if chords_per_bar is None:
            chords_per_bar = max(0.25, min(4.0, self.genre.harmonic_rhythm))

        # 코드 하나의 길이(박). 마디당 코드 수의 역수다.
        chord_beats = beats_per_bar / chords_per_bar
        # 박자에 안 맞는 길이가 나오면 가장 가까운 자연스러운 값으로 맞춘다
        allowed = [beats_per_bar * 4, beats_per_bar * 2, beats_per_bar,
                   beats_per_bar / 2, beats_per_bar / 4]
        chord_beats = min(allowed, key=lambda v: abs(v - chord_beats))
        count = max(1, int(round(bars * beats_per_bar / chord_beats)))

        function_table = MINOR_FUNCTION if self.key.is_minor else MAJOR_FUNCTION
        degree_table = DEGREE_WEIGHTS_MINOR if self.key.is_minor else DEGREE_WEIGHTS_MAJOR

        cadence_romans: list[str] = []
        if cadence:
            key_name = cadence
            if self.key.is_minor and f"minor_{cadence}" in CADENCES:
                key_name = f"minor_{cadence}"
            if key_name not in CADENCES:
                raise HarmonyError(
                    f"모르는 종지입니다: {cadence!r} (가능: {', '.join(CADENCES)})"
                )
            cadence_romans = CADENCES[key_name]

        body_count = max(1, count - len(cadence_romans))
        degrees: list[int] = [start_degree]
        current_function = function_table[start_degree]
        for _ in range(body_count - 1):
            next_function = _weighted_choice(FUNCTION_TRANSITIONS[current_function], self.random)
            weights = dict(degree_table[next_function])
            # 바로 앞과 같은 화음이 이어지는 것을 줄인다
            if degrees[-1] in weights:
                weights[degrees[-1]] *= 0.25
            degree = _weighted_choice(weights, self.random)
            degrees.append(degree)
            current_function = next_function

        events: list[ChordEvent] = []
        beat = 0.0
        for index, degree in enumerate(degrees):
            roman_base = self.key.roman_numeral(self.key.diatonic_chord(degree))
            # 조성 밖 화음은 첫 자리와 마지막 자리에는 넣지 않는다
            if (0 < index < len(degrees) - 1
                    and self.random.random() < self.genre.borrowed_chords):
                borrowed = self._borrowed()
                if borrowed is not None:
                    chord, label = borrowed
                    events.append(ChordEvent(chord, beat, chord_beats, label, "S"))
                    beat += chord_beats
                    continue
            chord, roman = self._decorate(degree, roman_base, index == len(degrees) - 1)
            events.append(ChordEvent(chord, beat, chord_beats, roman, function_table[degree]))
            beat += chord_beats

        for roman in cadence_romans:
            chord = self.key.chord_from_roman(roman)
            degree = self.key.degree_of_chord(chord) or 1
            if self.random.random() < self.genre.seventh_chords and roman.upper() == "V":
                chord = chord.as_kind("7")
                roman = roman + "7"
            events.append(ChordEvent(chord, beat, chord_beats, roman, function_table[degree]))
            beat += chord_beats

        return ChordProgression(events, self.key)

    def from_template(
        self, bars: int, beats_per_bar: int = 4, template: Sequence[str] | None = None,
        chords_per_bar: float = 1.0,
    ) -> ChordProgression:
        """장르에서 흔한 진행을 가져다 필요한 길이만큼 반복한다.

        처음 만드는 곡에서는 이쪽이 훨씬 자연스럽다. 전이 확률로 만든 진행은
        문법적으로는 맞지만 '어디서 들어본 것 같은' 친숙함이 없다.
        """
        if bars < 1:
            raise HarmonyError(f"마디 수는 1 이상이어야 합니다: {bars}")
        if chords_per_bar <= 0:
            raise HarmonyError(f"마디당 코드 수는 0보다 커야 합니다: {chords_per_bar}")

        if template is None:
            template = self._pick_template()

        chord_beats = beats_per_bar / chords_per_bar
        total_beats = bars * beats_per_bar
        function_table = MINOR_FUNCTION if self.key.is_minor else MAJOR_FUNCTION

        events: list[ChordEvent] = []
        beat = 0.0
        while beat < total_beats - 1e-9:
            for roman in template:
                if beat >= total_beats - 1e-9:
                    break
                try:
                    chord = self.key.chord_from_roman(roman)
                except TheoryError as error:
                    raise HarmonyError(
                        f"도수 표기를 읽을 수 없습니다: {roman!r} ({error})"
                    ) from error
                degree = self.key.degree_of_chord(chord)
                function = function_table.get(degree, "S") if degree else "S"
                length = min(chord_beats, total_beats - beat)
                events.append(ChordEvent(chord, beat, length, roman, function))
                beat += length
        return ChordProgression(events, self.key)

    def _pick_template(self) -> list[str]:
        """장르와 조성에 맞는 진행 틀을 고른다."""
        mode = "minor" if self.key.is_minor else "major"
        candidates: list[list[str]] = []
        # 혼합 장르면 섞인 이름이 "kpop+orchestral+rock" 처럼 들어온다.
        names = self.genre.name.split("+")
        for name in names:
            entry = COMMON_PROGRESSIONS.get(name)
            if entry and entry.get(mode):
                candidates.extend(entry[mode])
        if not candidates:
            candidates = COMMON_PROGRESSIONS["pop"][mode]
        return list(self.random.choice(candidates))


def apply_cadence(
    progression: ChordProgression, cadence: str, beats_per_bar: int = 4,
    cadence_bars: int = 2,
) -> ChordProgression:
    """진행의 마지막 몇 마디를 종지로 바꾼다. 전체 길이는 그대로 둔다.

    길이를 늘리면 안 된다. 구간 길이는 곡 구조가 정한 값이고, 여기서 한 마디라도
    늘어나면 그 편곡이 다음 구간으로 넘쳐 들어가 음이 겹친다.
    """
    key = progression.key
    name = cadence
    if key.is_minor and f"minor_{cadence}" in CADENCES:
        name = f"minor_{cadence}"
    if name not in CADENCES:
        raise HarmonyError(f"모르는 종지입니다: {cadence!r} (가능: {', '.join(CADENCES)})")
    romans = CADENCES[name]

    total = progression.total_beats
    span = min(total, cadence_bars * beats_per_bar)
    start = total - span
    if start <= 0:
        # 진행이 종지 길이보다 짧으면 전체를 종지로 채운다
        start = 0.0
        span = total

    function_table = MINOR_FUNCTION if key.is_minor else MAJOR_FUNCTION
    events: list[ChordEvent] = []
    for event in progression:
        if event.start_beat >= start - 1e-9:
            continue
        if event.end_beat > start + 1e-9:
            # 경계를 걸친 코드는 경계에서 자른다
            events.append(ChordEvent(event.chord, event.start_beat, start - event.start_beat,
                                     event.roman, event.function))
        else:
            events.append(event)

    each = span / len(romans)
    beat = start
    for roman in romans:
        chord = key.chord_from_roman(roman)
        degree = key.degree_of_chord(chord)
        function = function_table.get(degree, "D") if degree else "D"
        events.append(ChordEvent(chord, beat, each, roman, function))
        beat += each
    return ChordProgression(events, key)
