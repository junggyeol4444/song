"""
음소와 시간 — 33번의 Lyrics -> Phoneme -> Timing.

음 하나에 가사 한 음절이 붙는다. 그 음절은 다시 자음과 모음으로 나뉘고,
각각 언제 시작해 얼마나 이어지는지가 정해진다. 그 시간이 그대로

    · 노래 합성의 발음 시간
    · 영상의 입 모양 시간 (33번)

이 된다. 두 곳에서 따로 계산하면 반드시 어긋난다.

시간을 나누는 원칙: 자음은 짧고 모음이 길다. 그리고 **자음은 박자 앞에
놓인다.** 사람이 노래할 때 '나' 를 부르면 'ㄴ' 이 박자보다 살짝 앞서고
'ㅏ' 가 박자에 떨어진다. 반대로 하면 모든 음이 늦게 들린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .korean import (
    FINAL_SOUND, KoreanError, Syllable, apply_sound_changes, decompose,
    is_hangul_syllable, split_syllables,
)


class PhonemeKind(Enum):
    """음소의 종류. 합성 방식과 입 모양이 여기서 갈린다."""

    VOWEL = "vowel"              # 모음. 소리의 중심.
    NASAL = "nasal"              # ㄴ ㅁ ㅇ. 코로 울린다.
    LIQUID = "liquid"            # ㄹ. 혀를 튕기거나 옆으로 흘린다.
    PLOSIVE = "plosive"          # ㄱ ㄷ ㅂ. 막았다 터뜨린다.
    FRICATIVE = "fricative"      # ㅅ ㅎ. 좁은 틈으로 바람을 낸다.
    AFFRICATE = "affricate"      # ㅈ ㅊ. 막았다가 마찰로 푼다.
    SILENCE = "silence"          # 소리 없음.


# 모음의 포먼트. (F1, F2, F3) Hz.
# 포먼트는 입과 목의 공명 주파수다. 이 값이 모음을 결정한다.
# 성대가 내는 소리는 모음마다 똑같고, 입 모양이 만드는 공명이 다를 뿐이다.
# 값은 한국어 모음 음향 연구의 성인 여성 평균치를 기준으로 했다.
VOWEL_FORMANTS: dict[str, tuple[float, float, float]] = {
    "ㅏ": (900.0, 1400.0, 2900.0),
    "ㅓ": (650.0, 1200.0, 2700.0),
    "ㅗ": (480.0, 800.0, 2700.0),
    "ㅜ": (380.0, 800.0, 2600.0),
    "ㅡ": (420.0, 1600.0, 2500.0),
    "ㅣ": (320.0, 2700.0, 3200.0),
    "ㅔ": (500.0, 2300.0, 2900.0),
    "ㅐ": (600.0, 2100.0, 2800.0),
}

# 겹모음은 반모음 + 단모음이다. ㅑ = j + ㅏ
DIPHTHONGS: dict[str, tuple[str, str]] = {
    "ㅑ": ("j", "ㅏ"), "ㅕ": ("j", "ㅓ"), "ㅛ": ("j", "ㅗ"), "ㅠ": ("j", "ㅜ"),
    "ㅒ": ("j", "ㅐ"), "ㅖ": ("j", "ㅔ"),
    "ㅘ": ("w", "ㅏ"), "ㅝ": ("w", "ㅓ"), "ㅙ": ("w", "ㅐ"), "ㅞ": ("w", "ㅔ"),
    "ㅚ": ("w", "ㅔ"), "ㅟ": ("w", "ㅣ"), "ㅢ": ("ɰ", "ㅣ"),
}

# 반모음의 포먼트. 짧게 스쳐 지나간다.
GLIDE_FORMANTS: dict[str, tuple[float, float, float]] = {
    "j": (300.0, 2500.0, 3100.0),     # ㅣ 에 가깝다
    "w": (320.0, 700.0, 2500.0),      # ㅜ 에 가깝다
    "ɰ": (400.0, 1500.0, 2500.0),     # ㅡ 에 가깝다
}

# 자음의 성질.
# (종류, 조음 위치의 대표 주파수, 기본 길이(초), 유성 여부, 기식 정도)
CONSONANTS: dict[str, tuple[PhonemeKind, float, float, bool, float]] = {
    "ㄱ": (PhonemeKind.PLOSIVE, 2000.0, 0.045, False, 0.35),
    "ㄲ": (PhonemeKind.PLOSIVE, 2100.0, 0.040, False, 0.10),
    "ㅋ": (PhonemeKind.PLOSIVE, 2000.0, 0.075, False, 0.85),
    "ㄷ": (PhonemeKind.PLOSIVE, 3500.0, 0.040, False, 0.30),
    "ㄸ": (PhonemeKind.PLOSIVE, 3600.0, 0.035, False, 0.08),
    "ㅌ": (PhonemeKind.PLOSIVE, 3500.0, 0.070, False, 0.85),
    "ㅂ": (PhonemeKind.PLOSIVE, 800.0, 0.040, False, 0.30),
    "ㅃ": (PhonemeKind.PLOSIVE, 800.0, 0.035, False, 0.08),
    "ㅍ": (PhonemeKind.PLOSIVE, 800.0, 0.070, False, 0.85),
    "ㅅ": (PhonemeKind.FRICATIVE, 6500.0, 0.085, False, 0.55),
    "ㅆ": (PhonemeKind.FRICATIVE, 6800.0, 0.095, False, 0.35),
    "ㅎ": (PhonemeKind.FRICATIVE, 1500.0, 0.055, False, 0.95),
    "ㅈ": (PhonemeKind.AFFRICATE, 3000.0, 0.060, False, 0.30),
    "ㅉ": (PhonemeKind.AFFRICATE, 3100.0, 0.055, False, 0.08),
    "ㅊ": (PhonemeKind.AFFRICATE, 3000.0, 0.085, False, 0.85),
    "ㄴ": (PhonemeKind.NASAL, 250.0, 0.055, True, 0.0),
    "ㅁ": (PhonemeKind.NASAL, 200.0, 0.060, True, 0.0),
    "ㅇ": (PhonemeKind.NASAL, 300.0, 0.060, True, 0.0),
    "ㄹ": (PhonemeKind.LIQUID, 450.0, 0.040, True, 0.0),
}

# 입 모양 (viseme). 여러 음소가 같은 입 모양을 쓴다.
# 영상에서는 이 단위로 입을 움직인다 (33번).
VISEMES: dict[str, str] = {
    "ㅏ": "A", "ㅐ": "A", "ㅓ": "E", "ㅔ": "E",
    "ㅗ": "O", "ㅜ": "U", "ㅡ": "I", "ㅣ": "I",
    "j": "I", "w": "U", "ɰ": "I",
    "ㅂ": "M", "ㅃ": "M", "ㅍ": "M", "ㅁ": "M",     # 입술을 붙인다
    "ㅅ": "S", "ㅆ": "S", "ㅈ": "S", "ㅉ": "S", "ㅊ": "S",
    "ㄷ": "T", "ㄸ": "T", "ㅌ": "T", "ㄴ": "T", "ㄹ": "T",
    "ㄱ": "K", "ㄲ": "K", "ㅋ": "K", "ㅇ": "K",
    "ㅎ": "H",
    "": "X",                                        # 다문 입
}


@dataclass(frozen=True, slots=True)
class Phoneme:
    """음소 하나와 그 시간."""

    symbol: str
    kind: PhonemeKind
    start: float            # 음 시작으로부터 몇 초 뒤 (음수면 박자보다 앞)
    duration: float
    formants: tuple[float, float, float] = (500.0, 1500.0, 2500.0)
    aspiration: float = 0.0
    voiced: bool = True
    center_hz: float = 0.0  # 자음의 조음 위치 주파수

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def viseme(self) -> str:
        return VISEMES.get(self.symbol, "X")

    def __str__(self) -> str:
        return f"{self.symbol}({self.kind.value} {self.start * 1000:+.0f}~{self.end * 1000:.0f}ms)"


@dataclass(slots=True)
class SungSyllable:
    """부르는 음절 하나. 음 하나에 대응한다."""

    text: str
    phonemes: list[Phoneme] = field(default_factory=list)
    note_start: float = 0.0     # 실제 시각(초)
    note_duration: float = 0.0

    @property
    def onset_offset(self) -> float:
        """첫 소리가 박자보다 얼마나 앞서는가 (초). 항상 0 이하."""
        return min((p.start for p in self.phonemes), default=0.0)

    @property
    def vowel(self) -> Phoneme | None:
        """이 음절의 중심 모음. 음높이가 실리는 곳이다."""
        for phoneme in self.phonemes:
            if phoneme.kind is PhonemeKind.VOWEL:
                return phoneme
        return None

    def viseme_timeline(self) -> list[tuple[float, float, str]]:
        """입 모양이 언제 바뀌는지. (시작 초, 끝 초, 입 모양)"""
        return [
            (self.note_start + p.start, self.note_start + p.end, p.viseme)
            for p in self.phonemes
        ]


def _vowel_phonemes(medial: str) -> list[tuple[str, PhonemeKind, tuple[float, float, float]]]:
    """중성을 반모음 + 모음으로 편다."""
    if medial in DIPHTHONGS:
        glide, vowel = DIPHTHONGS[medial]
        return [
            (glide, PhonemeKind.VOWEL, GLIDE_FORMANTS[glide]),
            (vowel, PhonemeKind.VOWEL, VOWEL_FORMANTS[vowel]),
        ]
    if medial in VOWEL_FORMANTS:
        return [(medial, PhonemeKind.VOWEL, VOWEL_FORMANTS[medial])]
    raise KoreanError(f"모르는 중성입니다: {medial!r}")


def syllable_to_phonemes(
    syllable: Syllable, duration: float, tempo_scale: float = 1.0,
) -> list[Phoneme]:
    """한 음절을 음소와 시간으로 편다.

    duration 은 이 음이 이어지는 시간(초)이다.
    자음은 박자 앞에 놓는다. 사람이 노래할 때 그렇게 하기 때문이다.
    """
    if duration <= 0:
        raise KoreanError(f"음 길이는 0보다 커야 합니다: {duration}")

    result: list[Phoneme] = []

    # --- 초성 ---
    initial = syllable.initial
    onset_length = 0.0
    if initial and initial != "ㅇ":       # 초성 ㅇ 은 소리가 없다
        kind, center, base_length, voiced, aspiration = CONSONANTS[initial]
        # 음이 아주 짧으면 자음도 줄인다. 안 그러면 자음이 음을 다 먹는다.
        onset_length = min(base_length * tempo_scale, duration * 0.4)
        result.append(Phoneme(
            initial, kind, -onset_length, onset_length,
            formants=(center * 0.3, center, center * 1.6),
            aspiration=aspiration, voiced=voiced, center_hz=center,
        ))

    # --- 종성 ---
    final = FINAL_SOUND.get(syllable.final, syllable.final)
    coda_length = 0.0
    if final:
        kind, center, base_length, voiced, aspiration = CONSONANTS.get(
            final, (PhonemeKind.PLOSIVE, 1500.0, 0.05, False, 0.2)
        )
        # 받침은 음 끝에 붙는다. 울림소리(ㄴㅁㅇㄹ)는 길게, 막음소리는 짧게.
        coda_length = min(
            base_length * tempo_scale * (1.6 if voiced else 0.8),
            duration * 0.35,
        )

    # --- 중성 ---
    vowel_parts = _vowel_phonemes(syllable.medial)
    vowel_total = max(0.02, duration - coda_length)
    if len(vowel_parts) == 2:
        # 반모음은 짧게 스친다
        glide_length = min(0.06 * tempo_scale, vowel_total * 0.3)
        symbol, kind, formants = vowel_parts[0]
        result.append(Phoneme(symbol, kind, 0.0, glide_length, formants=formants))
        symbol, kind, formants = vowel_parts[1]
        result.append(Phoneme(symbol, kind, glide_length, vowel_total - glide_length,
                              formants=formants))
    else:
        symbol, kind, formants = vowel_parts[0]
        result.append(Phoneme(symbol, kind, 0.0, vowel_total, formants=formants))

    if final and coda_length > 0:
        kind, center, _, voiced, aspiration = CONSONANTS.get(
            final, (PhonemeKind.PLOSIVE, 1500.0, 0.05, False, 0.2)
        )
        result.append(Phoneme(
            final, kind, vowel_total, coda_length,
            formants=(center * 0.5, center, center * 2.0),
            aspiration=aspiration, voiced=voiced, center_hz=center,
        ))
    return result


def align_lyrics(
    text: str, note_times: Sequence[tuple[float, float]], tempo_scale: float = 1.0,
) -> list[SungSyllable]:
    """가사를 음에 붙인다.

    note_times 는 [(시작 초, 길이 초), ...] 다.
    음절 수와 음 개수가 다르면 알린다. 조용히 맞추면 가사가 밀려서
    나중에 어디서 어긋났는지 찾을 수 없게 된다.
    """
    pieces = split_syllables(text)
    if not pieces:
        return []
    if len(pieces) != len(note_times):
        raise KoreanError(
            f"가사 {len(pieces)}음절과 음 {len(note_times)}개가 맞지 않습니다.\n"
            f"  가사: {' '.join(pieces)}\n"
            f"  음절 수를 맞추거나, 한 음에 여러 음절을 붙일지 정해야 합니다."
        )

    # 소리 바뀜 규칙은 글자들을 이어 놓고 봐야 한다
    syllables: list[Syllable | None] = []
    hangul_indices: list[int] = []
    for index, piece in enumerate(pieces):
        if len(piece) == 1 and is_hangul_syllable(piece):
            syllables.append(decompose(piece))
            hangul_indices.append(index)
        else:
            syllables.append(None)

    hangul = [syllables[i] for i in hangul_indices]
    if hangul:
        changed = apply_sound_changes([s for s in hangul if s is not None])
        for position, index in enumerate(hangul_indices):
            syllables[index] = changed[position]

    result: list[SungSyllable] = []
    for piece, syllable, (start, duration) in zip(pieces, syllables, note_times):
        if syllable is None:
            # 한글이 아닌 것은 'ㅏ' 로 부른다. 없는 것보다 낫다.
            phonemes = [Phoneme("ㅏ", PhonemeKind.VOWEL, 0.0, duration,
                                formants=VOWEL_FORMANTS["ㅏ"])]
        else:
            phonemes = syllable_to_phonemes(syllable, duration, tempo_scale)
        result.append(SungSyllable(piece, phonemes, start, duration))
    return result


def viseme_timeline(syllables: Sequence[SungSyllable]) -> list[tuple[float, float, str]]:
    """전체 입 모양 변화. 33번·34번의 립싱크가 이걸 그대로 쓴다."""
    timeline: list[tuple[float, float, str]] = []
    for syllable in syllables:
        timeline.extend(syllable.viseme_timeline())
    timeline.sort(key=lambda item: item[0])
    # 소리 사이의 빈틈은 입을 다문 것으로 채운다
    filled: list[tuple[float, float, str]] = []
    previous_end = 0.0
    for start, end, viseme in timeline:
        if start > previous_end + 0.04:
            filled.append((previous_end, start, "X"))
        filled.append((start, end, viseme))
        previous_end = max(previous_end, end)
    return filled
