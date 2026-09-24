"""
목소리 학습 연습 — 6번.

    "고음의 'ㅣ' 발음 데이터가 부족합니다.
     다음 문장을 G4 부근에서 불러주세요."

이것을 하려면 두 가지가 필요하다.

    1. 지금까지 모은 자료가 모음 × 음높이 구역마다 얼마나 되는지 (커버리지)
    2. 비어 있는 칸을 채울 연습을 만드는 것

연습은 '가이드 음을 먼저 들려주고, 조용할 때 따라 부르는' 방식이다.
노래하는 동안 가이드가 울리면 스피커로 들을 때 마이크에 가이드가 섞여서,
사람 목소리 대신 가이드 음을 분석하게 된다. 들려주고 → 부르고 를 번갈아 하면
헤드폰이 없어도 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .analysis import GuideNote, NoteMeasure, hz_to_midi, midi_to_hz, note_name, nucleus_vowel
from .korean import decompose, is_hangul_syllable

VOWELS: tuple[str, ...] = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ", "ㅔ", "ㅐ")
VOWEL_SYLLABLE: dict[str, str] = {
    "ㅏ": "아", "ㅓ": "어", "ㅗ": "오", "ㅜ": "우", "ㅡ": "으", "ㅣ": "이", "ㅔ": "에", "ㅐ": "애",
}
ZONES: tuple[str, ...] = ("low", "mid", "high")
ZONE_NAMES: dict[str, str] = {"low": "저음", "mid": "중음", "high": "고음"}

# 칸 하나에 필요한 자료. 음 두 개 이상, 소리 낸 시간 2초 이상.
NEEDED_SECONDS = 2.0
NEEDED_NOTES = 2

# 모음마다 그 모음이 많이 들어간 문장. 한 음절에 한 음씩 부른다.
# 그 모음 비율이 높은 것을 앞에 둔다 (첫 문장은 57~100%).
VOWEL_SENTENCES: dict[str, tuple[str, ...]] = {
    "ㅏ": ("바다 가자 나랑 같이", "하나 둘 다 가 봐", "파란 하늘 날아가"),
    "ㅓ": ("저 너머 먼 거리", "허공 너머 떠나", "어서 너를 만나러"),
    "ㅗ": ("모두 오고 보고", "오솔길 도는 소리", "고요한 노을 속"),
    "ㅜ": ("두루 부는 루루", "우두커니 누운 우리", "구름 위 무지개"),
    "ㅡ": ("스르르 드는 그늘", "크는 그 느낌", "그늘 아래 흐르는 물"),
    "ㅣ": ("이 길 지나 미리", "비가 내리니 시리다", "기다린 이 밤 끝에 피어난 빛"),
    "ㅔ": ("제게 네 게 데려가", "메아리 세 번 네게", "베개 위 세레나데"),
    "ㅐ": ("태양 새 새벽 해", "내 애 대답해 봐", "새해엔 해맑게 노래해"),
}

# 발음 연습용: 초성 19개가 골고루 들어가게 만든 문장
CONSONANT_SENTENCES: tuple[str, ...] = (
    "가나다라 마바사 아자차",       # ㄱㄴㄷㄹ ㅁㅂㅅ ㅇㅈㅊ
    "카타파하 까따빠 싸짜",          # ㅋㅌㅍㅎ ㄲㄸㅃ ㅆㅉ
)
ALL_INITIALS: tuple[str, ...] = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ",
    "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)


# ==========================================================================
# 연습
# ==========================================================================

@dataclass(slots=True)
class Exercise:
    """연습 하나. 들려줄 음과 부를 구간."""

    title: str
    instruction: str
    notes: list[GuideNote]            # 부를 구간 (녹음 시각 기준)
    cue_seconds: float = 0.6          # 각 음 앞에서 가이드 음을 들려주는 시간
    kind: str = "sentence"

    @property
    def duration(self) -> float:
        last = self.notes[-1]
        return last.start + last.duration + 0.6

    @property
    def text(self) -> str:
        return "".join(n.syllable for n in self.notes)


def _call_and_response(syllables: Sequence[str], midis: Sequence[int], sing_seconds: float,
                       cue_seconds: float = 0.6, lead_in: float = 0.8) -> list[GuideNote]:
    """들려주고(cue) → 부르고(sing) 를 번갈아 하는 시간표."""
    notes = []
    cursor = lead_in
    for syllable, midi in zip(syllables, midis):
        cursor += cue_seconds + 0.15          # 가이드 음 + 숨 쉴 틈
        notes.append(GuideNote(cursor, sing_seconds, int(midi), syllable))
        # 부른 뒤 다음 가이드 음까지 0.5초. 장치 지연(보통 0.3초 이하)만큼 녹음이
        # 늦게 들어와도 부른 구간이 다음 가이드 음과 겹치지 않게.
        cursor += sing_seconds + 0.5
    return notes


def sustain_exercise(vowel: str, midis: Sequence[int], sing_seconds: float = 1.4) -> Exercise:
    syllable = VOWEL_SYLLABLE[vowel]
    names = ", ".join(note_name(midi_to_hz(m)) for m in midis)
    return Exercise(
        title=f"'{syllable}' 길게 내기",
        instruction=f"가이드 음을 듣고, 그 음으로 '{syllable}' 를 {sing_seconds:.1f}초 동안 "
                    f"길게 불러 주세요. ({names})",
        notes=_call_and_response([syllable] * len(midis), midis, sing_seconds),
        kind="sustain",
    )


def sentence_exercise(sentence: str, center_midi: int, sing_seconds: float = 0.55,
                      zone_label: str = "") -> Exercise:
    """문장을 한 음절에 한 음씩. 가운데 음 둘레 ±2반음에서 오르내린다."""
    syllables = [c for c in sentence if is_hangul_syllable(c)]
    shape = (0, 2, 0, -2, 0, 2, 4, 2, 0, -1, 0)
    midis = [center_midi + shape[i % len(shape)] for i in range(len(syllables))]
    where = f"{zone_label} " if zone_label else ""
    return Exercise(
        title=f"문장: {sentence}",
        instruction=f"다음 문장을 {where}{note_name(midi_to_hz(center_midi))} 부근에서 "
                    f"불러주세요.\n“{sentence}”\n한 음절마다 가이드 음을 먼저 들려줍니다.",
        notes=_call_and_response(syllables, midis, sing_seconds),
    )


def range_exercise(start_midi: int, stop_midi: int, vowel: str = "ㅏ") -> Exercise:
    """음역 재기. 한 음씩 올라간다. 안 나오는 음은 부르지 않고 넘어가면 된다."""
    step = 1 if stop_midi >= start_midi else -1
    midis = list(range(start_midi, stop_midi + step, step))
    syllable = VOWEL_SYLLABLE[vowel]
    cue = 0.5
    return Exercise(
        title="음역 재기",
        instruction=f"가이드를 따라 '{syllable}' 로 반음씩 올라갑니다. 편하게 낼 수 없는 음은 "
                    f"부르지 말고 쉬세요. 무리하지 않는 것이 중요합니다.",
        # 시간표와 가이드 소리의 길이가 같아야 한다. 다르면 가이드가 앞 음 부르는
        # 구간에 겹쳐서 녹음에 섞인다.
        notes=_call_and_response([syllable] * len(midis), midis, 0.8, cue_seconds=cue),
        cue_seconds=cue,
        kind="range",
    )


def starter_exercises(low_midi: int, high_midi: int) -> list[Exercise]:
    """처음 녹음할 때의 기본 연습들. 음역을 모르면 성종 기본값에서 시작한다."""
    center = (low_midi + high_midi) // 2
    third = max(2, (high_midi - low_midi) // 3)
    return [
        range_exercise(low_midi, high_midi),
        sustain_exercise("ㅏ", [center - 2, center, center + 2]),
        sustain_exercise("ㅣ", [center - 2, center, center + 2]),
        sentence_exercise(VOWEL_SENTENCES["ㅏ"][0], center),
        sentence_exercise(CONSONANT_SENTENCES[0], center - third // 2),
        sentence_exercise(CONSONANT_SENTENCES[1], center + third // 2),
    ]


# ==========================================================================
# 커버리지
# ==========================================================================

@dataclass(slots=True)
class Cell:
    seconds: float = 0.0
    notes: int = 0

    @property
    def enough(self) -> bool:
        return self.seconds >= NEEDED_SECONDS and self.notes >= NEEDED_NOTES


@dataclass(slots=True)
class Coverage:
    """모음 × 음높이 구역마다 모은 자료."""

    low_midi: int
    high_midi: int
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    initials: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for vowel in VOWELS:
            for zone in ZONES:
                self.cells.setdefault((vowel, zone), Cell())

    def zone_of(self, midi: float) -> str:
        span = max(1.0, self.high_midi - self.low_midi)
        position = (midi - self.low_midi) / span
        return "low" if position < 1 / 3 else ("mid" if position < 2 / 3 else "high")

    def zone_center(self, zone: str) -> int:
        span = self.high_midi - self.low_midi
        index = ZONES.index(zone)
        return int(round(self.low_midi + span * (index + 0.5) / 3))

    def add(self, notes: Sequence[NoteMeasure], syllables: Sequence[str] = ()) -> None:
        for note in notes:
            if not note.vowel or note.vowel not in VOWELS:
                continue
            cell = self.cells[(note.vowel, self.zone_of(hz_to_midi(note.pitch_hz)))]
            cell.seconds += note.voiced_seconds
            cell.notes += 1
        for syllable in syllables:
            if syllable and is_hangul_syllable(syllable[0]):
                initial = decompose(syllable[0]).initial
                self.initials[initial] = self.initials.get(initial, 0) + 1

    @property
    def filled_ratio(self) -> float:
        return sum(1 for c in self.cells.values() if c.enough) / len(self.cells)

    def missing(self) -> list[tuple[str, str]]:
        """비어 있는 칸. 중음 → 고음 → 저음, 모음 표 순서로 (자주 쓰는 순서)."""
        order = {"mid": 0, "high": 1, "low": 2}
        empty = [key for key, cell in self.cells.items() if not cell.enough]
        return sorted(empty, key=lambda k: (order[k[1]], VOWELS.index(k[0])))

    def missing_initials(self) -> list[str]:
        return [c for c in ALL_INITIALS if self.initials.get(c, 0) == 0]

    def table(self) -> str:
        lines = ["      " + "  ".join(f"{VOWEL_SYLLABLE[v]:>4}" for v in VOWELS)]
        for zone in ZONES:
            row = []
            for vowel in VOWELS:
                cell = self.cells[(vowel, zone)]
                row.append(f"{cell.seconds:4.1f}" + ("✓" if cell.enough else " "))
            lines.append(f"{ZONE_NAMES[zone]:>4}  " + " ".join(row))
        return "\n".join(lines)


@dataclass(slots=True)
class Suggestion:
    message: str
    exercise: Exercise


def next_suggestions(coverage: Coverage, count: int = 3,
                     used_sentences: Sequence[str] = ()) -> list[Suggestion]:
    """부족한 자료를 채울 다음 연습들."""
    result: list[Suggestion] = []
    used = set(used_sentences)
    for vowel, zone in coverage.missing():
        if len(result) >= count:
            break
        center = coverage.zone_center(zone)
        choices = [s for s in VOWEL_SENTENCES[vowel] if s not in used] or list(VOWEL_SENTENCES[vowel])
        sentence = choices[0]
        used.add(sentence)
        cell = coverage.cells[(vowel, zone)]
        have = f" (지금 {cell.seconds:.1f}초)" if cell.seconds > 0 else ""
        message = (f"{ZONE_NAMES[zone]}의 '{vowel}' 발음 "
                   f"데이터가 부족합니다{have}.\n다음 문장을 {note_name(midi_to_hz(center))} "
                   f"부근에서 불러주세요.\n“{sentence}”")
        result.append(Suggestion(message, sentence_exercise(sentence, center)))
    if len(result) < count:
        missing = coverage.missing_initials()
        if missing:
            sentence = next((s for s in CONSONANT_SENTENCES
                             if any(c in missing for c in _initials_of(s))), CONSONANT_SENTENCES[0])
            center = coverage.zone_center("mid")
            message = (f"자음 {', '.join(missing[:6])}{' 등' if len(missing) > 6 else ''} 의 "
                       f"발음 데이터가 없습니다.\n다음 문장을 {note_name(midi_to_hz(center))} "
                       f"부근에서 불러주세요.\n“{sentence}”")
            result.append(Suggestion(message, sentence_exercise(sentence, center)))
    return result


def _initials_of(sentence: str) -> list[str]:
    return [decompose(c).initial for c in sentence if is_hangul_syllable(c)]
