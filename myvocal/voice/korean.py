"""
한국어 발음 분해 — 33번 Lyrics -> Phoneme -> Timing 의 첫 단계.

한글은 음소를 조합해 한 글자를 만든다. 이 성질 덕분에 발음을 기계적으로
정확히 뽑아낼 수 있다. 영어처럼 철자와 발음이 따로 노는 언어와 다르다.

    가 = ㄱ + ㅏ
    강 = ㄱ + ㅏ + ㅇ
    닭 = ㄷ + ㅏ + ㄺ

다만 글자 그대로 읽으면 안 된다. 한국어에는 소리가 바뀌는 규칙이 있다.

    같이 -> 가치      (구개음화)
    독립 -> 동닙      (비음화)
    좋아 -> 조아      (ㅎ 탈락)
    닭이 -> 달기      (연음)

노래는 글자가 아니라 소리를 낸다. 이 규칙을 안 넣으면 '같이' 를 '갇이' 라고
부르게 되고, 듣는 사람은 바로 알아챈다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3

# 초성 19개
INITIALS: tuple[str, ...] = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

# 중성 21개
MEDIALS: tuple[str, ...] = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)

# 종성 28개 (첫 번째는 받침 없음)
FINALS: tuple[str, ...] = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)


class KoreanError(ValueError):
    """한국어 처리 오류."""


@dataclass(frozen=True, slots=True)
class Syllable:
    """한 글자를 이룬 소리들."""

    initial: str        # 초성
    medial: str         # 중성
    final: str = ""     # 종성 (받침). 없으면 빈 문자열.
    original: str = ""  # 원래 글자

    @property
    def has_final(self) -> bool:
        return bool(self.final)

    def compose(self) -> str:
        """다시 한 글자로 합친다."""
        try:
            initial_index = INITIALS.index(self.initial)
            medial_index = MEDIALS.index(self.medial)
            final_index = FINALS.index(self.final)
        except ValueError as error:
            raise KoreanError(f"합칠 수 없는 소리입니다: {self} ({error})") from error
        return chr(HANGUL_BASE + (initial_index * 21 + medial_index) * 28 + final_index)

    def __str__(self) -> str:
        return f"{self.initial}+{self.medial}" + (f"+{self.final}" if self.final else "")


def is_hangul_syllable(character: str) -> bool:
    return len(character) == 1 and HANGUL_BASE <= ord(character) <= HANGUL_LAST


def decompose(character: str) -> Syllable:
    """한 글자를 초성·중성·종성으로 나눈다."""
    if not is_hangul_syllable(character):
        raise KoreanError(f"한글 글자가 아닙니다: {character!r}")
    offset = ord(character) - HANGUL_BASE
    final_index = offset % 28
    medial_index = (offset // 28) % 21
    initial_index = offset // 28 // 21
    return Syllable(
        INITIALS[initial_index], MEDIALS[medial_index], FINALS[final_index], character
    )


def compose(initial: str, medial: str, final: str = "") -> str:
    return Syllable(initial, medial, final).compose()


# ==========================================================================
# 소리 바뀜 규칙
# ==========================================================================

# 받침이 실제로 나는 소리 (음절 끝소리 규칙).
# 받침에는 27가지가 올 수 있지만 실제로 나는 소리는 7가지뿐이다.
FINAL_SOUND: dict[str, str] = {
    "": "", "ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄳ": "ㄱ", "ㄺ": "ㄱ",
    "ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㄶ": "ㄴ",
    "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ",
    "ㅌ": "ㄷ", "ㅎ": "ㄷ",
    "ㄹ": "ㄹ", "ㄼ": "ㄹ", "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㅀ": "ㄹ",
    "ㅁ": "ㅁ", "ㄻ": "ㅁ",
    "ㅂ": "ㅂ", "ㅍ": "ㅂ", "ㅄ": "ㅂ", "ㄿ": "ㅂ",
    "ㅇ": "ㅇ",
}

# 겹받침을 뒤 글자로 넘길 때 어느 쪽이 넘어가는가
DOUBLE_FINAL_SPLIT: dict[str, tuple[str, str]] = {
    "ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"), "ㅄ": ("ㅂ", "ㅅ"),
}

# 비음화: 받침 + 다음 초성 -> 바뀐 받침
NASALIZATION: dict[tuple[str, str], str] = {}
for _final in ("ㄱ", "ㄲ", "ㅋ", "ㄳ", "ㄺ"):
    for _initial in ("ㄴ", "ㅁ"):
        NASALIZATION[(_final, _initial)] = "ㅇ"
for _final in ("ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㅎ"):
    for _initial in ("ㄴ", "ㅁ"):
        NASALIZATION[(_final, _initial)] = "ㄴ"
for _final in ("ㅂ", "ㅍ", "ㄼ", "ㄿ", "ㅄ"):
    for _initial in ("ㄴ", "ㅁ"):
        NASALIZATION[(_final, _initial)] = "ㅁ"

# ㄹ 이 앞 소리를 따라 바뀌는 경우 (유음화·비음화)
LIQUID_RULES: dict[tuple[str, str], tuple[str, str]] = {
    ("ㄴ", "ㄹ"): ("ㄹ", "ㄹ"),        # 신라 -> 실라
    ("ㄹ", "ㄴ"): ("ㄹ", "ㄹ"),        # 칼날 -> 칼랄
    ("ㅁ", "ㄹ"): ("ㅁ", "ㄴ"),        # 담력 -> 담녁
    ("ㅇ", "ㄹ"): ("ㅇ", "ㄴ"),        # 종로 -> 종노
    ("ㄱ", "ㄹ"): ("ㅇ", "ㄴ"),        # 국립 -> 궁닙
    ("ㅂ", "ㄹ"): ("ㅁ", "ㄴ"),        # 협력 -> 혐녁
}

# 된소리되기: 받침 + 다음 초성 -> 된소리
TENSIFICATION: dict[str, str] = {
    "ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ",
}
TENSE_TRIGGERS: frozenset[str] = frozenset({"ㄱ", "ㄲ", "ㅋ", "ㄷ", "ㅅ", "ㅆ",
                                            "ㅈ", "ㅊ", "ㅌ", "ㅂ", "ㅍ"})

# 거센소리되기: ㅎ 과 만나면 거센소리가 된다
ASPIRATION: dict[str, str] = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}

# 구개음화: ㄷ/ㅌ 받침 + '이' -> ㅈ/ㅊ
PALATALIZATION: dict[str, str] = {"ㄷ": "ㅈ", "ㅌ": "ㅊ"}


def apply_sound_changes(syllables: Sequence[Syllable]) -> list[Syllable]:
    """소리 바뀜 규칙을 적용해 '실제로 나는 소리' 로 바꾼다.

    규칙 순서가 중요하다. 실제 발음이 일어나는 순서를 따른다.
      1. ㅎ 관련 (거센소리되기, ㅎ 탈락)
      2. 구개음화
      3. 연음 (받침이 다음 글자 초성으로)
      4. 끝소리 규칙 (남은 받침을 7가지로)
      5. 비음화 / 유음화
      6. 된소리되기
    """
    result = [
        Syllable(s.initial, s.medial, s.final, s.original) for s in syllables
    ]

    for index in range(len(result)):
        current = result[index]
        following = result[index + 1] if index + 1 < len(result) else None
        if following is None:
            continue

        # --- 1. ㅎ 관련 ---
        if current.final in ("ㅎ", "ㄶ", "ㅀ"):
            base = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[current.final]
            if following.initial in ASPIRATION:
                # 좋다 -> 조타
                result[index] = Syllable(current.initial, current.medial, base,
                                         current.original)
                result[index + 1] = Syllable(
                    ASPIRATION[following.initial], following.medial, following.final,
                    following.original,
                )
                continue
            if following.initial == "ㅇ":
                # 좋아 -> 조아
                result[index] = Syllable(current.initial, current.medial, base,
                                         current.original)
                continue
        if current.final in ASPIRATION and following.initial == "ㅎ":
            # 축하 -> 추카
            result[index] = Syllable(current.initial, current.medial, "",
                                     current.original)
            result[index + 1] = Syllable(
                ASPIRATION[current.final], following.medial, following.final,
                following.original,
            )
            continue

    for index in range(len(result) - 1):
        current = result[index]
        following = result[index + 1]

        # --- 2. 구개음화 (같이 -> 가치) ---
        if (current.final in PALATALIZATION and following.initial == "ㅇ"
                and following.medial in ("ㅣ", "ㅕ", "ㅑ", "ㅛ", "ㅠ")):
            result[index] = Syllable(current.initial, current.medial, "",
                                     current.original)
            result[index + 1] = Syllable(
                PALATALIZATION[current.final], following.medial, following.final,
                following.original,
            )
            continue

        # --- 3. 연음 (받침이 다음 글자로 넘어간다) ---
        if current.final and following.initial == "ㅇ":
            if current.final in DOUBLE_FINAL_SPLIT:
                stay, move = DOUBLE_FINAL_SPLIT[current.final]
                result[index] = Syllable(current.initial, current.medial, stay,
                                         current.original)
                result[index + 1] = Syllable(move, following.medial, following.final,
                                             following.original)
            else:
                result[index] = Syllable(current.initial, current.medial, "",
                                         current.original)
                result[index + 1] = Syllable(
                    current.final, following.medial, following.final, following.original
                )

    # --- 4. 끝소리 규칙 ---
    for index, syllable in enumerate(result):
        sound = FINAL_SOUND.get(syllable.final, syllable.final)
        if sound != syllable.final:
            result[index] = Syllable(syllable.initial, syllable.medial, sound,
                                     syllable.original)

    # --- 5. 비음화 / 유음화 ---
    for index in range(len(result) - 1):
        current = result[index]
        following = result[index + 1]
        pair = (current.final, following.initial)
        if pair in LIQUID_RULES:
            new_final, new_initial = LIQUID_RULES[pair]
            result[index] = Syllable(current.initial, current.medial, new_final,
                                     current.original)
            result[index + 1] = Syllable(new_initial, following.medial,
                                         following.final, following.original)
        elif pair in NASALIZATION:
            result[index] = Syllable(current.initial, current.medial,
                                     NASALIZATION[pair], current.original)

    # --- 6. 된소리되기 ---
    for index in range(len(result) - 1):
        current = result[index]
        following = result[index + 1]
        if (current.final in ("ㄱ", "ㄷ", "ㅂ")
                and following.initial in TENSIFICATION):
            result[index + 1] = Syllable(
                TENSIFICATION[following.initial], following.medial,
                following.final, following.original,
            )

    return result


def pronounce(text: str) -> str:
    """글자 그대로가 아니라 실제로 나는 소리로 바꾼다.

        같이 -> 가치,  독립 -> 동닙,  좋아 -> 조아
    """
    parts: list[str] = []
    buffer: list[Syllable] = []

    def flush() -> None:
        if buffer:
            for syllable in apply_sound_changes(buffer):
                parts.append(syllable.compose())
            buffer.clear()

    for character in text:
        if is_hangul_syllable(character):
            buffer.append(decompose(character))
        else:
            flush()
            parts.append(character)
    flush()
    return "".join(parts)


def split_syllables(text: str) -> list[str]:
    """노래에 쓸 음절 단위로 나눈다.

    한글은 한 글자가 한 음절이다. 그래서 음 하나에 글자 하나를 붙이면 된다.
    영어 숫자 같은 것이 섞이면 그것도 하나로 센다.
    """
    result: list[str] = []
    buffer = ""
    for character in text:
        if is_hangul_syllable(character):
            if buffer:
                result.append(buffer)
                buffer = ""
            result.append(character)
        elif character.isspace():
            if buffer:
                result.append(buffer)
                buffer = ""
        elif character in ".,!?~…·\"'()":
            if buffer:
                result.append(buffer)
                buffer = ""
        else:
            buffer += character
    if buffer:
        result.append(buffer)
    return result


def count_syllables(text: str) -> int:
    """음절 수. 작사할 때 멜로디의 음 개수와 맞춰야 한다."""
    return len(split_syllables(text))
