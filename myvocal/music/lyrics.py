"""
작사 — 5번.

가사는 멜로디에 붙는다. 음 하나에 한 음절이다. 그래서 가사를 '쓴다' 는 것은
사실 '이 줄은 7음절, 다음 줄은 9음절' 이라는 틀을 채우는 일이다.
틀을 무시하고 쓴 가사는 남는 음이 생기거나 모자라서 부를 수 없다.

    1. 보컬 트랙에서 줄을 찾는다      쉼표로 끊긴 곳이 줄 경계다
    2. 줄마다 음절 수를 센다
    3. Provider 에게 틀을 넘기고 가사를 받는다   (로컬 규칙 또는 Claude)
    4. 받은 가사가 틀과 조금 다르면 멜로디를 가사에 맞춘다
       음절이 모자라면 짧은 음 둘을 하나로, 남으면 긴 음을 둘로 나눈다
    5. 전부 한 번의 편집 명령으로 묶는다   Ctrl+Z 한 번에 되돌아간다

같은 선율을 쓰는 후렴은 같은 가사를 부른다. 그래야 후렴이다.
벌스는 선율이 같아도 가사가 다르다. 그래야 이야기가 나아간다.

가사는 음표에만 있다. 다른 곳에 따로 적어 두지 않는다. 두 곳에 있으면
한쪽만 고쳐졌을 때 어느 것이 맞는지 알 수 없게 된다.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

from ..core.history import AddNotes, CompositeCommand, EditNotes, RemoveNotes, Command
from ..core.notes import Note
from ..core.project import Project
from ..core.tracks import Track
from ..core.units import PPQ
from ..providers.base import (
    Capability, ProviderError, ProviderRegistry, ProviderRequest, ProviderResult,
)
from ..voice.korean import (
    count_syllables, decompose, is_hangul_syllable, split_syllables, syllables_with_word_ends,
)
from .lyric_lexicon import (
    COMMON_OPENERS, FUTURE_OPENERS, GENERIC_SHORTS, GENRE_DEFAULT_THEME, IMAGE_OPENERS,
    MANNER_OPENERS, STATE_ENDINGS, THEMES, Theme,
)
from .structure import Section


class LyricsError(ValueError):
    """작사 관련 오류."""


# 이만큼 비면 줄이 끝난 것으로 본다. 16분음표 하나보다 짧은 틈은
# 연주상의 떨어짐이지 숨 쉬는 자리가 아니다.
LINE_GAP_TICKS = PPQ // 4

# 한 줄이 이보다 길면 둘로 나눈다. 숨 한 번에 부르기 어렵다.
MAX_LINE_SYLLABLES = 16

# 구간 종류 -> 어휘의 역할 표시
SECTION_ROLE: dict[str, str] = {
    "verse": "v", "pre_chorus": "p", "build": "p", "chorus": "c", "post_chorus": "c",
    "drop": "c", "bridge": "b", "breakdown": "b", "interlude": "b",
    "intro": "v", "outro": "c", "instrumental": "v", "solo": "v",
}

# 선율이 같으면 가사도 같아야 하는 구간
REPEATING_KINDS = frozenset({"chorus", "post_chorus", "drop"})


# ==========================================================================
# 1. 멜로디에서 틀을 찾는다
# ==========================================================================

@dataclass(slots=True)
class LyricLine:
    """가사 한 줄이 들어갈 자리."""

    notes: list[Note]
    text: str = ""

    @property
    def syllables(self) -> int:
        return len(self.notes)


@dataclass(slots=True)
class SectionSlots:
    """한 구간의 줄들."""

    section: Section
    lines: list[LyricLine]
    same_as: "SectionSlots | None" = None   # 같은 가사를 부르는 앞 구간

    @property
    def key(self) -> str:
        return f"s{self.section.section_id}"

    @property
    def pattern(self) -> list[int]:
        return [line.syllables for line in self.lines]


def _split_long(notes: list[Note]) -> list[list[Note]]:
    """너무 긴 줄을 가장 긴 음 뒤에서 나눈다. 긴 음 뒤가 숨 쉬기 자연스럽다."""
    if len(notes) <= MAX_LINE_SYLLABLES:
        return [notes]
    middle = len(notes) // 2
    window = range(max(1, middle - 4), min(len(notes) - 1, middle + 4))
    cut = max(window, key=lambda i: (notes[i - 1].duration_ticks, -abs(i - middle)))
    return _split_long(notes[:cut]) + _split_long(notes[cut:])


def melody_lines(notes: Sequence[Note]) -> list[list[Note]]:
    """음들을 줄로 나눈다."""
    ordered = sorted(notes, key=lambda n: n.start_tick)
    lines: list[list[Note]] = []
    current: list[Note] = []
    for note in ordered:
        if current and note.start_tick - current[-1].end_tick >= LINE_GAP_TICKS:
            lines.append(current)
            current = []
        current.append(note)
    if current:
        lines.append(current)
    result: list[list[Note]] = []
    for line in lines:
        result.extend(_split_long(line))
    return result


def _relative_shape(project: Project, section: Section, lines: list[LyricLine]) -> tuple:
    start = project.section_range(section).start_tick
    return tuple(
        tuple((n.midi, n.start_tick - start, n.duration_ticks) for n in line.notes)
        for line in lines
    )


def collect_slots(project: Project, track: Track) -> list[SectionSlots]:
    """보컬 트랙에서 구간별 줄을 모은다."""
    if track.kind != "vocal":
        raise LyricsError(f"'{track.name}' 은 보컬 트랙이 아닙니다.")
    if not project.structure:
        raise LyricsError("곡 구조가 없어서 구간을 나눌 수 없습니다.")
    result: list[SectionSlots] = []
    shapes: dict[tuple, SectionSlots] = {}
    for section in project.structure:
        span = project.section_range(section)
        inside = [n for n in track.notes if span.start_tick <= n.start_tick < span.end_tick]
        if not inside:
            continue
        lines = [LyricLine(list(group)) for group in melody_lines(inside)]
        slots = SectionSlots(section, lines)
        if section.kind in REPEATING_KINDS:
            shape = (section.kind, _relative_shape(project, section, lines))
            if shape in shapes:
                slots.same_as = shapes[shape]
            else:
                shapes[shape] = slots
        result.append(slots)
    if not result:
        raise LyricsError(f"'{track.name}' 트랙에 음이 없습니다. 멜로디를 먼저 만들어 주세요.")
    return result


# ==========================================================================
# 2. Provider 에게 넘길 요청
# ==========================================================================

@dataclass(slots=True)
class LyricsBrief:
    """무엇에 대한 가사인가."""

    subject: str = ""
    title: str = ""
    mood: str = ""
    genre: str = ""
    extra: str = ""            # "마지막 후렴은 희망적으로" 같은 지시
    seed: int | None = None


def build_payload(project: Project, slots: Sequence[SectionSlots],
                  brief: LyricsBrief) -> dict:
    """Provider 가 받는 형식. 로컬이든 Claude 든 같은 것을 받는다."""
    sections = []
    for slot in slots:
        sections.append({
            "key": slot.key,
            "kind": slot.section.kind,
            "label": slot.section.label,
            "lines": slot.pattern,
            "same_as": slot.same_as.key if slot.same_as else None,
            "direction": slot.section.notes_for_ai,
        })
    return {
        "title": brief.title or project.meta.title,
        "subject": brief.subject or project.meta.description,
        "mood": brief.mood,
        "genre": brief.genre or project.genre.primary.name,
        "extra": brief.extra,
        "language": "ko",
        "seed": brief.seed,
        "sections": sections,
    }


def check_response(payload: dict, sections: dict, syllables: bool = True) -> list[str]:
    """받은 가사가 틀에 맞는지. 문제 목록을 돌려준다 (비면 정확히 맞음).

    syllables=False 면 구간이 빠졌거나 줄 수가 다른 것만 본다. 그건 멜로디에
    붙일 수 없는 문제이고, 음절 수 차이는 멜로디 쪽에서 맞출 수 있다.
    """
    problems: list[str] = []
    for item in payload["sections"]:
        if item["same_as"]:
            continue
        lines = sections.get(item["key"])
        if lines is None:
            problems.append(f"{item['label']}: 가사가 없습니다.")
            continue
        if len(lines) != len(item["lines"]):
            problems.append(
                f"{item['label']}: {len(item['lines'])}줄이어야 하는데 {len(lines)}줄입니다."
            )
            continue
        if not syllables:
            continue
        for index, (text, wanted) in enumerate(zip(lines, item["lines"]), 1):
            got = count_syllables(text)
            if got != wanted:
                problems.append(
                    f"{item['label']} {index}번째 줄 \"{text}\": "
                    f"{wanted}음절이어야 하는데 {got}음절입니다."
                )
    return problems


# ==========================================================================
# 3. 로컬 작사 (키 없이)
# ==========================================================================

def detect_theme(text: str, genre: str = "") -> tuple[Theme, str]:
    """설명에서 주제를 고른다. (주제, 이유) 를 돌려준다."""
    scores: dict[str, float] = {}
    found: dict[str, list[str]] = {}
    for name, theme in THEMES.items():
        for word, weight in theme.keywords.items():
            if word in text:
                scores[name] = scores.get(name, 0.0) + weight
                found.setdefault(name, []).append(word)
    if scores:
        best = max(scores, key=lambda n: scores[n])
        return THEMES[best], f"설명의 '{', '.join(found[best])}' 에서 정함"
    primary = genre.split("+")[0].strip()
    name = GENRE_DEFAULT_THEME.get(primary, "love")
    reason = (f"설명에 주제가 없어 장르({primary})에 맞춰 정함" if primary
              else "설명에 주제가 없어 기본값으로 정함")
    return THEMES[name], reason


def detect_images(text: str) -> list[str]:
    """설명에 나온 장면 낱말. 가사에 넣는다.

    낱말이 다른 낱말의 일부인 경우는 뺀다. '준비' 의 '비', '눈물' 의 '눈',
    '달려' 의 '달' 은 장면이 아니다. 앞은 낱말 경계, 뒤는 낱말 경계나 조사여야 한다.
    """
    particles = "이가은는을를에의과와도로만처"
    found: list[str] = []
    for word in IMAGE_OPENERS:
        position = text.find(word)
        while position != -1:
            before = text[position - 1] if position > 0 else ""
            after = text[position + len(word): position + len(word) + 1]
            before_ok = not (before and is_hangul_syllable(before))
            after_ok = not (after and is_hangul_syllable(after)) or after in particles
            if before_ok and after_ok:
                found.append(word)
                break
            position = text.find(word, position + 1)
    return found


def _is_past(text: str) -> bool:
    """과거형으로 끝나는가. '돌아왔어', '그리웠어' 처럼 끝 앞 글자 받침이 ㅆ."""
    word = text.split()[-1] if text.split() else ""
    return (len(word) >= 2 and is_hangul_syllable(word[-2])
            and decompose(word[-2]).final == "ㅆ")


def _last_vowel(text: str) -> str:
    for character in reversed(text):
        if is_hangul_syllable(character):
            return decompose(character).medial
    return ""


@dataclass(slots=True)
class _Piece:
    text: str
    kind: str          # "o" 여는 말, "l" 잇는 말, "e" 맺는 말, "h" 후렴 첫 줄, "s" 짧은 말
    role: str = ""
    length: int = field(init=False)

    def __post_init__(self) -> None:
        self.length = count_syllables(self.text)


class LocalLyricist:
    """규칙으로 가사를 쓴다."""

    PATTERNS: tuple[tuple[str, ...], ...] = (
        ("e",), ("h",), ("o", "e"), ("l", "e"), ("o", "l", "e"),
        ("l", "l", "e"), ("o", "l", "l", "e"), ("o", "h"), ("h", "e"),
    )

    def __init__(self, theme: Theme, images: Sequence[str] = (),
                 seed: int | None = None) -> None:
        self.theme = theme
        self.random = random.Random(seed)
        self.images = list(images)
        image_openers = [o for word in self.images for o in IMAGE_OPENERS[word]]
        self.image_openers = set(image_openers)
        openers = list(dict.fromkeys(list(theme.openers) + image_openers + list(COMMON_OPENERS)))
        self.pieces: dict[str, list[_Piece]] = {
            "o": [_Piece(t, "o") for t in openers],
            "l": [_Piece(t, "l", r) for t, r in theme.links],
            "e": [_Piece(t, "e", r) for t, r in theme.endings],
            "h": [_Piece(t, "h", "c") for t in theme.hooks],
        }
        self.by_length: dict[str, dict[int, list[_Piece]]] = {}
        for kind, pieces in self.pieces.items():
            table: dict[int, list[_Piece]] = {}
            for piece in pieces:
                table.setdefault(piece.length, []).append(piece)
            self.by_length[kind] = table
        self.used: dict[str, int] = {}
        self.images_used: set[str] = set()

    # ---- 한 줄 ----

    def _length_splits(self, pattern: tuple[str, ...], total: int) -> list[tuple[int, ...]]:
        """이 모양으로 total 음절을 만들 수 있는 길이 조합들."""
        results: list[tuple[int, ...]] = []

        def walk(index: int, remaining: int, chosen: tuple[int, ...]) -> None:
            if index == len(pattern):
                if remaining == 0:
                    results.append(chosen)
                return
            for length in self.by_length[pattern[index]]:
                if length <= remaining:
                    walk(index + 1, remaining - length, chosen + (length,))

        walk(0, total, ())
        return results

    def _score(self, pieces: Sequence[_Piece], role: str, rhyme: str,
               first_of_chorus: bool) -> float:
        score = 0.0
        for piece in pieces:
            score -= 6.0 * self.used.get(piece.text, 0)     # 이미 쓴 말은 피한다
            if piece.role and role in piece.role:
                score += 1.5
            if piece.text in self.image_openers and piece.text not in self.images_used:
                score += 2.5
        texts = [p.text for p in pieces]
        if len(set(texts)) < len(texts):
            score -= 10.0
        # "오늘 오늘도 기다려" 처럼 한 줄에 같은 낱말이 두 번 나오면 어색하다
        words = [w for p in pieces for w in p.text.split()]
        stems = [w[:2] for w in words if len(w) >= 2]
        if len(set(stems)) < len(stems):
            score -= 5.0
        # 어울리지 않는 짝
        ending = pieces[-1]
        for index, piece in enumerate(pieces[:-1]):
            if piece.kind != "o":
                continue
            if piece.text in FUTURE_OPENERS and _is_past(ending.text):
                score -= 4.0
            nxt = pieces[index + 1]
            if piece.text in MANNER_OPENERS and nxt.text in STATE_ENDINGS:
                score -= 4.0
        if rhyme and _last_vowel(pieces[-1].text) == rhyme:
            score += 1.0
        if first_of_chorus and pieces[-1].kind == "h":
            score += 4.0
        if first_of_chorus and any(p.kind == "h" for p in pieces):
            score += 2.0
        score -= 0.4 * (len(pieces) - 2) ** 2               # 조각이 너무 많으면 어수선하다
        return score + self.random.random() * 1.2

    def line(self, syllables: int, role: str = "v", rhyme: str = "",
             first_of_chorus: bool = False) -> str:
        """정확히 syllables 음절인 한 줄."""
        if syllables <= 0:
            raise LyricsError(f"음절 수는 1 이상이어야 합니다: {syllables}")
        candidates: list[list[_Piece]] = []
        for pattern in self.PATTERNS:
            if "h" in pattern and role != "c":
                continue
            for lengths in self._length_splits(pattern, syllables):
                pools = [self.by_length[k][n] for k, n in zip(pattern, lengths)]
                size = 1
                for pool in pools:
                    size *= len(pool)
                if size <= 60:
                    # 경우가 적으면 전부 본다. 뽑다 보면 안 쓴 말을 놓친다.
                    candidates.extend(list(combo) for combo in itertools.product(*pools))
                else:
                    for _ in range(20):
                        candidates.append([self.random.choice(pool) for pool in pools])
        if not candidates:
            return self._short(syllables)
        best = max(candidates, key=lambda c: self._score(c, role, rhyme, first_of_chorus))
        for piece in best:
            self.used[piece.text] = self.used.get(piece.text, 0) + 1
            if piece.text in self.image_openers:
                self.images_used.add(piece.text)
        return " ".join(p.text for p in best)

    def _short(self, syllables: int) -> str:
        """음이 한두 개뿐이거나 조각으로 못 채울 때. 짧은 말을 이어 붙인다."""
        pool = [s for s in list(self.theme.shorts) + list(GENERIC_SHORTS)]
        words: list[str] = []
        remaining = syllables
        guard = 0
        while remaining > 0 and guard < 50:
            guard += 1
            fitting = [w for w in pool if count_syllables(w) <= remaining]
            if not fitting:
                break
            word = self.random.choice(fitting)
            words.append(word)
            remaining -= count_syllables(word)
        text = " ".join(words)
        if count_syllables(text) != syllables:
            raise LyricsError(f"{syllables}음절 줄을 채우지 못했습니다.")
        return text

    # ---- 한 구간 ----

    def section(self, kind: str, pattern: Sequence[int]) -> list[str]:
        role = SECTION_ROLE.get(kind, "v")
        # 구간 안에서 줄 끝 모음을 맞춘다 (각운). 첫 줄 끝 모음을 따른다.
        rhyme = ""
        lines: list[str] = []
        for index, count in enumerate(pattern):
            text = self.line(count, role, rhyme, first_of_chorus=(role == "c" and index == 0))
            if not rhyme:
                rhyme = _last_vowel(text)
            lines.append(text)
        return lines

    def write(self, payload: dict) -> dict[str, list[str]]:
        written: dict[str, list[str]] = {}
        for item in payload["sections"]:
            if item["same_as"]:
                continue
            written[item["key"]] = self.section(item["kind"], item["lines"])
        return written


def local_lyrics(payload: dict) -> dict:
    """로컬 작사. LYRICS Provider 가 부르는 함수."""
    source = " ".join(str(payload.get(k, "")) for k in ("subject", "title", "mood", "extra"))
    theme, reason = detect_theme(source, str(payload.get("genre", "")))
    images = detect_images(source)
    lyricist = LocalLyricist(theme, images, payload.get("seed"))
    sections = lyricist.write(payload)
    note = f"주제: {theme.display_name} ({reason})"
    if images:
        note += f" / 넣은 장면: {', '.join(images)}"
    return {"sections": sections, "theme": theme.display_name, "note": note}


# ==========================================================================
# 4. 멜로디를 가사에 맞춘다
# ==========================================================================

def fit_notes(notes: Sequence[Note], syllables: Sequence[str],
              word_ends: Sequence[bool] | None = None) -> list[Note]:
    """음 개수를 음절 수에 맞추고 가사를 붙인다.

    음절이 모자라면: 가장 짧은 이웃 두 음을 하나로 합친다 (앞 음 높이, 두 길이 합).
    음절이 남으면:   가장 긴 음을 반으로 나눈다 (같은 높이).
    마지막 음은 되도록 건드리지 않는다. 줄 끝의 긴 음은 숨 쉬기 전 여운이다.
    """
    if not notes:
        raise LyricsError("음이 없는 줄에는 가사를 붙일 수 없습니다.")
    if not syllables:
        raise LyricsError("빈 가사를 붙일 수 없습니다.")
    working = [n.copy() for n in sorted(notes, key=lambda n: n.start_tick)]
    changed = False

    while len(working) > len(syllables):
        pairs = range(len(working) - 1)
        preferred = [i for i in pairs if i + 1 < len(working) - 1] or list(pairs)
        index = min(preferred, key=lambda i: working[i].duration_ticks
                    + working[i + 1].duration_ticks)
        first, second = working[index], working[index + 1]
        merged = replace(first, duration_ticks=second.end_tick - first.start_tick, note_id=0)
        working[index:index + 2] = [merged]
        changed = True

    minimum = PPQ // 4       # 16분음표보다 짧게는 나누지 않는다. 발음할 시간이 없다.
    while len(working) < len(syllables):
        splittable = [i for i, n in enumerate(working) if n.duration_ticks >= 2 * minimum]
        if not splittable:
            raise LyricsError(
                f"음 {len(notes)}개에 {len(syllables)}음절을 넣을 수 없습니다. "
                f"음이 너무 짧아 더 나눌 수 없습니다."
            )
        inner = [i for i in splittable if i < len(working) - 1] or splittable
        index = max(inner, key=lambda i: working[i].duration_ticks)
        note = working[index]
        half = max(minimum, (note.duration_ticks // 2) // minimum * minimum)
        left = replace(note, duration_ticks=half, note_id=0)
        right = replace(note, start_tick=note.start_tick + half,
                        duration_ticks=note.duration_ticks - half, note_id=0)
        working[index:index + 1] = [left, right]
        changed = True

    if changed:
        # 새로 만든 음은 note_id 를 트랙이 다시 매긴다
        working = [replace(n, note_id=0) for n in working]
    ends = list(word_ends) if word_ends is not None else [False] * len(syllables)
    if len(ends) != len(syllables):
        raise LyricsError("띄어쓰기 표시 개수가 음절 수와 다릅니다.")
    return [n.with_lyric(s, word_end=e) for n, s, e in zip(working, syllables, ends)]


# ==========================================================================
# 5. 전체 흐름
# ==========================================================================

@dataclass(slots=True)
class LyricsDraft:
    """받은 가사. 아직 음표에 붙이기 전이다."""

    slots: list[SectionSlots]
    sections: dict[str, list[str]]
    provider: str = ""
    fell_back: bool = False
    note: str = ""
    mismatches: list[str] = field(default_factory=list)
    cost_note: str = ""

    def is_stale(self, track: Track) -> bool:
        """가사를 받는 동안 멜로디가 바뀌었는가. 바뀌었으면 붙이면 안 된다."""
        present = {id(n) for n in track.notes}
        return any(id(n) not in present
                   for slot in self.slots for line in slot.lines for n in line.notes)

    def lines_for(self, slot: SectionSlots) -> list[str]:
        source = slot.same_as or slot
        return self.sections.get(source.key, [])

    def text(self) -> str:
        """사람이 읽는 가사 전체."""
        blocks: list[str] = []
        for slot in self.slots:
            lines = self.lines_for(slot)
            blocks.append(f"[{slot.section.label}]\n" + "\n".join(lines))
        return "\n\n".join(blocks)


def prepare_lyrics(project: Project, track: Track,
                   brief: LyricsBrief | None = None) -> tuple[list[SectionSlots], dict]:
    """멜로디에서 틀을 읽는다. 화면 실뜨기에서 부른다 (프로젝트를 읽으므로)."""
    slots = collect_slots(project, track)
    return slots, build_payload(project, slots, brief or LyricsBrief())


def request_lyrics(slots: list[SectionSlots], payload: dict, registry: ProviderRegistry,
                   on_progress: Callable[[str], None] | None = None,
                   timeout_seconds: float = 300.0) -> LyricsDraft:
    """가사를 받아 온다. 프로젝트를 건드리지 않으므로 딴 실뜨기에서 불러도 된다.

    바깥 서비스는 수십 초가 걸릴 수 있다. 그동안 화면이 멈추면 안 된다.
    """
    request = ProviderRequest(Capability.LYRICS, payload, timeout_seconds=timeout_seconds,
                              max_retries=1)
    result: ProviderResult = registry.run(request, on_progress)
    sections = result.require("sections")
    if not isinstance(sections, dict):
        raise ProviderError("가사 응답 형식이 잘못됐습니다 (sections 가 사전이 아님).")
    return LyricsDraft(
        slots=slots,
        sections={k: [str(line) for line in v] for k, v in sections.items()},
        provider=result.provider,
        fell_back=result.fell_back,
        note=str(result.data.get("note", "")),
        mismatches=check_response(payload, sections),
        cost_note=result.cost_note,
    )


def write_lyrics(project: Project, track: Track, registry: ProviderRegistry,
                 brief: LyricsBrief | None = None,
                 on_progress: Callable[[str], None] | None = None,
                 timeout_seconds: float = 300.0) -> LyricsDraft:
    """틀을 읽고 가사를 받아 온다. 프로젝트는 아직 바꾸지 않는다."""
    slots, payload = prepare_lyrics(project, track, brief)
    return request_lyrics(slots, payload, registry, on_progress, timeout_seconds)


def lyrics_command(track: Track, draft: LyricsDraft,
                   label: str = "AI 작사") -> CompositeCommand:
    """가사를 음표에 붙이는 편집 명령. 한 번에 되돌릴 수 있다."""
    commands: list[Command] = []
    edits: list[tuple[Note, Note]] = []
    for slot in draft.slots:
        texts = draft.lines_for(slot)
        if len(texts) != len(slot.lines):
            raise LyricsError(
                f"{slot.section.label}: 줄 수가 맞지 않습니다 "
                f"(멜로디 {len(slot.lines)}줄, 가사 {len(texts)}줄)."
            )
        for line, text in zip(slot.lines, texts):
            pairs = syllables_with_word_ends(text)
            if not pairs:
                raise LyricsError(f"{slot.section.label}: 빈 줄이 있습니다.")
            fitted = fit_notes(line.notes, [p[0] for p in pairs], [p[1] for p in pairs])
            if len(fitted) == len(line.notes):
                ordered = sorted(line.notes, key=lambda n: n.start_tick)
                for old, new in zip(ordered, fitted):
                    new = replace(new, note_id=old.note_id)
                    if (new.lyric, new.word_end) != (old.lyric, old.word_end):
                        edits.append((old, new))
            else:
                commands.append(RemoveNotes(track, line.notes))
                commands.append(AddNotes(track, fitted))
    if edits:
        commands.insert(0, EditNotes(track, edits))
    if not commands:
        raise LyricsError("바뀔 것이 없습니다. 이미 같은 가사가 붙어 있습니다.")
    return CompositeCommand(commands, label)


def vocal_track(project: Project) -> Track:
    """가사를 붙일 보컬 트랙. 음이 있는 첫 번째 것."""
    for track in project.tracks:
        if track.kind == "vocal" and len(track.notes):
            return track
    raise LyricsError("음이 있는 보컬 트랙이 없습니다. 멜로디를 먼저 만들어 주세요.")
