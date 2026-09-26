"""
MY MUSIC STYLE (8번) 와 Reference (9번).

    8번  내가 만들었거나 학습 권한이 있는 곡 여러 개 → 공통 성향을 모아 '내 스타일' 장르를
         만든다. 이후 "내 음악 스타일로 새로운 곡 만들어줘" 가 된다.

    9번  학습 권한이 없는 참고 곡 하나 → 템포, 구조, 에너지 흐름, 악기 구성, 코드 성격,
         편곡 밀도, 믹스 성향만 이번 곡 하나에 참고로 쓴다. 저장해 두고 되풀이해 쓰는
         스타일로 만들지 않는다. 원곡의 코드 진행과 멜로디는 가져오지 않는다.

둘 다 결과는 작곡기가 이미 쓰는 '장르 설정값(GenreProfile)' 이다. 그래서 작곡기,
편곡기, 믹스는 새로 배울 것이 없다. 장르 22개에 하나가 더 생긴 것처럼 동작한다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..music.genre import GENRES, GenreBlend, GenreError, GenreProfile
from ..music.harmony import COMMON_PROGRESSIONS
from ..music.instruments import available_instruments
from ..music.structure import SECTION_NAMES, STRUCTURE_TEMPLATES, Section, SongStructure
from .analysis import MusicAnalysis, chord_roots_and_kinds


class StyleError(ValueError):
    """스타일 관련 오류."""


MAJOR_DEGREES = {0: "I", 2: "II", 4: "III", 5: "IV", 7: "V", 9: "VI", 11: "VII"}
MINOR_DEGREES = {0: "I", 2: "II", 3: "III", 5: "IV", 7: "V", 8: "VI", 10: "VII"}


def template_numerals(chords: Sequence[str], tonic: int, minor: bool,
                      extensions: bool = True) -> list[str | None]:
    """코드 이름들 -> 작곡기의 도수 표기. 조성 밖 코드는 None.

    extensions=True 면 7화음 표기를 살린다 ('ii7', 'V7', 'Imaj7', 'ii7b5').
    재즈·R&B 스타일의 색은 7화음에 있다. 떼어 내면 그 스타일로 만든 곡에서
    7화음이 사라진다 (처음에 그렇게 해서 재즈 스타일 곡의 7화음이 21% 로 나왔다).
    """
    table = MINOR_DEGREES if minor else MAJOR_DEGREES
    result: list[str | None] = []
    for symbol in chords:
        parsed = chord_roots_and_kinds(symbol)
        if parsed is None:
            result.append(None)
            continue
        root, kind = parsed
        numeral = table.get((root - tonic) % 12)
        if numeral is None:
            result.append(None)
            continue
        is_minor = kind.startswith("m") and not kind.startswith("maj")
        if kind.startswith("dim") or kind == "m7b5":
            base = numeral.lower() + ("7b5" if extensions and kind == "m7b5" else "°")
            if kind == "m7b5" and extensions:
                base = numeral.lower() + "7b5"
            result.append(base)
            continue
        base = numeral.lower() if is_minor else numeral
        if extensions and kind in ("7", "m7"):
            base += "7"
        elif extensions and kind == "maj7":
            base += "maj7"
        result.append(base)
    return result


def base_numeral(numeral: str | None) -> str | None:
    """'ii7' -> 'ii', 'Imaj7' -> 'I', 'ii7b5' -> 'ii°'. 7화음 장식을 뗀 뼈대."""
    if numeral is None:
        return None
    if numeral.endswith("7b5"):
        return numeral[:-3] + "°"
    for suffix in ("maj7", "7"):
        if numeral.endswith(suffix):
            return numeral[: -len(suffix)]
    return numeral


def frequent_progressions(analyses: Sequence[MusicAnalysis], count: int = 4) -> dict[str, list[list[str]]]:
    """자주 나온 4코드 진행 (조성 안의 코드만). 장조/단조 따로."""
    counts = {"major": {}, "minor": {}}
    for analysis in analyses:
        if analysis.key_tonic_pc is None or not analysis.chords:
            continue
        mode = "minor" if analysis.key_minor else "major"
        numerals = template_numerals([c for _, c in analysis.chords], analysis.key_tonic_pc,
                                     bool(analysis.key_minor))
        for i in range(len(numerals) - 3):
            window = numerals[i:i + 4]
            if None in window or len(set(window)) < 3:
                continue
            key = tuple(window)
            counts[mode][key] = counts[mode].get(key, 0) + 1
    result = {}
    for mode, table in counts.items():
        ranked = sorted(table.items(), key=lambda p: -p[1])[:count]
        if ranked:
            result[mode] = [list(k) for k, _ in ranked]
    return result


def structure_template(analyses: Sequence[MusicAnalysis]) -> list[tuple[str, int]]:
    """구간 순서와 길이. 구간이 가장 잘 나뉜 곡 하나를 기준으로, 길이는 짝수 마디로."""
    candidates = [a for a in analyses if len(a.sections) >= 3]
    if not candidates:
        return []
    chosen = max(candidates, key=lambda a: len({s.kind for s in a.sections}) * 10 + len(a.sections))
    template = []
    for section in chosen.sections:
        kind = section.kind if section.kind in SECTION_NAMES else "verse"
        bars = max(2, int(round(section.bars / 2)) * 2)
        template.append((kind, bars))
    return template


def _mean(values) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


@dataclass(slots=True)
class MusicStyleProfile:
    """내 음악 스타일."""

    name: str
    style_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    sources: list[str] = field(default_factory=list)
    values: dict = field(default_factory=dict)          # GenreProfile 에 들어갈 값
    progressions: dict[str, list[list[str]]] = field(default_factory=dict)
    structure: list[tuple[str, int]] = field(default_factory=list)
    base_genres: dict[str, float] = field(default_factory=dict)   # 모자란 값을 채운 장르
    measured: list[str] = field(default_factory=list)

    @property
    def genre_name(self) -> str:
        return f"mystyle{self.style_id}"

    def to_genre_profile(self) -> GenreProfile:
        base = GenreBlend(self.base_genres or {"pop": 1.0}).resolve()
        values = dict(self.values)
        instruments = values.pop("instrument_weights", None) or dict(base.instrument_weights)
        fields = {name: getattr(base, name) for name in (
            "bpm_low", "bpm_high", "bpm_typical", "minor_tendency", "seventh_chords",
            "tension_notes", "borrowed_chords", "harmonic_rhythm", "swing", "syncopation",
            "drum_density", "arrangement_density", "brightness", "compression", "target_lufs",
            "stereo_width", "low_end", "vocal_vibrato", "vocal_breathiness", "vocal_power",
            "harmony_layers")}
        fields.update(values)
        low, typical, high = fields.pop("bpm_low"), fields.pop("bpm_typical"), fields.pop("bpm_high")
        return GenreProfile(self.genre_name, f"내 스타일: {self.name}", low, high, typical,
                            instrument_weights=instruments, **fields)

    def describe(self) -> str:
        profile = self.to_genre_profile()
        lines = [f"MY MUSIC STYLE — {self.name}", f"곡 {len(self.sources)}개에서: {', '.join(self.sources)}",
                 profile.describe()]
        if self.progressions:
            for mode, items in self.progressions.items():
                lines.append(f"  자주 쓴 진행 ({'단조' if mode == 'minor' else '장조'}): "
                             + " / ".join("-".join(p) for p in items))
        if self.structure:
            lines.append("  곡 구조: " + " → ".join(f"{SECTION_NAMES[k]}{b}" for k, b in self.structure))
        base = ", ".join(f"{n} {w:.0%}" for n, w in self.base_genres.items())
        lines.append(f"  잰 항목: {', '.join(self.measured)}")
        lines.append(f"  못 잰 값은 가장 닮은 장르({base})의 값으로 채웠습니다.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"name": self.name, "style_id": self.style_id, "sources": self.sources,
                "values": self.values, "progressions": self.progressions,
                "structure": [list(s) for s in self.structure], "base_genres": self.base_genres,
                "measured": self.measured, "format": 1}

    @classmethod
    def from_dict(cls, data: dict) -> "MusicStyleProfile":
        return cls(name=data["name"], style_id=data["style_id"], sources=list(data.get("sources", [])),
                   values=dict(data.get("values", {})), progressions=data.get("progressions", {}),
                   structure=[tuple(s) for s in data.get("structure", [])],
                   base_genres=dict(data.get("base_genres", {})), measured=list(data.get("measured", [])))


def _measured_values(analyses: Sequence[MusicAnalysis]) -> tuple[dict, list[str]]:
    """여러 곡의 분석을 장르 설정값으로. (값, 잰 항목 이름들)"""
    values: dict = {}
    measured: list[str] = []
    bpms = [a.bpm for a in analyses if a.bpm]
    if bpms:
        values["bpm_typical"] = float(np.median(bpms))
        values["bpm_low"] = float(max(40.0, min(bpms) - 6.0))
        values["bpm_high"] = float(min(240.0, max(bpms) + 6.0))
        measured.append("BPM")
    modes = [a.key_minor for a in analyses if a.key_minor is not None]
    if modes:
        values["minor_tendency"] = float(np.mean(modes))
        measured.append("장조/단조 성향")
    for key, attribute, label in (
        ("seventh_chords", "seventh_ratio", "7화음"),
        ("borrowed_chords", "borrowed_ratio", "조성 밖 화음"),
        ("swing", "swing", "스윙"),
        ("syncopation", "syncopation", "엇박"),
        ("drum_density", "drum_density", "드럼 밀도"),
        ("arrangement_density", "arrangement_density", "편곡 밀도"),
    ):
        mean = _mean([getattr(a, attribute) for a in analyses])
        if mean is not None:
            values[key] = float(np.clip(mean, 0.0, 1.0 if key != "swing" else 0.66))
            measured.append(label)
    rhythm = _mean([a.harmonic_rhythm for a in analyses])
    if rhythm is not None:
        # 작곡기의 화성 리듬은 마디당 코드 수. 흔한 값(0.5/1/2)으로 맞춘다.
        values["harmonic_rhythm"] = min((0.5, 1.0, 2.0), key=lambda v: abs(v - rhythm))
        measured.append("화성 리듬")

    # 악기: 악보에서 잰 것만 (오디오는 악기를 모른다)
    known = set(available_instruments())
    totals: dict[str, float] = {}
    counted = 0
    for analysis in analyses:
        shares = {n: w for n, w in analysis.instruments.items() if n in known}
        if shares:
            counted += 1
            for name, weight in shares.items():
                totals[name] = totals.get(name, 0.0) + weight
    if totals:
        top = max(totals.values())
        values["instrument_weights"] = {n: round(w / top, 3) for n, w in totals.items()}
        measured.append("악기 구성")

    # 믹스: 오디오에서 잰 것 중, 장르 설정값과의 관계를 확인한 것만 옮긴다.
    # 작곡기로 만든 6곡(장르 설정을 아는 곡)을 렌더링해서 재 보았다:
    #   압축 ↔ 봉우리-평균 차이   상관 -0.91  → 직선으로 옮김 (잔차 0.08)
    #   저역 ↔ 250Hz 아래 비율    상관 +0.77  → 직선으로 옮김 (잔차 0.18)
    #   밝기 ↔ 소리 무게중심      상관 -0.51 (방향이 반대) → 근거가 없어 옮기지 않음
    #   좌우 넓이                 설정값이 1.0/1.1/1.3 뿐이라 관계를 세울 수 없어 옮기지 않음
    #   라우드니스               렌더러가 스트리밍 기준 -14 LUFS 로 맞추므로 옮기지 않음
    mixes = [a.mix for a in analyses if a.mix]
    if mixes:
        crest = _mean([m.get("봉우리-평균 차이 dB") for m in mixes])
        if crest is not None:
            values["compression"] = float(np.clip(-0.248 * crest + 4.286, 0.0, 1.0))
        low = _mean([m.get("저역 비율 % (<250Hz)") for m in mixes])
        if low is not None:
            values["low_end"] = float(np.clip(0.0133 * low - 0.188, -1.0, 1.0))
        measured.append("믹스 성향 (압축·저역)")
    return values, measured


def _base_genres(analyses: Sequence[MusicAnalysis]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for analysis in analyses:
        for name, weight in analysis.genre_guess:
            if name in GENRES and not name.startswith(("mystyle", "reference")):
                totals[name] = totals.get(name, 0.0) + weight
    if not totals:
        return {"pop": 1.0}
    top = sorted(totals.items(), key=lambda p: -p[1])[:3]
    total = sum(w for _, w in top)
    return {n: round(w / total, 3) for n, w in top}


def build_style(name: str, analyses: Sequence[MusicAnalysis]) -> MusicStyleProfile:
    if not name.strip():
        raise StyleError("스타일 이름을 적어 주세요.")
    if not analyses:
        raise StyleError("분석한 곡이 없습니다.")
    values, measured = _measured_values(analyses)
    style = MusicStyleProfile(
        name=name.strip(), sources=[a.source for a in analyses], values=values,
        progressions=frequent_progressions(analyses), structure=structure_template(analyses),
        base_genres=_base_genres(analyses), measured=measured,
    )
    if style.progressions:
        style.measured.append("코드 진행")
    if style.structure:
        style.measured.append("곡 구조")
    style.to_genre_profile()          # 값이 장르 설정으로 말이 되는지 여기서 확인
    return style


def register_style(style: MusicStyleProfile) -> str:
    """작곡기가 쓸 수 있게 등록한다. 장르 이름을 돌려준다."""
    profile = style.to_genre_profile()
    GENRES[profile.name] = profile
    if style.progressions:
        COMMON_PROGRESSIONS[profile.name] = {mode: [list(p) for p in items]
                                             for mode, items in style.progressions.items()}
    if style.structure:
        STRUCTURE_TEMPLATES[profile.name] = [tuple(s) for s in style.structure]
    return profile.name


# ==========================================================================
# Reference (9번)
# ==========================================================================

@dataclass(slots=True)
class ReferenceHints:
    """참고 곡에서 이번 곡 하나에 가져올 것."""

    source: str
    genre_name: str                     # 임시로 등록한 장르 이름
    bpm: float | None
    structure: SongStructure | None
    notes: list[str]                    # 무엇을 참고했는지

    def describe(self) -> str:
        return f"Reference: {self.source}\n" + "\n".join(f"  · {n}" for n in self.notes)


def reference_hints(analysis: MusicAnalysis) -> ReferenceHints:
    """참고 곡 하나를 이번 곡의 설정으로. 코드 진행과 멜로디는 가져오지 않는다."""
    values, measured = _measured_values([analysis])
    # 원곡 복제를 막기 위해 코드 진행 틀은 넣지 않는다 (코드 '성격' 수치만 쓴다)
    base = _base_genres([analysis])
    style = MusicStyleProfile(name=f"참고: {analysis.source}", values=values, base_genres=base,
                              measured=measured)
    profile = style.to_genre_profile()
    name = f"reference{style.style_id}"
    profile = GenreProfile(**{**{f: getattr(profile, f) for f in profile.__dataclass_fields__},
                              "name": name, "display_name": f"참고: {analysis.source}"})
    GENRES[name] = profile
    notes = [f"{m}" for m in measured]
    structure = None
    if len(analysis.sections) >= 3:
        structure = SongStructure()
        bar = 1
        for section in analysis.sections:
            kind = section.kind if section.kind in SECTION_NAMES else "verse"
            bars = max(2, int(round(section.bars / 2)) * 2)
            structure.add(Section(kind, bar, bars, energy=float(np.clip(section.energy, 0.05, 1.0))))
            bar += bars
        structure.renumber_labels()
        notes.append("곡 구조와 구간별 에너지")
    return ReferenceHints(analysis.source, name, analysis.bpm, structure,
                          [f"참고한 것: {', '.join(notes)}",
                           "가져오지 않은 것: 코드 진행, 멜로디, 가사 (원곡을 복제하지 않습니다)"])
