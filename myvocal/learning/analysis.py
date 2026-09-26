"""
곡 분석 결과의 공통 모양, 그리고 악보(MIDI · 프로젝트) 분석.

8번이 요구하는 분석 항목:

    BPM / Key / Chord / Melody / 곡 구조 / 장르 / 리듬 / 악기 / 편곡 / 가사 / 보컬 표현 / Mix

악보(MIDI, .mvp)에는 음이 그대로 적혀 있으므로 대부분을 정확히 잴 수 있다.
Mix(소리의 크기, 공간감)는 악보에 없다. 오디오 분석(audio_analysis.py)이 잰다.
잴 수 없는 항목은 비워 두고 '왜 못 쟀는지' 를 unavailable 에 적는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core.project import Project
from ..core.units import PPQ
from ..music.theory import CHORD_INTERVALS, Chord, Key, Pitch, detect_key


class MusicAnalysisError(ValueError):
    """곡 분석 오류."""


# 코드 판정에 쓰는 종류. 텐션 코드까지 전부 후보로 넣으면 음 하나 차이로
# 이상한 이름이 붙는다. 대중음악에서 자주 쓰는 것만 쓴다.
CHORD_KINDS: tuple[str, ...] = ("", "m", "dim", "aug", "sus4", "sus2", "7", "maj7", "m7",
                                "m7b5", "dim7", "5")
SEVENTH_KINDS = {"7", "maj7", "m7", "m7b5", "dim7"}


@dataclass(slots=True)
class DetectedSection:
    start_bar: int              # 1부터
    bars: int
    label: str                  # 'A', 'B', ... 같은 글자면 같은 내용이 되풀이된 것
    kind: str                   # 'intro' / 'verse' / 'chorus' / 'bridge' / 'outro' 추정
    energy: float               # 0~1
    start_seconds: float = 0.0  # 곡 처음부터 몇 초


@dataclass(slots=True)
class MusicAnalysis:
    """곡 하나의 분석 결과."""

    source: str
    kind: str                                   # 'midi' / 'project' / 'audio'
    duration_seconds: float = 0.0
    bpm: float | None = None
    bpm_confidence: float = 0.0
    bpm_alternatives: list[float] = field(default_factory=list)   # 두 배/절반 후보 (오디오)
    beats_per_bar: int = 4
    key: str | None = None                      # 'A minor' 처럼
    key_tonic_pc: int | None = None
    key_minor: bool | None = None
    key_confidence: float = 0.0
    chords: list[tuple[float, str]] = field(default_factory=list)   # (몇 번째 박부터, 코드)
    seventh_ratio: float | None = None
    borrowed_ratio: float | None = None
    harmonic_rhythm: float | None = None        # 마디당 코드 바뀜
    progressions: list[tuple[str, int]] = field(default_factory=list)  # (도수 4개, 횟수)
    melody: list[tuple[float, float, int]] = field(default_factory=list)  # (박, 길이, 음)
    melody_range: tuple[int, int] | None = None
    melody_leap_ratio: float | None = None
    sections: list[DetectedSection] = field(default_factory=list)
    energy_curve: list[float] = field(default_factory=list)   # 마디마다 0~1
    swing: float | None = None                  # 0 = 정박, 0.66 = 완전 스윙
    syncopation: float | None = None            # 엇박 음 비율
    drum_density: float | None = None           # 드럼 음 / 16분음표 칸
    instruments: dict[str, float] = field(default_factory=dict)    # 악기 -> 음 비중
    arrangement_density: float | None = None    # 동시에 울리는 트랙 수 / 전체 (0~1)
    lyrics: str = ""
    vocal_expression: dict[str, float] = field(default_factory=dict)
    mix: dict[str, float] = field(default_factory=dict)
    genre_guess: list[tuple[str, float]] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["sections"] = [vars_section(s) for s in self.sections]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MusicAnalysis":
        values = dict(data)
        values["sections"] = [DetectedSection(**s) for s in data.get("sections", [])]
        values["chords"] = [tuple(c) for c in data.get("chords", [])]
        values["progressions"] = [tuple(p) for p in data.get("progressions", [])]
        values["melody"] = [tuple(m) for m in data.get("melody", [])]
        if data.get("melody_range"):
            values["melody_range"] = tuple(data["melody_range"])
        values["genre_guess"] = [tuple(g) for g in data.get("genre_guess", [])]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in values.items() if k in known})

    def summary(self) -> str:
        lines = [f"{self.source} ({self.kind})"]
        if self.duration_seconds:
            lines.append(f"길이: {int(self.duration_seconds // 60)}:{self.duration_seconds % 60:04.1f}")
        if self.bpm:
            extra = (f" (빠르기는 두 배/절반으로 헷갈릴 수 있습니다: "
                     f"{' 또는 '.join(f'{b:.1f}' for b in self.bpm_alternatives)})"
                     if self.bpm_alternatives else "")
            lines.append(f"BPM: {self.bpm:.1f}{extra}")
        if self.key:
            lines.append(f"조성: {self.key} (확신도 {self.key_confidence:.2f})")
        if self.chords:
            shown = " ".join(c for _, c in self.chords[:16])
            lines.append(f"코드 (앞부분): {shown}")
        if self.seventh_ratio is not None:
            lines.append(f"7화음 비율 {self.seventh_ratio:.0%}, 조성 밖 화음 {self.borrowed_ratio:.0%}, "
                         f"마디당 코드 {self.harmonic_rhythm:.2f}개")
        if self.progressions:
            lines.append("자주 쓴 진행: " + ", ".join(f"{p} ({n}번)" for p, n in self.progressions[:3]))
        if self.melody_range:
            lines.append(f"멜로디 음역: {Pitch.from_midi(self.melody_range[0])}~"
                         f"{Pitch.from_midi(self.melody_range[1])}, 도약 비율 {self.melody_leap_ratio:.0%}")
        if self.sections:
            text = " / ".join(f"{s.kind}({s.label}) {s.start_bar}~{s.start_bar + s.bars - 1}마디"
                              for s in self.sections)
            lines.append(f"곡 구조: {text}")
        if self.syncopation is not None:
            lines.append(f"리듬: 엇박 {self.syncopation:.0%}, 스윙 {self.swing:.2f}, "
                         f"드럼 밀도 {self.drum_density:.0%}" if self.drum_density is not None
                         else f"리듬: 엇박 {self.syncopation:.0%}, 스윙 {self.swing:.2f}")
        if self.instruments:
            top = sorted(self.instruments.items(), key=lambda p: -p[1])[:6]
            lines.append("악기: " + ", ".join(f"{n} {w:.0%}" for n, w in top))
        if self.arrangement_density is not None:
            lines.append(f"편곡 밀도: {self.arrangement_density:.0%}")
        if self.genre_guess:
            lines.append("장르 추정: " + ", ".join(f"{n} {w:.0%}" for n, w in self.genre_guess[:3]))
        if self.lyrics:
            lines.append(f"가사 (앞부분): {self.lyrics[:60]}")
        if self.mix:
            lines.append("믹스: " + ", ".join(f"{k} {v:.1f}" for k, v in self.mix.items()))
        for item, reason in self.unavailable.items():
            lines.append(f"[못 잰 것] {item}: {reason}")
        return "\n".join(lines)


def vars_section(section: DetectedSection) -> dict:
    return {"start_bar": section.start_bar, "bars": section.bars, "label": section.label,
            "kind": section.kind, "energy": section.energy, "start_seconds": section.start_seconds}


# ==========================================================================
# 코드
# ==========================================================================

def best_chord(weights: np.ndarray, bass_pc: int | None = None) -> tuple[str, float]:
    """12개 음이름의 무게로 가장 그럴듯한 코드. (이름, 점수)

    점수 = 코드 구성음에 든 무게 - 구성음 밖 무게의 절반 + 근음이 베이스면 가점.
    구성음이 많은 코드일수록 우연히 맞기 쉬우므로 음 하나당 조금씩 깎는다.
    """
    total = float(weights.sum())
    if total <= 0:
        return "N", 0.0
    normalized = weights / total
    best_name, best_score = "N", -1e9
    for root in range(12):
        for kind in CHORD_KINDS:
            tones = {(root + i) % 12 for i in CHORD_INTERVALS[kind]}
            inside = sum(normalized[t] for t in tones)
            outside = 1.0 - inside
            missing = sum(1 for t in tones if normalized[t] < 0.02)
            score = inside - 0.5 * outside - 0.12 * missing - 0.02 * len(tones)
            if bass_pc is not None and bass_pc == root:
                score += 0.08
            if score > best_score:
                best_score = score
                best_name = Chord(Pitch.from_midi(60 + root, prefer_flat=root in (1, 3, 8, 10)),
                                  kind).symbol if kind != "5" else \
                    f"{Pitch.from_midi(60 + root, prefer_flat=root in (1, 3, 8, 10)).name}5"
    return best_name, best_score


def chord_roots_and_kinds(symbol: str) -> tuple[int, str] | None:
    """코드 이름 -> (근음 음이름 번호, 종류)."""
    if symbol == "N" or not symbol:
        return None
    letters = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    root = letters.get(symbol[0])
    if root is None:
        return None
    rest = symbol[1:]
    while rest[:1] in ("#", "b"):
        root += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return root % 12, rest


def roman_of(symbol: str, key_tonic: int, minor: bool) -> str:
    parsed = chord_roots_and_kinds(symbol)
    if parsed is None:
        return "N"
    root, kind = parsed
    degree_semitones = (root - key_tonic) % 12
    names = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V",
             8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
    numeral = names[degree_semitones]
    if kind.startswith("m") and not kind.startswith("maj"):
        numeral = numeral.lower() if not numeral.startswith("b") else "b" + numeral[1:].lower()
    elif kind.startswith("dim"):
        numeral = numeral.lower() + "°"
    return numeral


def diatonic_ratio(chords: Sequence[str], tonic: int, minor: bool) -> float:
    """조성 안의 음만으로 된 코드의 비율 (근음 기준 대략)."""
    scale = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    scale_pcs = {(tonic + s) % 12 for s in scale}
    counted = inside = 0
    for symbol in chords:
        parsed = chord_roots_and_kinds(symbol)
        if parsed is None:
            continue
        root, kind = parsed
        base = kind if kind in CHORD_INTERVALS else ("5" if kind == "5" else "")
        tones = {(root + i) % 12 for i in CHORD_INTERVALS.get(base, (0, 4, 7))}
        counted += 1
        if tones <= scale_pcs or (minor and tones <= scale_pcs | {(tonic + 11) % 12}):
            inside += 1
    return inside / counted if counted else 1.0


# ==========================================================================
# 구조
# ==========================================================================

def _novelty(ssm: np.ndarray, width: int) -> np.ndarray:
    """Foote 의 체커보드 커널로 '여기서 달라지는 정도' 를 마디마다 잰다.

    앞 width 마디끼리 비슷하고, 뒤 width 마디끼리 비슷하고, 앞뒤는 다를수록 크다.
    """
    n = ssm.shape[0]
    kernel = np.ones((2 * width, 2 * width))
    kernel[:width, width:] = -1
    kernel[width:, :width] = -1
    padded = np.pad(ssm, width, mode="edge")
    result = np.zeros(n)
    for i in range(n):
        block = padded[i:i + 2 * width, i:i + 2 * width]
        result[i] = float((block * kernel).sum())
    result -= result.min()
    return result / (result.max() or 1.0)


def detect_sections(features: np.ndarray, energy: np.ndarray,
                    min_bars: int = 4) -> list[DetectedSection]:
    """마디마다의 특징으로 곡을 구간으로 나누고 이름을 붙인다.

    1. 마디끼리의 닮음 행렬을 만든다 (코드 색채, 멜로디, 어떤 악기가 울리는가)
    2. 체커보드 커널로 '달라지는 지점' 의 세기를 구하고, 봉우리를 경계로 본다.
       구간은 최소 min_bars 마디. 대중음악은 구간이 대개 짝수 마디라서,
       봉우리가 홀수 자리이고 옆 짝수 자리도 거의 같은 세기면 짝수 쪽을 쓴다.
    3. 서로 닮은 구간끼리 같은 글자 (A, B, ...) 를 붙인다
    4. 되풀이되면서 에너지가 가장 큰 글자를 후렴, 첫/끝의 조용한 구간을 인트로/아웃트로로 본다
    """
    bars = features.shape[0]
    if bars == 0:
        return []
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    unit = features / norms
    ssm = unit @ unit.T
    novelty = _novelty(ssm, width=min(4, max(1, bars // 4)))
    # 에너지가 크게 바뀌는 곳도 경계 후보다 (같은 코드로 악기만 늘어나는 빌드업 등)
    energy_change = np.abs(np.diff(np.concatenate([[energy[0]], energy])))
    score = novelty + 0.8 * energy_change / (energy_change.max() or 1.0)

    threshold = float(np.mean(score) + 0.4 * np.std(score))
    candidates = [i for i in range(1, bars)
                  if score[i] >= threshold and score[i] >= score[i - 1]
                  and (i + 1 >= bars or score[i] >= score[i + 1])]
    # 센 것부터 고르되, 이미 고른 경계와 min_bars 보다 가까우면 버린다
    boundaries: list[int] = []
    for index in sorted(candidates, key=lambda i: -score[i]):
        if index % 2 == 1:
            neighbours = [j for j in (index - 1, index + 1) if 0 < j < bars]
            best_even = max(neighbours, key=lambda j: score[j]) if neighbours else index
            if score[best_even] >= 0.85 * score[index]:
                index = best_even
        if index < min_bars or bars - index < min_bars:
            continue
        if all(abs(index - b) >= min_bars for b in boundaries):
            boundaries.append(index)
    edges = [0] + sorted(boundaries) + [bars]

    segments = []
    for a, b in zip(edges, edges[1:]):
        segments.append((a, b, float(np.mean(energy[a:b]))))

    # 닮은 구간끼리 같은 글자: 두 구간을 겹쳐 놓고 (짧은 쪽 길이만큼) 마디끼리의 닮음 평균
    def similarity(first, second) -> float:
        """짧은 구간을 긴 구간 위에서 밀어 가며 가장 잘 맞는 자리의 닮음.

        프리코러스와 후렴이 한 구간으로 묶였을 때, 뒤에 후렴만 나온 구간도
        같은 글자로 알아보려면 처음끼리가 아니라 맞는 자리를 찾아야 한다.
        """
        (a1, b1, _), (a2, b2, _) = first, second
        if b1 - a1 < b2 - a2:
            (a1, b1), (a2, b2) = (a2, b2), (a1, b1)
        length = b2 - a2
        best = -1.0
        for shift in range(0, (b1 - a1) - length + 1, 2 if (b1 - a1 - length) > 1 else 1):
            value = float(np.mean([ssm[a1 + shift + k, a2 + k] for k in range(length)]))
            best = max(best, value)
        return best

    labels: list[int] = []
    for index, segment in enumerate(segments):
        label = None
        for previous in range(index):
            if similarity(segments[previous], segment) > 0.9:
                label = labels[previous]
                break
        labels.append(label if label is not None else (max(labels, default=-1) + 1))

    energies = [s[2] for s in segments]
    counts = {label: labels.count(label) for label in set(labels)}
    repeated = [l for l in counts if counts[l] >= 2]
    chorus_label = None
    if repeated:
        # 후렴(드롭) = 되풀이되는 구간 중 에너지가 크고 긴 것. 악보의 '에너지' 는 음의
        # 촘촘함이라 빌드업처럼 음만 많은 구간이 더 크게 나올 수 있다. 길이도 본다
        # (EDM 드롭은 16마디로 빌드업의 두 배다).
        longest = max(b - a for a, b, _ in segments)

        def chorus_score(label: int) -> float:
            members = [i for i, x in enumerate(labels) if x == label]
            mean_energy = float(np.mean([energies[i] for i in members]))
            mean_length = float(np.mean([segments[i][1] - segments[i][0] for i in members]))
            return mean_energy + 0.15 * mean_length / longest

        chorus_label = max(repeated, key=chorus_score)
    top = max(energies) if energies else 1.0
    result = []
    for index, ((a, b, segment_energy), label) in enumerate(zip(segments, labels)):
        relative = segment_energy / top if top > 0 else 0.0
        if label == chorus_label:
            kind = "chorus"
        elif index == 0 and relative < 0.7 and len(segments) > 2:
            kind = "intro"
        elif index == len(segments) - 1 and relative < 0.7 and len(segments) > 2:
            kind = "outro"
        elif counts[label] >= 2:
            kind = "verse"
        else:
            kind = "bridge"
        result.append(DetectedSection(a + 1, b - a, chr(ord("A") + min(label, 25)), kind,
                                      round(relative, 3)))
    return result


# ==========================================================================
# 악보 분석
# ==========================================================================

def _genre_guess(analysis: MusicAnalysis) -> list[tuple[str, float]]:
    """측정값이 장르 설정값과 얼마나 가까운가로 장르를 추정한다.

    장르를 판정하는 정답은 없다. 여기서는 '이 곡의 성질이 우리 장르 설정 중
    어느 것과 가장 닮았는가' 를 거리로 잰다. 그래서 추정이라고 적는다.
    """
    from ..music.genre import GENRES

    scores = []
    for name, profile in GENRES.items():
        if name.startswith("mystyle"):
            continue
        distance = 0.0
        weight = 0.0
        if analysis.bpm:
            if profile.bpm_low <= analysis.bpm <= profile.bpm_high:
                distance += 0.0
            else:
                gap = min(abs(analysis.bpm - profile.bpm_low), abs(analysis.bpm - profile.bpm_high))
                distance += min(1.0, gap / 30.0) * 2.0
            weight += 2.0
        for value, target, w in (
            (analysis.seventh_ratio, profile.seventh_chords, 1.5),
            (analysis.swing, profile.swing, 1.0),
            (analysis.syncopation, profile.syncopation, 1.0),
            (analysis.drum_density, profile.drum_density, 1.0),
            (analysis.arrangement_density, profile.arrangement_density, 0.5),
            (None if analysis.key_minor is None else float(analysis.key_minor),
             profile.minor_tendency, 0.5),
        ):
            if value is not None:
                distance += abs(value - target) * w
                weight += w
        if analysis.instruments and profile.instrument_weights:
            names = set(analysis.instruments) | set(profile.instrument_weights)
            total_profile = sum(profile.instrument_weights.values()) or 1.0
            overlap = sum(min(analysis.instruments.get(n, 0.0),
                              profile.instrument_weights.get(n, 0.0) / total_profile) for n in names)
            distance += (1.0 - overlap) * 1.5
            weight += 1.5
        if weight:
            scores.append((name, distance / weight))
    scores.sort(key=lambda p: p[1])
    if not scores:
        return []
    # 거리 -> 비중 (가까운 세 개)
    top = scores[:3]
    raw = [math.exp(-d * 8.0) for _, d in top]
    total = sum(raw) or 1.0
    return [(name, value / total) for (name, _), value in zip(top, raw)]


def analyze_project(project: Project, source: str = "", kind: str = "project") -> MusicAnalysis:
    """악보(프로젝트 또는 MIDI 에서 읽은 것)를 분석한다."""
    analysis = MusicAnalysis(source or project.meta.title, kind)
    tracks = [t for t in project.tracks if len(t.notes)]
    if not tracks:
        raise MusicAnalysisError("음이 하나도 없습니다.")
    pitched = [t for t in tracks if t.kind != "drum"]
    drums = [t for t in tracks if t.kind == "drum"]
    signature = project.meter.signature_at_tick(0)
    beats_per_bar = signature.numerator
    analysis.beats_per_bar = beats_per_bar
    end_tick = max(t.notes.end_tick for t in tracks)
    analysis.duration_seconds = project.tempo.tick_to_seconds(end_tick)
    # MIDI 는 템포를 마이크로초로 적어서 114 가 113.99995 가 된다. 0.01 로 반올림한다.
    analysis.bpm = round(float(project.tempo.bpm_at_tick(0)), 2)
    analysis.bpm_confidence = 1.0
    bar_ticks = beats_per_bar * PPQ
    bars = max(1, math.ceil(end_tick / bar_ticks))

    # ---- 조성 ----
    all_notes = [n for t in pitched for n in t.notes]
    if all_notes:
        guesses = detect_key([n.midi for n in all_notes], [n.duration_ticks for n in all_notes])
        if guesses:
            key, score = guesses[0]
            analysis.key = key.display_name if hasattr(key, "display_name") else str(key)
            analysis.key_tonic_pc = key.tonic.pitch_class
            analysis.key_minor = key.mode != "major"
            analysis.key_confidence = float(score)

    # ---- 코드: 반 마디(또는 한 마디)마다 ----
    half = bar_ticks // 2 if beats_per_bar % 2 == 0 else bar_ticks
    windows = math.ceil(end_tick / half)
    chords: list[tuple[float, str]] = []
    bar_chroma = np.zeros((bars, 12))
    for w in range(windows):
        start, stop = w * half, (w + 1) * half
        weights = np.zeros(12)
        bass = None
        lowest = 999
        for note in all_notes:
            overlap = min(stop, note.end_tick) - max(start, note.start_tick)
            if overlap > 0:
                weights[note.midi % 12] += overlap * (0.5 + note.velocity / 254)
                if note.midi < lowest:
                    lowest, bass = note.midi, note.midi % 12
        bar_index = min(bars - 1, start // bar_ticks)
        bar_chroma[bar_index] += weights
        name, _ = best_chord(weights, bass)
        if name == "N":
            continue
        beat = start / PPQ
        if not chords or chords[-1][1] != name:
            chords.append((beat, name))
    analysis.chords = chords
    names = [c for _, c in chords]
    if names:
        analysis.seventh_ratio = sum(1 for c in names if (chord_roots_and_kinds(c) or (0, ""))[1]
                                     in SEVENTH_KINDS) / len(names)
        if analysis.key_tonic_pc is not None:
            analysis.borrowed_ratio = 1.0 - diatonic_ratio(names, analysis.key_tonic_pc,
                                                           bool(analysis.key_minor))
            numerals = [roman_of(c, analysis.key_tonic_pc, bool(analysis.key_minor)) for c in names]
            counts: dict[str, int] = {}
            for i in range(len(numerals) - 3):
                pattern = "-".join(numerals[i:i + 4])
                counts[pattern] = counts.get(pattern, 0) + 1
            analysis.progressions = sorted(counts.items(), key=lambda p: -p[1])[:5]
        analysis.harmonic_rhythm = len(chords) / bars

    # ---- 멜로디: 보컬 트랙, 없으면 가장 높은 음들을 따라가는 선 ----
    vocal = [t for t in pitched if t.kind == "vocal"]
    if not vocal:
        # 가사가 없는 MIDI 도 트랙 이름이나 악기(합창·목소리)로 보컬을 알 수 있다
        vocal = [t for t in pitched
                 if any(word in t.name.lower() for word in ("vocal", "voice", "보컬", "멜로디", "melody", "노래"))
                 or t.instrument == "choir"]
    melody_notes = []
    if vocal:
        melody_notes = sorted(vocal[0].notes, key=lambda n: n.start_tick)
    elif pitched:
        # 멜로디는 대개 한 번에 한 음(단선율)이고 가운데~높은 음역이다
        def melody_score(track) -> float:
            notes = sorted(track.notes, key=lambda n: n.start_tick)
            overlaps = sum(1 for a, b in zip(notes, notes[1:]) if b.start_tick < a.end_tick - PPQ // 16)
            monophony = 1.0 - overlaps / max(1, len(notes) - 1)
            register = float(np.mean([n.midi for n in notes]))
            return monophony + 0.5 * float(np.clip((register - 55) / 25, 0.0, 1.0))
        candidate = max(pitched, key=melody_score)
        melody_notes = sorted(candidate.notes, key=lambda n: n.start_tick)
        analysis.unavailable["멜로디"] = (f"보컬 트랙이 없어 한 음씩 움직이는 트랙 "
                                        f"'{candidate.name}' 을 멜로디로 보았습니다 (틀릴 수 있음).")
    if melody_notes:
        analysis.melody = [(n.start_tick / PPQ, n.duration_ticks / PPQ, n.midi) for n in melody_notes]
        analysis.melody_range = (min(n.midi for n in melody_notes), max(n.midi for n in melody_notes))
        steps = [abs(b.midi - a.midi) for a, b in zip(melody_notes, melody_notes[1:])]
        analysis.melody_leap_ratio = (sum(1 for s in steps if s > 4) / len(steps)) if steps else 0.0
        if vocal:
            analysis.lyrics = "".join(n.lyric + (" " if n.word_end else "") for n in melody_notes).strip()

    # ---- 리듬 ----
    onsets = [n.start_tick for t in pitched for n in t.notes]
    if onsets:
        eighth = PPQ // 2
        off_beat = sum(1 for o in onsets if o % PPQ != 0)
        analysis.syncopation = off_beat / len(onsets)
        # 스윙: 8분음표 뒤쪽 음이 박의 몇 % 지점에 오는가 (50% 정박, 66% 셔플)
        positions = [(o % PPQ) / PPQ for o in onsets if 0.4 < (o % PPQ) / PPQ < 0.8]
        if len(positions) >= 8:
            middle = float(np.median(positions))
            analysis.swing = float(np.clip((middle - 0.5) / (0.667 - 0.5) * 0.66, 0.0, 0.66))
        else:
            analysis.swing = 0.0
    if drums:
        drum_onsets = {n.start_tick // (PPQ // 4) for t in drums for n in t.notes}
        analysis.drum_density = min(1.0, len(drum_onsets) / (bars * beats_per_bar * 4))
    else:
        analysis.drum_density = 0.0

    # ---- 악기와 편곡 ----
    total_notes = sum(len(t.notes) for t in tracks)
    shares: dict[str, float] = {}
    for track in tracks:
        name = track.instrument or track.kind
        shares[name] = shares.get(name, 0.0) + len(track.notes) / total_notes
    analysis.instruments = shares
    active = np.zeros(bars)
    for track in tracks:
        used = {min(bars - 1, n.start_tick // bar_ticks) for n in track.notes}
        for bar in used:
            active[bar] += 1
    analysis.arrangement_density = float(np.mean(active) / len(tracks)) if tracks else 0.0

    # ---- 에너지와 구조 ----
    density = np.zeros(bars)
    loudness = np.zeros(bars)
    rhythm_features = np.zeros((bars, 4))
    for track in tracks:
        for note in track.notes:
            bar = min(bars - 1, note.start_tick // bar_ticks)
            density[bar] += 1
            loudness[bar] += note.velocity
            beat_in_bar = (note.start_tick % bar_ticks) // PPQ
            rhythm_features[bar, min(3, beat_in_bar)] += 1
    energy = (density / (density.max() or 1.0)) * 0.6 + (active / (active.max() or 1.0)) * 0.4
    analysis.energy_curve = [round(float(e), 3) for e in energy]
    chroma = bar_chroma / (bar_chroma.sum(axis=1, keepdims=True) + 1e-12)
    rhythm = rhythm_features / (rhythm_features.sum(axis=1, keepdims=True) + 1e-12)
    # 어떤 트랙이 울리는가. 편곡은 구간마다 악기를 들이고 빼므로 강한 단서다.
    presence = np.zeros((bars, len(tracks)))
    for column, track in enumerate(tracks):
        for note in track.notes:
            presence[min(bars - 1, note.start_tick // bar_ticks), column] = 1.0
    melody_chroma = np.zeros((bars, 12))
    for note in melody_notes:
        melody_chroma[min(bars - 1, note.start_tick // bar_ticks), note.midi % 12] += note.duration_ticks
    melody_chroma /= (melody_chroma.sum(axis=1, keepdims=True) + 1e-12)
    features = np.hstack([chroma, melody_chroma, presence * 0.6, rhythm * 0.3])
    analysis.sections = detect_sections(features, energy)
    for section in analysis.sections:
        section.start_seconds = float(project.tempo.tick_to_seconds((section.start_bar - 1) * bar_ticks))

    # ---- 조성 다시 보기 ----
    # 나란한조(F 단조 / Ab 장조)는 음 분포가 거의 같다. 으뜸화음이 얼마나 나오는지,
    # 구간 첫 코드와 마지막 코드가 무엇인지로 가린다 (EDM 곡의 Fm-Db-Eb 가 Ab 장조로
    # 나오던 것을 이것으로 고쳤다).
    if names and all_notes:
        from .audio_analysis import _chord_at_beat, refine_key
        histogram = np.zeros(12)
        for note in all_notes:
            histogram[note.midi % 12] += note.duration_ticks
        anchors = [_chord_at_beat(chords, (s.start_bar - 1) * beats_per_bar) for s in analysis.sections]
        anchors = [a for a in anchors if a] + [names[-1]]
        tonic, minor, score = refine_key(histogram, names, anchors)
        key = Key(Pitch.from_midi(60 + tonic, prefer_flat=tonic in (1, 3, 5, 8, 10)),
                  "natural_minor" if minor else "major")
        analysis.key = key.display_name
        analysis.key_tonic_pc, analysis.key_minor, analysis.key_confidence = tonic, minor, float(score)
        analysis.borrowed_ratio = 1.0 - diatonic_ratio(names, tonic, minor)
        numerals = [roman_of(c, tonic, minor) for c in names]
        counts: dict[str, int] = {}
        for i in range(len(numerals) - 3):
            pattern = "-".join(numerals[i:i + 4])
            counts[pattern] = counts.get(pattern, 0) + 1
        analysis.progressions = sorted(counts.items(), key=lambda p: -p[1])[:5]

    analysis.genre_guess = _genre_guess(analysis)
    analysis.unavailable.setdefault(
        "믹스", "악보에는 소리의 크기·공간감이 없습니다. 오디오 파일을 넣으면 잽니다.")
    if not vocal:
        analysis.unavailable.setdefault("가사", "가사가 적힌 트랙이 없습니다.")
        analysis.unavailable.setdefault("보컬 표현", "보컬 트랙이 없습니다.")
    else:
        velocities = [n.velocity for n in melody_notes]
        analysis.vocal_expression = {
            "세기 폭": float(np.percentile(velocities, 95) - np.percentile(velocities, 5)),
            "레가토 비율": float(np.mean([1.0 if b.start_tick - a.end_tick <= PPQ // 16 else 0.0
                                         for a, b in zip(melody_notes, melody_notes[1:])]))
            if len(melody_notes) > 1 else 0.0,
        }
    return analysis


def analyze_midi(path) -> MusicAnalysis:
    from pathlib import Path
    from ..audio.export import import_midi

    project = import_midi(path)
    return analyze_project(project, Path(path).name, "midi")
