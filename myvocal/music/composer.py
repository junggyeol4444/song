"""
작곡 — 설명 한 줄에서 편집 가능한 프로젝트까지.

3번의 예시가 이 파일이 하는 일이다.

    애니메이션 록 발라드을 만들어줘.
    주제: 오래 헤어져 있던 친구를 다시 만나는 이야기
    장르: Rock 50% / K-Pop 30% / Orchestral 20%
    마지막 후렴은 웅장하게.

여기서 나오는 것은 완성된 음원 한 덩어리가 아니다. 트랙과 음으로 이루어진
프로젝트다. 5번이 그렇게 하라고 못박고 있다. 음원으로 나오면 드럼만 빼거나
베이스 한 음을 고치는 것이 불가능해진다.

구간마다 무엇을 넣고 뺄지가 편곡의 핵심이다. 처음부터 끝까지 같은 악기가
같은 밀도로 나오면 곡에 기승전결이 없다. 그래서 에너지에 따라 악기를
들이고 뺀다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..core.notes import Note
from ..core.project import Project
from ..core.tracks import Track
from ..core.units import PPQ
from .arranger import (
    AccompanimentGenerator, ArrangeError, BassGenerator, DrumGenerator,
)
from .genre import GenreBlend, GenreProfile, get_genre
from .harmony import (
    ChordProgression, HarmonyError, ProgressionGenerator, apply_cadence,
)
from .melody import MelodyError, MelodyGenerator, VocalRange
from .structure import Section, SongStructure, build_structure
from .theory import Key, Pitch, TheoryError


class ComposeError(ValueError):
    """작곡 관련 오류."""


# 악기가 나오기 시작하는 에너지 문턱.
# 값이 높을수록 늦게 들어온다. 이 차이가 곡의 기승전결을 만든다.
ENTRY_THRESHOLD: dict[str, float] = {
    "acoustic_piano": 0.0, "electric_piano": 0.0, "synth_pad": 0.0, "organ": 0.2,
    "acoustic_guitar": 0.15, "electric_bass": 0.2, "sub_bass": 0.2, "synth_bass": 0.25,
    "drum_kit": 0.25, "strings": 0.35, "electric_guitar": 0.4, "synth_lead": 0.5,
    "brass": 0.55, "choir": 0.6, "bell": 0.3, "flute": 0.35,
}

# 역할별 기본 음량과 좌우 위치. 믹스의 출발점이다.
ROLE_MIX: dict[str, tuple[float, float]] = {
    "vocal":        (0.0, 0.0),
    "bass":         (-3.0, 0.0),
    "drums":        (-3.5, 0.0),
    "chords":       (-7.0, -0.2),
    "pad":          (-13.0, 0.3),
    "counter":      (-11.0, 0.35),
    "lead":         (-8.0, -0.3),
}


@dataclass(slots=True)
class SongRequest:
    """만들 곡에 대한 요청. 3번의 입력을 그대로 담는다."""

    title: str = "제목 없음"
    genre: GenreBlend | str = "pop"
    key: Key | str | None = None        # None 이면 장르 성향으로 정한다
    bpm: float | None = None            # None 이면 장르 범위에서 정한다
    subject: str = ""                   # 주제. 작사에 쓴다.
    mood: str = ""
    vocal_range: VocalRange | None = None
    structure: SongStructure | None = None
    final_chorus_note: str = ""         # "마지막 후렴은 웅장하게"
    repeat_final_chorus: bool = False
    seed: int | None = None
    time_signature: str = "4/4"

    def resolved_genre(self) -> GenreBlend:
        return self.genre if isinstance(self.genre, GenreBlend) else GenreBlend.parse(self.genre)


def choose_key(profile: GenreProfile, generator: random.Random,
               vocal_range: VocalRange | None = None) -> Key:
    """장르 성향과 목소리 음역에 맞는 조성을 고른다.

    아무 조나 고르면 안 된다. 노래하는 사람의 음역 가운데에 곡의 중심이
    오도록 골라야 부를 수 있다. 같은 곡도 조가 두 반음 다르면 못 부르게 된다.
    """
    minor = generator.random() < profile.minor_tendency
    # 연주하기 쉬운 조를 앞에 둔다. 플랫이 다섯 개 넘는 조는 피한다.
    candidates = ["C", "G", "D", "A", "E", "F", "Bb", "Eb", "Ab"] if not minor else \
                 ["Am", "Em", "Bm", "F#m", "Dm", "Gm", "Cm", "Fm", "Bbm"]
    if vocal_range is None:
        return Key.parse(generator.choice(candidates))
    # 목소리 음역의 가운데에 으뜸음이 가깝게 오는 조를 고른다
    center = vocal_range.center
    scored: list[tuple[float, str]] = []
    for name in candidates:
        key = Key.parse(name)
        tonic = key.tonic.pitch_class
        # 음역 안에서 그 으뜸음에 가장 가까운 자리
        base = center - (center % 12) + tonic
        best = min((abs(candidate - center) for candidate in (base - 12, base, base + 12)),
                   default=12)
        scored.append((best + generator.random() * 1.5, name))
    scored.sort()
    return Key.parse(scored[0][1])


def choose_bpm(profile: GenreProfile, generator: random.Random) -> float:
    """장르 범위 안에서 BPM 을 고른다. 보통값 근처가 나올 확률이 높게 한다."""
    low, high, typical = profile.bpm_low, profile.bpm_high, profile.bpm_typical
    # 보통값을 중심으로 한 삼각 분포
    value = generator.triangular(low, high, typical)
    # 정수로 반올림하되 float 로 돌려준다. BPM 은 소수가 될 수 있는 값이고
    # (MIDI 는 템포를 정수 마이크로초로 저장하므로 왕복하면 소수가 된다),
    # 여기서 int 를 돌려주면 타입이 섞여서 비교할 때마다 걸린다.
    return float(round(value))


def select_instruments(
    profile: GenreProfile, energy: float, limit: int = 6,
) -> list[tuple[str, str]]:
    """이 에너지에서 나올 악기들. 돌려주는 값은 [(악기, 역할), ...] 다.

    에너지가 낮으면 적게, 높으면 많이 나온다. 이게 편곡의 기승전결이다.
    """
    weights = profile.instrument_weights
    if not weights:
        return [("acoustic_piano", "chords")]

    # 에너지에 따라 몇 개까지 쓸지
    density = profile.arrangement_density * (0.35 + 0.75 * energy)
    count = max(1, min(limit, int(round(1 + density * (limit - 1)))))

    chosen: list[tuple[str, str]] = []
    used: set[str] = set()

    def take(candidates: Sequence[str], role: str) -> str | None:
        for name in candidates:
            if name in used:
                continue
            if weights.get(name, 0.0) <= 0.0:
                continue
            if energy < ENTRY_THRESHOLD.get(name, 0.3):
                continue
            used.add(name)
            chosen.append((name, role))
            return name
        return None

    # 베이스와 드럼이 먼저다. 곡의 바닥이다.
    take(["sub_bass", "electric_bass", "synth_bass"]
         if profile.low_end > 0.7 else ["electric_bass", "synth_bass", "sub_bass"], "bass")
    take(["drum_kit"], "drums")
    # 화음을 담당할 악기
    take(["acoustic_piano", "electric_piano", "acoustic_guitar", "electric_guitar",
          "organ", "synth_pad"], "chords")

    # 남은 자리는 가중치가 큰 순서로 채운다
    ranked = sorted(weights.items(), key=lambda pair: pair[1], reverse=True)
    for name, weight in ranked:
        if len(chosen) >= count:
            break
        if name in used or weight <= 0.0:
            continue
        if energy < ENTRY_THRESHOLD.get(name, 0.3):
            continue
        role = "pad" if name in ("synth_pad", "strings", "choir") else (
            "lead" if name in ("synth_lead", "brass", "flute", "bell") else "counter"
        )
        used.add(name)
        chosen.append((name, role))
    return chosen


class Composer:
    """곡 하나를 만든다."""

    def __init__(self, request: SongRequest) -> None:
        self.request = request
        self.random = random.Random(request.seed)
        self.blend = request.resolved_genre()
        self.profile = self.blend.resolve()
        self.vocal_range = request.vocal_range or VocalRange.typical("alto")

        if request.key is not None:
            self.key = request.key if isinstance(request.key, Key) else Key.parse(request.key)
        else:
            self.key = choose_key(self.profile, self.random, self.vocal_range)
        self.bpm = request.bpm if request.bpm is not None else choose_bpm(self.profile, self.random)

    # ------------------------------------------------------------------

    def compose(self, with_melody: bool = True, with_arrangement: bool = True) -> Project:
        """프로젝트를 만든다."""
        project = Project(
            title=self.request.title,
            key=self.key,
            bpm=self.bpm,
            time_signature=self.request.time_signature,
            genre=self.blend,
        )
        project.meta.description = self.request.subject
        if self.request.mood:
            project.meta.tags.append(self.request.mood)

        structure = self.request.structure or build_structure(
            self.profile.name.split("+")[0] if "+" in self.profile.name else self.profile.name,
            self.request.repeat_final_chorus,
        )
        project.set_structure(structure)

        if self.request.final_chorus_note:
            last = structure.last_of_kind("chorus")
            if last is not None:
                last.notes_for_ai = self.request.final_chorus_note
                # '웅장하게' 같은 지시는 에너지를 최대로 올린다는 뜻이다
                last.energy = 1.0

        beats_per_bar = project.time_signature.numerator

        # 구간마다 코드 진행을 만든다. 같은 종류의 구간은 같은 진행을 쓴다.
        # 후렴이 나올 때마다 화음이 다르면 사람이 후렴으로 인식하지 못한다.
        progressions: dict[str, ChordProgression] = {}
        section_progressions: list[tuple[Section, ChordProgression]] = []
        for section in structure:
            blend = section.genre or self.blend
            profile = blend.resolve()
            key = section.key or self.key
            bars = section.length_bars
            cache_key = f"{section.kind}:{bars}:{key}:{profile.name}"
            if cache_key in progressions:
                progression = progressions[cache_key]
            else:
                generator = ProgressionGenerator(
                    key, profile, seed=self.random.randint(0, 2 ** 31)
                )
                cadence = "authentic" if section.kind in ("chorus", "outro") else (
                    "half" if section.kind in ("verse", "pre_chorus", "build") else None
                )
                progression = generator.from_template(bars, beats_per_bar)
                if cadence and bars >= 4 and section.kind not in (
                    "intro", "breakdown", "interlude", "drop"
                ):
                    # 마지막 두 마디를 종지로 바꾼다. 길이는 그대로 둔다.
                    progression = apply_cadence(progression, cadence, beats_per_bar)
                progressions[cache_key] = progression
            section_progressions.append((section, progression))

        # 트랙을 만든다. 구간마다 악기가 다르므로, 곡 전체에서 한 번이라도
        # 쓰이는 악기에 대해 트랙을 하나씩 만들고 거기에 음을 넣는다.
        tracks: dict[str, Track] = {}

        def track_for(instrument: str, role: str) -> Track:
            if instrument in tracks:
                return tracks[instrument]
            volume, pan = ROLE_MIX.get(role, (-9.0, 0.0))
            kind = "drum" if instrument == "drum_kit" else "instrument"
            track = project.add_track(
                _instrument_label(instrument), kind,
                instrument=instrument, volume_db=volume, pan=pan,
                generated_by="ai_arranger",
            )
            tracks[instrument] = track
            return track

        vocal_track: Track | None = None
        if with_melody:
            vocal_track = project.add_track(
                "리드 보컬", "vocal", instrument="choir",
                volume_db=ROLE_MIX["vocal"][0], pan=ROLE_MIX["vocal"][1],
                generated_by="ai_composer",
            )

        for section, progression in section_progressions:
            span = project.section_range(section)
            start_tick = span.start_tick
            energy = section.effective_energy()
            blend = section.genre or self.blend
            profile = blend.resolve()
            key = section.key or self.key
            seed = self.random.randint(0, 2 ** 31)

            if with_melody and vocal_track is not None and section.kind not in (
                "intro", "instrumental", "solo", "interlude", "outro", "breakdown", "drop"
            ):
                melody = MelodyGenerator(
                    key, profile, self.vocal_range, seed=seed
                ).generate(
                    progression, beats_per_bar, section.kind, energy,
                    bars_per_phrase=2, start_tick=start_tick,
                )
                vocal_track.add_notes(_clip_to_span(melody, start_tick, span.end_tick))

            if not with_arrangement:
                continue

            for instrument, role in select_instruments(profile, energy):
                track = track_for(instrument, role)
                if role == "bass":
                    low, high = _bass_range(instrument)
                    notes = BassGenerator(key, profile, low, high, seed=seed).generate(
                        progression, beats_per_bar, energy, start_tick=start_tick
                    )
                elif role == "drums":
                    notes = DrumGenerator(profile, seed=seed).generate(
                        section.length_bars, beats_per_bar, energy,
                        start_tick=start_tick,
                        crash_at_start=section.kind in ("chorus", "drop", "solo"),
                    )
                else:
                    low, high = _accompaniment_range(instrument, role)
                    style = None
                    if role == "pad":
                        style = "pad"
                    notes = AccompanimentGenerator(profile, seed=seed).generate(
                        progression, beats_per_bar, energy, style=style,
                        start_tick=start_tick, low_midi=low, high_midi=high,
                    )
                track.add_notes(_clip_to_span(notes, start_tick, span.end_tick))

        return project


def _instrument_label(instrument: str) -> str:
    labels = {
        "acoustic_piano": "피아노", "electric_piano": "일렉 피아노", "organ": "오르간",
        "acoustic_guitar": "통기타", "electric_guitar": "일렉 기타",
        "electric_bass": "베이스", "synth_bass": "신스 베이스", "sub_bass": "서브 베이스",
        "strings": "스트링", "brass": "브라스", "flute": "플루트",
        "synth_pad": "패드", "synth_lead": "리드", "bell": "벨",
        "choir": "코러스", "drum_kit": "드럼",
    }
    return labels.get(instrument, instrument)


def _bass_range(instrument: str) -> tuple[int, int]:
    from .instruments import create_instrument

    info = create_instrument(instrument).info
    return info.lowest_midi, min(info.highest_midi, info.lowest_midi + 27)


def _accompaniment_range(instrument: str, role: str) -> tuple[int, int]:
    from .instruments import create_instrument

    info = create_instrument(instrument).info
    if role == "pad":
        low = max(info.lowest_midi, 52)
        high = min(info.highest_midi, 84)
    elif role == "lead":
        low = max(info.lowest_midi, 60)
        high = min(info.highest_midi, 88)
    else:
        low = max(info.lowest_midi, 48)
        high = min(info.highest_midi, 79)
    if low >= high:
        low, high = info.lowest_midi, info.highest_midi
    return low, high


def compose(
    description: str = "", title: str = "제목 없음", genre: str = "pop",
    key: str | None = None, bpm: float | None = None, subject: str = "",
    final_chorus_note: str = "", seed: int | None = None,
    vocal_range: VocalRange | None = None,
) -> Project:
    """설명에서 곡을 만든다. 가장 간단한 진입점이다."""
    request = SongRequest(
        title=title, genre=genre, key=key, bpm=bpm, subject=subject or description,
        final_chorus_note=final_chorus_note, seed=seed, vocal_range=vocal_range,
    )
    return Composer(request).compose()


def _clip_to_span(notes: Iterable[Note], start_tick: int, end_tick: int) -> list[Note]:
    """구간 밖으로 넘치는 음을 잘라낸다.

    편곡기가 구간 길이를 조금 넘겨도 다음 구간을 침범하지 않게 하는 안전장치다.
    침범하면 같은 음이 겹쳐서 앞 음이 울리는 중에 다시 쳐지고, 소리가 지저분해진다.
    """
    result: list[Note] = []
    for note in notes:
        if note.start_tick >= end_tick:
            continue
        if note.end_tick > end_tick:
            length = end_tick - note.start_tick
            if length < 1:
                continue
            note = note.resized(length)
        result.append(note)
    return result
