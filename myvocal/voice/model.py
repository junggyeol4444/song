"""
목소리 모델과 창법 — 7번.

    VOICE MODEL     누구의 목소리인가   (성도 길이, 모음 포먼트, 음역, 숨, 밝기,
                                        비브라토 습관, 음 잡는 습관, 가성 전환점)
    SINGING STYLE   어떻게 부르는가     (Natural / Ballad / Soft / Power / Rock /
                                        Whisper / Falsetto / Emotional)

둘을 나눠 두면 "내 목소리로 록 창법" 과 "내 목소리로 속삭이듯" 을 같은
목소리 모델 하나로 만들 수 있다.

솔직하게 적어 둔다. 이 모델은 측정한 성질을 포먼트 합성기의 설정값으로
옮긴 것이다. 신경망 음성 복제처럼 그 사람 목소리를 그대로 재현하지는 않는다.
음역, 모음 공명 자리, 숨 섞임, 비브라토 같은 성질은 따라가지만, 목소리의
세밀한 결까지 같지는 않다. 신경망 복제는 Provider(VOICE_CLONE, SINGING)
자리로 남겨 두었다.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from scipy import signal as scipy_signal

from .analysis import (
    GuideNote, NoteMeasure, VoiceReport, align_offset, analyze_take, cents, extract_contour,
    guided_notes, hz_to_midi, midi_to_hz, note_name, note_pitch, summarize,
)
from .phonemes import VOWEL_FORMANTS, align_lyrics
from .synth import SingingSynth, VoiceError, VoiceTimbre


# ==========================================================================
# 창법
# ==========================================================================

@dataclass(frozen=True, slots=True)
class SingingStyle:
    """부르는 방식. 목소리 모델 위에 얹는 변화량이다."""

    name: str
    display_name: str
    description: str
    breathiness_add: float = 0.0
    breathiness_set: float | None = None
    tension_add: float = 0.0
    brightness_add: float = 0.0
    vibrato_depth_mul: float = 1.0
    vibrato_rate_mul: float = 1.0
    vibrato_onset_mul: float = 1.0
    scoop_cents: float | None = None      # None 이면 목소리 모델의 습관 그대로
    portamento: float | None = None
    roughness: float = 0.0
    falsetto: str = "natural"             # natural(목소리 모델의 전환점) / never / always


SINGING_STYLES: dict[str, SingingStyle] = {
    s.name: s for s in (
        SingingStyle("natural", "Natural", "평소 부르는 그대로."),
        SingingStyle("ballad", "Ballad", "숨을 조금 섞고, 비브라토를 늦게 깊게, 음을 아래에서 밀어 올림.",
                     breathiness_add=0.08, brightness_add=-0.1, vibrato_depth_mul=1.2,
                     vibrato_onset_mul=1.3, scoop_cents=-40.0, portamento=0.07),
        SingingStyle("soft", "Soft", "힘을 빼고 부드럽게.",
                     breathiness_add=0.15, tension_add=-0.2, brightness_add=-0.25,
                     vibrato_depth_mul=0.7),
        SingingStyle("power", "Power", "숨 없이 꽉 찬 소리로 강하게. 고음도 가성 없이.",
                     breathiness_add=-0.08, tension_add=0.3, brightness_add=0.3,
                     vibrato_depth_mul=1.2, vibrato_rate_mul=1.05, falsetto="never"),
        SingingStyle("rock", "Rock", "거칠고 밝게, 음을 아래에서 치고 올라감.",
                     tension_add=0.35, brightness_add=0.35, roughness=0.45, scoop_cents=-60.0,
                     vibrato_depth_mul=0.6, falsetto="never"),
        SingingStyle("whisper", "Whisper", "속삭이듯. 숨이 대부분.",
                     breathiness_set=0.85, tension_add=-0.3, brightness_add=0.1,
                     vibrato_depth_mul=0.2, falsetto="never"),
        SingingStyle("falsetto", "Falsetto", "모든 음을 가성으로.",
                     breathiness_add=0.03, brightness_add=-0.1, falsetto="always"),
        SingingStyle("emotional", "Emotional", "비브라토를 크게, 음 사이를 길게 이어 감정을 싣기.",
                     breathiness_add=0.05, vibrato_depth_mul=1.35, vibrato_onset_mul=1.5,
                     scoop_cents=-50.0, portamento=0.08),
    )
}


def get_style(name: str) -> SingingStyle:
    key = (name or "natural").lower()
    if key not in SINGING_STYLES:
        raise VoiceError(f"모르는 창법입니다: {name!r} (가능: {', '.join(SINGING_STYLES)})")
    return SINGING_STYLES[key]


# ==========================================================================
# 목소리 모델
# ==========================================================================

RIGHTS_CHOICES: dict[str, str] = {
    "my_voice": "내 목소리",
    "licensed": "사용 허가를 받은 목소리",
}


@dataclass(slots=True)
class VoiceModel:
    """한 사람의 목소리."""

    name: str
    voice_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    rights: str = "my_voice"
    created_at: float = field(default_factory=time.time)
    builtin: bool = False

    lowest_hz: float = 130.0
    highest_hz: float = 523.0
    comfortable_low_hz: float = 165.0
    comfortable_high_hz: float = 440.0

    formant_scale: float = 1.0
    vowel_formants: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    breathiness: float = 0.16
    tension: float = 0.5
    brightness: float = 0.0
    vibrato_rate: float = 5.4
    vibrato_depth: float = 28.0
    vibrato_onset: float = 0.32
    scoop_cents: float = 0.0
    scoop_time: float = 0.08
    portamento: float = 0.045
    falsetto_from_hz: float = 0.0

    takes_used: int = 0
    notes_used: int = 0
    report_text: str = ""
    measured: list[str] = field(default_factory=list)   # 실제로 잰 항목 (나머지는 기본값)
    reproduction: dict[str, float] = field(default_factory=dict)   # 모델로 다시 부른 차이

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise VoiceError("목소리 이름이 비어 있습니다.")
        if self.rights not in RIGHTS_CHOICES:
            raise VoiceError(
                f"목소리 모델은 본인 목소리나 허가받은 목소리로만 만들 수 있습니다 "
                f"(가능: {', '.join(RIGHTS_CHOICES.values())})."
            )
        if not (0 < self.lowest_hz <= self.highest_hz):
            raise VoiceError(f"음역이 잘못됐습니다: {self.lowest_hz} ~ {self.highest_hz}")

    @property
    def reference(self) -> str:
        """트랙에 적는 이름. 'user:<id>' 또는 'preset:<이름>'."""
        return f"{'preset' if self.builtin else 'user'}:{self.voice_id}"

    def vocal_range(self):
        """작곡기가 쓰는 음역으로. 멜로디를 이 사람이 부를 수 있는 범위에 만든다."""
        from ..music.melody import VocalRange

        low = int(round(hz_to_midi(self.lowest_hz)))
        high = max(low + 1, int(round(hz_to_midi(self.highest_hz))))
        comfortable_low = int(round(hz_to_midi(self.comfortable_low_hz)))
        comfortable_high = int(round(hz_to_midi(self.comfortable_high_hz)))
        if not (low <= comfortable_low < comfortable_high <= high):
            comfortable_low = comfortable_high = 0
        return VocalRange(low, high, comfortable_low, comfortable_high)

    def range_text(self) -> str:
        return (f"{note_name(self.lowest_hz)}~{note_name(self.highest_hz)} "
                f"(편한 음역 {note_name(self.comfortable_low_hz)}~{note_name(self.comfortable_high_hz)})")

    # ---- 합성기 설정으로 ----

    def timbre(self, style: SingingStyle | str = "natural") -> VoiceTimbre:
        style = get_style(style) if isinstance(style, str) else style
        breathiness = (style.breathiness_set if style.breathiness_set is not None
                       else self.breathiness + style.breathiness_add)
        if style.falsetto == "always":
            falsetto_from = 1.0
        elif style.falsetto == "never":
            falsetto_from = 0.0
        else:
            falsetto_from = self.falsetto_from_hz
        return VoiceTimbre(
            name=f"{self.name} · {style.display_name}",
            formant_scale=float(np.clip(self.formant_scale, 0.5, 1.6)),
            breathiness=float(np.clip(breathiness, 0.0, 1.0)),
            tension=float(np.clip(self.tension + style.tension_add, 0.0, 1.0)),
            brightness=float(np.clip(self.brightness + style.brightness_add, -1.0, 1.0)),
            vibrato_rate=self.vibrato_rate * style.vibrato_rate_mul,
            vibrato_depth=self.vibrato_depth * style.vibrato_depth_mul,
            vibrato_onset=self.vibrato_onset * style.vibrato_onset_mul,
            portamento=style.portamento if style.portamento is not None else self.portamento,
            scoop_cents=style.scoop_cents if style.scoop_cents is not None else self.scoop_cents,
            scoop_time=self.scoop_time,
            roughness=style.roughness,
            falsetto_from_hz=falsetto_from,
            vowel_formants=tuple((v, *f) for v, f in sorted(self.vowel_formants.items())),
        )

    # ---- 저장 ----

    def to_dict(self) -> dict:
        data = {k: getattr(self, k) for k in (
            "name", "voice_id", "rights", "created_at", "lowest_hz", "highest_hz",
            "comfortable_low_hz", "comfortable_high_hz", "formant_scale", "breathiness",
            "tension", "brightness", "vibrato_rate", "vibrato_depth", "vibrato_onset",
            "scoop_cents", "scoop_time", "portamento", "falsetto_from_hz", "takes_used",
            "notes_used", "report_text", "measured", "reproduction",
        )}
        data["vowel_formants"] = {v: list(f) for v, f in self.vowel_formants.items()}
        data["format"] = 1
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceModel":
        known = {k: data[k] for k in (
            "name", "voice_id", "rights", "created_at", "lowest_hz", "highest_hz",
            "comfortable_low_hz", "comfortable_high_hz", "formant_scale", "breathiness",
            "tension", "brightness", "vibrato_rate", "vibrato_depth", "vibrato_onset",
            "scoop_cents", "scoop_time", "portamento", "falsetto_from_hz", "takes_used",
            "notes_used", "report_text", "measured", "reproduction",
        ) if k in data}
        known["vowel_formants"] = {v: tuple(f) for v, f in data.get("vowel_formants", {}).items()}
        return cls(**known)


def preset_voices() -> dict[str, VoiceModel]:
    """학습 전에도 고를 수 있는 기본 목소리들. 음역은 작곡기(VocalRange)와 같은 표를 쓴다."""
    from ..music.melody import VocalRange

    names = {"soprano": "소프라노", "mezzo": "메조소프라노", "alto": "알토",
             "tenor": "테너", "baritone": "바리톤", "bass": "베이스"}
    result = {}
    for key, label in names.items():
        t = VoiceTimbre.preset(key)
        vocal = VocalRange.typical(key)
        result[key] = VoiceModel(
            name=label, voice_id=key, builtin=True,
            lowest_hz=midi_to_hz(vocal.lowest), highest_hz=midi_to_hz(vocal.highest),
            comfortable_low_hz=midi_to_hz(vocal.comfortable_low),
            comfortable_high_hz=midi_to_hz(vocal.comfortable_high),
            formant_scale=t.formant_scale, breathiness=t.breathiness, tension=t.tension,
            brightness=t.brightness, vibrato_rate=t.vibrato_rate, vibrato_depth=t.vibrato_depth,
            vibrato_onset=t.vibrato_onset, portamento=t.portamento,
        )
    return result


# ==========================================================================
# 녹음에서 모델 만들기
# ==========================================================================

@dataclass(slots=True)
class Take:
    """녹음 하나와, 그때 부르라고 한 것."""

    signal: np.ndarray
    rate: int
    guide: list[GuideNote] = field(default_factory=list)
    label: str = ""


def _synthesize_like(take_guide: Sequence[GuideNote], timbre: VoiceTimbre, rate: int) -> np.ndarray:
    times = [(n.start, n.duration) for n in take_guide]
    text = "".join(n.syllable for n in take_guide)
    syllables = align_lyrics(text, times)
    return SingingSynth(timbre, rate).render(syllables, [midi_to_hz(n.midi) for n in take_guide]).data[0]


# 합성은 실제로 곡을 만들 때와 같은 샘플레이트로 한다. 합성기의 숨소리 잡음은
# 샘플레이트에 따라 대역당 크기가 달라서, 16kHz 로 합성해 맞춘 값을 48kHz 곡에
# 쓰면 숨이 전혀 다르게 들린다 (같은 설정에서 HNR 이 10dB 넘게 차이 났다).
RENDER_RATE = 48000
# 재는 것은 비교하는 양쪽을 같은 샘플레이트로 내려서 한다.
MEASURE_RATE = 16000


def _resample(signal: np.ndarray, rate: int, target: int = MEASURE_RATE) -> np.ndarray:
    if rate == target:
        return signal
    divisor = math.gcd(rate, target)
    return scipy_signal.resample_poly(signal, target // divisor, rate // divisor)


def _measure_synth(guide: Sequence[GuideNote], timbre: VoiceTimbre) -> VoiceReport:
    audio = _synthesize_like(guide, timbre, RENDER_RATE)
    return summarize(analyze_take(_resample(audio, RENDER_RATE), MEASURE_RATE, guide))


def _attack_curves(signal: np.ndarray, rate: int, guide: Sequence[GuideNote],
                   frames: int = 9) -> list[np.ndarray]:
    """각 음의 소리가 난 순간부터 90ms 동안의 음높이 (안정 음높이 기준 센트).

    그 뒤로는 비브라토가 곡선을 차지한다. 비브라토의 위상은 녹음마다 제멋대로라
    비교하면 흐려지기만 한다.
    """
    contour = extract_contour(signal, rate)
    offset = align_offset(contour, guide)
    windows = [(n.start + offset, n.start + n.duration + offset, midi_to_hz(n.midi)) for n in guide]

    def expected(seconds: float) -> float | None:
        for begin, finish, hz in windows:
            if begin <= seconds < finish:
                return hz
        return None

    contour = extract_contour(signal, rate, expected)
    curves = []
    for note in guided_notes(contour, guide, offset):
        stable = note_pitch(note, contour)
        values = contour.f0[note.frames.start:note.frames.start + frames]
        curve = np.full(frames, np.nan)
        if not np.isnan(stable):
            curve[:len(values)] = cents(values, stable)
        curves.append(curve)
    return curves


def _fit_scoop(sample: "Take", model: VoiceModel) -> tuple[float, float] | None:
    """음 시작의 음높이 곡선을 맞춰 스쿱 (센트, 시간) 을 찾는다.

    '얼마나 아래에서' 와 '얼마나 빨리' 는 측정값 하나로는 구별되지 않는다.
    그래서 쉬었다 들어가는 음들의 처음 150ms 곡선 전체를 비교한다. 같은 음들을
    여러 조합으로 합성해서, 측정한 곡선이 녹음의 곡선과 가장 가까운 조합을 고른다.
    """
    guide = sample.guide
    chosen = [g for i, g in enumerate(guide)
              if g.duration >= 0.4 and (i == 0 or g.start - (guide[i - 1].start + guide[i - 1].duration) >= 0.1)
              and (not model.falsetto_from_hz or midi_to_hz(g.midi) < model.falsetto_from_hz)][:3]
    if len(chosen) < 2:
        return None
    # 고른 음들만 떼어 새 가이드로 (0.2초 쉬고 시작, 음 사이 0.3초)
    excerpt, cursor = [], 0.2
    for g in chosen:
        excerpt.append(GuideNote(cursor, g.duration, g.midi, g.syllable))
        cursor += g.duration + 0.3
    rate = 24000     # 음높이 곡선은 샘플레이트와 무관하다. 빠르게 하려고 낮춘다.
    target_curves = _attack_curves(_resample(sample.signal, sample.rate, rate), rate, chosen)
    if len(target_curves) != len(chosen):
        return None
    target = np.concatenate(target_curves)

    def error(scoop: float, duration: float) -> float:
        timbre = replace(model.timbre("natural"), scoop_cents=scoop, scoop_time=duration,
                         vibrato_depth=0.0)
        curves = _attack_curves(_synthesize_like(excerpt, timbre, rate), rate, excerpt)
        if len(curves) != len(chosen):
            return float("inf")
        trial = np.concatenate(curves)
        both = ~np.isnan(target) & ~np.isnan(trial)
        only_one = np.isnan(target) != np.isnan(trial)
        if both.sum() < 5:
            return float("inf")
        return float(np.sqrt(np.mean((target[both] - trial[both]) ** 2))) + 5.0 * float(only_one.mean())

    best = (float("inf"), 0.0, 0.08)
    for scoop in (-200.0, -150.0, -100.0, -60.0, -30.0, 0.0):
        for duration in (0.03, 0.06, 0.1, 0.15, 0.2):
            value = error(scoop, duration)
            if value < best[0]:
                best = (value, scoop, duration)
    _, scoop, duration = best
    for fine_scoop in (scoop - 20.0, scoop, scoop + 20.0):
        for fine_duration in (duration - 0.015, duration, duration + 0.015):
            if fine_duration < 0.02:
                continue
            value = error(fine_scoop, fine_duration)
            if value < best[0]:
                best = (value, fine_scoop, fine_duration)
    if not np.isfinite(best[0]):
        return None
    return float(np.clip(best[1], -250.0, 100.0)), float(np.clip(best[2], 0.02, 0.25))


def _calibrate(target: float, low: float, high: float, measure: Callable[[float], float | None],
               increasing: bool, steps: int = 6) -> float:
    """measure(x) 가 target 이 되는 x 를 이분법으로 찾는다 (단조 함수라고 보고)."""
    for _ in range(steps):
        middle = 0.5 * (low + high)
        value = measure(middle)
        if value is None:
            break
        if (value < target) == increasing:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def build_voice_model(
    name: str, takes: Sequence[Take], rights: str = "my_voice",
    calibrate: bool = True, progress: Callable[[str, float], None] | None = None,
) -> tuple[VoiceModel, VoiceReport]:
    """녹음들로 목소리 모델을 만든다.

    잰 값을 그대로 옮기는 것(음역, 포먼트, 비브라토, 스쿱, 가성 전환점)과,
    합성기로 다시 만들어 같은 측정값이 나오게 맞추는 것(숨 섞임, 밝기)이 있다.
    뒤의 것은 측정값과 합성기 설정값의 관계가 모음·음높이 구성에 따라 달라서,
    이 사람이 실제로 부른 연습과 같은 것을 합성해 보며 맞춘다.
    """
    def report_progress(text: str, value: float) -> None:
        if progress is not None:
            progress(text, value)

    if rights not in RIGHTS_CHOICES:
        raise VoiceError("목소리 모델은 본인 목소리나 허가받은 목소리로만 만들 수 있습니다.")
    if not takes:
        raise VoiceError("녹음이 없습니다.")
    all_notes: list[NoteMeasure] = []
    for index, take in enumerate(takes):
        report_progress(f"녹음 분석 중 ({index + 1}/{len(takes)})", 0.6 * index / len(takes))
        all_notes.extend(analyze_take(take.signal, take.rate, take.guide or None))
    report = summarize(all_notes)
    if not report.notes or report.lowest_hz is None:
        raise VoiceError("노래한 음을 찾지 못했습니다. 녹음이 너무 작거나 잡음이 많습니다.")

    model = VoiceModel(name=name, rights=rights)
    measured: list[str] = []
    model.lowest_hz, model.highest_hz = report.lowest_hz, report.highest_hz
    model.comfortable_low_hz = report.comfortable_low_hz or report.lowest_hz
    model.comfortable_high_hz = report.comfortable_high_hz or report.highest_hz
    measured.append("음역")
    if report.formant_scale:
        model.formant_scale = report.formant_scale
        model.vowel_formants = dict(report.vowel_formants)
        measured.append("성도 길이·모음 포먼트")
    if report.vibrato_rate:
        model.vibrato_rate = report.vibrato_rate
        model.vibrato_depth = report.vibrato_extent or model.vibrato_depth
        model.vibrato_onset = float(np.clip(report.vibrato_onset or 0.3, 0.05, 1.0))
        measured.append("비브라토")
    elif report.vibrato_share == 0.0 and any(n.duration >= 0.6 for n in report.notes):
        model.vibrato_depth = 4.0                 # 긴 음에서도 비브라토를 안 쓰는 사람
        measured.append("비브라토 (안 씀)")
    if report.scoop_cents is not None:
        # 첫값. 아래에서 합성해 보며 다시 맞춘다 (측정 창이 시작을 흐려서 작게 잡힌다)
        model.scoop_cents = float(np.clip(report.scoop_cents, -200.0, 100.0))
        if report.attack_seconds:
            model.scoop_time = float(np.clip(report.attack_seconds * 1.3, 0.03, 0.25))
        measured.append("음 잡는 습관")
    if report.falsetto_from_hz:
        model.falsetto_from_hz = report.falsetto_from_hz
        measured.append("가성 전환점")

    guided = [t for t in takes if t.guide]
    if calibrate and guided:
        # 합성해 보며 맞춘다. 비교는 같은 음들끼리 해야 한다. 숨 섞임과 밝기는
        # 모음과 음높이에 따라 달라서, 녹음 전체 평균과 연습 하나의 합성값을
        # 비교하면 엉뚱한 값이 나온다. 그래서 가장 긴 가이드 연습 하나를 골라,
        # 그 녹음에서 잰 값과 같은 연습을 합성해서 잰 값을 맞춘다.
        sample = max(guided, key=lambda t: len(t.guide))
        guide = [g for g in sample.guide
                 if not model.falsetto_from_hz or midi_to_hz(g.midi) < model.falsetto_from_hz][:8]
        if len(guide) >= 3:
            target = summarize(analyze_take(_resample(sample.signal, sample.rate),
                                            MEASURE_RATE, guide))

            def measured_with(**changes) -> VoiceReport:
                timbre = replace(model.timbre("natural"), **changes)
                return _measure_synth(guide, timbre)

            if target.hnr_db is not None:
                report_progress("숨 섞임 맞추는 중", 0.65)
                # 합성기에서 HNR 은 숨 0~0.7 사이에서 단조롭게 줄어든다 (측정으로 확인)
                model.breathiness = float(np.clip(_calibrate(
                    target.hnr_db, 0.0, 0.7,
                    lambda v: measured_with(breathiness=v).hnr_db, increasing=False), 0.0, 0.7))
                measured.append("숨 섞임")
            if target.tilt_db is not None:
                report_progress("밝기 맞추는 중", 0.78)
                model.brightness = float(np.clip(_calibrate(
                    target.tilt_db, -1.0, 1.0,
                    lambda v: measured_with(brightness=v).tilt_db, increasing=True), -1.0, 1.0))
                measured.append("밝기")
            report_progress("음 잡는 습관 맞추는 중", 0.88)
            fitted = _fit_scoop(sample, model)
            if fitted is not None:
                model.scoop_cents, model.scoop_time = fitted

    model.takes_used = len(takes)
    model.notes_used = len(report.notes)
    model.measured = measured
    text = report.summary()
    if calibrate and guided:
        report_progress("모델로 다시 불러 비교하는 중", 0.95)
        check = reproduction_check(model, max(guided, key=lambda t: len(t.guide)))
        if check:
            model.reproduction = check
            text += "\n\n" + describe_reproduction(check)
    model.report_text = text
    report_progress("완료", 1.0)
    return model, report


# 재현 비교 항목 -> (설명, 단위)
REPRODUCTION_ITEMS: dict[str, tuple[str, str]] = {
    "formant_scale": ("성도 배율", ""),
    "hnr_db": ("숨 섞임 (HNR)", "dB"),
    "tilt_db": ("밝기 (배음 기울기)", "dB"),
    "vibrato_rate": ("비브라토 빠르기", "Hz"),
    "vibrato_extent": ("비브라토 폭", "센트"),
    "attack_curve": ("음 시작 곡선", "센트"),
}


def reproduction_check(model: VoiceModel, take: "Take") -> dict[str, float]:
    """모델로 같은 연습을 불러 보고, 녹음과 측정값이 얼마나 다른지.

    설정값이 정답과 같은지는 알 수 없다 (사람 목소리에는 정답 설정이 없다).
    대신 '이 모델로 부르면 원래 녹음과 같은 측정값이 나오는가' 는 잴 수 있다.
    이것이 모델이 목소리를 얼마나 따라가는지의 실제 척도다.
    """
    guide = [g for g in take.guide
             if not model.falsetto_from_hz or midi_to_hz(g.midi) < model.falsetto_from_hz][:8]
    if len(guide) < 3:
        return {}
    original = summarize(analyze_take(_resample(take.signal, take.rate), MEASURE_RATE, guide))
    synthetic_audio = _synthesize_like(guide, model.timbre("natural"), RENDER_RATE)
    synthetic = summarize(analyze_take(_resample(synthetic_audio, RENDER_RATE),
                                       MEASURE_RATE, guide))
    result: dict[str, float] = {}
    for key in ("formant_scale", "hnr_db", "tilt_db", "vibrato_rate", "vibrato_extent"):
        a, b = getattr(original, key), getattr(synthetic, key)
        if a is not None and b is not None:
            result[key] = float(b - a)
    a_curves = _attack_curves(_resample(take.signal, take.rate, 24000), 24000, guide)
    b_curves = _attack_curves(_resample(synthetic_audio, RENDER_RATE, 24000), 24000, guide)
    if a_curves and len(a_curves) == len(b_curves):
        a, b = np.concatenate(a_curves), np.concatenate(b_curves)
        both = ~np.isnan(a) & ~np.isnan(b)
        if both.sum() >= 5:
            result["attack_curve"] = float(np.sqrt(np.mean((a[both] - b[both]) ** 2)))
    return result


def describe_reproduction(check: dict[str, float]) -> str:
    lines = ["모델로 다시 불러 본 결과 (녹음과의 차이):"]
    for key, value in check.items():
        label, unit = REPRODUCTION_ITEMS[key]
        if key == "attack_curve":
            lines.append(f"  {label}: 평균 {value:.0f}{unit} 차이")
        elif key == "formant_scale":
            lines.append(f"  {label}: {value:+.3f}")
        else:
            lines.append(f"  {label}: {value:+.1f}{unit}")
    return "\n".join(lines)
