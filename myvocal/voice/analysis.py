"""
목소리 분석 — 6번.

녹음한 노래에서 이 사람의 목소리가 어떤지 잰다.

    음역        가장 낮은/높은 음, 편하게 내는 범위
    음색        포먼트 (모음마다 입 안 공명 자리), 성도 길이, 밝기
    발음        모음·음높이별로 자료가 얼마나 있는가 (커버리지)
    Pitch       가이드 음과 얼마나 맞는가 (센트)
    Breath      숨 섞인 정도 (HNR)
    Dynamics    세기의 폭
    Vibrato     빠르기(Hz)와 폭(센트), 걸리기 시작하는 시점
    Attack      음을 잡을 때 아래에서 밀어 올리는가 (스쿱)
    Release     음을 놓을 때 얼마나 빨리 사라지는가
    고음/저음    편한 범위 밖
    Falsetto    가성으로 넘어가는 음높이

측정값은 전부 신호에서 계산한다. 추측으로 채우지 않는다. 잴 수 없으면
None 으로 두고 왜 못 쟀는지 적는다.

포먼트는 LPC 로 잰다. 음높이가 350Hz 를 넘으면 배음 간격이 넓어서 LPC 가
포먼트 대신 배음을 잡는 일이 잦다. 그래서 그 위의 자료는 포먼트 계산에서
빼고, 빠졌다는 것을 결과에 남긴다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import linalg as scipy_linalg
from scipy import signal as scipy_signal

from ..audio.analysis import yin_difference, yin_pitch
from ..music.theory import Pitch
from .korean import decompose, is_hangul_syllable
from .phonemes import DIPHTHONGS, VOWEL_FORMANTS


class VoiceAnalysisError(ValueError):
    """목소리 분석 오류."""


FMIN = 75.0          # E2 아래까지 (베이스 최저음 근처)
FMAX = 1400.0        # F6 위 (소프라노 최고음 근처)
HOP = 0.01
FRAME = 0.055        # 75Hz 네 주기
FORMANT_MAX_F0 = 350.0


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def note_name(hz: float) -> str:
    """'G4' 처럼 옥타브까지."""
    pitch = Pitch.from_midi(int(round(hz_to_midi(hz))))
    return f"{pitch.name}{pitch.octave}"


def cents(frequency: float | np.ndarray, reference: float | np.ndarray):
    return 1200.0 * np.log2(np.asarray(frequency) / np.asarray(reference))


def nucleus_vowel(syllable: str) -> str | None:
    """음절의 중심 모음. '여' -> 'ㅓ', '과' -> 'ㅏ'."""
    if not syllable or not is_hangul_syllable(syllable[0]):
        return None
    medial = decompose(syllable[0]).medial
    if medial in DIPHTHONGS:
        return DIPHTHONGS[medial][1]
    return medial if medial in VOWEL_FORMANTS else None


# ==========================================================================
# 음높이 곡선
# ==========================================================================

@dataclass(slots=True)
class Contour:
    """10ms 마다의 음높이와 크기."""

    signal: np.ndarray
    rate: int
    times: np.ndarray           # 프레임 가운데 시각
    f0: np.ndarray              # Hz. 무성이면 nan
    confidence: np.ndarray
    level_db: np.ndarray        # 프레임 RMS (dBFS)

    @property
    def voiced(self) -> np.ndarray:
        return ~np.isnan(self.f0)

    def index_at(self, seconds: float) -> int:
        return int(np.clip(round((seconds - FRAME / 2) / HOP), 0, len(self.times) - 1))

    def frame(self, index: int, length: float = FRAME) -> np.ndarray:
        center = int(self.times[index] * self.rate)
        half = int(length * self.rate / 2)
        return self.signal[max(0, center - half): center + half]


def _octave_check(frame: np.ndarray, rate: int, f0: float) -> float:
    """한 옥타브 위로 잘못 읽었는지 스펙트럼으로 확인한다.

    '아' 를 높게 부르면 첫 포먼트가 둘째 배음을 크게 키워서, 음높이 추정기가
    둘째 배음을 기본음으로 읽는 일이 생긴다 (B4 를 B5 로). 진짜 기본음이 f/2
    라면 f/2 와 3f/2 자리에도 배음 봉우리가 있어야 한다. 둘 다 뚜렷한 봉우리로
    있으면 f/2 로 고친다. 잡음을 봉우리로 착각하지 않게, 주변 중앙값보다
    6dB 이상 솟은 것만 봉우리로 본다.
    """
    half = f0 / 2.0
    if half < FMIN:
        return f0
    size = 1 << int(math.ceil(math.log2(len(frame) * 4)))
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), size))
    freqs = np.fft.rfftfreq(size, 1.0 / rate)

    def peak(center: float) -> float:
        band = (freqs > center * 0.96) & (freqs < center * 1.04)
        return float(spectrum[band].max()) if band.any() else 0.0

    def floor(low: float, high: float) -> float:
        band = (freqs > low) & (freqs < high)
        return float(np.median(spectrum[band])) + 1e-12 if band.any() else 1e-12

    main = peak(f0) + 1e-12
    sub = peak(half)
    third = peak(1.5 * f0)
    sub_floor = floor(half * 0.6, half * 0.85)
    third_floor = floor(f0 * 1.15, f0 * 1.35)
    # 잡음 스펙트럼의 최댓값은 중앙값의 2~3배쯤은 우연히 나온다. 12dB(4배) 넘게
    # 솟아야 진짜 배음 봉우리로 본다.
    if (sub / sub_floor > 4.0 and third / third_floor > 4.0
            and sub / main > 10 ** (-30 / 20)):
        return half
    return f0


def _nearest_candidate(frame: np.ndarray, rate: int, target: float,
                       window_cents: float = 300.0) -> tuple[float, float] | None:
    """목표 음 근처(±3반음)에서 주기성이 가장 뚜렷한 지연을 고른다. (Hz, 신뢰도)"""
    x = frame - frame.mean()
    if float(np.sqrt(np.mean(x * x))) < 1e-5:
        return None
    # 목표 음의 0.6배 ~ 4배 대역만 남긴다. 숨소리의 높은 잡음이 주기 찾기를 흐린다.
    band_low = max(30.0, target * 0.6)
    band_high = min(rate * 0.45, target * 4.0)
    sos = scipy_signal.butter(2, [band_low, band_high], btype="bandpass", fs=rate, output="sos")
    x = scipy_signal.sosfiltfilt(sos, x)
    low_hz = target * 2 ** (-window_cents / 1200)
    high_hz = target * 2 ** (window_cents / 1200)
    min_lag = max(2, int(math.floor(rate / high_hz)))
    max_lag = min(int(math.ceil(rate / low_hz)) + 1, len(x) // 2)
    if max_lag <= min_lag + 2:
        return None
    difference = yin_difference(x, max_lag + 1)
    normalized = np.ones_like(difference)
    running = np.cumsum(difference[1:])
    lags = np.arange(1, difference.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized[1:] = difference[1:] * lags / np.maximum(running, 1e-12)
    search = normalized[min_lag:max_lag + 1]
    best = int(np.argmin(search))
    lag = float(min_lag + best)
    value = float(search[best])
    if value > 0.5:
        return None                 # 목표 근처에 주기가 없다 (쉬는 곳이거나 딴 음)
    i = int(lag)
    if 0 < i < normalized.shape[0] - 1:
        y0, y1, y2 = normalized[i - 1], normalized[i], normalized[i + 1]
        denominator = y0 - 2.0 * y1 + y2
        if abs(denominator) > 1e-15:
            shift = 0.5 * (y0 - y2) / denominator
            if -1.0 < shift < 1.0:
                lag = i + shift
    return rate / lag, float(np.clip(1.0 - value, 0.0, 1.0))


def extract_contour(signal: np.ndarray, rate: int,
                    expected=None) -> Contour:
    """음높이 곡선을 뽑는다. 조용한 곳과 믿을 수 없는 곳은 무성으로 둔다.

    expected 는 시각(초) -> 기대 음높이(Hz 또는 None) 함수다. 가이드 연습일 때 준다.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise VoiceAnalysisError("모노 신호여야 합니다.")
    length = int(FRAME * rate)
    hop = int(HOP * rate)
    if signal.shape[0] < length:
        raise VoiceAnalysisError(f"녹음이 너무 짧습니다 ({signal.shape[0] / rate:.2f}초).")
    starts = np.arange(0, signal.shape[0] - length + 1, hop)
    f0 = np.full(starts.shape[0], np.nan)
    confidence = np.zeros(starts.shape[0])
    level = np.full(starts.shape[0], -120.0)
    for index, start in enumerate(starts):
        frame = signal[start:start + length]
        rms = float(np.sqrt(np.mean(frame * frame)))
        level[index] = 20.0 * math.log10(rms) if rms > 1e-9 else -120.0
        if rms < 1e-5:
            continue
        center = (start + length / 2) / rate
        target = expected(center) if expected is not None else None
        if target:
            # 가이드 음을 알면 그 근처의 주기 후보 중에서 고른다. 숨이 많은
            # 목소리(가성, 속삭임)는 주기성이 약해서 일반 기준으로는 음을 통째로
            # 놓치거나 배음을 기본음으로 읽는다. 무엇을 불렀는지 알면 헷갈릴 일이 없다.
            found = _nearest_candidate(frame, rate, target)
            if found is not None:
                f0[index], confidence[index] = found
            continue
        estimate = yin_pitch(frame, rate, FMIN, FMAX, threshold=0.2)
        confidence[index] = estimate.confidence
        if estimate.confidence >= 0.6 and estimate.frequency > 0:
            f0[index] = estimate.frequency
    # 옥타브 위로 잘못 읽은 프레임을 바로잡는다 (가이드로 고른 프레임은 이미 맞다)
    for index in np.where(~np.isnan(f0))[0]:
        start = int(starts[index])
        if expected is not None and expected((start + length / 2) / rate):
            continue
        f0[index] = _octave_check(signal[start:start + length], rate, float(f0[index]))
    # 가장 큰 곳보다 40dB 넘게 작은 곳은 숨소리나 방 잡음이다
    floor = float(np.max(level)) - 40.0
    f0[level < floor] = np.nan
    # 한두 프레임짜리 유성 조각은 잡음이다
    voiced = ~np.isnan(f0)
    run_start = None
    for index in range(len(voiced) + 1):
        on = index < len(voiced) and voiced[index]
        if on and run_start is None:
            run_start = index
        elif not on and run_start is not None:
            if index - run_start < 4:
                f0[run_start:index] = np.nan
            run_start = None
    times = (starts + length / 2) / rate
    return Contour(signal, rate, times, f0, confidence, level)


# ==========================================================================
# 음 나누기
# ==========================================================================

@dataclass(slots=True)
class SungNote:
    """녹음 속 음 하나."""

    start: float
    end: float
    frames: slice                  # 곡선의 프레임 범위
    target_hz: float | None = None   # 가이드가 있으면 목표 음
    syllable: str = ""
    vowel: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment_notes(contour: Contour, min_seconds: float = 0.12) -> list[SungNote]:
    """가이드 없이 음을 나눈다. 쉬는 곳, 그리고 음높이가 뛰는 곳에서 자른다.

    비브라토(±30~100센트, 5~7Hz)를 음 바뀜으로 보면 안 된다. 그래서 0.2초
    (비브라토 한 주기 이상) 중앙값으로 편 곡선을 보고 자른다.
    """
    f0 = contour.f0
    voiced = ~np.isnan(f0)
    notes: list[SungNote] = []
    index = 0
    total = len(f0)
    while index < total:
        if not voiced[index]:
            index += 1
            continue
        run_end = index
        while run_end < total and voiced[run_end]:
            run_end += 1
        run = f0[index:run_end]
        midi = 69.0 + 12.0 * np.log2(run / 440.0)
        smoothed = scipy_signal.medfilt(midi, kernel_size=min(21, len(midi) // 2 * 2 + 1)) \
            if len(midi) >= 3 else midi
        cuts = [0]
        anchor = smoothed[0]
        for position in range(1, len(smoothed)):
            if abs(smoothed[position] - anchor) > 0.6 and position - cuts[-1] >= 8:
                cuts.append(position)
            if position - cuts[-1] < 8:
                anchor = smoothed[position]
            else:
                anchor = 0.9 * anchor + 0.1 * smoothed[position]
        cuts.append(len(smoothed))
        for a, b in zip(cuts, cuts[1:]):
            first, last = index + a, index + b
            start = contour.times[first] - HOP / 2
            end = contour.times[last - 1] + HOP / 2
            if end - start >= min_seconds:
                notes.append(SungNote(start, end, slice(first, last)))
        index = run_end
    return notes


# ==========================================================================
# 한 음에서 재는 것들
# ==========================================================================

def _stable_region(note: SungNote, contour: Contour) -> np.ndarray:
    """음 가운데의 안정된 부분 (앞 15%, 뒤 10% 제외)의 유성 프레임 번호."""
    frames = np.arange(note.frames.start, note.frames.stop)
    count = len(frames)
    head = max(1, int(count * 0.15))
    tail = max(1, int(count * 0.10))
    middle = frames[head:count - tail] if count > head + tail + 2 else frames
    return middle[~np.isnan(contour.f0[middle])]


def note_pitch(note: SungNote, contour: Contour) -> float:
    region = _stable_region(note, contour)
    if region.size == 0:
        return float("nan")
    return float(np.median(contour.f0[region]))


@dataclass(slots=True)
class VibratoMeasure:
    rate_hz: float
    extent_cents: float      # 사인파 진폭 (위아래 각각)
    onset_seconds: float


def measure_vibrato(note: SungNote, contour: Contour) -> VibratoMeasure | None:
    """음 하나의 비브라토. 0.6초보다 짧은 음은 한두 주기뿐이라 재지 않는다."""
    if note.duration < 0.6:
        return None
    frames = np.arange(note.frames.start, note.frames.stop)
    f0 = contour.f0[frames]
    valid = ~np.isnan(f0)
    if valid.mean() < 0.9:
        return None
    f0 = np.interp(np.arange(len(f0)), np.where(valid)[0], f0[valid])
    skip_head, skip_tail = int(0.12 / HOP), int(0.08 / HOP)
    body = f0[skip_head:len(f0) - skip_tail]
    if len(body) < int(0.4 / HOP):
        return None
    track = cents(body, float(np.median(body)))
    x = np.arange(len(track))
    track = track - np.polyval(np.polyfit(x, track, 2), x)
    frame_rate = 1.0 / HOP
    size = 1 << 12
    spectrum = np.abs(np.fft.rfft(track * np.hanning(len(track)), size))
    freqs = np.fft.rfftfreq(size, 1.0 / frame_rate)
    band = (freqs >= 3.0) & (freqs <= 9.0)
    if not band.any():
        return None
    peak = int(np.argmax(np.where(band, spectrum, 0.0)))
    power_band = float(np.sum(spectrum[band] ** 2))
    power_total = float(np.sum(spectrum[freqs > 0.5] ** 2)) + 1e-12
    if power_band / power_total < 0.5:
        return None                      # 비브라토가 없거나 흔들림이 불규칙하다
    if 0 < peak < len(spectrum) - 1:
        y0, y1, y2 = np.log(spectrum[peak - 1:peak + 2] + 1e-12)
        shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
    else:
        shift = 0.0
    rate_hz = float((peak + shift) * frame_rate / size)
    # 3~9Hz 성분만 남겨 진폭을 잰다. 사인파의 진폭 = RMS × √2
    sos = scipy_signal.butter(2, [3.0, 9.0], btype="bandpass", fs=frame_rate, output="sos")
    band_signal = scipy_signal.sosfiltfilt(sos, track)
    steady = band_signal[len(band_signal) // 3:]
    extent = float(np.sqrt(np.mean(steady ** 2)) * math.sqrt(2.0))
    # 음높이는 창 하나(실제로 평균되는 길이 W = FRAME - 1/FMIN) 동안의 평균으로
    # 잡힌다. 빠른 흔들림은 그만큼 작게 보인다. 사각 창 평균의 이득 sinc 로 되돌린다.
    effective = FRAME - 1.0 / FMIN
    x = math.pi * rate_hz * effective
    attenuation = math.sin(x) / x if x > 1e-6 else 1.0
    if attenuation > 0.3:
        extent /= attenuation

    # 시작 시점: 음 처음부터(음 잡는 40ms 만 빼고) 흔들림 크기를 보고, 안정 크기의
    # 90% 에 처음 닿는 곳. 비브라토는 0 에서 곧게 커진다고 보고 0.9 로 나눈다.
    start_skip = int(0.04 / HOP)
    whole = f0[start_skip:len(f0) - skip_tail]
    onset = None
    if len(whole) > 30:
        whole_cents = cents(whole, float(np.median(whole)))
        xs = np.arange(len(whole_cents))
        whole_cents = whole_cents - np.polyval(np.polyfit(xs, whole_cents, 2), xs)
        whole_band = scipy_signal.sosfiltfilt(sos, whole_cents)
        envelope = np.abs(scipy_signal.hilbert(whole_band))
        envelope = np.convolve(envelope, np.ones(9) / 9, mode="same")
        raw_extent = extent * attenuation if attenuation > 0.3 else extent
        reached = np.where(envelope >= 0.9 * raw_extent)[0]
        if reached.size:
            onset = (start_skip + int(reached[0])) * HOP / 0.9
    return VibratoMeasure(rate_hz, extent, onset if onset is not None else 0.3)


def harmonic_to_noise(frame: np.ndarray, rate: int, f0: float) -> float | None:
    """HNR (dB). 한 주기 뒤의 자기 자신과 얼마나 닮았는가. 숨이 섞일수록 작다."""
    period = rate / f0
    low, high = int(period * 0.97), int(math.ceil(period * 1.03)) + 1
    if high * 2 >= len(frame):
        return None
    x = frame - frame.mean()
    best = 0.0
    for lag in range(low, high + 1):
        a, b = x[:-lag], x[lag:]
        denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        if denominator <= 0:
            continue
        best = max(best, float(np.dot(a, b)) / denominator)
    best = min(best, 0.99999)
    if best <= 0.0:
        return None
    return 10.0 * math.log10(best / (1.0 - best))


def harmonic_levels(frame: np.ndarray, rate: int, f0: float, count: int = 2) -> list[float]:
    """1~count 번째 배음의 크기 (dB)."""
    window = np.hanning(len(frame))
    size = 1 << int(math.ceil(math.log2(len(frame) * 8)))
    spectrum = np.abs(np.fft.rfft(frame * window, size))
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    levels = []
    for k in range(1, count + 1):
        band = (freqs > k * f0 * 0.9) & (freqs < k * f0 * 1.1)
        levels.append(20.0 * math.log10(float(spectrum[band].max()) + 1e-12)
                      if band.any() else -120.0)
    return levels


def spectral_tilt(frame: np.ndarray, rate: int, f0: float) -> float:
    """배음만 보고 잰 2~5kHz 대 0~1kHz 의 에너지 차이 (dB). 밝은 목소리일수록 크다.

    스펙트럼 전체를 더하면 숨소리 잡음까지 들어간다. 잡음은 고역이 두꺼워서
    숨이 조금만 바뀌어도 이 값이 크게 흔들리고, 밝기와 숨을 구별할 수 없게
    된다. 그래서 배음 자리의 봉우리만 더한다.
    """
    size = 1 << int(math.ceil(math.log2(len(frame) * 4)))
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), size)) ** 2
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    low = high = 1e-20
    top = min(5000.0, rate / 2 * 0.95)
    k = 1
    while k * f0 < top:
        center = k * f0
        band = (freqs > center - f0 * 0.25) & (freqs < center + f0 * 0.25)
        if band.any():
            value = float(spectrum[band].max())
            if center < 1000.0:
                low += value
            elif center >= 2000.0:
                high += value
        k += 1
    return 10.0 * math.log10(high / low)


# ---- 포먼트 (LPC) ----

LPC_RATE = 11025


def lpc(frame: np.ndarray, order: int) -> np.ndarray:
    r = np.correlate(frame, frame, mode="full")[len(frame) - 1: len(frame) + order]
    if r[0] <= 0:
        raise VoiceAnalysisError("무음 프레임입니다.")
    r = r.copy()
    r[0] *= 1.0 + 1e-6          # 수치 안정용 아주 작은 백색 잡음
    a = scipy_linalg.solve_toeplitz(r[:order], -r[1:order + 1])
    return np.concatenate([[1.0], a])


def frame_formants(frame: np.ndarray, rate: int) -> list[float]:
    """프레임 하나의 포먼트 후보들 (Hz, 낮은 순)."""
    if rate != LPC_RATE:
        gcd = math.gcd(rate, LPC_RATE)
        frame = scipy_signal.resample_poly(frame, LPC_RATE // gcd, rate // gcd)
    x = frame - frame.mean()
    emphasis = math.exp(-2.0 * math.pi * 50.0 / LPC_RATE)
    x = np.append(x[0], x[1:] - emphasis * x[:-1]) * np.hamming(len(x))
    order = 2 + LPC_RATE // 1000
    try:
        coefficients = lpc(x, order)
    except (VoiceAnalysisError, np.linalg.LinAlgError, ValueError):
        return []
    roots = np.roots(coefficients)
    roots = roots[np.imag(roots) > 0]
    result = []
    for root in roots:
        frequency = float(np.angle(root) * LPC_RATE / (2 * math.pi))
        bandwidth = float(-LPC_RATE / math.pi * math.log(abs(root)))
        if 150.0 < frequency < 5000.0 and bandwidth < 500.0:
            result.append(frequency)
    return sorted(result)


def note_formants(note: SungNote, contour: Contour) -> tuple[float, float, float] | None:
    """음 하나의 F1~F3. 음이 너무 높으면 None."""
    region = _stable_region(note, contour)
    if region.size < 5:
        return None
    if float(np.median(contour.f0[region])) > FORMANT_MAX_F0:
        return None
    picks: list[tuple[float, float, float]] = []
    for index in region[:: 2]:
        found = frame_formants(contour.frame(index, 0.03), contour.rate)
        if len(found) >= 3:
            picks.append((found[0], found[1], found[2]))
    if len(picks) < 3:
        return None
    array = np.array(picks)
    return tuple(float(v) for v in np.median(array, axis=0))   # type: ignore[return-value]


# ==========================================================================
# 전체 분석
# ==========================================================================

@dataclass(slots=True)
class NoteMeasure:
    """음 하나의 측정 결과."""

    start: float
    duration: float
    pitch_hz: float
    level_db: float
    target_hz: float | None = None
    error_cents: float | None = None
    vowel: str | None = None
    formants: tuple[float, float, float] | None = None
    vibrato: VibratoMeasure | None = None
    hnr_db: float | None = None
    h1_h2_db: float | None = None         # 포먼트 보정 전 (H1-H2)
    h1_h2_corrected: float | None = None  # 포먼트 영향을 뺀 값 (H1*-H2*)
    tilt_db: float | None = None
    scoop_cents: float | None = None      # 시작이 목표보다 얼마나 아래(음수)/위
    attack_seconds: float | None = None   # 음을 잡기까지
    release_seconds: float | None = None  # 놓을 때 20dB 떨어지기까지
    voiced_seconds: float = 0.0


def measure_note(note: SungNote, contour: Contour) -> NoteMeasure | None:
    region = _stable_region(note, contour)
    if region.size < 3:
        return None
    pitch = float(np.median(contour.f0[region]))
    level = float(np.median(contour.level_db[region]))
    result = NoteMeasure(note.start, note.duration, pitch, level,
                         target_hz=note.target_hz, vowel=note.vowel)
    frames = np.arange(note.frames.start, note.frames.stop)
    result.voiced_seconds = float(np.sum(~np.isnan(contour.f0[frames])) * HOP)
    if note.target_hz:
        result.error_cents = float(cents(pitch, note.target_hz))

    hnrs, h12, tilts = [], [], []
    for index in region[:: 3]:
        frame = contour.frame(index, 0.04)
        f0 = float(contour.f0[index])
        value = harmonic_to_noise(frame, contour.rate, f0)
        if value is not None:
            hnrs.append(value)
        h1, h2 = harmonic_levels(contour.frame(index, 0.06), contour.rate, f0)
        h12.append(h1 - h2)
        tilts.append(spectral_tilt(contour.frame(index, 0.06), contour.rate, f0))
    result.hnr_db = float(np.median(hnrs)) if hnrs else None
    result.h1_h2_db = float(np.median(h12)) if h12 else None
    result.tilt_db = float(np.median(tilts)) if tilts else None
    result.formants = note_formants(note, contour)
    result.vibrato = measure_vibrato(note, contour)

    # 어택: 첫 30ms 의 음높이가 안정 음높이보다 얼마나 다른가, 30센트 안에 들기까지
    f0 = contour.f0[frames]
    first = f0[: max(1, int(0.03 / HOP))]
    first = first[~np.isnan(first)]
    if first.size:
        result.scoop_cents = float(cents(float(np.median(first)), pitch))
        inside = np.where(np.abs(cents(np.nan_to_num(f0, nan=1.0), pitch)) <= 30.0)[0]
        if inside.size:
            result.attack_seconds = float(inside[0] * HOP)

    # 릴리스: 끝에서 크기가 안정값보다 20dB 떨어지는 데 걸린 시간
    after = contour.level_db[note.frames.stop - 1: min(len(contour.level_db),
                                                     note.frames.stop + int(0.5 / HOP))]
    below = np.where(after <= level - 20.0)[0]
    if below.size:
        result.release_seconds = float(below[0] * HOP)
    return result


@dataclass(slots=True)
class VoiceReport:
    """녹음 여러 개를 합친 분석 결과."""

    notes: list[NoteMeasure] = field(default_factory=list)
    lowest_hz: float | None = None
    highest_hz: float | None = None
    comfortable_low_hz: float | None = None
    comfortable_high_hz: float | None = None
    vowel_formants: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    formant_scale: float | None = None
    hnr_db: float | None = None
    h1_h2_db: float | None = None
    tilt_db: float | None = None
    dynamics_db: float | None = None
    vibrato_rate: float | None = None
    vibrato_extent: float | None = None
    vibrato_onset: float | None = None
    vibrato_share: float = 0.0             # 긴 음 중 비브라토가 있었던 비율
    scoop_cents: float | None = None
    attack_seconds: float | None = None
    release_seconds: float | None = None
    pitch_error_cents: float | None = None   # 평균 절대 오차 (가이드 연습만)
    pitch_bias_cents: float | None = None    # + 높게 / - 낮게
    falsetto_from_hz: float | None = None
    notes_skipped_formants: int = 0
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines: list[str] = []

        def hz(value):
            return f"{note_name(value)} ({value:.0f}Hz)" if value else "잴 수 없음"

        lines.append(f"분석한 음 {len(self.notes)}개")
        lines.append(f"음역: {hz(self.lowest_hz)} ~ {hz(self.highest_hz)}")
        lines.append(f"편한 음역: {hz(self.comfortable_low_hz)} ~ {hz(self.comfortable_high_hz)}")
        if self.formant_scale:
            lines.append(f"성도 배율: {self.formant_scale:.2f} (1보다 크면 굵고 낮은 음색)")
        if self.vowel_formants:
            text = ", ".join(f"{v} {f[0]:.0f}/{f[1]:.0f}" for v, f in sorted(self.vowel_formants.items()))
            lines.append(f"모음 포먼트 F1/F2: {text}")
        if self.hnr_db is not None:
            lines.append(f"숨 섞임 (HNR): {self.hnr_db:.1f}dB — 작을수록 숨이 많음")
        if self.dynamics_db is not None:
            lines.append(f"세기 폭: {self.dynamics_db:.1f}dB")
        if self.vibrato_rate:
            lines.append(f"비브라토: {self.vibrato_rate:.1f}Hz, ±{self.vibrato_extent:.0f}센트, "
                         f"{self.vibrato_onset:.2f}초 뒤 시작 (긴 음의 {self.vibrato_share:.0%})")
        else:
            lines.append("비브라토: 감지 안 됨")
        if self.scoop_cents is not None:
            lines.append(f"어택: 시작 {self.scoop_cents:+.0f}센트에서 {self.attack_seconds or 0:.2f}초에 음을 잡음")
        if self.release_seconds is not None:
            lines.append(f"릴리스: {self.release_seconds:.2f}초")
        if self.pitch_error_cents is not None:
            lines.append(f"음정: 평균 {self.pitch_error_cents:.0f}센트 차이 "
                         f"({'높게' if (self.pitch_bias_cents or 0) > 0 else '낮게'} "
                         f"{abs(self.pitch_bias_cents or 0):.0f}센트 치우침)")
        lines.append(f"가성 시작: {hz(self.falsetto_from_hz) if self.falsetto_from_hz else '감지 안 됨'}")
        lines.extend(self.messages)
        return "\n".join(lines)


def _median(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.median(clean)) if clean else None


def _resonance_db(frequency: float, formants: Sequence[float]) -> float:
    """포먼트 공명이 이 주파수를 얼마나 키우는가 (dB). 직류 정규화 2극 공명기의 곱."""
    total = 0.0
    # 대역폭은 실제 성도보다 조금 넓게 잡는다. 포먼트 추정이 조금 틀려도
    # 보정값이 크게 흔들리지 않게. (모음별로 모은 포먼트를 쓰므로 많이 넓힐
    # 필요는 없다. 너무 넓히면 둘째 배음이 첫 포먼트에 걸린 것을 덜 뺀다.)
    bandwidths = (100.0, 120.0, 180.0)
    for formant, bandwidth in zip(formants, bandwidths):
        # |H(f)| = F^2 / sqrt((F^2 - f^2)^2 + (B f)^2)   (직류에서 1)
        magnitude = formant ** 2 / math.sqrt((formant ** 2 - frequency ** 2) ** 2
                                             + (bandwidth * frequency) ** 2)
        total += 20.0 * math.log10(magnitude)
    return total


def correct_h1_h2(note: NoteMeasure, formants: Sequence[float]) -> float | None:
    """H1*-H2*. 포먼트가 첫째·둘째 배음을 키운 만큼을 뺀다 (Iseli·Alwan 방식).

    보정하지 않으면 '아' 를 높게 부를 때 둘째 배음이 첫 포먼트에 걸려 커지는
    것이, 성대 쪽의 변화(가성)와 구별되지 않는다.
    """
    if note.h1_h2_db is None:
        return None
    f0 = note.pitch_hz
    return (note.h1_h2_db - _resonance_db(f0, formants) + _resonance_db(2.0 * f0, formants))


def summarize(notes: Sequence[NoteMeasure]) -> VoiceReport:
    """음 측정들을 모아 목소리 전체의 성질로."""
    report = VoiceReport(notes=list(notes))
    if not notes:
        report.messages.append("노래한 음을 찾지 못했습니다. 녹음 크기를 확인해 주세요.")
        return report

    # 음역: 0.2초 이상 버틴 음만. 스쳐 지나간 음은 음역이 아니다.
    held = [n for n in notes if n.duration >= 0.2]
    if held:
        pitches = np.array([n.pitch_hz for n in held])
        report.lowest_hz = float(pitches.min())
        report.highest_hz = float(pitches.max())
        levels = np.array([n.level_db for n in held])
        # 편한 음역: 크기가 중앙값에서 6dB 안이고 숨이 너무 많지 않은 음들의 10~90%
        good = [n for n in held if abs(n.level_db - float(np.median(levels))) <= 6.0]
        if len(good) >= 3:
            values = np.array([n.pitch_hz for n in good])
            report.comfortable_low_hz = float(np.percentile(values, 10))
            report.comfortable_high_hz = float(np.percentile(values, 90))
        else:
            report.comfortable_low_hz, report.comfortable_high_hz = report.lowest_hz, report.highest_hz

    # 포먼트: 모음마다 중앙값, 그리고 성도 배율
    by_vowel: dict[str, list[tuple[float, float, float]]] = {}
    for n in notes:
        if n.vowel and n.formants:
            by_vowel.setdefault(n.vowel, []).append(n.formants)
        elif n.vowel and n.pitch_hz > FORMANT_MAX_F0:
            report.notes_skipped_formants += 1
    ratios = []
    for vowel, values in by_vowel.items():
        array = np.median(np.array(values), axis=0)
        report.vowel_formants[vowel] = (float(array[0]), float(array[1]), float(array[2]))
        reference = VOWEL_FORMANTS[vowel]
        # F2, F3 가 성도 길이를 가장 잘 반영한다 (F1 은 입 벌림에 크게 좌우된다)
        ratios.extend([reference[1] / array[1], reference[2] / array[2]])
    if ratios:
        report.formant_scale = float(np.clip(np.median(ratios), 0.5, 1.6))
    if report.notes_skipped_formants:
        report.messages.append(
            f"음이 높은 {report.notes_skipped_formants}개 음은 포먼트 계산에서 뺐습니다 "
            f"({FORMANT_MAX_F0:.0f}Hz 이상에서는 정확히 잴 수 없습니다).")

    # 포먼트 보정한 H1-H2. 모음을 알면 그 모음의 포먼트(재지 못했으면 표 값에
    # 성도 배율을 적용)로, 모르면 중립 모음 값으로 보정한다.
    factor = 1.0 / (report.formant_scale or 1.0)
    for n in notes:
        # 음 하나의 LPC 값보다 모음별로 모은 값이 안정하다. 포먼트가 배음 근처로
        # 잘못 잡히면 보정값이 크게 흔들리기 때문이다.
        if n.vowel and n.vowel in report.vowel_formants:
            formants = report.vowel_formants[n.vowel]
        elif n.vowel:
            formants = tuple(f * factor for f in VOWEL_FORMANTS[n.vowel])
        else:
            formants = (500.0 * factor, 1500.0 * factor, 2500.0 * factor)
        n.h1_h2_corrected = correct_h1_h2(n, formants)

    levels = [n.level_db for n in notes]
    report.dynamics_db = float(np.percentile(levels, 95) - np.percentile(levels, 5)) \
        if len(levels) >= 4 else None

    long_notes = [n for n in notes if n.duration >= 0.6]
    with_vibrato = [n for n in long_notes if n.vibrato]
    if long_notes:
        report.vibrato_share = len(with_vibrato) / len(long_notes)
    if with_vibrato:
        report.vibrato_rate = _median([n.vibrato.rate_hz for n in with_vibrato])
        report.vibrato_extent = _median([n.vibrato.extent_cents for n in with_vibrato])
        report.vibrato_onset = _median([n.vibrato.onset_seconds for n in with_vibrato])

    report.scoop_cents = _median([n.scoop_cents for n in notes])
    report.attack_seconds = _median([n.attack_seconds for n in notes])
    report.release_seconds = _median([n.release_seconds for n in notes])

    guided = [n for n in notes if n.error_cents is not None]
    if guided:
        errors = np.array([n.error_cents for n in guided])
        report.pitch_error_cents = float(np.mean(np.abs(errors)))
        report.pitch_bias_cents = float(np.median(errors))

    report.falsetto_from_hz = detect_falsetto(notes)
    # 목소리 자체의 성질은 가성이 아닌 음(흉성)에서 잰다
    chest = [n for n in notes
             if report.falsetto_from_hz is None or n.pitch_hz < report.falsetto_from_hz]
    report.hnr_db = _median([n.hnr_db for n in chest])
    report.h1_h2_db = _median([n.h1_h2_corrected for n in chest])
    report.tilt_db = _median([n.tilt_db for n in chest])
    return report


def detect_falsetto(notes: Sequence[NoteMeasure]) -> float | None:
    """가성으로 넘어가는 음높이.

    가성은 성대가 얇게 떨려서 성문 흐름이 사인파에 가깝다. 그래서 첫 배음이
    둘째 배음보다 훨씬 커진다. 포먼트 영향을 뺀 H1*-H2* 를 쓴다.

    포먼트 보정을 해도 모음마다 기준이 조금씩 다르다. 그래서 모음을 아는 음은
    같은 모음의 낮은 음들(아래 절반, 흉성일 가능성이 큰 쪽)의 값을 빼고 본다.
    음 높이 순으로 늘어놓고, 아래 음들보다 8dB 이상 크고 그 위 음들도 대부분
    6dB 이상 큰 첫 지점을 찾는다. 그런 지점이 없으면 가성을 안 쓴 것이다.
    """
    measured = [n for n in notes if n.h1_h2_corrected is not None]
    if len(measured) < 6:
        return None
    by_vowel: dict[str, list[NoteMeasure]] = {}
    for n in measured:
        if n.vowel:
            by_vowel.setdefault(n.vowel, []).append(n)
    baseline: dict[str, float] = {}
    for vowel, group in by_vowel.items():
        if len(group) >= 2:
            ordered = sorted(group, key=lambda n: n.pitch_hz)
            lower = ordered[: max(1, len(ordered) // 2)]
            baseline[vowel] = float(np.median([n.h1_h2_corrected for n in lower]))
    usable = [n for n in measured if not n.vowel or n.vowel in baseline]
    ordered = sorted(usable, key=lambda n: n.pitch_hz)
    if len(ordered) < 6:
        return None
    values = np.array([n.h1_h2_corrected - baseline.get(n.vowel or "", 0.0) for n in ordered])
    # 경계 위에 음이 두 개 이상 있어야 한다. 마지막 음 하나만 튄 것은 가성이 아니다.
    for index in range(3, len(ordered) - 1):
        below = float(np.median(values[:index]))
        above = values[index:]
        if values[index] - below >= 8.0 and float(np.mean(above - below >= 6.0)) >= 0.8:
            return float(ordered[index].pitch_hz)
    return None


# ==========================================================================
# 가이드 연습 녹음
# ==========================================================================

@dataclass(slots=True)
class GuideNote:
    """연습에서 부르라고 한 음."""

    start: float
    duration: float
    midi: int
    syllable: str


def align_offset(contour: Contour, guide: Sequence[GuideNote],
                 search_seconds: float = 0.35) -> float:
    """녹음이 가이드보다 얼마나 늦게 들어왔는가 (초).

    장치 지연 때문에 녹음은 항상 조금 늦다. 가이드의 '소리 내는 구간' 과
    녹음의 유성 구간이 가장 잘 겹치는 어긋남을 찾는다.
    """
    voiced = (~np.isnan(contour.f0)).astype(float)
    best_offset, best_score = 0.0, -1.0
    for step in range(0, int(search_seconds / HOP) + 1):
        offset = step * HOP
        score = 0.0
        for note in guide:
            a = contour.index_at(note.start + offset)
            b = contour.index_at(note.start + note.duration + offset)
            score += float(voiced[a:b].sum())
        if score > best_score:
            best_offset, best_score = offset, score
    return best_offset


def guided_notes(contour: Contour, guide: Sequence[GuideNote],
                 offset: float | None = None) -> list[SungNote]:
    if offset is None:
        offset = align_offset(contour, guide)
    result = []
    for note in guide:
        a = contour.index_at(note.start + offset)
        b = contour.index_at(note.start + note.duration + offset) + 1
        voiced = np.where(~np.isnan(contour.f0[a:b]))[0]
        if voiced.size < 5:
            continue
        first, last = a + int(voiced[0]), a + int(voiced[-1]) + 1
        result.append(SungNote(
            start=float(contour.times[first] - HOP / 2), end=float(contour.times[last - 1] + HOP / 2),
            frames=slice(first, last), target_hz=midi_to_hz(note.midi),
            syllable=note.syllable, vowel=nucleus_vowel(note.syllable),
        ))
    return result


def analyze_take(signal: np.ndarray, rate: int,
                 guide: Sequence[GuideNote] | None = None) -> list[NoteMeasure]:
    """녹음 하나를 분석한다. 가이드가 있으면 그 음들 기준으로, 없으면 스스로 나눈다.

    가이드가 있으면 두 번 본다. 먼저 일반 방법으로 녹음이 얼마나 늦게 들어왔는지
    찾고, 그 어긋남을 반영해 각 음을 기대 음높이 근처에서 다시 잰다.
    """
    contour = extract_contour(signal, rate)
    if guide:
        offset = align_offset(contour, guide)
        windows = [(n.start + offset, n.start + n.duration + offset, midi_to_hz(n.midi))
                   for n in guide]

        def expected(seconds: float) -> float | None:
            for begin, finish, hz in windows:
                if begin <= seconds < finish:
                    return hz
            return None

        contour = extract_contour(signal, rate, expected)
        notes = guided_notes(contour, guide, offset)
    else:
        notes = segment_notes(contour)
    measures = []
    for note in notes:
        measure = measure_note(note, contour)
        if measure is not None:
            measures.append(measure)
    return measures
