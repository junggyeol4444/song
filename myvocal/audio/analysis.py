"""
오디오 분석 — 소리에서 정보를 꺼낸다.

6번(목소리 학습: 음역/Pitch/Vibrato)과 8번(노래 학습: BPM/Key/Chord) 이 쓴다.
검증용으로도 쓴다. 악기가 지정한 음높이를 실제로 내는지 확인하려면
믿을 수 있는 측정기가 필요하다.

단순 자기상관은 쓰지 않는다. 정규화 없이 최댓값을 찾으면 지연이 짧을수록
값이 커서, 낮은 음에서 항상 엉뚱한 답이 나온다. YIN 은 그 문제를 누적
평균 정규화로 해결한 방법이고, 음성·악기 모두에서 표준으로 쓰인다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from .buffer import AudioBuffer, AudioError
from ..music.theory import Pitch, frequency_to_midi


class AnalysisError(ValueError):
    """분석 관련 오류."""


# ==========================================================================
# 음높이
# ==========================================================================

@dataclass(frozen=True, slots=True)
class PitchEstimate:
    """한 지점의 음높이 추정 결과."""

    frequency: float        # Hz. 무성음/무음이면 0
    confidence: float       # 0 ~ 1. 낮으면 믿지 말 것
    time: float = 0.0       # 초

    @property
    def is_voiced(self) -> bool:
        return self.frequency > 0.0 and self.confidence >= 0.5

    @property
    def midi(self) -> float:
        if self.frequency <= 0:
            raise AnalysisError("무성 구간에는 음높이가 없습니다.")
        return frequency_to_midi(self.frequency)

    @property
    def nearest_pitch(self) -> Pitch:
        return Pitch.from_midi(int(round(self.midi)))

    @property
    def cents_off(self) -> float:
        """가장 가까운 평균율 음에서 몇 센트 벗어났는가. 음정 정확도 판정에 쓴다."""
        midi = self.midi
        return (midi - round(midi)) * 100.0


def yin_difference(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """YIN 의 차이 함수 d(τ).

        d(τ) = Σ_j (x[j] - x[j+τ])²

    정의대로 이중 루프를 돌면 O(W·τmax) 라 느리다. 아래처럼 풀면
    자기상관 한 번(FFT)으로 끝난다.

        d(τ) = Σx[j]² + Σx[j+τ]² - 2·Σx[j]x[j+τ]
    """
    window = frame.shape[0] - max_lag
    if window <= 0:
        raise AnalysisError(
            f"프레임이 너무 짧습니다: {frame.shape[0]}샘플, 최대 지연 {max_lag}샘플"
        )

    # Σ_j x[j]·x[j+τ] 를 FFT 로 구한다.
    # 주의: 프레임 전체의 자기상관이 아니라, 앞쪽 W 샘플과 전체의 교차상관이다.
    # 전체 자기상관을 쓰면 겹치는 구간 길이가 τ 마다 달라져서 정규화가 무너진다.
    head = frame[:window]
    size = 1
    while size < frame.shape[0] + window:
        size *= 2
    head_spectrum = np.fft.rfft(head, size)
    full_spectrum = np.fft.rfft(frame, size)
    correlation = np.fft.irfft(np.conj(head_spectrum) * full_spectrum, size)[: max_lag + 1]

    # 누적합으로 각 구간의 제곱합을 구한다
    squared = np.concatenate([[0.0], np.cumsum(frame * frame)])
    first_term = squared[window] - squared[0]
    lags = np.arange(max_lag + 1)
    second_term = squared[window + lags] - squared[lags]
    return np.maximum(first_term + second_term - 2.0 * correlation, 0.0)


def yin_pitch(
    frame: np.ndarray, sample_rate: int, fmin: float = 55.0, fmax: float = 1600.0,
    threshold: float = 0.12,
) -> PitchEstimate:
    """한 프레임의 음높이를 YIN 으로 추정한다.

    threshold 가 낮을수록 엄격하다. 0.1 ~ 0.15 가 보통이다.

    측정된 정확도 (프레임 250ms 기준):

        깨끗한 신호                27.5Hz ~ 3.5kHz 에서 오차 ±2센트 이내
        배음 있는 소리 + 잡음       SNR 10dB 까지 ±5센트 이내
        순수 사인파 + 잡음          SNR 20dB 까지 ±10센트 이내
        그 이하                    옥타브를 잘못 고를 수 있다.
                                   이 경우 confidence 가 0.5 미만으로 나오므로
                                   is_voiced 를 확인하고 쓸 것.

    프레임 하나만 보고 판단하므로 심한 잡음에서는 한계가 있다. 여러 프레임의
    시간 연속성까지 쓰는 방법(pYIN 의 Viterbi 디코딩)이 더 강하지만, 실제
    녹음은 SNR 이 20dB 를 넘는 것이 보통이라 여기서는 이 정도로 충분하다.
    """
    if fmin <= 0 or fmax <= fmin:
        raise AnalysisError(f"주파수 범위가 잘못됐습니다: {fmin} ~ {fmax}")
    if fmax >= sample_rate / 2:
        raise AnalysisError(f"fmax 가 나이퀴스트를 넘습니다: {fmax}")

    signal = np.asarray(frame, dtype=np.float64)
    signal = signal - signal.mean()
    energy = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0
    if energy < 1e-6:
        return PitchEstimate(0.0, 0.0)

    min_lag = max(2, int(math.floor(sample_rate / fmax)))
    max_lag = min(int(math.ceil(sample_rate / fmin)), signal.shape[0] // 2)
    if max_lag <= min_lag:
        return PitchEstimate(0.0, 0.0)

    difference = yin_difference(signal, max_lag)

    # 누적 평균 정규화. 이게 YIN 의 핵심이다.
    # 짧은 지연이 무조건 유리해지는 문제를 없앤다.
    normalized = np.ones_like(difference)
    running = np.cumsum(difference[1:])
    lags = np.arange(1, difference.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized[1:] = difference[1:] * lags / np.maximum(running, 1e-12)

    search = normalized[min_lag : max_lag + 1]
    if search.size == 0:
        return PitchEstimate(0.0, 0.0)

    # 기준 아래로 처음 내려간 뒤의 지역 최솟값을 고른다.
    # 전역 최솟값을 그냥 쓰면 옥타브 아래를 고르는 경우가 생긴다.
    below = np.where(search < threshold)[0]
    crossed = below.size > 0
    if crossed:
        start = int(below[0])
        index = start
        while index + 1 < search.size and search[index + 1] < search[index]:
            index += 1
    else:
        # 어느 지연도 기준 아래로 안 내려갔다. 주기성이 약하거나 잡음이 많다는 뜻이다.
        # 그래도 최솟값을 돌려주되, 신뢰도를 낮춰서 믿지 말라고 알린다.
        index = int(np.argmin(search))

    lag = min_lag + index

    # 옥타브 오류 보정.
    #
    # 주기가 T 인 신호는 2T, 3T, ... 에서도 차이 함수가 작다. 잡음이 섞이면
    # 진짜 T 에서의 값이 배수 위치보다 살짝 커져서 한 옥타브(또는 그 이상)
    # 아래로 잘못 읽는 일이 생긴다. 고음일수록 탐색 범위에 들어오는 배수가
    # 많아져서(880Hz 면 17배까지) 확률이 올라간다.
    #
    # 그래서 후보 지연의 '약수 위치'만 확인한다. 진짜 주기는 반드시 lag/k 형태다.
    # 짧은 쪽부터(k 가 큰 쪽부터) 보고, 조건을 만족하는 첫 번째를 쓴다.
    #
    # 아무 짧은 지연이나 허용하면 안 된다. 피아노처럼 소리가 줄어드는 악기는
    # 차이 함수가 전체적으로 들려서, 관계없는 짧은 지연이 기준을 통과해버린다.
    # 그러면 음이 반음쯤 높게 나온다.
    best_value = float(normalized[lag])
    limit = min(0.45, max(threshold, best_value) * 1.35)
    max_divisor = lag // max(min_lag, 1)
    for divisor in range(max_divisor, 1, -1):
        candidate = lag / divisor
        if candidate < min_lag:
            continue
        low = max(min_lag, int(candidate * 0.97))
        high = min(max_lag, int(candidate * 1.03) + 1)
        if high <= low:
            continue
        local = normalized[low:high]
        local_best = int(np.argmin(local))
        if float(local[local_best]) < limit:
            lag = low + local_best
            break

    value = float(normalized[lag])

    # 포물선 보간으로 정수 지연보다 정밀하게
    if 0 < lag < normalized.shape[0] - 1:
        y0, y1, y2 = normalized[lag - 1], normalized[lag], normalized[lag + 1]
        denominator = y0 - 2.0 * y1 + y2
        if abs(denominator) > 1e-15:
            shift = 0.5 * (y0 - y2) / denominator
            if -1.0 < shift < 1.0:
                lag = lag + shift

    if lag <= 0:
        return PitchEstimate(0.0, 0.0)
    frequency = sample_rate / lag
    if not (fmin * 0.9 <= frequency <= fmax * 1.1):
        return PitchEstimate(0.0, 0.0)
    confidence = float(np.clip(1.0 - value, 0.0, 1.0))
    if not crossed:
        # 기준을 못 넘긴 추정은 옥타브를 잘못 고를 수 있다. 확실히 낮게 표시한다.
        confidence = min(confidence, 0.45)
    return PitchEstimate(frequency, confidence)


def pitch_track(
    buffer: AudioBuffer, fmin: float = 55.0, fmax: float = 1600.0,
    frame_seconds: float = 0.093, hop_seconds: float = 0.010,
) -> list[PitchEstimate]:
    """시간에 따른 음높이 변화를 따라간다.

    프레임 길이는 가장 낮은 음의 네 주기 이상이어야 한다. 93ms 면 43Hz 까지 잡힌다.
    """
    mono = buffer.to_mono()
    rate = mono.sample_rate
    frame_length = int(round(frame_seconds * rate))
    hop_length = int(round(hop_seconds * rate))
    if frame_length < 2 or hop_length < 1:
        raise AnalysisError("프레임/홉 길이가 너무 짧습니다.")
    minimum_needed = int(math.ceil(rate / fmin)) * 4
    if frame_length < minimum_needed:
        raise AnalysisError(
            f"{fmin}Hz 를 잡으려면 프레임이 최소 {minimum_needed / rate * 1000:.0f}ms "
            f"여야 합니다 (현재 {frame_seconds * 1000:.0f}ms)."
        )

    signal = mono.data[0]
    results: list[PitchEstimate] = []
    for start in range(0, max(1, signal.shape[0] - frame_length + 1), hop_length):
        frame = signal[start : start + frame_length]
        estimate = yin_pitch(frame, rate, fmin, fmax)
        results.append(
            PitchEstimate(estimate.frequency, estimate.confidence, start / rate)
        )
    return results


def median_pitch(estimates: list[PitchEstimate], min_confidence: float = 0.5) -> float:
    """유성 구간의 중앙값 주파수. 한 음의 대표 음높이를 구할 때 쓴다.

    평균이 아니라 중앙값을 쓰는 이유: 음 시작과 끝의 불안정한 구간이
    평균을 크게 끌어당기기 때문이다.
    """
    voiced = [e.frequency for e in estimates if e.confidence >= min_confidence and e.frequency > 0]
    if not voiced:
        return 0.0
    return float(np.median(voiced))


def measure_note_frequency(
    signal: np.ndarray, sample_rate: int, fmin: float = 25.0, fmax: float = 4000.0,
    skip_seconds: float = 0.15, length_seconds: float = 0.5,
) -> float:
    """악기가 낸 한 음의 주파수를 잰다.

    시작 부분(어택)은 음정이 불안정하므로 건너뛴다.
    """
    start = int(round(skip_seconds * sample_rate))
    length = int(round(length_seconds * sample_rate))
    segment = signal[start : start + length]
    if segment.shape[0] < length // 2:
        segment = signal[: max(1, len(signal))]
    # YIN 은 '프레임 길이 - 최대 지연' 만큼의 창으로 비교한다. 최대 지연이
    # 가장 낮은 음의 한 주기이므로, 창에 최소 두세 주기가 남으려면 프레임이
    # 네 주기는 되어야 한다. 두 주기로 잡으면 창이 한 주기밖에 안 남아
    # 저음에서 값이 크게 흔들린다.
    needed = int(math.ceil(sample_rate / fmin)) * 4
    if segment.shape[0] < needed:
        raise AnalysisError(
            f"{fmin}Hz 를 재려면 최소 {needed}샘플({needed / sample_rate * 1000:.0f}ms)이 "
            f"필요한데 {segment.shape[0]}샘플뿐입니다."
        )
    estimates = []
    # 창이 길수록 정밀해진다. 구간이 허락하는 만큼 길게 잡되 최소 needed 는 지킨다.
    frame_length = min(segment.shape[0], max(needed, int(0.25 * sample_rate)))
    hop = max(1, frame_length // 8)
    for begin in range(0, segment.shape[0] - frame_length + 1, hop):
        estimate = yin_pitch(segment[begin : begin + frame_length], sample_rate, fmin, fmax)
        if estimate.frequency > 0:
            estimates.append(estimate)
    if not estimates:
        return 0.0
    # 믿을 만한 프레임이 있으면 그것만 쓴다. 하나도 없으면 전부 쓰되,
    # 이 경우 값이 틀릴 수 있다 (신뢰도를 보려면 pitch_track 을 쓸 것).
    confident = [e for e in estimates if e.confidence >= 0.5]
    return median_pitch(confident if confident else estimates, min_confidence=0.0)


# ==========================================================================
# 음색
# ==========================================================================

def spectral_centroid(buffer: AudioBuffer, frame_seconds: float = 0.046) -> float:
    """스펙트럼 무게중심(Hz). 클수록 밝은 소리다.

    6번의 '음색' 분석과, 세기에 따라 음색이 실제로 바뀌는지 확인할 때 쓴다.
    """
    mono = buffer.to_mono()
    rate = mono.sample_rate
    frame_length = max(64, int(round(frame_seconds * rate)))
    signal = mono.data[0]
    if signal.shape[0] < frame_length:
        frame_length = signal.shape[0]
    if frame_length < 8:
        raise AnalysisError("분석하기에 너무 짧습니다.")
    hop = max(1, frame_length // 2)
    window = np.hanning(frame_length)
    freqs = np.fft.rfftfreq(frame_length, 1.0 / rate)
    values: list[float] = []
    for start in range(0, signal.shape[0] - frame_length + 1, hop):
        spectrum = np.abs(np.fft.rfft(signal[start : start + frame_length] * window))
        total = spectrum.sum()
        if total > 1e-9:
            values.append(float((spectrum * freqs).sum() / total))
    return float(np.mean(values)) if values else 0.0


def spectral_rolloff(buffer: AudioBuffer, percentile: float = 0.85) -> float:
    """에너지의 percentile 만큼이 이 주파수 아래에 있다. 밝기의 다른 척도."""
    if not (0.0 < percentile < 1.0):
        raise AnalysisError(f"백분위는 0 과 1 사이여야 합니다: {percentile}")
    mono = buffer.to_mono()
    signal = mono.data[0]
    if signal.shape[0] < 8:
        raise AnalysisError("분석하기에 너무 짧습니다.")
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.shape[0])))
    freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / mono.sample_rate)
    cumulative = np.cumsum(spectrum)
    if cumulative[-1] <= 0:
        return 0.0
    index = int(np.searchsorted(cumulative, cumulative[-1] * percentile))
    return float(freqs[min(index, freqs.shape[0] - 1)])


def harmonic_energy_ratio(
    signal: np.ndarray, sample_rate: int, fundamental: float,
    low_band: tuple[float, float] = (0.5, 3.5), high_band: tuple[float, float] = (6.0, 30.0),
) -> float:
    """기본음 배수 기준으로 고음 대역 / 저음 대역 에너지 비율.

    세게 칠수록 이 값이 커져야 실제 악기처럼 들린다. 음량만 바뀌면 일정하다.
    """
    if fundamental <= 0:
        raise AnalysisError("기본 주파수는 양수여야 합니다.")
    windowed = signal * np.hanning(signal.shape[0])
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / sample_rate)
    nyquist = sample_rate / 2.0
    low_mask = (freqs >= fundamental * low_band[0]) & (freqs < min(fundamental * low_band[1], nyquist))
    high_mask = (freqs >= fundamental * high_band[0]) & (freqs < min(fundamental * high_band[1], nyquist))
    low = float(spectrum[low_mask].sum())
    high = float(spectrum[high_mask].sum())
    return high / low if low > 1e-12 else 0.0


def find_partials(
    signal: np.ndarray, sample_rate: int, fundamental: float, count: int = 12,
    search_width: float = 0.45,
) -> list[tuple[float, float]]:
    """기본음 위에 실제로 어떤 부분음이 있는지 찾는다.

    피아노 줄은 뻣뻣해서 배음이 정수배보다 조금씩 높다(인하모니시티). 높은
    배음일수록 더 많이 밀린다. 그래서 'k배 근처'를 고정 폭으로 뒤지면 안 된다.
    폭을 넓히면 옆 배음을 잡고, 좁히면 밀려난 배음을 놓친다.

    실제 분석기가 하는 대로, 앞에서 찾은 부분음으로 다음 위치를 예측하며
    따라간다. 예측에는 줄의 뻣뻣함 모형을 쓴다.

        f_k = k · f0 · sqrt(1 + B·k²)

    search_width 는 예측 위치 기준 탐색 반경이며 기본음 간격의 배수다.
    0.45 면 이웃 부분음까지 침범하지 않는다.

    돌려주는 값은 [(주파수, 가장 센 성분 대비 dB), ...] 다.
    """
    if fundamental <= 0:
        raise AnalysisError("기본 주파수는 양수여야 합니다.")
    if not (0.0 < search_width < 0.5):
        raise AnalysisError(f"탐색 반경은 0 ~ 0.5 여야 합니다: {search_width}")
    if signal.shape[0] < 16:
        raise AnalysisError("분석하기에 너무 짧습니다.")

    windowed = signal * np.hanning(signal.shape[0])
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / sample_rate)
    peak = float(spectrum.max())
    if peak <= 0:
        return []
    nyquist = sample_rate / 2.0
    radius = fundamental * search_width

    results: list[tuple[float, float]] = []
    stiffness = 0.0     # 앞선 부분음에서 추정한 줄의 뻣뻣함
    for k in range(1, count + 1):
        predicted = k * fundamental * math.sqrt(1.0 + stiffness * k * k)
        if predicted >= nyquist * 0.98:
            break
        mask = (freqs > predicted - radius) & (freqs < predicted + radius)
        if not mask.any():
            continue
        band = spectrum[mask]
        band_freqs = freqs[mask]
        index = int(np.argmax(band))
        found = float(band_freqs[index])
        results.append((found, float(20.0 * np.log10(max(band[index], 1e-12) / peak))))
        # 찾은 위치로 뻣뻣함을 다시 추정해 다음 예측에 쓴다
        if k >= 2:
            ratio = found / (k * fundamental)
            estimated = (ratio * ratio - 1.0) / (k * k)
            if estimated > 0:
                # 급격히 튀지 않게 천천히 따라간다
                stiffness = stiffness * 0.5 + estimated * 0.5 if stiffness > 0 else estimated
    return results
