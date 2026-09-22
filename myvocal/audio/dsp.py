"""
DSP — 소리를 실제로 바꾸는 처리들.

믹서(11번)와 마스터(64번)가 이걸 쓴다. 겉모양만 있는 노브가 아니라
실제로 계수를 계산하고 샘플을 처리한다.

필터 계수는 Robert Bristow-Johnson 의 Audio EQ Cookbook 공식을 쓴다.
오디오 업계 표준이고, 주파수 응답을 수학적으로 검증할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import signal as scipy_signal

from .buffer import AudioBuffer, AudioError, DType, db_to_linear, linear_to_db

FilterKind = Literal[
    "lowpass", "highpass", "bandpass", "notch", "allpass", "peaking", "lowshelf", "highshelf"
]


# ==========================================================================
# 필터
# ==========================================================================

def biquad_coefficients(
    kind: FilterKind, sample_rate: int, frequency: float, q: float = 0.7071, gain_db: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """2차 필터 계수 (b, a). Audio EQ Cookbook 공식.

    frequency  차단/중심 주파수 (Hz)
    q          대역폭. 클수록 좁고 뾰족하다. 0.7071 이 평탄한 기본값.
    gain_db    peaking / shelf 에서만 쓴다. 올리거나 내릴 양.
    """
    nyquist = sample_rate / 2.0
    if not (0.0 < frequency < nyquist):
        raise AudioError(
            f"주파수는 0 보다 크고 나이퀴스트({nyquist:.0f}Hz)보다 작아야 합니다: {frequency}Hz"
        )
    if q <= 0.0:
        raise AudioError(f"Q 는 양수여야 합니다: {q}")

    omega = 2.0 * math.pi * frequency / sample_rate
    sin_w = math.sin(omega)
    cos_w = math.cos(omega)
    alpha = sin_w / (2.0 * q)
    amplitude = 10.0 ** (gain_db / 40.0)   # shelf/peaking 은 진폭의 제곱근을 쓴다

    if kind == "lowpass":
        b = np.array([(1 - cos_w) / 2, 1 - cos_w, (1 - cos_w) / 2])
        a = np.array([1 + alpha, -2 * cos_w, 1 - alpha])
    elif kind == "highpass":
        b = np.array([(1 + cos_w) / 2, -(1 + cos_w), (1 + cos_w) / 2])
        a = np.array([1 + alpha, -2 * cos_w, 1 - alpha])
    elif kind == "bandpass":
        # 대역 안에서 최대 이득이 0dB 가 되는 형태
        b = np.array([alpha, 0.0, -alpha])
        a = np.array([1 + alpha, -2 * cos_w, 1 - alpha])
    elif kind == "notch":
        b = np.array([1.0, -2 * cos_w, 1.0])
        a = np.array([1 + alpha, -2 * cos_w, 1 - alpha])
    elif kind == "allpass":
        b = np.array([1 - alpha, -2 * cos_w, 1 + alpha])
        a = np.array([1 + alpha, -2 * cos_w, 1 - alpha])
    elif kind == "peaking":
        b = np.array([1 + alpha * amplitude, -2 * cos_w, 1 - alpha * amplitude])
        a = np.array([1 + alpha / amplitude, -2 * cos_w, 1 - alpha / amplitude])
    elif kind == "lowshelf":
        sqrt_term = 2.0 * math.sqrt(amplitude) * alpha
        b = np.array([
            amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w + sqrt_term),
            2 * amplitude * ((amplitude - 1) - (amplitude + 1) * cos_w),
            amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w - sqrt_term),
        ])
        a = np.array([
            (amplitude + 1) + (amplitude - 1) * cos_w + sqrt_term,
            -2 * ((amplitude - 1) + (amplitude + 1) * cos_w),
            (amplitude + 1) + (amplitude - 1) * cos_w - sqrt_term,
        ])
    elif kind == "highshelf":
        sqrt_term = 2.0 * math.sqrt(amplitude) * alpha
        b = np.array([
            amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w + sqrt_term),
            -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos_w),
            amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w - sqrt_term),
        ])
        a = np.array([
            (amplitude + 1) - (amplitude - 1) * cos_w + sqrt_term,
            2 * ((amplitude - 1) - (amplitude + 1) * cos_w),
            (amplitude + 1) - (amplitude - 1) * cos_w - sqrt_term,
        ])
    else:
        raise AudioError(f"알 수 없는 필터 종류입니다: {kind!r}")

    return b / a[0], a / a[0]


@dataclass(frozen=True, slots=True)
class FilterBand:
    """EQ 한 밴드."""

    kind: FilterKind
    frequency: float
    q: float = 0.7071
    gain_db: float = 0.0
    enabled: bool = True

    def coefficients(self, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        return biquad_coefficients(self.kind, sample_rate, self.frequency, self.q, self.gain_db)

    def response_db(self, frequencies: Sequence[float], sample_rate: int) -> np.ndarray:
        """지정 주파수들에서의 이득(dB). EQ 그래프 그릴 때 쓴다."""
        b, a = self.coefficients(sample_rate)
        worN = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64) / sample_rate
        _, h = scipy_signal.freqz(b, a, worN=worN)
        return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))

    def describe(self) -> str:
        names = {
            "lowpass": "로우패스", "highpass": "하이패스", "bandpass": "밴드패스",
            "notch": "노치", "allpass": "올패스", "peaking": "피킹",
            "lowshelf": "로우셸프", "highshelf": "하이셸프",
        }
        text = f"{names.get(self.kind, self.kind)} {self.frequency:.0f}Hz Q{self.q:.2f}"
        if self.kind in ("peaking", "lowshelf", "highshelf"):
            text += f" {self.gain_db:+.1f}dB"
        return text


class Equalizer:
    """여러 밴드를 직렬로 건다."""

    __slots__ = ("bands",)

    def __init__(self, bands: Sequence[FilterBand] | None = None) -> None:
        self.bands: list[FilterBand] = list(bands or [])

    def add(self, band: FilterBand) -> "Equalizer":
        self.bands.append(band)
        return self

    def process(self, buffer: AudioBuffer) -> AudioBuffer:
        active = [b for b in self.bands if b.enabled]
        if not active:
            return buffer.copy()
        signal = buffer.data.astype(np.float64)
        for band in active:
            b, a = band.coefficients(buffer.sample_rate)
            signal = scipy_signal.lfilter(b, a, signal, axis=1)
        return AudioBuffer(signal, buffer.sample_rate)

    def response_db(self, frequencies: Sequence[float], sample_rate: int) -> np.ndarray:
        """전체 합산 응답. EQ 화면 그래프용."""
        total = np.zeros(len(frequencies), dtype=np.float64)
        for band in self.bands:
            if band.enabled:
                total += band.response_db(frequencies, sample_rate)
        return total

    def describe(self) -> str:
        if not self.bands:
            return "EQ 없음"
        return " / ".join(b.describe() for b in self.bands if b.enabled)


def highpass(buffer: AudioBuffer, frequency: float, order: int = 2) -> AudioBuffer:
    """저역 차단. order 는 12dB/oct 단위 (2 = 12dB/oct, 4 = 24dB/oct)."""
    if order < 1:
        raise AudioError("차수는 1 이상이어야 합니다.")
    sos = scipy_signal.butter(
        order, frequency, btype="highpass", fs=buffer.sample_rate, output="sos"
    )
    return AudioBuffer(
        scipy_signal.sosfilt(sos, buffer.data.astype(np.float64), axis=1), buffer.sample_rate
    )


def lowpass(buffer: AudioBuffer, frequency: float, order: int = 2) -> AudioBuffer:
    sos = scipy_signal.butter(
        order, frequency, btype="lowpass", fs=buffer.sample_rate, output="sos"
    )
    return AudioBuffer(
        scipy_signal.sosfilt(sos, buffer.data.astype(np.float64), axis=1), buffer.sample_rate
    )


# ==========================================================================
# 다이내믹스
# ==========================================================================

@dataclass(frozen=True, slots=True)
class CompressorSettings:
    """컴프레서 설정.

    threshold_db  이 음량을 넘으면 누르기 시작한다
    ratio         얼마나 누를지. 4 면 4dB 넘칠 때 1dB 만 넘어가게 한다
    attack_ms     넘친 뒤 누르기 시작하는 데 걸리는 시간
    release_ms    다시 놓아주는 데 걸리는 시간
    knee_db       threshold 주변을 부드럽게 넘길 폭. 0 이면 딱 꺾인다
    makeup_db     눌러서 작아진 만큼 다시 올리는 양. None 이면 자동 계산
    """

    threshold_db: float = -18.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    knee_db: float = 6.0
    makeup_db: float | None = None
    lookahead_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.ratio < 1.0:
            raise AudioError(f"비율은 1.0 이상이어야 합니다: {self.ratio}")
        if self.attack_ms < 0 or self.release_ms < 0:
            raise AudioError("어택/릴리즈는 0 이상이어야 합니다.")
        if self.knee_db < 0:
            raise AudioError("니 폭은 0 이상이어야 합니다.")
        if self.lookahead_ms < 0:
            raise AudioError("룩어헤드는 0 이상이어야 합니다.")


def _smooth_envelope(
    level_db: np.ndarray, attack_ms: float, release_ms: float, sample_rate: int
) -> np.ndarray:
    """어택/릴리즈 시정수로 포락선을 부드럽게 만든다.

    dB 영역에서 1차 지수 평활을 한다. 올라갈 때는 어택, 내려갈 때는 릴리즈 계수를
    쓴다. 계수는 '시정수 안에 목표의 (1 - 1/e) ≈ 63.2% 까지 간다' 기준이다.
    """
    attack_coefficient = (
        math.exp(-1.0 / (attack_ms * 0.001 * sample_rate)) if attack_ms > 0 else 0.0
    )
    release_coefficient = (
        math.exp(-1.0 / (release_ms * 0.001 * sample_rate)) if release_ms > 0 else 0.0
    )
    output = np.empty_like(level_db)
    state = level_db[0]
    for i in range(level_db.shape[0]):
        target = level_db[i]
        coefficient = attack_coefficient if target > state else release_coefficient
        state = target + coefficient * (state - target)
        output[i] = state
    return output


try:  # 포락선 계산은 샘플 단위 루프라 파이썬으로는 느리다. 가능하면 컴파일한다.
    from numba import njit as _njit  # type: ignore

    _smooth_envelope = _njit(cache=True, fastmath=True)(_smooth_envelope)  # type: ignore
except Exception:  # numba 가 없으면 순수 파이썬으로 돈다 (느리지만 결과는 같다)
    pass


def compress(buffer: AudioBuffer, settings: CompressorSettings) -> tuple[AudioBuffer, np.ndarray]:
    """컴프레서. 돌려주는 값은 (처리된 오디오, 샘플별 적용 이득 dB) 다.

    이득 곡선을 같이 돌려주는 이유: 화면에 '지금 몇 dB 눌리고 있는지'를 보여줘야
    사용자가 설정을 판단할 수 있다. 숫자 없이 노브만 있으면 아무도 못 쓴다.
    """
    if buffer.frames == 0:
        return buffer.copy(), np.zeros(0)

    signal = buffer.data.astype(np.float64)
    # 검출은 채널 최대값으로 한다. 채널별로 따로 누르면 스테레오 상이 흔들린다.
    detector = np.max(np.abs(signal), axis=0)
    level_db = 20.0 * np.log10(np.maximum(detector, 1e-10))

    threshold = settings.threshold_db
    knee = settings.knee_db
    ratio = settings.ratio

    # 정적 곡선: 입력 dB -> 출력 dB
    over = level_db - threshold
    if knee > 0:
        # 니 구간 안에서는 2차 곡선으로 부드럽게 넘어간다
        below = over <= -knee / 2
        above = over >= knee / 2
        middle = ~below & ~above
        gain_db = np.zeros_like(level_db)
        gain_db[above] = (1.0 / ratio - 1.0) * over[above]
        knee_value = over[middle] + knee / 2
        gain_db[middle] = (1.0 / ratio - 1.0) * (knee_value ** 2) / (2.0 * knee)
    else:
        gain_db = np.where(over > 0, (1.0 / ratio - 1.0) * over, 0.0)

    # 시간 특성 적용. 이득은 음수이므로 부호를 뒤집어 '누르는 양'으로 평활한다.
    reduction_db = _smooth_envelope(
        -gain_db, settings.attack_ms, settings.release_ms, buffer.sample_rate
    )
    applied_db = -reduction_db

    if settings.lookahead_ms > 0:
        delay = int(round(settings.lookahead_ms * 0.001 * buffer.sample_rate))
        if delay > 0:
            # 이득 곡선을 앞당겨 신호보다 먼저 반응하게 한다
            applied_db = np.concatenate([applied_db[delay:], np.full(delay, applied_db[-1])])

    makeup = settings.makeup_db
    if makeup is None:
        # 자동 보정: threshold 를 기준점으로 잡았을 때 잃는 양을 되돌린다
        makeup = -(1.0 / ratio - 1.0) * (0.0 - threshold) * 0.5

    gain_linear = np.power(10.0, (applied_db + makeup) / 20.0)
    return AudioBuffer(signal * gain_linear, buffer.sample_rate), applied_db


def limit(
    buffer: AudioBuffer, ceiling_db: float = -0.3, lookahead_ms: float = 5.0,
    release_ms: float = 50.0, true_peak: bool = True, oversample: int = 4,
) -> tuple[AudioBuffer, float]:
    """리미터. 지정한 천장을 절대 넘지 않게 한다.

    돌려주는 값은 (처리된 오디오, 최대로 누른 dB) 다.

    룩어헤드를 쓰는 이유: 소리가 온 뒤에 반응하면 첫 순간이 이미 넘어간다.
    신호를 조금 늦추고 이득 곡선을 먼저 내려서 그 순간을 막는다.

    true_peak 를 켜면 샘플 사이에 숨은 피크까지 막는다. 디지털 샘플값이
    천장 아래여도, 스피커나 변환기가 아날로그로 되돌릴 때 그 사이에서 더
    올라갈 수 있다. MP3/AAC 로 압축하면 더 심해진다. 방송·스트리밍 규격이
    보는 값이 이것이라, 끄면 규격을 못 맞춘다.
    """
    if buffer.frames == 0:
        return buffer.copy(), 0.0
    if oversample < 1:
        raise AudioError(f"오버샘플 배수는 1 이상이어야 합니다: {oversample}")
    ceiling = db_to_linear(ceiling_db)
    signal = buffer.data.astype(np.float64)
    rate = buffer.sample_rate

    if true_peak and oversample > 1:
        # 샘플 사이를 실제로 복원해서 최대값을 본다
        upsampled = scipy_signal.resample_poly(signal, oversample, 1, axis=1)
        detector_fine = np.max(np.abs(upsampled), axis=0)
        # 오버샘플된 검출값을 원래 해상도로 되돌린다. 각 구간의 최대를 취해야
        # 사이에 있던 봉우리를 놓치지 않는다.
        usable = (detector_fine.shape[0] // oversample) * oversample
        detector = detector_fine[:usable].reshape(-1, oversample).max(axis=1)
        if detector.shape[0] < signal.shape[1]:
            detector = np.concatenate([
                detector, np.zeros(signal.shape[1] - detector.shape[0])
            ])
        else:
            detector = detector[: signal.shape[1]]
    else:
        detector = np.max(np.abs(signal), axis=0)

    lookahead_frames = max(1, int(round(lookahead_ms * 0.001 * rate)))

    # 앞으로 lookahead 구간의 최대값을 미리 본다
    padded = np.concatenate([detector, np.zeros(lookahead_frames)])
    windowed_max = scipy_signal.order_filter(
        padded, np.ones(2 * lookahead_frames + 1), 2 * lookahead_frames
    )[: detector.shape[0]]

    with np.errstate(divide="ignore", invalid="ignore"):
        needed_gain = np.where(windowed_max > ceiling, ceiling / windowed_max, 1.0)
    needed_gain_db = 20.0 * np.log10(np.maximum(needed_gain, 1e-10))

    # 내려갈 때는 즉시, 올라올 때만 release 로 천천히
    smoothed_db = _smooth_envelope(needed_gain_db, 0.0, release_ms, rate)
    smoothed_db = np.minimum(smoothed_db, needed_gain_db)   # 절대 천장을 못 넘게 한다

    delayed = np.zeros_like(signal)
    if lookahead_frames < signal.shape[1]:
        delayed[:, lookahead_frames:] = signal[:, : -lookahead_frames]
    result = delayed * np.power(10.0, smoothed_db / 20.0)

    output = AudioBuffer(result, rate)
    if true_peak and oversample > 1:
        # 보간 오차가 남아 아주 살짝 넘는 경우를 위해 한 번 더 확인한다
        actual = output.true_peak(oversample=oversample)
        if actual > ceiling and actual > 0:
            output = output.with_gain(ceiling / actual)
    else:
        result = np.clip(result, -ceiling, ceiling)
        output = AudioBuffer(result, rate)
    return output, float(-np.min(smoothed_db))


def gate(
    buffer: AudioBuffer, threshold_db: float = -50.0, attack_ms: float = 1.0,
    release_ms: float = 100.0, hold_ms: float = 10.0
) -> AudioBuffer:
    """노이즈 게이트. 기준보다 조용하면 소리를 닫는다. 녹음 잡음 정리에 쓴다."""
    if buffer.frames == 0:
        return buffer.copy()
    signal = buffer.data.astype(np.float64)
    detector = np.max(np.abs(signal), axis=0)
    level_db = 20.0 * np.log10(np.maximum(detector, 1e-10))

    open_mask = (level_db > threshold_db).astype(np.float64)
    hold_frames = int(round(hold_ms * 0.001 * buffer.sample_rate))
    if hold_frames > 0:
        # 잠깐 조용해져도 바로 닫지 않는다 (홀드)
        open_mask = scipy_signal.order_filter(
            np.concatenate([open_mask, np.zeros(hold_frames)]),
            np.ones(2 * hold_frames + 1), 2 * hold_frames
        )[: detector.shape[0]]

    target_db = np.where(open_mask > 0.5, 0.0, -80.0)
    smoothed = _smooth_envelope(target_db, attack_ms, release_ms, buffer.sample_rate)
    return AudioBuffer(signal * np.power(10.0, smoothed / 20.0), buffer.sample_rate)


def saturate(buffer: AudioBuffer, drive_db: float = 0.0, mode: str = "tanh") -> AudioBuffer:
    """새츄레이션. 아날로그 기기처럼 큰 소리를 부드럽게 둥글린다.

    배음이 생겨 소리가 두꺼워진다. 록/메탈 기타, 드럼 버스, 마스터 마무리에 쓴다.
    """
    drive = db_to_linear(drive_db)
    signal = buffer.data.astype(np.float64) * drive
    if mode == "tanh":
        result = np.tanh(signal)
        compensation = 1.0 / math.tanh(1.0) if drive_db != 0 else 1.0
    elif mode == "soft":
        # 3차 소프트클립. 2/3 까지는 선형, 그 위는 둥글게
        absolute = np.abs(signal)
        result = np.where(
            absolute < 1.0,
            signal - (signal ** 3) / 3.0,
            np.sign(signal) * (2.0 / 3.0),
        )
        compensation = 1.5
    elif mode == "hard":
        result = np.clip(signal, -1.0, 1.0)
        compensation = 1.0
    else:
        raise AudioError(f"알 수 없는 새츄레이션 방식입니다: {mode!r}")
    return AudioBuffer(result * compensation / max(drive, 1e-9), buffer.sample_rate)


# ==========================================================================
# 공간계
# ==========================================================================

# Freeverb 의 지연 길이. 44100Hz 기준 샘플 수다.
# 서로 소수에 가깝게 잡혀 있어서 반사음이 규칙적으로 겹치지 않는다.
_COMB_TUNINGS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
_ALLPASS_TUNINGS = (556, 441, 341, 225)
_STEREO_SPREAD = 23        # 오른쪽 채널의 지연을 이만큼 어긋나게 해 넓이를 만든다
_REFERENCE_RATE = 44100


def _feedback_block_size(delay: int, frames: int) -> int:
    """되먹임 지연 필터를 블록 단위로 푼다.

    y[n] 이 D 샘플 전 값에만 기대므로, 길이 D 짜리 블록 안에서는 필요한 과거 값이
    전부 '이전 블록'에 이미 확정돼 있다. 그래서 블록 단위로 한 번에 계산할 수 있다.

    이렇게 안 하고 lfilter 에 D차 계수를 통째로 넘기면 샘플마다 D번 곱셈을 해서
    O(N·D) 가 된다. 잔향은 빗살 8개 × 좌우 2채널이라 실제로 실시간보다 느려진다.
    블록 처리는 O(N) 이다.
    """
    return max(1, min(delay, frames))


def _comb_filter(signal: np.ndarray, delay: int, feedback: float, damping: float) -> np.ndarray:
    """감쇠가 들어간 빗살 필터. 잔향의 꼬리를 만든다.

    되먹임 경로에 1차 저역통과가 걸려 있어서, 반복될수록 고역이 먼저 사라진다.
    실제 방에서 벽이 고음을 더 많이 흡수하는 것과 같다.

        y[n] = b[n-D]
        f[n] = (1-damp)·y[n] + damp·f[n-1]
        b[n] = x[n] + fb·f[n]
    """
    if delay < 1:
        raise AudioError(f"지연은 1 이상이어야 합니다: {delay}")
    if not (0.0 <= feedback < 1.0):
        raise AudioError(f"되먹임은 0 이상 1 미만이어야 합니다: {feedback}")
    if not (0.0 <= damping < 1.0):
        raise AudioError(f"감쇠는 0 이상 1 미만이어야 합니다: {damping}")

    x = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    channels, frames = x.shape
    if frames == 0:
        return np.zeros_like(x) if signal.ndim > 1 else np.zeros(0)

    # store[k] 는 b[k-D] 를 담는다. 따라서 y[n] = store[n] 이다.
    store = np.zeros((channels, frames + delay))
    lowpass_b = np.array([1.0 - damping])
    lowpass_a = np.array([1.0, -damping])
    state = np.zeros((channels, 1))

    block = _feedback_block_size(delay, frames)
    for begin in range(0, frames, block):
        end = min(begin + block, frames)
        y_block = store[:, begin:end]
        f_block, state = scipy_signal.lfilter(
            lowpass_b, lowpass_a, y_block, axis=1, zi=state
        )
        store[:, begin + delay : end + delay] = x[:, begin:end] + feedback * f_block

    result = store[:, :frames]
    return result if signal.ndim > 1 else result[0]


def _allpass_filter(signal: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    """Freeverb 의 확산기. 반사음을 촘촘하게 퍼뜨려 금속성 울림을 없앤다.

        y[n] = -x[n] + b[n-D]
        b[n] =  x[n] + fb·b[n-D]
    """
    if delay < 1:
        raise AudioError(f"지연은 1 이상이어야 합니다: {delay}")
    if not (0.0 <= feedback < 1.0):
        raise AudioError(f"되먹임은 0 이상 1 미만이어야 합니다: {feedback}")

    x = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    channels, frames = x.shape
    if frames == 0:
        return np.zeros_like(x) if signal.ndim > 1 else np.zeros(0)

    store = np.zeros((channels, frames + delay))
    block = _feedback_block_size(delay, frames)
    for begin in range(0, frames, block):
        end = min(begin + block, frames)
        past = store[:, begin:end]
        store[:, begin + delay : end + delay] = x[:, begin:end] + feedback * past

    result = store[:, :frames] - x
    return result if signal.ndim > 1 else result[0]


def _resonate(signal: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    """초기 지연 없는 되먹임.  y[n] = x[n] + fb·y[n-D]

    _feedback_delay 와 달리 첫 소리가 바로 나온다. 이미 지연을 준 신호에
    반복만 더 걸고 싶을 때 쓴다 (핑퐁 딜레이).
    """
    if delay < 1:
        raise AudioError(f"지연은 1 이상이어야 합니다: {delay}")
    if not (0.0 <= feedback < 1.0):
        raise AudioError(f"되먹임은 0 이상 1 미만이어야 합니다: {feedback}")

    x = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    channels, frames = x.shape
    if frames == 0:
        return np.zeros_like(x) if signal.ndim > 1 else np.zeros(0)

    y = np.zeros((channels, frames + delay))
    block = _feedback_block_size(delay, frames)
    for begin in range(0, frames, block):
        end = min(begin + block, frames)
        y[:, begin + delay : end + delay] = x[:, begin:end] + feedback * y[:, begin:end]
    # y 배열은 D 만큼 앞이 비어 있으므로 그만큼 당겨서 돌려준다
    result = y[:, delay : delay + frames]
    return result if signal.ndim > 1 else result[0]


def _feedback_delay(signal: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    """단순 되먹임 지연.  y[n] = x[n-D] + fb·y[n-D]"""
    if delay < 1:
        raise AudioError(f"지연은 1 이상이어야 합니다: {delay}")
    if not (0.0 <= feedback < 1.0):
        raise AudioError(f"되먹임은 0 이상 1 미만이어야 합니다: {feedback}")

    x = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    channels, frames = x.shape
    if frames == 0:
        return np.zeros_like(x) if signal.ndim > 1 else np.zeros(0)

    store = np.zeros((channels, frames + delay))
    block = _feedback_block_size(delay, frames)
    for begin in range(0, frames, block):
        end = min(begin + block, frames)
        store[:, begin + delay : end + delay] = x[:, begin:end] + feedback * store[:, begin:end]

    result = store[:, :frames]
    return result if signal.ndim > 1 else result[0]


@dataclass(frozen=True, slots=True)
class ReverbSettings:
    """잔향 설정.

    room_size   공간 크기. 클수록 잔향이 길다 (0 ~ 1)
    damping     고역 흡수. 클수록 어둡고 부드럽다 (0 ~ 1)
    width       좌우 넓이 (0 = 모노, 1 = 최대)
    wet_db      잔향 소리의 양
    dry_db      원래 소리의 양
    predelay_ms 원음과 잔향 사이의 틈. 클수록 공간이 멀게 느껴진다
    """

    room_size: float = 0.5
    damping: float = 0.5
    width: float = 1.0
    wet_db: float = -12.0
    dry_db: float = 0.0
    predelay_ms: float = 20.0

    def __post_init__(self) -> None:
        for name, value in (
            ("room_size", self.room_size), ("damping", self.damping), ("width", self.width)
        ):
            if not (0.0 <= value <= 1.0):
                raise AudioError(f"{name} 은 0 ~ 1 범위여야 합니다: {value}")
        if self.predelay_ms < 0:
            raise AudioError("프리딜레이는 0 이상이어야 합니다.")


def reverb(buffer: AudioBuffer, settings: ReverbSettings | None = None) -> AudioBuffer:
    """잔향. Freeverb 구조 (빗살 8개 병렬 + 올패스 4개 직렬).

    보컬과 악기를 같은 공간에 있는 것처럼 들리게 한다. 이게 없으면 각 트랙이
    서로 다른 방에서 녹음한 것처럼 따로 논다.
    """
    settings = settings or ReverbSettings()
    if buffer.frames == 0:
        return buffer.copy()

    rate = buffer.sample_rate
    scale = rate / _REFERENCE_RATE

    stereo = buffer.to_channels(2)
    source = stereo.data.astype(np.float64)

    predelay_frames = int(round(settings.predelay_ms * 0.001 * rate))
    if predelay_frames > 0:
        delayed = np.zeros_like(source)
        delayed[:, predelay_frames:] = source[:, :-predelay_frames]
        wet_input = delayed
    else:
        wet_input = source

    # 잔향 입력은 모노로 합친 뒤 좌우에 다른 지연을 준다
    mono_input = wet_input.mean(axis=0) * 0.015    # Freeverb 의 고정 입력 이득

    feedback = 0.28 + settings.room_size * 0.7      # 0.28 ~ 0.98
    damping = settings.damping * 0.4                # 과하면 소리가 먹으므로 제한

    channels: list[np.ndarray] = []
    for channel_index in range(2):
        offset = _STEREO_SPREAD if channel_index == 1 else 0
        accumulated = np.zeros_like(mono_input)
        for tuning in _COMB_TUNINGS:
            delay = max(1, int(round((tuning + offset) * scale)))
            accumulated += _comb_filter(mono_input, delay, feedback, damping)
        for tuning in _ALLPASS_TUNINGS:
            delay = max(1, int(round((tuning + offset) * scale)))
            accumulated = _allpass_filter(accumulated, delay, 0.5)
        channels.append(accumulated)

    wet = np.vstack(channels)

    # 좌우 넓이 조절 (M/S)
    if settings.width < 1.0:
        mid = (wet[0] + wet[1]) * 0.5
        side = (wet[0] - wet[1]) * 0.5 * settings.width
        wet = np.vstack([mid + side, mid - side])

    result = source * db_to_linear(settings.dry_db) + wet * db_to_linear(settings.wet_db)
    return AudioBuffer(result, rate)


def delay_effect(
    buffer: AudioBuffer, time_ms: float, feedback: float = 0.35,
    wet_db: float = -9.0, dry_db: float = 0.0, ping_pong: bool = False,
    damping_hz: float | None = 6000.0,
) -> AudioBuffer:
    """딜레이(에코). 박자에 맞추면 리듬감이 생긴다.

    ping_pong 을 켜면 좌우를 번갈아 튀어 공간이 넓어진다.
    damping_hz 를 주면 반복될수록 고역이 깎여 자연스럽게 멀어진다.
    """
    if time_ms <= 0:
        raise AudioError(f"딜레이 시간은 양수여야 합니다: {time_ms}ms")
    if not (0.0 <= feedback < 1.0):
        raise AudioError(f"되먹임은 0 이상 1 미만이어야 합니다: {feedback}")
    if buffer.frames == 0:
        return buffer.copy()

    rate = buffer.sample_rate
    delay_frames = max(1, int(round(time_ms * 0.001 * rate)))
    stereo = buffer.to_channels(2)
    source = stereo.data.astype(np.float64)

    if ping_pong:
        mono = source.mean(axis=0)[np.newaxis, :]
        # 왼쪽은 D, 3D, 5D... / 오른쪽은 2D, 4D, 6D... 에서 울린다.
        # 먼저 각각 D / 2D 만큼 밀어 놓고, 주기 2D 로 되먹인다.
        left = _resonate(_feedback_delay(mono, delay_frames, 0.0), 2 * delay_frames, feedback)
        right = _resonate(_feedback_delay(mono, 2 * delay_frames, 0.0), 2 * delay_frames, feedback)
        wet = np.vstack([left[0], right[0]])
    else:
        wet = _feedback_delay(source, delay_frames, feedback)

    if damping_hz is not None and damping_hz < rate / 2:
        sos = scipy_signal.butter(1, damping_hz, btype="lowpass", fs=rate, output="sos")
        wet = scipy_signal.sosfilt(sos, wet, axis=1)

    result = source * db_to_linear(dry_db) + wet * db_to_linear(wet_db)
    return AudioBuffer(result, rate)


def stereo_width(buffer: AudioBuffer, width: float) -> AudioBuffer:
    """좌우 넓이. 1.0 = 그대로, 0 = 모노, 2.0 = 두 배로 벌림.

    M/S 방식이다. 가운데(보컬/베이스/킥)는 그대로 두고 양옆만 넓힌다.
    """
    if width < 0:
        raise AudioError(f"넓이는 0 이상이어야 합니다: {width}")
    stereo = buffer.to_channels(2)
    data = stereo.data.astype(np.float64)
    mid = (data[0] + data[1]) * 0.5
    side = (data[0] - data[1]) * 0.5 * width
    return AudioBuffer(np.vstack([mid + side, mid - side]), buffer.sample_rate)


def mono_below(buffer: AudioBuffer, frequency: float = 120.0, order: int = 4) -> AudioBuffer:
    """저역만 모노로 만든다.

    저음이 좌우로 벌어져 있으면 클럽 시스템과 바이닐에서 문제가 된다.
    마스터링에서 거의 항상 하는 처리다.

    좌우를 각각 저역/고역으로 나눠 저역만 합치는 방식은 쓰지 않는다. 필터가
    벽이 아니라서 고역 쪽에 남은 저역 성분이 그대로 스테레오로 남기 때문이다.
    대신 M/S 로 바꿔 '차이 성분(Side)' 만 저역 차단한다. Mid 는 손대지 않으므로
    가운데 소리는 전 대역 그대로 남고, 기준 아래에서는 차이가 사라져 모노가 된다.
    """
    if order < 1:
        raise AudioError(f"차수는 1 이상이어야 합니다: {order}")
    stereo = buffer.to_channels(2)
    rate = stereo.sample_rate
    if not (0.0 < frequency < rate / 2):
        raise AudioError(f"기준 주파수는 0 보다 크고 나이퀴스트보다 작아야 합니다: {frequency}Hz")
    data = stereo.data.astype(np.float64)
    mid = (data[0] + data[1]) * 0.5
    side = (data[0] - data[1]) * 0.5
    sos = scipy_signal.butter(order, frequency, btype="highpass", fs=rate, output="sos")
    side = scipy_signal.sosfilt(sos, side)
    return AudioBuffer(np.vstack([mid + side, mid - side]), rate)


def haas_widen(buffer: AudioBuffer, delay_ms: float = 12.0) -> AudioBuffer:
    """한쪽을 아주 조금 늦춰 넓게 들리게 한다 (하스 효과).

    30ms 안쪽이면 사람 귀는 두 소리로 듣지 않고 하나가 넓게 퍼진 것으로 듣는다.
    다만 모노로 합치면 빗살 간섭이 생기므로 과하게 쓰면 안 된다.
    """
    if not (0.0 < delay_ms <= 40.0):
        raise AudioError(f"하스 지연은 0 초과 40ms 이하여야 합니다: {delay_ms}")
    stereo = buffer.to_channels(2)
    frames = int(round(delay_ms * 0.001 * stereo.sample_rate))
    data = stereo.data.astype(np.float64)
    shifted = np.zeros_like(data)
    shifted[0] = data[0]
    if frames < data.shape[1]:
        shifted[1, frames:] = data[1, :-frames]
    return AudioBuffer(shifted, stereo.sample_rate)
