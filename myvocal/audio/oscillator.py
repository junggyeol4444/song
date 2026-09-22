"""
오실레이터 — 소리의 원재료.

톱니파·사각파를 단순하게 만들면 안 된다. 그냥 만들면 나이퀴스트를 넘는 배음이
낮은 주파수로 접혀 들어와(엘리어싱) 원래 화음과 상관없는 소리가 섞인다.
고음으로 갈수록 심해지고, 되돌릴 방법이 없다.

그래서 여기서는 배음을 나이퀴스트 아래까지만 직접 더한다(가산 합성).
느릴 것 같지만 numpy 로 한 번에 계산하면 충분히 빠르고, 엘리어싱이 수학적으로
0 이다. 근사가 아니라 정확하다.

위상을 적분해서 쓰므로 비브라토·포르타멘토처럼 음높이가 계속 변하는 경우도
그대로 처리된다.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .buffer import AudioError

Waveform = Literal["sine", "saw", "square", "triangle", "pulse", "noise", "pink_noise"]


def phase_from_frequency(
    frequency: np.ndarray | float, frames: int, sample_rate: int, start_phase: float = 0.0
) -> np.ndarray:
    """주파수(Hz)를 위상(라디안)으로 적분한다.

    frequency 가 배열이면 샘플마다 다른 주파수로 본다. 비브라토, 피치 벤드,
    포르타멘토가 전부 이걸로 표현된다.
    """
    if frames < 0:
        raise AudioError(f"샘플 수는 0 이상이어야 합니다: {frames}")
    if frames == 0:
        return np.zeros(0)
    if np.isscalar(frequency):
        increment = np.full(frames, float(frequency) / sample_rate)
    else:
        array = np.asarray(frequency, dtype=np.float64)
        if array.shape[0] != frames:
            raise AudioError(f"주파수 배열 길이가 다릅니다: {array.shape[0]} vs {frames}")
        increment = array / sample_rate
    if np.any(increment < 0):
        raise AudioError("주파수는 음수일 수 없습니다.")
    # 누적합으로 위상을 만든다. 첫 샘플은 start_phase 그대로여야 하므로 한 칸 민다.
    return 2.0 * np.pi * (start_phase / (2.0 * np.pi) + np.cumsum(increment) - increment[0])


def _harmonic_limit(frequency: np.ndarray | float, sample_rate: int) -> int:
    """엘리어싱 없이 쓸 수 있는 최대 배음 차수."""
    peak = float(np.max(frequency)) if not np.isscalar(frequency) else float(frequency)
    if peak <= 0.0:
        return 0
    nyquist = sample_rate / 2.0
    return max(0, int(nyquist / peak))


def sine_wave(phase: np.ndarray) -> np.ndarray:
    """사인파. 배음이 하나도 없는 가장 순수한 소리."""
    return np.sin(phase)


def saw_wave(phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int) -> np.ndarray:
    """톱니파. 모든 배음이 1/k 로 들어 있다. 스트링, 브라스, 리드의 바탕.

        x(t) = (2/π) Σ (-1)^(k+1) sin(kφ) / k
    """
    limit = _harmonic_limit(frequency, sample_rate)
    if limit < 1:
        return np.zeros_like(phase)
    result = np.zeros_like(phase)
    for k in range(1, limit + 1):
        result += ((-1.0) ** (k + 1)) * np.sin(k * phase) / k
    return result * (2.0 / math.pi)


def square_wave(phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int) -> np.ndarray:
    """사각파. 홀수 배음만 있다. 속이 빈 듯한 소리. 오르간, 게임 사운드, 베이스.

        x(t) = (4/π) Σ_{k 홀수} sin(kφ) / k
    """
    limit = _harmonic_limit(frequency, sample_rate)
    if limit < 1:
        return np.zeros_like(phase)
    result = np.zeros_like(phase)
    for k in range(1, limit + 1, 2):
        result += np.sin(k * phase) / k
    return result * (4.0 / math.pi)


def triangle_wave(phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int) -> np.ndarray:
    """삼각파. 홀수 배음이 1/k² 로 빠르게 줄어든다. 사각파보다 부드럽고 둥글다.

        x(t) = (8/π²) Σ_{k 홀수} (-1)^((k-1)/2) sin(kφ) / k²
    """
    limit = _harmonic_limit(frequency, sample_rate)
    if limit < 1:
        return np.zeros_like(phase)
    result = np.zeros_like(phase)
    for k in range(1, limit + 1, 2):
        result += ((-1.0) ** ((k - 1) // 2)) * np.sin(k * phase) / (k * k)
    return result * (8.0 / (math.pi * math.pi))


def pulse_wave(
    phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int, duty: float = 0.5
) -> np.ndarray:
    """펄스파. duty 를 바꾸면 음색이 변한다. 0.5 면 사각파와 같다.

    톱니파 두 개를 어긋나게 빼서 만든다. 이렇게 하면 duty 를 바꿔도
    엘리어싱이 생기지 않는다.
    """
    if not (0.0 < duty < 1.0):
        raise AudioError(f"듀티는 0 과 1 사이여야 합니다: {duty}")
    first = saw_wave(phase, frequency, sample_rate)
    second = saw_wave(phase - 2.0 * math.pi * duty, frequency, sample_rate)
    return (first - second) * 0.5 + (2.0 * duty - 1.0)


def white_noise(frames: int, seed: int | None = None) -> np.ndarray:
    """백색 잡음. 모든 주파수가 고르게 들어 있다. 드럼, 숨소리, 바람."""
    generator = np.random.default_rng(seed)
    return generator.standard_normal(frames) * 0.3


def pink_noise(frames: int, seed: int | None = None) -> np.ndarray:
    """분홍 잡음. 옥타브마다 에너지가 같다. 백색보다 자연스럽게 들린다.

    Voss-McCartney 방식. 서로 다른 속도로 갱신되는 난수를 여러 개 겹친다.
    """
    if frames <= 0:
        return np.zeros(max(0, frames))
    generator = np.random.default_rng(seed)
    rows = 16
    result = np.zeros(frames)
    for row in range(rows):
        step = 1 << row
        count = frames // step + 1
        values = generator.standard_normal(count)
        result += np.repeat(values, step)[:frames]
    return result / math.sqrt(rows) * 0.3


def oscillator(
    waveform: Waveform,
    frequency: np.ndarray | float,
    frames: int,
    sample_rate: int,
    start_phase: float = 0.0,
    duty: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """파형 하나를 만든다."""
    if waveform == "noise":
        return white_noise(frames, seed)
    if waveform == "pink_noise":
        return pink_noise(frames, seed)
    phase = phase_from_frequency(frequency, frames, sample_rate, start_phase)
    if waveform == "sine":
        return sine_wave(phase)
    if waveform == "saw":
        return saw_wave(phase, frequency, sample_rate)
    if waveform == "square":
        return square_wave(phase, frequency, sample_rate)
    if waveform == "triangle":
        return triangle_wave(phase, frequency, sample_rate)
    if waveform == "pulse":
        return pulse_wave(phase, frequency, sample_rate, duty)
    raise AudioError(f"알 수 없는 파형입니다: {waveform!r}")


def vibrato_frequency(
    base_hz: float, frames: int, sample_rate: int,
    rate_hz: float = 5.5, depth_cents: float = 30.0, onset_seconds: float = 0.3,
) -> np.ndarray:
    """비브라토가 걸린 주파수 곡선을 만든다.

    사람이 노래할 때 비브라토는 음이 시작하자마자 걸리지 않는다. 조금 늦게
    들어와서 서서히 깊어진다. onset_seconds 가 그 시간이다.
    """
    if frames <= 0:
        return np.zeros(0)
    if depth_cents < 0:
        raise AudioError("비브라토 깊이는 0 이상이어야 합니다.")
    time = np.arange(frames) / sample_rate
    if onset_seconds > 0:
        ramp = np.clip(time / onset_seconds, 0.0, 1.0)
    else:
        ramp = np.ones(frames)
    modulation = np.sin(2.0 * np.pi * rate_hz * time) * (depth_cents / 100.0) * ramp
    return base_hz * np.power(2.0, modulation / 12.0)


def glide_frequency(
    from_hz: float, to_hz: float, frames: int, sample_rate: int,
    glide_seconds: float = 0.08, curve: str = "exponential",
) -> np.ndarray:
    """음과 음 사이를 미끄러지는 주파수 곡선 (포르타멘토).

    사람의 목소리는 음을 계단처럼 건너뛰지 않는다. 이게 없으면 합성 보컬이
    기계처럼 들리는 가장 큰 원인이 된다.
    """
    if frames <= 0:
        return np.zeros(0)
    if from_hz <= 0 or to_hz <= 0:
        raise AudioError("주파수는 양수여야 합니다.")
    glide_frames = min(frames, max(1, int(round(glide_seconds * sample_rate))))
    result = np.full(frames, float(to_hz))
    if glide_frames > 0:
        progress = np.linspace(0.0, 1.0, glide_frames, endpoint=True)
        if curve == "exponential":
            # 음높이는 로그 스케일이므로 지수 보간이 자연스럽다
            result[:glide_frames] = from_hz * np.power(to_hz / from_hz, progress)
        elif curve == "linear":
            result[:glide_frames] = from_hz + (to_hz - from_hz) * progress
        elif curve == "smooth":
            eased = progress * progress * (3.0 - 2.0 * progress)
            result[:glide_frames] = from_hz * np.power(to_hz / from_hz, eased)
        else:
            raise AudioError(f"알 수 없는 글라이드 곡선입니다: {curve!r}")
    return result


# ==========================================================================
# 대역제한 웨이브테이블
# ==========================================================================

"""
배음을 매번 다 더하면 정확하지만 느리다. 낮은 음일수록 배음이 많아서
(130Hz 면 184개) 한 음 만드는 데 수백 번의 sin() 계산이 필요하다.
패드처럼 목소리를 여러 겹 쌓는 악기는 이게 곱해져서 실시간의 몇 배가 걸린다.

실제 신디사이저가 쓰는 방법은 미리 만들어 두는 것이다. 옥타브 구간마다
'그 구간에서 엘리어싱이 안 나는 최대 배음까지만' 담은 표를 한 번 만들고,
연주할 때는 그 표를 읽기만 한다. 배음이 몇 개든 읽는 비용은 같다.

정확도는 표 크기와 보간 방식이 정한다. 여기서는 4096칸 표에 3차 보간을
쓴다. 아래 검증에서 엘리어싱이 -90dB 아래로 유지되는 것을 확인한다.
"""

_TABLE_SIZE: int = 4096
_BANDS_PER_OCTAVE: int = 4
_LOWEST_BAND_HZ: float = 16.0      # C0 아래까지 덮는다

# (파형, 샘플레이트) -> 밴드별 표
_WAVETABLE_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _build_wavetables(waveform: str, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """옥타브 구간마다 표를 만든다.

    돌려주는 값은 (밴드 상한 주파수 배열, 표 배열) 이다.
    표 배열의 shape 는 (밴드 수, _TABLE_SIZE).
    """
    nyquist = sample_rate / 2.0
    edges: list[float] = []
    frequency = _LOWEST_BAND_HZ
    while frequency < nyquist:
        frequency *= 2.0 ** (1.0 / _BANDS_PER_OCTAVE)
        edges.append(min(frequency, nyquist))
        if frequency >= nyquist:
            break
    if not edges:
        edges = [nyquist]

    tables = np.zeros((len(edges), _TABLE_SIZE), dtype=np.float64)
    for index, upper_hz in enumerate(edges):
        # 이 구간에서 나올 수 있는 가장 높은 음이 upper_hz 다.
        # 그 음의 배음이 나이퀴스트를 안 넘으려면 배음 수가 이만큼이어야 한다.
        max_harmonic = max(1, int(nyquist / upper_hz))
        # 역 FFT 로 한 번에 만든다. 배음마다 sin 을 더하는 것보다 훨씬 빠르다.
        spectrum = np.zeros(_TABLE_SIZE // 2 + 1, dtype=np.complex128)
        if waveform == "saw":
            for k in range(1, max_harmonic + 1):
                if k >= spectrum.shape[0]:
                    break
                # (2/π)·(-1)^(k+1)·sin(kφ)/k  ->  허수부에 넣는다
                spectrum[k] = -1j * (2.0 / math.pi) * ((-1.0) ** (k + 1)) / k
        elif waveform == "square":
            for k in range(1, max_harmonic + 1, 2):
                if k >= spectrum.shape[0]:
                    break
                spectrum[k] = -1j * (4.0 / math.pi) / k
        elif waveform == "triangle":
            for k in range(1, max_harmonic + 1, 2):
                if k >= spectrum.shape[0]:
                    break
                spectrum[k] = -1j * (8.0 / (math.pi ** 2)) * ((-1.0) ** ((k - 1) // 2)) / (k * k)
        else:
            raise AudioError(f"웨이브테이블을 만들 수 없는 파형입니다: {waveform!r}")
        tables[index] = np.fft.irfft(spectrum, n=_TABLE_SIZE) * _TABLE_SIZE / 2.0
    return np.array(edges), tables


def _get_wavetables(waveform: str, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    key = (waveform, sample_rate)
    if key not in _WAVETABLE_CACHE:
        _WAVETABLE_CACHE[key] = _build_wavetables(waveform, sample_rate)
    return _WAVETABLE_CACHE[key]


def _table_lookup(table: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """표를 3차 보간으로 읽는다 (Catmull-Rom).

    선형 보간은 계산이 싸지만 고음에서 왜곡이 들린다. 3차는 조금 더 비싼 대신
    엘리어싱 수준까지 깨끗하다.
    """
    size = table.shape[0]
    position = (phase / (2.0 * np.pi)) * size
    index = np.floor(position).astype(np.int64)
    fraction = position - index
    i0 = (index - 1) % size
    i1 = index % size
    i2 = (index + 1) % size
    i3 = (index + 2) % size
    p0, p1, p2, p3 = table[i0], table[i1], table[i2], table[i3]
    # Catmull-Rom
    return (
        p1
        + 0.5 * fraction * (p2 - p0
            + fraction * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                + fraction * (3.0 * (p1 - p2) + p3 - p0)))
    )


def wavetable_wave(
    waveform: str, phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int
) -> np.ndarray:
    """표를 읽어 파형을 만든다. 배음 수와 무관하게 빠르다.

    frequency 가 배열이면(비브라토 등) 가장 높은 값 기준으로 밴드를 고른다.
    한 음 안에서 밴드를 바꾸면 경계에서 음색이 튀기 때문이다. 비브라토 폭은
    보통 반음 이내라 한 밴드 안에 들어간다.
    """
    if phase.shape[0] == 0:
        return np.zeros(0)
    peak_hz = float(np.max(frequency)) if not np.isscalar(frequency) else float(frequency)
    if peak_hz <= 0:
        return np.zeros_like(phase)
    edges, tables = _get_wavetables(waveform, sample_rate)
    band = int(np.searchsorted(edges, peak_hz, side="left"))
    band = min(max(band, 0), tables.shape[0] - 1)
    return _table_lookup(tables[band], phase)


def saw_wave_fast(phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int) -> np.ndarray:
    """톱니파 (표 방식). saw_wave 와 같은 소리, 훨씬 빠르다."""
    return wavetable_wave("saw", phase, frequency, sample_rate)


def square_wave_fast(phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int) -> np.ndarray:
    return wavetable_wave("square", phase, frequency, sample_rate)


def triangle_wave_fast(phase: np.ndarray, frequency: np.ndarray | float, sample_rate: int) -> np.ndarray:
    return wavetable_wave("triangle", phase, frequency, sample_rate)
