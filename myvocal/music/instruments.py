"""
악기 — 실제로 소리를 만드는 음원.

샘플 라이브러리를 쓰지 않고 합성한다. 수 GB 짜리 샘플을 같이 배포할 수 없기
때문이다. 대신 각 악기의 물리적 특징을 실제로 구현한다.

중요한 원칙 두 가지:

1. 세게 칠수록 커지기만 하는 게 아니라 '밝아진다'
   실제 악기는 세게 칠수록 고음 배음이 많아진다. 음량만 바꾸면 세게 친 소리가
   아니라 '가까이서 녹음한 소리'가 된다. 사람은 그 차이를 바로 알아챈다.

2. 배음은 같은 속도로 사라지지 않는다
   피아노 줄을 치면 고음 배음이 먼저 죽고 낮은 배음이 남는다. 모든 배음을
   같은 엔벨로프로 처리하면 오르간 같은 소리가 된다.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, ClassVar

import numpy as np
from scipy import signal as scipy_signal

from ..audio.buffer import AudioError, db_to_linear
from ..audio.envelope import ADSR, exponential_decay, percussive_envelope
from ..audio.oscillator import (
    phase_from_frequency, pink_noise, white_noise,
    saw_wave_fast as saw_wave,
    square_wave_fast as square_wave,
    triangle_wave_fast as triangle_wave,
)
from .theory import Pitch, midi_to_frequency

# 악기 분류. UI 에서 묶어 보여줄 때 쓴다.
CATEGORIES: dict[str, str] = {
    "keys": "건반",
    "guitar": "기타",
    "bass": "베이스",
    "strings": "현악",
    "brass": "금관",
    "wind": "목관",
    "synth": "신스",
    "percussion": "타악",
    "drums": "드럼",
    "choir": "합창",
}


class InstrumentError(ValueError):
    """악기 관련 오류."""


def velocity_to_amplitude(velocity: int) -> float:
    """세기(0~127) -> 진폭.

    선형이 아니다. MIDI 표준 권고대로 제곱 곡선을 쓴다. 선형으로 하면
    약하게 친 음이 거의 안 들리고 중간 세기가 뭉친다.
    """
    if not (0 <= velocity <= 127):
        raise InstrumentError(f"세기는 0~127 이어야 합니다: {velocity}")
    normalized = velocity / 127.0
    return normalized * normalized


def velocity_to_brightness(velocity: int) -> float:
    """세기 -> 배음 강조 정도 (0 ~ 1).

    세게 칠수록 고음 배음이 살아난다. 이게 없으면 세기 변화가 음량 변화로만
    들려서 연주가 밋밋해진다.
    """
    if not (0 <= velocity <= 127):
        raise InstrumentError(f"세기는 0~127 이어야 합니다: {velocity}")
    return velocity / 127.0


@dataclass(frozen=True, slots=True)
class InstrumentInfo:
    """악기 정보. 화면 표시와 편곡 판단에 쓴다."""

    name: str
    display_name: str
    category: str
    lowest_midi: int = 21
    highest_midi: int = 108
    polyphonic: bool = True
    default_velocity: int = 90

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise InstrumentError(
                f"알 수 없는 분류입니다: {self.category!r} "
                f"(사용 가능: {', '.join(CATEGORIES)})"
            )
        if not (0 <= self.lowest_midi <= self.highest_midi <= 127):
            raise InstrumentError(
                f"{self.name}: 음역이 잘못됐습니다 ({self.lowest_midi} ~ {self.highest_midi})"
            )

    @property
    def range_text(self) -> str:
        return f"{Pitch.from_midi(self.lowest_midi)} ~ {Pitch.from_midi(self.highest_midi)}"

    def contains(self, midi: int) -> bool:
        return self.lowest_midi <= midi <= self.highest_midi


class Instrument(ABC):
    """악기 하나. 음 하나를 모노 배열로 만들어 돌려준다.

    스테레오 배치, 음량, 이펙트는 여기서 하지 않는다. 그건 믹서 일이다.
    악기는 '어떤 소리인가'만 책임진다.
    """

    info: ClassVar[InstrumentInfo]

    @abstractmethod
    def render_note(
        self, midi: int, velocity: int, duration: float, sample_rate: int
    ) -> np.ndarray:
        """음 하나를 만든다. 길이는 duration 보다 길 수 있다 (여운)."""

    def tail_seconds(self, midi: int, velocity: int) -> float:
        """음을 뗀 뒤 남는 소리의 길이. 렌더러가 버퍼 길이를 잡을 때 쓴다."""
        return 0.5

    def check_range(self, midi: int) -> None:
        if not self.info.contains(midi):
            raise InstrumentError(
                f"{self.info.display_name} 의 음역을 벗어났습니다: "
                f"{Pitch.from_midi(midi)} (가능 범위 {self.info.range_text})"
            )

    def _frames(self, duration: float, sample_rate: int, tail: float) -> int:
        if duration < 0:
            raise InstrumentError(f"길이는 0 이상이어야 합니다: {duration}")
        return max(1, int(round((duration + tail) * sample_rate)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.info.display_name})"


# ==========================================================================
# 건반
# ==========================================================================

class AcousticPiano(Instrument):
    """어쿠스틱 피아노.

    실제 피아노 줄은 이상적인 줄이 아니라 뻣뻣해서, 배음이 정확히 정수배보다
    조금씩 높다(인하모니시티). 이게 피아노를 피아노처럼 들리게 하는 핵심이다.
    정수배로 만들면 오르간이나 신스에 가까운 소리가 난다.

        f_k = k · f0 · sqrt(1 + B·k²)     B 는 줄의 뻣뻣함 계수

    낮은 음일수록 줄이 굵고 짧아 B 가 크다.
    """

    info = InstrumentInfo("acoustic_piano", "어쿠스틱 피아노", "keys", 21, 108)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        # 낮은 음일수록 오래 울린다
        return float(np.interp(midi, [21, 60, 108], [6.0, 3.0, 0.8]))

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)

        # 뻣뻣함 계수. 낮은 음일수록 크다.
        stiffness = float(np.interp(midi, [21, 48, 72, 108], [4e-4, 8e-5, 2e-5, 6e-6]))
        # 전체 감쇠 반감기. 낮은 음이 훨씬 오래 간다.
        base_half_life = float(np.interp(midi, [21, 60, 108], [3.0, 1.2, 0.25]))

        time = np.arange(frames) / sample_rate
        result = np.zeros(frames, dtype=np.float64)
        nyquist = sample_rate / 2.0

        # 배음마다 '들리는 동안만' 계산한다.
        # 고음 배음은 반감기가 짧아서 몇십 ms 만에 사라지는데, 그걸 꼬리 끝까지
        # (낮은 음은 6초) 계산하면 대부분이 0 을 곱하는 헛일이다.
        silence_level = 1e-4        # -80dB 아래는 안 들린다고 본다
        for k in range(1, 41):
            partial_hz = k * fundamental * math.sqrt(1.0 + stiffness * k * k)
            if partial_hz >= nyquist * 0.98:
                break
            # 고음 배음일수록 빨리 죽는다
            half_life = base_half_life / (1.0 + 0.35 * (k - 1))
            # 세게 칠수록 고음 배음이 더 크게 시작한다
            level = (1.0 / (k ** 1.3)) * (0.35 + 0.65 * brightness) ** (0.5 * (k - 1) / 4.0)
            if level < silence_level:
                break
            audible_seconds = half_life * math.log2(level / silence_level)
            span = min(frames, max(1, int(audible_seconds * sample_rate)))
            local_time = time[:span]
            phase = 2.0 * math.pi * (k * 0.37 % 1.0)   # 배음마다 위상을 흩어 뭉침을 없앤다
            result[:span] += level * np.sin(2.0 * np.pi * partial_hz * local_time + phase) \
                * np.power(0.5, local_time / half_life)

        # 해머가 줄을 때리는 순간의 타격음. 이게 없으면 시작이 흐리다.
        hammer_frames = min(frames, int(0.012 * sample_rate))
        if hammer_frames > 0:
            noise = white_noise(hammer_frames, seed=midi * 7 + velocity)
            sos = scipy_signal.butter(
                2, min(nyquist * 0.9, 2500.0 + 3000.0 * brightness),
                btype="lowpass", fs=sample_rate, output="sos"
            )
            noise = scipy_signal.sosfilt(sos, noise)
            decay = np.exp(-np.arange(hammer_frames) / (0.0025 * sample_rate))
            result[:hammer_frames] += noise * decay * 0.18 * brightness

        # 건반을 떼면 댐퍼가 줄을 잡는다. 완전히 끊기지 않고 빠르게 죽는다.
        release_frames = int(round(duration * sample_rate))
        if 0 < release_frames < frames:
            damper = np.ones(frames)
            remaining = frames - release_frames
            damper[release_frames:] = np.exp(
                -np.arange(remaining) / (0.12 * sample_rate)
            )
            result *= damper

        peak = np.max(np.abs(result))
        if peak > 0:
            result = result / peak * amplitude
        return result


class ElectricPiano(Instrument):
    """일렉트릭 피아노 (로즈 계열).

    금속 막대(tine)를 해머로 때리고 그 진동을 픽업으로 받는다. 구조상 두 가지가
    겹쳐서 들린다.

        1. 막대의 기본 진동 — 거의 사인파에 가깝고 오래 남는다
        2. 때리는 순간의 고음 '깡' — 기본음의 10~16배 부근, 0.1초 안에 사라진다

    두 번째가 로즈를 로즈로 만든다. 다만 이걸 주 음에 '변조'로 걸면 안 된다.
    변조비 14 를 걸면 측대역이 13배·15배에 생겨서 정수 배음 구조가 깨지고,
    사람 귀에도 기계에도 음정이 다르게 들린다(실측 +125센트). 그래서 변조가
    아니라 따로 더한다.
    """

    info = InstrumentInfo("electric_piano", "일렉트릭 피아노", "keys", 28, 103)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return float(np.interp(midi, [28, 60, 103], [4.0, 2.5, 1.0]))

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        time = np.arange(frames) / sample_rate

        half_life = float(np.interp(midi, [28, 60, 103], [2.0, 1.1, 0.4]))
        body_envelope = np.power(0.5, time / half_life)

        # --- 1. 막대의 기본 진동 ---
        # 변조비 1:1 FM. 이러면 측대역이 2f, 3f, ... 정수 배음에만 생긴다.
        # 변조 지수가 시간에 따라 줄어서 처음엔 밝고 금방 부드러워진다.
        phase = 2.0 * np.pi * fundamental * time
        index = (0.9 + 1.6 * brightness) * np.exp(-time / 0.25)
        body = np.sin(phase + index * np.sin(phase))

        # --- 2. 때리는 순간의 고음 ---
        # 기본음의 정수배에 놓아 배음 구조를 깨지 않는다. 낮은 음일수록 배수가 크다.
        tine_ratio = float(np.interp(midi, [28, 60, 103], [16.0, 10.0, 6.0]))
        tine_ratio = float(round(tine_ratio))
        tine_hz = fundamental * tine_ratio
        tine = np.zeros(frames)
        if tine_hz < nyquist * 0.95:
            tine = np.sin(2.0 * np.pi * tine_hz * time) * np.exp(-time / 0.09) \
                * (0.10 + 0.30 * brightness)

        # --- 3. 해머가 막대를 때리는 소리 ---
        knock_frames = min(frames, int(0.008 * sample_rate))
        knock = np.zeros(frames)
        if knock_frames > 0:
            noise = white_noise(knock_frames, seed=midi * 19 + velocity)
            sos = scipy_signal.butter(2, min(nyquist * 0.9, 3000.0),
                                      btype="lowpass", fs=sample_rate, output="sos")
            knock[:knock_frames] = scipy_signal.sosfilt(sos, noise) * np.exp(
                -np.arange(knock_frames) / (0.0018 * sample_rate)
            ) * 0.10 * brightness

        result = (body + tine) * body_envelope + knock

        # 톤바 공명. 픽업과 공명봉이 특정 대역을 살짝 밀어준다.
        resonance_hz = min(nyquist * 0.8, max(180.0, fundamental * 2.0))
        omega = 2.0 * math.pi * resonance_hz / sample_rate
        q = 1.2
        alpha = math.sin(omega) / (2.0 * q)
        gain = 10.0 ** (2.5 / 40.0)
        b = np.array([1 + alpha * gain, -2 * math.cos(omega), 1 - alpha * gain])
        a = np.array([1 + alpha / gain, -2 * math.cos(omega), 1 - alpha / gain])
        result = scipy_signal.lfilter(b / a[0], a / a[0], result)

        release_frames = int(round(duration * sample_rate))
        if 0 < release_frames < frames:
            damper = np.ones(frames)
            damper[release_frames:] = np.exp(
                -np.arange(frames - release_frames) / (0.15 * sample_rate)
            )
            result *= damper

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


class Organ(Instrument):
    """오르간 (드로바 방식).

    피아노와 정반대다. 누르고 있는 동안 크기가 전혀 안 변한다. 감쇠가 없다.
    여러 배음을 고정 비율로 섞는다. 숫자가 드로바 설정이다.
    """

    info = InstrumentInfo("organ", "오르간", "keys", 24, 108)

    # (배음 비율, 세기). 16' 8' 5⅓' 4' 2⅔' 2' 등
    # 16' (0.5배) 는 한 옥타브 아래 소리다. 크게 넣으면 연주한 음보다 한 옥타브
    # 낮게 들려서 베이스와 부딪힌다. 두께만 더하는 정도로 제한한다.
    DRAWBARS: tuple[tuple[float, float], ...] = (
        (0.5, 0.20), (1.0, 1.00), (1.5, 0.35), (2.0, 0.70),
        (3.0, 0.25), (4.0, 0.40), (5.0, 0.15), (6.0, 0.20), (8.0, 0.30),
    )

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 0.08

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        time = np.arange(frames) / sample_rate
        nyquist = sample_rate / 2.0

        result = np.zeros(frames, dtype=np.float64)
        for ratio, level in self.DRAWBARS:
            partial_hz = fundamental * ratio
            if partial_hz >= nyquist * 0.98:
                continue
            result += level * np.sin(2.0 * np.pi * partial_hz * time + ratio)

        # 레슬리 스피커의 회전. 약한 트레몰로로 흉내낸다.
        result *= 1.0 + 0.06 * np.sin(2.0 * np.pi * 6.7 * time)

        envelope = ADSR(attack=0.008, decay=0.0, sustain=1.0, release=0.06,
                        curve="exponential", attack_curve="linear")
        gate = min(int(round(duration * sample_rate)), frames)
        curve = envelope.render(gate, sample_rate, frames - gate)
        result *= curve[:frames]

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


# ==========================================================================
# 줄을 뜯는 악기
# ==========================================================================

def karplus_strong(
    frequency: float, frames: int, sample_rate: int, damping: float = 0.5,
    brightness: float = 0.5, seed: int | None = None,
) -> np.ndarray:
    """뜯은 줄 소리. Karplus-Strong 방식.

    짧은 잡음을 지연선에 넣고, 되먹임할 때마다 살짝 저역통과를 건다.
    그러면 처음엔 밝다가 점점 부드러워지면서 줄어든다. 실제 줄이 그렇게 울린다.

    음정을 맞추려면 '루프 한 바퀴의 총 지연'이 정확히 1주기여야 한다.
    총 지연은 세 부분의 합이다.

        정수 지연 D  +  루프 저역통과의 군지연 g  +  올패스의 소수 지연 a

    루프 필터의 군지연을 빼먹고 D 와 a 만 맞추면 그만큼 주기가 길어져서
    음이 평균 5센트, 고음에서는 20센트 넘게 낮아진다. 조율이 안 맞는 기타가 된다.

    되먹임 루프는 DC 에서 이득이 1 이라, 여기 신호에 치우침이 있으면 그게
    수백 번 돌면서 남는다. 스펙트럼을 0Hz 성분이 덮어버리므로 미리 없앤다.
    """
    if frequency <= 0:
        raise InstrumentError(f"주파수는 양수여야 합니다: {frequency}")
    if frames <= 0:
        return np.zeros(max(0, frames))
    if not (0.0 <= damping <= 1.0):
        raise InstrumentError(f"감쇠는 0 ~ 1 이어야 합니다: {damping}")
    if not (0.0 <= brightness <= 1.0):
        raise InstrumentError(f"밝기는 0 ~ 1 이어야 합니다: {brightness}")

    target_delay = sample_rate / frequency
    smooth = 0.5 + 0.48 * damping          # 루프 저역통과 계수. 클수록 고역이 빨리 죽는다
    loop_group_delay = smooth              # FIR [1-s, s] 의 DC 군지연은 s 다

    remaining = target_delay - loop_group_delay
    if remaining < 2.0:
        raise InstrumentError(
            f"{frequency:.1f}Hz 는 {sample_rate}Hz 에서 뜯는 줄로 표현하기엔 너무 높습니다."
        )
    integer_delay = int(remaining)
    fractional = remaining - integer_delay
    # 올패스는 소수 지연이 0 근처일 때 오차가 크다. 0.1 아래면 한 칸 당겨 온다.
    if fractional < 0.1:
        integer_delay -= 1
        fractional += 1.0
    allpass_c = (1.0 - fractional) / (1.0 + fractional)

    # --- 초기 여기(exciter) ---
    generator = np.random.default_rng(seed)
    excitation = generator.standard_normal(integer_delay + 1)
    if brightness < 1.0:
        cutoff = 300.0 + brightness * (sample_rate / 2.0 - 400.0)
        sos = scipy_signal.butter(2, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        excitation = scipy_signal.sosfilt(sos, excitation)
    excitation -= excitation.mean()          # DC 제거
    peak = np.max(np.abs(excitation))
    if peak > 0:
        excitation /= peak

    decay_gain = 0.999 - 0.0008 * damping

    # --- 되먹임 루프를 블록 단위로 푼다 ---
    # 길이 D 짜리 블록 안에서는 필요한 과거 값이 전부 이전 블록에서 확정돼 있다.
    # 그래서 블록마다 1차 필터 두 개만 돌리면 된다. 샘플 단위 파이썬 루프보다 훨씬 빠르다.
    total = frames + integer_delay + 1
    buffer = np.zeros(total)
    buffer[: integer_delay + 1] = excitation

    lowpass_b = np.array([1.0 - smooth, smooth])
    lowpass_a = np.array([1.0])
    allpass_b = np.array([allpass_c, 1.0])
    allpass_a = np.array([1.0, allpass_c])
    lowpass_state = np.zeros(1)
    allpass_state = np.zeros(1)

    begin = integer_delay + 1
    while begin < total:
        end = min(begin + integer_delay, total)
        source = buffer[begin - integer_delay : end - integer_delay]
        filtered, lowpass_state = scipy_signal.lfilter(
            lowpass_b, lowpass_a, source, zi=lowpass_state
        )
        filtered, allpass_state = scipy_signal.lfilter(
            allpass_b, allpass_a, filtered, zi=allpass_state
        )
        buffer[begin:end] = filtered * decay_gain
        begin = end

    result = buffer[:frames]
    # 루프가 DC 에서 이득 1 이라 아주 느린 치우침이 남을 수 있다. 마지막에 막는다.
    sos_dc = scipy_signal.butter(1, max(20.0, frequency * 0.4), btype="highpass",
                                 fs=sample_rate, output="sos")
    return scipy_signal.sosfilt(sos_dc, result)


class AcousticGuitar(Instrument):
    """어쿠스틱 기타. Karplus-Strong 에 울림통 공명을 더한다."""

    info = InstrumentInfo("acoustic_guitar", "어쿠스틱 기타", "guitar", 40, 88)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return float(np.interp(midi, [40, 64, 88], [3.0, 2.0, 1.0]))

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)

        string = karplus_strong(
            midi_to_frequency(midi), frames, sample_rate,
            damping=0.45, brightness=0.35 + 0.5 * brightness, seed=midi * 13 + velocity,
        )

        # 울림통 공명. 기타 바디는 대략 100Hz 와 200Hz 부근에서 크게 울린다.
        for resonance_hz, gain_db, q in ((100.0, 5.0, 3.0), (200.0, 3.0, 4.0), (400.0, 2.0, 5.0)):
            if resonance_hz < sample_rate / 2.0 * 0.9:
                omega = 2.0 * math.pi * resonance_hz / sample_rate
                alpha = math.sin(omega) / (2.0 * q)
                gain = 10.0 ** (gain_db / 40.0)
                b = np.array([1 + alpha * gain, -2 * math.cos(omega), 1 - alpha * gain])
                a = np.array([1 + alpha / gain, -2 * math.cos(omega), 1 - alpha / gain])
                string = scipy_signal.lfilter(b / a[0], a / a[0], string)

        # 피크가 줄을 긁는 소리
        pick_frames = min(frames, int(0.006 * sample_rate))
        if pick_frames > 0:
            noise = white_noise(pick_frames, seed=midi + velocity)
            string[:pick_frames] += noise * np.exp(
                -np.arange(pick_frames) / (0.0015 * sample_rate)
            ) * 0.12 * brightness

        release_frames = int(round(duration * sample_rate))
        if 0 < release_frames < frames:
            mute = np.ones(frames)
            mute[release_frames:] = np.exp(
                -np.arange(frames - release_frames) / (0.25 * sample_rate)
            )
            string *= mute

        peak = np.max(np.abs(string))
        return string / peak * amplitude if peak > 0 else string


class ElectricGuitar(Instrument):
    """일렉트릭 기타. 뜯은 줄에 왜곡을 걸고 앰프 특성을 흉내낸다."""

    info = InstrumentInfo("electric_guitar", "일렉트릭 기타", "guitar", 40, 88)

    def __init__(self, drive: float = 0.5) -> None:
        if not (0.0 <= drive <= 1.0):
            raise InstrumentError(f"드라이브는 0 ~ 1 이어야 합니다: {drive}")
        self.drive = drive

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 2.0 + self.drive * 1.5

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)

        string = karplus_strong(
            midi_to_frequency(midi), frames, sample_rate,
            damping=0.25, brightness=0.5 + 0.4 * brightness, seed=midi * 17 + velocity,
        )

        # 왜곡. 세게 걸수록 배음이 늘고 서스테인이 길어진다.
        gain = 1.0 + self.drive * 24.0
        distorted = np.tanh(string * gain) / math.tanh(gain) if gain > 1 else string
        mixed = distorted * (0.3 + 0.7 * self.drive) + string * (0.7 - 0.7 * self.drive)

        # 기타 앰프 스피커는 5kHz 위를 거의 안 낸다. 이게 없으면 지직거린다.
        cutoff = min(sample_rate / 2.0 * 0.95, 5000.0)
        sos = scipy_signal.butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        mixed = scipy_signal.sosfilt(sos, mixed)
        sos_high = scipy_signal.butter(2, 90.0, btype="highpass", fs=sample_rate, output="sos")
        mixed = scipy_signal.sosfilt(sos_high, mixed)

        release_frames = int(round(duration * sample_rate))
        if 0 < release_frames < frames:
            mute = np.ones(frames)
            mute[release_frames:] = np.exp(
                -np.arange(frames - release_frames) / (0.18 * sample_rate)
            )
            mixed *= mute

        peak = np.max(np.abs(mixed))
        return mixed / peak * amplitude if peak > 0 else mixed


# ==========================================================================
# 베이스
# ==========================================================================

class ElectricBass(Instrument):
    """일렉트릭 베이스.

    낮은 음역은 스피커에서 기본음이 잘 안 들린다. 그래서 배음을 적당히 살려야
    작은 스피커에서도 '무슨 음인지' 들린다. 기본음만 크게 하면 큰 스피커에서는
    웅웅거리고 작은 스피커에서는 아예 안 들린다.
    """

    info = InstrumentInfo("electric_bass", "일렉트릭 베이스", "bass", 28, 67, polyphonic=False)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 1.2

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)

        string = karplus_strong(
            midi_to_frequency(midi), frames, sample_rate,
            damping=0.62, brightness=0.25 + 0.45 * brightness, seed=midi * 11 + velocity,
        )
        # 기본음을 따로 더해 저역을 든든하게 만든다
        time = np.arange(frames) / sample_rate
        fundamental = midi_to_frequency(midi)
        sub = np.sin(2.0 * np.pi * fundamental * time) * np.power(0.5, time / 0.9)
        mixed = string * 0.75 + sub * 0.45

        # 손가락이 줄을 뜯는 소리
        attack_frames = min(frames, int(0.008 * sample_rate))
        if attack_frames > 0:
            noise = white_noise(attack_frames, seed=midi * 3)
            sos = scipy_signal.butter(2, min(sample_rate / 2 * 0.9, 1800.0),
                                      btype="lowpass", fs=sample_rate, output="sos")
            mixed[:attack_frames] += scipy_signal.sosfilt(sos, noise) * np.exp(
                -np.arange(attack_frames) / (0.002 * sample_rate)
            ) * 0.15 * brightness

        # 픽업과 앰프 특성
        sos = scipy_signal.butter(3, min(sample_rate / 2 * 0.9, 4000.0),
                                  btype="lowpass", fs=sample_rate, output="sos")
        mixed = scipy_signal.sosfilt(sos, mixed)

        release_frames = int(round(duration * sample_rate))
        if 0 < release_frames < frames:
            mute = np.ones(frames)
            mute[release_frames:] = np.exp(
                -np.arange(frames - release_frames) / (0.08 * sample_rate)
            )
            mixed *= mute

        peak = np.max(np.abs(mixed))
        return mixed / peak * amplitude if peak > 0 else mixed


class SynthBass(Instrument):
    """신스 베이스. EDM / Hip-Hop / Synthwave 용.

    필터 엔벨로프가 핵심이다. 음이 시작할 때 필터가 열렸다가 닫히면서
    '뚱' 하는 특유의 소리가 난다.
    """

    info = InstrumentInfo("synth_bass", "신스 베이스", "bass", 24, 67, polyphonic=False)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 0.35

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)

        phase = phase_from_frequency(fundamental, frames, sample_rate)
        # 살짝 어긋난 톱니 두 개를 겹쳐 두께를 만든다. 기준 음 위아래로 같은
        # 만큼 벌려야 전체 음정이 안 밀린다.
        low_phase = phase_from_frequency(fundamental * 0.998, frames, sample_rate)
        high_phase = phase_from_frequency(fundamental * 1.002, frames, sample_rate)
        wave = (
            saw_wave(low_phase, fundamental, sample_rate) * 0.45
            + saw_wave(high_phase, fundamental * 1.002, sample_rate) * 0.45
            + np.sin(phase) * 0.45      # 기본음을 사인으로 보강한다.
        )
        # 옥타브 아래 오실레이터는 넣지 않는다. 연주한 음보다 한 옥타브 낮게
        # 들려서 코드와 부딪힌다. 저음의 무게는 위 사인과 필터로 만든다.

        # 필터 엔벨로프: 열렸다가 닫힌다
        time = np.arange(frames) / sample_rate
        cutoff_start = fundamental * (6.0 + 10.0 * brightness)
        cutoff_end = fundamental * 2.2
        nyquist = sample_rate / 2.0
        cutoff_start = min(cutoff_start, nyquist * 0.9)
        cutoff_end = min(cutoff_end, nyquist * 0.9)
        # 구간을 나눠 시변 필터를 근사한다. 샘플마다 계수를 다시 계산하면 너무 느리다.
        segments = 24
        result = np.zeros(frames)
        bounds = np.linspace(0, frames, segments + 1).astype(int)
        for index in range(segments):
            begin, end = bounds[index], bounds[index + 1]
            if end <= begin:
                continue
            progress = index / max(1, segments - 1)
            cutoff = max(40.0, cutoff_start * math.exp(math.log(cutoff_end / cutoff_start) * (1.0 - math.exp(-4.0 * progress))))
            sos = scipy_signal.butter(2, min(cutoff, nyquist * 0.95),
                                      btype="lowpass", fs=sample_rate, output="sos")
            # 구간 경계에서 끊기지 않도록 앞쪽을 조금 더 넣고 자른다
            pad = min(begin, 512)
            filtered = scipy_signal.sosfilt(sos, wave[begin - pad : end])
            result[begin:end] = filtered[pad:]

        envelope = ADSR(attack=0.004, decay=0.08, sustain=0.75, release=0.06)
        gate = min(int(round(duration * sample_rate)), frames)
        result *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


class SubBass(Instrument):
    """서브 베이스. 거의 사인파. 808 계열, 힙합/트랩의 저음."""

    info = InstrumentInfo("sub_bass", "서브 베이스", "bass", 21, 55, polyphonic=False)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 1.5

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        time = np.arange(frames) / sample_rate

        # 808 은 시작할 때 음이 살짝 위에서 떨어진다. 이게 '펀치'를 만든다.
        pitch_drop = fundamental * (1.0 + 0.7 * np.exp(-time / 0.02))
        phase = phase_from_frequency(pitch_drop, frames, sample_rate)
        wave = np.sin(phase)
        # 아주 약한 배음으로 작은 스피커에서도 음정이 들리게 한다
        wave += np.sin(2.0 * phase) * 0.12 + np.sin(3.0 * phase) * 0.05

        envelope = ADSR(attack=0.002, decay=0.5, sustain=0.55, release=0.35)
        gate = min(int(round(duration * sample_rate)), frames)
        wave *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(wave))
        return wave / peak * amplitude if peak > 0 else wave


# ==========================================================================
# 현악 / 관악
# ==========================================================================

class Strings(Instrument):
    """스트링 앙상블.

    여러 연주자가 같이 켜면 음정과 타이밍이 미세하게 다 다르다. 그 어긋남이
    스트링 특유의 두꺼움을 만든다. 하나만 쓰면 얇은 신스 소리가 난다.
    """

    info = InstrumentInfo("strings", "스트링 앙상블", "strings", 36, 96)

    PLAYERS: int = 7

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 0.9

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        generator = np.random.default_rng(midi * 31 + velocity)

        result = np.zeros(frames)
        for player in range(self.PLAYERS):
            # 연주자마다 음정이 조금씩 다르다 (±8센트)
            detune_cents = generator.uniform(-8.0, 8.0)
            player_hz = fundamental * (2.0 ** (detune_cents / 1200.0))
            # 비브라토도 속도와 깊이가 다르다
            rate_hz = generator.uniform(4.6, 6.2)
            depth = generator.uniform(12.0, 26.0)
            onset = generator.uniform(0.15, 0.4)
            time = np.arange(frames) / sample_rate
            ramp = np.clip(time / onset, 0.0, 1.0)
            modulation = np.sin(2.0 * np.pi * rate_hz * time + generator.uniform(0, 6.28)) \
                * (depth / 100.0) * ramp
            frequency = player_hz * np.power(2.0, modulation / 12.0)
            phase = phase_from_frequency(frequency, frames, sample_rate,
                                         start_phase=generator.uniform(0, 6.28))
            result += saw_wave(phase, float(np.max(frequency)), sample_rate) / self.PLAYERS

        # 활이 현을 긁는 소리
        bow_noise = pink_noise(frames, seed=midi + velocity) * 0.04 * brightness
        sos = scipy_signal.butter(2, min(nyquist * 0.9, 3000.0),
                                  btype="highpass", fs=sample_rate, output="sos")
        result += scipy_signal.sosfilt(sos, bow_noise)

        # 현악기 몸통은 아주 높은 배음을 별로 안 낸다
        cutoff = min(nyquist * 0.9, 3500.0 + 4000.0 * brightness)
        sos = scipy_signal.butter(2, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        result = scipy_signal.sosfilt(sos, result)

        envelope = ADSR(attack=0.13, decay=0.25, sustain=0.85, release=0.7,
                        attack_curve="linear")
        gate = min(int(round(duration * sample_rate)), frames)
        result *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


class Brass(Instrument):
    """금관. 세게 불수록 배음이 폭발적으로 늘어난다.

    이 성질이 금관을 금관처럼 만든다. 크게 부는 것과 밝게 부는 것이 분리되지
    않는 악기다. 세기만 올리고 음색이 그대로면 트럼펫으로 안 들린다.
    """

    info = InstrumentInfo("brass", "금관", "brass", 40, 84)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 0.35

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        time = np.arange(frames) / sample_rate

        # 입술 떨림에서 오는 미세한 음정 흔들림
        wobble = 1.0 + 0.0016 * np.sin(2.0 * np.pi * 5.2 * time) * np.clip(time / 0.25, 0, 1)
        phase = phase_from_frequency(fundamental * wobble, frames, sample_rate)
        wave = saw_wave(phase, fundamental * 1.01, sample_rate)

        # 필터가 열리면서 배음이 늘어난다 (브라스 특유의 '밀고 들어오는' 느낌)
        open_cutoff = min(nyquist * 0.9, fundamental * (4.0 + 22.0 * brightness))
        start_cutoff = min(open_cutoff, fundamental * 2.5)
        segments = 20
        result = np.zeros(frames)
        bounds = np.linspace(0, frames, segments + 1).astype(int)
        for index in range(segments):
            begin, end = bounds[index], bounds[index + 1]
            if end <= begin:
                continue
            progress = min(1.0, (bounds[index] / sample_rate) / 0.09)
            cutoff = start_cutoff + (open_cutoff - start_cutoff) * progress
            sos = scipy_signal.butter(2, min(max(cutoff, 60.0), nyquist * 0.95),
                                      btype="lowpass", fs=sample_rate, output="sos")
            pad = min(begin, 512)
            filtered = scipy_signal.sosfilt(sos, wave[begin - pad : end])
            result[begin:end] = filtered[pad:]

        envelope = ADSR(attack=0.045, decay=0.1, sustain=0.9, release=0.2,
                        attack_curve="linear")
        gate = min(int(round(duration * sample_rate)), frames)
        result *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


class Flute(Instrument):
    """플루트 계열. 배음이 적고 숨소리가 섞인다."""

    info = InstrumentInfo("flute", "플루트", "wind", 59, 96)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 0.25

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        time = np.arange(frames) / sample_rate

        vibrato = 1.0 + 0.004 * np.sin(2.0 * np.pi * 5.0 * time) * np.clip(time / 0.35, 0, 1)
        phase = phase_from_frequency(fundamental * vibrato, frames, sample_rate)
        # 거의 사인파에 가깝고, 2·3배음이 조금 섞인다
        wave = np.sin(phase) + 0.18 * np.sin(2 * phase) + 0.07 * np.sin(3 * phase) * brightness

        # 숨소리. 플루트 소리의 절반은 이것이다.
        breath = white_noise(frames, seed=midi * 5 + velocity)
        sos = scipy_signal.butter(
            2, [min(nyquist * 0.8, fundamental * 1.5), min(nyquist * 0.9, fundamental * 6.0)],
            btype="bandpass", fs=sample_rate, output="sos"
        )
        breath = scipy_signal.sosfilt(sos, breath)
        breath_peak = np.max(np.abs(breath))
        if breath_peak > 0:
            breath = breath / breath_peak
        wave += breath * (0.16 - 0.08 * brightness)

        envelope = ADSR(attack=0.06, decay=0.08, sustain=0.92, release=0.14,
                        attack_curve="linear")
        gate = min(int(round(duration * sample_rate)), frames)
        wave *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(wave))
        return wave / peak * amplitude if peak > 0 else wave


# ==========================================================================
# 신스
# ==========================================================================

class SynthPad(Instrument):
    """패드. 천천히 부풀어 오르고 오래 남는다. 배경을 채운다."""

    info = InstrumentInfo("synth_pad", "신스 패드", "synth", 24, 100)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 2.2

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        time = np.arange(frames) / sample_rate

        result = np.zeros(frames)
        for index, cents in enumerate((-11.0, -4.0, 0.0, 5.0, 12.0)):
            voice_hz = fundamental * (2.0 ** (cents / 1200.0))
            # 아주 느린 흔들림으로 소리가 살아 있게 만든다
            drift = 1.0 + 0.0009 * np.sin(2.0 * np.pi * (0.13 + 0.07 * index) * time + index)
            phase = phase_from_frequency(voice_hz * drift, frames, sample_rate,
                                         start_phase=index * 1.7)
            result += saw_wave(phase, voice_hz * 1.01, sample_rate) * 0.2

        cutoff = min(nyquist * 0.9, fundamental * (3.0 + 8.0 * brightness) + 400.0)
        sos = scipy_signal.butter(2, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        result = scipy_signal.sosfilt(sos, result)

        envelope = ADSR(attack=0.55, decay=0.5, sustain=0.8, release=1.8, attack_curve="linear")
        gate = min(int(round(duration * sample_rate)), frames)
        result *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


class SynthLead(Instrument):
    """리드. 앞에 나서는 선율용. 날카롭고 분명하다."""

    info = InstrumentInfo("synth_lead", "신스 리드", "synth", 48, 108, polyphonic=False)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 0.3

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        time = np.arange(frames) / sample_rate

        vibrato = 1.0 + 0.005 * np.sin(2.0 * np.pi * 5.8 * time) * np.clip((time - 0.2) / 0.3, 0, 1)
        # 두 오실레이터를 기준 음 위아래로 같은 만큼 벌린다. 한쪽만 올리면
        # 두 소리의 가운데가 그만큼 높아져서 악기 전체가 조금 높게 들린다.
        low_phase = phase_from_frequency(fundamental * 0.9970 * vibrato, frames, sample_rate)
        high_phase = phase_from_frequency(fundamental * 1.0030 * vibrato, frames, sample_rate)
        wave = (
            saw_wave(low_phase, fundamental * 1.01, sample_rate) * 0.55
            + square_wave(high_phase, fundamental * 1.01, sample_rate) * 0.35
        )

        cutoff = min(nyquist * 0.9, fundamental * (5.0 + 14.0 * brightness))
        sos = scipy_signal.butter(2, max(cutoff, 200.0), btype="lowpass",
                                  fs=sample_rate, output="sos")
        wave = scipy_signal.sosfilt(sos, wave)

        envelope = ADSR(attack=0.012, decay=0.12, sustain=0.8, release=0.18)
        gate = min(int(round(duration * sample_rate)), frames)
        wave *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(wave))
        return wave / peak * amplitude if peak > 0 else wave


class Bell(Instrument):
    """종. 배음이 정수배가 아니다. 그래서 화음처럼 들리지 않고 '땡' 하고 울린다."""

    info = InstrumentInfo("bell", "벨", "percussion", 48, 108)

    # 실제 종의 부분음 비율 (정수배가 아니다)
    PARTIALS: tuple[tuple[float, float, float], ...] = (
        (0.5, 0.30, 4.0), (1.0, 1.00, 3.0), (1.2, 0.45, 2.2), (1.5, 0.55, 1.8),
        (2.0, 0.40, 1.4), (2.5, 0.30, 1.0), (3.0, 0.25, 0.8), (4.2, 0.18, 0.5),
        (5.4, 0.12, 0.35), (6.8, 0.08, 0.25),
    )

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 4.0

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        time = np.arange(frames) / sample_rate

        result = np.zeros(frames)
        silence_level = 1e-4
        for ratio, level, half_life in self.PARTIALS:
            partial_hz = fundamental * ratio
            if partial_hz >= nyquist * 0.98:
                continue
            gain = level * (0.5 + 0.5 * brightness) ** (ratio * 0.3)
            if gain < silence_level:
                continue
            # 부분음이 안 들리게 된 뒤로는 계산하지 않는다
            span = min(frames, max(1, int(half_life * math.log2(gain / silence_level) * sample_rate)))
            local_time = time[:span]
            result[:span] += gain * np.sin(2.0 * np.pi * partial_hz * local_time + ratio) \
                * np.power(0.5, local_time / half_life)

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


class Choir(Instrument):
    """합창 '아'. 사람 목소리의 포먼트를 흉내낸다.

    모음은 배음 구조가 아니라 포먼트(특정 주파수대의 공명)가 결정한다.
    '아' 는 대략 730Hz / 1090Hz / 2440Hz 에 공명이 있다.
    """

    info = InstrumentInfo("choir", "합창", "choir", 40, 84)

    FORMANTS: tuple[tuple[float, float, float], ...] = (
        (730.0, 0.0, 80.0), (1090.0, -6.0, 90.0), (2440.0, -12.0, 120.0),
        (3400.0, -22.0, 130.0),
    )

    def tail_seconds(self, midi: int, velocity: int) -> float:
        return 1.0

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        self.check_range(midi)
        amplitude = velocity_to_amplitude(velocity)
        tail = self.tail_seconds(midi, velocity)
        frames = self._frames(duration, sample_rate, tail)
        fundamental = midi_to_frequency(midi)
        nyquist = sample_rate / 2.0
        generator = np.random.default_rng(midi * 41 + velocity)

        # 여러 명이 부르므로 음정이 조금씩 다르다
        source = np.zeros(frames)
        for singer in range(5):
            cents = generator.uniform(-14.0, 14.0)
            singer_hz = fundamental * (2.0 ** (cents / 1200.0))
            time = np.arange(frames) / sample_rate
            vibrato = 1.0 + 0.006 * np.sin(
                2.0 * np.pi * generator.uniform(4.5, 6.0) * time + generator.uniform(0, 6.28)
            ) * np.clip(time / 0.4, 0, 1)
            phase = phase_from_frequency(singer_hz * vibrato, frames, sample_rate,
                                         start_phase=generator.uniform(0, 6.28))
            # 성대 진동은 톱니에 가깝다
            source += saw_wave(phase, singer_hz * 1.02, sample_rate) * 0.2

        # 포먼트 공명을 건다
        result = np.zeros(frames)
        for center, gain_db, bandwidth in self.FORMANTS:
            if center >= nyquist * 0.9:
                continue
            q = center / bandwidth
            omega = 2.0 * math.pi * center / sample_rate
            alpha = math.sin(omega) / (2.0 * q)
            b = np.array([alpha, 0.0, -alpha])
            a = np.array([1 + alpha, -2 * math.cos(omega), 1 - alpha])
            result += scipy_signal.lfilter(b / a[0], a / a[0], source) * db_to_linear(gain_db)

        result += source * 0.12   # 원음도 조금 남겨 두께를 유지한다

        envelope = ADSR(attack=0.18, decay=0.3, sustain=0.85, release=0.8, attack_curve="linear")
        gate = min(int(round(duration * sample_rate)), frames)
        result *= envelope.render(gate, sample_rate, frames - gate)[:frames]

        peak = np.max(np.abs(result))
        return result / peak * amplitude if peak > 0 else result


# ==========================================================================
# 드럼
# ==========================================================================

# 일반 MIDI 드럼 노트 번호. DAW 와 MIDI 파일이 이 번호를 쓴다.
DRUM_NOTES: dict[int, str] = {
    35: "kick_soft", 36: "kick", 37: "rim", 38: "snare", 39: "clap", 40: "snare_tight",
    41: "tom_low", 42: "hihat_closed", 43: "tom_low", 44: "hihat_pedal",
    45: "tom_mid", 46: "hihat_open", 47: "tom_mid", 48: "tom_high",
    49: "crash", 50: "tom_high", 51: "ride", 52: "china", 53: "ride_bell",
    54: "tambourine", 55: "splash", 56: "cowbell", 57: "crash", 59: "ride",
    69: "shaker", 70: "shaker", 75: "clave", 76: "woodblock",
}

DRUM_DISPLAY_NAMES: dict[str, str] = {
    "kick": "킥", "kick_soft": "킥(약)", "snare": "스네어", "snare_tight": "스네어(타이트)",
    "rim": "림샷", "clap": "클랩", "hihat_closed": "클로즈 하이햇",
    "hihat_open": "오픈 하이햇", "hihat_pedal": "페달 하이햇",
    "tom_low": "로우 탐", "tom_mid": "미드 탐", "tom_high": "하이 탐",
    "crash": "크래시", "ride": "라이드", "ride_bell": "라이드 벨", "china": "차이나",
    "splash": "스플래시", "cowbell": "카우벨", "tambourine": "탬버린",
    "shaker": "셰이커", "clave": "클라베", "woodblock": "우드블록",
}


class DrumKit:
    """드럼 한 세트. 음높이가 아니라 '어떤 악기를 쳤는가'로 소리를 고른다.

    각 소리를 실제 구조대로 만든다.
        킥      음이 순식간에 아래로 떨어지는 사인파 + 때리는 소리
        스네어  낮은 몸통 울림 + 아래 울림줄의 잡음
        하이햇  금속판 여러 장이라 비정수 배음 + 잡음
        심벌    같은 원리에 훨씬 긴 여운
    """

    info = InstrumentInfo("drum_kit", "드럼 킷", "drums", 27, 87, default_velocity=100)

    def available(self) -> list[str]:
        return sorted(set(DRUM_NOTES.values()))

    def render_note(self, midi: int, velocity: int, duration: float, sample_rate: int) -> np.ndarray:
        """MIDI 번호로 친다. duration 은 오픈 하이햇 같은 일부에만 영향을 준다."""
        name = DRUM_NOTES.get(midi)
        if name is None:
            raise InstrumentError(
                f"드럼 노트 {midi} 에 배정된 악기가 없습니다. "
                f"사용 가능한 번호: {sorted(DRUM_NOTES)}"
            )
        return self.render(name, velocity, sample_rate, duration)

    def tail_seconds(self, midi: int, velocity: int) -> float:
        name = DRUM_NOTES.get(midi, "kick")
        return {"crash": 3.5, "china": 3.0, "splash": 1.5, "ride": 2.5,
                "ride_bell": 2.0, "hihat_open": 0.9, "tom_low": 0.8,
                "tom_mid": 0.7, "tom_high": 0.6}.get(name, 0.5)

    def render(self, name: str, velocity: int = 100, sample_rate: int = 48000,
               duration: float = 0.0) -> np.ndarray:
        """이름으로 친다."""
        if name not in DRUM_DISPLAY_NAMES:
            raise InstrumentError(
                f"모르는 드럼 악기입니다: {name!r} "
                f"(사용 가능: {', '.join(sorted(DRUM_DISPLAY_NAMES))})"
            )
        amplitude = velocity_to_amplitude(velocity)
        brightness = velocity_to_brightness(velocity)
        maker = getattr(self, f"_{name}", None)
        if maker is None:
            maker = {
                "kick_soft": lambda v, b, r: self._kick(v, b * 0.6, r),
                "snare_tight": lambda v, b, r: self._snare(v, b, r, tight=True),
                "hihat_pedal": lambda v, b, r: self._hihat_closed(v, b * 0.7, r),
                "tom_low": lambda v, b, r: self._tom(v, b, r, 95.0),
                "tom_mid": lambda v, b, r: self._tom(v, b, r, 140.0),
                "tom_high": lambda v, b, r: self._tom(v, b, r, 200.0),
                "china": lambda v, b, r: self._crash(v, b, r, decay=2.2, bright=1.3),
                "splash": lambda v, b, r: self._crash(v, b, r, decay=1.0, bright=1.5),
                "ride_bell": lambda v, b, r: self._ride(v, b, r, bell=True),
            }.get(name)
        if maker is None:
            raise InstrumentError(f"{name} 의 소리 생성기가 없습니다.")
        signal = maker(amplitude, brightness, sample_rate)
        if duration > 0:
            wanted = int(round(duration * sample_rate))
            if 0 < wanted < signal.shape[0] and name in ("hihat_open", "ride", "crash"):
                # 오픈 하이햇은 닫으면 소리가 끊긴다
                fade = np.ones(signal.shape[0])
                remaining = signal.shape[0] - wanted
                fade[wanted:] = np.exp(-np.arange(remaining) / (0.02 * sample_rate))
                signal = signal * fade
        peak = np.max(np.abs(signal))
        return signal / peak * amplitude if peak > 0 else signal

    # ---------------------------------------------------------------- 개별 소리

    def _kick(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.9 * rate)
        time = np.arange(frames) / rate
        # 음이 순식간에 떨어진다. 이 하강이 킥의 '펀치'를 만든다.
        start_hz = 110.0 + 60.0 * brightness
        end_hz = 45.0
        frequency = end_hz + (start_hz - end_hz) * np.exp(-time / 0.028)
        phase = phase_from_frequency(frequency, frames, rate)
        body = np.sin(phase) * np.exp(-time / (0.22 + 0.08 * brightness))
        # 비터가 가죽을 때리는 소리
        click_frames = int(0.006 * rate)
        click = np.zeros(frames)
        noise = white_noise(click_frames, seed=1)
        sos = scipy_signal.butter(2, [800.0, min(rate / 2 * 0.9, 6000.0)],
                                  btype="bandpass", fs=rate, output="sos")
        click[:click_frames] = scipy_signal.sosfilt(sos, noise) * np.exp(
            -np.arange(click_frames) / (0.0012 * rate)
        ) * (0.25 + 0.45 * brightness)
        return body + click

    def _snare(self, amplitude: float, brightness: float, rate: int,
               tight: bool = False) -> np.ndarray:
        frames = int((0.35 if tight else 0.55) * rate)
        time = np.arange(frames) / rate
        # 몸통의 두 공명
        body = (
            np.sin(2.0 * np.pi * 185.0 * time) * np.exp(-time / 0.10)
            + np.sin(2.0 * np.pi * 330.0 * time) * np.exp(-time / 0.07) * 0.6
        ) * 0.5
        # 아래 울림줄(스네어 와이어)의 잡음. 스네어 소리의 절반 이상이 이것이다.
        noise = white_noise(frames, seed=2)
        sos = scipy_signal.butter(2, [180.0, min(rate / 2 * 0.9, 9000.0 + 3000.0 * brightness)],
                                  btype="bandpass", fs=rate, output="sos")
        wires = scipy_signal.sosfilt(sos, noise)
        wires_peak = np.max(np.abs(wires))
        if wires_peak > 0:
            wires /= wires_peak
        wires *= np.exp(-time / (0.055 if tight else 0.11))
        return body * 0.55 + wires * (0.6 + 0.3 * brightness)

    def _rim(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.12 * rate)
        time = np.arange(frames) / rate
        tone = np.sin(2.0 * np.pi * 800.0 * time) * np.exp(-time / 0.012)
        noise = white_noise(frames, seed=3)
        sos = scipy_signal.butter(2, [1500.0, min(rate / 2 * 0.9, 8000.0)],
                                  btype="bandpass", fs=rate, output="sos")
        crack = scipy_signal.sosfilt(sos, noise) * np.exp(-time / 0.006)
        return tone * 0.5 + crack * 1.2

    def _clap(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.45 * rate)
        result = np.zeros(frames)
        noise = white_noise(frames, seed=4)
        sos = scipy_signal.butter(2, [900.0, min(rate / 2 * 0.9, 7000.0)],
                                  btype="bandpass", fs=rate, output="sos")
        filtered = scipy_signal.sosfilt(sos, noise)
        # 손뼉은 한 번이 아니라 아주 짧은 간격으로 서너 번 겹쳐 들린다.
        # 이게 없으면 그냥 짧은 잡음이 된다.
        for offset_ms, level in ((0.0, 1.0), (9.0, 0.85), (19.0, 0.7), (30.0, 0.5)):
            offset = int(offset_ms * 0.001 * rate)
            if offset >= frames:
                continue
            length = frames - offset
            envelope = np.exp(-np.arange(length) / (0.008 * rate))
            result[offset:] += filtered[:length] * envelope * level
        # 마지막 잔향
        tail = np.exp(-np.arange(frames) / (0.09 * rate))
        result += filtered * tail * 0.18
        return result

    def _metal(self, frames: int, rate: int, ratios: tuple[float, ...], base_hz: float,
               half_life: float, seed: int) -> np.ndarray:
        """금속판 공통. 비정수 배음 여러 개 + 잡음."""
        time = np.arange(frames) / rate
        nyquist = rate / 2.0
        result = np.zeros(frames)
        for index, ratio in enumerate(ratios):
            partial = base_hz * ratio
            if partial >= nyquist * 0.95:
                continue
            result += np.sin(2.0 * np.pi * partial * time + index) * np.exp(
                -time / (half_life * (1.0 - 0.06 * index))
            ) / (1.0 + 0.25 * index)
        noise = white_noise(frames, seed=seed)
        sos = scipy_signal.butter(2, min(nyquist * 0.9, base_hz * 2.0),
                                  btype="highpass", fs=rate, output="sos")
        hiss = scipy_signal.sosfilt(sos, noise)
        hiss_peak = np.max(np.abs(hiss))
        if hiss_peak > 0:
            hiss /= hiss_peak
        return result * 0.6 + hiss * np.exp(-time / half_life) * 0.8

    def _hihat_closed(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.16 * rate)
        return self._metal(frames, rate, (1.0, 1.34, 1.79, 2.41, 3.17, 4.09),
                           2600.0 + 600.0 * brightness, 0.022, seed=5)

    def _hihat_open(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.9 * rate)
        return self._metal(frames, rate, (1.0, 1.34, 1.79, 2.41, 3.17, 4.09),
                           2500.0 + 600.0 * brightness, 0.22, seed=6)

    def _crash(self, amplitude: float, brightness: float, rate: int,
               decay: float = 1.4, bright: float = 1.0) -> np.ndarray:
        frames = int(min(4.0, decay * 2.2) * rate)
        return self._metal(frames, rate, (1.0, 1.21, 1.47, 1.83, 2.29, 2.87, 3.61, 4.53),
                           (1700.0 + 500.0 * brightness) * bright, decay, seed=7)

    def _ride(self, amplitude: float, brightness: float, rate: int,
              bell: bool = False) -> np.ndarray:
        frames = int(2.6 * rate)
        time = np.arange(frames) / rate
        base = self._metal(frames, rate, (1.0, 1.19, 1.42, 1.71, 2.13, 2.66),
                           1400.0 + 400.0 * brightness, 0.9, seed=8)
        # 라이드는 '땅' 하는 타점이 분명하다
        ping = np.sin(2.0 * np.pi * 2400.0 * time) * np.exp(-time / 0.05) * 0.5
        if bell:
            ping = np.sin(2.0 * np.pi * 1180.0 * time) * np.exp(-time / 0.35) * 1.1
            ping += np.sin(2.0 * np.pi * 2360.0 * time) * np.exp(-time / 0.2) * 0.4
        return base * (0.45 if bell else 0.8) + ping

    def _tom(self, amplitude: float, brightness: float, rate: int,
             base_hz: float = 140.0) -> np.ndarray:
        frames = int(0.8 * rate)
        time = np.arange(frames) / rate
        # 탐도 음이 조금 떨어진다
        frequency = base_hz * (1.0 + 0.28 * np.exp(-time / 0.05))
        phase = phase_from_frequency(frequency, frames, rate)
        body = (np.sin(phase) + 0.3 * np.sin(2.0 * phase)) * np.exp(-time / 0.20)
        strike_frames = int(0.005 * rate)
        strike = np.zeros(frames)
        noise = white_noise(strike_frames, seed=9)
        sos = scipy_signal.butter(2, min(rate / 2 * 0.9, 4000.0),
                                  btype="lowpass", fs=rate, output="sos")
        strike[:strike_frames] = scipy_signal.sosfilt(sos, noise) * np.exp(
            -np.arange(strike_frames) / (0.0015 * rate)
        ) * 0.3 * brightness
        return body + strike

    def _cowbell(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.4 * rate)
        time = np.arange(frames) / rate
        # 카우벨은 두 개의 서로 안 어울리는 주파수가 특징이다
        envelope = np.exp(-time / 0.12)
        return (np.sin(2.0 * np.pi * 540.0 * time) + np.sin(2.0 * np.pi * 800.0 * time)) \
            * envelope * 0.5

    def _tambourine(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.45 * rate)
        return self._metal(frames, rate, (1.0, 1.51, 2.13, 2.88, 3.77),
                           5200.0, 0.10, seed=10)

    def _shaker(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.20 * rate)
        time = np.arange(frames) / rate
        noise = white_noise(frames, seed=11)
        sos = scipy_signal.butter(2, [4000.0, min(rate / 2 * 0.9, 13000.0)],
                                  btype="bandpass", fs=rate, output="sos")
        filtered = scipy_signal.sosfilt(sos, noise)
        # 알갱이가 한쪽으로 몰렸다가 부딪히므로 소리가 부드럽게 올라갔다 내려온다
        envelope = np.exp(-((time - 0.022) ** 2) / (2 * 0.016 ** 2))
        return filtered * envelope

    def _clave(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.18 * rate)
        time = np.arange(frames) / rate
        return (np.sin(2.0 * np.pi * 1200.0 * time) * np.exp(-time / 0.02)
                + np.sin(2.0 * np.pi * 2400.0 * time) * np.exp(-time / 0.012) * 0.4)

    def _woodblock(self, amplitude: float, brightness: float, rate: int) -> np.ndarray:
        frames = int(0.15 * rate)
        time = np.arange(frames) / rate
        return (np.sin(2.0 * np.pi * 900.0 * time) * np.exp(-time / 0.018)
                + np.sin(2.0 * np.pi * 1700.0 * time) * np.exp(-time / 0.010) * 0.5)


# ==========================================================================
# 악기 등록
# ==========================================================================

_REGISTRY: dict[str, Callable[[], Instrument]] = {
    "acoustic_piano": AcousticPiano,
    "electric_piano": ElectricPiano,
    "organ": Organ,
    "acoustic_guitar": AcousticGuitar,
    "electric_guitar": ElectricGuitar,
    "electric_bass": ElectricBass,
    "synth_bass": SynthBass,
    "sub_bass": SubBass,
    "strings": Strings,
    "brass": Brass,
    "flute": Flute,
    "synth_pad": SynthPad,
    "synth_lead": SynthLead,
    "bell": Bell,
    "choir": Choir,
    "drum_kit": DrumKit,
}


def create_instrument(name: str, **kwargs) -> Instrument:
    """이름으로 악기를 만든다."""
    if name not in _REGISTRY:
        raise InstrumentError(
            f"모르는 악기입니다: {name!r}\n사용 가능: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


def available_instruments() -> list[str]:
    return sorted(_REGISTRY)


def instruments_by_category() -> dict[str, list[str]]:
    """분류별 악기 목록. 화면에 묶어 보여줄 때 쓴다."""
    grouped: dict[str, list[str]] = {key: [] for key in CATEGORIES}
    for name, factory in _REGISTRY.items():
        info = factory.info if hasattr(factory, "info") else factory().info
        grouped[info.category].append(name)
    return {key: sorted(value) for key, value in grouped.items() if value}
