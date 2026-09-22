"""
엔벨로프 — 소리의 시간에 따른 변화.

같은 파형이라도 엔벨로프가 다르면 완전히 다른 악기가 된다.
피아노와 오르간은 배음 구조보다 엔벨로프 차이가 더 크다.

    피아노   때리자마자 최대, 곧바로 줄어들기 시작, 떼면 빨리 사라짐
    오르간   눌러진 동안 계속 같은 크기, 떼면 바로 사라짐
    스트링   서서히 커지고, 떼도 천천히 사라짐
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .buffer import AudioError


@dataclass(frozen=True, slots=True)
class ADSR:
    """네 단계 엔벨로프.

    attack   무음에서 최대까지 걸리는 시간(초)
    decay    최대에서 sustain 높이까지 내려오는 시간(초)
    sustain  누르고 있는 동안 유지되는 높이 (0 ~ 1)
    release  떼고 나서 무음까지 걸리는 시간(초)
    curve    'exponential' 이면 실제 악기처럼 처음 빠르게 변한다.
             'linear' 는 기계적으로 들린다.
    """

    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.3
    curve: str = "exponential"
    attack_curve: str = "linear"

    def __post_init__(self) -> None:
        for name, value in (
            ("attack", self.attack), ("decay", self.decay), ("release", self.release)
        ):
            if value < 0:
                raise AudioError(f"{name} 은 0 이상이어야 합니다: {value}")
        if not (0.0 <= self.sustain <= 1.0):
            raise AudioError(f"sustain 은 0 ~ 1 이어야 합니다: {self.sustain}")
        if self.curve not in ("exponential", "linear"):
            raise AudioError(f"알 수 없는 곡선입니다: {self.curve!r}")
        if self.attack_curve not in ("exponential", "linear"):
            raise AudioError(f"알 수 없는 어택 곡선입니다: {self.attack_curve!r}")

    @property
    def tail_seconds(self) -> float:
        """음을 뗀 뒤에도 소리가 남는 시간. 버퍼 길이 계산에 쓴다."""
        return self.release

    def _attack_shape(self, progress: np.ndarray) -> np.ndarray:
        """0 -> 1 로 올라가는 곡선."""
        if self.attack_curve == "exponential":
            # 처음 빠르게 올라온 뒤 완만해진다. 타악기 계열에 맞는다.
            full = 1.0 - math.exp(-5.0)
            return (1.0 - np.exp(-5.0 * progress)) / full
        return progress

    def _decay_shape(self, progress: np.ndarray) -> np.ndarray:
        """1 -> 0 으로 내려가는 곡선."""
        if self.curve == "exponential":
            full = 1.0 - math.exp(-5.0)
            return (np.exp(-5.0 * progress) - math.exp(-5.0)) / full
        return 1.0 - progress

    def render(self, gate_frames: int, sample_rate: int, tail_frames: int | None = None) -> np.ndarray:
        """엔벨로프 곡선을 만든다.

        gate_frames  음을 누르고 있는 길이(샘플)
        tail_frames  그 뒤에 릴리즈를 위해 더 그릴 길이. None 이면 release 만큼.

        각 구간은 정해진 속도로 진행하며, 게이트가 짧으면 도중에 잘린다.
        어택 0.1초짜리를 0.05초만 누르면 절반까지만 올라가고 거기서 릴리즈가
        시작된다. 게이트 길이에 맞춰 어택을 압축하면 짧은 음이 전부 최대
        음량으로 울려서 스타카토가 스타카토처럼 안 들린다.
        """
        if gate_frames < 0:
            raise AudioError(f"게이트 길이는 0 이상이어야 합니다: {gate_frames}")
        release_frames = int(round(self.release * sample_rate))
        if tail_frames is None:
            tail_frames = release_frames
        total = gate_frames + tail_frames
        if total <= 0:
            return np.zeros(0)

        result = np.zeros(total, dtype=np.float64)
        attack_full = int(round(self.attack * sample_rate))
        decay_full = int(round(self.decay * sample_rate))

        # --- 어택: 정해진 속도로 진행하다 게이트가 끝나면 거기서 잘린다 ---
        attack_used = min(attack_full, gate_frames)
        if attack_used > 0:
            progress = np.arange(attack_used, dtype=np.float64) / attack_full
            result[:attack_used] = self._attack_shape(progress)
        cursor = attack_used
        # 어택이 0 이면 곧바로 최대에서 시작한다
        peak_level = 1.0

        # --- 디케이: 최대에서 sustain 까지 ---
        if cursor < gate_frames:
            decay_used = min(decay_full, gate_frames - cursor)
            if decay_used > 0:
                progress = np.arange(decay_used, dtype=np.float64) / decay_full
                result[cursor : cursor + decay_used] = (
                    self.sustain + (peak_level - self.sustain) * self._decay_shape(progress)
                )
                cursor += decay_used
            elif decay_full == 0:
                pass

        # --- 서스테인 ---
        if cursor < gate_frames:
            result[cursor:gate_frames] = self.sustain

        # --- 릴리즈: 뗀 순간의 실제 높이에서 시작한다 ---
        if gate_frames == 0:
            release_start = 0.0
        elif attack_full == 0 and decay_full == 0:
            release_start = self.sustain if gate_frames > 0 else 0.0
        else:
            release_start = float(result[gate_frames - 1])
        if tail_frames > 0 and release_start > 0.0:
            if release_frames > 0:
                progress = np.arange(tail_frames, dtype=np.float64) / release_frames
                shape = self._decay_shape(np.minimum(progress, 1.0))
                shape[progress > 1.0] = 0.0
            else:
                shape = np.zeros(tail_frames)
            result[gate_frames:] = release_start * shape
        return result

    def describe(self) -> str:
        return (
            f"A {self.attack * 1000:.0f}ms / D {self.decay * 1000:.0f}ms / "
            f"S {self.sustain:.2f} / R {self.release * 1000:.0f}ms"
        )


def exponential_decay(frames: int, sample_rate: int, half_life_seconds: float,
                      start: float = 1.0) -> np.ndarray:
    """지수 감쇠 곡선. 피아노·기타처럼 때리고 나면 계속 줄어드는 소리에 쓴다.

    half_life_seconds 마다 절반이 된다.
    """
    if half_life_seconds <= 0:
        raise AudioError(f"반감기는 양수여야 합니다: {half_life_seconds}")
    if frames <= 0:
        return np.zeros(max(0, frames))
    time = np.arange(frames) / sample_rate
    return start * np.power(0.5, time / half_life_seconds)


def percussive_envelope(frames: int, sample_rate: int, attack_ms: float = 1.0,
                        decay_seconds: float = 0.3, curve: float = 4.0) -> np.ndarray:
    """타악기 엔벨로프. 아주 빠르게 올라갔다가 지수로 떨어진다."""
    if frames <= 0:
        return np.zeros(max(0, frames))
    attack_frames = max(1, int(round(attack_ms * 0.001 * sample_rate)))
    result = np.zeros(frames, dtype=np.float64)
    attack_frames = min(attack_frames, frames)
    result[:attack_frames] = np.linspace(0.0, 1.0, attack_frames, endpoint=False) \
        if attack_frames > 1 else 1.0
    remaining = frames - attack_frames
    if remaining > 0:
        time = np.arange(remaining) / sample_rate
        result[attack_frames:] = np.exp(-curve * time / max(decay_seconds, 1e-6))
    return result


def apply_envelope(signal: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """엔벨로프를 신호에 건다. 길이가 다르면 짧은 쪽에 맞춘다."""
    length = min(signal.shape[-1], envelope.shape[-1])
    return signal[..., :length] * envelope[:length]
