"""
녹음 — 6번 "프로그램이 사용자의 노래를 직접 녹음한다".

연습 하나를 녹음한다. 가이드 음을 먼저 들려주고(cue), 조용할 때 사용자가
따라 부른다. 노래하는 동안에는 아무것도 틀지 않으므로 스피커로 들어도
가이드가 녹음에 섞이지 않는다.

재생과 녹음을 한 번에 여닫는다 (sounddevice.playrec). 따로 열면 둘의 시작
시각이 어긋나서, 녹음이 가이드보다 얼마나 늦었는지 알 수 없게 된다.
남는 장치 지연은 분석할 때 녹음과 가이드를 맞춰 보며 찾는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..voice.analysis import midi_to_hz
from ..voice.training import Exercise


class RecordingError(RuntimeError):
    """녹음 오류."""


def render_cues(exercise: Exercise, rate: int, level: float = 0.25) -> np.ndarray:
    """가이드 음들. 각 부를 구간 바로 앞에서 그 음을 짧게 들려준다.

    배음을 몇 개 섞은 부드러운 소리다. 순수한 사인파는 음높이를 잡기 어렵다는
    사람이 많고, 너무 밝으면 귀에 거슬린다.
    """
    total = int(math.ceil(exercise.duration * rate))
    out = np.zeros(total, dtype=np.float64)
    for note in exercise.notes:
        begin = note.start - exercise.cue_seconds - 0.15
        if begin < 0:
            continue
        length = int(exercise.cue_seconds * rate)
        t = np.arange(length) / rate
        hz = midi_to_hz(note.midi)
        tone = (np.sin(2 * np.pi * hz * t) + 0.35 * np.sin(4 * np.pi * hz * t)
                + 0.12 * np.sin(6 * np.pi * hz * t))
        attack = np.clip(t / 0.02, 0.0, 1.0)
        release = np.clip((exercise.cue_seconds - t) / 0.08, 0.0, 1.0)
        start = int(begin * rate)
        stop = min(total, start + length)
        out[start:stop] += (tone * attack * release)[: stop - start] * level / 1.47
    return out


@dataclass(slots=True)
class LevelCheck:
    """녹음 상태."""

    peak_db: float
    noise_db: float
    clipped: float           # 찢어진 샘플 비율
    messages: list[str]

    @property
    def ok(self) -> bool:
        return not self.messages


def check_level(signal: np.ndarray, rate: int) -> LevelCheck:
    """녹음이 분석할 만한지. 너무 작거나, 찢어졌거나, 주변이 시끄러운지."""
    x = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    peak_db = 20 * math.log10(peak) if peak > 1e-9 else -120.0
    hop = max(1, int(0.05 * rate))
    frames = x[: x.size // hop * hop].reshape(-1, hop) if x.size >= hop else x.reshape(1, -1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    noise_db = 20 * math.log10(float(np.percentile(rms, 10)))
    loud_db = 20 * math.log10(float(np.percentile(rms, 95)))
    clipped = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    messages = []
    if peak_db < -30:
        messages.append("소리가 너무 작습니다. 마이크 음량을 올리거나 가까이에서 불러 주세요.")
    if clipped > 0.0005:
        messages.append("소리가 찢어졌습니다 (너무 큼). 마이크 음량을 낮추거나 조금 떨어져서 불러 주세요.")
    if loud_db - noise_db < 20 and peak_db >= -30:
        messages.append(f"주변 소음이 큽니다 (노래와 소음 차이 {loud_db - noise_db:.0f}dB). "
                        f"조용한 곳에서 녹음하면 분석이 정확해집니다.")
    return LevelCheck(peak_db, noise_db, clipped, messages)


def input_available() -> tuple[bool, str]:
    """녹음할 수 있는 장치가 있는가."""
    try:
        import sounddevice
    except Exception as error:          # 설치 안 됨, 또는 PortAudio 없음
        return False, f"녹음 장치를 쓸 수 없습니다 (sounddevice: {error})"
    try:
        device = sounddevice.query_devices(kind="input")
    except Exception as error:
        return False, f"마이크를 찾을 수 없습니다: {error}"
    return True, f"마이크: {device['name']}"


class Recorder:
    """마이크 녹음."""

    def __init__(self, rate: int = 48000) -> None:
        self.rate = rate

    def record(self, exercise: Exercise) -> np.ndarray:
        """가이드를 틀면서 녹음한다. 끝날 때까지 돌아오지 않는다 (딴 실뜨기에서 부를 것)."""
        available, message = input_available()
        if not available:
            raise RecordingError(message)
        import sounddevice

        cues = render_cues(exercise, self.rate).astype(np.float32).reshape(-1, 1)
        try:
            recorded = sounddevice.playrec(cues, samplerate=self.rate, channels=1,
                                           dtype="float32", blocking=True)
        except Exception as error:
            raise RecordingError(f"녹음하지 못했습니다: {error}") from error
        return np.asarray(recorded[:, 0], dtype=np.float64)

    def play_cues(self, exercise: Exercise) -> None:
        """가이드만 들어 보기."""
        try:
            import sounddevice
            sounddevice.play(render_cues(exercise, self.rate).astype(np.float32), self.rate)
        except Exception as error:
            raise RecordingError(f"재생하지 못했습니다: {error}") from error

    @staticmethod
    def stop() -> None:
        try:
            import sounddevice
            sounddevice.stop()
        except Exception:
            pass
