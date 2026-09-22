"""
재생 — 만든 소리를 실제로 듣는다.

편집기에서 소리를 못 들으면 아무 판단도 할 수 없다. 화면의 네모를 보고
음악을 상상할 수 있는 사람은 없다.

오디오 장치가 없는 환경(서버, 원격 접속, 소리 장치가 꺼진 컴퓨터)에서도
프로그램이 죽으면 안 된다. 그런 곳에서도 곡을 만들고 파일로 내보내는 것은
되어야 한다. 그래서 장치가 없으면 '들을 수 없음' 상태로 계속 동작한다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from .buffer import AudioBuffer, AudioError


class PlaybackError(ValueError):
    """재생 관련 오류."""


class PlaybackState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class OutputDevice:
    """소리를 낼 장치."""

    index: int
    name: str
    channels: int
    default_sample_rate: float
    is_default: bool = False

    def describe(self) -> str:
        mark = " (기본)" if self.is_default else ""
        return f"{self.name}{mark} — {self.channels}채널, {self.default_sample_rate:.0f}Hz"


def list_output_devices() -> list[OutputDevice]:
    """쓸 수 있는 출력 장치 목록. 없으면 빈 목록."""
    try:
        import sounddevice
    except Exception:
        return []
    try:
        devices = sounddevice.query_devices()
        default_index = sounddevice.default.device[1]
    except Exception:
        return []
    result: list[OutputDevice] = []
    for index, info in enumerate(devices):
        channels = int(info.get("max_output_channels", 0))
        if channels < 1:
            continue
        result.append(OutputDevice(
            index=index,
            name=str(info.get("name", f"장치 {index}")),
            channels=channels,
            default_sample_rate=float(info.get("default_samplerate", 48000.0)),
            is_default=(index == default_index),
        ))
    return result


def audio_available() -> tuple[bool, str]:
    """소리를 낼 수 있는지와 그 이유."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False, (
            "sounddevice 가 설치돼 있지 않습니다. "
            "'pip install sounddevice' 로 설치하면 소리를 들을 수 있습니다."
        )
    except OSError as error:
        return False, (
            f"오디오 라이브러리를 불러오지 못했습니다: {error}\n"
            f"윈도우에서는 보통 자동으로 해결됩니다. 리눅스라면 libportaudio2 가 필요합니다."
        )
    devices = list_output_devices()
    if not devices:
        return False, (
            "소리를 낼 장치를 찾지 못했습니다. 스피커나 헤드폰이 연결돼 있는지, "
            "소리 장치가 꺼져 있지는 않은지 확인해 주세요.\n"
            "장치가 없어도 곡을 만들고 파일로 내보내는 것은 됩니다."
        )
    return True, f"출력 장치 {len(devices)}개를 찾았습니다."


class PlaybackEngine:
    """미리 만들어 둔 오디오를 재생한다.

    실시간으로 합성하지 않는다. 악기 합성은 실시간보다 빠르긴 하지만
    (3~4배) 여유가 크지 않아서, 재생 도중 한 번이라도 늦으면 소리가 끊긴다.
    미리 만들어 두면 그 문제가 없다.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 2,
                 block_size: int = 1024) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size

        self._buffer: AudioBuffer | None = None
        self._position = 0                  # 샘플 단위
        self._state = PlaybackState.STOPPED
        self._stream = None
        self._lock = threading.RLock()
        self._loop: tuple[int, int] | None = None     # 샘플 범위
        self._volume = 1.0
        self._error = ""
        self._finished_callbacks: list[Callable[[], None]] = []
        self._peak_left = 0.0
        self._peak_right = 0.0

        available, message = audio_available()
        self._available = available
        if not available:
            self._error = message

    # ---------------------------------------------------------------- 상태

    @property
    def available(self) -> bool:
        """소리를 낼 수 있는가."""
        return self._available

    @property
    def unavailable_reason(self) -> str:
        return self._error

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state is PlaybackState.PLAYING

    @property
    def position_seconds(self) -> float:
        with self._lock:
            return self._position / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            return self._buffer.duration if self._buffer is not None else 0.0

    @property
    def peak_levels(self) -> tuple[float, float]:
        """지금 나오고 있는 소리의 크기. 미터 표시에 쓴다."""
        return self._peak_left, self._peak_right

    def set_volume(self, value: float) -> None:
        if not (0.0 <= value <= 2.0):
            raise PlaybackError(f"음량은 0~2 여야 합니다: {value}")
        self._volume = value

    # ---------------------------------------------------------------- 자료

    def load(self, audio: AudioBuffer, keep_position: bool = False) -> None:
        """재생할 소리를 넣는다."""
        if audio.sample_rate != self.sample_rate:
            audio = audio.resample(self.sample_rate)
        if audio.channels != self.channels:
            audio = audio.to_channels(self.channels)
        with self._lock:
            previous = self._position
            self._buffer = audio
            if keep_position:
                self._position = min(previous, max(0, audio.frames - 1))
            else:
                self._position = 0

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self._buffer = None
            self._position = 0

    # ---------------------------------------------------------------- 조작

    def play(self, from_seconds: float | None = None) -> bool:
        """재생한다. 소리를 낼 수 없으면 False 를 돌려준다."""
        with self._lock:
            if self._buffer is None:
                raise PlaybackError("재생할 소리가 없습니다.")
            if from_seconds is not None:
                self._position = max(0, min(self._buffer.frames - 1,
                                            int(from_seconds * self.sample_rate)))
        if not self._available:
            return False
        if self._state is PlaybackState.PLAYING:
            return True
        if not self._open_stream():
            return False
        self._state = PlaybackState.PLAYING
        return True

    def pause(self) -> None:
        if self._state is PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED
            self._close_stream()

    def stop(self) -> None:
        self._state = PlaybackState.STOPPED
        self._close_stream()
        with self._lock:
            self._position = 0
        self._peak_left = self._peak_right = 0.0

    def toggle(self) -> bool:
        if self._state is PlaybackState.PLAYING:
            self.pause()
            return False
        return self.play()

    def seek(self, seconds: float) -> None:
        with self._lock:
            if self._buffer is None:
                self._position = 0
                return
            self._position = max(0, min(self._buffer.frames - 1,
                                        int(seconds * self.sample_rate)))

    def set_loop(self, start_seconds: float | None, end_seconds: float | None) -> None:
        """반복 구간. 둘 중 하나라도 None 이면 반복을 끈다."""
        if start_seconds is None or end_seconds is None:
            self._loop = None
            return
        if end_seconds <= start_seconds:
            raise PlaybackError(
                f"반복 구간의 끝이 시작보다 앞입니다: {start_seconds} ~ {end_seconds}"
            )
        self._loop = (int(start_seconds * self.sample_rate),
                      int(end_seconds * self.sample_rate))

    def on_finished(self, callback: Callable[[], None]) -> None:
        self._finished_callbacks.append(callback)

    # ---------------------------------------------------------------- 내부

    def _open_stream(self) -> bool:
        if self._stream is not None:
            return True
        try:
            import sounddevice

            self._stream = sounddevice.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._callback,
                finished_callback=self._on_stream_finished,
            )
            self._stream.start()
            return True
        except Exception as error:
            self._error = f"소리를 시작할 수 없습니다: {error}"
            self._available = False
            self._stream = None
            return False

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """오디오 장치가 다음 소리를 달라고 부른다.

        이 함수 안에서는 시간을 오래 쓰면 안 된다. 늦으면 소리가 끊긴다.
        그래서 여기서는 이미 만들어 둔 배열을 복사하기만 한다.
        """
        with self._lock:
            buffer = self._buffer
            if buffer is None or self._state is not PlaybackState.PLAYING:
                outdata.fill(0.0)
                return
            position = self._position
            total = buffer.frames

            if self._loop is not None and position >= self._loop[1]:
                position = self._loop[0]

            end = min(position + frames, total)
            if self._loop is not None:
                end = min(end, self._loop[1])
            length = max(0, end - position)

            if length > 0:
                chunk = buffer.data[:, position:end].T
                outdata[:length] = chunk * self._volume
            if length < frames:
                outdata[length:].fill(0.0)
            self._position = end

            if length > 0:
                self._peak_left = float(np.max(np.abs(outdata[:length, 0])))
                self._peak_right = float(
                    np.max(np.abs(outdata[:length, min(1, self.channels - 1)]))
                )
            else:
                self._peak_left = self._peak_right = 0.0

            if end >= total and self._loop is None:
                self._state = PlaybackState.STOPPED
                for callback in self._finished_callbacks:
                    try:
                        callback()
                    except Exception:
                        pass

    def _on_stream_finished(self) -> None:
        self._stream = None

    def __repr__(self) -> str:
        status = "소리 가능" if self._available else "소리 불가"
        return (
            f"PlaybackEngine({status}, {self._state.value}, "
            f"{self.position_seconds:.1f}/{self.duration_seconds:.1f}초)"
        )


class OfflinePlayback(PlaybackEngine):
    """소리를 못 내는 환경에서 위치만 흘려보낸다.

    오디오 장치가 없어도 재생 위치가 움직이고 화면이 따라가면, 곡의 구조를
    눈으로 확인하며 작업할 수 있다. 아무것도 안 움직이면 프로그램이 멈춘
    것처럼 보인다.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 2) -> None:
        super().__init__(sample_rate, channels)
        self._available = False
        self._started_at = 0.0
        self._started_position = 0

    def play(self, from_seconds: float | None = None) -> bool:
        with self._lock:
            if self._buffer is None:
                raise PlaybackError("재생할 소리가 없습니다.")
            if from_seconds is not None:
                self._position = max(0, int(from_seconds * self.sample_rate))
            self._started_at = time.monotonic()
            self._started_position = self._position
        self._state = PlaybackState.PLAYING
        return False        # 소리는 안 난다

    def pause(self) -> None:
        self._advance()
        self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        self._state = PlaybackState.STOPPED
        with self._lock:
            self._position = 0

    def _advance(self) -> None:
        if self._state is not PlaybackState.PLAYING:
            return
        elapsed = time.monotonic() - self._started_at
        with self._lock:
            position = self._started_position + int(elapsed * self.sample_rate)
            if self._loop is not None and position >= self._loop[1]:
                span = self._loop[1] - self._loop[0]
                position = self._loop[0] + (position - self._loop[0]) % max(1, span)
            total = self._buffer.frames if self._buffer is not None else 0
            if position >= total:
                position = total
                self._state = PlaybackState.STOPPED
                for callback in self._finished_callbacks:
                    try:
                        callback()
                    except Exception:
                        pass
            self._position = position

    @property
    def position_seconds(self) -> float:
        self._advance()
        with self._lock:
            return self._position / self.sample_rate


def create_engine(sample_rate: int = 48000, channels: int = 2) -> PlaybackEngine:
    """이 컴퓨터에 맞는 재생기를 만든다. 소리를 못 내면 위치만 흘려보내는 것을 준다."""
    available, _ = audio_available()
    if available:
        return PlaybackEngine(sample_rate, channels)
    return OfflinePlayback(sample_rate, channels)
