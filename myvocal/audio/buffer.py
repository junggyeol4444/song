"""
오디오 버퍼 — 소리를 담는 단 하나의 그릇.

악기 렌더, 목소리 녹음, 목소리 합성, 믹서, 마스터, 내보내기가 전부 이걸 주고받는다.
중간에 포맷이 바뀌면 그 지점에서 음질이 깨지므로, 내부는 항상 다음으로 고정한다.

    float32, 값 범위 [-1.0, 1.0] 기준 (초과 허용 — 클리핑은 마지막에만 한다)
    shape = (채널 수, 샘플 수)

'클리핑은 마지막에만' 이 중요하다. 중간 단계에서 미리 잘라버리면 그 뒤에 음량을
낮춰도 이미 깨진 소리가 된다. 실제 DAW 가 내부를 부동소수점으로 쓰는 이유다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..core.units import DEFAULT_SAMPLE_RATE

DType = np.float32


class AudioError(ValueError):
    """오디오 처리 관련 오류."""


def db_to_linear(db: float) -> float:
    """데시벨 -> 배율. 0dB = 1.0배, -6dB ≈ 0.501배, -inf = 0."""
    if db <= -300.0:
        return 0.0
    return float(10.0 ** (db / 20.0))


def linear_to_db(linear: float) -> float:
    """배율 -> 데시벨. 0 은 -inf 대신 -300dB 로 돌려준다."""
    if linear <= 1e-15:
        return -300.0
    return float(20.0 * math.log10(abs(linear)))


class AudioBuffer:
    """다채널 오디오 한 덩어리.

    numpy 배열을 감싸되, 샘플레이트·채널 수가 안 맞는 연산을 조용히 넘어가지
    않게 막는다. 샘플레이트가 다른 두 소리를 그냥 더하면 피치가 틀어진다.
    """

    __slots__ = ("_data", "_sample_rate")

    # ---------------------------------------------------------------- 생성

    def __init__(self, data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        array = np.asarray(data)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise AudioError(f"오디오는 (채널, 샘플) 2차원이어야 합니다. 받은 모양: {array.shape}")
        if array.shape[0] < 1:
            raise AudioError("채널이 0개입니다.")
        if array.shape[0] > 64:
            raise AudioError(f"채널이 너무 많습니다: {array.shape[0]}")
        if sample_rate <= 0:
            raise AudioError(f"샘플레이트는 양수여야 합니다: {sample_rate}")
        self._data: np.ndarray = np.ascontiguousarray(array, dtype=DType)
        self._sample_rate: int = int(sample_rate)

    @classmethod
    def silence(
        cls, frames: int, channels: int = 2, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> "AudioBuffer":
        if frames < 0:
            raise AudioError(f"샘플 수는 0 이상이어야 합니다: {frames}")
        return cls(np.zeros((channels, frames), dtype=DType), sample_rate)

    @classmethod
    def silence_seconds(
        cls, seconds: float, channels: int = 2, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> "AudioBuffer":
        return cls.silence(int(round(seconds * sample_rate)), channels, sample_rate)

    @classmethod
    def from_mono(cls, samples: Sequence[float] | np.ndarray,
                  sample_rate: int = DEFAULT_SAMPLE_RATE) -> "AudioBuffer":
        return cls(np.asarray(samples, dtype=DType).reshape(1, -1), sample_rate)

    # ---------------------------------------------------------------- 속성

    @property
    def data(self) -> np.ndarray:
        """내부 배열. 직접 고치면 이 버퍼가 바뀐다."""
        return self._data

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._data.shape[0]

    @property
    def frames(self) -> int:
        return self._data.shape[1]

    @property
    def duration(self) -> float:
        """길이(초)."""
        return self.frames / self._sample_rate

    @property
    def is_silent(self) -> bool:
        return bool(np.all(self._data == 0.0))

    def copy(self) -> "AudioBuffer":
        return AudioBuffer(self._data.copy(), self._sample_rate)

    # ------------------------------------------------------- 채널 / 길이 맞추기

    def _require_same_rate(self, other: "AudioBuffer", what: str) -> None:
        if other._sample_rate != self._sample_rate:
            raise AudioError(
                f"{what}: 샘플레이트가 다릅니다 ({self._sample_rate}Hz vs {other._sample_rate}Hz). "
                f"먼저 resample() 로 맞추세요. 그냥 섞으면 음높이가 틀어집니다."
            )

    def to_mono(self) -> "AudioBuffer":
        """모든 채널을 평균낸다. 분석(피치/BPM)에 쓴다."""
        if self.channels == 1:
            return self.copy()
        return AudioBuffer(self._data.mean(axis=0, keepdims=True), self._sample_rate)

    def to_channels(self, channels: int) -> "AudioBuffer":
        """채널 수를 맞춘다. 1->2 는 복제, 2->1 은 평균, 그 외는 앞에서부터 채운다."""
        if channels < 1:
            raise AudioError(f"채널 수는 1 이상이어야 합니다: {channels}")
        if channels == self.channels:
            return self.copy()
        if self.channels == 1:
            return AudioBuffer(np.repeat(self._data, channels, axis=0), self._sample_rate)
        if channels == 1:
            return self.to_mono()
        out = np.zeros((channels, self.frames), dtype=DType)
        common = min(channels, self.channels)
        out[:common] = self._data[:common]
        return AudioBuffer(out, self._sample_rate)

    def padded_to(self, frames: int) -> "AudioBuffer":
        """뒤에 무음을 붙여 길이를 맞춘다. 이미 길면 그대로 둔다."""
        if frames <= self.frames:
            return self.copy()
        out = np.zeros((self.channels, frames), dtype=DType)
        out[:, : self.frames] = self._data
        return AudioBuffer(out, self._sample_rate)

    def slice_frames(self, start: int, end: int | None = None) -> "AudioBuffer":
        """구간 잘라내기. 범위를 벗어나면 무음으로 채운다."""
        end = self.frames if end is None else end
        if end < start:
            raise AudioError(f"끝이 시작보다 앞입니다: start={start} end={end}")
        length = end - start
        out = np.zeros((self.channels, length), dtype=DType)
        src_start = max(0, start)
        src_end = min(self.frames, end)
        if src_end > src_start:
            out[:, src_start - start : src_end - start] = self._data[:, src_start:src_end]
        return AudioBuffer(out, self._sample_rate)

    def slice_seconds(self, start: float, end: float | None = None) -> "AudioBuffer":
        start_frame = int(round(start * self._sample_rate))
        end_frame = None if end is None else int(round(end * self._sample_rate))
        return self.slice_frames(start_frame, end_frame)

    # ---------------------------------------------------------------- 합성

    def mix(self, other: "AudioBuffer", at_frame: int = 0, gain: float = 1.0) -> "AudioBuffer":
        """다른 소리를 지정 위치에 더한다. 길이가 모자라면 늘린다.

        클리핑하지 않는다. 더해서 1.0 을 넘어도 그대로 둔다. 마지막 마스터에서만 다룬다.
        """
        self._require_same_rate(other, "mix")
        source = other.to_channels(self.channels) if other.channels != self.channels else other
        if at_frame < 0:
            # 음수 위치면 앞부분을 잘라내고 0 에 붙인다
            source = source.slice_frames(-at_frame, source.frames)
            at_frame = 0
        needed = max(self.frames, at_frame + source.frames)
        out = np.zeros((self.channels, needed), dtype=DType)
        out[:, : self.frames] = self._data
        if source.frames:
            out[:, at_frame : at_frame + source.frames] += source._data * DType(gain)
        return AudioBuffer(out, self._sample_rate)

    def mix_in_place(self, other: "AudioBuffer", at_frame: int = 0, gain: float = 1.0) -> None:
        """길이를 늘리지 않고 제자리에서 더한다. 넘치는 부분은 버린다.

        렌더러가 큰 버퍼에 수백 개 음을 쌓을 때 쓴다. 매번 새 배열을 만들면 느리다.
        """
        self._require_same_rate(other, "mix_in_place")
        source = other.to_channels(self.channels) if other.channels != self.channels else other
        src_from = max(0, -at_frame)
        dst_from = max(0, at_frame)
        length = min(source.frames - src_from, self.frames - dst_from)
        if length <= 0:
            return
        self._data[:, dst_from : dst_from + length] += (
            source._data[:, src_from : src_from + length] * DType(gain)
        )

    def concat(self, other: "AudioBuffer") -> "AudioBuffer":
        self._require_same_rate(other, "concat")
        source = other.to_channels(self.channels)
        return AudioBuffer(np.concatenate([self._data, source._data], axis=1), self._sample_rate)

    @staticmethod
    def sum_buffers(buffers: Iterable["AudioBuffer"]) -> "AudioBuffer":
        """여러 버퍼를 0 위치 기준으로 전부 합친다. 믹서가 쓴다."""
        items = list(buffers)
        if not items:
            raise AudioError("합칠 버퍼가 없습니다.")
        rate = items[0].sample_rate
        channels = max(b.channels for b in items)
        frames = max(b.frames for b in items)
        out = AudioBuffer.silence(frames, channels, rate)
        for buffer in items:
            out.mix_in_place(buffer, 0)
        return out

    # ---------------------------------------------------------------- 음량

    def with_gain(self, gain: float) -> "AudioBuffer":
        return AudioBuffer(self._data * DType(gain), self._sample_rate)

    def with_gain_db(self, db: float) -> "AudioBuffer":
        return self.with_gain(db_to_linear(db))

    def apply_gain_envelope(self, envelope: np.ndarray) -> "AudioBuffer":
        """샘플별 음량 곡선을 적용한다. 페이드/오토메이션이 쓴다."""
        curve = np.asarray(envelope, dtype=DType)
        if curve.shape[-1] != self.frames:
            raise AudioError(
                f"음량 곡선 길이가 다릅니다: 곡선 {curve.shape[-1]} vs 오디오 {self.frames}"
            )
        return AudioBuffer(self._data * curve, self._sample_rate)

    def faded(self, fade_in: float = 0.0, fade_out: float = 0.0,
              shape: str = "equal_power") -> "AudioBuffer":
        """앞뒤 페이드. shape 는 'linear' 또는 'equal_power'.

        equal_power 는 크로스페이드할 때 음량이 가운데서 꺼지지 않게 한다.
        """
        if fade_in < 0 or fade_out < 0:
            raise AudioError("페이드 길이는 0 이상이어야 합니다.")
        rate = self._sample_rate
        in_frames = min(int(round(fade_in * rate)), self.frames)
        out_frames = min(int(round(fade_out * rate)), self.frames)
        curve = np.ones(self.frames, dtype=DType)
        if in_frames > 0:
            ramp = np.linspace(0.0, 1.0, in_frames, endpoint=False, dtype=DType)
            curve[:in_frames] = np.sin(ramp * np.pi / 2) if shape == "equal_power" else ramp
        if out_frames > 0:
            ramp = np.linspace(1.0, 0.0, out_frames, endpoint=False, dtype=DType)
            curve[-out_frames:] = np.sin(ramp * np.pi / 2) if shape == "equal_power" else ramp
        return self.apply_gain_envelope(curve)

    def panned(self, pan: float) -> "AudioBuffer":
        """좌우 배치. -1 = 완전 왼쪽, 0 = 가운데, +1 = 완전 오른쪽.

        -3dB 등파워 법칙을 쓴다. 가운데에서 음량이 튀지 않는다.
        """
        if not (-1.0 <= pan <= 1.0):
            raise AudioError(f"팬 값은 -1.0 ~ 1.0 이어야 합니다: {pan}")
        stereo = self.to_channels(2)
        angle = (pan + 1.0) * math.pi / 4.0      # 0 ~ pi/2
        left = math.cos(angle)
        right = math.sin(angle)
        out = stereo.data.copy()
        out[0] *= DType(left)
        out[1] *= DType(right)
        return AudioBuffer(out, self._sample_rate)

    def normalized(self, target_db: float = -1.0) -> "AudioBuffer":
        """최대값이 target_db 가 되도록 전체 음량을 맞춘다."""
        peak = self.peak()
        if peak <= 0.0:
            return self.copy()
        return self.with_gain(db_to_linear(target_db) / peak)

    def clipped(self, ceiling: float = 1.0) -> "AudioBuffer":
        """단순 잘라내기. 마스터 맨 끝에서만 쓴다."""
        return AudioBuffer(np.clip(self._data, -ceiling, ceiling), self._sample_rate)

    def reversed(self) -> "AudioBuffer":
        return AudioBuffer(self._data[:, ::-1].copy(), self._sample_rate)

    # ---------------------------------------------------------------- 측정

    def peak(self) -> float:
        """최대 절대값."""
        return float(np.max(np.abs(self._data))) if self.frames else 0.0

    def peak_db(self) -> float:
        return linear_to_db(self.peak())

    def rms(self) -> float:
        """실효값. 체감 음량에 더 가깝다."""
        if self.frames == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self._data.astype(np.float64)))))

    def rms_db(self) -> float:
        return linear_to_db(self.rms())

    def dc_offset(self) -> list[float]:
        """채널별 DC 치우침. 0 에서 멀면 녹음이나 처리에 문제가 있다."""
        if self.frames == 0:
            return [0.0] * self.channels
        return [float(v) for v in self._data.mean(axis=1)]

    def clip_count(self, threshold: float = 0.999) -> int:
        """한계를 넘은 샘플 수. 0 이 아니면 소리가 깨진 것이다."""
        return int(np.count_nonzero(np.abs(self._data) >= threshold))

    def true_peak(self, oversample: int = 4) -> float:
        """샘플 사이에 숨은 최대값까지 잡는다.

        디지털 최대값이 1.0 이어도, 아날로그로 복원하면 그 사이에서 더 올라갈 수
        있다. 방송/스트리밍 규격(EBU R128)이 이 값을 본다.
        """
        if self.frames == 0:
            return 0.0
        if oversample < 1:
            raise AudioError("오버샘플 배수는 1 이상이어야 합니다.")
        if oversample == 1:
            return self.peak()
        from scipy import signal as scipy_signal

        upsampled = scipy_signal.resample_poly(
            self._data.astype(np.float64), oversample, 1, axis=1
        )
        return float(np.max(np.abs(upsampled)))

    def true_peak_db(self, oversample: int = 4) -> float:
        return linear_to_db(self.true_peak(oversample))

    # ---------------------------------------------------------------- 변환

    def resample(self, target_rate: int) -> "AudioBuffer":
        """샘플레이트 변환. 다른 샘플레이트끼리 섞기 전에 반드시 거친다."""
        if target_rate <= 0:
            raise AudioError(f"샘플레이트는 양수여야 합니다: {target_rate}")
        if target_rate == self._sample_rate:
            return self.copy()
        if self.frames == 0:
            return AudioBuffer.silence(0, self.channels, target_rate)
        from math import gcd

        from scipy import signal as scipy_signal

        divisor = gcd(target_rate, self._sample_rate)
        up = target_rate // divisor
        down = self._sample_rate // divisor
        converted = scipy_signal.resample_poly(
            self._data.astype(np.float64), up, down, axis=1
        )
        return AudioBuffer(converted, target_rate)

    # ---------------------------------------------------------------- 파일

    def save(self, path: str | Path, subtype: str = "PCM_24") -> Path:
        """WAV/FLAC 로 저장. 확장자로 포맷을 정한다."""
        import soundfile as sf

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # soundfile 은 (샘플, 채널) 순서를 받는다
        sf.write(str(target), self._data.T, self._sample_rate, subtype=subtype)
        return target

    @classmethod
    def load(cls, path: str | Path, target_rate: int | None = None) -> "AudioBuffer":
        """오디오 파일 읽기. target_rate 를 주면 바로 변환까지 한다."""
        import soundfile as sf

        source = Path(path)
        if not source.exists():
            raise AudioError(f"파일이 없습니다: {source}")
        samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
        buffer = cls(samples.T, rate)
        return buffer.resample(target_rate) if target_rate and target_rate != rate else buffer

    # ---------------------------------------------------------------- 기타

    def __len__(self) -> int:
        return self.frames

    def __repr__(self) -> str:
        return (
            f"AudioBuffer({self.channels}ch, {self.frames}샘플, "
            f"{self.duration:.3f}초, {self._sample_rate}Hz, peak {self.peak_db():.1f}dB)"
        )
