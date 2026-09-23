"""
노래 합성 — 가사를 실제로 부른다.

사람의 목소리는 두 부분으로 이루어진다.

    성대     일정한 주기로 열렸다 닫히며 '부저' 같은 소리를 낸다.
             여기서 음높이가 정해진다. 모음이 무엇이든 이 소리는 같다.

    입과 목  그 소리가 지나가며 특정 주파수대가 크게 울린다(포먼트).
             여기서 'ㅏ' 인지 'ㅣ' 인지가 정해진다.

그래서 성대 소리를 만들고 포먼트 공명을 걸면 모음이 된다. 자음은 그 사이에
막음(파열음), 바람(마찰음), 코울림(비음)을 끼워 넣는다. 이 방식을 포먼트
합성이라고 하고, 신경망이 나오기 전 음성 합성이 쓰던 방법이다.

이걸로 나오는 소리는 '사람 목소리를 흉내낸 소리' 다. 진짜 누군가의 목소리는
아니다. 그건 그 사람 목소리를 학습해야 한다(6~9번). 다만 가사를 실제로
발음하며 노래하므로, 코러스 음색으로 멜로디만 내는 것과는 완전히 다르다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import signal as scipy_signal

from ..audio.buffer import AudioBuffer, AudioError
from ..audio.oscillator import phase_from_frequency
from .phonemes import Phoneme, PhonemeKind, SungSyllable


class VoiceError(ValueError):
    """노래 합성 오류."""


# 포먼트의 대역폭(Hz). 좁을수록 또렷하고 넓을수록 뭉툭하다.
# 높은 포먼트일수록 넓다. 실제 성도가 그렇다.
FORMANT_BANDWIDTHS: tuple[float, ...] = (80.0, 100.0, 160.0, 250.0, 320.0)

# 4·5번째 포먼트는 모음과 거의 무관하고 목소리의 개성을 만든다.
HIGHER_FORMANTS: tuple[float, float] = (3800.0, 4600.0)


@dataclass(frozen=True, slots=True)
class VoiceTimbre:
    """목소리의 성질. 7번의 VOICE MODEL 이 담을 값들이다.

    formant_scale   성도 길이. 1보다 작으면 포먼트가 올라가 어리고 가늘게,
                    크면 내려가 굵고 낮게 들린다. 남녀 차이의 대부분이 이것이다.
    breathiness     숨 섞임. 높으면 속삭이듯, 낮으면 또렷하다.
    tension         성대 긴장. 높으면 날카롭고 낮으면 부드럽다.
    """

    name: str = "기본"
    formant_scale: float = 1.0
    breathiness: float = 0.16
    tension: float = 0.5
    brightness: float = 0.0        # -1 어두움 ~ +1 밝음
    vibrato_rate: float = 5.4      # Hz
    vibrato_depth: float = 28.0    # 센트
    vibrato_onset: float = 0.32    # 초
    portamento: float = 0.045      # 앞 음에서 미끄러져 오는 시간(초)

    def __post_init__(self) -> None:
        if not (0.5 <= self.formant_scale <= 1.6):
            raise VoiceError(f"성도 배율은 0.5~1.6 이어야 합니다: {self.formant_scale}")
        for label, value in (("breathiness", self.breathiness), ("tension", self.tension)):
            if not (0.0 <= value <= 1.0):
                raise VoiceError(f"{label} 은 0~1 이어야 합니다: {value}")
        if self.vibrato_depth < 0:
            raise VoiceError("비브라토 깊이는 0 이상이어야 합니다.")

    @classmethod
    def preset(cls, name: str) -> "VoiceTimbre":
        """대표적인 목소리. 학습 전에 고를 수 있게 한다."""
        table = {
            "soprano": dict(formant_scale=0.86, breathiness=0.14, tension=0.55,
                            brightness=0.25, vibrato_rate=5.8, vibrato_depth=34.0),
            "mezzo": dict(formant_scale=0.92, breathiness=0.16, tension=0.5,
                          brightness=0.1, vibrato_rate=5.5, vibrato_depth=30.0),
            "alto": dict(formant_scale=0.97, breathiness=0.18, tension=0.45,
                         brightness=0.0, vibrato_rate=5.2, vibrato_depth=26.0),
            "tenor": dict(formant_scale=1.08, breathiness=0.15, tension=0.55,
                          brightness=0.05, vibrato_rate=5.4, vibrato_depth=28.0),
            "baritone": dict(formant_scale=1.16, breathiness=0.16, tension=0.45,
                             brightness=-0.1, vibrato_rate=5.0, vibrato_depth=24.0),
            "bass": dict(formant_scale=1.26, breathiness=0.18, tension=0.4,
                         brightness=-0.2, vibrato_rate=4.8, vibrato_depth=22.0),
            "whisper": dict(formant_scale=0.95, breathiness=0.85, tension=0.25,
                            brightness=0.15, vibrato_depth=8.0),
            "power": dict(formant_scale=0.94, breathiness=0.08, tension=0.85,
                          brightness=0.35, vibrato_rate=6.0, vibrato_depth=38.0),
        }
        if name not in table:
            raise VoiceError(
                f"모르는 목소리입니다: {name!r} (가능: {', '.join(sorted(table))})"
            )
        return cls(name=name, **table[name])


def glottal_pulse(phase: np.ndarray, tension: float = 0.5) -> np.ndarray:
    """성대가 여닫히는 파형 (Rosenberg 모형).

    단순한 톱니나 사각파를 쓰면 안 된다. 성대의 실제 움직임은 '천천히 열리고
    빠르게 닫히는' 비대칭이고, 그 비대칭이 사람 목소리 특유의 배음 구조를
    만든다. 톱니파를 쓰면 신스 리드처럼 들린다.

    tension 이 높으면 닫히는 순간이 빨라져 배음이 늘고 날카로워진다.
    """
    normalized = (phase / (2.0 * np.pi)) % 1.0
    open_phase = 0.46 - 0.12 * tension       # 열리는 구간
    close_phase = 0.20 - 0.09 * tension      # 닫히는 구간
    result = np.zeros_like(normalized)

    rising = normalized < open_phase
    result[rising] = 0.5 * (1.0 - np.cos(np.pi * normalized[rising] / open_phase))

    falling = (normalized >= open_phase) & (normalized < open_phase + close_phase)
    result[falling] = np.cos(
        np.pi * (normalized[falling] - open_phase) / (2.0 * close_phase)
    )
    # 나머지는 닫혀 있어 0 이다
    return result - result.mean()


def _resonator(frequency: float, bandwidth: float, sample_rate: int,
               normalize: str = "dc"):
    """포먼트 하나를 만드는 2극 공명기 계수.

    정규화 방식이 중요하다.

    normalize="dc"    직류에서 이득 1. 공명점에서는 크게 솟는다.
                      이걸 직렬로 이어야 포먼트 합성기가 된다.
    normalize="peak"  공명점에서 이득 1. 병렬로 더해 쓸 때 맞다.

    직렬 연결에 "peak" 를 쓰면 안 된다. 각 공명기가 자기 주파수에서만 1 이고
    다른 데서는 훨씬 작으므로, 세 개를 이으면 전체 이득이 10^-9 수준으로
    사라진다. 모음 포먼트가 전부 죽고 소리가 나오지 않는다.
    """
    frequency = max(60.0, min(frequency, sample_rate * 0.48))
    radius = math.exp(-math.pi * bandwidth / sample_rate)
    theta = 2.0 * math.pi * frequency / sample_rate
    a1 = -2.0 * radius * math.cos(theta)
    a2 = radius * radius
    if normalize == "dc":
        gain = 1.0 + a1 + a2
    elif normalize == "peak":
        gain = (1.0 - radius) * math.sqrt(
            1.0 - 2.0 * radius * math.cos(2.0 * theta) + radius * radius
        )
    else:
        raise VoiceError(f"알 수 없는 정규화 방식입니다: {normalize!r}")
    return np.array([gain, 0.0, 0.0]), np.array([1.0, a1, a2])


class SingingSynth:
    """가사와 음높이를 받아 노래를 만든다."""

    BLOCK = 128       # 포먼트를 갱신하는 단위(샘플). 128이면 2.7ms 마다 바뀐다.

    def __init__(self, timbre: VoiceTimbre | None = None,
                 sample_rate: int = 48000) -> None:
        self.timbre = timbre or VoiceTimbre()
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------

    def _parameter_tracks(
        self, syllables: Sequence[SungSyllable], pitches_hz: Sequence[float],
        total_frames: int,
    ) -> dict[str, np.ndarray]:
        """시간에 따라 변하는 값들을 미리 만든다.

        포먼트는 음소가 바뀔 때 뚝 끊기면 안 된다. 실제 입은 서서히 움직인다.
        갑자기 바뀌면 딸깍거리는 소리가 나고 발음이 뭉개진다. 그래서 목표값을
        찍어 놓고 부드럽게 잇는다.
        """
        rate = self.sample_rate
        scale = self.timbre.formant_scale

        formants = np.zeros((3, total_frames))
        amplitude = np.zeros(total_frames)
        voicing = np.zeros(total_frames)      # 1 = 성대 울림, 0 = 바람소리만
        noise_level = np.zeros(total_frames)      # 성문 바람 (숨소리). 성도를 지난다.
        frication = np.zeros(total_frames)        # 입 안 좁은 곳의 바람 (ㅅ, 터짐). 성도를 안 지난다.
        noise_center = np.full(total_frames, 2000.0)
        nasal = np.zeros(total_frames)

        # 기본값: 중립 모음 자리
        formants[0, :] = 500.0 * scale
        formants[1, :] = 1500.0 * scale
        formants[2, :] = 2500.0 * scale

        for syllable in syllables:
            for phoneme in syllable.phonemes:
                start = int(round((syllable.note_start + phoneme.start) * rate))
                end = int(round((syllable.note_start + phoneme.end) * rate))
                start = max(0, start)
                end = min(total_frames, end)
                if end <= start:
                    continue
                span = slice(start, end)
                f1, f2, f3 = phoneme.formants

                if phoneme.kind is PhonemeKind.VOWEL:
                    formants[0, span] = f1 * scale
                    formants[1, span] = f2 * scale
                    formants[2, span] = f3 * scale
                    amplitude[span] = 1.0
                    voicing[span] = 1.0
                    noise_level[span] = self.timbre.breathiness * 0.35
                    noise_center[span] = 2600.0

                elif phoneme.kind is PhonemeKind.NASAL:
                    # 코로 울린다. 낮은 포먼트 하나가 강하고 나머지는 눌린다.
                    formants[0, span] = 260.0 * scale
                    formants[1, span] = 1100.0 * scale
                    formants[2, span] = 2400.0 * scale
                    amplitude[span] = 0.55
                    voicing[span] = 1.0
                    nasal[span] = 1.0
                    noise_level[span] = 0.02

                elif phoneme.kind is PhonemeKind.LIQUID:
                    formants[0, span] = 420.0 * scale
                    formants[1, span] = 1300.0 * scale
                    formants[2, span] = 2600.0 * scale
                    amplitude[span] = 0.7
                    voicing[span] = 1.0
                    noise_level[span] = 0.03

                elif phoneme.kind is PhonemeKind.PLOSIVE:
                    # 막았다 터뜨린다. 앞쪽은 무음, 끝에서 짧게 터진다.
                    # 터짐의 바람은 입 앞쪽에서 나므로 성도 공명을 지나지 않는다 (Klatt 병렬 가지).
                    burst_start = max(start, end - int(0.012 * rate))
                    amplitude[span] = 0.0
                    voicing[span] = 0.0
                    frication[burst_start:end] = 0.8 + 0.4 * phoneme.aspiration
                    noise_center[span] = phoneme.center_hz or 2000.0

                elif phoneme.kind in (PhonemeKind.FRICATIVE, PhonemeKind.AFFRICATE):
                    if phoneme.kind is PhonemeKind.AFFRICATE:
                        # 파찰음은 앞이 막음, 뒤가 마찰이다
                        stop_end = start + int((end - start) * 0.35)
                        amplitude[start:stop_end] = 0.0
                        voicing[start:stop_end] = 0.0
                        span = slice(stop_end, end)
                    amplitude[span] = 0.0
                    voicing[span] = 0.0
                    # 'ㅎ' 는 성문에서 나는 바람이라 성도를 지난다. 나머지는 입 안에서 난다.
                    if phoneme.symbol == "ㅎ":
                        amplitude[span] = 0.5
                        noise_level[span] = 1.0
                    else:
                        frication[span] = 1.0
                    noise_center[span] = phoneme.center_hz or 4000.0

        # 값이 뚝 끊기지 않게 잇는다. 5ms 이동평균이면 딸깍거림이 사라지고
        # 발음은 그대로 남는다.
        window = max(3, int(0.006 * rate))
        kernel = np.ones(window) / window
        for index in range(3):
            formants[index] = np.convolve(formants[index], kernel, mode="same")
        amplitude = np.convolve(amplitude, kernel, mode="same")
        voicing = np.convolve(voicing, kernel, mode="same")
        noise_level = np.convolve(noise_level, kernel, mode="same")
        frication = np.convolve(frication, kernel, mode="same")
        noise_center = np.convolve(noise_center, np.ones(window * 2) / (window * 2),
                                   mode="same")

        return {
            "formants": formants, "amplitude": amplitude, "voicing": voicing,
            "noise": noise_level, "frication": frication, "noise_center": noise_center,
            "nasal": nasal,
        }

    def _pitch_track(
        self, syllables: Sequence[SungSyllable], pitches_hz: Sequence[float],
        total_frames: int,
    ) -> np.ndarray:
        """음높이 곡선. 음 사이를 미끄러지고 비브라토가 걸린다.

        음높이가 계단처럼 바뀌면 기계 소리가 된다. 사람은 앞 음에서 다음 음으로
        아주 짧게 미끄러진다. 그 시간이 포르타멘토다.
        """
        rate = self.sample_rate
        track = np.zeros(total_frames)
        if not syllables:
            return track

        glide_frames = max(1, int(self.timbre.portamento * rate))
        previous_hz = pitches_hz[0]
        for syllable, target_hz in zip(syllables, pitches_hz):
            start = max(0, int(round(syllable.note_start * rate)))
            end = min(total_frames, int(round(
                (syllable.note_start + syllable.note_duration) * rate
            )))
            if end <= start:
                continue
            length = end - start
            glide = min(glide_frames, length)
            if abs(previous_hz - target_hz) > 0.5 and glide > 1:
                # 음높이는 로그로 들리므로 지수 보간이 자연스럽다
                progress = np.linspace(0.0, 1.0, glide)
                eased = progress * progress * (3.0 - 2.0 * progress)
                track[start:start + glide] = previous_hz * np.power(
                    target_hz / previous_hz, eased
                )
                track[start + glide:end] = target_hz
            else:
                track[start:end] = target_hz
            previous_hz = target_hz

        # 빈 구간은 앞 값으로 채운다 (0Hz 가 되면 위상 계산이 멈춘다)
        filled = track.copy()
        last = pitches_hz[0]
        for index in range(total_frames):
            if filled[index] <= 0:
                filled[index] = last
            else:
                last = filled[index]

        # 비브라토. 음이 시작하자마자 걸리지 않고 조금 늦게 들어온다.
        time = np.arange(total_frames) / rate
        onset = np.zeros(total_frames)
        for syllable in syllables:
            start = max(0, int(round(syllable.note_start * rate)))
            end = min(total_frames, int(round(
                (syllable.note_start + syllable.note_duration) * rate
            )))
            if end <= start:
                continue
            local = np.arange(end - start) / rate
            onset[start:end] = np.clip(local / max(1e-6, self.timbre.vibrato_onset), 0.0, 1.0)
        modulation = (
            np.sin(2.0 * np.pi * self.timbre.vibrato_rate * time)
            * (self.timbre.vibrato_depth / 100.0) * onset
        )
        return filled * np.power(2.0, modulation / 12.0)

    # ------------------------------------------------------------------

    def render(
        self, syllables: Sequence[SungSyllable], pitches_hz: Sequence[float],
        tail_seconds: float = 0.25,
    ) -> AudioBuffer:
        """노래를 만든다."""
        if len(syllables) != len(pitches_hz):
            raise VoiceError(
                f"음절 {len(syllables)}개와 음높이 {len(pitches_hz)}개가 맞지 않습니다."
            )
        if not syllables:
            return AudioBuffer.silence(0, 1, self.sample_rate)
        for hz in pitches_hz:
            if hz <= 0:
                raise VoiceError(f"음높이는 양수여야 합니다: {hz}")

        rate = self.sample_rate
        last = syllables[-1]
        total_seconds = last.note_start + last.note_duration + tail_seconds
        earliest = min((s.note_start + s.onset_offset for s in syllables), default=0.0)
        if earliest < 0:
            # 첫 자음이 0 보다 앞서면 그만큼 뒤로 민다
            shift = -earliest
            syllables = [
                SungSyllable(s.text, s.phonemes, s.note_start + shift, s.note_duration)
                for s in syllables
            ]
            total_seconds += shift
        total_frames = max(1, int(round(total_seconds * rate)))

        tracks = self._parameter_tracks(syllables, pitches_hz, total_frames)
        pitch = self._pitch_track(syllables, pitches_hz, total_frames)

        # --- 성대 소리 ---
        phase = phase_from_frequency(pitch, total_frames, rate)
        source = glottal_pulse(phase, self.timbre.tension)
        source *= tracks["voicing"]

        # --- 바람 소리 ---
        generator = np.random.default_rng(12345)
        noise = generator.standard_normal(total_frames) * 0.5
        oral_noise = generator.standard_normal(total_frames)

        # --- 포먼트 공명 ---
        # 계수가 계속 바뀌므로 블록마다 다시 만들어 상태를 이어 간다.
        formants = tracks["formants"]
        amplitude = tracks["amplitude"]
        noise_level = tracks["noise"]
        noise_center = tracks["noise_center"]
        nasal = tracks["nasal"]
        frication = tracks["frication"]

        output = np.zeros(total_frames)
        fricative = np.zeros(total_frames)
        states = [np.zeros(2) for _ in range(5)]
        noise_state = np.zeros(2)
        fricative_state = np.zeros(2)
        scale = self.timbre.formant_scale

        for begin in range(0, total_frames, self.BLOCK):
            end = min(begin + self.BLOCK, total_frames)
            middle = (begin + end) // 2

            block_source = source[begin:end]
            block_noise = noise[begin:end] * noise_level[begin:end]

            # 마찰음의 바람은 조음 위치에 맞춘 대역으로 거른다
            center = float(noise_center[middle])
            b, a = _resonator(center, max(400.0, center * 0.45), rate)
            filtered_noise, noise_state = scipy_signal.lfilter(
                b, a, block_noise, zi=noise_state
            )

            mixed = block_source + filtered_noise * 1.4

            # 입 안에서 나는 바람 (ㅅ, 터짐). 성도 공명을 지나지 않는 병렬 가지다.
            # 이걸 포먼트 직렬 사슬에 넣으면, 평평한 잡음이 높은 포먼트에서 40dB
            # 넘게 부풀어 모음보다 수십 배 큰 봉우리가 생긴다. Klatt 합성기가
            # 마찰음을 병렬 가지로 따로 내는 이유가 이것이다.
            block_fric = frication[begin:end]
            if block_fric.any() or fricative_state.any():
                b, a = _resonator(center, max(1000.0, center * 0.6), rate, normalize="peak")
                fricative[begin:end], fricative_state = scipy_signal.lfilter(
                    b, a, oral_noise[begin:end] * block_fric, zi=fricative_state
                )

            # 포먼트 다섯 개를 직렬로 잇는다. 성도는 하나의 관이고 공명이
            # 그 안에서 차례로 일어나므로, 병렬로 더하는 것보다 직렬이 맞다.
            # 1~3번은 모음마다 바뀌고, 4~5번은 고정이라 목소리의 개성을 만든다.
            block = mixed
            for index in range(5):
                if index < 3:
                    frequency = float(formants[index, middle])
                    bandwidth = FORMANT_BANDWIDTHS[index]
                    if nasal[middle] > 0.5:
                        bandwidth *= 1.6      # 코울림은 공명이 넓고 흐리다
                else:
                    frequency = HIGHER_FORMANTS[index - 3] * scale
                    bandwidth = FORMANT_BANDWIDTHS[index]
                b, a = _resonator(frequency, bandwidth, rate)
                block, states[index] = scipy_signal.lfilter(b, a, block, zi=states[index])

            output[begin:end] = block * amplitude[begin:end]

        # --- 입술에서 나오며 생기는 고역 강조 ---
        # 소리는 입 밖으로 나오면서 미분되는 효과를 받는다. 이게 없으면 먹먹하다.
        output = np.concatenate([[0.0], np.diff(output)]) + output * 0.25

        # 밝기 조정
        if abs(self.timbre.brightness) > 0.02:
            gain_db = self.timbre.brightness * 5.0
            sos = scipy_signal.butter(1, 3000.0, btype="highpass", fs=rate, output="sos")
            high = scipy_signal.sosfilt(sos, output)
            output = output + high * (10.0 ** (gain_db / 20.0) - 1.0)

        # 아주 낮은 성분은 들리지 않고 자리만 차지한다
        sos = scipy_signal.butter(2, 70.0, btype="highpass", fs=rate, output="sos")
        output = scipy_signal.sosfilt(sos, output)

        output = self._level_syllables(output, syllables)
        output = output + self._scale_frication(fricative, frication)
        return AudioBuffer.from_mono(self._soft_limit(output), rate)

    # 마찰음 크기. 모음보다 8dB 작게. 노래에서 'ㅅ' 이 안 들리면 발음이 뭉개지고,
    # 너무 크면 치찰음이 귀를 찌른다.
    FRICATION_RELATIVE = 0.4

    def _scale_frication(self, fricative: np.ndarray, track: np.ndarray) -> np.ndarray:
        """마찰 소리를 모음 크기에 맞춘다. 온전히 마찰하는 구간의 크기를 기준으로 한다."""
        steady = track > 0.9
        if not steady.any():
            steady = track > 0.3
        if not steady.any():
            return fricative
        level = float(np.sqrt(np.mean(fricative[steady] ** 2)))
        if level <= 1e-12:
            return fricative
        return fricative * (self.TARGET_VOWEL_RMS * self.FRICATION_RELATIVE / level)

    @staticmethod
    def _soft_limit(signal: np.ndarray, knee: float = 0.7) -> np.ndarray:
        """봉우리만 부드럽게 누른다. knee 까지는 그대로 둔다."""
        result = signal.copy()
        magnitude = np.abs(result)
        over = magnitude > knee
        if over.any():
            squeezed = knee + (1.0 - knee) * np.tanh((magnitude[over] - knee) / (1.0 - knee))
            result[over] = np.sign(result[over]) * squeezed * 0.98
        return result

    # 노래하는 모음의 목표 크기 (RMS). -14 dBFS 쯤이다. 반주 악기 한 대와
    # 비슷한 크기라서, 믹스에서 보컬이 묻히지 않는다.
    TARGET_VOWEL_RMS = 0.2

    def _level_syllables(self, output: np.ndarray,
                         syllables: Sequence[SungSyllable]) -> np.ndarray:
        """음절마다 모음 크기를 맞춘다.

        포먼트 공명은 모음마다, 음높이마다 전체 이득이 크게 다르다. 배음이
        포먼트 봉우리에 걸리면 커지고 비켜 가면 작아진다. 재 보면 같은 세기로
        불러도 '다' 와 '여' 가 20dB 넘게 차이 난다. 사람은 그렇게 부르지 않는다.
        가수는 모음이 바뀌어도 크기를 고르게 유지한다.

        또 전체를 가장 큰 순간(대개 자음 터짐) 기준으로 맞추면 나머지가 전부
        작아져서 반주에 묻힌다. 그래서 봉우리가 아니라 모음의 평균 크기로 맞추고,
        삐져나오는 자음 봉우리만 부드럽게 누른다.

        자음과 모음의 비율은 음절 안에서 그대로 둔다 (음절 전체에 같은 이득).
        """
        rate = self.sample_rate
        frames = len(output)
        spans: list[tuple[int, int, float]] = []
        for index, syllable in enumerate(syllables):
            vowel = syllable.vowel
            if vowel is None:
                continue
            a = max(0, int((syllable.note_start + vowel.start) * rate))
            b = min(frames, int((syllable.note_start + vowel.end) * rate))
            if b - a < int(0.02 * rate):
                continue
            # 평균이 아니라 10ms 창 크기들의 중앙값. 전이 구간의 순간 봉우리 하나가
            # 평균을 끌어올려 모음 전체를 작게 만드는 일을 막는다.
            window = max(1, int(0.01 * rate))
            segment = output[a:b]
            usable = len(segment) // window * window
            if usable >= window:
                frames_rms = np.sqrt(np.mean(segment[:usable].reshape(-1, window) ** 2, axis=1))
                level = float(np.median(frames_rms))
            else:
                level = float(np.sqrt(np.mean(segment ** 2)))
            if level <= 1e-9:
                continue
            begin = max(0, int((syllable.note_start + syllable.onset_offset) * rate))
            if index + 1 < len(syllables):
                following = syllables[index + 1]
                finish = int((following.note_start + following.onset_offset) * rate)
            else:
                finish = frames
            spans.append((begin, max(begin + 1, min(frames, finish)), level))

        if not spans:
            level = float(np.sqrt(np.mean(output ** 2))) if frames else 0.0
            return output * (self.TARGET_VOWEL_RMS / level) if level > 1e-12 else output

        gain = np.zeros(frames)
        covered = np.zeros(frames, dtype=bool)
        for begin, finish, level in spans:
            # 너무 작은 것을 끝없이 키우면 잡음까지 커진다. -12 ~ +24 dB 로 묶는다.
            value = float(np.clip(self.TARGET_VOWEL_RMS / level, 0.25, 16.0))
            gain[begin:finish] = value
            covered[begin:finish] = True
        # 음절 밖(꼬리, 첫 자음 앞)은 가장 가까운 음절의 이득을 쓴다
        if not covered.all():
            indices = np.where(covered)[0]
            nearest = np.searchsorted(indices, np.arange(frames)).clip(0, len(indices) - 1)
            gain = np.where(covered, gain, gain[indices[nearest]])
        # 이득이 음절 경계에서 뚝 바뀌면 딸깍 소리가 난다. 15ms 로 부드럽게.
        width = max(1, int(0.015 * rate))
        kernel = np.ones(width) / width
        gain = np.convolve(np.pad(gain, (width // 2, width - width // 2 - 1), mode="edge"),
                           kernel, mode="valid")
        return output * gain


def sing(
    text: str, note_times: Sequence[tuple[float, float]],
    pitches_hz: Sequence[float], timbre: VoiceTimbre | None = None,
    sample_rate: int = 48000,
) -> AudioBuffer:
    """가사와 음을 받아 바로 노래로 만든다."""
    from .phonemes import align_lyrics

    syllables = align_lyrics(text, note_times)
    return SingingSynth(timbre, sample_rate).render(syllables, pitches_hz)
