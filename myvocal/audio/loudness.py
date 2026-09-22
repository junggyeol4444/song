"""
라우드니스 측정 — ITU-R BS.1770-4 / EBU R128.

"소리가 얼마나 크게 들리는가"를 재는 국제 표준이다. 최대값(peak)이나 실효값(RMS)
과는 다르다. 사람 귀는 주파수마다 민감도가 다르므로, K-weighting 이라는 필터를
먼저 걸고 재야 실제 체감 음량과 맞는다.

이게 필요한 이유:
  스트리밍 서비스들은 업로드된 곡을 자기네 기준 음량으로 자동 조절한다.
  기준보다 크게 만들어 올리면 서비스가 음량을 깎는데, 이때 음압만 잃고
  음질은 이미 뭉개진 상태가 된다. 그래서 마스터링은 peak 가 아니라
  이 값을 보고 해야 한다.

  Spotify / YouTube  약 -14 LUFS
  Apple Music        약 -16 LUFS
  방송 (EBU R128)    -23 LUFS
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from .buffer import AudioBuffer, AudioError

# BS.1770-4 가 규정한 K-weighting 필터 상수
_SHELF_FREQ = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_SHELF_VB_EXP = 0.4996667741545416

_HIGHPASS_FREQ = 38.13547087602444
_HIGHPASS_Q = 0.5003270373238773

# 측정값을 dB 로 바꿀 때 더하는 보정. 표준이 정한 값이다.
_OFFSET_DB = -0.691

# 블록 길이. 표준이 정한 값.
_MOMENTARY_SECONDS = 0.400     # 순간 라우드니스
_SHORT_TERM_SECONDS = 3.000    # 단기 라우드니스
_BLOCK_OVERLAP = 0.75          # 75% 겹침

_ABSOLUTE_GATE_LUFS = -70.0    # 이보다 조용한 블록은 무음으로 보고 뺀다
_RELATIVE_GATE_DB = -10.0      # 전체 평균보다 10dB 이상 조용한 블록도 뺀다

# 채널별 가중치. 서라운드 뒤쪽 채널은 더 크게 센다.
# 순서: L, R, C, LFE, Ls, Rs
_CHANNEL_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 1.0, 0.0, 1.41, 1.41)

# 스트리밍 기준 (2024~2025 기준 공개 정보. 서비스가 바꿀 수 있다)
STREAMING_TARGETS: dict[str, float] = {
    "spotify": -14.0,
    "youtube": -14.0,
    "apple_music": -16.0,
    "amazon_music": -14.0,
    "tidal": -14.0,
    "soundcloud": -14.0,
    "broadcast_ebu_r128": -23.0,
    "broadcast_atsc_a85": -24.0,
}


def k_weighting_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """K-weighting 2단 필터 계수를 만든다.

    표준 문서에는 48kHz 계수만 표로 나와 있다. 다른 샘플레이트에서도 맞으려면
    아날로그 원형에서 쌍선형 변환으로 다시 설계해야 한다. 여기서 그걸 한다.

    돌려주는 값은 (b, a) 두 벌을 세로로 쌓은 배열이다. 1단이 고역 셸빙,
    2단이 저역 차단(RLB)이다.
    """
    if sample_rate <= 0:
        raise AudioError(f"샘플레이트는 양수여야 합니다: {sample_rate}")
    if sample_rate < 2 * _SHELF_FREQ:
        raise AudioError(
            f"샘플레이트가 너무 낮아 K-weighting 을 설계할 수 없습니다: {sample_rate}Hz "
            f"(최소 {int(2 * _SHELF_FREQ) + 1}Hz 필요)"
        )

    # 1단: 고역 셸빙
    k = math.tan(math.pi * _SHELF_FREQ / sample_rate)
    vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    vb = vh ** _SHELF_VB_EXP
    denom = 1.0 + k / _SHELF_Q + k * k
    shelf_b = np.array([
        (vh + vb * k / _SHELF_Q + k * k) / denom,
        2.0 * (k * k - vh) / denom,
        (vh - vb * k / _SHELF_Q + k * k) / denom,
    ])
    shelf_a = np.array([
        1.0,
        2.0 * (k * k - 1.0) / denom,
        (1.0 - k / _SHELF_Q + k * k) / denom,
    ])

    # 2단: 저역 차단 (RLB)
    k2 = math.tan(math.pi * _HIGHPASS_FREQ / sample_rate)
    denom2 = 1.0 + k2 / _HIGHPASS_Q + k2 * k2
    hp_b = np.array([1.0, -2.0, 1.0])
    hp_a = np.array([
        1.0,
        2.0 * (k2 * k2 - 1.0) / denom2,
        (1.0 - k2 / _HIGHPASS_Q + k2 * k2) / denom2,
    ])

    return np.vstack([shelf_b, hp_b]), np.vstack([shelf_a, hp_a])


def apply_k_weighting(buffer: AudioBuffer) -> np.ndarray:
    """K-weighting 을 걸어 float64 배열로 돌려준다."""
    b_stack, a_stack = k_weighting_coefficients(buffer.sample_rate)
    signal = buffer.data.astype(np.float64)
    for b, a in zip(b_stack, a_stack):
        signal = scipy_signal.lfilter(b, a, signal, axis=1)
    return signal


def _channel_weights(channels: int) -> np.ndarray:
    if channels <= len(_CHANNEL_WEIGHTS):
        return np.array(_CHANNEL_WEIGHTS[:channels])
    # 표준에 없는 채널 수면 나머지는 1.0 으로 센다
    extra = [1.0] * (channels - len(_CHANNEL_WEIGHTS))
    return np.array(list(_CHANNEL_WEIGHTS) + extra)


def _block_mean_squares(
    weighted: np.ndarray, sample_rate: int, block_seconds: float, overlap: float
) -> np.ndarray:
    """겹치는 블록마다 채널별 평균제곱을 구한다. shape = (블록 수, 채널 수)."""
    block_frames = int(round(block_seconds * sample_rate))
    if block_frames <= 0:
        raise AudioError("블록 길이가 0 입니다.")
    step = max(1, int(round(block_frames * (1.0 - overlap))))
    total = weighted.shape[1]
    if total < block_frames:
        return np.empty((0, weighted.shape[0]))

    squared = np.square(weighted)
    # 누적합으로 모든 블록의 합을 한 번에 구한다. 블록마다 다시 더하면 느리다.
    cumulative = np.concatenate(
        [np.zeros((squared.shape[0], 1)), np.cumsum(squared, axis=1)], axis=1
    )
    starts = np.arange(0, total - block_frames + 1, step)
    sums = cumulative[:, starts + block_frames] - cumulative[:, starts]
    return (sums / block_frames).T


def _blocks_to_lufs(mean_squares: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """블록별 평균제곱 -> 블록별 LUFS."""
    if mean_squares.size == 0:
        return np.empty(0)
    total = mean_squares @ weights
    with np.errstate(divide="ignore"):
        return _OFFSET_DB + 10.0 * np.log10(np.maximum(total, 1e-30))


@dataclass(frozen=True, slots=True)
class LoudnessReport:
    """한 곡의 라우드니스 측정 결과."""

    integrated_lufs: float          # 곡 전체 (게이팅 적용). 마스터링이 맞추는 값.
    loudness_range_lu: float        # 조용한 데와 큰 데의 차이. 다이내믹의 척도.
    max_momentary_lufs: float       # 400ms 기준 최대
    max_short_term_lufs: float      # 3초 기준 최대
    true_peak_dbtp: float           # 샘플 사이까지 본 최대값
    sample_peak_dbfs: float
    duration_seconds: float

    def headroom_to(self, target_lufs: float) -> float:
        """목표 음량까지 몇 dB 올려야/내려야 하는가. 양수면 올려야 한다."""
        return target_lufs - self.integrated_lufs

    def check_streaming(self, service: str) -> str:
        """스트리밍 기준과 비교한 한 줄 진단."""
        key = service.lower().replace(" ", "_").replace("-", "_")
        if key not in STREAMING_TARGETS:
            raise AudioError(
                f"모르는 서비스입니다: {service} "
                f"(사용 가능: {', '.join(sorted(STREAMING_TARGETS))})"
            )
        target = STREAMING_TARGETS[key]
        difference = self.integrated_lufs - target
        if abs(difference) <= 0.5:
            verdict = "기준에 맞습니다"
        elif difference > 0:
            verdict = f"{difference:.1f}dB 큽니다. 서비스가 음량을 깎습니다"
        else:
            verdict = f"{-difference:.1f}dB 작습니다. 다른 곡보다 조용하게 들립니다"
        peak_note = ""
        if self.true_peak_dbtp > -1.0:
            peak_note = f" / True Peak {self.true_peak_dbtp:+.1f}dBTP — -1.0 이하 권장"
        return f"{service}: {self.integrated_lufs:.1f} LUFS (기준 {target:.1f}) — {verdict}{peak_note}"

    def summary(self) -> str:
        return (
            f"Integrated {self.integrated_lufs:.1f} LUFS / "
            f"Range {self.loudness_range_lu:.1f} LU / "
            f"Short-term max {self.max_short_term_lufs:.1f} / "
            f"True Peak {self.true_peak_dbtp:+.1f} dBTP"
        )


def integrated_loudness(buffer: AudioBuffer) -> float:
    """곡 전체 라우드니스(LUFS). BS.1770-4 의 2단계 게이팅을 적용한다.

    게이팅이 필요한 이유: 곡 중간의 무음이나 아주 조용한 부분까지 평균에 넣으면
    실제보다 조용하게 측정된다. 그래서 (1) 절대 -70 LUFS 미만 블록을 빼고,
    (2) 남은 것의 평균보다 10dB 이상 조용한 블록도 뺀 뒤 다시 평균낸다.
    """
    if buffer.frames == 0:
        return float("-inf")
    weighted = apply_k_weighting(buffer)
    weights = _channel_weights(buffer.channels)
    mean_squares = _block_mean_squares(
        weighted, buffer.sample_rate, _MOMENTARY_SECONDS, _BLOCK_OVERLAP
    )
    if mean_squares.shape[0] == 0:
        # 400ms 도 안 되는 짧은 소리는 블록을 못 만든다. 전체를 한 블록으로 본다.
        whole = np.mean(np.square(weighted), axis=1)[np.newaxis, :]
        value = float(whole @ weights)
        return _OFFSET_DB + 10.0 * math.log10(max(value, 1e-30))

    block_lufs = _blocks_to_lufs(mean_squares, weights)

    # 1단계: 절대 게이트
    passed = block_lufs > _ABSOLUTE_GATE_LUFS
    if not np.any(passed):
        return float("-inf")

    # 2단계: 상대 게이트
    gated_mean = float(np.mean(mean_squares[passed] @ weights))
    relative_threshold = _OFFSET_DB + 10.0 * math.log10(max(gated_mean, 1e-30)) + _RELATIVE_GATE_DB
    passed = passed & (block_lufs > relative_threshold)
    if not np.any(passed):
        return float("-inf")

    final_mean = float(np.mean(mean_squares[passed] @ weights))
    return _OFFSET_DB + 10.0 * math.log10(max(final_mean, 1e-30))


def loudness_range(buffer: AudioBuffer) -> float:
    """라우드니스 레인지(LU). EBU Tech 3342.

    3초 블록들의 라우드니스 분포에서 하위 10% ~ 상위 95% 구간의 폭을 잰다.
    이 값이 작으면 처음부터 끝까지 음량이 평평하다는 뜻이다. 과도한 압축의 신호다.
    """
    if buffer.frames == 0:
        return 0.0
    weighted = apply_k_weighting(buffer)
    weights = _channel_weights(buffer.channels)
    mean_squares = _block_mean_squares(weighted, buffer.sample_rate, _SHORT_TERM_SECONDS, 0.90)
    if mean_squares.shape[0] < 2:
        return 0.0
    block_lufs = _blocks_to_lufs(mean_squares, weights)

    passed = block_lufs > _ABSOLUTE_GATE_LUFS
    if np.count_nonzero(passed) < 2:
        return 0.0
    gated_mean = float(np.mean(mean_squares[passed] @ weights))
    # LRA 의 상대 게이트는 -20 LU 다 (통합 라우드니스의 -10 LU 와 다르다)
    threshold = _OFFSET_DB + 10.0 * math.log10(max(gated_mean, 1e-30)) - 20.0
    selected = block_lufs[passed & (block_lufs > threshold)]
    if selected.size < 2:
        return 0.0
    low, high = np.percentile(selected, [10.0, 95.0])
    return float(high - low)


def max_windowed_loudness(buffer: AudioBuffer, window_seconds: float) -> float:
    """지정 길이 창에서의 최대 라우드니스. 순간(0.4초)/단기(3초)에 쓴다."""
    if buffer.frames == 0:
        return float("-inf")
    weighted = apply_k_weighting(buffer)
    weights = _channel_weights(buffer.channels)
    mean_squares = _block_mean_squares(weighted, buffer.sample_rate, window_seconds, _BLOCK_OVERLAP)
    if mean_squares.shape[0] == 0:
        return integrated_loudness(buffer)
    return float(np.max(_blocks_to_lufs(mean_squares, weights)))


def measure(buffer: AudioBuffer) -> LoudnessReport:
    """한 번에 전부 측정한다."""
    return LoudnessReport(
        integrated_lufs=integrated_loudness(buffer),
        loudness_range_lu=loudness_range(buffer),
        max_momentary_lufs=max_windowed_loudness(buffer, _MOMENTARY_SECONDS),
        max_short_term_lufs=max_windowed_loudness(buffer, _SHORT_TERM_SECONDS),
        true_peak_dbtp=buffer.true_peak_db(oversample=4),
        sample_peak_dbfs=buffer.peak_db(),
        duration_seconds=buffer.duration,
    )


def normalize_to_lufs(
    buffer: AudioBuffer, target_lufs: float, true_peak_ceiling_db: float = -1.0
) -> tuple[AudioBuffer, float]:
    """목표 라우드니스로 맞춘다. True Peak 한계를 넘으면 그만큼 덜 올린다.

    돌려주는 값은 (조절된 오디오, 실제로 적용한 dB) 다. 목표에 못 미치면
    적용 dB 를 보고 알 수 있다.
    """
    current = integrated_loudness(buffer)
    if current == float("-inf"):
        return buffer.copy(), 0.0
    gain_db = target_lufs - current

    from .buffer import db_to_linear, linear_to_db

    predicted_peak_db = buffer.true_peak_db(oversample=4) + gain_db
    if predicted_peak_db > true_peak_ceiling_db:
        gain_db -= predicted_peak_db - true_peak_ceiling_db
    return buffer.with_gain(db_to_linear(gain_db)), gain_db


def master_to_lufs(
    buffer: AudioBuffer,
    target_lufs: float = -14.0,
    true_peak_ceiling_db: float = -1.0,
    max_limiting_db: float = 6.0,
    passes: int = 3,
) -> tuple[AudioBuffer, "MasteringResult"]:
    """목표 라우드니스까지 실제로 끌어올린다.

    normalize_to_lufs 와 다르다. 그쪽은 피크가 천장에 닿으면 이득을 줄여서
    목표에 못 미친 채로 끝난다. 안전하지만 곡이 다른 곡보다 조용해진다.

    실제 마스터링은 그렇게 하지 않는다. 이득을 올리고, 넘치는 순간만 리미터로
    눌러서 천장을 지키면서 평균 음량을 올린다. 여기서 그걸 한다.

    라우드니스는 리미팅 뒤에 다시 재야 정확하다. 리미터가 파형을 바꾸기
    때문이다. 그래서 몇 번 반복해서 목표에 맞춘다.

    max_limiting_db 는 안전장치다. 이보다 많이 눌러야 목표에 닿는다면,
    그건 곡 자체가 목표보다 훨씬 조용하거나 다이내믹이 큰 것이다. 억지로
    누르면 소리가 납작해지므로 거기서 멈추고 결과에 알린다.
    """
    if passes < 1:
        raise AudioError(f"반복 횟수는 1 이상이어야 합니다: {passes}")
    if max_limiting_db < 0:
        raise AudioError(f"최대 리미팅 양은 0 이상이어야 합니다: {max_limiting_db}")

    from . import dsp
    from .buffer import db_to_linear

    original = measure(buffer)
    if original.integrated_lufs == float("-inf"):
        return buffer.copy(), MasteringResult(
            applied_gain_db=0.0, limiting_db=0.0, reached_target=False,
            before=original, after=original,
            note="무음이라 라우드니스를 맞출 수 없습니다.",
        )

    working = buffer
    total_gain = 0.0
    total_limiting = 0.0
    note = ""

    for _ in range(passes):
        current = integrated_loudness(working)
        remaining = target_lufs - current
        if abs(remaining) < 0.1:
            break
        # 음량을 낮춰야 하면 리미터가 필요 없다. 그냥 낮춘다.
        if remaining < 0:
            working = working.with_gain(db_to_linear(remaining))
            total_gain += remaining
            continue
        # 올릴 때는, 천장을 넘는 만큼만 리미터가 누르게 한다.
        headroom = true_peak_ceiling_db - working.true_peak_db(oversample=4)
        limiting_needed = max(0.0, remaining - headroom)
        allowed = min(limiting_needed, max(0.0, max_limiting_db - total_limiting))
        step = headroom + allowed
        if step <= 0.0:
            note = (
                f"허용한 리미팅 {max_limiting_db:.1f}dB 를 다 썼습니다. "
                f"목표까지 {target_lufs - current:.1f}dB 남았습니다."
            )
            break
        working = working.with_gain(db_to_linear(step))
        total_gain += step
        if allowed > 0:
            working, reduction = dsp.limit(
                working, ceiling_db=true_peak_ceiling_db, lookahead_ms=5.0, release_ms=60.0
            )
            total_limiting += min(allowed, reduction)

    # 마지막으로 천장을 확실히 지킨다
    working, final_reduction = dsp.limit(
        working, ceiling_db=true_peak_ceiling_db, lookahead_ms=5.0, release_ms=60.0
    )
    total_limiting += final_reduction

    after = measure(working)
    reached = abs(after.integrated_lufs - target_lufs) <= 0.5
    if not reached and not note:
        note = (
            f"목표 {target_lufs:.1f} LUFS 에 {after.integrated_lufs - target_lufs:+.1f}dB "
            f"못 미쳤습니다."
        )
    if after.loudness_range_lu < 3.0 and original.loudness_range_lu >= 3.0:
        note += (
            f" 다이내믹이 {original.loudness_range_lu:.1f} -> "
            f"{after.loudness_range_lu:.1f} LU 로 줄었습니다. 과하게 눌렸을 수 있습니다."
        )
    return working, MasteringResult(
        applied_gain_db=total_gain,
        limiting_db=total_limiting,
        reached_target=reached,
        before=original,
        after=after,
        note=note.strip(),
    )


@dataclass(frozen=True, slots=True)
class MasteringResult:
    """마스터링 결과 보고. 무슨 일이 있었는지 숫자로 남긴다."""

    applied_gain_db: float
    limiting_db: float
    reached_target: bool
    before: LoudnessReport
    after: LoudnessReport
    note: str = ""

    def summary(self) -> str:
        lines = [
            f"이득 {self.applied_gain_db:+.1f}dB, 리미팅 {self.limiting_db:.1f}dB",
            f"전: {self.before.summary()}",
            f"후: {self.after.summary()}",
        ]
        if self.note:
            lines.append(f"참고: {self.note}")
        return "\n".join(lines)
