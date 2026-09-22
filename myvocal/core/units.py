"""
시간 좌표계 — MYVOCAL Studio 전체가 공유하는 단 하나의 시간 축.

음악(트랙/노트/코드)과 영상(Storyboard/Camera/Lip Sync)이 같은 좌표계를 쓴다.
그래야 62번 "음악 Timeline과 영상 Timeline을 동시에 편집" 이 성립한다.

세 가지 단위를 쓴다.

    tick    음악적 시간. PPQ 기준 정수. 템포가 바뀌어도 악보상 위치는 안 변한다.
    second  실제 시간. 영상/립싱크/렌더가 쓴다.
    sample  오디오 샘플 인덱스. 렌더러가 쓴다.

TempoMap 이 이 셋을 서로 변환한다. 템포 변화(Accelerando 포함)를 지원한다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Sequence

# 4분음표 하나당 tick 수. 셋잇단(3), 5잇단, 7잇단, 64분음표까지 정수로 떨어지도록 선택.
# 960 = 2^6 * 3 * 5  ->  64분음표(60tick), 3연음(320tick), 5연음(192tick), 7연음은 근사.
PPQ: int = 960

# 내부 오디오 처리 샘플레이트 기본값. 프로젝트별로 바뀔 수 있다.
DEFAULT_SAMPLE_RATE: int = 48000


class TimeError(ValueError):
    """시간 좌표계 관련 오류."""


def beats_to_ticks(beats: float) -> int:
    """박(beat) 수를 tick 으로. 4분음표 = 1 beat 기준."""
    return int(round(beats * PPQ))


def ticks_to_beats(ticks: int) -> float:
    return ticks / PPQ


def note_value_to_ticks(numerator: int, denominator: int) -> int:
    """음표 길이를 tick 으로. (1, 4) = 4분음표, (3, 8) = 점4분음표, (1, 3) = 2분음표 셋잇단.

    denominator 는 '온음표를 몇 등분했는가'다. 4분음표 = 1/4 온음표 = PPQ tick.
    """
    if numerator <= 0 or denominator <= 0:
        raise TimeError(f"음표 길이는 양수여야 합니다: {numerator}/{denominator}")
    ticks = Fraction(numerator, denominator) * 4 * PPQ
    if ticks.denominator != 1:
        # 정수로 안 떨어지면 반올림하되, 오차가 1 tick 을 넘으면 알린다.
        rounded = int(round(float(ticks)))
        if abs(float(ticks) - rounded) > 0.5:
            raise TimeError(f"PPQ={PPQ} 로 표현 불가능한 음표 길이: {numerator}/{denominator}")
        return rounded
    return int(ticks)


@dataclass(frozen=True, slots=True)
class TimeSignature:
    """박자표. 4/4, 3/4, 6/8, 7/8 등."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise TimeError(f"박자표 분자는 양수여야 합니다: {self.numerator}")
        if self.denominator not in (1, 2, 4, 8, 16, 32):
            raise TimeError(f"지원하지 않는 박자표 분모입니다: {self.denominator}")

    @property
    def ticks_per_bar(self) -> int:
        """한 마디의 tick 수."""
        return note_value_to_ticks(self.numerator, self.denominator)

    @property
    def ticks_per_beat(self) -> int:
        """박자표가 정의하는 1박의 tick 수. 6/8 에서는 8분음표가 1박."""
        return note_value_to_ticks(1, self.denominator)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True, slots=True)
class TempoChange:
    """특정 tick 위치에서의 템포. curve='linear' 이면 다음 변화점까지 서서히 변한다."""

    tick: int
    bpm: float
    curve: str = "step"  # "step" | "linear"

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise TimeError(f"템포 변화 위치는 0 이상이어야 합니다: {self.tick}")
        if not (1.0 <= self.bpm <= 999.0):
            raise TimeError(f"BPM 범위를 벗어났습니다 (1~999): {self.bpm}")
        if self.curve not in ("step", "linear"):
            raise TimeError(f"알 수 없는 템포 커브입니다: {self.curve}")


@dataclass(frozen=True, slots=True)
class MeterChange:
    """특정 마디에서의 박자표 변경."""

    tick: int
    signature: TimeSignature


class TempoMap:
    """tick <-> second 변환기.

    템포 변화 목록을 들고 각 변화점의 누적 시간을 미리 계산해 둔다.
    변환은 이진 탐색이라 노트가 수만 개여도 빠르다.
    """

    __slots__ = ("_changes", "_cum_seconds", "_ticks")

    def __init__(self, changes: Iterable[TempoChange] | None = None) -> None:
        raw = sorted(changes or [], key=lambda c: c.tick)
        if not raw or raw[0].tick != 0:
            raw = [TempoChange(0, 120.0)] + [c for c in raw if c.tick != 0]
        # 같은 tick 에 여러 개면 마지막 것만 쓴다.
        deduped: list[TempoChange] = []
        for change in raw:
            if deduped and deduped[-1].tick == change.tick:
                deduped[-1] = change
            else:
                deduped.append(change)
        self._changes: tuple[TempoChange, ...] = tuple(deduped)
        self._ticks: tuple[int, ...] = tuple(c.tick for c in deduped)
        self._cum_seconds: tuple[float, ...] = self._build_cumulative()

    @property
    def changes(self) -> tuple[TempoChange, ...]:
        return self._changes

    @property
    def initial_bpm(self) -> float:
        return self._changes[0].bpm

    def _build_cumulative(self) -> tuple[float, ...]:
        cumulative = [0.0]
        for i in range(len(self._changes) - 1):
            span = self._segment_seconds(i, self._changes[i + 1].tick)
            cumulative.append(cumulative[-1] + span)
        return tuple(cumulative)

    def _segment_seconds(self, index: int, end_tick: int) -> float:
        """index 번째 템포 구간의 시작부터 end_tick 까지 걸리는 실제 초."""
        start = self._changes[index]
        delta_ticks = end_tick - start.tick
        if delta_ticks <= 0:
            return 0.0
        beats = delta_ticks / PPQ

        next_change = self._changes[index + 1] if index + 1 < len(self._changes) else None
        if start.curve == "linear" and next_change is not None:
            # 템포가 구간 안에서 선형으로 변한다. 박 위치에 대해 BPM 이 선형이면
            # 소요 시간은 로그 적분이 된다:  t = 60 * (b1-b0) / (B1-B0) * ln(B1/B0)
            total_beats = (next_change.tick - start.tick) / PPQ
            if total_beats > 0 and abs(next_change.bpm - start.bpm) > 1e-9:
                slope = (next_change.bpm - start.bpm) / total_beats
                bpm_at_end = start.bpm + slope * beats
                if bpm_at_end <= 0:
                    raise TimeError("템포 커브가 0 이하 BPM 으로 내려갑니다.")
                import math

                return 60.0 / slope * math.log(bpm_at_end / start.bpm)
        return beats * 60.0 / start.bpm

    def bpm_at_tick(self, tick: int) -> float:
        """해당 tick 위치에서의 순간 BPM."""
        if tick < 0:
            raise TimeError(f"tick 은 0 이상이어야 합니다: {tick}")
        index = max(0, bisect.bisect_right(self._ticks, tick) - 1)
        start = self._changes[index]
        next_change = self._changes[index + 1] if index + 1 < len(self._changes) else None
        if start.curve == "linear" and next_change is not None:
            total_beats = (next_change.tick - start.tick) / PPQ
            if total_beats > 0:
                progress = (tick - start.tick) / PPQ / total_beats
                return start.bpm + (next_change.bpm - start.bpm) * min(1.0, max(0.0, progress))
        return start.bpm

    def tick_to_seconds(self, tick: int | float) -> float:
        if tick < 0:
            raise TimeError(f"tick 은 0 이상이어야 합니다: {tick}")
        index = max(0, bisect.bisect_right(self._ticks, tick) - 1)
        return self._cum_seconds[index] + self._segment_seconds(index, int(tick))

    def seconds_to_tick(self, seconds: float) -> int:
        """실제 시간 -> tick. 영상 쪽에서 '이 장면은 몇 마디인가'를 물을 때 쓴다."""
        if seconds < 0:
            raise TimeError(f"시간은 0 이상이어야 합니다: {seconds}")
        index = max(0, bisect.bisect_right(self._cum_seconds, seconds) - 1)
        remaining = seconds - self._cum_seconds[index]
        start = self._changes[index]
        next_change = self._changes[index + 1] if index + 1 < len(self._changes) else None

        if start.curve == "linear" and next_change is not None:
            total_beats = (next_change.tick - start.tick) / PPQ
            if total_beats > 0 and abs(next_change.bpm - start.bpm) > 1e-9:
                import math

                slope = (next_change.bpm - start.bpm) / total_beats
                # 위 적분의 역함수
                beats = start.bpm / slope * (math.exp(slope * remaining / 60.0) - 1.0)
                return start.tick + int(round(beats * PPQ))
        beats = remaining * start.bpm / 60.0
        return start.tick + int(round(beats * PPQ))

    def tick_to_sample(self, tick: int | float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
        return int(round(self.tick_to_seconds(tick) * sample_rate))

    def sample_to_tick(self, sample: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
        return self.seconds_to_tick(sample / sample_rate)

    def with_change(self, change: TempoChange) -> "TempoMap":
        """템포 변화를 추가한 새 TempoMap. (불변 객체라 새로 만든다)"""
        kept = [c for c in self._changes if c.tick != change.tick]
        return TempoMap(kept + [change])

    def __repr__(self) -> str:
        parts = ", ".join(f"{c.tick}:{c.bpm:g}" for c in self._changes)
        return f"TempoMap({parts})"


class MeterMap:
    """tick <-> 마디/박 변환기. Storyboard 가 '2절 시작 마디'를 말할 때 쓴다."""

    __slots__ = ("_changes", "_ticks", "_cum_bars")

    def __init__(self, changes: Iterable[MeterChange] | None = None) -> None:
        raw = sorted(changes or [], key=lambda c: c.tick)
        if not raw or raw[0].tick != 0:
            raw = [MeterChange(0, TimeSignature(4, 4))] + [c for c in raw if c.tick != 0]
        deduped: list[MeterChange] = []
        for change in raw:
            if deduped and deduped[-1].tick == change.tick:
                deduped[-1] = change
            else:
                deduped.append(change)
        self._changes: tuple[MeterChange, ...] = tuple(deduped)
        self._ticks: tuple[int, ...] = tuple(c.tick for c in deduped)

        cum: list[int] = [0]
        for i in range(len(deduped) - 1):
            span = deduped[i + 1].tick - deduped[i].tick
            bar_ticks = deduped[i].signature.ticks_per_bar
            cum.append(cum[-1] + span // bar_ticks)
        self._cum_bars: tuple[int, ...] = tuple(cum)

    @property
    def changes(self) -> tuple[MeterChange, ...]:
        return self._changes

    def signature_at_tick(self, tick: int) -> TimeSignature:
        index = max(0, bisect.bisect_right(self._ticks, tick) - 1)
        return self._changes[index].signature

    def tick_to_bar_beat(self, tick: int) -> tuple[int, float]:
        """tick -> (마디 번호, 마디 안에서의 박). 마디와 박 모두 1부터 시작한다."""
        if tick < 0:
            raise TimeError(f"tick 은 0 이상이어야 합니다: {tick}")
        index = max(0, bisect.bisect_right(self._ticks, tick) - 1)
        change = self._changes[index]
        offset = tick - change.tick
        bar_ticks = change.signature.ticks_per_bar
        bars = offset // bar_ticks
        rest = offset % bar_ticks
        beat = rest / change.signature.ticks_per_beat + 1.0
        return self._cum_bars[index] + bars + 1, beat

    def bar_to_tick(self, bar: int) -> int:
        """마디 번호(1부터) -> 그 마디 첫 tick."""
        if bar < 1:
            raise TimeError(f"마디 번호는 1 이상이어야 합니다: {bar}")
        target = bar - 1
        index = max(0, bisect.bisect_right(self._cum_bars, target) - 1)
        change = self._changes[index]
        return change.tick + (target - self._cum_bars[index]) * change.signature.ticks_per_bar

    def bars_in_range(self, start_tick: int, end_tick: int) -> int:
        """구간에 들어 있는 마디 수 (내림)."""
        start_bar, _ = self.tick_to_bar_beat(start_tick)
        end_bar, _ = self.tick_to_bar_beat(max(start_tick, end_tick - 1))
        return end_bar - start_bar + 1

    def __repr__(self) -> str:
        parts = ", ".join(f"{c.tick}:{c.signature}" for c in self._changes)
        return f"MeterMap({parts})"


@dataclass(frozen=True, slots=True)
class TimeRange:
    """구간. 음악 구간(Verse/Chorus)과 영상 장면(Scene) 둘 다 이걸 쓴다."""

    start_tick: int
    end_tick: int

    def __post_init__(self) -> None:
        if self.start_tick < 0:
            raise TimeError(f"구간 시작은 0 이상이어야 합니다: {self.start_tick}")
        if self.end_tick <= self.start_tick:
            raise TimeError(
                f"구간 끝은 시작보다 커야 합니다: start={self.start_tick} end={self.end_tick}"
            )

    @property
    def length_ticks(self) -> int:
        return self.end_tick - self.start_tick

    def contains(self, tick: int) -> bool:
        return self.start_tick <= tick < self.end_tick

    def overlaps(self, other: "TimeRange") -> bool:
        return self.start_tick < other.end_tick and other.start_tick < self.end_tick

    def intersection(self, other: "TimeRange") -> "TimeRange | None":
        start = max(self.start_tick, other.start_tick)
        end = min(self.end_tick, other.end_tick)
        return TimeRange(start, end) if end > start else None

    def shifted(self, delta_ticks: int) -> "TimeRange":
        return TimeRange(self.start_tick + delta_ticks, self.end_tick + delta_ticks)

    def seconds(self, tempo: TempoMap) -> tuple[float, float]:
        return tempo.tick_to_seconds(self.start_tick), tempo.tick_to_seconds(self.end_tick)


def format_timecode(seconds: float) -> str:
    """0:00 / 1:23.5 형태. Storyboard 표시에 쓴다."""
    if seconds < 0:
        raise TimeError(f"시간은 0 이상이어야 합니다: {seconds}")
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}:{rest:04.1f}"
