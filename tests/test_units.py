"""시간 좌표계 검증.

기대값은 전부 손으로 계산할 수 있는 값이다. 코드가 뱉은 값을 기대값으로
적어두면 검증이 아니라 '현재 동작 박제'가 되므로, 음악 이론상 맞는 값만 쓴다.
"""

import math
import sys

sys.path.insert(0, ".")

from myvocal.core.units import (
    PPQ,
    MeterChange,
    MeterMap,
    TempoChange,
    TempoMap,
    TimeRange,
    TimeSignature,
    beats_to_ticks,
    format_timecode,
    note_value_to_ticks,
)

failures: list[str] = []


def check(label: str, got, expected, tol: float = 1e-9) -> None:
    ok = (abs(got - expected) <= tol) if isinstance(expected, float) else (got == expected)
    if not ok:
        failures.append(f"{label}: 기대 {expected!r} 실제 {got!r}")
    print(f"  {'OK ' if ok else 'FAIL'} {label}: {got!r} (기대 {expected!r})")


def check_raises(label: str, fn, exc=Exception) -> None:
    try:
        fn()
    except exc:
        print(f"  OK  {label}: 예외 발생함")
        return
    failures.append(f"{label}: 예외가 나와야 하는데 안 남")
    print(f"  FAIL {label}: 예외가 안 남")


print("[1] 음표 길이 -> tick  (4분음표 = PPQ = 960)")
check("4분음표", note_value_to_ticks(1, 4), 960)
check("8분음표", note_value_to_ticks(1, 8), 480)
check("온음표", note_value_to_ticks(1, 1), 3840)
check("점4분음표 3/8", note_value_to_ticks(3, 8), 1440)
check("64분음표", note_value_to_ticks(1, 64), 60)
check("2분음표 셋잇단 1/3", note_value_to_ticks(1, 3), 1280)

print("\n[2] 박자표")
check("4/4 한 마디", TimeSignature(4, 4).ticks_per_bar, 3840)
check("3/4 한 마디", TimeSignature(3, 4).ticks_per_bar, 2880)
check("6/8 한 마디", TimeSignature(6, 8).ticks_per_bar, 2880)
check("6/8 1박은 8분음표", TimeSignature(6, 8).ticks_per_beat, 480)
check("7/8 한 마디", TimeSignature(7, 8).ticks_per_bar, 3360)
check_raises("분모 5는 거부", lambda: TimeSignature(4, 5))
check_raises("분자 0은 거부", lambda: TimeSignature(0, 4))

print("\n[3] 고정 템포 120BPM -> 1박 = 0.5초, 1마디(4/4) = 2.0초")
tm = TempoMap([TempoChange(0, 120.0)])
check("tick 0", tm.tick_to_seconds(0), 0.0)
check("tick 960 (1박)", tm.tick_to_seconds(960), 0.5)
check("tick 3840 (1마디)", tm.tick_to_seconds(3840), 2.0)
check("tick 38400 (10마디)", tm.tick_to_seconds(38400), 20.0)
check("역변환 2.0초", tm.seconds_to_tick(2.0), 3840)
check("샘플 변환 1마디 @48k", tm.tick_to_sample(3840, 48000), 96000)

print("\n[4] 템포 변화: 1마디 뒤 120 -> 60 BPM")
tm2 = TempoMap([TempoChange(0, 120.0), TempoChange(3840, 60.0)])
check("변화 직전 tick 3840", tm2.tick_to_seconds(3840), 2.0)
# 60BPM 에서 4박 = 4초. 총 2.0 + 4.0 = 6.0
check("그 다음 1마디 끝", tm2.tick_to_seconds(7680), 6.0)
check("중간 BPM 조회", tm2.bpm_at_tick(5000), 60.0)
check("변화 전 BPM 조회", tm2.bpm_at_tick(1000), 120.0)
check("역변환 6.0초", tm2.seconds_to_tick(6.0), 7680)

print("\n[5] 선형 템포 커브: 4박에 걸쳐 120 -> 240 BPM")
# BPM(b) = 120 + 30b.  t = ∫0^4 60/(120+30b) db = (60/30)·ln(240/120) = 2·ln2
tm3 = TempoMap([TempoChange(0, 120.0, "linear"), TempoChange(3840, 240.0)])
expected = 2.0 * math.log(2.0)
check("4박 소요시간 = 2·ln2", tm3.tick_to_seconds(3840), expected, tol=1e-9)
check("중간 2박 지점 BPM", tm3.bpm_at_tick(1920), 180.0)
check("역변환 왕복", tm3.seconds_to_tick(expected), 3840)
# 수치적분으로 독립 검증
steps = 200000
acc = 0.0
for i in range(steps):
    b = (i + 0.5) * 4.0 / steps
    acc += 60.0 / (120.0 + 30.0 * b) * (4.0 / steps)
check("수치적분과 일치", tm3.tick_to_seconds(3840), acc, tol=1e-6)

print("\n[6] 마디/박 변환 (4/4)")
mm = MeterMap([MeterChange(0, TimeSignature(4, 4))])
check("tick 0 -> 1마디 1박", mm.tick_to_bar_beat(0), (1, 1.0))
check("tick 960 -> 1마디 2박", mm.tick_to_bar_beat(960), (1, 2.0))
check("tick 3840 -> 2마디 1박", mm.tick_to_bar_beat(3840), (2, 1.0))
check("tick 4800 -> 2마디 2박", mm.tick_to_bar_beat(4800), (2, 2.0))
check("9마디 첫 tick", mm.bar_to_tick(9), 30720)

print("\n[7] 박자표 변경: 4마디 4/4 뒤 3/4 로")
mm2 = MeterMap([MeterChange(0, TimeSignature(4, 4)), MeterChange(15360, TimeSignature(3, 4))])
check("4/4 4마디 = 15360tick, 그게 5마디 시작", mm2.tick_to_bar_beat(15360), (5, 1.0))
check("3/4 한 마디 뒤는 6마디", mm2.tick_to_bar_beat(15360 + 2880), (6, 1.0))
check("6마디 첫 tick 역산", mm2.bar_to_tick(6), 18240)
check("박자표 조회", str(mm2.signature_at_tick(16000)), "3/4")

print("\n[8] 구간(TimeRange)")
verse = TimeRange(0, 15360)
chorus = TimeRange(15360, 30720)
check("길이", verse.length_ticks, 15360)
check("겹치지 않음", verse.overlaps(chorus), False)
check("포함", verse.contains(1000), True)
check("경계는 미포함", verse.contains(15360), False)
check("교집합 없음", verse.intersection(chorus), None)
check("이동", verse.shifted(3840).start_tick, 3840)
check("겹침 판정", TimeRange(0, 100).overlaps(TimeRange(50, 200)), True)
check_raises("끝<=시작 거부", lambda: TimeRange(100, 100))
check("초 변환", verse.seconds(tm), (0.0, 8.0))

print("\n[9] 타임코드 표시")
check("0초", format_timecode(0.0), "0:00.0")
check("83.5초", format_timecode(83.5), "1:23.5")
check("125초", format_timecode(125.0), "2:05.0")

print("\n[10] 왕복 변환 무결성 (템포 3회 변경, 500개 지점)")
tm4 = TempoMap([
    TempoChange(0, 90.0),
    TempoChange(7680, 140.0, "linear"),
    TempoChange(23040, 72.0),
    TempoChange(38400, 128.0),
])
worst = 0
for t in range(0, 60000, 120):
    back = tm4.seconds_to_tick(tm4.tick_to_seconds(t))
    worst = max(worst, abs(back - t))
check("최대 왕복 오차 (tick)", worst <= 1, True)
print(f"       실제 최대 오차: {worst} tick")

print("\n" + "=" * 60)
if failures:
    print(f"실패 {len(failures)}건:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("전부 통과")
