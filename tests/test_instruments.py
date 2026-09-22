"""악기 검증.

'소리가 났다'는 검증이 아니다. 다음을 확인한다.

  1. 지정한 음높이를 실제로 내는가          (음정 검출기로 측정)
  2. 세게 치면 음색이 바뀌는가              (음량만 바뀌면 실패)
  3. 악기의 물리적 특징이 실제로 있는가      (피아노 인하모니시티, 벨 비정수 배음 등)
  4. 음역 밖 요청을 조용히 넘기지 않는가
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from myvocal.audio.analysis import (
    find_partials, harmonic_energy_ratio, measure_note_frequency, spectral_centroid,
)
from myvocal.audio.buffer import AudioBuffer
from myvocal.music import instruments as I
from myvocal.music.theory import Pitch, midi_to_frequency

RATE = 48000
failures: list[str] = []
checks = 0


def check(label, got, expected, tol=1e-9):
    global checks
    checks += 1
    if isinstance(expected, float) and isinstance(got, (int, float)):
        ok = abs(got - expected) <= tol
    else:
        ok = got == expected
    if not ok:
        failures.append(f"{label}: 기대 {expected!r} 실제 {got!r}")
        print(f"  FAIL {label}: {got!r} (기대 {expected!r})")


def check_true(label, condition, detail=""):
    global checks
    checks += 1
    if not condition:
        failures.append(f"{label} {detail}")
        print(f"  FAIL {label} {detail}")


def check_raises(label, fn):
    global checks
    checks += 1
    try:
        fn()
    except I.InstrumentError:
        return
    failures.append(f"{label}: InstrumentError 가 나와야 하는데 안 남")
    print(f"  FAIL {label}")


def check_pitch(instrument, midi, tolerance_cents=10.0, label=""):
    """악기가 지정한 음높이를 내는지 확인한다."""
    global checks
    checks += 1
    expected_hz = midi_to_frequency(midi)
    signal = instrument.render_note(midi, 100, 1.2, RATE)
    measured = measure_note_frequency(
        signal, RATE, fmin=max(20.0, expected_hz * 0.55), fmax=min(6000.0, expected_hz * 2.5),
        skip_seconds=0.12, length_seconds=0.6,
    )
    if measured <= 0:
        failures.append(f"{label} {Pitch.from_midi(midi)}: 음높이를 측정할 수 없음")
        print(f"  FAIL {label} {Pitch.from_midi(midi)}: 측정 불가")
        return
    cents = 1200.0 * np.log2(measured / expected_hz)
    if abs(cents) > tolerance_cents:
        failures.append(
            f"{label} {Pitch.from_midi(midi)}: {cents:+.1f}센트 벗어남 "
            f"(기대 {expected_hz:.2f}Hz, 실측 {measured:.2f}Hz)"
        )
        print(f"  FAIL {label} {Pitch.from_midi(midi)}: {cents:+.1f}센트")


print("[1] 음높이 정확도 — 모든 유선율 악기, 음역 전체")
TONAL = [
    (I.AcousticPiano(), "피아노", [21, 33, 45, 57, 69, 81, 93, 105]),
    (I.ElectricPiano(), "일렉피아노", [28, 40, 52, 64, 76, 88, 100]),
    (I.Organ(), "오르간", [24, 36, 48, 60, 72, 84, 96]),
    (I.AcousticGuitar(), "통기타", [40, 48, 56, 64, 72, 80, 88]),
    (I.ElectricGuitar(0.6), "일렉기타", [40, 48, 56, 64, 72, 80, 88]),
    (I.ElectricBass(), "일렉베이스", [28, 34, 40, 46, 52, 58, 64]),
    (I.SynthBass(), "신스베이스", [24, 32, 40, 48, 56, 64]),
    (I.SubBass(), "서브베이스", [21, 28, 35, 42, 49, 55]),
    (I.Strings(), "스트링", [36, 48, 60, 72, 84, 96]),
    (I.Brass(), "금관", [40, 50, 60, 70, 80]),
    (I.Flute(), "플루트", [59, 68, 77, 86, 95]),
    (I.SynthPad(), "패드", [24, 38, 52, 66, 80, 94]),
    (I.SynthLead(), "리드", [48, 60, 72, 84, 96, 106]),
    (I.Choir(), "합창", [40, 50, 60, 70, 80]),
]
start = time.time()
for instrument, name, midis in TONAL:
    for midi in midis:
        check_pitch(instrument, midi, 10.0, name)
print(f"  {sum(len(m) for _, _, m in TONAL)}개 음 검사, {time.time() - start:.1f}초")

print("[2] 세기가 음색을 바꾸는가 (음량만 바뀌면 실패)")
for instrument, name, midi in (
    (I.AcousticPiano(), "피아노", 60), (I.ElectricPiano(), "일렉피아노", 60),
    (I.ElectricGuitar(0.5), "일렉기타", 52), (I.Brass(), "금관", 60),
    (I.SynthBass(), "신스베이스", 40), (I.SynthLead(), "리드", 72),
):
    fundamental = midi_to_frequency(midi)
    ratios = []
    for velocity in (40, 80, 120):
        signal = instrument.render_note(midi, velocity, 0.8, RATE)
        segment = signal[int(0.05 * RATE) : int(0.55 * RATE)]
        ratios.append(harmonic_energy_ratio(segment, RATE, fundamental))
    check_true(
        f"{name}: 세게 칠수록 배음이 많아짐",
        ratios[0] < ratios[1] < ratios[2],
        f"(약 {ratios[0]:.4f} / 중 {ratios[1]:.4f} / 강 {ratios[2]:.4f})",
    )

print("[3] 세기가 음량도 바꾸는가")
for instrument, name, midi in ((I.AcousticPiano(), "피아노", 60), (I.Organ(), "오르간", 60)):
    peaks = [
        float(np.max(np.abs(instrument.render_note(midi, v, 0.6, RATE))))
        for v in (30, 70, 110)
    ]
    check_true(f"{name}: 세게 칠수록 커짐", peaks[0] < peaks[1] < peaks[2],
               f"({peaks[0]:.3f} / {peaks[1]:.3f} / {peaks[2]:.3f})")

print("[4] 피아노 인하모니시티 — 실제 피아노 줄의 뻣뻣함")
piano = I.AcousticPiano()
signal = piano.render_note(36, 100, 3.0, RATE)
fundamental = midi_to_frequency(36)
partials = find_partials(signal[: RATE * 2], RATE, fundamental, count=16)
deviations = [
    1200.0 * np.log2(hz / (fundamental * (index + 1)))
    for index, (hz, _) in enumerate(partials)
]
check_true("1배음은 제자리", abs(deviations[0]) < 10.0, f"({deviations[0]:+.1f}센트)")
check_true("높은 배음일수록 위로 밀린다", deviations[-1] > 15.0,
           f"({len(deviations)}배음 {deviations[-1]:+.1f}센트)")
check_true("단조 증가", all(deviations[i] <= deviations[i + 1] + 3.0
                         for i in range(len(deviations) - 1)),
           f"({[round(d, 1) for d in deviations]})")

print("[5] 벨은 배음이 정수배가 아니다")
bell = I.Bell()
signal = bell.render_note(72, 100, 1.0, RATE)
fundamental = midi_to_frequency(72)
spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.shape[0])))
freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / RATE)
for ratio in (0.5, 1.0, 1.2, 1.5, 2.0, 2.5):
    target = fundamental * ratio
    mask = (freqs > target * 0.98) & (freqs < target * 1.02)
    found = float(freqs[mask][np.argmax(spectrum[mask])]) if mask.any() else 0.0
    check(f"벨 {ratio}배 부분음", 1200.0 * np.log2(found / target) if found else 999.0,
          0.0, tol=5.0)
# 1.2배, 1.5배 같은 비정수 부분음이 실제로 세게 있어야 한다
mask = (freqs > fundamental * 1.18) & (freqs < fundamental * 1.22)
check_true("1.2배 부분음이 실제로 강하다",
           float(spectrum[mask].max()) > float(spectrum.max()) * 0.05)

print("[6] 오르간은 줄어들지 않는다 (피아노와 정반대)")
organ = I.Organ()
signal = organ.render_note(60, 100, 2.0, RATE)
early = float(np.sqrt(np.mean(signal[int(0.3 * RATE) : int(0.5 * RATE)] ** 2)))
late = float(np.sqrt(np.mean(signal[int(1.5 * RATE) : int(1.7 * RATE)] ** 2)))
check_true("오르간: 2초 동안 음량 유지", abs(20 * np.log10(late / early)) < 1.0,
           f"({20 * np.log10(late / early):+.2f}dB 변화)")
signal = piano.render_note(60, 100, 2.0, RATE)
early = float(np.sqrt(np.mean(signal[int(0.05 * RATE) : int(0.25 * RATE)] ** 2)))
late = float(np.sqrt(np.mean(signal[int(1.5 * RATE) : int(1.7 * RATE)] ** 2)))
# C4 의 반감기는 1.2초로 잡았다. 1.5초 뒤면 이론상 -7.5dB 안팎이고,
# 고음 배음이 먼저 죽으므로 실제로는 그보다 조금 더 줄어든다.
piano_decay_db = 20 * np.log10(late / early)
check_true("피아노: 같은 시간에 확실히 줄어든다", piano_decay_db < -6.0,
           f"({piano_decay_db:+.1f}dB)")
check_true("피아노: 그래도 완전히 사라지지는 않는다", piano_decay_db > -25.0,
           f"({piano_decay_db:+.1f}dB)")

print("[7] 음을 떼면 소리가 멈추는가")
for instrument, name in ((I.AcousticPiano(), "피아노"), (I.Organ(), "오르간"),
                         (I.Strings(), "스트링"), (I.Flute(), "플루트")):
    signal = instrument.render_note(60, 100, 0.5, RATE)
    during = float(np.max(np.abs(signal[int(0.2 * RATE) : int(0.45 * RATE)])))
    after = float(np.max(np.abs(signal[int(1.2 * RATE) :]))) if signal.shape[0] > RATE * 1.2 else 0.0
    check_true(f"{name}: 뗀 뒤 크게 줄어든다", after < during * 0.35,
               f"(연주 중 {during:.3f} -> 뗀 뒤 {after:.3f})")

print("[8] 음역 밖은 거부한다")
check_raises("피아노 음역 아래", lambda: I.AcousticPiano().render_note(20, 100, 1.0, RATE))
check_raises("피아노 음역 위", lambda: I.AcousticPiano().render_note(109, 100, 1.0, RATE))
check_raises("베이스 음역 위", lambda: I.ElectricBass().render_note(80, 100, 1.0, RATE))
check_raises("플루트 음역 아래", lambda: I.Flute().render_note(40, 100, 1.0, RATE))
check_raises("세기 범위 초과", lambda: I.AcousticPiano().render_note(60, 200, 1.0, RATE))
check_raises("음수 길이", lambda: I.AcousticPiano().render_note(60, 100, -1.0, RATE))

print("[9] Karplus-Strong 튜닝")
from myvocal.music.instruments import karplus_strong
worst = 0.0
for midi in range(40, 90, 2):
    expected = midi_to_frequency(midi)
    signal = karplus_strong(expected, RATE * 2, RATE, damping=0.45, brightness=0.7, seed=1)
    measured = measure_note_frequency(signal, RATE, fmin=expected * 0.6, fmax=expected * 2.0,
                                      skip_seconds=0.2, length_seconds=0.8)
    cents = abs(1200.0 * np.log2(measured / expected)) if measured > 0 else 999.0
    worst = max(worst, cents)
check_true("E2~F6 전체에서 5센트 이내", worst < 5.0, f"(최대 {worst:.2f}센트)")
check_raises("너무 높은 음 거부", lambda: karplus_strong(30000.0, 1000, RATE))
check_raises("감쇠 범위 초과", lambda: karplus_strong(440.0, 1000, RATE, damping=1.5))
check("DC 성분 없음", float(abs(karplus_strong(196.0, RATE, RATE, seed=1).mean())), 0.0, tol=1e-4)

print("[10] 드럼")
kit = I.DrumKit()
for name in sorted(I.DRUM_DISPLAY_NAMES):
    signal = kit.render(name, 100, RATE)
    check_true(f"{I.DRUM_DISPLAY_NAMES[name]}: 소리가 난다",
               float(np.max(np.abs(signal))) > 0.1)
    check_true(f"{I.DRUM_DISPLAY_NAMES[name]}: 길이가 적당하다",
               0.05 <= signal.shape[0] / RATE <= 4.5,
               f"({signal.shape[0] / RATE:.2f}초)")
check_raises("모르는 드럼 거부", lambda: kit.render("bogus"))
check_raises("배정 안 된 노트 거부", lambda: kit.render_note(1, 100, 0.1, RATE))
check("MIDI 36 은 킥", I.DRUM_NOTES[36], "kick")
check("MIDI 38 은 스네어", I.DRUM_NOTES[38], "snare")
check("MIDI 42 은 클로즈 하이햇", I.DRUM_NOTES[42], "hihat_closed")
# 각 드럼이 있어야 할 주파수 대역에 있는가
centroids = {
    name: spectral_centroid(AudioBuffer.from_mono(kit.render(name, 100, RATE), RATE))
    for name in ("kick", "snare", "hihat_closed", "tom_low", "crash")
}
check_true("킥은 저역", centroids["kick"] < 200.0, f"({centroids['kick']:.0f}Hz)")
check_true("로우 탐은 저역", centroids["tom_low"] < 400.0, f"({centroids['tom_low']:.0f}Hz)")
check_true("스네어는 중고역", 2000.0 < centroids["snare"] < 12000.0,
           f"({centroids['snare']:.0f}Hz)")
check_true("하이햇은 고역", centroids["hihat_closed"] > 7000.0,
           f"({centroids['hihat_closed']:.0f}Hz)")
check_true("크래시는 고역이고 길다", centroids["crash"] > 6000.0)
check_true("킥 < 탐 < 스네어 < 하이햇 순으로 밝다",
           centroids["kick"] < centroids["tom_low"] < centroids["snare"] < centroids["hihat_closed"])
# 오픈 하이햇은 클로즈보다 길다
check_true("오픈 하이햇이 클로즈보다 길다",
           kit.render("hihat_open", 100, RATE).shape[0]
           > kit.render("hihat_closed", 100, RATE).shape[0] * 3)

print("[11] 악기 등록부")
check("등록된 악기 수", len(I.available_instruments()), 16)
check("이름으로 생성", type(I.create_instrument("acoustic_piano")).__name__, "AcousticPiano")
check("인자 전달", I.create_instrument("electric_guitar", drive=0.9).drive, 0.9)
check_raises("모르는 악기 거부", lambda: I.create_instrument("theremin"))
check_raises("드라이브 범위 초과", lambda: I.ElectricGuitar(1.5))
grouped = I.instruments_by_category()
check_true("건반 분류에 3종", len(grouped["keys"]) == 3, f"({grouped['keys']})")
check_true("모든 악기가 어느 분류엔가 속한다",
           sum(len(v) for v in grouped.values()) == len(I.available_instruments()))
for name in I.available_instruments():
    instrument = I.create_instrument(name)
    check_true(f"{name}: 음역이 유효", instrument.info.lowest_midi < instrument.info.highest_midi)
    check_true(f"{name}: 표시 이름 있음", bool(instrument.info.display_name))

print("[12] 세기 -> 진폭 변환")
check("세기 0", I.velocity_to_amplitude(0), 0.0)
check("세기 127", I.velocity_to_amplitude(127), 1.0)
check("세기 64 는 절반이 아니다 (제곱 곡선)",
      abs(I.velocity_to_amplitude(64) - 0.5) > 0.2, True)
check("세기 64", I.velocity_to_amplitude(64), (64 / 127) ** 2, tol=1e-9)
check_raises("세기 범위 초과", lambda: I.velocity_to_amplitude(128))
check_raises("음수 세기", lambda: I.velocity_to_amplitude(-1))

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
