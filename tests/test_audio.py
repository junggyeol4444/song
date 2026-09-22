"""오디오 엔진 검증.

기대값은 물리/신호처리 이론에서 나오는 값이거나 국제 표준(EBU Tech 3341)의
적합성 시험값이다. 코드가 뱉은 값을 그대로 적어두지 않는다.
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal as ss

sys.path.insert(0, ".")

from myvocal.audio import dsp, loudness as L
from myvocal.audio.buffer import AudioBuffer, AudioError, db_to_linear, linear_to_db

failures: list[str] = []
checks = 0
RATE = 48000


def check(label, got, expected, tol=1e-9):
    global checks
    checks += 1
    if isinstance(expected, float) and isinstance(got, (int, float)):
        if math.isinf(expected) or math.isinf(got) or math.isnan(expected) or math.isnan(got):
            ok = (got == expected) or (math.isnan(got) and math.isnan(expected))
        else:
            ok = abs(got - expected) <= tol
    else:
        ok = got == expected
    if not ok:
        failures.append(f"{label}: 기대 {expected!r} 실제 {got!r}")
        print(f"  FAIL {label}: {got!r} (기대 {expected!r}, 허용오차 {tol})")


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
    except AudioError:
        return
    failures.append(f"{label}: AudioError 가 나와야 하는데 안 남")
    print(f"  FAIL {label}")


def sine(freq, seconds=1.0, amp=1.0, rate=RATE):
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2.0 * np.pi * freq * t) * amp


def gain_at(b, a, freq, rate=RATE):
    _, h = ss.freqz(b, a, worN=[2.0 * np.pi * freq / rate])
    return 20.0 * math.log10(abs(h[0]))


print("[1] dB 변환")
check("0dB = 1.0배", db_to_linear(0.0), 1.0)
check("-6.0206dB = 0.5배", db_to_linear(-6.020599913279624), 0.5, tol=1e-12)
check("+6.0206dB = 2.0배", db_to_linear(6.020599913279624), 2.0, tol=1e-12)
check("-20dB = 0.1배", db_to_linear(-20.0), 0.1, tol=1e-12)
check("0.5배 = -6.0206dB", linear_to_db(0.5), -6.020599913279624, tol=1e-9)
check("0 은 -300dB", linear_to_db(0.0), -300.0)
check("왕복", db_to_linear(linear_to_db(0.333)), 0.333, tol=1e-9)

print("[2] 버퍼 기본")
b = AudioBuffer.from_mono(sine(440, 1.0, 0.5))
check("채널", b.channels, 1)
check("샘플 수", b.frames, RATE)
check("길이(초)", b.duration, 1.0)
check("peak", b.peak(), 0.5, tol=1e-6)
check("사인파 RMS = peak/√2", b.rms(), 0.5 / math.sqrt(2.0), tol=1e-4)
check("무음 판정", AudioBuffer.silence(100).is_silent, True)
check("1->2채널 복제", b.to_channels(2).channels, 2)
check("2->1채널 평균", b.to_channels(2).to_mono().channels, 1)
check("길이 늘리기", b.padded_to(RATE * 2).frames, RATE * 2)
check("길이 줄이기 시도는 무시", b.padded_to(100).frames, RATE)
check("구간 자르기", b.slice_seconds(0.25, 0.75).frames, RATE // 2)
check("범위 밖 자르기는 무음", AudioBuffer.from_mono(np.ones(10)).slice_frames(5, 20).frames, 15)
check_raises("샘플레이트 0 거부", lambda: AudioBuffer(np.zeros((1, 10)), 0))
check_raises("3차원 거부", lambda: AudioBuffer(np.zeros((2, 2, 2))))

print("[3] 합성 / 섞기")
a1 = AudioBuffer.from_mono(np.full(100, 0.3))
a2 = AudioBuffer.from_mono(np.full(100, 0.4))
check("더하면 합", a1.mix(a2).peak(), 0.7, tol=1e-6)
check("클리핑하지 않는다 (1.0 넘어도 그대로)",
      AudioBuffer.from_mono(np.full(10, 0.8)).mix(AudioBuffer.from_mono(np.full(10, 0.8))).peak(),
      1.6, tol=1e-6)
check("위치 지정 합성 길이", a1.mix(a2, at_frame=50).frames, 150)
check("겹치는 구간은 더해짐 (0.3+0.4)", float(a1.mix(a2, at_frame=50).data[0, 60]), 0.7, tol=1e-6)
check("안 겹치는 앞쪽은 원래대로", float(a1.mix(a2, at_frame=50).data[0, 10]), 0.3, tol=1e-6)
check("안 겹치는 뒤쪽은 새 소리만", float(a1.mix(a2, at_frame=50).data[0, 120]), 0.4, tol=1e-6)
check("이어붙이기", a1.concat(a2).frames, 200)
check("여러 개 합산", AudioBuffer.sum_buffers([a1, a1, a1]).peak(), 0.9, tol=1e-6)
mixed = AudioBuffer.silence(100, 1)
mixed.mix_in_place(a2, 0, gain=0.5)
check("제자리 합성 (이득 적용)", float(mixed.data[0, 0]), 0.2, tol=1e-6)
check_raises("샘플레이트 다르면 거부",
             lambda: AudioBuffer.silence(10, 1, 48000).mix(AudioBuffer.silence(10, 1, 44100)))

print("[4] 음량 / 팬")
check("+6dB 는 2배", b.with_gain_db(6.020599913279624).peak(), 1.0, tol=1e-5)
check("노멀라이즈 -1dB", b.normalized(-1.0).peak_db(), -1.0, tol=1e-4)
check("클리핑", AudioBuffer.from_mono(np.full(10, 2.0)).clipped(1.0).peak(), 1.0, tol=1e-6)
centered = AudioBuffer.from_mono(np.full(100, 1.0)).panned(0.0)
check("가운데 팬은 좌우 같음", float(centered.data[0, 0]), float(centered.data[1, 0]), tol=1e-6)
check("가운데 팬은 -3dB (등파워)", linear_to_db(float(centered.data[0, 0])),
      -3.0102999566398116, tol=1e-4)
left = AudioBuffer.from_mono(np.full(100, 1.0)).panned(-1.0)
check("완전 왼쪽: 왼쪽 1.0", float(left.data[0, 0]), 1.0, tol=1e-6)
check("완전 왼쪽: 오른쪽 0", abs(float(left.data[1, 0])) < 1e-6, True)
check_raises("팬 범위 초과 거부", lambda: b.panned(1.5))
faded = AudioBuffer.from_mono(np.ones(RATE)).faded(fade_in=0.1)
check("페이드인 시작은 0", abs(float(faded.data[0, 0])) < 1e-6, True)
check("페이드인 끝나면 1.0", float(faded.data[0, RATE // 2]), 1.0, tol=1e-6)

print("[5] 측정")
check("DC 치우침 감지", AudioBuffer.from_mono(np.full(100, 0.5)).dc_offset()[0], 0.5, tol=1e-6)
check("사인파는 DC 0", abs(b.dc_offset()[0]) < 1e-4, True)
check("클리핑 샘플 수", AudioBuffer.from_mono(np.full(10, 1.0)).clip_count(), 10)
check("정상 신호는 클리핑 0", b.clip_count(), 0)
# 인터샘플 피크: 나이퀴스트 근처 사인은 샘플 사이가 더 높다
near_nyquist = AudioBuffer.from_mono(sine(RATE * 0.45, 0.1, 1.0))
check_true("트루피크 >= 샘플피크",
           near_nyquist.true_peak() >= near_nyquist.peak() - 1e-6,
           f"(true {near_nyquist.true_peak():.4f} vs sample {near_nyquist.peak():.4f})")

print("[6] 샘플레이트 변환")
original = AudioBuffer.from_mono(sine(1000, 1.0, 0.5), 48000)
converted = original.resample(44100)
check("샘플레이트 바뀜", converted.sample_rate, 44100)
check("길이(초) 유지", converted.duration, 1.0, tol=1e-3)
check("진폭 유지", converted.peak(), 0.5, tol=0.01)
# 변환 후에도 1000Hz 가 맞는지 스펙트럼으로 확인
spectrum = np.abs(np.fft.rfft(converted.data[0] * np.hanning(converted.frames)))
peak_hz = np.fft.rfftfreq(converted.frames, 1 / 44100)[np.argmax(spectrum)]
check("주파수 유지 (1000Hz)", float(peak_hz), 1000.0, tol=2.0)
check("왕복 변환 길이", original.resample(44100).resample(48000).duration, 1.0, tol=1e-3)

print("[7] 파일 저장 / 읽기")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "test.wav"
    stereo = AudioBuffer(np.vstack([sine(440, 0.5, 0.4), sine(554.37, 0.5, 0.4)]), RATE)
    stereo.save(path)
    check("파일 생성됨", path.exists(), True)
    check_true("파일 크기 > 0", path.stat().st_size > 1000, f"({path.stat().st_size} bytes)")
    loaded = AudioBuffer.load(path)
    check("샘플레이트 보존", loaded.sample_rate, RATE)
    check("채널 수 보존", loaded.channels, 2)
    check("샘플 수 보존", loaded.frames, stereo.frames)
    check("파형 보존 (24bit 오차 안)",
          float(np.max(np.abs(loaded.data - stereo.data))), 0.0, tol=1e-6)
    check("읽으면서 변환", AudioBuffer.load(path, target_rate=44100).sample_rate, 44100)
    check_raises("없는 파일 거부", lambda: AudioBuffer.load(Path(tmp) / "없음.wav"))

print("[8] 라우드니스 — EBU Tech 3341 적합성 시험")
for target in (-23.0, -33.0):
    for rate in (44100, 48000, 96000):
        s = sine(1000, 10.0, 10 ** (target / 20.0), rate)
        got = L.integrated_loudness(AudioBuffer(np.vstack([s, s]), rate))
        check(f"{rate}Hz 1kHz {target}dBFS 스테레오", got, target, tol=0.1)
# 채널 합산 법칙: 한 채널만 울리면 3dB 작다
s = sine(1000, 10.0, 10 ** (-23.0 / 20.0))
check("한 채널만 -> 3dB 작음",
      L.integrated_loudness(AudioBuffer(np.vstack([s, np.zeros_like(s)]), RATE)), -26.0, tol=0.1)
# 게이팅: 뒤에 무음을 길게 붙여도 측정값이 거의 안 변해야 한다
with_silence = AudioBuffer(np.vstack([
    np.concatenate([s, np.zeros(RATE * 10)]),
    np.concatenate([s, np.zeros(RATE * 10)]),
]), RATE)
check("무음 구간은 게이팅으로 제외", L.integrated_loudness(with_silence), -23.0, tol=0.3)
check("무음은 -inf", L.integrated_loudness(AudioBuffer.silence(RATE, 2)), float("-inf"))
# LRA: 음량이 일정하면 0 에 가깝고, 변하면 커야 한다
quiet = sine(1000, 10.0, 10 ** (-33.0 / 20.0))
varied = AudioBuffer(np.vstack([np.concatenate([s, quiet])] * 2), RATE)
flat = AudioBuffer(np.vstack([np.concatenate([s, s])] * 2), RATE)
check_true("일정한 음량은 LRA 작음", L.loudness_range(flat) < 1.0, f"({L.loudness_range(flat):.2f} LU)")
check_true("변하는 음량은 LRA 큼", L.loudness_range(varied) > 5.0, f"({L.loudness_range(varied):.2f} LU)")

print("[9] 라우드니스 정규화")
loud = AudioBuffer(np.vstack([sine(440, 5.0, 0.7)] * 2), RATE)
normalized, applied = L.normalize_to_lufs(loud, -14.0, true_peak_ceiling_db=-1.0)
check("목표 LUFS 도달", L.integrated_loudness(normalized), -14.0, tol=0.2)
check_true("True Peak 한계 지킴", normalized.true_peak_db() <= -1.0 + 0.05,
           f"({normalized.true_peak_db():.2f} dBTP)")
# 목표가 너무 높아 True Peak 에 걸리는 경우
_, applied2 = L.normalize_to_lufs(loud, 0.0, true_peak_ceiling_db=-1.0)
check_true("천장에 걸리면 덜 올림", applied2 < (0.0 - L.integrated_loudness(loud)),
           f"(적용 {applied2:.2f}dB)")
report = L.measure(AudioBuffer(np.vstack([sine(1000, 5.0, 10 ** (-16 / 20))] * 2), RATE))
check("리포트 LUFS", report.integrated_lufs, -16.0, tol=0.2)
check("헤드룸 계산", report.headroom_to(-14.0), 2.0, tol=0.2)
check_true("스트리밍 진단 문자열", "spotify" in report.check_streaming("spotify").lower())

print("[10] 필터 응답")
for kind in ("lowpass", "highpass"):
    b_c, a_c = dsp.biquad_coefficients(kind, RATE, 1000.0, 0.7071)
    check(f"{kind} 차단주파수 -3dB", gain_at(b_c, a_c, 1000.0), -3.0103, tol=0.02)
b_c, a_c = dsp.biquad_coefficients("lowpass", RATE, 1000.0, 0.7071)
check_true("로우패스는 고역 차단", gain_at(b_c, a_c, 10000.0) < -35.0,
           f"({gain_at(b_c, a_c, 10000.0):.1f}dB)")
check("로우패스는 저역 통과", gain_at(b_c, a_c, 50.0), 0.0, tol=0.05)
b_c, a_c = dsp.biquad_coefficients("peaking", RATE, 1000.0, 1.0, 6.0)
check("피킹 +6dB 는 중심에서 정확히 +6", gain_at(b_c, a_c, 1000.0), 6.0, tol=1e-6)
b_c, a_c = dsp.biquad_coefficients("peaking", RATE, 1000.0, 1.0, -9.0)
check("피킹 -9dB", gain_at(b_c, a_c, 1000.0), -9.0, tol=1e-6)
b_c, a_c = dsp.biquad_coefficients("highshelf", RATE, 4000.0, 0.7071, 6.0)
check("하이셸프: 저역은 0dB", gain_at(b_c, a_c, 50.0), 0.0, tol=0.02)
check("하이셸프: 코너에서 절반(+3dB)", gain_at(b_c, a_c, 4000.0), 3.0, tol=0.05)
check("하이셸프: 고역은 +6dB", gain_at(b_c, a_c, 20000.0), 6.0, tol=0.05)
b_c, a_c = dsp.biquad_coefficients("lowshelf", RATE, 200.0, 0.7071, -6.0)
check("로우셸프 -6dB: 저역", gain_at(b_c, a_c, 20.0), -6.0, tol=0.1)
check("로우셸프 -6dB: 고역은 0", gain_at(b_c, a_c, 10000.0), 0.0, tol=0.05)
b_c, a_c = dsp.biquad_coefficients("notch", RATE, 1000.0, 10.0)
check_true("노치는 중심을 깊게 판다", gain_at(b_c, a_c, 1000.0) < -40.0,
           f"({gain_at(b_c, a_c, 1000.0):.1f}dB)")
check_raises("나이퀴스트 초과 거부", lambda: dsp.biquad_coefficients("lowpass", RATE, 30000.0))
check_raises("Q 0 거부", lambda: dsp.biquad_coefficients("lowpass", RATE, 1000.0, 0.0))
check_raises("모르는 필터 거부", lambda: dsp.biquad_coefficients("bogus", RATE, 1000.0))

print("[11] EQ 체인")
eq = dsp.Equalizer([
    dsp.FilterBand("highpass", 80.0),
    dsp.FilterBand("peaking", 3000.0, 1.5, 4.0),
    dsp.FilterBand("highshelf", 8000.0, 0.7071, 2.0),
])
response = eq.response_db([100.0, 3000.0, 16000.0], RATE)
check("3kHz 부스트 반영", float(response[1]), 4.0, tol=0.3)
check_true("80Hz 하이패스 반영", response[0] < 0.0, f"({response[0]:.2f}dB)")
check_true("16kHz 셸프 반영", response[2] > 1.5, f"({response[2]:.2f}dB)")
processed = eq.process(AudioBuffer.from_mono(sine(3000, 0.5, 0.2)))
check("실제 처리도 +4dB", linear_to_db(processed.peak()) - linear_to_db(0.2), 4.0, tol=0.4)
check("빈 EQ 는 그대로", dsp.Equalizer().process(b).peak(), b.peak(), tol=1e-9)

print("[12] 컴프레서 정적 곡선")
settings = dsp.CompressorSettings(threshold_db=-20.0, ratio=4.0, attack_ms=0.0,
                                  release_ms=0.0, knee_db=0.0, makeup_db=0.0)
for input_db in (-40.0, -30.0, -20.0, -10.0, 0.0, 6.0):
    buffer = AudioBuffer.from_mono(np.full(4800, 10 ** (input_db / 20.0)))
    out, _ = dsp.compress(buffer, settings)
    got = linear_to_db(float(np.max(np.abs(out.data[:, 2400:]))))
    expected = input_db if input_db <= -20.0 else -20.0 + (input_db + 20.0) / 4.0
    check(f"입력 {input_db:+.0f}dB", got, expected, tol=0.02)
# 비율 1:1 이면 아무것도 안 해야 한다
flat_settings = dsp.CompressorSettings(threshold_db=-40.0, ratio=1.0, knee_db=0.0, makeup_db=0.0)
out, _ = dsp.compress(AudioBuffer.from_mono(np.full(4800, 0.5)), flat_settings)
check("1:1 은 변화 없음", float(np.max(np.abs(out.data))), 0.5, tol=1e-6)
# 소프트 니는 threshold 아래에서도 조금 눌린다
knee_settings = dsp.CompressorSettings(threshold_db=-20.0, ratio=4.0, attack_ms=0.0,
                                       release_ms=0.0, knee_db=12.0, makeup_db=0.0)
out, reduction = dsp.compress(AudioBuffer.from_mono(np.full(4800, 10 ** (-22.0 / 20.0))),
                              knee_settings)
check_true("소프트 니는 threshold 아래도 살짝 누름", float(np.min(reduction)) < -0.01,
           f"({float(np.min(reduction)):.3f}dB)")
check_raises("비율 1 미만 거부", lambda: dsp.CompressorSettings(ratio=0.5))

print("[13] 리미터")
for ceiling in (-0.1, -1.0, -3.0):
    loud = AudioBuffer.from_mono(sine(200, 1.0, 1.8))
    out, reduction = dsp.limit(loud, ceiling_db=ceiling, lookahead_ms=5.0)
    check_true(f"천장 {ceiling}dB 를 안 넘음", out.peak_db() <= ceiling + 0.02,
               f"(실제 {out.peak_db():.3f}dB)")
    check_true(f"천장 {ceiling}dB 에서 실제로 누름", reduction > 0.0, f"({reduction:.2f}dB)")
quiet = AudioBuffer.from_mono(sine(200, 1.0, 0.1))
out, reduction = dsp.limit(quiet, ceiling_db=-1.0)
check("작은 소리는 안 누름", reduction, 0.0, tol=1e-6)

print("[14] 게이트 / 새츄레이션")
noisy = np.concatenate([sine(440, 0.5, 0.5), sine(440, 0.5, 0.0005)])
gated = dsp.gate(AudioBuffer.from_mono(noisy), threshold_db=-40.0, release_ms=20.0, hold_ms=5.0)
check_true("큰 부분은 남김", float(np.max(np.abs(gated.data[0, :RATE // 2]))) > 0.4)
check_true("조용한 부분은 닫음",
           float(np.max(np.abs(gated.data[0, int(RATE * 0.8):]))) < 0.0001,
           f"({float(np.max(np.abs(gated.data[0, int(RATE * 0.8):]))):.6f})")
saturated = dsp.saturate(AudioBuffer.from_mono(sine(440, 0.2, 0.9)), drive_db=12.0, mode="tanh")
spectrum = np.abs(np.fft.rfft(saturated.data[0] * np.hanning(saturated.frames)))
freqs = np.fft.rfftfreq(saturated.frames, 1 / RATE)
third = float(spectrum[np.argmin(np.abs(freqs - 1320.0))])
fundamental = float(spectrum[np.argmin(np.abs(freqs - 440.0))])
check_true("새츄레이션이 배음을 만든다", third / fundamental > 0.01,
           f"(3배음/기본음 = {third / fundamental:.4f})")
check_raises("모르는 새츄레이션 거부", lambda: dsp.saturate(b, mode="bogus"))

print("[15] 공간계")
impulse = np.zeros(RATE * 3)
impulse[0] = 1.0
impulse_buffer = AudioBuffer(np.vstack([impulse, impulse]), RATE)


def tail_length(buffer):
    envelope = np.max(np.abs(buffer.data), axis=0)
    above = np.where(envelope > envelope.max() * 1e-3)[0]
    return float(above[-1] / RATE) if above.size else 0.0


big = dsp.reverb(impulse_buffer, dsp.ReverbSettings(room_size=0.85, wet_db=0.0, dry_db=-90.0))
small = dsp.reverb(impulse_buffer, dsp.ReverbSettings(room_size=0.15, wet_db=0.0, dry_db=-90.0))
check_true("큰 방이 꼬리가 길다", tail_length(big) > tail_length(small) * 2.0,
           f"(큰 {tail_length(big):.2f}초 vs 작은 {tail_length(small):.2f}초)")
check_true("잔향이 실제로 소리를 만든다", big.peak() > 0.0)
bright = dsp.reverb(impulse_buffer, dsp.ReverbSettings(room_size=0.7, damping=0.0,
                                                      wet_db=0.0, dry_db=-90.0))
dark = dsp.reverb(impulse_buffer, dsp.ReverbSettings(room_size=0.7, damping=1.0,
                                                    wet_db=0.0, dry_db=-90.0))


def high_energy(buffer):
    spectrum = np.abs(np.fft.rfft(buffer.data[0]))
    freqs = np.fft.rfftfreq(buffer.frames, 1 / RATE)
    return float(np.sum(spectrum[freqs > 5000.0]))


check_true("감쇠를 올리면 고역이 줄어든다", high_energy(dark) < high_energy(bright),
           f"(어두움 {high_energy(dark):.0f} vs 밝음 {high_energy(bright):.0f})")

echo = dsp.delay_effect(impulse_buffer, 500.0, feedback=0.5, wet_db=0.0, dry_db=0.0,
                        damping_hz=None)
envelope = np.max(np.abs(echo.data), axis=0)
peaks = [i for i in range(1, len(envelope) - 1)
         if envelope[i] > 0.01 and envelope[i] >= envelope[i - 1] and envelope[i] >= envelope[i + 1]]
check("첫 반향 0.5초", round(peaks[0] / RATE, 3), 0.5)
check("두번째 1.0초", round(peaks[1] / RATE, 3), 1.0)
check("세번째 1.5초", round(peaks[2] / RATE, 3), 1.5)
check("되먹임 0.5 -> 절반", float(envelope[peaks[1]]) / float(envelope[peaks[0]]), 0.5, tol=1e-4)
check("그 다음도 절반", float(envelope[peaks[2]]) / float(envelope[peaks[1]]), 0.5, tol=1e-4)

pp = dsp.delay_effect(impulse_buffer, 300.0, feedback=0.4, wet_db=0.0, dry_db=-90.0,
                      ping_pong=True, damping_hz=None)


def peak_times(channel):
    envelope = np.abs(channel)
    return [round(i / RATE, 3) for i in range(1, len(envelope) - 1)
            if envelope[i] > 0.005 and envelope[i] >= envelope[i - 1] and envelope[i] >= envelope[i + 1]]


check("핑퐁 왼쪽 = 홀수배", peak_times(pp.data[0])[:3], [0.3, 0.9, 1.5])
check("핑퐁 오른쪽 = 짝수배", peak_times(pp.data[1])[:3], [0.6, 1.2, 1.8])

noise_l = np.random.RandomState(0).randn(RATE) * 0.1
noise_r = np.random.RandomState(1).randn(RATE) * 0.1
stereo_noise = AudioBuffer(np.vstack([noise_l, noise_r]), RATE)
narrow = dsp.stereo_width(stereo_noise, 0.0)
check("폭 0 은 완전 모노", float(np.corrcoef(narrow.data[0], narrow.data[1])[0, 1]), 1.0, tol=1e-6)
wide = dsp.stereo_width(stereo_noise, 2.0)
check_true("폭 2.0 은 더 벌어짐",
           float(np.corrcoef(wide.data[0], wide.data[1])[0, 1])
           < float(np.corrcoef(stereo_noise.data[0], stereo_noise.data[1])[0, 1]))
check("폭 1.0 은 그대로",
      float(np.max(np.abs(dsp.stereo_width(stereo_noise, 1.0).data - stereo_noise.data))),
      0.0, tol=1e-6)
mono_low = dsp.mono_below(stereo_noise, 120.0)


def band_correlation(buffer, low_hz, high_hz):
    """지정 대역만 남기고 좌우 상관을 잰다. 1.0 이면 그 대역은 모노다."""
    sos = ss.butter(8, [low_hz, high_hz], btype="bandpass", fs=RATE, output="sos")
    filtered = ss.sosfilt(sos, buffer.data, axis=1)
    return float(np.corrcoef(filtered[0], filtered[1])[0, 1])


check_true("크로스오버 아래(20~60Hz)는 모노",
           band_correlation(mono_low, 20.0, 60.0) > 0.98,
           f"(상관 {band_correlation(mono_low, 20.0, 60.0):.4f})")
check_true("크로스오버 위(1k~5kHz)는 스테레오 유지",
           abs(band_correlation(mono_low, 1000.0, 5000.0)) < 0.2,
           f"(상관 {band_correlation(mono_low, 1000.0, 5000.0):.4f})")
check_true("원본은 저역도 스테레오였음",
           abs(band_correlation(stereo_noise, 20.0, 60.0)) < 0.5,
           f"(상관 {band_correlation(stereo_noise, 20.0, 60.0):.4f})")
check_raises("기준 주파수 범위 초과 거부", lambda: dsp.mono_below(stereo_noise, 30000.0))
check_raises("하스 40ms 초과 거부", lambda: dsp.haas_widen(stereo_noise, 50.0))

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
