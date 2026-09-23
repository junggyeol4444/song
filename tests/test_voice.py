"""노래 합성 검증.

'소리가 났다' 는 검증이 아니다. 다음을 확인한다.

  1. 한국어 발음 규칙이 표준 발음법대로 적용되는가
  2. 자음이 박자 앞에 놓이는가 (사람이 노래하는 방식)
  3. 모음마다 포먼트가 실제로 다른가 ('아'와 '이'가 구분되는가)
  4. 성도 배율이 포먼트를 옮기는가 (남녀 목소리 차이)
  5. 지정한 음높이를 내는가
  6. 숨소리 설정이 실제 잡음량을 바꾸는가
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from myvocal.audio.analysis import measure_note_frequency
from myvocal.music.theory import Pitch
from myvocal.voice.korean import (
    FINAL_SOUND, KoreanError, compose, count_syllables, decompose,
    is_hangul_syllable, pronounce, split_syllables,
)
from myvocal.voice.phonemes import (
    PhonemeKind, VOWEL_FORMANTS, align_lyrics, syllable_to_phonemes, viseme_timeline,
)
from myvocal.voice.synth import (
    SingingSynth, VoiceError, VoiceTimbre, _resonator, glottal_pulse, sing,
)

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


def check_raises(label, fn, exc=Exception):
    global checks
    checks += 1
    try:
        fn()
    except exc:
        return
    failures.append(f"{label}: 예외가 나와야 하는데 안 남")
    print(f"  FAIL {label}")


def envelope_peaks(signal, rate=RATE, smooth_hz=320.0, fmax=4000.0):
    """포먼트 포락선의 봉우리. 배음 간격보다 넓게 평활해서 찾는다.

    LPC 는 음높이가 높으면 포먼트 대신 배음을 잡는 일이 잦다. 평활 스펙트럼은
    그 문제가 없다.
    """
    length = len(signal)
    spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(length)))
    freqs = np.fft.rfftfreq(length, 1.0 / rate)
    bin_hz = freqs[1] - freqs[0]
    width = max(3, int(smooth_hz / bin_hz) | 1)
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    envelope = 20.0 * np.log10(np.convolve(spectrum, kernel, mode="same") + 1e-12)
    peaks = []
    for index in range(2, len(envelope) - 2):
        if freqs[index] > fmax:
            break
        if freqs[index] < 180.0:
            continue
        if (envelope[index] > envelope[index - 1]
                and envelope[index] >= envelope[index + 1]
                and envelope[index] > envelope.max() - 30.0):
            peaks.append(float(freqs[index]))
    return peaks


print("[1] 글자 분해와 조립")
check("강", str(decompose("강")), "ㄱ+ㅏ+ㅇ")
check("닭", str(decompose("닭")), "ㄷ+ㅏ+ㄺ")
check("아", str(decompose("아")), "ㅇ+ㅏ")
check("다시 합침", decompose("값").compose(), "값")
check("직접 조립", compose("ㄱ", "ㅏ", "ㅇ"), "강")
check("받침 없이", compose("ㄴ", "ㅏ"), "나")
check_true("한글 판정", is_hangul_syllable("가") and not is_hangul_syllable("a"))
check_raises("한글 아닌 것 거부", lambda: decompose("a"), KoreanError)
check_raises("두 글자 거부", lambda: decompose("가나"), KoreanError)
for character in "가나다라마바사아자차카타파하강닭값없":
    check(f"'{character}' 왕복", decompose(character).compose(), character)

print("[2] 표준 발음법 — 소리 바뀜")
# 국립국어원 표준 발음법의 예시들
CASES = [
    ("같이", "가치", "구개음화 (제17항)"),
    ("굳이", "구지", "구개음화"),
    ("독립", "동닙", "비음화 (제19항)"),
    ("국물", "궁물", "비음화 (제18항)"),
    ("밥물", "밤물", "비음화"),
    ("좋아", "조아", "ㅎ 탈락 (제12항)"),
    ("놓은", "노은", "ㅎ 탈락"),
    ("좋다", "조타", "거센소리 (제12항)"),
    ("축하", "추카", "거센소리"),
    ("신라", "실라", "유음화 (제20항)"),
    ("칼날", "칼랄", "유음화"),
    ("종로", "종노", "비음화"),
    ("국립", "궁닙", "비음화"),
    ("닭이", "달기", "연음 (제14항)"),
    ("꽃이", "꼬치", "연음"),
    ("앉아", "안자", "겹받침 연음"),
    ("값이", "갑씨", "겹받침 연음 + 된소리 (제14항)"),
    ("없어", "업써", "겹받침 연음 + 된소리"),
    ("먹다", "먹따", "된소리 (제23항)"),
    ("옷장", "옫짱", "끝소리 + 된소리"),
    ("낮", "낟", "끝소리 (제9항)"),
    ("꽃", "꼳", "끝소리"),
    ("앞", "압", "끝소리"),
    ("밖", "박", "끝소리"),
]
for source, expected, rule in CASES:
    check(f"{source} ({rule})", pronounce(source), expected)
check("바뀔 것이 없으면 그대로", pronounce("나비"), "나비")
check("한글 아닌 것은 그대로", pronounce("hello"), "hello")
check("섞여 있어도", pronounce("a같이b"), "a가치b")
# 받침은 7가지 소리만 난다
sounds = {FINAL_SOUND[f] for f in FINAL_SOUND if f}
check("받침 소리는 7가지", sorted(sounds), ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"])

print("[3] 음절 나누기")
check("한글", split_syllables("다시 만나는 날"), ["다", "시", "만", "나", "는", "날"])
check("음절 수", count_syllables("안녕하세요"), 5)
check("띄어쓰기는 안 센다", count_syllables("다시 만나는 날"), 6)
check("문장부호도 안 센다", count_syllables("안녕, 반가워!"), 5)  # 안녕반가워
check("영어는 덩어리로", split_syllables("Hello 세계"), ["Hello", "세", "계"])
check("빈 문자열", split_syllables(""), [])

print("[4] 음소와 시간 — 자음은 박자 앞에")
phonemes = syllable_to_phonemes(decompose("강"), 0.5)
check("강 = 3음소", len(phonemes), 3)
check("첫소리는 ㄱ", phonemes[0].symbol, "ㄱ")
check_true("자음이 박자보다 앞선다", phonemes[0].start < 0,
           f"({phonemes[0].start * 1000:.0f}ms)")
check("모음은 박자에 시작", phonemes[1].start, 0.0)
check("모음이 ㅏ", phonemes[1].symbol, "ㅏ")
check("받침은 끝에", phonemes[2].symbol, "ㅇ")
check_true("받침이 음 안에서 끝난다", phonemes[2].end <= 0.5 + 1e-9)
# 받침 없는 음절
phonemes = syllable_to_phonemes(decompose("나"), 0.5)
check("나 = 2음소", len(phonemes), 2)
check_true("모음이 끝까지", abs(phonemes[1].end - 0.5) < 1e-6)
# 겹모음은 반모음 + 모음
phonemes = syllable_to_phonemes(decompose("별"), 0.5)
check("별 = ㅂ + j + ㅓ + ㄹ", [p.symbol for p in phonemes], ["ㅂ", "j", "ㅓ", "ㄹ"])
check_true("반모음이 짧다", phonemes[1].duration < phonemes[2].duration)
# 아주 짧은 음에서도 자음이 음을 다 먹지 않는다
phonemes = syllable_to_phonemes(decompose("가"), 0.08)
vowel = next(p for p in phonemes if p.kind is PhonemeKind.VOWEL)
check_true("짧은 음도 모음이 절반 이상", vowel.duration > 0.04,
           f"(모음 {vowel.duration * 1000:.0f}ms / 전체 80ms)")
check_raises("길이 0 거부", lambda: syllable_to_phonemes(decompose("가"), 0.0), KoreanError)

print("[5] 가사 붙이기")
times = [(index * 0.5, 0.45) for index in range(6)]
syllables = align_lyrics("다시 만나는 날", times)
check("음절 수", len(syllables), 6)
check("첫 음절", syllables[0].text, "다")
check("시각이 붙는다", syllables[2].note_start, 1.0)
check_true("모든 음절에 모음이 있다", all(s.vowel is not None for s in syllables))
check_true("자음이 앞선다", all(s.onset_offset <= 0 for s in syllables))
check_raises("음절 수가 안 맞으면 거부",
             lambda: align_lyrics("안녕", [(0.0, 0.5)]), KoreanError)
# 발음 규칙이 가사 전체에 적용된다
sung = align_lyrics("같이", [(0.0, 0.5), (0.5, 0.5)])
check("같이 -> 두 번째가 ㅊ", sung[1].phonemes[0].symbol, "ㅊ")

print("[6] 입 모양 — 33번 립싱크")
timeline = viseme_timeline(syllables)
check_true("시간표가 만들어진다", len(timeline) > len(syllables))
check_true("시간순이다", all(timeline[i][0] <= timeline[i + 1][0]
                          for i in range(len(timeline) - 1)))
visemes = {item[2] for item in timeline}
check_true("입 모양이 여러 가지", len(visemes) >= 3, f"({visemes})")
check_true("모든 구간이 유효", all(end >= start for start, end, _ in timeline))
# 같은 모음은 같은 입 모양
one = align_lyrics("아", [(0.0, 0.5)])
two = align_lyrics("하", [(0.0, 0.5)])
check("'아'와 '하'의 모음 입 모양이 같다",
      one[0].vowel.viseme, two[0].vowel.viseme)
check("ㅁ 은 입을 다문다", align_lyrics("마", [(0, 0.5)])[0].phonemes[0].viseme, "M")

print("[7] 성대 파형")
phase = np.linspace(0, 2 * np.pi * 20, 20000)
pulse = glottal_pulse(phase, 0.5)
check("직류 성분 없음", float(abs(pulse.mean())), 0.0, tol=1e-6)
check_true("비대칭이다 (천천히 열리고 빨리 닫힌다)",
           abs(pulse.max()) > abs(pulse.min()) * 1.5,
           f"(최대 {pulse.max():.3f}, 최소 {pulse.min():.3f})")
# 긴장이 높으면 배음이 늘어난다
soft = glottal_pulse(np.linspace(0, 2 * np.pi * 50, 48000), 0.1)
hard = glottal_pulse(np.linspace(0, 2 * np.pi * 50, 48000), 0.9)


def high_ratio(signal):
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.arange(len(spectrum))
    return float(spectrum[freqs > 200].sum() / max(spectrum[freqs <= 200].sum(), 1e-9))


check_true("긴장이 높으면 배음이 많다", high_ratio(hard) > high_ratio(soft),
           f"(부드러움 {high_ratio(soft):.4f} vs 긴장 {high_ratio(hard):.4f})")

print("[8] 공명기 정규화")
from scipy.signal import freqz

b, a = _resonator(900.0, 80.0, RATE, "dc")
_, response = freqz(b, a, worN=4096, fs=RATE)
check("직류 이득 1", float(abs(response[0])), 1.0, tol=1e-6)
check_true("공명점에서 크게 솟는다", abs(response[np.argmin(abs(_ - 900))]) > 5.0)
# 다섯 개를 이어도 살아남아야 한다 (예전에 여기서 소리가 사라졌다)
total = np.ones(4096, dtype=complex)
for frequency in (900.0, 1400.0, 2900.0, 3800.0, 4600.0):
    b, a = _resonator(frequency, 120.0, RATE, "dc")
    _, single = freqz(b, a, worN=4096, fs=RATE)
    total = total * single
check("5개 직렬 후에도 직류 이득 1", float(abs(total[0])), 1.0, tol=1e-6)
check_true("포먼트가 살아 있다", abs(total[np.argmin(abs(_ - 900))]) > 1.0,
           f"({abs(total[np.argmin(abs(_ - 900))]):.2f})")
b, a = _resonator(900.0, 80.0, RATE, "peak")
_, response = freqz(b, a, worN=4096, fs=RATE)
check("peak 정규화는 공명점에서 1", float(abs(response[np.argmin(abs(_ - 900))])), 1.0, tol=0.02)
check_raises("모르는 정규화 거부", lambda: _resonator(900.0, 80.0, RATE, "bogus"), VoiceError)

print("[9] 모음이 서로 구분되는가")
timbre = VoiceTimbre.preset("mezzo")
scale = timbre.formant_scale
for vowel, (want1, want2) in (
    ("아", (900.0, 1400.0)), ("에", (500.0, 2300.0)), ("이", (320.0, 2700.0)),
):
    audio = sing(vowel, [(0.0, 1.0)], [110.0], timbre)
    peaks = envelope_peaks(audio.data[0][int(0.25 * RATE):int(0.9 * RATE)])
    target = want2 * scale
    near = [p for p in peaks if abs(p - target) < target * 0.25]
    check_true(f"'{vowel}' 의 F2 가 {target:.0f}Hz 근처", bool(near),
               f"(찾은 봉우리 {[f'{p:.0f}' for p in peaks[:5]]})")
# 서로 다른 모음은 스펙트럼이 달라야 한다
def spectrum_of(text):
    audio = sing(text, [(0.0, 1.0)], [110.0], timbre)
    segment = audio.data[0][int(0.3 * RATE):int(0.8 * RATE)]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    return spectrum / (spectrum.sum() + 1e-12)


a_spectrum = spectrum_of("아")
i_spectrum = spectrum_of("이")
u_spectrum = spectrum_of("우")
difference = float(np.abs(a_spectrum - i_spectrum).sum())
check_true("'아'와 '이'의 스펙트럼이 다르다", difference > 0.3, f"(차이 {difference:.3f})")
check_true("'아'와 '우'도 다르다",
           float(np.abs(a_spectrum - u_spectrum).sum()) > 0.3)

print("[10] 목소리 종류 — 성도 배율이 포먼트를 옮긴다")
for preset in ("soprano", "mezzo", "alto", "tenor", "baritone", "bass"):
    voice = VoiceTimbre.preset(preset)
    audio = sing("이", [(0.0, 1.0)], [110.0], voice)
    peaks = envelope_peaks(audio.data[0][int(0.25 * RATE):int(0.9 * RATE)])
    target = 2700.0 * voice.formant_scale
    near = [p for p in peaks if abs(p - target) < target * 0.35]
    found = min(near, key=lambda p: abs(p - target)) if near else 0.0
    error = abs(found - target) / target if found else 1.0
    check_true(f"{preset}: F2 가 {target:.0f}Hz 근처", error < 0.15,
               f"(실측 {found:.0f}Hz, 오차 {error * 100:.1f}%)")
check_raises("모르는 목소리 거부", lambda: VoiceTimbre.preset("bogus"), VoiceError)
check_raises("성도 배율 범위 초과", lambda: VoiceTimbre(formant_scale=3.0), VoiceError)
check_raises("숨소리 범위 초과", lambda: VoiceTimbre(breathiness=2.0), VoiceError)

print("[11] 숨소리가 실제 잡음량을 바꾼다")


def harmonic_to_noise(text, breathiness):
    voice = VoiceTimbre(breathiness=breathiness)
    audio = sing(text, [(0.0, 1.0)], [220.0], voice)
    segment = audio.data[0][int(0.3 * RATE):int(0.9 * RATE)]
    spectrum = np.abs(np.fft.rfft((segment - segment.mean()) * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / RATE)
    harmonics, between = [], []
    for k in range(1, 25):
        frequency = 220.0 * k
        if frequency > 8000.0:
            break
        index = int(np.argmin(abs(freqs - frequency)))
        harmonics.append(spectrum[max(0, index - 2):index + 3].max())
        middle = int(np.argmin(abs(freqs - (frequency + 110.0))))
        between.append(np.median(spectrum[max(0, middle - 4):middle + 5]))
    return 20.0 * np.log10(np.mean(harmonics) / max(np.mean(between), 1e-12))


clear = harmonic_to_noise("아", 0.05)
breathy = harmonic_to_noise("아", 0.85)
check_true("숨이 많으면 잡음이 늘어난다", breathy < clear - 5.0,
           f"(또렷 {clear:.1f}dB vs 숨 많음 {breathy:.1f}dB)")

print("[12] 지정한 음높이를 내는가")
melody = [60, 64, 67, 72, 67, 64, 60]
text = "도미솔도솔미도"
times = [(index * 0.6, 0.55) for index in range(len(melody))]
pitches = [Pitch.from_midi(m).frequency for m in melody]
audio = sing(text, times, pitches, VoiceTimbre.preset("mezzo"))
check_true("소리가 난다", audio.peak() > 0.3, f"({audio.peak():.3f})")
check("길이", audio.duration, times[-1][0] + times[-1][1] + 0.25, tol=0.05)
worst = 0.0
for midi, (start, duration) in zip(melody, times):
    want = Pitch.from_midi(midi).frequency
    segment = audio.data[0][int((start + 0.15) * RATE):int((start + duration - 0.05) * RATE)]
    if len(segment) < 4000:
        continue
    measured = measure_note_frequency(
        segment, RATE, fmin=want * 0.6, fmax=want * 1.7,
        skip_seconds=0.0, length_seconds=len(segment) / RATE,
    )
    if measured > 0:
        worst = max(worst, abs(1200.0 * np.log2(measured / want)))
check_true("음높이 오차 25센트 이내", worst < 25.0, f"(최대 {worst:.1f}센트)")

print("[13] 잘못된 입력")
check_raises("음절 수 불일치", lambda: sing("가나", [(0, 0.5)], [440.0]), KoreanError)
check_raises("음높이 개수 불일치",
             lambda: SingingSynth().render(align_lyrics("가", [(0, 0.5)]), [440.0, 880.0]),
             VoiceError)
check_raises("음높이 0", lambda: sing("가", [(0, 0.5)], [0.0]), VoiceError)
empty = SingingSynth().render([], [])
check("빈 입력은 빈 소리", empty.frames, 0)

# ---------------------------------------------------------------------------
print("[크기] 모음마다 고른 크기, 튀는 소리 없음")
# 예전에는 마찰음 잡음이 포먼트 직렬 사슬을 지나며 40dB 넘게 부풀었고,
# 전체를 그 봉우리에 맞추느라 모음이 반주보다 30dB 작아 들리지 않았다.
LEVEL_RATE = 24000


def vowel_levels(text, pitches, rate=LEVEL_RATE):
    times = [(i * 0.45, 0.42) for i in range(len(pitches))]
    syllables = align_lyrics(text, times)
    audio = SingingSynth(VoiceTimbre.preset("mezzo"), rate).render(syllables, pitches).data[0]
    shift = max(0.0, -min(s.note_start + s.onset_offset for s in syllables))
    vowels, consonants = [], []
    for syllable in syllables:
        vowel = syllable.vowel
        a = int((syllable.note_start + shift + vowel.start) * rate)
        b = int((syllable.note_start + shift + vowel.end) * rate)
        vowels.append(20 * np.log10(np.sqrt(np.mean(audio[a:b] ** 2))))
        for phoneme in syllable.phonemes:
            if phoneme.symbol in ("ㅅ", "ㅆ"):
                a = int((syllable.note_start + shift + phoneme.start) * rate)
                b = int((syllable.note_start + shift + phoneme.end) * rate)
                consonants.append(20 * np.log10(np.sqrt(np.mean(audio[a:b] ** 2)) + 1e-12))
    return audio, vowels, consonants


audio, vowels, _ = vowel_levels("아이우에오여기", [220, 262, 330, 392, 440, 494, 523])
median = float(np.median(vowels))
check_true("모음 크기가 목표 근처 (-14 dBFS ±3)", abs(median + 14.0) <= 3.0, f"({median:.1f})")
spread = max(abs(v - median) for v in vowels)
check_true("모음마다 크기 차이 ±4dB 안", spread <= 4.0,
           f"({', '.join(f'{v:.1f}' for v in vowels)})")
audio, vowels, fricatives = vowel_levels("사시사철 쓸쓸히", [330, 349, 392, 440, 392, 349, 330])
frames = np.sqrt(np.mean(audio[: len(audio) // 240 * 240].reshape(-1, 240) ** 2, axis=1))
loud = frames[frames > 1e-4]
jump = 20 * np.log10(loud.max() / np.median(loud))
check_true("마찰음 많은 문장에서도 튀는 봉우리 없음 (10ms 최대/중앙 12dB 이하)",
           jump <= 12.0, f"({jump:.1f}dB)")
check_true("넘치지 않는다", float(np.abs(audio).max()) <= 1.0, f"({np.abs(audio).max():.2f})")
relative = float(np.median(fricatives) - np.median(vowels))
check_true("ㅅ 은 들리되 모음보다 작다 (-16 ~ -3dB)", -16.0 <= relative <= -3.0,
           f"({relative:.1f}dB)")

print("\n" + "=" * 62)
print(f"검증 항목 {checks}개")
if failures:
    print(f"실패 {len(failures)}건:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("전부 통과")
